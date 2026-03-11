use std::collections::HashMap;
use tch::{nn, Device, IndexOp, Kind, Tensor};
use chrono::{DateTime, Duration, Utc};
use RL::env::*

pub struct ScatterGather;

impl ScatterGather {
    pub fn scatter(src: &Tensor, index: &Tensor, dim_size: i64, reduce: &str) -> Tensor {
        let device = src.device();
        let feature_dim = src.size()[1];

        let mut out = Tensor::zeros(&[dim_size, feature_dim], (src.kind(), device));

        match reduce {
            "sum" => {
                out.index_add_(0, index, src);
                out
            }
            "mean" => {
                let sum = out.index_add_(0, index, src);

                let ones = Tensor::ones(&[src.size()[0], 1], (Kind::Float, device));
                let mut count = Tensor::zeros(&[dim_size, 1], (Kind::Float, device));
                count.index_add_(0, index, &ones);

                let count = count.clamp_min(1.0);
                sum / count
            }
            "max" => Self::scatter_max(src, index, dim_size, device),
            _ => panic!("Unsupported reduce operation: {}", reduce),
        }
    }

    /// Scatter max operation (special case requiring iteration)
    fn scatter_max(src: &Tensor, index: &Tensor, dim_size: i64, device: Device) -> Tensor {
        let feature_dim = src.size()[1];
        let mut out = Tensor::full(
            &[dim_size, feature_dim],
            f32::NEG_INFINITY,
            (Kind::Float, device),
        );

        let index_vec: Vec<i64> = index.to_kind(Kind::Int64).into();

        for (i, &idx) in index_vec.iter().enumerate() {
            let src_row = src.get(i as i64);
            let out_row = out.get(idx);
            let max_row = src_row.max_other(&out_row);
            out = out.index_put_(&[Some(Tensor::of_slice(&[idx]))], &max_row, false);
        }

        out
    }

    pub fn gather(src: &Tensor, index: &Tensor) -> Tensor {
        src.index_select(0, index)
    }
}

/// Node-level attention: computes attention between neighboring nodes
pub struct NodeLevelAttention {
    query: nn::Linear,
    key: nn::Linear,
    value: nn::Linear,
    num_heads: i64,
    head_dim: i64,
}

impl NodeLevelAttention {
    pub fn new(vs: &nn::Path, input_dim: i64, hidden_dim: i64, num_heads: i64) -> Self {
        assert_eq!(
            hidden_dim % num_heads,
            0,
            "hidden_dim must be divisible by num_heads"
        );
        let head_dim = hidden_dim / num_heads;

        Self {
            query: nn::linear(vs / "query", input_dim, hidden_dim, Default::default()),
            key: nn::linear(vs / "key", input_dim, hidden_dim, Default::default()),
            value: nn::linear(vs / "value", input_dim, hidden_dim, Default::default()),
            num_heads,
            head_dim,
        }
    }

    pub fn forward(&self, node_features: &Tensor, edge_index: &Tensor, num_nodes: i64) -> Tensor {
        // edge_index: [2, num_edges] where first row is source, second row is target
        let src_index = edge_index.i(0);
        let tgt_index = edge_index.i(1);

        // Compute Q, K, V
        let q = self.query.forward(node_features); // [num_nodes, hidden_dim]
        let k = self.key.forward(node_features);
        let v = self.value.forward(node_features);

        let batch_size = q.size()[0];
        let q = q.view([batch_size, self.num_heads, self.head_dim]);
        let k = k.view([batch_size, self.num_heads, self.head_dim]);
        let v = v.view([batch_size, self.num_heads, self.head_dim]);

        // Gather source and target features
        let q_tgt = ScatterGather::gather(&q, &tgt_index); // [num_edges, num_heads, head_dim]
        let k_src = ScatterGather::gather(&k, &src_index);
        let v_src = ScatterGather::gather(&v, &src_index);

        let scores = (q_tgt * k_src).sum_dim_intlist(&[2i64][..], false, Kind::Float)
            / (self.head_dim as f64).sqrt(); // [num_edges, num_heads]

        let attention_weights = Self::edge_softmax(&scores, &tgt_index, num_nodes);

        // Apply attention to values
        let attention_weights = attention_weights.unsqueeze(-1); // [num_edges, num_heads, 1]
        let weighted_values = v_src * attention_weights; // [num_edges, num_heads, head_dim]

        let num_edges = weighted_values.size()[0];
        let weighted_values_flat =
            weighted_values.view([num_edges, self.num_heads * self.head_dim]);

        let aggregated =
            ScatterGather::scatter(&weighted_values_flat, &tgt_index, num_nodes, "sum");

        aggregated
    }

    /// Softmax over edges grouped by target node
    fn edge_softmax(scores: &Tensor, tgt_index: &Tensor, num_nodes: i64) -> Tensor {
        let device = scores.device();
        let num_heads = scores.size()[1];

        let max_scores = ScatterGather::scatter(scores, tgt_index, num_nodes, "max"); // [num_nodes, num_heads]

        let max_per_edge = ScatterGather::gather(&max_scores, tgt_index);

        let exp_scores = (scores - max_per_edge).exp();

        let sum_exp = ScatterGather::scatter(&exp_scores, tgt_index, num_nodes, "sum");

        let sum_per_edge = ScatterGather::gather(&sum_exp, tgt_index);

        exp_scores / (sum_per_edge + 1e-10)
    }
}

/// Semantic-level attention: aggregates across different edge types (meta-paths)
pub struct SemanticLevelAttention {
    attention_weights: nn::Linear,
}

impl SemanticLevelAttention {
    pub fn new(vs: &nn::Path, hidden_dim: i64, num_edge_types: i64) -> Self {
        Self {
            attention_weights: nn::linear(
                vs / "semantic_attn",
                hidden_dim,
                num_edge_types,
                Default::default(),
            ),
        }
    }

    pub fn forward(&self, embeddings_by_type: &[Tensor]) -> Tensor {
        if embeddings_by_type.is_empty() {
            panic!("No embeddings provided");
        }

        if embeddings_by_type.len() == 1 {
            return embeddings_by_type[0].shallow_clone();
        }

        // Stack embeddings: [num_types, num_nodes, hidden_dim]
        let stacked = Tensor::stack(embeddings_by_type, 0);

        let mean_embeddings = stacked.mean_dim(Some(&[1i64][..]), false, Kind::Float);

        let attention_logits = self.attention_weights.forward(&mean_embeddings);
        let attention_scores = attention_logits.mean_dim(Some(&[1i64][..]), false, Kind::Float); // [num_types]

        let attention_weights = attention_scores.softmax(0, Kind::Float);

        let attention_weights = attention_weights.unsqueeze(-1).unsqueeze(-1); // [num_types, 1, 1]

        let weighted = stacked * attention_weights;
        weighted.sum_dim_intlist(&[0i64][..], false, Kind::Float)
    }
}

pub struct HANLayer {
    node_attention: NodeLevelAttention,
    semantic_attention: SemanticLevelAttention,
    num_edge_types: i64,
}

impl HANLayer {
    pub fn new(
        vs: &nn::Path,
        input_dim: i64,
        hidden_dim: i64,
        num_heads: i64,
        num_edge_types: i64,
    ) -> Self {
        Self {
            node_attention: NodeLevelAttention::new(
                &(vs / "node_attn"),
                input_dim,
                hidden_dim,
                num_heads,
            ),
            semantic_attention: SemanticLevelAttention::new(
                &(vs / "semantic_attn"),
                hidden_dim,
                num_edge_types,
            ),
            num_edge_types,
        }
    }

    pub fn forward(
        &self,
        node_features: &Tensor,
        edge_index: &Tensor,
        edge_types: &Tensor,
    ) -> Tensor {
        let num_nodes = node_features.size()[0];
        let device = node_features.device();

        let edge_types_vec: Vec<i64> = edge_types.to_kind(Kind::Int64).into();
        let mut edges_by_type: HashMap<i64, Vec<i64>> = HashMap::new();

        for (i, &edge_type) in edge_types_vec.iter().enumerate() {
            edges_by_type
                .entry(edge_type)
                .or_insert_with(Vec::new)
                .push(i as i64);
        }

        let mut embeddings_by_type = Vec::new();

        for edge_type in 0..self.num_edge_types {
            if let Some(edge_indices) = edges_by_type.get(&edge_type) {
                if edge_indices.is_empty() {
                    continue;
                }

                let edge_mask = Tensor::of_slice(edge_indices).to_device(device);
                let typed_edge_index = edge_index.index_select(1, &edge_mask);

                let embedding =
                    self.node_attention
                        .forward(node_features, &typed_edge_index, num_nodes);

                embeddings_by_type.push(embedding);
            } else {
                let zero_embedding = Tensor::zeros_like(node_features);
                embeddings_by_type.push(zero_embedding);
            }
        }

        if embeddings_by_type.is_empty() {
            node_features.shallow_clone()
        } else {
            self.semantic_attention.forward(&embeddings_by_type)
        }
    }
}

pub struct HAN {
    layers: Vec<HANLayer>,
    input_projection: nn::Linear,
    output_projection: nn::Linear,
    num_layers: i64,
    hidden_dim: i64,
}

impl HAN {
    pub fn new(
        vs: &nn::Path,
        input_dim: i64,
        hidden_dim: i64,
        num_layers: i64,
        num_heads: i64,
        num_edge_types: i64,
    ) -> Self {
        let input_projection =
            nn::linear(vs / "input_proj", input_dim, hidden_dim, Default::default());

        // HAN layers
        let mut layers = Vec::new();
        for i in 0..num_layers {
            let layer = HANLayer::new(
                &(vs / format!("han_layer_{}", i)),
                hidden_dim,
                hidden_dim,
                num_heads,
                num_edge_types,
            );
            layers.push(layer);
        }

        let output_projection = nn::linear(
            vs / "output_proj",
            hidden_dim,
            hidden_dim,
            Default::default(),
        );

        Self {
            layers,
            input_projection,
            output_projection,
            num_layers,
            hidden_dim,
        }
    }

    pub fn forward(&self, graph_tensors: &GraphTensors) -> Tensor {
        let mut x = self.input_projection.forward(&graph_tensors.node_features);
        x = x.relu();

        for layer in &self.layers {
            let h = layer.forward(&x, &graph_tensors.edge_index, &graph_tensors.edge_types);

            x = (x + h).relu();
        }

        // Final projection
        self.output_projection.forward(&x)
    }

    pub fn get_node_type_embeddings(&self, graph_tensors: &GraphTensors, node_type: i64) -> Tensor {
        let full_embeddings = self.forward(graph_tensors);

        let node_types = &graph_tensors.node_types;
        let mask: Vec<i64> = node_types
            .to_kind(Kind::Int64)
            .into::<Vec<i64>>()
            .iter()
            .enumerate()
            .filter(|(_, &t)| t == node_type)
            .map(|(i, _)| i as i64)
            .collect();

        if mask.is_empty() {
            let device = full_embeddings.device();
            return Tensor::zeros(&[0, self.hidden_dim], (Kind::Float, device));
        }

        let mask_tensor = Tensor::of_slice(&mask).to_device(full_embeddings.device());
        full_embeddings.index_select(0, &mask_tensor)
    }
}

#[derive(Debug, Clone)]
pub struct GraphTensors {
    pub node_features: Tensor,
    pub node_types: Tensor,
    pub edge_index: Tensor,
    pub edge_types: Tensor,
    pub num_nodes: i64,
    pub cluster_indices: Vec<i64>,
    pub pending_indices: Vec<i64>,
    pub running_indices: Vec<i64>,
}

impl GraphTensor{
    pub fn new() -> Self { 
        Self { 
            node_features: Tensor::new([] , device=Device), 
            node_type: Tensor::new([] , device =Device), 
            edge_index: Tensor::new([] , device =Device), 
            edge_type: Tensor::new([] , device =Device), 
            num_nodes: 0.0,
            cluster_indices :Vec::new(),
            pending_indices: Vec::new(),
            running_indices: Vec::new()
        };
    }
    
    pub fn add_cluster(&self , cluster_features: &[ClusterFeatures]){
        
    }
}

// resize handles both directions —
// pads with 0.0 if too short, silently truncates if somehow over 32.
//  Call it inside every add_* method in HANGraph before pushing to node_features, so the padding logic lives in exactly one place.
fn pad_to_feature_dim(mut v: Vec<f32>) -> Vec<f32> {
    v.resize(GRAPH_FEATURE_DIM, 0.0);
    v
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_scatter_sum() {
        let device = Device::Cpu;

        let src = Tensor::of_slice(&[
            1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0,
        ])
        .view([5, 3]);

        let index = Tensor::of_slice(&[0i64, 0, 1, 1, 2]);

        let result = ScatterGather::scatter(&src, &index, 3, "sum");

        println!("Scatter sum result: {:?}", result);
        assert_eq!(result.size(), vec![3, 3]);
    }

    #[test]
    fn test_han_forward() {
        let vs = nn::VarStore::new(Device::Cpu);
        let han = HAN::new(&vs.root(), 10, 64, 2, 4, 3);

        let num_nodes = 20;
        let num_edges = 40;

        let graph_tensors = GraphTensors {
            node_features: Tensor::randn(&[num_nodes, 10], (Kind::Float, Device::Cpu)),
            node_types: Tensor::randint(4, &[num_nodes], (Kind::Int64, Device::Cpu)),
            edge_index: Tensor::randint(num_nodes, &[2, num_edges], (Kind::Int64, Device::Cpu)),
            edge_types: Tensor::randint(3, &[num_edges], (Kind::Int64, Device::Cpu)),
            num_nodes: num_nodes as i64,
            cluster_indices: vec![0, 1, 2],
            pending_indices: vec![3, 4, 5, 6],
            running_indices: vec![7, 8],
        };

        let embeddings = han.forward(&graph_tensors);
        println!("Output shape: {:?}", embeddings.size());
        assert_eq!(embeddings.size(), vec![num_nodes, 64]);
    }
}
