use tch::{nn, nn::Module, Device, Kind, Tensor};
use crate::structures::_graph::{GraphTensors, HAN};

pub struct TapFingerActor {
    pub han: HAN,
    pub task_selection: nn::Sequential,
    pub pointer_query: nn::Linear,
    pub task_encoder: nn::Sequential,
    pub pointer_key: nn::Linear,
    pub cpu_allocator: nn::Sequential,
    pub gpu_allocator: nn::Sequential,
    pub memory_allocator: nn::Sequential,
    pub hidden_dim: i64,
    pub resource_bins: i64,
}

impl TapFingerActor {
    pub fn new(vs: &nn::Path, input_dim: i64, hidden_dim: i64, num_resource_bins: i64) -> Self {
        let han = HAN::new(
            &(vs / "han"),
            input_dim,
            hidden_dim,
            2,  // num_layers
            4,  // num_heads
            3,  // num_edge_types
        );

        let task_encoder = nn::seq()
            .add(nn::linear(
                vs / "task_enc_1",
                hidden_dim,
                hidden_dim,
                Default::default(),
            ))
            .add_fn(|x| x.relu())
            .add(nn::linear(
                vs / "task_enc_2",
                hidden_dim,
                hidden_dim,
                Default::default(),
            ));

        let pointer_query = nn::linear(vs / "ptr_q", hidden_dim, hidden_dim, Default::default());
        let pointer_key = nn::linear(vs / "ptr_k", hidden_dim, hidden_dim, Default::default());

        let cpu_allocator = nn::seq()
            .add(nn::linear(
                vs / "cpu_1",
                hidden_dim * 2,
                hidden_dim / 2,
                Default::default(),
            ))
            .add_fn(|x| x.relu())
            .add(nn::linear(
                vs / "cpu_2",
                hidden_dim / 2,
                num_resource_bins,
                Default::default(),
            ));

        let gpu_allocator = nn::seq()
            .add(nn::linear(
                vs / "gpu_1",
                hidden_dim * 2,
                hidden_dim / 2,
                Default::default(),
            ))
            .add_fn(|x| x.relu())
            .add(nn::linear(
                vs / "gpu_2",
                hidden_dim / 2,
                num_resource_bins,
                Default::default(),
            ));

        let memory_allocator = nn::seq()
            .add(nn::linear(
                vs / "mem_1",
                hidden_dim * 2,
                hidden_dim / 2,
                Default::default(),
            ))
            .add_fn(|x| x.relu())
            .add(nn::linear(
                vs / "mem_2",
                hidden_dim / 2,
                num_resource_bins,
                Default::default(),
            ));

        // task_selection is used for the pointer network scoring
        let task_selection = nn::seq()
            .add(nn::linear(
                vs / "ts_1",
                hidden_dim,
                hidden_dim,
                Default::default(),
            ))
            .add_fn(|x| x.relu());

        Self {
            han,
            task_encoder,
            pointer_query,
            pointer_key,
            cpu_allocator,
            gpu_allocator,
            memory_allocator,
            task_selection,
            hidden_dim,
            resource_bins: num_resource_bins,
        }
    }

    /// High-level forward: runs HAN then pointer network, returns (task_probs, resource_logits)
    pub fn forward(
        &self,
        graph_tensors: &GraphTensors,
        action_mask: &ActionMask,
    ) -> (Tensor, Tensor) {
        let full_embedding = self.han.forward(graph_tensors);

        let cluster_embedding = self.extract_cluster_embedding(&full_embedding, graph_tensors);
        let pending_embeddings = self.extract_pending_embeddings(&full_embedding, graph_tensors);

        let output = self.forward_detailed(&cluster_embedding, &pending_embeddings, action_mask);

        let resource_logits = if let Some(ref res) = output.resource_allocation {
            Tensor::cat(
                &[
                    res.cpu.unsqueeze(0),
                    res.gpu.unsqueeze(0),
                    res.memory.unsqueeze(0),
                ],
                0,
            )
        } else {
            Tensor::zeros(&[3, 1], (Kind::Float, cluster_embedding.device()))
        };

        (output.task_probs, resource_logits)
    }

    pub fn forward_detailed(
        &self,
        cluster_embedding: &Tensor,
        pending_embeddings: &Tensor,
        action_mask: &ActionMask,
    ) -> ActorOutput {
        let num_pending = pending_embeddings.size()[0];

        // Prepend a "no-action" zero embedding
        let no_action_emb = Tensor::zeros(
            &[1, self.hidden_dim],
            (pending_embeddings.kind(), pending_embeddings.device()),
        );
        let task_embeddings = Tensor::cat(&[&no_action_emb, pending_embeddings], 0);

        let encoded_tasks = self.task_encoder.forward(&task_embeddings);
        let query = self.pointer_query.forward(cluster_embedding); // [1, hidden]
        let keys = self.pointer_key.forward(&encoded_tasks);       // [num_pending+1, hidden]

        // Pointer attention scores
        let scores = query.matmul(&keys.transpose(0, 1)).squeeze_dim(0); // [num_pending+1]

        let masked_scores = scores + &action_mask.task_mask;
        let task_probs = masked_scores.softmax(0, Kind::Float);

        let task_action = task_probs.multinomial(1, true).squeeze();
        let task_idx = i64::from(&task_action);

        // Only allocate resources if a real task was selected (not no-action at index 0)
        let resource_allocation = if task_idx > 0 {
            let selected_task_emb = encoded_tasks.get(task_idx);
            let context = Tensor::cat(&[cluster_embedding, &selected_task_emb.unsqueeze(0)], 1);

            let cpu_logits = self.cpu_allocator.forward(&context).squeeze_dim(0);
            let cpu_masked = cpu_logits + &action_mask.cpu_mask;
            let cpu_probs = cpu_masked.softmax(0, Kind::Float);
            let cpu_action = cpu_probs.multinomial(1, true);

            let gpu_logits = self.gpu_allocator.forward(&context).squeeze_dim(0);
            let gpu_masked = gpu_logits + &action_mask.gpu_mask;
            let gpu_probs = gpu_masked.softmax(0, Kind::Float);
            let gpu_action = gpu_probs.multinomial(1, true);

            let mem_logits = self.memory_allocator.forward(&context).squeeze_dim(0);
            let mem_masked = mem_logits + &action_mask.memory_mask;
            let mem_probs = mem_masked.softmax(0, Kind::Float);
            let mem_action = mem_probs.multinomial(1, true);

            Some(ResourceAction {
                cpu: cpu_action,
                gpu: gpu_action,
                memory: mem_action,
            })
        } else {
            None
        };

        ActorOutput {
            task_action,
            task_probs,
            resource_allocation,
        }
    }

    fn extract_cluster_embedding(&self, full_embedding: &Tensor, graph: &GraphTensors) -> Tensor {
        if graph.cluster_indices.is_empty() {
            return Tensor::zeros(
                &[1, self.hidden_dim],
                (Kind::Float, full_embedding.device()),
            );
        }

        let cluster_idx =
            Tensor::of_slice(&graph.cluster_indices[0..1]).to_device(full_embedding.device());
        full_embedding.index_select(0, &cluster_idx)
    }

    fn extract_pending_embeddings(&self, full_embedding: &Tensor, graph: &GraphTensors) -> Tensor {
        if graph.pending_indices.is_empty() {
            return Tensor::zeros(
                &[0, self.hidden_dim],
                (Kind::Float, full_embedding.device()),
            );
        }

        let pending_idx =
            Tensor::of_slice(&graph.pending_indices).to_device(full_embedding.device());
        full_embedding.index_select(0, &pending_idx)
    }
}

// ============================================================================
// ACTION MASK
// ============================================================================

pub struct ActionMask {
    pub task_mask: Tensor,   // [num_pending+1] — 0 for valid, -inf for invalid
    pub cpu_mask: Tensor,    // [num_cpu_bins]
    pub gpu_mask: Tensor,    // [num_gpus+1]
    pub memory_mask: Tensor, // [num_memory_bins]
}

impl ActionMask {
    pub fn new(
        num_pending: i64,
        num_cpu_bins: i64,
        num_gpus: i64,
        num_memory_bins: i64,
        device: Device,
    ) -> Self {
        Self {
            task_mask: Tensor::zeros(&[num_pending + 1], (Kind::Float, device)),
            cpu_mask: Tensor::zeros(&[num_cpu_bins], (Kind::Float, device)),
            gpu_mask: Tensor::zeros(&[num_gpus + 1], (Kind::Float, device)),
            memory_mask: Tensor::zeros(&[num_memory_bins], (Kind::Float, device)),
        }
    }

    /// Mark a task index as invalid (set to -inf)
    pub fn mask_task(&mut self, task_idx: i64) {
        let _ = self.task_mask.get(task_idx).fill_(f64::NEG_INFINITY);
    }

    /// Mark a cpu bin index as invalid
    pub fn mask_cpu(&mut self, cpu_idx: i64) {
        let _ = self.cpu_mask.get(cpu_idx).fill_(f64::NEG_INFINITY);
    }

    /// Mark a gpu index as invalid
    pub fn mask_gpu(&mut self, gpu_idx: i64) {
        let _ = self.gpu_mask.get(gpu_idx).fill_(f64::NEG_INFINITY);
    }

    /// Mark a memory bin index as invalid
    pub fn mask_memory(&mut self, mem_idx: i64) {
        let _ = self.memory_mask.get(mem_idx).fill_(f64::NEG_INFINITY);
    }
}

// ============================================================================
// OUTPUT TYPES
// ============================================================================

pub struct ActorOutput {
    pub task_action: Tensor,
    pub task_probs: Tensor,
    pub resource_allocation: Option<ResourceAction>,
}

pub struct ResourceAction {
    pub cpu: Tensor,
    pub gpu: Tensor,
    pub memory: Tensor,
}
