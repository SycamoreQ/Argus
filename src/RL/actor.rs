use crate::structures::_graph::{GraphTensors, HAN, GRAPH_FEATURE_DIM};
use crate::RL::env::EdgeMLEnv;
use tch::{nn, nn::Module, Device, Kind, Tensor};

pub struct TapFingerActor {
    pub han: HAN,
    pub task_encoder: nn::Sequential,
    pub pointer_query: nn::Linear,
    pub pointer_key: nn::Linear,
    pub cpu_allocator: nn::Sequential,
    pub gpu_allocator: nn::Sequential,
    pub memory_allocator: nn::Sequential,
    pub task_selection: nn::Sequential,
    pub hidden_dim: i64,
    pub resource_bins: i64,
}

impl TapFingerActor {
    pub fn new(vs: &nn::Path, hidden_dim: i64, num_resource_bins: i64) -> Self {
        let han = HAN::new(
            &(vs / "han"),
            GRAPH_FEATURE_DIM as i64,
            hidden_dim,
            2,
            4,
            3,
        );

        let task_encoder = nn::seq()
            .add(nn::linear(vs / "task_enc_1", hidden_dim, hidden_dim, Default::default()))
            .add_fn(|x| x.relu())
            .add(nn::linear(vs / "task_enc_2", hidden_dim, hidden_dim, Default::default()));

        let pointer_query =
            nn::linear(vs / "ptr_q", hidden_dim, hidden_dim, Default::default());
        let pointer_key =
            nn::linear(vs / "ptr_k", hidden_dim, hidden_dim, Default::default());

        let cpu_allocator = nn::seq()
            .add(nn::linear(vs / "cpu_1", hidden_dim * 2, hidden_dim / 2, Default::default()))
            .add_fn(|x| x.relu())
            .add(nn::linear(vs / "cpu_2", hidden_dim / 2, num_resource_bins, Default::default()));

        let gpu_allocator = nn::seq()
            .add(nn::linear(vs / "gpu_1", hidden_dim * 2, hidden_dim / 2, Default::default()))
            .add_fn(|x| x.relu())
            .add(nn::linear(vs / "gpu_2", hidden_dim / 2, num_resource_bins, Default::default()));

        let memory_allocator = nn::seq()
            .add(nn::linear(vs / "mem_1", hidden_dim * 2, hidden_dim / 2, Default::default()))
            .add_fn(|x| x.relu())
            .add(nn::linear(vs / "mem_2", hidden_dim / 2, num_resource_bins, Default::default()));

        let task_selection = nn::seq()
            .add(nn::linear(vs / "ts_1", hidden_dim, hidden_dim, Default::default()))
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

    /// High-level forward pass.
    /// Returns (task_probs [num_pending+1], resource_logits [3, num_resource_bins])
    pub fn forward(
        &self,
        graph_tensors: &GraphTensors,
        action_mask: &ActionMask,
    ) -> (Tensor, Tensor) {
        let full_embedding = self.han.forward(graph_tensors);

        let cluster_embedding =
            self.extract_cluster_embedding(&full_embedding, graph_tensors);
        let pending_embeddings =
            self.extract_pending_embeddings(&full_embedding, graph_tensors);

        let output =
            self.forward_detailed(&cluster_embedding, &pending_embeddings, action_mask);

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
            Tensor::zeros(
                &[3, self.resource_bins],
                (Kind::Float, cluster_embedding.device()),
            )
        };

        (output.task_probs, resource_logits)
    }

    pub fn forward_detailed(
        &self,
        cluster_embedding: &Tensor,
        pending_embeddings: &Tensor,
        action_mask: &ActionMask,
    ) -> ActorOutput {
        let no_action_emb = Tensor::zeros(
            &[1, self.hidden_dim],
            (pending_embeddings.kind(), pending_embeddings.device()),
        );
        let task_embeddings = Tensor::cat(&[&no_action_emb, pending_embeddings], 0);

        let encoded_tasks = self.task_encoder.forward(&task_embeddings);
        let query = self.pointer_query.forward(cluster_embedding);
        let keys = self.pointer_key.forward(&encoded_tasks);

        let scores = query.matmul(&keys.transpose(0, 1)).squeeze_dim(0);
        let masked_scores = scores + &action_mask.task_mask;
        let task_probs = masked_scores.softmax(0, Kind::Float);

        let task_action = task_probs.multinomial(1, true).squeeze();
        let task_idx = i64::from(&task_action);

        let resource_allocation = if task_idx > 0 {
            let selected_task_emb = encoded_tasks.get(task_idx);
            let context =
                Tensor::cat(&[cluster_embedding, &selected_task_emb.unsqueeze(0)], 1);

            let cpu_probs =
                (self.cpu_allocator.forward(&context).squeeze_dim(0) + &action_mask.cpu_mask)
                    .softmax(0, Kind::Float);
            let cpu_action = cpu_probs.multinomial(1, true);

            let gpu_probs =
                (self.gpu_allocator.forward(&context).squeeze_dim(0) + &action_mask.gpu_mask)
                    .softmax(0, Kind::Float);
            let gpu_action = gpu_probs.multinomial(1, true);

            let mem_probs =
                (self.memory_allocator.forward(&context).squeeze_dim(0)
                    + &action_mask.memory_mask)
                    .softmax(0, Kind::Float);
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

    fn extract_cluster_embedding(
        &self,
        full_embedding: &Tensor,
        graph: &GraphTensors,
    ) -> Tensor {
        if graph.cluster_indices.is_empty() {
            return Tensor::zeros(
                &[1, self.hidden_dim],
                (Kind::Float, full_embedding.device()),
            );
        }
        let idx =
            Tensor::of_slice(&graph.cluster_indices[0..1]).to_device(full_embedding.device());
        full_embedding.index_select(0, &idx)
    }

    fn extract_pending_embeddings(
        &self,
        full_embedding: &Tensor,
        graph: &GraphTensors,
    ) -> Tensor {
        if graph.pending_indices.is_empty() {
            return Tensor::zeros(
                &[0, self.hidden_dim],
                (Kind::Float, full_embedding.device()),
            );
        }
        let idx =
            Tensor::of_slice(&graph.pending_indices).to_device(full_embedding.device());
        full_embedding.index_select(0, &idx)
    }
}

// ============================================================================
// ACTION MASK
// ============================================================================

pub struct ActionMask {
    pub task_mask: Tensor,
    pub cpu_mask: Tensor,
    pub gpu_mask: Tensor,
    pub memory_mask: Tensor,
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

    pub fn mask_task(&mut self, idx: i64) {
        let _ = self.task_mask.get(idx).fill_(f64::NEG_INFINITY);
    }

    pub fn mask_cpu(&mut self, bin: i64) {
        let _ = self.cpu_mask.get(bin).fill_(f64::NEG_INFINITY);
    }

    pub fn mask_gpu(&mut self, idx: i64) {
        let _ = self.gpu_mask.get(idx).fill_(f64::NEG_INFINITY);
    }

    pub fn mask_memory(&mut self, bin: i64) {
        let _ = self.memory_mask.get(bin).fill_(f64::NEG_INFINITY);
    }

    /// Build a fully populated ActionMask from a live environment cluster.
    ///
    /// Task masking  — mask tasks where any single resource requirement
    ///                 cannot be met (cpu, any-gpu, memory).
    /// CPU masking   — mask bins requesting more cores than available.
    /// GPU masking   — mask GPU slots with zero available cores (slot 0 = no GPU, never masked).
    /// Memory masking — mask bins requesting more MB than available.
    pub fn from_environment(env: &EdgeMLEnv, cluster_id: usize, device: Device) -> Self {
        let cluster = &env.clusters[cluster_id];
        let num_pending = cluster.pending_tasks.len() as i64;
        let num_gpus = cluster.gpus.len() as i64;

        let mut mask = Self::new(num_pending, 17, num_gpus, 20, device);

        let cpu_available = cluster.total_cpu_available();
        let mem_available = cluster.total_memory_available();

        // ── Task masks ───────────────────────────────────────────────────────
        for (i, task) in cluster.pending_tasks.iter().enumerate() {
            let mask_idx = i as i64 + 1; // 0 is always the no-op slot

            if task.min_cpu > cpu_available {
                mask.mask_task(mask_idx);
                continue;
            }

            // Check if at least one GPU can satisfy both core and memory requirements
            if task.min_gpu_core > 0 {
                let fits = cluster.gpus.iter().any(|g| {
                    g.core_available >= task.min_gpu_core
                        && g.memory_available_mb >= task.min_gpu_memory
                });
                if !fits {
                    mask.mask_task(mask_idx);
                    continue;
                }
            }

            if task.min_memory_mb > mem_available {
                mask.mask_task(mask_idx);
            }
        }

        // ── CPU bin masks ────────────────────────────────────────────────────
        // bin 0 = 0 cores (valid no-resource request), bins 1–16 scale linearly
        for bin in 1i64..17 {
            let requested = (cpu_available * bin as usize) / 16;
            if requested > cpu_available {
                mask.mask_cpu(bin);
            }
        }

        // ── GPU slot masks ───────────────────────────────────────────────────
        // Slot 0 = no GPU — always valid.
        // Slot N = GPU (N-1). Mask if core_available == 0.
        for (gpu_id, gpu) in cluster.gpus.iter().enumerate() {
            if gpu.core_available == 0 {
                mask.mask_gpu(gpu_id as i64 + 1);
            }
        }

        // ── Memory bin masks ─────────────────────────────────────────────────
        for bin in 1i64..20 {
            let requested = (mem_available * bin as usize) / 20;
            if requested > mem_available {
                mask.mask_memory(bin);
            }
        }

        mask
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
