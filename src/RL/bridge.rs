use crate::RL::env::{AgentAction, ClusterResources, SchedulingContext};
use crate::planner::world::ActionKey;

// ============================================================================
// RESOURCE BINS
// ============================================================================

/// Translates bin indices produced by the actor into actual resource quantities
/// using the cluster's current totals as the scaling reference.
pub struct ResourceBins {
    pub total_cpu_cores: usize,
    pub gpu_total_cores: usize,
    pub total_memory_mb: usize,
}

impl ResourceBins {
    pub fn from_cluster_resources(res: &ClusterResources) -> Self {
        // Use the sum of available GPU cores as the per-GPU reference.
        // If no GPUs, default to 100 (standard unit).
        let gpu_total = res.gpu_available.first().copied().unwrap_or(100);

        Self {
            total_cpu_cores: res.cpu_available.max(1),
            gpu_total_cores: gpu_total.max(1),
            total_memory_mb: res.memory_available.max(1),
        }
    }

    /// cpu_bin is 0–16 → maps linearly to 0–total_cpu_cores
    pub fn cpu_bin_to_cores(&self, bin: i64) -> usize {
        (self.total_cpu_cores * bin as usize) / 16
    }

    /// mem_bin is 0–19 → maps linearly to 0–total_memory_mb
    pub fn mem_bin_to_mb(&self, bin: i64) -> usize {
        (self.total_memory_mb * bin as usize) / 20
    }

    /// gpu_bin is 0–16 → maps linearly to 0–gpu_total_cores
    pub fn gpu_bin_to_cores(&self, bin: i64) -> usize {
        (self.gpu_total_cores * bin as usize) / 16
    }
}

// ============================================================================
// ACTION CONVERSION
// ============================================================================

/// Convert an ActionKey (RL space) into an AgentAction (environment space).
///
/// ActionKey field conventions:
///   task_idx == 0  → no-op
///   gpu_idx  == 0  → no GPU requested; gpu_idx >= 1 maps to GPU (gpu_idx - 1)
///
/// The task_id string is looked up from the SchedulingContext's TaskLookup
/// using the graph node index stored in task_idx.
pub fn action_key_to_agent_action(
    key: &ActionKey,
    ctx: &SchedulingContext,
    bins: &ResourceBins,
) -> AgentAction {
    // No-op
    if key.task_idx == 0 {
        return AgentAction::no_op();
    }

    // Look up the real task ID from the graph node index
    let task_id = ctx
        .task_lookup
        .get(&key.task_idx)
        .map(|info| info.task_id.clone());

    // If the index isn't in the lookup the task doesn't exist — treat as no-op
    if task_id.is_none() {
        return AgentAction::no_op();
    }

    // GPU: index 0 = no GPU, index N = GPU (N-1)
    let (allocated_gpu_id, allocated_gpu_cores) = if key.gpu_idx > 0 {
        let gpu_id = (key.gpu_idx - 1) as usize;
        let cores = bins.gpu_bin_to_cores(key.cpu_bin).max(1);
        (Some(gpu_id), cores)
    } else {
        (None, 0)
    };

    AgentAction {
        selected_task: task_id,
        allocated_cpu: bins.cpu_bin_to_cores(key.cpu_bin).max(1),
        allocated_gpu_id,
        allocated_gpu_cores,
        allocated_memory_mb: bins.mem_bin_to_mb(key.mem_bin).max(1),
    }
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::RL::env::{ClusterResources, TaskInfo, TaskLookup, MLTaskType};
    use crate::structures::_graph::GraphTensors;
    use tch::{Device, Kind, Tensor};

    fn dummy_graph_tensors() -> GraphTensors {
        GraphTensors {
            node_features: Tensor::zeros(&[2, 32], (Kind::Float, Device::Cpu)),
            node_types: Tensor::zeros(&[2], (Kind::Int64, Device::Cpu)),
            edge_index: Tensor::zeros(&[2, 0], (Kind::Int64, Device::Cpu)),
            edge_types: Tensor::zeros(&[0], (Kind::Int64, Device::Cpu)),
            num_nodes: 2,
            cluster_indices: vec![0],
            pending_indices: vec![1],
            running_indices: vec![],
        }
    }

    fn dummy_context() -> SchedulingContext {
        let mut task_lookup = TaskLookup::new();
        task_lookup.insert(
            1,
            TaskInfo {
                task_id: "task-abc".to_string(),
                min_cpu: 4,
                min_gpu_core: 25,
                min_gpu_memory: 2000,
                min_memory_mb: 4096,
                task_type: MLTaskType::LinearRegression,
            },
        );

        SchedulingContext {
            graph: dummy_graph_tensors(),
            task_lookup,
            cluster_resources: ClusterResources {
                cpu_available: 64,
                gpu_available: vec![100, 100, 100, 100],
                memory_available: 32768,
            },
        }
    }

    #[test]
    fn test_no_op() {
        let key = ActionKey::no_op();
        let ctx = dummy_context();
        let bins = ResourceBins::from_cluster_resources(&ctx.cluster_resources);
        let action = action_key_to_agent_action(&key, &ctx, &bins);
        assert!(action.selected_task.is_none());
        assert_eq!(action.allocated_cpu, 0);
    }

    #[test]
    fn test_valid_action() {
        let key = ActionKey {
            task_idx: 1,
            cpu_bin: 4,   // 4/16 * 64 = 16 cores
            gpu_idx: 1,   // GPU 0
            mem_bin: 5,   // 5/20 * 32768 = 8192 MB
        };
        let ctx = dummy_context();
        let bins = ResourceBins::from_cluster_resources(&ctx.cluster_resources);
        let action = action_key_to_agent_action(&key, &ctx, &bins);

        assert_eq!(action.selected_task, Some("task-abc".to_string()));
        assert_eq!(action.allocated_cpu, 16);
        assert_eq!(action.allocated_gpu_id, Some(0));
        assert_eq!(action.allocated_memory_mb, 8192);
    }

    #[test]
    fn test_missing_task_idx_is_noop() {
        let key = ActionKey {
            task_idx: 99, // not in lookup
            cpu_bin: 4,
            gpu_idx: 0,
            mem_bin: 5,
        };
        let ctx = dummy_context();
        let bins = ResourceBins::from_cluster_resources(&ctx.cluster_resources);
        let action = action_key_to_agent_action(&key, &ctx, &bins);
        assert!(action.selected_task.is_none());
    }
}
