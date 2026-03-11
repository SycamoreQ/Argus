use serde::{Deserialize, Serialize};
use tracing::{debug, error, info};

// ============================================================================
// GPU UNIT — resource request from a single container
// ============================================================================

#[derive(Default, Clone, Debug, Serialize, Deserialize)]
pub struct GPUUnit {
    pub core: usize,
    pub memory: usize,
    pub gpu_count: usize,
}

impl std::fmt::Display for GPUUnit {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "core:{}, memory:{}, count:{}",
            self.core, self.memory, self.gpu_count
        )
    }
}

// ============================================================================
// SINGLE GPU
// ============================================================================

#[derive(Debug, Deserialize, Serialize, Clone, PartialEq, Eq)]
pub struct GPU {
    pub id: usize,
    pub core_available: usize,
    pub memory_available: usize,
    pub core_total: usize,
    pub memory_total: usize,
}

impl GPU {
    pub fn new(id: usize, core_total: usize, memory_total: usize) -> Self {
        Self {
            id,
            core_available: core_total,
            memory_available: memory_total,
            core_total,
            memory_total,
        }
    }

    /// Consume resources (allocation).
    pub fn add(&mut self, resource: &GPUUnit) {
        if resource.gpu_count > 0 {
            // Claiming the whole card
            self.core_available = 0;
            self.memory_available = 0;
        } else {
            self.core_available = self.core_available.saturating_sub(resource.core);
            self.memory_available = self.memory_available.saturating_sub(resource.memory);
        }
    }

    /// Release resources (de-allocation).
    pub fn sub(&mut self, resource: &GPUUnit) {
        if resource.gpu_count > 0 {
            // Releasing the whole card
            self.core_available = self.core_total;
            self.memory_available = self.memory_total;
        } else {
            self.core_available = (self.core_available + resource.core).min(self.core_total);
            self.memory_available =
                (self.memory_available + resource.memory).min(self.memory_total);
        }
    }

    pub fn can_allocate(&self, resource: &GPUUnit) -> bool {
        if resource.gpu_count > 0 {
            // Needs the full card — only allocatable if completely free
            self.core_available == self.core_total && self.memory_available == self.memory_total
        } else {
            self.core_available >= resource.core && self.memory_available >= resource.memory
        }
    }
}

// ============================================================================
// GPU COLLECTION
// ============================================================================

/// Newtype wrapper so we can implement helper methods on `Vec<GPU>`.
#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct GPUs(pub Vec<GPU>);

impl std::ops::Index<usize> for GPUs {
    type Output = GPU;
    fn index(&self, idx: usize) -> &GPU {
        &self.0[idx]
    }
}

impl std::ops::IndexMut<usize> for GPUs {
    fn index_mut(&mut self, idx: usize) -> &mut GPU {
        &mut self.0[idx]
    }
}

impl GPUs {
    pub fn new(gpus: Vec<GPU>) -> Self {
        Self(gpus)
    }

    pub fn len(&self) -> usize {
        self.0.len()
    }

    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    pub fn iter(&self) -> std::slice::Iter<'_, GPU> {
        self.0.iter()
    }

    pub fn iter_mut(&mut self) -> std::slice::IterMut<'_, GPU> {
        self.0.iter_mut()
    }

    /// Returns indices of GPUs with 100% available resources.
    pub fn get_free_gpus(&self) -> Vec<usize> {
        self.0
            .iter()
            .enumerate()
            .filter(|(_, gpu)| {
                gpu.core_available == gpu.core_total && gpu.memory_available == gpu.memory_total
            })
            .map(|(i, _)| i)
            .collect()
    }

    /// DFS-based allocation search across containers.
    /// Returns a `GPUOption` describing which GPU indices serve each container.
    pub fn trade(
        &mut self,
        rater: &dyn Rater,
        request: &[GPUUnit],
    ) -> Result<GPUOption, Box<dyn std::error::Error>> {
        let mut indexes: Vec<Vec<usize>> = vec![vec![]; request.len()];
        let mut found = false;
        let mut option = GPUOption::new(request.to_vec());

        Self::dfs(0, self, request, &mut indexes, &mut found, &mut option, rater);

        if !found {
            return Err("no enough resource to allocate".into());
        }
        Ok(option)
    }

    fn dfs(
        container_idx: usize,
        gpus: &mut GPUs,
        request: &[GPUUnit],
        indexes: &mut Vec<Vec<usize>>,
        found: &mut bool,
        option: &mut GPUOption,
        rater: &dyn Rater,
    ) {
        if container_idx == request.len() {
            *found = true;

            let rate_indexes: Vec<i32> = indexes
                .iter()
                .map(|idx_list| {
                    if idx_list.len() == 1 {
                        idx_list[0] as i32
                    } else {
                        -1
                    }
                })
                .collect();

            let curr_score = rater.rate(gpus, &rate_indexes);
            if option.score >= curr_score {
                return;
            }

            for (i, gpu_indices) in indexes.iter().enumerate() {
                option.allocated[i] = gpu_indices.clone();
            }
            option.score = curr_score;
            return;
        }

        let req = &request[container_idx];
        info!("Allocating for container {}", container_idx);

        if req.gpu_count > 0 {
            let free_gpus = gpus.get_free_gpus();
            if free_gpus.len() < req.gpu_count {
                return;
            }

            let selected = free_gpus[..req.gpu_count].to_vec();
            for &gpu_idx in &selected {
                gpus[gpu_idx].add(req);
            }
            indexes[container_idx] = selected.clone();

            Self::dfs(container_idx + 1, gpus, request, indexes, found, option, rater);

            for &gpu_idx in &selected {
                gpus[gpu_idx].sub(req);
            }
        } else {
            for i in 0..gpus.len() {
                if gpus[i].can_allocate(req) {
                    gpus[i].add(req);
                    indexes[container_idx] = vec![i];

                    Self::dfs(container_idx + 1, gpus, request, indexes, found, option, rater);

                    gpus[i].sub(req);
                }
            }
        }
    }

    /// Commit an option to the GPUs (deduct resources).
    pub fn transact(&mut self, option: &GPUOption) -> Result<(), String> {
        debug!("Transacting option on GPUs");

        for (i, allocation) in option.allocated.iter().enumerate() {
            let request = &option.request[i];

            if request.gpu_count > 0 {
                for &gpu_index in allocation {
                    if !self[gpu_index].can_allocate(request) {
                        let msg = format!("GPU {} insufficient resources", gpu_index);
                        error!("{}", msg);
                        return Err(msg);
                    }
                    self[gpu_index].add(request);
                }
            } else if let Some(&gpu_index) = allocation.first() {
                if !self[gpu_index].can_allocate(request) {
                    let msg = format!("GPU {} insufficient resources", gpu_index);
                    error!("{}", msg);
                    return Err(msg);
                }
                self[gpu_index].add(request);
            }
        }
        Ok(())
    }

    /// Undo an option (release resources).
    pub fn cancel(&mut self, option: &GPUOption) -> Result<(), String> {
        debug!("Cancelling GPU option");

        for (i, request) in option.request.iter().enumerate() {
            let allocation = &option.allocated[i];

            if request.gpu_count > 0 {
                for &gpu_index in allocation {
                    self[gpu_index].sub(request);
                }
            } else if let Some(&gpu_index) = allocation.first() {
                self[gpu_index].sub(request);
            }
        }
        Ok(())
    }
}

// ============================================================================
// GPU OPTION — result of an allocation search
// ============================================================================

#[derive(Debug, Clone)]
pub struct GPUOption {
    pub request: Vec<GPUUnit>,
    pub allocated: Vec<Vec<usize>>,
    pub score: f64,
}

impl GPUOption {
    pub fn new(request: Vec<GPUUnit>) -> Self {
        let len = request.len();
        Self {
            request,
            allocated: vec![vec![]; len],
            score: f64::NEG_INFINITY,
        }
    }
}

// ============================================================================
// RATER TRAIT & IMPLEMENTATIONS
// ============================================================================

pub trait Rater: Send + Sync {
    fn rate(&self, gpus: &GPUs, indexes: &[i32]) -> f64;
}

/// BinPacker: prefers allocations that minimise fragmentation
/// (smaller range of remaining resources = better packing).
pub struct BinPacker;

impl Rater for BinPacker {
    fn rate(&self, gpus: &GPUs, indexes: &[i32]) -> f64 {
        let mut gpu_used: Vec<bool> = vec![false; gpus.len()];
        let mut gpu_count = 0usize;

        for &idx in indexes {
            if idx >= 0 {
                let i = idx as usize;
                if !gpu_used[i] {
                    gpu_used[i] = true;
                    gpu_count += 1;
                }
            }
        }

        if gpus.is_empty() {
            return 0.0;
        }

        let mut max_mem = gpus[0].memory_available;
        let mut min_mem = gpus[0].memory_available;
        let mut max_core = gpus[0].core_available;
        let mut min_core = gpus[0].core_available;

        for gpu in gpus.iter() {
            max_mem = max_mem.max(gpu.memory_available);
            min_mem = min_mem.min(gpu.memory_available);
            max_core = max_core.max(gpu.core_available);
            min_core = min_core.min(gpu.core_available);
        }

        let range = ((max_mem + max_core).saturating_sub(min_mem + min_core)) as f64 / 2.0;
        // Lower range = better packing → higher score
        if gpu_count == 0 {
            0.0
        } else {
            -(range / (gpu_count + 1) as f64)
        }
    }
}

/// Spread: prefers allocations that distribute load evenly.
pub struct Spread;

impl Rater for Spread {
    fn rate(&self, gpus: &GPUs, indexes: &[i32]) -> f64 {
        // Count how many distinct GPUs are used
        let used: std::collections::HashSet<i32> =
            indexes.iter().filter(|&&i| i >= 0).cloned().collect();

        // More GPUs used = better spread = higher score
        used.len() as f64
    }
}
