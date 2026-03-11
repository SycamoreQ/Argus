// Rater trait and implementations have been consolidated into gpu/gpu.rs.
// This file re-exports them for any callers that import from `rater` directly.
pub use crate::gpu::gpu::{BinPacker, GPUOption, GPUs, Rater, Spread};
