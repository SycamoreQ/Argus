#![allow(non_snake_case)]

// RL subsystem
pub mod RL {
    pub mod action;
    pub mod actor;
    pub mod argus;
    pub mod conflict_resolver;
    pub mod critic;
    pub mod env;
    pub mod mappo;
    pub mod network;
    pub mod reward;
    pub mod state;
}

// Neural network structures
pub mod structures {
    pub mod _abc;
    pub mod _enums;
    pub mod _events;
    pub mod _graph;
    pub mod _query;
    pub mod _struct;
}

// GPU resource management
pub mod gpu {
    pub mod allocate;
    pub mod cache;
    pub mod gpu;
    pub mod node;
    pub mod pod;
    pub mod rater;
}

// Database layer
pub mod database {
    pub mod low;
    pub mod memory;
    pub mod unified;
}

// Event brokers
pub mod eventbrokers {
    pub mod base;
    pub mod nats;
    pub mod redis;
}

// Planning (world model + MCTS)
pub mod planner {
    pub mod attn;
    pub mod world;
}

// Re-exports for convenience
pub use structures::_graph::{GraphTensors, HAN};
pub use RL::env::EdgeMLEnv;
pub use RL::mappo::MAPPOTrainer;
pub use planner::world::{ActionKey, WorldModel, MCTS};
