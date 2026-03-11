use std::sync::Arc;
use tch::{nn, Device, Kind, Tensor};

use crate::planner::world::{ActionKey, WorldModel, MCTS};
use crate::RL::actor::TapFingerActor;
use crate::RL::critic::TapFingerCritic;
use crate::structures::_graph::GraphTensors;

pub struct MuZeroScheduler {
    pub world_model: Arc<WorldModel>,
    pub actor: Arc<TapFingerActor>,
    pub critic: Arc<TapFingerCritic>,
    pub mcts: MCTS,

    vs: nn::VarStore,
    optimizer: nn::Optimizer,

    model_buffer: Vec<ModelTransition>,
}

#[derive(Clone)]
pub struct ModelTransition {
    pub state: GraphTensors,
    pub action: ActionKey,
    pub reward: f64,
    pub next_state: GraphTensors,
}

impl MuZeroScheduler {
    pub fn new(
        vs: nn::VarStore,
        state_dim: i64,
        action_dim: i64,
        hidden_dim: i64,
        input_dim: i64,
        num_resource_bins: i64,
    ) -> Self {
        let world_model = Arc::new(WorldModel::new(
            &vs.root() / "world_model",
            state_dim,
            action_dim,
            hidden_dim,
        ));

        let actor = Arc::new(TapFingerActor::new(
            &vs.root() / "actor",
            input_dim,
            hidden_dim,
            num_resource_bins,
        ));

        let critic = Arc::new(TapFingerCritic::new(&vs.root() / "critic", hidden_dim));

        let mcts = MCTS::new(
            Arc::clone(&world_model),
            Arc::clone(&actor),
            50,
        );

        let optimizer = nn::Adam::default().build(&vs, 1e-4).unwrap();

        Self {
            world_model,
            actor,
            critic,
            mcts,
            vs,
            optimizer,
            model_buffer: Vec::new(),
        }
    }

    /// Select action using MCTS planning
    pub fn select_action(&self, state: &GraphTensors) -> ActionKey {
        let (action, _value) = self.mcts.search(state);
        action
    }

    /// Store a transition for world model training
    pub fn store_transition(&mut self, transition: ModelTransition) {
        self.model_buffer.push(transition);
        if self.model_buffer.len() > 10_000 {
            self.model_buffer.remove(0);
        }
    }

    /// Train the world model on a random mini-batch from the buffer
    pub fn train_world_model(&mut self, batch_size: usize) -> f64 {
        if self.model_buffer.len() < batch_size {
            return 0.0;
        }

        // Sample without replacement using index shuffling
        let n = self.model_buffer.len();
        let mut indices: Vec<usize> = (0..n).collect();
        // Fisher-Yates partial shuffle for batch_size elements
        for i in 0..batch_size {
            let j = i + (pseudo_rand(i as u64) as usize % (n - i));
            indices.swap(i, j);
        }
        let batch: Vec<&ModelTransition> =
            indices[..batch_size].iter().map(|&i| &self.model_buffer[i]).collect();

        let device = Device::cuda_if_available();

        let states: Vec<Tensor> = batch
            .iter()
            .map(|t| t.state.node_features.shallow_clone())
            .collect();

        let actions: Vec<ActionKey> = batch.iter().map(|t| t.action.clone()).collect();
        let rewards: Vec<f64> = batch.iter().map(|t| t.reward).collect();

        self.optimizer.zero_grad();
        let loss = self.world_model.train_step(&states, &actions, &rewards);
        loss.backward();
        self.optimizer.step();

        f64::try_from(&loss).unwrap_or(0.0)
    }

    pub fn train_step(&mut self, _batch: &[ModelTransition]) -> TrainingMetrics {
        let model_loss = self.train_world_model(32);

        // Policy improvement via MCTS targets is implemented in Milestone 4
        TrainingMetrics {
            model_loss,
            policy_loss: 0.0,
            value_loss: 0.0,
        }
    }
}

pub struct TrainingMetrics {
    pub model_loss: f64,
    pub policy_loss: f64,
    pub value_loss: f64,
}

/// Minimal pseudo-random helper (replace with `rand` crate in Milestone 4)
fn pseudo_rand(seed: u64) -> u64 {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};
    let mut h = DefaultHasher::new();
    seed.hash(&mut h);
    h.finish()
}
