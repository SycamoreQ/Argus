use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use tch::nn::Module;
use tch::{nn, Device, Kind, Tensor};

const N_TASKS: usize = 100;

// ============================================================================
// WORLD MODEL
// ============================================================================

pub struct WorldModel {
    pub representation_net: nn::Sequential,
    pub action_encoder: nn::Linear,
    pub dynamics_net: nn::Sequential,
    pub reward_net: nn::Sequential,
    pub value_net: nn::Sequential,
    pub hidden_dim: i64,
}

impl WorldModel {
    pub fn new(vs: &nn::Path, state_dim: i64, action_dim: i64, hidden_dim: i64) -> Self {
        let representation_net = nn::seq()
            .add(nn::linear(
                vs / "repr_1",
                state_dim,
                hidden_dim * 2,
                Default::default(),
            ))
            .add_fn(|x| x.relu())
            .add(nn::linear(
                vs / "repr_2",
                hidden_dim * 2,
                hidden_dim,
                Default::default(),
            ))
            .add_fn(|x| x.tanh());

        let action_encoder = nn::linear(
            vs / "action_enc",
            action_dim,
            hidden_dim / 4,
            Default::default(),
        );

        let dynamics_net = nn::seq()
            .add(nn::linear(
                vs / "dyn_1",
                hidden_dim + hidden_dim / 4,
                hidden_dim * 2,
                Default::default(),
            ))
            .add_fn(|x| x.relu())
            .add(nn::linear(
                vs / "dyn_2",
                hidden_dim * 2,
                hidden_dim,
                Default::default(),
            ))
            .add_fn(|x| x.tanh());

        let reward_net = nn::seq()
            .add(nn::linear(
                vs / "rew_1",
                hidden_dim,
                128,
                Default::default(),
            ))
            .add_fn(|x| x.relu())
            .add(nn::linear(vs / "rew_2", 128, 1, Default::default()));

        let value_net = nn::seq()
            .add(nn::linear(
                vs / "val_1",
                hidden_dim,
                128,
                Default::default(),
            ))
            .add_fn(|x| x.relu())
            .add(nn::linear(vs / "val_2", 128, 1, Default::default()));

        Self {
            representation_net,
            action_encoder,
            dynamics_net,
            reward_net,
            value_net,
            hidden_dim,
        }
    }

    pub fn represent(&self, state: &Tensor) -> Tensor {
        self.representation_net.forward(state)
    }

    pub fn step(&self, latent_state: &Tensor, action: &ActionKey) -> WorldModelOutput {
        let device = latent_state.device();

        let action_data = vec![
            action.task_idx as f32 / N_TASKS as f32,
            action.cpu_bin as f32 / 16.0,
            action.gpu_idx as f32 / 8.0,
            action.mem_bin as f32 / 20.0,
        ];
        let action_tensor = Tensor::of_slice(&action_data).to_device(device);

        let action_proj = self.action_encoder.forward(&action_tensor);
        let dyn_input = Tensor::cat(&[latent_state, &action_proj], -1);
        let next_latent = self.dynamics_net.forward(&dyn_input);

        let reward = self.reward_net.forward(&next_latent);
        let value = self.value_net.forward(&next_latent);

        WorldModelOutput {
            next_latent_state: next_latent,
            reward,
            value,
        }
    }

    pub fn predict_reward(&self, latent_state: &Tensor) -> Tensor {
        self.reward_net.forward(latent_state)
    }

    pub fn predict_value(&self, latent_state: &Tensor) -> Tensor {
        self.value_net.forward(latent_state)
    }

    pub fn unroll(&self, initial_latent: &Tensor, actions: &[ActionKey]) -> Vec<WorldModelOutput> {
        let mut outputs = Vec::new();
        let mut current_latent = initial_latent.shallow_clone();

        for action in actions {
            let output = self.step(&current_latent, action);
            current_latent = output.next_latent_state.shallow_clone();
            outputs.push(output);
        }

        outputs
    }

    pub fn train_step(
        &self,
        real_states: &[Tensor],
        actions: &[ActionKey],
        real_rewards: &[f64],
    ) -> Tensor {
        let mut latent = self.represent(&real_states[0]);
        let mut total_loss = Tensor::zeros(&[], (Kind::Float, Device::cuda_if_available()));

        for i in 0..actions.len().min(real_states.len() - 1) {
            let output = self.step(&latent, &actions[i]);
            let real_next_latent = self.represent(&real_states[i + 1]);

            let state_loss = (&output.next_latent_state - &real_next_latent)
                .pow_tensor_scalar(2)
                .mean(Kind::Float);

            let reward_loss = (&output.reward - real_rewards[i])
                .pow_tensor_scalar(2)
                .mean(Kind::Float);

            total_loss = total_loss + state_loss + reward_loss;
            latent = output.next_latent_state;
        }

        total_loss / actions.len() as f64
    }
}

#[derive(Clone)]
pub struct WorldModelOutput {
    pub next_latent_state: Tensor,
    pub reward: Tensor,
    pub value: Tensor,
}

// ============================================================================
// ACTION KEY
// ============================================================================

#[derive(Debug, Clone, Hash, Eq, PartialEq)]
pub struct ActionKey {
    pub task_idx: i64,
    pub cpu_bin: i64,
    pub gpu_idx: i64,
    pub mem_bin: i64,
}

impl ActionKey {
    pub fn no_op() -> Self {
        Self {
            task_idx: 0,
            cpu_bin: 0,
            gpu_idx: 0,
            mem_bin: 0,
        }
    }

    pub fn to_tensor(&self, device: Device) -> Tensor {
        Tensor::of_slice(&[
            self.task_idx as f32,
            self.cpu_bin as f32,
            self.gpu_idx as f32,
            self.mem_bin as f32,
        ])
        .to_device(device)
    }
}

// ============================================================================
// MCTS NODE
// ============================================================================

pub struct MCTSNode {
    pub latent_state: Tensor,
    pub parent: Option<Arc<Mutex<MCTSNode>>>,
    pub children: HashMap<ActionKey, Arc<Mutex<MCTSNode>>>,
    pub visit_count: usize,
    pub total_value: f64,
    pub prior_prob: f64,
    pub action: Option<ActionKey>,
    pub reward: f64,
}

impl MCTSNode {
    pub fn new(latent_state: Tensor) -> Self {
        Self {
            latent_state,
            parent: None,
            children: HashMap::new(),
            visit_count: 0,
            total_value: 0.0,
            prior_prob: 1.0,
            action: None,
            reward: 0.0,
        }
    }

    pub fn q_value(&self) -> f64 {
        if self.visit_count == 0 {
            0.0
        } else {
            self.total_value / self.visit_count as f64
        }
    }

    pub fn ucb_score(&self, parent_visits: usize, c_puct: f64) -> f64 {
        let exploitation = self.q_value();
        let exploration = c_puct * self.prior_prob * (parent_visits as f64).sqrt()
            / (1.0 + self.visit_count as f64);
        exploitation + exploration
    }

    pub fn select_child(&self, c_puct: f64) -> Option<(ActionKey, Arc<Mutex<MCTSNode>>)> {
        if self.children.is_empty() {
            return None;
        }

        let parent_visits = self.visit_count;

        self.children
            .iter()
            .max_by(|(_, a), (_, b)| {
                let score_a = a.lock().unwrap().ucb_score(parent_visits, c_puct);
                let score_b = b.lock().unwrap().ucb_score(parent_visits, c_puct);
                score_a.partial_cmp(&score_b).unwrap()
            })
            .map(|(action, node)| (action.clone(), Arc::clone(node)))
    }

    pub fn add_child(
        &mut self,
        action: ActionKey,
        child_latent: Tensor,
        prior: f64,
        reward: f64,
    ) -> Arc<Mutex<MCTSNode>> {
        let child = MCTSNode {
            latent_state: child_latent,
            parent: None,
            children: HashMap::new(),
            reward,
            visit_count: 0,
            total_value: 0.0,
            prior_prob: prior,
            action: Some(action.clone()),
        };

        let child_ref = Arc::new(Mutex::new(child));
        self.children.insert(action, Arc::clone(&child_ref));
        child_ref
    }

    pub fn backpropagate(&mut self, value: f64) {
        self.visit_count += 1;
        self.total_value += value;
    }

    pub fn is_leaf(&self) -> bool {
        self.children.is_empty()
    }
}

// ============================================================================
// MCTS
// ============================================================================

use crate::structures::_graph::GraphTensors;
use crate::RL::actor::{ActionMask, TapFingerActor};

pub struct MCTS {
    pub world_model: Arc<WorldModel>,
    pub actor: Arc<TapFingerActor>,
    pub c_puct: f64,
    pub num_simulations: usize,
    pub max_depth: usize,
    pub gamma: f64,
}

impl MCTS {
    pub fn new(
        world_model: Arc<WorldModel>,
        actor: Arc<TapFingerActor>,
        num_simulations: usize,
    ) -> Self {
        Self {
            world_model,
            actor,
            c_puct: 1.5,
            num_simulations,
            max_depth: 10,
            gamma: 0.99,
        }
    }

    pub fn search(&self, root_state: &GraphTensors) -> (ActionKey, f64) {
        let root_latent = self.world_model.represent(&root_state.node_features);
        let root = Arc::new(Mutex::new(MCTSNode::new(root_latent)));

        for _ in 0..self.num_simulations {
            self.simulate(Arc::clone(&root), root_state);
        }

        let root_lock = root.lock().unwrap();

        let (best_action, best_child) = root_lock
            .children
            .iter()
            .max_by_key(|(_, child)| child.lock().unwrap().visit_count)
            .map(|(action, child)| (action.clone(), Arc::clone(child)))
            .expect("No children found in root — tree was never expanded");

        let value = best_child.lock().unwrap().q_value();
        (best_action, value)
    }

    fn simulate(&self, root: Arc<Mutex<MCTSNode>>, state: &GraphTensors) {
        let mut path: Vec<(Arc<Mutex<MCTSNode>>, ActionKey)> = Vec::new();
        let mut current = Arc::clone(&root);
        let mut depth = 0;

        // Selection: walk down tree using UCB
        loop {
            let is_leaf = current.lock().unwrap().is_leaf();
            if is_leaf || depth >= self.max_depth {
                break;
            }

            let child_opt = current.lock().unwrap().select_child(self.c_puct);
            match child_opt {
                Some((action, child)) => {
                    path.push((Arc::clone(&current), action));
                    current = child;
                    depth += 1;
                }
                None => break,
            }
        }

        // Expansion
        let should_expand = {
            let node = current.lock().unwrap();
            node.visit_count >= 1 && node.is_leaf() && depth < self.max_depth
        };

        if should_expand {
            let latent_state = current.lock().unwrap().latent_state.shallow_clone();
            let device = latent_state.device();

            let mask = ActionMask::new(state.pending_indices.len() as i64, 17, 8, 20, device);

            let (task_probs, _resource_logits) = self.actor.forward(state, &mask);

            // Expand top-K actions by probability
            let num_actions = task_probs.size()[0].min(8);
            let top_indices = task_probs.argsort(0, true).i(..num_actions);
            let top_indices_vec: Vec<i64> = top_indices.into();

            for task_idx in top_indices_vec {
                let prob = f64::try_from(task_probs.get(task_idx)).unwrap_or(0.0);
                let action = ActionKey {
                    task_idx,
                    cpu_bin: 8, // default mid-range bins; refined in Milestone 3
                    gpu_idx: 0,
                    mem_bin: 10,
                };

                let output = self.world_model.step(&latent_state, &action);
                let pred_reward = f64::try_from(&output.reward.squeeze()).unwrap_or(0.0);

                current.lock().unwrap().add_child(
                    action,
                    output.next_latent_state,
                    prob,
                    pred_reward,
                );
            }

            // Move into the best child for evaluation
            let child_opt = current.lock().unwrap().select_child(self.c_puct);
            if let Some((action, child)) = child_opt {
                path.push((Arc::clone(&current), action));
                current = child;
            }
        }

        // Evaluation via value network
        let mut value = {
            let latent = current.lock().unwrap().latent_state.shallow_clone();
            let v = self.world_model.predict_value(&latent);
            f64::try_from(v.squeeze()).unwrap_or(0.0)
        };

        // Backpropagation
        current.lock().unwrap().backpropagate(value);
        for (node_arc, _) in path.iter().rev() {
            let reward = node_arc.lock().unwrap().reward;
            value = reward + self.gamma * value;
            node_arc.lock().unwrap().backpropagate(value);
        }
    }

    fn select_top_k_actions(
        &self,
        action_probs: &[(ActionKey, f64)],
        k: usize,
    ) -> Vec<(ActionKey, f64)> {
        let mut sorted = action_probs.to_vec();
        sorted.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        sorted.into_iter().take(k).collect()
    }
}
