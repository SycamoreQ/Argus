use crate::RL::actor::TapFingerActor;
use crate::RL::critic::TapFingerCritic;
use crate::structures::_graph::GraphTensors;
use std::collections::HashMap;
use tch::{nn, Device, Kind, Tensor};

pub struct MAPPOTrainer {
    pub actors: Vec<TapFingerActor>,
    pub critics: Vec<TapFingerCritic>,
    optimizer: nn::Optimizer,
    clip_epsilon: f64,
    value_loss_coef: f64,
    entropy_coef: f64,
}

impl MAPPOTrainer {
    pub fn new(
        vs: &nn::VarStore,
        num_agents: usize,
        input_dim: i64,
        hidden_dim: i64,
        num_resource_bins: i64,
        learning_rate: f64,
    ) -> Self {
        let mut actors = Vec::new();
        let mut critics = Vec::new();

        for i in 0..num_agents {
            let actor_vs = vs.root() / format!("actor_{}", i);
            let critic_vs = vs.root() / format!("critic_{}", i);

            actors.push(TapFingerActor::new(&actor_vs, input_dim, hidden_dim, num_resource_bins));
            critics.push(TapFingerCritic::new(&critic_vs, hidden_dim));
        }

        let optimizer = nn::Adam::default().build(vs, learning_rate).unwrap();

        Self {
            actors,
            critics,
            optimizer,
            clip_epsilon: 0.2,
            value_loss_coef: 0.5,
            entropy_coef: 0.01,
        }
    }

    /// Compute clipped PPO loss for all agents.
    /// `returns` are the bootstrapped target values used for critic regression.
    pub fn compute_loss(
        &self,
        states: &[GraphTensors],
        actions: &[Tensor],
        old_log_probs: &[Tensor],
        advantages: &[Tensor],
        returns: &[Tensor],
        masks: &[crate::RL::actor::ActionMask],
    ) -> Tensor {
        let mut total_loss = Tensor::zeros(&[], (Kind::Float, Device::Cpu));

        for (i, actor) in self.actors.iter().enumerate() {
            let (task_probs, _resource_logits) = actor.forward(&states[i], &masks[i]);

            // Log-probs for actions taken
            let log_probs = task_probs.log();
            let action_log_probs = log_probs.gather(0, &actions[i], false);

            // PPO clipped surrogate objective
            let ratio = (action_log_probs - &old_log_probs[i]).exp();
            let surr1 = &ratio * &advantages[i];
            let surr2 = ratio
                .clamp(1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon)
                * &advantages[i];
            let policy_loss = -surr1.min_other(&surr2).mean(Kind::Float);

            // Critic (value) loss
            let value = self.critics[i].forward(&states[i].node_features);
            let value_loss = (value - &returns[i])
                .pow_tensor_scalar(2)
                .mean(Kind::Float);

            // Entropy bonus to encourage exploration
            let entropy = -(task_probs.clamp_min(1e-8) * task_probs.clamp_min(1e-8).log())
                .sum_dim_intlist(&[0i64][..], false, Kind::Float)
                .mean(Kind::Float);

            total_loss = total_loss
                + policy_loss
                + self.value_loss_coef * value_loss
                - self.entropy_coef * entropy;
        }

        total_loss
    }

    pub fn train_step(&mut self, batch: TrainingBatch) -> f64 {
        self.optimizer.zero_grad();

        let loss = self.compute_loss(
            &batch.states,
            &batch.actions,
            &batch.old_log_probs,
            &batch.advantages,
            &batch.returns,
            &batch.masks,
        );

        loss.backward();
        self.optimizer.step();

        f64::try_from(&loss).unwrap_or(0.0)
    }

    /// Compute Generalised Advantage Estimation (GAE).
    /// Call this after collecting a rollout before calling `train_step`.
    pub fn compute_advantages(
        &self,
        rewards: &[f64],
        values: &[f64],
        dones: &[bool],
        gamma: f64,
        lam: f64,
    ) -> (Vec<f64>, Vec<f64>) {
        let n = rewards.len();
        let mut advantages = vec![0.0f64; n];
        let mut returns = vec![0.0f64; n];
        let mut gae = 0.0f64;

        for t in (0..n).rev() {
            let next_value = if t + 1 < n && !dones[t] { values[t + 1] } else { 0.0 };
            let delta = rewards[t] + gamma * next_value - values[t];
            gae = delta + gamma * lam * if dones[t] { 0.0 } else { gae };
            advantages[t] = gae;
            returns[t] = advantages[t] + values[t];
        }

        (advantages, returns)
    }
}

pub struct TrainingBatch {
    pub states: Vec<GraphTensors>,
    pub actions: Vec<Tensor>,
    pub old_log_probs: Vec<Tensor>,
    pub advantages: Vec<Tensor>,
    pub returns: Vec<Tensor>,
    pub masks: Vec<crate::RL::actor::ActionMask>,
}
