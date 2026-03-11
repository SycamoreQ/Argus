use tch::{nn, nn::Module, Device, Kind, Tensor};

pub struct ActorCriticNet {
    actor: nn::Sequential,
    critic: nn::Sequential,
    device: Device,
}

impl ActorCriticNet {
    pub fn new(vs: &nn::Path, state_dim: i64, num_gpus: usize) -> Self {
        let device = vs.device();

        let action_dim = num_gpus + 3;

        // Fixed: `default::Default()` → `Default::default()`
        // Fixed: shared_layers cannot be clone()d directly in tch — build actor/critic with their own layers
        let actor = nn::seq()
            .add(nn::linear(vs / "shared_1", state_dim, 256, Default::default()))
            .add_fn(|x| x.relu())
            .add(nn::linear(vs / "shared_2", 256, 128, Default::default()))
            .add_fn(|x| x.relu())
            .add(nn::linear(vs / "actor1", 128, 128, Default::default()))
            .add_fn(|x| x.relu())
            .add(nn::linear(vs / "actor_out", 128, action_dim as i64, Default::default()));

        let critic = nn::seq()
            .add(nn::linear(vs / "shared_1_c", state_dim, 256, Default::default()))
            .add_fn(|x| x.relu())
            .add(nn::linear(vs / "shared_2_c", 256, 128, Default::default()))
            .add_fn(|x| x.relu())
            .add(nn::linear(vs / "critic1", 128, 64, Default::default()))
            .add_fn(|x| x.relu())
            .add(nn::linear(vs / "critic_out", 64, 1, Default::default()));

        Self { actor, critic, device }
    }

    pub fn forward(&self, state: &Tensor) -> (Tensor, Tensor) {
        // Fixed: was returning `actor_logits` (undefined) — renamed from `action_logits`
        let action_logits = self.actor.forward(state);
        let critic_value = self.critic.forward(state);
        (action_logits, critic_value)
    }

    pub fn get_action(&self, state: &Tensor, deterministic: bool) -> (Tensor, Tensor) {
        let (action_logits, value) = self.forward(state);

        if deterministic {
            (action_logits, value)
        } else {
            let action = action_logits + Tensor::randn_like(&action_logits) * 0.1;
            (action, value)
        }
    }
}
