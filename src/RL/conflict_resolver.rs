use std::collections::HashMap;

pub struct ConflictResolver {
    pub temperature: f64,
}

impl ConflictResolver {
    pub fn new(temperature: f64) -> Self {
        Self { temperature }
    }

    /// Resolve conflicts where multiple agents selected the same task.
    /// Input: `(agent_id, Option<task_id>, selection_probability)`
    /// Output: `(agent_id, Option<task_id>)` — losers receive `None`
    pub fn resolve(
        &self,
        agent_actions: Vec<(usize, Option<String>, f64)>,
    ) -> Vec<(usize, Option<String>)> {
        let mut resolved: Vec<(usize, Option<String>)> = Vec::new();

        // Separate no-ops (no conflict possible) from task selections
        let mut task_to_agents: HashMap<String, Vec<(usize, f64)>> = HashMap::new();

        for (agent_id, task_id, prob) in agent_actions {
            if let Some(tid) = task_id {
                task_to_agents
                    .entry(tid.clone())
                    .or_insert_with(Vec::new)
                    .push((agent_id, prob));
            } else {
                resolved.push((agent_id, None));
            }
        }

        for (task_id, agents) in task_to_agents {
            if agents.len() == 1 {
                // No conflict — assign directly
                resolved.push((agents[0].0, Some(task_id)));
            } else {
                // Multiple agents want the same task — sample winner by probability
                let total: f64 = agents.iter().map(|(_, p)| p).sum();
                let normalized: Vec<(usize, f64)> =
                    agents.iter().map(|&(aid, p)| (aid, p / total)).collect();

                let winner = self.sample_categorical(&normalized);
                resolved.push((winner, Some(task_id.clone())));

                // Losers get no-action
                for (aid, _) in &agents {
                    if *aid != winner {
                        resolved.push((*aid, None));
                    }
                }
            }
        }

        resolved
    }

    /// Weighted random sampling: returns the `agent_id` of the sampled winner.
    fn sample_categorical(&self, agents: &[(usize, f64)]) -> usize {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        use std::time::{SystemTime, UNIX_EPOCH};

        // Simple deterministic-ish random via system time — replace with `rand` crate in Milestone 4
        let seed = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .subsec_nanos();

        let mut hasher = DefaultHasher::new();
        seed.hash(&mut hasher);
        let rand_val = (hasher.finish() as f64) / (u64::MAX as f64);

        let mut cumulative = 0.0;
        for &(agent_id, prob) in agents {
            cumulative += prob;
            if rand_val <= cumulative {
                return agent_id;
            }
        }

        // Fallback: return last agent
        agents.last().map(|&(id, _)| id).unwrap_or(0)
    }
}
