use anyhow::Result;
use database::low::DatabaseManager;
use lru::LruCache;
use std::collections::{HashMap, VecDeque};
use std::num::NonZeroUsize;
use std::sync::{Arc, RwLock};
use tch::{nn, Device, Kind, Tensor};
use tokio::sync::RwLock as AsyncRwLock;
use tracing::{debug, info};

pub struct AdvancedMCTS {
    world_model: Arc<WorldModel>,
    actor: Arc<TapFingerActor>,
    c_puct: f64,
    num_simulations: usize,
    max_depth: usize,
    gamma: f64,
    transposition_table: Arc<RwLock<LruCache<StateHash, Arc<Mutex<MCTSNode>>>>>,
    tree_cache: Arc<RwLock<HashMap<String, Arc<Mutex<MCTSNode>>>>>,
    db: Option<Arc<DatabaseManager>>,
    enable_persistence: bool,
    virtual_loss: f64,
    progressive_widening_alpha: f64,
    progressive_widening_c: f64,
    min_action_prob: f64,
    cache_hits: Arc<RwLock<usize>>,
    cache_misses: Arc<RwLock<usize>>,
}

pub struct

impl AdvancedMCTS {
    pub fn new(
        world_model: Arc<WorldModel>,
        actor: Arc<TapFingerActor>,
        num_simulations: usize,
        db: Option<Arc<DatabaseManager>>,
    ) -> Self {
        Self {
            world_model,
            actor,
            c_puct: 1.5,
            num_simulations,
            max_depth: 15,
            gamma: 0.99,

            // Transposition table with LRU eviction
            transposition_table: Arc::new(RwLock::new(LruCache::new(
                NonZeroUsize::new(10000).unwrap(),
            ))),

            // Tree reuse cache
            tree_cache: Arc::new(RwLock::new(HashMap::new())),

            db,
            enable_persistence: true,

            virtual_loss: 10.0,
            progressive_widening_alpha: 0.5,
            progressive_widening_c: 2.0,

            min_action_prob: 0.01,

            cache_hits: Arc::new(RwLock::new(0)),
            cache_misses: Arc::new(RwLock::new(0)),
        }
    }

    pub async fn batch_search(
        &self,
        contexts: Vec<SchedulingContext>,
    ) -> Result<Vec<(ActionKey, f64)>> {
        let mut handles = Vec::new();

        for ctx in contexts {
            let mcts = self.clone_for_thread();
            let handle = tokio::spawn(async move {
                mcts.search_with_reuse(&ctx).await
            });
            handles.push(handle);
        }

        let mut results = Vec::new();
        for handle in handles {
            results.push(handle.await??);
        }

        Ok(results)
    }

    pub async fn search_reuse(&self , ctx: &SchedulingContext) -> Result<(ActionKey , f64)>{
        let cluster_idx = ctx.cluster_resources.cluster_id;
        let cache_key = format!("mcts_cluster_{}", cluster_id);

        let root = if Some(cached_root) = self.load_from_cache(&cache_key).await?{
            info!("reusing MCTS tree from cluster {}" , cluster_id);
            *self.cache_hits.write().unwrap() += 1 ;
            cached_root
        }
        else{
            info!("Building new MCTS tree for cluster {}", cluster_id);
            *self.cache_misses.write().unwrap() += 1;
            let root_latent = self.world_model.represent(&ctx.graph.node_features);
            Arc::new(Mutex::new(MCTSNode::new(root_latent)))
        }

        for i in 0..self.num_simulations {
            self.simulate(Arc::clone(&root) ,ctx).await?;

            if self.enable_persistance == True{
                self.persist_tree(&cache_key , &root).await?;
            }
        }

        self.save_to_cache(&cache_key, Arc::clone(&root)).await?;

        if self.enable_persistence {
            self.persist_tree(&cache_key, &root).await?;
        }

        let root_lock = root.lock().unwrap(){
            let (best_action , best_child) = root_lock
                .children
                .iter()
                .max_by_key(|(_ , child) child.lock().unwrap().visit_count)
                .map(|(action , child)| (action.clone() , Arc::clone(child)))
                .ok_or_else(||anyhow::anyhow!("no children found"))?;
        }

        let value = best_child.lock().unwrap().q_value();

        info!("Selected action for cluster {}: {:?} (value: {:.3})",
            cluster_id, best_action, value);

        Ok((best_action, value))
    }

    pub async fn simulate(&self , root: Arc<Mutex<MCTSNode>>, ctx: &SchedulingContext) -> Result<()>{
        let mut path = Vec::new();
        let mut current = Arc::clone(&root);
        let depth = 0;

        loop{
            current.lock().unwrap().apply_virtual_loss(self.virtual_loss);

            if let Some(transposed) = self.check_transposition_table(&state_hash).await {
                debug!("Transposition table hit at depth {}", depth);
                current = transposed;
            }

            let node_lock = current.lock().unwrap()

            if node_lock.is_leaf() || depth >= self.max_depth{
                drop(node_lock);
                break;
            }

            if let Some((action , child)) = node_lock.select_child(self.c_puct){
                path.push(Arc::clone(&current) , action.clone());
                drop(node_lock);
                current = child;
                depth += 1;
            }
            else{
                drop(node_lock);
                break;
            }
        }

        let should_expand = {
            let node_lock = current.lock().unwrap();
            node_lock.visit_count >= 1 && node_lock.is_leaf() && depth < self.max_depth
        };

        if should_expand(){
            self.expand(&current , ctx)?await;
        }

        let value = self.evaluate(&current);

        for (node_arc, _) in path.iter().rev() {
            let mut node = node_arc.lock().unwrap();

            // Remove virtual loss
            node.revert_virtual_loss(self.virtual_loss);

            // Regular backprop
            let combined_value = node.reward + (self.gamma * value);
            node.backpropagate(combined_value);
        }
        current.lock().unwrap().revert_virtual_loss(self.virtual_loss);
        current.lock().unwrap().backpropagate(value);

        Ok(())
    }

    async fn expand(
        &self,
        node: &Arc<Mutex<MCTSNode>>,
        ctx: &SchedulingContext,
    ) -> Result<()> {
        let latent_state = node.lock().unwrap().latent_state.shallow_clone();
        let device = latent_state.device();

        let mask = ActionMask::from_context(ctx , device);
        let valid_candidates = mask.get_valid_candidates(
            &ctx.cluster_resources,
            &ctx.task_lookup,
        );

        let action_probs = self.get_action_probabilities(&valid_candidates , &ctx.graph)?;
        let filtered: Vec<_> = action_probs
            .into_iter()
            .filter(|(_ , probs)| *probs >= self.min_action_prob as f64)
            .collect();

        let visit_count = node.lock().unwrap().visit_count;
        let max_children = (self.progressive_widening_c) * (visit_count as f64).powf(self.progressive_widening_alpha)) as usize;

        let num_to_expand = max_children.min(filtered.len());

        debug!("Progressive widening: expanding {}/{} actions at visit_count={}",
            num_to_expand, filtered.len(), visit_count);

        let top_actions = self.select_top_k_actions(&filtered , num_to_expand);

        for (action , prob) in top_actions{
            let output = self.world_model.step(&latent_state , &action);
            let pred_reward = f64::try_from(&output.reward).unwrap_or(0.0);

            let child = node.lock().unwrap().add_child(
                action.clone(),
                output.next_latent_state.shallow_clone(),
                prob,
                pred_reward,
            );

            let state_hash = self.compute_state_hash(&child);
            self.add_to_transposition_table(state_hash, Arc::clone(&child)).await;
        }

        Ok(())
    }

    fn state_hash(&self , &Arc<Mutex<MCTSNode>>) -> StateHash{
        let node_lock = node.lock().unwrap();
        let latent = &node_lock.latent_state;

        // Simple hash: sum of latent state values
        let data: Vec<f32> = latent.view(-1).try_into().unwrap_or_default();
        let sum: f32 = data.iter().sum();
        let hash = (sum * 1000000.0) as u64;

        StateHash(hash)
    }

    async fn check_transposition_table(
        &self,
        hash: &StateHash,
    ) -> Option<Arc<Mutex<MCTSNode>>> {
        let table = self.transposition_table.read().unwrap();
        table.peek(hash).cloned()
    }

    async fn add_to_transposition_table(
        &self,
        hash: StateHash,
        node: Arc<Mutex<MCTSNode>>,
    ) {
        let mut table = self.transposition_table.write().unwrap();
        table.put(hash, node);
    }

    async fn save_to_cache(
        &self,
        key: &str,
        root: Arc<Mutex<MCTSNode>>,
    ) -> Result<()> {
        let mut cache = self.tree_cache.write().unwrap();
        cache.insert(key.to_string(), root);
        Ok(())
    }

    async fn load_from_cache(
        &self,
        key: &str,
    ) -> Result<Option<Arc<Mutex<MCTSNode>>>> {
        let cache = self.tree_cache.read().unwrap();
        Ok(cache.get(key).cloned())
    }

    async fn persist_tree(
        &self,
        key: &str,
        root: &Arc<Mutex<MCTSNode>>,
    ) -> Result<()> {
        if let Some(db) = &self.db {
            let node_data = self.serialize_node_recursive(root, None)?;
            db.cache.store_mcts_node(&node_data).await?;
        }
        Ok(())
    }


    fn serialize_node_recursive(
        &self,
        node: &Arc<Mutex<MCTSNode>>,
        parent_id: Option<String>,
    ) -> Result<MCTSNodeData> {
        let node_lock = node.lock().unwrap();
        let node_id = uuid::Uuid::new_v4().to_string();

        let action_data = node_lock.action.as_ref().map(|a| ActionKeyData {
            task_idx: a.task_idx,
            cpu_bin: a.cpu_bin,
            gpu_idx: a.gpu_idx,
            mem_bin: a.mem_bin,
        });

        let mut children_ids = Vec::new();
        for (_, child_arc) in &node_lock.children {
            let child_data = self.serialize_node_recursive(child_arc, Some(node_id.clone()))?;
            children_ids.push(child_data.node_id.clone());
        }

        Ok(MCTSNodeData {
            node_id,
            parent_id,
            visit_count: node_lock.visit_count,
            total_value: node_lock.total_value,
            prior_prob: node_lock.prior_prob,
            reward: node_lock.reward,
            action: action_data,
            children_ids,
        })
    }

    /// Evaluation with neural network
    fn evaluate(&self, node: &Arc<Mutex<MCTSNode>>) -> f64 {
        let latent_state = node.lock().unwrap().latent_state.shallow_clone();
        let value = self.world_model.predict_value(&latent_state);
        f64::try_from(&value).unwrap_or(0.0)
    }

    /// Get action probabilities from actor network
    fn get_action_probs_for_candidates(
        &self,
        candidates: &[ActionKey],
        graph: &GraphTensors,
    ) -> Result<Vec<(ActionKey, f64)>> {
        if candidates.is_empty(){
            Ok(Vec::new())
        }

        let device = graph.node_features.device();

        let cluster_emb = if !graph.pending_indices.is_empty{
            let cluster_idx = Tensor::of_slice(graph.pending_indices).to_device(&device);
            graph.node_features.index_select(0 , &idx);
        }

        else{
            Tensor::zeros(&[0 , 256] , (Kind::Float , device));
        }

        let num_pending = graph.pending_indices.len() as i64;
        let mask = ActionMask::new(num_pending , 17 , 8 , 20 , device);

        let (task_probs, _resource_logits) = self.actor.forward_detailed(
            &cluster_emb,
            &pending_emb,
            &mask,
        );

        let task_probs_vec: Vec<f32> = task_probs.try_into().unwrap_or_default();

        let mut action_probs = Vec::new();
        for candidate in candidates {
            let task_prob = if candidate.task_idx == 0 {
                // No-op action (first index)
                task_probs_vec.get(0).copied().unwrap_or(0.01)
            } else if candidate.task_idx > 0 && (candidate.task_idx as usize) < task_probs_vec.len() {
                task_probs_vec[candidate.task_idx as usize]
            } else {
                0.01 // Default low probability for out-of-range
            };

            action_probs.push((candidate.clone(), task_prob as f64));
        }

        Ok(action_probs)
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

    /// Clone for parallel execution
    fn clone_for_thread(&self) -> Self {
        Self {
            world_model: Arc::clone(&self.world_model),
            actor: Arc::clone(&self.actor),
            c_puct: self.c_puct,
            num_simulations: self.num_simulations,
            max_depth: self.max_depth,
            gamma: self.gamma,
            transposition_table: Arc::clone(&self.transposition_table),
            tree_cache: Arc::clone(&self.tree_cache),
            db: self.db.clone(),
            enable_persistence: self.enable_persistence,
            virtual_loss: self.virtual_loss,
            progressive_widening_alpha: self.progressive_widening_alpha,
            progressive_widening_c: self.progressive_widening_c,
            min_action_prob: self.min_action_prob,
            cache_hits: Arc::clone(&self.cache_hits),
            cache_misses: Arc::clone(&self.cache_misses),
        }
    }

    /// Get statistics
    pub fn get_stats(&self) -> MCTSStats {
        let hits = *self.cache_hits.read().unwrap();
        let misses = *self.cache_misses.read().unwrap();
        let hit_rate = if hits + misses > 0 {
            hits as f64 / (hits + misses) as f64
        } else {
            0.0
        };

        MCTSStats {
            cache_hits: hits,
            cache_misses: misses,
            hit_rate,
            transposition_table_size: self.transposition_table.read().unwrap().len(),
            tree_cache_size: self.tree_cache.read().unwrap().len(),
        }
    }
}

pub struct MCTSNode {
    pub latent_state: Tensor,
    pub parent: Option<Arc<Mutex<MCTSNode>>>,
    pub children: HashMap<ActionKey, Arc<Mutex<MCTSNode>>>,
    pub visit_count: usize,
    pub total_value: f64,
    pub prior_prob: f64,
    pub action: Option<ActionKey>,
    pub reward: f64,
    virtual_loss_applied: f64,
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
            virtual_loss_applied: 0.0,
        }
    }

    pub fn q_value(&self) -> f64 {
        if self.visit_count == 0 {
            0.0
        } else {
            (self.total_value - self.virtual_loss_applied) / self.visit_count as f64
        }
    }

    pub fn ucb_score(&self, parent_visits: usize, c_puct: f64) -> f64 {
        let exploitation = self.q_value();
        let exploration = c_puct * self.prior_prob * (parent_visits as f64).sqrt()
            / (1.0 + self.visit_count as f64);
        exploitation + exploration
    }

    pub fn apply_virtual_loss(&mut self, loss: f64) {
        self.virtual_loss_applied += loss;
        self.visit_count += 1;
    }

    pub fn revert_virtual_loss(&mut self, loss: f64) {
        self.virtual_loss_applied -= loss;
        self.visit_count -= 1;
    }

    pub fn backpropagate(&mut self, value: f64) {
        self.visit_count += 1;
        self.total_value += value;
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
            virtual_loss_applied: 0.0,
        };

        let child_ref = Arc::new(Mutex::new(child));
        self.children.insert(action, Arc::clone(&child_ref));
        child_ref
    }

    pub fn is_leaf(&self) -> bool {
        self.children.is_empty()
    }
}

#[derive(Debug, Clone, Copy, Hash, Eq, PartialEq)]
pub struct StateHash(u64);

#[derive(Debug, Clone)]
pub struct MCTSStats {
    pub cache_hits: usize,
    pub cache_misses: usize,
    pub hit_rate: f64,
    pub transposition_table_size: usize,
    pub tree_cache_size: usize,
}
