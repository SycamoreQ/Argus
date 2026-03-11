#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt::init();

    info!("Starting Training with Advanced MCTS");

    // Setup
    let db = Arc::new(DatabaseManager::new(
        "postgres://scheduler:pass@localhost:5432/gpu_scheduler",
        "valkey://localhost:6379",
    ).await?);

    let env_config = EnvironmentConfig::default();
    let mut env = EdgeMLEnv::new(env_config.clone(), Arc::clone(&db));

    let device = Device::cuda_if_available();
    let vs = nn::VarStore::new(device);

    let world_model = Arc::new(WorldModel::new(&(vs.root() / "world"), 256, 4, 256));
    let actor = Arc::new(TapFingerActor::new(&vs.root(), 256, 17));

    // Create Advanced MCTS with all optimizations
    let mcts = Arc::new(AdvancedMCTS::new(
        Arc::clone(&world_model),
        Arc::clone(&actor),
        200,  // num_simulations
        Some(Arc::clone(&db)),
    ));

    info!("Advanced MCTS initialized with:");
    info!("   - Transposition Table (10k states)");
    info!("   - Tree Reuse between searches");
    info!("   - Progressive Widening");
    info!("   - Virtual Loss for parallel sims");
    info!("   - Redis persistence");

    // Training loop
    for episode in 0..1000 {
        env.reset();
        let mut done = false;

        while !done {
            // Build contexts for all clusters
            let contexts: Vec<_> = (0..env_config.num_clusters)
                .map(|i| (env.build_scheduling_context(i), i))
                .collect();

            // BATCH MCTS: Process all clusters in parallel
            let mcts_results = mcts.batch_search(contexts).await?;

            // Convert to agent actions
            let agent_actions: Vec<_> = mcts_results.iter()
                .enumerate()
                .map(|(cluster_id, (action_key, _value))| {
                    let ctx = env.build_scheduling_context(cluster_id);
                    action_key.to_agent_action(&ctx)
                })
                .collect();

            let result = env.step(agent_actions).await?;
            done = result.done;
        }

        // Print stats every 10 episodes
        if episode % 10 == 0 {
            let stats = mcts.get_stats();
            info!("Episode {}: Cache Hit Rate: {:.2}%, Transposition Table: {} entries",
                episode, stats.hit_rate * 100.0, stats.transposition_table_size);
        }

        // Save checkpoint
        if episode % 100 == 0 {
            vs.save(format!("checkpoints/episode_{}.pt", episode))?;
        }
    }

    info!("Training completed with advanced MCTS");

    Ok(())
}