use anyhow::Result;
use fred::prelude::*;
use serde::{Deserialize, Serialize};
use sqlx::PgPool;
use std::sync::Arc;

#[derive(Debug, Clone, Serialize, Deserialize, sqlx::FromRow)]
pub struct JobRecord {
    pub id: i64,
    pub job_id: String,
    pub task_type: String,
    pub cluster_id: i32,
    pub arrival_time: chrono::DateTime<chrono::Utc>,
    pub start_time: Option<chrono::DateTime<chrono::Utc>>,
    pub completion_time: Option<chrono::DateTime<chrono::Utc>>,
    pub status: String,
    pub cpu_allocated: i32,
    pub gpu_id: Option<i32>,
    pub gpu_cores_allocated: i32,
    pub memory_mb_allocated: i64,
    pub priority: f32,
    pub queue_wait_time_sec: Option<f32>,
    pub execution_time_sec: Option<f32>,
}

#[derive(Debug, Clone, Serialize, Deserialize, sqlx::FromRow)]
pub struct ClusterMetrics {
    pub id: i64,
    pub cluster_id: i32,
    pub timestamp: chrono::DateTime<chrono::Utc>,
    pub cpu_utilization: f32,
    pub gpu_utilization: f32,
    pub memory_utilization: f32,
    pub queue_length: i32,
    pub active_jobs: i32,
    pub throughput: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MCTSNodeData {
    pub node_id: String,
    pub parent_id: Option<String>,
    pub visit_count: usize,
    pub total_value: f64,
    pub prior_prob: f64,
    pub reward: f64,
    pub action: Option<ActionKeyData>,
    pub children_ids: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActionKeyData {
    pub task_idx: i64,
    pub cpu_bin: i64,
    pub gpu_idx: i64,
    pub mem_bin: i64,
}

#[async_trait::async_trait]
pub trait StorageLayer: Send + Sync {
    async fn store_job(&self, job: &JobRecord) -> Result<i64>;
    async fn update_job_status(&self, job_id: &str, status: &str) -> Result<()>;
    async fn get_job(&self, job_id: &str) -> Result<Option<JobRecord>>;
    async fn store_metrics(&self, metrics: &ClusterMetrics) -> Result<()>;
    async fn get_recent_metrics(&self, cluster_id: i32, limit: i64) -> Result<Vec<ClusterMetrics>>;
}

pub struct PostgresStorage {
    pool: PgPool,
}

impl PostgresStorage {
    pub async fn new(database_url: &str) -> Result<Self> {
        let pool = PgPool::connect(database_url).await?;
        Ok(Self { pool })
    }

    pub async fn initialize(&self) -> Result<()> {
        // Fixed: added missing commas between column definitions,
        // removed UNIQUE from task_type, removed UNIQUE from job_id
        sqlx::query(
            r#"
            CREATE TABLE IF NOT EXISTS jobs (
                id BIGSERIAL PRIMARY KEY,
                job_id VARCHAR(255) NOT NULL,
                task_type VARCHAR(225) NOT NULL,
                cluster_id INTEGER NOT NULL,
                arrival_time TIMESTAMPTZ NOT NULL,
                start_time TIMESTAMPTZ,
                completion_time TIMESTAMPTZ,
                status VARCHAR(50) NOT NULL,
                cpu_allocated INTEGER,
                gpu_id INTEGER,
                gpu_cores_allocated INTEGER,
                memory_mb_allocated BIGINT,
                priority REAL,
                queue_wait_time_sec REAL,
                execution_time_sec REAL,
                created_at TIMESTAMPTZ DEFAULT NOW()
            )
            "#,
        )
        .execute(&self.pool)
        .await?;

        sqlx::query(
            r#"
            CREATE TABLE IF NOT EXISTS cluster_metrics (
                id BIGSERIAL PRIMARY KEY,
                cluster_id INTEGER NOT NULL,
                timestamp TIMESTAMPTZ NOT NULL,
                cpu_utilization REAL NOT NULL,
                gpu_utilization REAL NOT NULL,
                memory_utilization REAL NOT NULL,
                queue_length INTEGER NOT NULL,
                active_jobs INTEGER NOT NULL,
                throughput REAL NOT NULL,
                created_at TIMESTAMPTZ DEFAULT NOW()
            )
            "#,
        )
        .execute(&self.pool)
        .await?;

        sqlx::query("CREATE INDEX IF NOT EXISTS idx_jobs_job_id ON jobs(job_id)")
            .execute(&self.pool)
            .await?;

        sqlx::query("CREATE INDEX IF NOT EXISTS idx_jobs_cluster_id ON jobs(cluster_id)")
            .execute(&self.pool)
            .await?;

        sqlx::query(
            "CREATE INDEX IF NOT EXISTS idx_metrics_cluster_timestamp ON cluster_metrics(cluster_id, timestamp DESC)",
        )
        .execute(&self.pool)
        .await?;

        Ok(())
    }
}

#[async_trait::async_trait]
impl StorageLayer for PostgresStorage {
    async fn store_job(&self, job: &JobRecord) -> Result<i64> {
        let rec = sqlx::query!(
            r#"
            INSERT INTO jobs (
                job_id, task_type, cluster_id, arrival_time, start_time,
                completion_time, status, cpu_allocated, gpu_id,
                gpu_cores_allocated, memory_mb_allocated, priority,
                queue_wait_time_sec, execution_time_sec
            )
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14)
            RETURNING id
            "#,
            job.job_id,
            job.task_type,
            job.cluster_id,
            job.arrival_time,
            job.start_time,
            job.completion_time,
            job.status,
            job.cpu_allocated,
            job.gpu_id,
            job.gpu_cores_allocated,
            job.memory_mb_allocated,
            job.priority,
            job.queue_wait_time_sec,
            job.execution_time_sec,
        )
        .fetch_one(&self.pool)
        .await?;

        Ok(rec.id)
    }

    async fn update_job_status(&self, job_id: &str, status: &str) -> Result<()> {
        // Fixed: was referencing `job.id` (undefined) — use the `job_id` parameter directly
        sqlx::query!(
            r#"UPDATE jobs SET status = $1 WHERE job_id = $2"#,
            status,
            job_id,
        )
        .execute(&self.pool)
        .await?;
        Ok(())
    }

    async fn get_job(&self, job_id: &str) -> Result<Option<JobRecord>> {
        // Fixed: was referencing `job.id` — use the `job_id` parameter directly
        let job = sqlx::query_as!(
            JobRecord,
            r#"SELECT * FROM jobs WHERE job_id = $1"#,
            job_id
        )
        .fetch_optional(&self.pool)
        .await?;
        Ok(job)
    }

    async fn store_metrics(&self, metrics: &ClusterMetrics) -> Result<()> {
        sqlx::query!(
            r#"
            INSERT INTO cluster_metrics (
                cluster_id, timestamp, cpu_utilization, gpu_utilization,
                memory_utilization, queue_length, active_jobs, throughput
            )
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
            "#,
            metrics.cluster_id,
            metrics.timestamp,
            metrics.cpu_utilization,
            metrics.gpu_utilization,
            metrics.memory_utilization,
            metrics.queue_length,
            metrics.active_jobs,
            metrics.throughput,
        )
        .execute(&self.pool)
        .await?;
        Ok(())
    }

    async fn get_recent_metrics(&self, cluster_id: i32, limit: i64) -> Result<Vec<ClusterMetrics>> {
        // Fixed: typo `ClusterMetricsd` → `ClusterMetrics`
        let metrics = sqlx::query_as!(
            ClusterMetrics,
            r#"
            SELECT * FROM cluster_metrics
            WHERE cluster_id = $1
            ORDER BY timestamp DESC
            LIMIT $2
            "#,
            cluster_id,
            limit
        )
        .fetch_all(&self.pool)
        .await?;
        Ok(metrics)
    }
}

impl PostgresStorage {
    pub async fn get_job_statistics(&self, hours: i64) -> Result<JobStatistics> {
        let cutoff = chrono::Utc::now() - chrono::Duration::hours(hours);

        let stats = sqlx::query!(
            r#"
            SELECT
                COUNT(*) as total_jobs,
                AVG(execution_time_sec) as avg_execution_time,
                AVG(queue_wait_time_sec) as avg_wait_time,
                COUNT(*) FILTER (WHERE status = 'completed') as completed_jobs,
                COUNT(*) FILTER (WHERE status = 'failed') as failed_jobs
            FROM jobs
            WHERE arrival_time > $1
            "#,
            cutoff
        )
        .fetch_one(&self.pool)
        .await?;

        Ok(JobStatistics {
            total_jobs: stats.total_jobs.unwrap_or(0),
            avg_execution_time: stats.avg_execution_time.unwrap_or(0.0) as f32,
            avg_wait_time: stats.avg_wait_time.unwrap_or(0.0) as f32,
            completed_jobs: stats.completed_jobs.unwrap_or(0),
            failed_jobs: stats.failed_jobs.unwrap_or(0),
        })
    }

    pub async fn get_cluster_utilization_history(
        &self,
        cluster_id: i32,
        hours: i64,
    ) -> Result<Vec<ClusterMetrics>> {
        let cutoff = chrono::Utc::now() - chrono::Duration::hours(hours);

        let metrics = sqlx::query_as!(
            ClusterMetrics,
            r#"
            SELECT * FROM cluster_metrics
            WHERE cluster_id = $1 AND timestamp > $2
            ORDER BY timestamp ASC
            "#,
            cluster_id,
            cutoff
        )
        .fetch_all(&self.pool)
        .await?;

        Ok(metrics)
    }
}

// ============================================================================
// CACHE STORAGE (Valkey / Redis-compatible)
// ============================================================================

pub struct CacheStorage {
    client: RedisClient,
    ttl_seconds: i64,
    cache_type: String,
}

#[derive(Debug)]
pub struct JobStatistics {
    pub total_jobs: i64,
    pub avg_execution_time: f32,
    pub avg_wait_time: f32,
    pub completed_jobs: i64,
    pub failed_jobs: i64,
}

impl CacheStorage {
    /// Accepts `valkey://` or `redis://` URLs.
    pub async fn new(cache_url: &str, ttl_seconds: i64) -> Result<Self> {
        // Fixed: the original if-let chain was missing else branches and wouldn't compile
        let cache_type = if cache_url.starts_with("valkey://") {
            "valkey"
        } else if cache_url.starts_with("dragonfly://") {
            "dragonfly"
        } else {
            "redis"
        };

        // Fred uses the Redis protocol which Valkey/DragonflyDB implement
        let config = RedisConfig::from_url(cache_url)?;
        let client = RedisClient::new(config, None, None, None);
        client.connect();
        client.wait_for_connect().await?;

        println!("Connected to {} cache", cache_type);

        Ok(Self {
            client,
            ttl_seconds,
            cache_type: cache_type.to_string(),
        })
    }

    pub async fn store_mcts_node(&self, node: &MCTSNodeData) -> Result<()> {
        // Fixed: key used `node.id` (non-existent field) — use `node.node_id`
        let key = format!("mcts:node:{}", &node.node_id);
        let value = serde_json::to_string(node)?;

        self.client
            .set(
                &key,
                value,
                Some(Expiration::EX(self.ttl_seconds)),
                None,
                false,
            )
            .await?;

        if let Some(parent_id) = &node.parent_id {
            let parent_children_key = format!("mcts:children:{}", parent_id);
            self.client
                .sadd(&parent_children_key, &node.node_id)
                .await?;
            self.client
                .expire(&parent_children_key, self.ttl_seconds)
                .await?;
        }

        Ok(())
    }

    pub async fn get_mcts_node(&self, node_id: &str) -> Result<Option<MCTSNodeData>> {
        let key = format!("mcts:node:{}", node_id);
        let value: Option<String> = self.client.get(&key).await?;

        match value {
            Some(json) => Ok(Some(serde_json::from_str(&json)?)),
            None => Ok(None),
        }
    }

    /// Returns the child node IDs for a given parent, not the full node.
    /// Fixed: original return type was `Option<MCTSNodeData>` but the data is a list of IDs.
    pub async fn get_children_ids(&self, parent_id: &str) -> Result<Vec<String>> {
        // Fixed: key format was inconsistent — standardised to `mcts:children:{id}`
        let key = format!("mcts:children:{}", parent_id);
        let ids: Vec<String> = self.client.smembers(&key).await?;
        Ok(ids)
    }

    pub async fn update_node_stats(
        &self,
        node_id: &str,
        visit_count: usize,
        total_value: f64,
    ) -> Result<()> {
        if let Some(mut node) = self.get_mcts_node(node_id).await? {
            node.visit_count = visit_count;
            node.total_value = total_value;
            self.store_mcts_node(&node).await?;
        }
        Ok(())
    }

    pub async fn cache_scheduling_state(
        &self,
        cluster_id: usize,
        state: &SchedulingStateCache,
    ) -> Result<()> {
        let key = format!("state:cluster:{}", cluster_id);
        let value = serde_json::to_string(state)?;
        self.client
            .set(&key, value, Some(Expiration::EX(300)), None, false)
            .await?;
        Ok(())
    }

    pub async fn get_scheduling_state(
        &self,
        cluster_id: usize,
    ) -> Result<Option<SchedulingStateCache>> {
        let key = format!("state:cluster:{}", cluster_id);
        let value: Option<String> = self.client.get(&key).await?;

        match value {
            Some(json) => Ok(Some(serde_json::from_str(&json)?)),
            None => Ok(None),
        }
    }

    pub async fn add_active_job(&self, cluster_id: usize, job_id: &str) -> Result<()> {
        let key = format!("active:cluster:{}", cluster_id);
        self.client.sadd(&key, job_id).await?;
        Ok(())
    }

    pub async fn remove_active_job(&self, cluster_id: usize, job_id: &str) -> Result<()> {
        let key = format!("active:cluster:{}", cluster_id);
        self.client.srem(&key, job_id).await?;
        Ok(())
    }

    pub async fn get_active_jobs(&self, cluster_id: usize) -> Result<Vec<String>> {
        let key = format!("active:cluster:{}", cluster_id);
        let jobs: Vec<String> = self.client.smembers(&key).await?;
        Ok(jobs)
    }

    pub fn cache_type(&self) -> &str {
        &self.cache_type
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SchedulingStateCache {
    pub cluster_id: usize,
    pub cpu_available: usize,
    pub gpu_states: Vec<GPUStateCache>,
    pub queue_length: usize,
    pub timestamp: chrono::DateTime<chrono::Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GPUStateCache {
    pub gpu_id: usize,
    pub core_available: usize,
    pub memory_available_mb: usize,
}

// ============================================================================
// DATABASE MANAGER — unified facade
// ============================================================================

pub struct DatabaseManager {
    pub postgres: Arc<PostgresStorage>,
    pub cache: Arc<CacheStorage>,
}

impl DatabaseManager {
    pub async fn new(postgres_url: &str, cache_url: &str) -> Result<Self> {
        let postgres = Arc::new(PostgresStorage::new(postgres_url).await?);
        postgres.initialize().await?;

        let cache = Arc::new(CacheStorage::new(cache_url, 3600).await?);

        println!(
            "Database initialized with {} cache backend",
            cache.cache_type()
        );

        Ok(Self { postgres, cache })
    }

    pub async fn record_job_start(&self, job: &JobRecord) -> Result<()> {
        self.postgres.store_job(job).await?;
        self.cache
            .add_active_job(job.cluster_id as usize, &job.job_id)
            .await?;
        Ok(())
    }

    pub async fn record_job_completion(
        &self,
        job_id: &str,
        cluster_id: usize,
        completion_time: chrono::DateTime<chrono::Utc>,
        execution_time_sec: f32,
    ) -> Result<()> {
        sqlx::query!(
            r#"
            UPDATE jobs
            SET status = 'completed',
                completion_time = $1,
                execution_time_sec = $2
            WHERE job_id = $3
            "#,
            completion_time,
            execution_time_sec,
            job_id
        )
        .execute(&self.postgres.pool)
        .await?;

        self.cache.remove_active_job(cluster_id, job_id).await?;
        Ok(())
    }

    pub async fn store_metrics_with_cache(&self, metrics: &ClusterMetrics) -> Result<()> {
        self.postgres.store_metrics(metrics).await?;

        let state = SchedulingStateCache {
            cluster_id: metrics.cluster_id as usize,
            cpu_available: (100.0 - metrics.cpu_utilization) as usize,
            gpu_states: vec![],
            queue_length: metrics.queue_length as usize,
            timestamp: metrics.timestamp,
        };

        self.cache
            .cache_scheduling_state(metrics.cluster_id as usize, &state)
            .await?;

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_with_valkey() -> Result<()> {
        let db = DatabaseManager::new(
            "postgres://user:pass@localhost/scheduler",
            "valkey://localhost:6379",
        )
        .await?;

        assert_eq!(db.cache.cache_type(), "valkey");
        Ok(())
    }
}
