use async_nats::jetstream::consumer::pull::Consumer as PullConsumer;
use async_nats::jetstream::stream::{Config, RetentionPolicy, StorageType};
use futures::StreamExt;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tokio::time::Duration;

use crate::eventbrokers::base::{AnyEvent, Event, LocalEventBroker};

pub async fn setup_jetstream(
    js: async_nats::jetstream::Context,
) -> Result<(), async_nats::Error> {
    js.get_or_create_stream(Config {
        name: "JOBS_STREAM".to_string(),
        subjects: vec!["jobs.>".to_string()],
        retention: RetentionPolicy::WorkQueue,
        storage: StorageType::File,
        ..Default::default()
    })
    .await?;

    Ok(())
}

pub struct NatsEventBroker {
    client: async_nats::Client,
    jetstream: async_nats::jetstream::Context,
    subject: String,
    local_broker: LocalEventBroker,
}

impl NatsEventBroker {
    // Fixed: `new` must be async because `connect` is async; was using `?` without await
    pub async fn new(url: &str, subject: String) -> Result<Self, async_nats::Error> {
        let client = async_nats::connect(url).await?;
        let jetstream = async_nats::jetstream::new(client.clone());

        Ok(Self {
            client,
            jetstream,
            subject,
            local_broker: LocalEventBroker::new(),
        })
    }

    pub async fn publish<E: AnyEvent + Serialize>(
        &self,
        event: &E,
    ) -> Result<(), async_nats::Error> {
        // Fixed: format string had missing dot separator and spurious `?` on `to_string()`
        let subject = format!("jobs.{}", self.subject);
        let payload = serde_json::to_vec(event).map_err(|e| {
            async_nats::Error::from(Box::<dyn std::error::Error + Send + Sync>::from(
                e.to_string(),
            ))
        })?;

        self.jetstream.publish(subject, payload.into()).await?;

        // Also dispatch to local in-process subscribers
        self.local_broker
            .publish_local(Arc::new(event.clone()) as Arc<dyn AnyEvent>)
            .await;

        Ok(())
    }

    pub async fn start_listening(&self) -> Result<(), Box<dyn std::error::Error>> {
        // Fixed: stream name was "JOB_STREAM" (inconsistent) — use "JOBS_STREAM"
        let stream = self.jetstream.get_stream("JOBS_STREAM").await?;
        let consumer = stream
            .get_or_create_consumer(
                "schedule_listener",
                async_nats::jetstream::consumer::pull::Config::default(),
            )
            .await?;

        let local = Arc::new(self.local_broker.clone());

        tokio::spawn(async move {
            let mut messages = consumer.messages().await.unwrap();
            while let Some(Ok(msg)) = messages.next().await {
                // Fixed: turbofish syntax `from_slice<Event>` → `from_slice::<Event>`
                if let Ok(event) = serde_json::from_slice::<Event>(&msg.payload) {
                    let shared_event = Arc::new(event) as Arc<dyn AnyEvent>;
                    local.publish_local(shared_event).await;
                } else {
                    tracing::error!("Failed to deserialize event from NATS");
                }
                // Fixed: `double_ack` does not exist — use `ack`
                msg.ack().await.ok();
            }
        });

        Ok(())
    }

    async fn create_gpu_consumer(
        js: &async_nats::jetstream::Context,
    ) -> PullConsumer {
        let stream = js
            .get_stream("JOBS_STREAM")
            .await
            .expect("Stream must exist");

        stream
            .get_or_create_consumer(
                "gpu_worker_group",
                async_nats::jetstream::consumer::pull::Config {
                    durable_name: Some("gpu_worker_group".to_string()),
                    ack_wait: Duration::from_secs(30),
                    ..Default::default()
                },
            )
            .await
            .expect("Failed to create consumer")
    }
}
