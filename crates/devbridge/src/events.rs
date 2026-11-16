//! Per-stream event buffers with replay, backing the server-sent event endpoints.

use std::collections::{HashMap, VecDeque};
use std::convert::Infallible;
use std::sync::{Arc, Mutex, MutexGuard, PoisonError};

use axum::response::sse::Event;
use futures_util::Stream;
use serde::Serialize;
use tokio::sync::watch;

/// How many streams a registry remembers before it forgets the oldest finished ones.
const MAX_STREAMS: usize = 256;

#[derive(Default)]
struct LogState {
    events: VecDeque<String>,
    /// The index of `events[0]` in the whole stream: how many events were forgotten.
    first: usize,
    /// How many events are kept; `None` keeps all of them.
    capacity: Option<usize>,
    closed: bool,
}

/// An append-only buffer of serialized events for one run, job, or editor session.
///
/// Subscribers see the stream from its first event, however late they connect, and
/// their stream finishes after the closing event. A bounded log forgets its oldest
/// events beyond its capacity; a subscriber that is further behind than that skips
/// ahead to the oldest event still kept.
pub struct EventLog {
    state: Mutex<LogState>,
    changed: watch::Sender<()>,
}

impl EventLog {
    /// Creates an empty, open log that keeps every event.
    pub fn new() -> Arc<Self> {
        Arc::new(Self {
            state: Mutex::new(LogState::default()),
            changed: watch::channel(()).0,
        })
    }

    /// Creates an empty, open log that keeps the latest `capacity` events: for
    /// streams with no natural end (an editor session's messages).
    pub fn bounded(capacity: usize) -> Arc<Self> {
        Arc::new(Self {
            state: Mutex::new(LogState {
                capacity: Some(capacity.max(1)),
                ..LogState::default()
            }),
            changed: watch::channel(()).0,
        })
    }

    fn state(&self) -> MutexGuard<'_, LogState> {
        self.state.lock().unwrap_or_else(PoisonError::into_inner)
    }

    /// Appends an event; `last` closes the log. Events pushed after closing are dropped.
    /// Callable from any thread, including non-async ones.
    pub fn push<E: Serialize>(&self, event: &E, last: bool) {
        let Ok(json) = serde_json::to_string(event) else {
            tracing::error!("dropping an event that failed to serialize");
            return;
        };
        {
            let mut state = self.state();
            if state.closed {
                return;
            }
            state.events.push_back(json);
            if state
                .capacity
                .is_some_and(|capacity| state.events.len() > capacity)
            {
                state.events.pop_front();
                state.first += 1;
            }
            state.closed = last;
        }
        self.changed.send_replace(());
    }

    fn is_closed(&self) -> bool {
        self.state().closed
    }

    /// The first event at or after `index` that is still kept, with its own index, or
    /// `Err(closed)` when there is none yet.
    fn get(&self, index: usize) -> Result<(usize, String), bool> {
        let state = self.state();
        let index = index.max(state.first);
        state
            .events
            .get(index - state.first)
            .map(|json| (index, json.clone()))
            .ok_or(state.closed)
    }

    /// A stream of every event from index `from` on (0 is the first event), finishing
    /// after the closing event. Each event carries its index as the SSE `id`, so a
    /// browser that reconnects sends it back as `Last-Event-ID` and can resume after
    /// it instead of seeing the stream again.
    pub fn subscribe(
        self: Arc<Self>,
        from: usize,
    ) -> impl Stream<Item = Result<Event, Infallible>> + Send {
        let updates = self.changed.subscribe();
        futures_util::stream::unfold(
            (self, updates, from),
            |(log, mut updates, next)| async move {
                loop {
                    match log.get(next) {
                        Ok((index, json)) => {
                            let event = Event::default().id(index.to_string()).data(json);
                            return Some((Ok(event), (log, updates, index + 1)));
                        }
                        Err(true) => return None,
                        Err(false) => updates.changed().await.ok()?,
                    }
                }
            },
        )
    }
}

#[derive(Default)]
struct Logs {
    by_id: HashMap<String, Arc<EventLog>>,
    /// Ids in insertion order, for eviction.
    order: VecDeque<String>,
}

/// Event logs by id, for one kind of stream (runs, jobs, or editor sessions).
#[derive(Default)]
pub struct LogRegistry {
    logs: Mutex<Logs>,
}

impl LogRegistry {
    /// Registers `log` under `id`, forgetting the oldest finished logs beyond the cap.
    pub fn insert(&self, id: String, log: Arc<EventLog>) {
        let mut guard = self.logs.lock().unwrap_or_else(PoisonError::into_inner);
        let Logs { by_id, order } = &mut *guard;
        by_id.insert(id.clone(), log);
        order.push_back(id);
        while order.len() > MAX_STREAMS {
            let finished = order
                .iter()
                .position(|id| by_id.get(id).is_none_or(|log| log.is_closed()));
            let Some(id) = finished.and_then(|position| order.remove(position)) else {
                break;
            };
            by_id.remove(&id);
        }
    }

    /// Looks up the log for `id`.
    pub fn get(&self, id: &str) -> Option<Arc<EventLog>> {
        self.logs
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .by_id
            .get(id)
            .cloned()
    }
}

#[cfg(test)]
mod tests {
    use futures_util::StreamExt;

    use super::*;

    /// What a subscriber from index `from` is given, as `index:json`. An axum `Event`
    /// cannot be read back, so the content comes from the log and the stream is only
    /// checked to deliver that many events and then finish.
    async fn collect(log: &Arc<EventLog>, from: usize) -> Vec<String> {
        let mut seen = Vec::new();
        let mut next = from;
        while let Ok((index, json)) = log.get(next) {
            seen.push(format!("{index}:{json}"));
            next = index + 1;
        }
        let delivered = Arc::clone(log).subscribe(from).count().await;
        assert_eq!(delivered, seen.len());
        seen
    }

    #[tokio::test]
    async fn an_unbounded_log_replays_from_any_index() {
        let log = EventLog::new();
        for n in 0..4 {
            log.push(&n, n == 3);
        }
        log.push(&99, false);
        assert_eq!(collect(&log, 0).await, ["0:0", "1:1", "2:2", "3:3"]);
        assert_eq!(collect(&log, 2).await, ["2:2", "3:3"]);
        assert!(collect(&log, 7).await.is_empty());
    }

    #[tokio::test]
    async fn a_bounded_log_forgets_its_oldest_events_and_keeps_their_numbering() {
        let log = EventLog::bounded(3);
        for n in 0..10 {
            log.push(&n, n == 9);
        }
        // A subscriber from the start skips ahead to what is still kept.
        assert_eq!(collect(&log, 0).await, ["7:7", "8:8", "9:9"]);
        assert_eq!(collect(&log, 9).await, ["9:9"]);
    }
}
