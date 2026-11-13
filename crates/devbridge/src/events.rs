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
    events: Vec<String>,
    closed: bool,
}

/// An append-only buffer of serialized events for one run or job.
///
/// Subscribers always see the stream from its first event, however late they connect,
/// and their stream finishes after the closing event.
pub struct EventLog {
    state: Mutex<LogState>,
    changed: watch::Sender<()>,
}

impl EventLog {
    /// Creates an empty, open log.
    pub fn new() -> Arc<Self> {
        Arc::new(Self {
            state: Mutex::new(LogState::default()),
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
            state.events.push(json);
            state.closed = last;
        }
        self.changed.send_replace(());
    }

    fn is_closed(&self) -> bool {
        self.state().closed
    }

    /// The event at `index`, or `Err(closed)` when there is none yet.
    fn get(&self, index: usize) -> Result<String, bool> {
        let state = self.state();
        state.events.get(index).cloned().ok_or(state.closed)
    }

    /// A stream of every event from the first, finishing after the closing event.
    pub fn subscribe(self: Arc<Self>) -> impl Stream<Item = Result<Event, Infallible>> + Send {
        let updates = self.changed.subscribe();
        futures_util::stream::unfold((self, updates, 0), |(log, mut updates, next)| async move {
            loop {
                match log.get(next) {
                    Ok(json) => {
                        return Some((Ok(Event::default().data(json)), (log, updates, next + 1)));
                    }
                    Err(true) => return None,
                    Err(false) => updates.changed().await.ok()?,
                }
            }
        })
    }
}

#[derive(Default)]
struct Logs {
    by_id: HashMap<String, Arc<EventLog>>,
    /// Ids in insertion order, for eviction.
    order: VecDeque<String>,
}

/// Event logs by id, for one kind of stream (runs or install jobs).
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
