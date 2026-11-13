//! Transport-agnostic event delivery.

use std::sync::Arc;

/// Receives a stream of events from core.
///
/// Tauri wraps a `Channel`, the devbridge wraps a replay buffer, and tests collect into a
/// vector. Any `Fn(E) + Send + Sync + 'static` closure is a sink.
pub trait EventSink<E>: Send + Sync + 'static {
    /// Delivers one event. Must not block for long: it is called from the run thread.
    fn emit(&self, event: E);
}

impl<E, F> EventSink<E> for F
where
    F: Fn(E) + Send + Sync + 'static,
{
    fn emit(&self, event: E) {
        self(event);
    }
}

/// A shared, type-erased sink.
pub type SharedSink<E> = Arc<dyn EventSink<E>>;
