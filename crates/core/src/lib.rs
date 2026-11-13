//! Whetstone core: all application logic lives here. See `docs/ARCHITECTURE.md`.
//!
//! The crate has no Tauri dependency. Shells (the Tauri app, the devbridge) construct a
//! [`Core`] for a content root and map their commands onto its methods; streaming
//! commands deliver events through an [`EventSink`].
//!
//! Module map:
//! - [`model`]: wire types (camelCase JSON, mirrors the TypeScript contract)
//! - [`root`]: content root resolution
//! - [`catalog`]: tracks and problems on disk, with a cheap incremental reload
//! - [`solutions`]: the user's files under `workspace/`
//! - [`store`]: SQLite state (submissions, editorial unlocks, settings)
//! - [`progress`]: status derivation and streak math
//! - [`protocol`], [`harness`], [`target`], [`run`]: the run pipeline
//! - [`uv`], [`env`]: the Python environment
//! - [`app`]: the [`Core`] facade

pub mod app;
pub mod catalog;
pub mod env;
pub mod error;
pub mod harness;
pub mod model;
pub mod progress;
pub mod protocol;
pub mod root;
pub mod run;
pub mod sink;
pub mod solutions;
pub mod store;
pub mod target;
pub mod uv;

pub use app::{Core, CoreOptions};
pub use error::{CoreError, Result};
pub use harness::{CommandSpec, HarnessCommand, HarnessJob, ProgramHarness, UvHarness};
pub use model::*;
pub use root::{ResolvedRoot, RootInputs, RootSource, resolve_root};
pub use run::RunLimits;
pub use sink::EventSink;
pub use target::{LOCAL_TARGET, LocalTarget, Process, Target};

/// The current time as ISO 8601 in UTC with millisecond precision.
pub(crate) fn now_iso8601() -> String {
    chrono::Utc::now().to_rfc3339_opts(chrono::SecondsFormat::Millis, true)
}
