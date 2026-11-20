//! Learn AI core: all application logic lives here. See `docs/ARCHITECTURE.md`.
//!
//! The crate has no Tauri dependency. Shells (the Tauri app, the devbridge) construct a
//! [`Core`] for a content root and map their commands onto its methods; streaming
//! commands deliver events through an [`EventSink`].
//!
//! Module map:
//! - [`model`]: wire types (camelCase JSON, mirrors the TypeScript contract)
//! - [`root`]: content root resolution
//! - [`catalog`]: tracks and problems on disk, with a cheap incremental reload
//! - [`areas`]: subject areas under `areas/`
//! - [`assets`]: statement figures served as `data:` URLs
//! - [`solutions`]: the user's files under `workspace/`
//! - [`store`]: SQLite state (submissions, editorial unlocks, settings)
//! - [`progress`]: status derivation and streak math
//! - [`review`]: spaced re-solve (the schedule math and the injectable clock)
//! - [`protocol`], [`harness`], [`target`], [`run`]: the run pipeline
//! - [`stream`]: reading a subprocess's output as lines (runs, installs, fetches)
//! - [`uv`], [`env`]: the Python environment
//! - [`models`]: model assets (list, fetch, cancel, remove)
//! - [`lsp`], `docs`: editor intelligence (the language server and the runtime docs
//!   helper)
//! - [`ssh`], [`remote`], [`targets`]: SSH run targets (configuration, remote processes,
//!   the target registry and check)
//! - [`app`]: the [`Core`] facade

pub mod app;
pub mod areas;
pub mod assets;
pub mod catalog;
mod docs;
pub mod env;
pub mod error;
pub mod harness;
pub mod lsp;
pub mod model;
pub mod models;
pub mod progress;
pub mod protocol;
pub mod remote;
pub mod review;
pub mod root;
pub mod run;
pub mod sink;
pub mod solutions;
pub mod ssh;
pub mod store;
mod stream;
pub mod target;
mod targets;
pub mod uv;

pub use app::{Core, CoreOptions};
pub use error::{CoreError, Result};
pub use harness::{CommandSpec, HarnessCommand, HarnessJob, ProgramHarness, UvHarness};
pub use lsp::{EditorOptions, RestartPolicy};
pub use model::*;
pub use remote::{SshProcess, SshTarget};
pub use review::{Clock, SystemClock};
pub use root::{ResolvedRoot, RootInputs, RootSource, resolve_root};
pub use run::RunLimits;
pub use sink::EventSink;
pub use ssh::{Ssh, SshOptions};
pub use target::{LOCAL_TARGET, LocalTarget, Process, Target};

/// The current time as ISO 8601 in UTC with millisecond precision.
pub(crate) fn now_iso8601() -> String {
    chrono::Utc::now().to_rfc3339_opts(chrono::SecondsFormat::Millis, true)
}
