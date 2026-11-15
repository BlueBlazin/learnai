//! The error type returned by every core operation.

use std::path::PathBuf;

use crate::model::Framework;

/// Convenience alias used throughout the crate.
pub type Result<T, E = CoreError> = std::result::Result<T, E>;

/// Everything that can go wrong in core.
#[derive(Debug, thiserror::Error)]
pub enum CoreError {
    /// No usable content root could be found.
    #[error("content root not found: {0}")]
    Root(String),

    /// The requested problem does not exist (or is malformed and was skipped).
    #[error("unknown problem `{0}`")]
    ProblemNotFound(String),

    /// The problem does not support the requested framework.
    #[error("problem `{problem}` does not support the `{framework}` framework")]
    FrameworkNotSupported {
        /// Problem id.
        problem: String,
        /// The unsupported framework.
        framework: Framework,
    },

    /// The requested submission does not exist.
    #[error("unknown submission {0}")]
    SubmissionNotFound(i64),

    /// The requested run target does not exist.
    #[error("unknown target `{0}`")]
    UnknownTarget(String),

    /// The reference solution was requested before the editorial was unlocked.
    #[error("the editorial for `{0}` is locked")]
    EditorialLocked(String),

    /// A required file of a valid problem is missing.
    #[error("missing file {}", .0.display())]
    MissingFile(PathBuf),

    /// A statement asset is larger than core is willing to serve.
    #[error("asset {} is {size} bytes; the limit is {limit} bytes", .path.display())]
    AssetTooLarge {
        /// The asset file.
        path: PathBuf,
        /// Its size in bytes.
        size: u64,
        /// The largest size served.
        limit: u64,
    },

    /// Another install is already in progress.
    #[error("an environment install is already running")]
    InstallInProgress,

    /// A model asset fetch is already in progress.
    #[error("a model download is already running; wait for it to finish or cancel it")]
    FetchInProgress,

    /// Model assets could not be listed.
    #[error("{0}")]
    Assets(String),

    /// A review was requested for a problem that has no active review schedule.
    #[error("no review is scheduled for `{0}`")]
    NoActiveReview(String),

    /// A review was requested while reviews are turned off.
    #[error("reviews are turned off in Settings")]
    ReviewsDisabled,

    /// A caller-supplied argument is invalid.
    #[error("{0}")]
    InvalidArgument(String),

    /// A file-system operation failed.
    #[error("{context}: {source}")]
    Io {
        /// What was being done.
        context: String,
        /// The underlying error.
        #[source]
        source: std::io::Error,
    },

    /// The state database failed.
    #[error("state database error: {0}")]
    Database(#[from] rusqlite::Error),

    /// Stored or transmitted JSON was invalid.
    #[error("invalid JSON: {0}")]
    Json(#[from] serde_json::Error),

    /// The state database was written by a newer version of the app.
    #[error("state database schema version {found} is newer than supported version {supported}")]
    SchemaTooNew {
        /// Version found in the database.
        found: i64,
        /// Newest version this build understands.
        supported: i64,
    },
}

impl CoreError {
    /// Builds an [`CoreError::Io`] with context naming the path involved.
    pub(crate) fn io(action: &str, path: &std::path::Path, source: std::io::Error) -> Self {
        CoreError::Io {
            context: format!("failed to {action} {}", path.display()),
            source,
        }
    }

    /// Whether the error means "the thing you asked for does not exist".
    pub fn is_not_found(&self) -> bool {
        matches!(
            self,
            CoreError::ProblemNotFound(_)
                | CoreError::SubmissionNotFound(_)
                | CoreError::UnknownTarget(_)
                | CoreError::MissingFile(_)
        )
    }

    /// Whether the error is caused by the caller's request rather than by core.
    pub fn is_invalid_request(&self) -> bool {
        matches!(
            self,
            CoreError::FrameworkNotSupported { .. }
                | CoreError::EditorialLocked(_)
                | CoreError::InstallInProgress
                | CoreError::FetchInProgress
                | CoreError::InvalidArgument(_)
                | CoreError::NoActiveReview(_)
                | CoreError::ReviewsDisabled
                | CoreError::AssetTooLarge { .. }
        )
    }
}
