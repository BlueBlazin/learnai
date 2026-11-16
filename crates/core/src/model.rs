//! Wire types shared with the front end.
//!
//! These mirror the TypeScript declarations in `docs/ARCHITECTURE.md` section 4 exactly:
//! field names are camelCase on the wire and enum values use the same strings.

use std::collections::BTreeMap;
use std::fmt;
use std::str::FromStr;

use serde::{Deserialize, Serialize};

/// Declares a string-valued wire enum with `as_str`, `Display`, and `FromStr`.
macro_rules! wire_enum {
    (
        $(#[$meta:meta])*
        $name:ident { $( $(#[$vmeta:meta])* $variant:ident => $text:literal ),+ $(,)? }
    ) => {
        $(#[$meta])*
        #[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
        pub enum $name {
            $( $(#[$vmeta])* #[serde(rename = $text)] $variant ),+
        }

        impl $name {
            /// Every value, in declaration order.
            pub const ALL: &'static [$name] = &[$($name::$variant),+];

            /// The wire representation of this value.
            pub const fn as_str(self) -> &'static str {
                match self { $($name::$variant => $text),+ }
            }
        }

        impl fmt::Display for $name {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                f.write_str(self.as_str())
            }
        }

        impl FromStr for $name {
            type Err = String;

            fn from_str(s: &str) -> Result<Self, Self::Err> {
                match s {
                    $($text => Ok($name::$variant),)+
                    other => Err(format!(
                        concat!("unknown ", stringify!($name), " `{}`"),
                        other
                    )),
                }
            }
        }
    };
}

wire_enum! {
    /// A framework a solution may be written in.
    Framework {
        /// PyTorch, the oracle framework.
        Torch => "torch",
        /// JAX.
        Jax => "jax",
        /// MLX.
        Mlx => "mlx",
        /// Framework-free Python.
        Python => "python",
    }
}

impl Framework {
    /// The optional uv extra that provides this framework, if it is not a base dependency.
    pub const fn extra(self) -> Option<Extra> {
        match self {
            Framework::Jax => Some(Extra::Jax),
            Framework::Mlx => Some(Extra::Mlx),
            Framework::Torch | Framework::Python => None,
        }
    }
}

wire_enum! {
    /// Problem difficulty.
    Difficulty {
        /// Easy.
        Easy => "easy",
        /// Medium.
        Medium => "medium",
        /// Hard.
        Hard => "hard",
    }
}

wire_enum! {
    /// The kind of judge a problem uses.
    JudgeKind {
        /// Output matches the oracle within tolerance.
        Numeric => "numeric",
        /// An invariant holds on seeded random inputs.
        Property => "property",
        /// Output distribution matches the target.
        Statistical => "statistical",
        /// Instrumented work counts.
        Cost => "cost",
        /// Time relative to the reference.
        Speed => "speed",
        /// Discrete-event simulation with virtual time.
        Simclock => "simclock",
        /// HTTP client against the solution's ASGI app.
        Service => "service",
    }
}

wire_enum! {
    /// Hardware a problem requires.
    Requires {
        /// Runs on CPU.
        Cpu => "cpu",
        /// Needs the Apple GPU through MLX.
        MlxGpu => "mlx-gpu",
        /// Needs an NVIDIA GPU.
        Cuda => "cuda",
    }
}

wire_enum! {
    /// Progress on a problem.
    ProblemStatus {
        /// Untouched.
        Todo => "todo",
        /// A solution differs from the starter, or a submit exists.
        Attempted => "attempted",
        /// At least one framework has an accepted submit.
        Solved => "solved",
    }
}

wire_enum! {
    /// Whether a run executes public cases only or all cases.
    RunMode {
        /// Public cases only; results are ephemeral.
        Run => "run",
        /// All cases; stored as a submission.
        Submit => "submit",
    }
}

wire_enum! {
    /// Outcome of a single case.
    CaseStatus {
        /// The case passed.
        Pass => "pass",
        /// The case produced a wrong result.
        Fail => "fail",
        /// The solution raised.
        Error => "error",
        /// The case exceeded its time limit.
        Timeout => "timeout",
        /// The case did not run.
        Skipped => "skipped",
    }
}

wire_enum! {
    /// Final outcome of a run.
    Verdict {
        /// Every case passed.
        Accepted => "accepted",
        /// At least one case failed.
        WrongAnswer => "wrong_answer",
        /// The solution raised.
        RuntimeError => "runtime_error",
        /// A case or the whole run exceeded its time limit.
        TimeLimit => "time_limit",
        /// The solution could not be imported.
        ImportError => "import_error",
        /// The harness itself failed.
        HarnessError => "harness_error",
        /// The user canceled the run.
        Canceled => "canceled",
        /// Nothing that ran failed, but a case was skipped for a model asset that is
        /// not downloaded. Not judged: neither accepted nor a failure.
        AssetsMissing => "assets_missing",
        /// Nothing that ran failed, but this machine lacks the hardware the problem
        /// (or one of its cases) requires. Not judged.
        HardwareMissing => "hardware_missing",
    }
}

impl Verdict {
    /// Whether the run ended without judging the solution on every case: nothing that
    /// ran failed, but something was skipped for a missing asset or missing hardware.
    pub const fn is_not_judged(self) -> bool {
        matches!(self, Verdict::AssetsMissing | Verdict::HardwareMissing)
    }
}

wire_enum! {
    /// An optional dependency set (a uv extra) of the Python project.
    Extra {
        /// JAX.
        Jax => "jax",
        /// MLX.
        Mlx => "mlx",
        /// Downloading model assets (`learnai assets fetch`) and authoring fixtures.
        Models => "models",
    }
}

wire_enum! {
    /// Whether a model asset is usable.
    AssetState {
        /// The declared files are on disk at the declared revision.
        Present => "present",
        /// Nothing usable is on disk.
        Missing => "missing",
        /// A directory exists but it is another revision, lacks files, or is damaged.
        Incomplete => "incomplete",
    }
}

/// A track: an ordered group of problems.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Track {
    /// Track id; equals the directory name.
    pub id: String,
    /// Display title.
    pub title: String,
    /// Short description.
    pub description: String,
    /// Position among tracks.
    pub order: i64,
    /// Number of valid problems in the track.
    pub problem_count: u32,
    /// Number of those problems that are solved.
    pub solved_count: u32,
}

/// Catalog metadata of a problem plus the user's progress on it.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ProblemSummary {
    /// Globally unique id; equals the directory name.
    pub id: String,
    /// Display title.
    pub title: String,
    /// Id of the owning track.
    pub track: String,
    /// Position within the track.
    pub order: i64,
    /// Difficulty.
    pub difficulty: Difficulty,
    /// Frameworks a solution may be written in.
    pub frameworks: Vec<Framework>,
    /// Judges the problem uses.
    pub judges: Vec<JudgeKind>,
    /// Free-form tags.
    pub tags: Vec<String>,
    /// Ids of prerequisite problems.
    pub prerequisites: Vec<String>,
    /// Hardware requirement.
    pub requires: Requires,
    /// Derived progress.
    pub status: ProblemStatus,
    /// Frameworks with an accepted submit.
    pub solved_frameworks: Vec<Framework>,
    /// Names of the model assets the problem declares.
    #[serde(default)]
    pub assets: Vec<String>,
}

/// Everything the problem screen needs.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ProblemDetail {
    /// Summary fields, flattened.
    #[serde(flatten)]
    pub summary: ProblemSummary,
    /// Statement Markdown.
    pub statement: String,
    /// Progressive hints, Markdown each.
    pub hints: Vec<String>,
    /// Editorial Markdown; `None` while locked.
    pub editorial: Option<String>,
    /// True once solved or explicitly revealed.
    pub editorial_unlocked: bool,
    /// Names the solution must define.
    pub entry: Vec<String>,
    /// Default per-case time limit in seconds.
    pub time_limit_s: f64,
}

/// A solution file as shown in the editor.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SolutionFile {
    /// File contents (the starter when nothing has been saved).
    pub code: String,
    /// True when `code` equals the starter.
    pub is_starter: bool,
    /// Absolute path of the workspace file (it may not exist yet).
    pub path: String,
}

/// A named measurement reported by a judge.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Metric {
    /// Metric name.
    pub name: String,
    /// Measured value.
    pub value: f64,
    /// Unit, if any.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub unit: Option<String>,
    /// Limit the value is judged against, if any.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub limit: Option<f64>,
    /// True for metrics where larger is better (throughput): the limit is then a
    /// minimum. The harness writes `higher_is_better`.
    #[serde(
        default,
        alias = "higher_is_better",
        skip_serializing_if = "std::ops::Not::not"
    )]
    pub higher_is_better: bool,
}

impl Metric {
    /// A metric with no unit and no limit.
    pub fn new(name: impl Into<String>, value: f64) -> Self {
        Self {
            name: name.into(),
            value,
            unit: None,
            limit: None,
            higher_is_better: false,
        }
    }
}

/// A case announced at the start of a run.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CaseInfo {
    /// Case id.
    pub id: String,
    /// Display name (generic for hidden cases).
    pub name: String,
    /// Whether the case is public.
    pub public: bool,
    /// Hardware the case needs; absent when it runs everywhere.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub requires: Option<Requires>,
    /// The stage the case belongs to (capstones); absent for unstaged problems. Public
    /// even for hidden cases.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stage: Option<String>,
}

impl CaseInfo {
    /// A case that runs everywhere.
    pub fn new(id: impl Into<String>, name: impl Into<String>, public: bool) -> Self {
        Self {
            id: id.into(),
            name: name.into(),
            public,
            requires: None,
            stage: None,
        }
    }
}

/// Why a case was not run.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "lowercase")]
pub enum Skip {
    /// The machine lacks the hardware the case requires.
    Hardware {
        /// The missing hardware.
        requires: Requires,
    },
    /// A model asset the case needs is not downloaded.
    Asset {
        /// The asset's name.
        asset: String,
    },
    /// The run stopped before reaching the case.
    Aborted,
}

/// Reads an optional field, treating a value this build cannot interpret as absent.
/// Additions to the protocol (a new skip kind, say) must never make a whole event or a
/// stored row unreadable.
pub(crate) fn lenient<'de, D, T>(deserializer: D) -> Result<Option<T>, D::Error>
where
    D: serde::Deserializer<'de>,
    T: serde::de::DeserializeOwned,
{
    let value = Option::<serde_json::Value>::deserialize(deserializer)?;
    Ok(value.and_then(|value| serde_json::from_value(value).ok()))
}

/// A model asset a run needed and did not find.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct MissingAsset {
    /// Asset name.
    pub name: String,
    /// Hugging Face repository id.
    #[serde(default)]
    pub repo: String,
    /// Declared approximate download size in megabytes. The harness writes `size_mb`.
    #[serde(default, alias = "size_mb")]
    pub size_mb: f64,
}

/// The result of one case.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CaseResult {
    /// Case id.
    pub id: String,
    /// Outcome.
    pub status: CaseStatus,
    /// Duration in milliseconds.
    pub duration_ms: f64,
    /// Short failure reason.
    pub message: Option<String>,
    /// Longer failure text.
    pub detail: Option<String>,
    /// Captured prints.
    pub stdout: String,
    /// Metrics reported for the case.
    pub metrics: Vec<Metric>,
    /// Why the case was skipped; present exactly when `status` is `skipped` and the
    /// harness said why.
    #[serde(
        default,
        deserialize_with = "lenient",
        skip_serializing_if = "Option::is_none"
    )]
    pub skip: Option<Skip>,
}

/// Which output stream a diagnostic line came from.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum LogStream {
    /// Standard output (non-protocol lines).
    Stdout,
    /// Standard error.
    Stderr,
}

/// An event in the stream produced by `start_run`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(
    tag = "type",
    rename_all = "camelCase",
    rename_all_fields = "camelCase"
)]
pub enum RunEvent {
    /// Progress before the harness reports anything (environment sync, spawn).
    Preparing {
        /// Human-readable progress message.
        message: String,
    },
    /// The harness started and announced its cases.
    Start {
        /// Cases that will run.
        cases: Vec<CaseInfo>,
        /// Runtime environment description.
        env: BTreeMap<String, String>,
        /// Hardware the problem requires to be accepted.
        requires: Requires,
        /// Whether this machine has it.
        requires_met: bool,
    },
    /// A case began.
    CaseStart {
        /// Case id.
        id: String,
    },
    /// A case finished.
    CaseEnd {
        /// The case result.
        result: CaseResult,
    },
    /// Diagnostic output.
    Log {
        /// Source stream.
        stream: LogStream,
        /// One line of text, without the trailing newline.
        text: String,
    },
    /// The run finished. Always the last event, emitted exactly once.
    End {
        /// Final verdict.
        verdict: Verdict,
        /// Number of passed cases.
        passed: u32,
        /// Number of cases.
        total: u32,
        /// Duration in milliseconds.
        duration_ms: f64,
        /// Run-level metrics.
        metrics: Vec<Metric>,
        /// Id of the stored submission, for `submit` runs that were stored.
        submission_id: Option<i64>,
        /// Number of cases that were not run.
        skipped: u32,
        /// With `assets_missing`: the assets to download.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        missing_assets: Option<Vec<MissingAsset>>,
        /// With `hardware_missing`: the hardware this machine lacks.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        missing_hardware: Option<Requires>,
        /// Short reason the run as a whole failed (import and harness errors, limits),
        /// or what was skipped and why for the not-judged verdicts.
        message: Option<String>,
        /// Longer text for that failure, such as a traceback.
        detail: Option<String>,
    },
}

/// A stored submit run.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Submission {
    /// Submission id.
    pub id: i64,
    /// Problem id.
    pub problem_id: String,
    /// Framework of the solution.
    pub framework: Framework,
    /// Always `submit` for stored submissions.
    pub mode: RunMode,
    /// Final verdict.
    pub verdict: Verdict,
    /// Number of passed cases.
    pub passed: u32,
    /// Number of cases.
    pub total: u32,
    /// Number of cases that were not run.
    #[serde(default)]
    pub skipped: u32,
    /// Duration in milliseconds.
    pub duration_ms: f64,
    /// Name of the target the run executed on.
    pub target: String,
    /// Creation time, ISO 8601 in UTC.
    pub created_at: String,
    /// Whether this was a review attempt, judged from the review file.
    #[serde(default)]
    pub review: bool,
}

/// A case result as stored with a submission: the result plus the case's display name
/// and visibility, so history reads the same as the live run.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct StoredCase {
    /// The case result, flattened.
    #[serde(flatten)]
    pub result: CaseResult,
    /// Display name (generic for hidden cases). Rows stored before names were kept
    /// fall back to the case id.
    #[serde(default)]
    pub name: String,
    /// Whether the case is public.
    #[serde(default = "default_public")]
    pub public: bool,
    /// The stage the case belongs to, if the problem has stages.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stage: Option<String>,
}

fn default_public() -> bool {
    true
}

impl StoredCase {
    /// Pairs a result with the case announced under the same id, if any.
    pub fn new(result: CaseResult, info: Option<&CaseInfo>) -> Self {
        Self {
            name: info.map_or_else(|| result.id.clone(), |i| i.name.clone()),
            public: info.is_none_or(|i| i.public),
            stage: info.and_then(|i| i.stage.clone()),
            result,
        }
    }
}

/// A malformed track or problem that the catalog skipped, or an inconsistency in it.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CatalogIssue {
    /// The offending file or directory.
    pub path: String,
    /// What is wrong.
    pub message: String,
}

/// One stage of a study path.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PathStage {
    /// Display title.
    pub title: String,
    /// One-sentence description.
    pub description: String,
    /// Ids of existing problems, in study order.
    pub problems: Vec<String>,
}

/// A curated, ordered study plan with progress. `Path` in the TypeScript contract; named
/// `StudyPath` here so it does not collide with [`std::path::Path`].
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct StudyPath {
    /// Path id.
    pub id: String,
    /// Display title.
    pub title: String,
    /// One-sentence description.
    pub description: String,
    /// Stages in order.
    pub stages: Vec<PathStage>,
    /// Number of problems on the path.
    pub problem_count: u32,
    /// Number of those that are solved.
    pub solved_count: u32,
}

/// A submission with its code snapshot and per-case results.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SubmissionDetail {
    /// Summary fields, flattened.
    #[serde(flatten)]
    pub submission: Submission,
    /// The submitted code.
    pub code: String,
    /// Per-case results, with names.
    pub cases: Vec<StoredCase>,
    /// Run-level metrics.
    pub metrics: Vec<Metric>,
}

/// Installation state of one framework.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct FrameworkEnv {
    /// The framework.
    pub framework: Framework,
    /// Whether it is importable in the Python environment.
    pub installed: bool,
    /// Installed version, if any.
    pub version: Option<String>,
}

/// State of the local toolchain and Python environment.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct EnvStatus {
    /// uv version, or `None` when uv is not available.
    pub uv: Option<String>,
    /// Python version of the project environment, or `None` when unavailable.
    pub python: Option<String>,
    /// One entry per framework.
    pub frameworks: Vec<FrameworkEnv>,
    /// Absolute content root.
    pub root: String,
}

/// Aggregate progress.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Stats {
    /// Number of solved problems.
    pub solved: u32,
    /// Number of problems.
    pub total: u32,
    /// `(solved, total)` per difficulty; every difficulty is present.
    pub by_difficulty: BTreeMap<Difficulty, (u32, u32)>,
    /// Consecutive days with an accepted submit, ending today or yesterday.
    pub streak_days: u32,
    /// Number of reviews that are due now; 0 while reviews are turned off.
    #[serde(default)]
    pub reviews_due: u32,
}

/// The review schedule of a solved problem (spaced re-solve).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Review {
    /// Problem id.
    pub problem_id: String,
    /// 0-based index into the review intervals.
    pub stage: u32,
    /// The interval of this stage in days.
    pub interval_days: u32,
    /// When the review is due, ISO 8601 in UTC.
    pub due_at: String,
    /// When the problem was last reviewed successfully.
    pub last_reviewed_at: Option<String>,
    /// Whether the review is due now.
    pub due: bool,
}

/// How the learner resolves a review without solving it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ReviewAction {
    /// The learner gave up and looked at their solution: back to the first interval.
    Forgot,
    /// Ask again a day later.
    Snooze,
    /// Stop reviewing the problem.
    Retire,
}

/// App settings stored in the state database.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Settings {
    /// Whether solved problems are scheduled for review.
    pub reviews_enabled: bool,
}

/// An event in the stream produced by `env_install`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(
    tag = "type",
    rename_all = "camelCase",
    rename_all_fields = "camelCase"
)]
pub enum InstallEvent {
    /// One line of installer output.
    Log {
        /// The line, without the trailing newline.
        text: String,
    },
    /// The install finished. Always the last event, emitted exactly once.
    Done {
        /// Whether the install succeeded.
        ok: bool,
        /// Failure reason, if any.
        message: Option<String>,
    },
}

/// A model asset declared by the catalog, with its state on disk.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ModelAsset {
    /// Directory name under the models directory.
    pub name: String,
    /// Hugging Face repository id.
    pub repo: String,
    /// Pinned commit hash.
    pub revision: String,
    /// Declared file names or globs.
    pub files: Vec<String>,
    /// Declared approximate download size in megabytes.
    pub size_mb: f64,
    /// Whether the asset is usable.
    pub state: AssetState,
    /// Why the asset is incomplete.
    pub reason: Option<String>,
    /// Absolute directory of the asset (it may not exist).
    pub path: String,
    /// Bytes currently on disk.
    pub bytes_on_disk: u64,
    /// Ids of the problems that declare the asset.
    pub problems: Vec<String>,
}

/// Every model asset the catalog declares.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AssetList {
    /// Absolute directory that holds the assets.
    pub models_dir: String,
    /// Assets sorted by name.
    pub assets: Vec<ModelAsset>,
}

/// An event in the stream produced by `fetch_assets`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(
    tag = "type",
    rename_all = "camelCase",
    rename_all_fields = "camelCase"
)]
pub enum AssetEvent {
    /// Progress before the fetch reports anything (environment sync).
    Preparing {
        /// Human-readable progress message.
        message: String,
    },
    /// The file list of an asset is known and its download begins.
    Start {
        /// Asset name.
        asset: String,
        /// Size of the whole asset in bytes.
        total_bytes: u64,
    },
    /// Download progress of an asset.
    Progress {
        /// Asset name.
        asset: String,
        /// File being downloaded.
        file: String,
        /// Bytes of the asset that are complete.
        done_bytes: u64,
        /// Size of the whole asset in bytes.
        total_bytes: u64,
    },
    /// An asset finished. Emitted once per asset.
    AssetEnd {
        /// Asset name.
        asset: String,
        /// Whether it is now present.
        ok: bool,
        /// One line saying what went wrong.
        message: Option<String>,
    },
    /// Diagnostic output.
    Log {
        /// One line of text, without the trailing newline.
        text: String,
    },
    /// The fetch finished. Always the last event, emitted exactly once.
    Done {
        /// Whether every asset succeeded.
        ok: bool,
        /// Whether the fetch was canceled.
        canceled: bool,
        /// Failure reason, if any.
        message: Option<String>,
    },
}

/// A host alias from the user's SSH configuration (no wildcard patterns).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SshHost {
    /// The alias, as written after `Host`.
    pub alias: String,
    /// `HostName`, if the block sets it.
    pub host_name: Option<String>,
    /// `User`, if the block sets it.
    pub user: Option<String>,
    /// `Port`, if the block sets it.
    pub port: Option<u16>,
}

wire_enum! {
    /// A step of a target check, in the order the steps run.
    CheckStepId {
        /// Reach the host over SSH without a prompt.
        Connect => "connect",
        /// Find uv.
        Uv => "uv",
        /// Copy the `python/` project.
        Sync => "sync",
        /// Build the Python environment.
        Environment => "environment",
        /// Run `learnai doctor --json`.
        Doctor => "doctor",
    }
}

/// The outcome of one step of a target check.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CheckStep {
    /// Which step.
    pub id: CheckStepId,
    /// Whether it succeeded.
    pub ok: bool,
    /// One line: what was found, or what went wrong.
    pub message: String,
    /// A command or instruction for the user when the step failed.
    pub fix: Option<String>,
}

/// What `learnai doctor --json` reported on a target, reduced to what the app shows.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct RemoteDoctor {
    /// Python version.
    pub python: String,
    /// `platform.system()`: `Linux`, `Darwin`.
    pub system: String,
    /// `platform.machine()`: `x86_64`, `arm64`.
    pub machine: String,
    /// Installed version of each tensor framework, or `None` when it is not installed.
    pub frameworks: BTreeMap<Framework, Option<String>>,
    /// The hardware that is present; always includes `cpu`.
    pub hardware: Vec<Requires>,
    /// Name of the GPU (CUDA device 0, else the Apple GPU).
    pub gpu_name: Option<String>,
    /// Its memory in bytes.
    pub gpu_memory_bytes: Option<u64>,
}

/// The result of checking a target.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TargetCheck {
    /// Whether every step succeeded.
    pub ok: bool,
    /// When the check finished (ISO 8601).
    pub checked_at: String,
    /// The steps that ran, in order; the last one is the failing one when `ok` is false.
    pub steps: Vec<CheckStep>,
    /// The doctor report, when the check got that far.
    pub doctor: Option<RemoteDoctor>,
}

wire_enum! {
    /// What kind of place a target is.
    TargetKind {
        /// This machine.
        Local => "local",
        /// A remote host reached over SSH.
        Ssh => "ssh",
    }
}

/// A place the harness can run.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TargetInfo {
    /// `local` or `ssh:<alias>`.
    pub name: String,
    /// Local or SSH.
    pub kind: TargetKind,
    /// Display name.
    pub label: String,
    /// The SSH host alias; `None` for the local target.
    pub host: Option<String>,
    /// The directory on the remote host the app works in; `None` for the local target.
    pub remote_dir: Option<String>,
    /// The latest check, if one has run.
    pub last_check: Option<TargetCheck>,
}

/// An event in the stream produced by `check_target`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(
    tag = "type",
    rename_all = "camelCase",
    rename_all_fields = "camelCase"
)]
pub enum TargetCheckEvent {
    /// A step finished.
    Step {
        /// Its outcome.
        step: CheckStep,
    },
    /// Diagnostic output.
    Log {
        /// One line of text, without the trailing newline.
        text: String,
    },
    /// The check finished. Always the last event, emitted exactly once.
    Done {
        /// The whole result, as saved in `lastCheck`.
        check: TargetCheck,
    },
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn enums_use_contract_strings() {
        assert_eq!(json!(Requires::MlxGpu), json!("mlx-gpu"));
        assert_eq!(json!(Verdict::WrongAnswer), json!("wrong_answer"));
        assert_eq!(json!(Verdict::Canceled), json!("canceled"));
        assert_eq!(json!(JudgeKind::Simclock), json!("simclock"));
        assert_eq!("time_limit".parse::<Verdict>(), Ok(Verdict::TimeLimit));
        assert!("nope".parse::<Framework>().is_err());
    }

    #[test]
    fn run_events_are_tagged_and_camel_case() {
        let end = RunEvent::End {
            verdict: Verdict::Accepted,
            passed: 2,
            total: 2,
            duration_ms: 1.5,
            metrics: vec![Metric {
                limit: Some(4.0),
                ..Metric::new("n", 3.0)
            }],
            submission_id: None,
            skipped: 0,
            missing_assets: None,
            missing_hardware: None,
            message: Some("m".into()),
            detail: None,
        };
        assert_eq!(
            json!(end),
            json!({
                "type": "end", "verdict": "accepted", "passed": 2, "total": 2,
                "durationMs": 1.5, "metrics": [{"name": "n", "value": 3.0, "limit": 4.0}],
                "submissionId": null, "skipped": 0, "message": "m", "detail": null
            })
        );
        let case = RunEvent::CaseEnd {
            result: CaseResult {
                id: "c1".into(),
                status: CaseStatus::Fail,
                duration_ms: 2.0,
                message: Some("m".into()),
                detail: None,
                stdout: String::new(),
                metrics: vec![],
                skip: None,
            },
        };
        assert_eq!(
            json!(case),
            json!({"type": "caseEnd", "result": {
                "id": "c1", "status": "fail", "durationMs": 2.0, "message": "m",
                "detail": null, "stdout": "", "metrics": []
            }})
        );
        assert_eq!(
            json!(RunEvent::CaseStart { id: "c1".into() }),
            json!({"type": "caseStart", "id": "c1"})
        );
        assert_eq!(
            json!(RunEvent::Log {
                stream: LogStream::Stderr,
                text: "x".into()
            }),
            json!({"type": "log", "stream": "stderr", "text": "x"})
        );
    }

    #[test]
    fn stored_cases_are_flat_and_tolerate_old_rows() {
        let result = CaseResult {
            id: "c1".into(),
            status: CaseStatus::Pass,
            duration_ms: 2.0,
            message: None,
            detail: None,
            stdout: String::new(),
            metrics: vec![],
            skip: None,
        };
        let info = CaseInfo::new("c1", "Hidden case 1", false);
        let stored = StoredCase::new(result.clone(), Some(&info));
        assert_eq!(
            json!(stored),
            json!({"id": "c1", "status": "pass", "durationMs": 2.0, "message": null,
                   "detail": null, "stdout": "", "metrics": [],
                   "name": "Hidden case 1", "public": false})
        );
        let unnamed = StoredCase::new(result.clone(), None);
        assert_eq!((unnamed.name.as_str(), unnamed.public), ("c1", true));

        // A row written before names were stored.
        let old: StoredCase = serde_json::from_value(json!(result)).expect("old row");
        assert_eq!((old.name.as_str(), old.public), ("", true));
        assert_eq!(
            json!(CatalogIssue {
                path: "/r/problems/t/p/problem.toml".into(),
                message: "title is empty".into()
            }),
            json!({"path": "/r/problems/t/p/problem.toml", "message": "title is empty"})
        );
    }

    #[test]
    fn skipped_cases_and_not_judged_endings_match_the_contract() {
        assert_eq!(json!(Verdict::AssetsMissing), json!("assets_missing"));
        assert_eq!(
            "hardware_missing".parse::<Verdict>(),
            Ok(Verdict::HardwareMissing)
        );
        assert!(Verdict::AssetsMissing.is_not_judged());
        assert!(Verdict::HardwareMissing.is_not_judged());
        assert!(!Verdict::Accepted.is_not_judged());
        assert!(!Verdict::Canceled.is_not_judged());

        assert_eq!(
            json!(Skip::Hardware {
                requires: Requires::Cuda
            }),
            json!({"kind": "hardware", "requires": "cuda"})
        );
        assert_eq!(
            json!(Skip::Asset {
                asset: "granite".into()
            }),
            json!({"kind": "asset", "asset": "granite"})
        );
        assert_eq!(json!(Skip::Aborted), json!({"kind": "aborted"}));

        // A skip kind from a newer harness is dropped, not an error.
        let result: CaseResult = serde_json::from_value(json!({
            "id": "c1", "status": "skipped", "durationMs": 0.0, "message": "m",
            "detail": null, "stdout": "", "metrics": [], "skip": {"kind": "quota", "n": 3}
        }))
        .expect("unknown skip kind");
        assert_eq!(result.skip, None);

        let end = RunEvent::End {
            verdict: Verdict::AssetsMissing,
            passed: 2,
            total: 3,
            duration_ms: 1.0,
            metrics: vec![Metric {
                unit: Some("tok/s".into()),
                limit: Some(20.0),
                higher_is_better: true,
                ..Metric::new("throughput", 41.0)
            }],
            submission_id: Some(7),
            skipped: 1,
            missing_assets: Some(vec![MissingAsset {
                name: "granite".into(),
                repo: "ibm/granite".into(),
                size_mb: 712.0,
            }]),
            missing_hardware: None,
            message: Some("m".into()),
            detail: None,
        };
        assert_eq!(
            json!(end),
            json!({
                "type": "end", "verdict": "assets_missing", "passed": 2, "total": 3,
                "durationMs": 1.0, "submissionId": 7, "skipped": 1,
                "metrics": [{"name": "throughput", "value": 41.0, "unit": "tok/s",
                             "limit": 20.0, "higherIsBetter": true}],
                "missingAssets": [{"name": "granite", "repo": "ibm/granite", "sizeMb": 712.0}],
                "message": "m", "detail": null
            })
        );
        assert_eq!(
            json!(RunEvent::Start {
                cases: vec![CaseInfo {
                    requires: Some(Requires::Cuda),
                    ..CaseInfo::new("c1", "On CUDA", true)
                }],
                env: BTreeMap::new(),
                requires: Requires::Cuda,
                requires_met: false,
            }),
            json!({"type": "start", "env": {}, "requires": "cuda", "requiresMet": false,
                   "cases": [{"id": "c1", "name": "On CUDA", "public": true, "requires": "cuda"}]})
        );
    }

    #[test]
    fn assets_match_the_contract() {
        let asset = ModelAsset {
            name: "granite".into(),
            repo: "ibm/granite".into(),
            revision: "abc".into(),
            files: vec!["config.json".into()],
            size_mb: 712.0,
            state: AssetState::Incomplete,
            reason: Some("model.safetensors is missing".into()),
            path: "/m/granite".into(),
            bytes_on_disk: 12,
            problems: vec!["p".into()],
        };
        assert_eq!(
            json!(AssetList {
                models_dir: "/m".into(),
                assets: vec![asset]
            }),
            json!({"modelsDir": "/m", "assets": [{
                "name": "granite", "repo": "ibm/granite", "revision": "abc",
                "files": ["config.json"], "sizeMb": 712.0, "state": "incomplete",
                "reason": "model.safetensors is missing", "path": "/m/granite",
                "bytesOnDisk": 12, "problems": ["p"]
            }]})
        );
        assert_eq!(
            json!(AssetEvent::Progress {
                asset: "granite".into(),
                file: "model.safetensors".into(),
                done_bytes: 5,
                total_bytes: 10
            }),
            json!({"type": "progress", "asset": "granite", "file": "model.safetensors",
                   "doneBytes": 5, "totalBytes": 10})
        );
        assert_eq!(
            json!(AssetEvent::AssetEnd {
                asset: "granite".into(),
                ok: false,
                message: Some("no network".into())
            }),
            json!({"type": "assetEnd", "asset": "granite", "ok": false, "message": "no network"})
        );
        assert_eq!(
            json!(AssetEvent::Done {
                ok: false,
                canceled: true,
                message: None
            }),
            json!({"type": "done", "ok": false, "canceled": true, "message": null})
        );
    }

    #[test]
    fn paths_match_the_contract() {
        let path = StudyPath {
            id: "inference-75".into(),
            title: "Inference 75".into(),
            description: "One sentence.".into(),
            stages: vec![PathStage {
                title: "Numerics".into(),
                description: "Another.".into(),
                problems: vec!["stable-softmax".into()],
            }],
            problem_count: 1,
            solved_count: 0,
        };
        assert_eq!(
            json!(path),
            json!({
                "id": "inference-75", "title": "Inference 75", "description": "One sentence.",
                "stages": [{"title": "Numerics", "description": "Another.",
                            "problems": ["stable-softmax"]}],
                "problemCount": 1, "solvedCount": 0
            })
        );
    }

    #[test]
    fn stats_and_install_events_match_the_contract() {
        let stats = Stats {
            solved: 1,
            total: 3,
            by_difficulty: Difficulty::ALL.iter().map(|d| (*d, (0, 1))).collect(),
            streak_days: 2,
            reviews_due: 4,
        };
        assert_eq!(
            json!(stats),
            json!({"solved": 1, "total": 3, "streakDays": 2, "reviewsDue": 4,
                   "byDifficulty": {"easy": [0, 1], "medium": [0, 1], "hard": [0, 1]}})
        );
        assert_eq!(
            json!(InstallEvent::Done {
                ok: false,
                message: None
            }),
            json!({"type": "done", "ok": false, "message": null})
        );
    }

    #[test]
    fn reviews_stages_and_settings_match_the_contract() {
        let review = Review {
            problem_id: "stable-softmax".into(),
            stage: 1,
            interval_days: 7,
            due_at: "2026-11-15T10:00:00.000Z".into(),
            last_reviewed_at: None,
            due: true,
        };
        assert_eq!(
            json!(review),
            json!({"problemId": "stable-softmax", "stage": 1, "intervalDays": 7,
                   "dueAt": "2026-11-15T10:00:00.000Z", "lastReviewedAt": null, "due": true})
        );
        for (action, text) in [
            (ReviewAction::Forgot, "forgot"),
            (ReviewAction::Snooze, "snooze"),
            (ReviewAction::Retire, "retire"),
        ] {
            assert_eq!(json!(action), json!(text));
            assert_eq!(
                serde_json::from_value::<ReviewAction>(json!(text)).expect("action"),
                action
            );
        }
        assert_eq!(
            json!(Settings {
                reviews_enabled: true
            }),
            json!({"reviewsEnabled": true})
        );

        // A stage travels from the announcement to the stored case; without one the
        // field is absent, so unstaged problems serialize exactly as before.
        let staged: CaseInfo = serde_json::from_value(
            json!({"id": "c1", "name": "Tokenize", "public": true, "stage": "1: Tokenizer"}),
        )
        .expect("case");
        assert_eq!(staged.stage.as_deref(), Some("1: Tokenizer"));
        let result = CaseResult {
            id: "c1".into(),
            status: CaseStatus::Pass,
            duration_ms: 1.0,
            message: None,
            detail: None,
            stdout: String::new(),
            metrics: vec![],
            skip: None,
        };
        let stored = json!(StoredCase::new(result.clone(), Some(&staged)));
        assert_eq!(stored["stage"], "1: Tokenizer");
        let plain = json!(StoredCase::new(
            result,
            Some(&CaseInfo::new("c1", "Tokenize", true))
        ));
        assert!(plain.get("stage").is_none());
        assert!(json!(CaseInfo::new("c1", "n", true)).get("stage").is_none());
        // Rows stored before stages existed still load.
        let old: StoredCase = serde_json::from_value(plain).expect("stored case");
        assert_eq!(old.stage, None);
    }

    #[test]
    fn targets_match_the_contract() {
        let check = TargetCheck {
            ok: false,
            checked_at: "2026-11-15T10:00:00.000Z".into(),
            steps: vec![CheckStep {
                id: CheckStepId::Uv,
                ok: false,
                message: "uv was not found".into(),
                fix: Some("curl -LsSf https://astral.sh/uv/install.sh | sh".into()),
            }],
            doctor: Some(RemoteDoctor {
                python: "3.15.2".into(),
                system: "Linux".into(),
                machine: "x86_64".into(),
                frameworks: BTreeMap::from([
                    (Framework::Torch, Some("2.15.0".into())),
                    (Framework::Mlx, None),
                ]),
                hardware: vec![Requires::Cpu, Requires::Cuda],
                gpu_name: Some("NVIDIA GeForce RTX 4090".into()),
                gpu_memory_bytes: Some(25_757_220_864),
            }),
        };
        let info = TargetInfo {
            name: "ssh:gpu-box".into(),
            kind: TargetKind::Ssh,
            label: "gpu-box".into(),
            host: Some("gpu-box".into()),
            remote_dir: Some("~/.learnai".into()),
            last_check: Some(check.clone()),
        };
        assert_eq!(
            json!(info),
            json!({
                "name": "ssh:gpu-box", "kind": "ssh", "label": "gpu-box", "host": "gpu-box",
                "remoteDir": "~/.learnai",
                "lastCheck": {
                    "ok": false, "checkedAt": "2026-11-15T10:00:00.000Z",
                    "steps": [{"id": "uv", "ok": false, "message": "uv was not found",
                               "fix": "curl -LsSf https://astral.sh/uv/install.sh | sh"}],
                    "doctor": {"python": "3.15.2", "system": "Linux", "machine": "x86_64",
                               "frameworks": {"torch": "2.15.0", "mlx": null},
                               "hardware": ["cpu", "cuda"],
                               "gpuName": "NVIDIA GeForce RTX 4090",
                               "gpuMemoryBytes": 25_757_220_864u64}
                }
            })
        );
        assert_eq!(
            json!(TargetCheckEvent::Done {
                check: check.clone()
            })["type"],
            json!("done")
        );
        assert_eq!(
            json!(TargetCheckEvent::Step {
                step: check.steps[0].clone()
            }),
            json!({"type": "step", "step": {"id": "uv", "ok": false, "message": "uv was not found",
                   "fix": "curl -LsSf https://astral.sh/uv/install.sh | sh"}})
        );
        assert_eq!(
            json!(SshHost {
                alias: "gpu-box".into(),
                host_name: Some("10.0.0.5".into()),
                user: None,
                port: Some(2222)
            }),
            json!({"alias": "gpu-box", "hostName": "10.0.0.5", "user": null, "port": 2222})
        );
    }
}
