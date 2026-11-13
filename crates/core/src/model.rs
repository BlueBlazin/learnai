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
    pub const fn extra(self) -> Option<&'static str> {
        match self {
            Framework::Jax => Some("jax"),
            Framework::Mlx => Some("mlx"),
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
        /// Short reason the run as a whole failed (import and harness errors, limits).
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
    /// Duration in milliseconds.
    pub duration_ms: f64,
    /// Name of the target the run executed on.
    pub target: String,
    /// Creation time, ISO 8601 in UTC.
    pub created_at: String,
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
                name: "n".into(),
                value: 3.0,
                unit: None,
                limit: Some(4.0),
            }],
            submission_id: None,
            message: Some("m".into()),
            detail: None,
        };
        assert_eq!(
            json!(end),
            json!({
                "type": "end", "verdict": "accepted", "passed": 2, "total": 2,
                "durationMs": 1.5, "metrics": [{"name": "n", "value": 3.0, "limit": 4.0}],
                "submissionId": null, "message": "m", "detail": null
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
        };
        let info = CaseInfo {
            id: "c1".into(),
            name: "Hidden case 1".into(),
            public: false,
        };
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
    fn stats_and_install_events_match_the_contract() {
        let stats = Stats {
            solved: 1,
            total: 3,
            by_difficulty: Difficulty::ALL.iter().map(|d| (*d, (0, 1))).collect(),
            streak_days: 2,
        };
        assert_eq!(
            json!(stats),
            json!({"solved": 1, "total": 3, "streakDays": 2,
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
}
