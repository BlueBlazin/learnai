//! The harness event protocol (`docs/ARCHITECTURE.md` section 3.6).
//!
//! The harness writes protocol lines to stdout, each one [`PREFIX`] followed by a
//! single-line JSON object. Every other stdout line is diagnostic output.

use std::collections::BTreeMap;

use serde::Deserialize;
use serde_json::Value;

use crate::model::{
    CaseInfo, CaseResult, CaseStatus, Metric, MissingAsset, Requires, Skip, Verdict,
};

/// Marker that starts every protocol line.
pub const PREFIX: &str = "@@WS@@ ";

/// An event as the harness reports it (snake_case on the wire).
#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(tag = "event", rename_all = "snake_case")]
pub enum HarnessEvent {
    /// The harness loaded the problem and is about to run cases.
    Start {
        /// Cases that will run.
        #[serde(default)]
        cases: Vec<CaseInfo>,
        /// Runtime environment description; values are stringified.
        #[serde(default)]
        env: BTreeMap<String, Value>,
        /// Hardware the problem requires to be accepted.
        #[serde(default = "default_requires")]
        requires: Requires,
        /// Whether this machine has it.
        #[serde(default = "default_requires_met")]
        requires_met: bool,
    },
    /// A case began.
    CaseStart {
        /// Case id.
        id: String,
    },
    /// A case finished.
    CaseEnd {
        /// Case id.
        id: String,
        /// Outcome.
        status: CaseStatus,
        /// Duration in milliseconds.
        #[serde(default)]
        duration_ms: f64,
        /// Short failure reason.
        #[serde(default)]
        message: Option<String>,
        /// Longer failure text.
        #[serde(default)]
        detail: Option<String>,
        /// Captured prints.
        #[serde(default)]
        stdout: String,
        /// Metrics reported for the case.
        #[serde(default)]
        metrics: Vec<Metric>,
        /// Why the case was skipped.
        #[serde(default, deserialize_with = "crate::model::lenient")]
        skip: Option<Skip>,
    },
    /// The harness finished.
    End {
        /// Final status.
        status: Verdict,
        /// Number of passed cases.
        #[serde(default)]
        passed: u32,
        /// Number of cases.
        #[serde(default)]
        total: u32,
        /// Duration in milliseconds.
        #[serde(default)]
        duration_ms: f64,
        /// Run-level metrics.
        #[serde(default)]
        metrics: Vec<Metric>,
        /// Number of cases that were not run.
        #[serde(default)]
        skipped: Option<u32>,
        /// With `assets_missing`: the assets to download.
        #[serde(default, deserialize_with = "crate::model::lenient")]
        missing_assets: Option<Vec<MissingAsset>>,
        /// With `hardware_missing`: the hardware this machine lacks.
        #[serde(default, deserialize_with = "crate::model::lenient")]
        missing_hardware: Option<Requires>,
        /// Why the run as a whole failed (`import_error`, `harness_error`), or what was
        /// skipped and why (`assets_missing`, `hardware_missing`).
        #[serde(default)]
        message: Option<String>,
        /// Longer text for that failure: the traceback.
        #[serde(default)]
        detail: Option<String>,
    },
}

fn default_requires() -> Requires {
    Requires::Cpu
}

fn default_requires_met() -> bool {
    true
}

impl HarnessEvent {
    /// Converts a `case_end` event into a [`CaseResult`]; other events yield `None`.
    pub fn into_case_result(self) -> Option<CaseResult> {
        match self {
            HarnessEvent::CaseEnd {
                id,
                status,
                duration_ms,
                message,
                detail,
                stdout,
                metrics,
                skip,
            } => Some(CaseResult {
                id,
                status,
                duration_ms,
                message,
                detail,
                stdout,
                metrics,
                skip,
            }),
            _ => None,
        }
    }
}

/// One classified line of harness stdout.
#[derive(Debug, Clone, PartialEq)]
pub enum StdoutLine {
    /// A well-formed protocol event.
    Event(HarnessEvent),
    /// Anything else, to be forwarded as diagnostic output.
    Log(String),
}

/// Classifies one stdout line (without its trailing newline).
///
/// A line with the prefix but a malformed or unknown payload is treated as diagnostic
/// output rather than dropped, so protocol bugs stay visible.
pub fn parse_stdout_line(line: &str) -> StdoutLine {
    line.strip_prefix(PREFIX)
        .and_then(|payload| serde_json::from_str::<HarnessEvent>(payload).ok())
        .map_or_else(|| StdoutLine::Log(line.to_owned()), StdoutLine::Event)
}

/// Stringifies harness `env` values: strings as is, everything else as compact JSON.
pub fn stringify_env(env: BTreeMap<String, Value>) -> BTreeMap<String, String> {
    env.into_iter()
        .map(|(key, value)| match value {
            Value::String(text) => (key, text),
            other => (key, other.to_string()),
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_the_documented_events() {
        let start = r#"@@WS@@ {"event":"start","problem":"stable-softmax","framework":"torch","mode":"run","cases":[{"id":"c1","name":"Basic","public":true}],"env":{"python":"3.15","mps":true}}"#;
        match parse_stdout_line(start) {
            StdoutLine::Event(HarnessEvent::Start {
                cases,
                env,
                requires,
                requires_met,
            }) => {
                assert_eq!(cases, vec![CaseInfo::new("c1", "Basic", true)]);
                // A harness that predates hardware requirements: CPU, and met.
                assert_eq!((requires, requires_met), (Requires::Cpu, true));
                let env = stringify_env(env);
                assert_eq!(env["python"], "3.15");
                assert_eq!(env["mps"], "true");
            }
            other => panic!("unexpected {other:?}"),
        }

        let case_end = r#"@@WS@@ {"event":"case_end","id":"c1","status":"pass","duration_ms":12.3,"message":null,"detail":null,"stdout":"","metrics":[{"name":"model_tokens","value":37,"unit":"tokens","limit":37}]}"#;
        let StdoutLine::Event(event) = parse_stdout_line(case_end) else {
            panic!("not an event")
        };
        let result = event.into_case_result().expect("case result");
        assert_eq!(result.status, CaseStatus::Pass);
        assert_eq!(result.metrics[0].limit, Some(37.0));

        let end = r#"@@WS@@ {"event":"end","status":"wrong_answer","passed":4,"total":5,"duration_ms":210.0,"metrics":[]}"#;
        assert!(matches!(
            parse_stdout_line(end),
            StdoutLine::Event(HarnessEvent::End {
                status: Verdict::WrongAnswer,
                passed: 4,
                total: 5,
                message: None,
                detail: None,
                ..
            })
        ));

        let failed = r#"@@WS@@ {"event":"end","status":"import_error","passed":0,"total":2,"duration_ms":1.0,"metrics":[],"message":"the solution failed to import: SyntaxError","detail":"  File \"s.py\", line 1\n"}"#;
        match parse_stdout_line(failed) {
            StdoutLine::Event(HarnessEvent::End {
                status,
                message,
                detail,
                ..
            }) => {
                assert_eq!(status, Verdict::ImportError);
                assert_eq!(
                    message.as_deref(),
                    Some("the solution failed to import: SyntaxError")
                );
                assert_eq!(detail.as_deref(), Some("  File \"s.py\", line 1\n"));
            }
            other => panic!("unexpected {other:?}"),
        }
    }

    #[test]
    fn parses_skipped_cases_and_not_judged_endings() {
        let start = r#"@@WS@@ {"event":"start","problem":"cuda-scale","framework":"torch","mode":"submit","cases":[{"id":"cpu","name":"scales on the CPU","public":true},{"id":"hidden-2","name":"Hidden case 2","public":false,"requires":"cuda"}],"env":{"python":"3.15.2"},"requires":"cuda","requires_met":false,"assets":[]}"#;
        match parse_stdout_line(start) {
            StdoutLine::Event(HarnessEvent::Start {
                cases,
                requires,
                requires_met,
                ..
            }) => {
                assert_eq!(cases[0].requires, None);
                assert_eq!(cases[1].requires, Some(Requires::Cuda));
                assert_eq!((requires, requires_met), (Requires::Cuda, false));
            }
            other => panic!("unexpected {other:?}"),
        }

        let skipped = r#"@@WS@@ {"event":"case_end","id":"hidden-2","status":"skipped","duration_ms":0,"message":"skipped: needs an NVIDIA GPU (CUDA), which this machine does not have","detail":null,"stdout":"","metrics":[],"skip":{"kind":"hardware","requires":"cuda"}}"#;
        let StdoutLine::Event(event) = parse_stdout_line(skipped) else {
            panic!("not an event")
        };
        let result = event.into_case_result().expect("case result");
        assert_eq!(result.status, CaseStatus::Skipped);
        assert_eq!(
            result.skip,
            Some(Skip::Hardware {
                requires: Requires::Cuda
            })
        );

        for (payload, want) in [
            (
                r#"{"kind":"asset","asset":"granite-4.0-350m"}"#,
                Some(Skip::Asset {
                    asset: "granite-4.0-350m".into(),
                }),
            ),
            (r#"{"kind":"aborted"}"#, Some(Skip::Aborted)),
            // Unknown kinds and extra fields never break the event.
            (r#"{"kind":"license","spdx":"x"}"#, None),
            (r#"{"kind":"aborted","after":"c1"}"#, Some(Skip::Aborted)),
        ] {
            let line = format!(
                r#"@@WS@@ {{"event":"case_end","id":"c","status":"skipped","new_field":[1],"skip":{payload}}}"#
            );
            let StdoutLine::Event(event) = parse_stdout_line(&line) else {
                panic!("not an event: {line}")
            };
            assert_eq!(event.into_case_result().expect("result").skip, want);
        }

        let hardware = r#"@@WS@@ {"event":"end","status":"hardware_missing","passed":2,"total":3,"skipped":1,"duration_ms":210.0,"metrics":[],"message":"2 of 2 cases that could run passed","missing_hardware":"cuda"}"#;
        match parse_stdout_line(hardware) {
            StdoutLine::Event(HarnessEvent::End {
                status,
                skipped,
                missing_hardware,
                missing_assets,
                message,
                ..
            }) => {
                assert_eq!(status, Verdict::HardwareMissing);
                assert_eq!(skipped, Some(1));
                assert_eq!(missing_hardware, Some(Requires::Cuda));
                assert_eq!(missing_assets, None);
                assert!(message.is_some());
            }
            other => panic!("unexpected {other:?}"),
        }

        let assets = r#"@@WS@@ {"event":"end","status":"assets_missing","passed":2,"total":4,"skipped":2,"duration_ms":3.0,"metrics":[{"name":"throughput","value":41.0,"unit":"tok/s","limit":20,"higher_is_better":true}],"message":"m","missing_assets":[{"name":"tiny","repo":"o/tiny","size_mb":0.01,"fetch_command":"ws assets fetch tiny"}]}"#;
        match parse_stdout_line(assets) {
            StdoutLine::Event(HarnessEvent::End {
                status,
                missing_assets,
                metrics,
                ..
            }) => {
                assert_eq!(status, Verdict::AssetsMissing);
                assert_eq!(
                    missing_assets,
                    Some(vec![MissingAsset {
                        name: "tiny".into(),
                        repo: "o/tiny".into(),
                        size_mb: 0.01,
                    }])
                );
                assert!(metrics[0].higher_is_better);
            }
            other => panic!("unexpected {other:?}"),
        }
    }

    #[test]
    fn anything_else_is_log_output() {
        for line in [
            "hello",
            "@@WS@@ not json",
            r#"@@WS@@ {"event":"mystery"}"#,
            r#" @@WS@@ {"event":"case_start","id":"c1"}"#,
            r#"@@WS@@{"event":"case_start","id":"c1"}"#,
            "",
        ] {
            assert_eq!(parse_stdout_line(line), StdoutLine::Log(line.to_owned()));
        }
    }
}
