//! The harness event protocol (`docs/ARCHITECTURE.md` section 3.6).
//!
//! The harness writes protocol lines to stdout, each one [`PREFIX`] followed by a
//! single-line JSON object. Every other stdout line is diagnostic output.

use std::collections::BTreeMap;

use serde::Deserialize;
use serde_json::Value;

use crate::model::{CaseInfo, CaseResult, CaseStatus, Metric, Verdict};

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
    },
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
            } => Some(CaseResult {
                id,
                status,
                duration_ms,
                message,
                detail,
                stdout,
                metrics,
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
            StdoutLine::Event(HarnessEvent::Start { cases, env }) => {
                assert_eq!(
                    cases,
                    vec![CaseInfo {
                        id: "c1".into(),
                        name: "Basic".into(),
                        public: true
                    }]
                );
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
                ..
            })
        ));
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
