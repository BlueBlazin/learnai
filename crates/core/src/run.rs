//! Run orchestration: launch the harness on a target, translate its output into
//! [`RunEvent`]s, enforce wall-clock limits, support cancel, and store submissions.
//!
//! Guarantee: every run emits exactly one `end` event, and it is the last event,
//! whatever happens (launch failure, harness crash, cancel, timeout).

use std::collections::HashMap;
use std::sync::mpsc::{self, RecvTimeoutError, Sender};
use std::sync::{Arc, Mutex, PoisonError};
use std::thread;
use std::time::{Duration, Instant};

use crate::harness::HarnessJob;
use crate::model::{
    CaseInfo, CaseStatus, LogStream, Metric, MissingAsset, Requires, RunEvent, RunMode, StoredCase,
    Verdict,
};
use crate::protocol::{HarnessEvent, StdoutLine, parse_stdout_line, stringify_env};
use crate::sink::SharedSink;
use crate::store::{NewSubmission, Store};
use crate::stream::{Message, spawn_reader};
use crate::target::Target;

/// Wall-clock limits core enforces on a run, as a backstop to the harness's own
/// per-case limits.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RunLimits {
    /// Maximum time from launch until the harness reports `start`. Generous, because the
    /// first run may have to sync the Python environment.
    pub startup: Duration,
    /// Slack added to the case budget. After `start`, the run may take
    /// `slack + cases x time_limit_s` before it is killed.
    pub slack: Duration,
    /// How long the process may linger after reporting `end` (or closing its output)
    /// before it is killed.
    pub exit_grace: Duration,
}

impl Default for RunLimits {
    fn default() -> Self {
        Self {
            startup: Duration::from_secs(600),
            slack: Duration::from_secs(30),
            exit_grace: Duration::from_secs(5),
        }
    }
}

impl RunLimits {
    /// The budget for the cases themselves, once the harness has started.
    fn case_budget(&self, cases: usize, time_limit_s: f64) -> Duration {
        let per_case = Duration::try_from_secs_f64(time_limit_s).unwrap_or(Duration::MAX);
        self.slack
            .saturating_add(per_case.saturating_mul(u32::try_from(cases).unwrap_or(u32::MAX)))
    }
}

/// Everything a run needs, resolved up front so the run thread cannot fail on lookups.
pub(crate) struct RunPlan {
    pub problem_id: String,
    pub time_limit_s: f64,
    pub job: HarnessJob,
    /// Snapshot of the solution, stored with the submission.
    pub code: String,
    pub target: Arc<dyn Target>,
    pub limits: RunLimits,
    pub store: Arc<Store>,
    pub sink: SharedSink<RunEvent>,
}

/// Active runs, by run id.
#[derive(Default)]
pub(crate) struct RunRegistry {
    runs: Mutex<HashMap<String, Sender<Message>>>,
}

impl RunRegistry {
    /// Starts `plan` on its own thread and returns the run id.
    pub fn start(self: &Arc<Self>, plan: RunPlan) -> std::io::Result<String> {
        let run_id = uuid::Uuid::new_v4().to_string();
        let (tx, rx) = mpsc::channel();
        self.lock().insert(run_id.clone(), tx.clone());

        let registry = Arc::clone(self);
        let id = run_id.clone();
        let spawned = thread::Builder::new()
            .name(format!("run-{run_id}"))
            .spawn(move || {
                Run::new(plan).execute(&tx, &rx);
                registry.lock().remove(&id);
            });
        if let Err(e) = spawned {
            self.lock().remove(&run_id);
            return Err(e);
        }
        Ok(run_id)
    }

    /// Requests cancellation. A run that already finished (or never existed) is a no-op,
    /// so a cancel that races with completion is not an error.
    pub fn cancel(&self, run_id: &str) {
        if let Some(tx) = self.lock().get(run_id) {
            // The receiver is gone only if the run just finished.
            let _ = tx.send(Message::Cancel);
        }
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, HashMap<String, Sender<Message>>> {
        self.runs.lock().unwrap_or_else(PoisonError::into_inner)
    }
}

/// Why the event loop stopped before the harness reported `end`.
enum Interruption {
    Canceled,
    TimedOut { limit: Duration, started: bool },
    Exited,
}

struct Run {
    plan: RunPlan,
    began: Instant,
    /// Cases announced by `start`, for naming stored results.
    announced: Vec<CaseInfo>,
    results: Vec<StoredCase>,
    ended: bool,
}

/// How a run finished, as far as the `end` event is concerned.
struct Outcome {
    verdict: Verdict,
    metrics: Vec<Metric>,
    /// The harness's own `(passed, total, duration_ms)`; without it the numbers are
    /// derived from what was observed.
    reported: Option<(u32, u32, f64)>,
    /// The harness's own count of skipped cases; without it the skipped results are
    /// counted.
    skipped: Option<u32>,
    missing_assets: Option<Vec<MissingAsset>>,
    missing_hardware: Option<Requires>,
    /// False when the harness process never started.
    launched: bool,
    message: Option<String>,
    detail: Option<String>,
}

impl Outcome {
    /// An outcome decided by core rather than reported by the harness.
    fn from_core(verdict: Verdict, launched: bool, message: Option<String>) -> Self {
        Self {
            verdict,
            metrics: Vec::new(),
            reported: None,
            skipped: None,
            missing_assets: None,
            missing_hardware: None,
            launched,
            message,
            detail: None,
        }
    }
}

/// What to tell the user when the judge process could not be started.
fn launch_failure(error: &std::io::Error) -> String {
    if error.kind() == std::io::ErrorKind::NotFound {
        "uv was not found, so the judge cannot run. Install uv \
         (https://docs.astral.sh/uv/), then reopen Whetstone."
            .to_owned()
    } else {
        format!("The judge could not be started: {error}")
    }
}

impl Run {
    fn new(plan: RunPlan) -> Self {
        Self {
            plan,
            began: Instant::now(),
            announced: Vec::new(),
            results: Vec::new(),
            ended: false,
        }
    }

    fn emit(&self, event: RunEvent) {
        if !self.ended {
            self.plan.sink.emit(event);
        }
    }

    fn log(&self, stream: LogStream, text: impl Into<String>) {
        self.emit(RunEvent::Log {
            stream,
            text: text.into(),
        });
    }

    fn execute(mut self, tx: &Sender<Message>, rx: &mpsc::Receiver<Message>) {
        let target = Arc::clone(&self.plan.target);
        self.emit(RunEvent::Preparing {
            message: format!(
                "Preparing `{}` on the {} target",
                self.plan.problem_id,
                target.name()
            ),
        });

        let launched = {
            let sink = Arc::clone(&self.plan.sink);
            target.launch(&self.plan.job, &move |message| {
                sink.emit(RunEvent::Preparing { message })
            })
        };
        let mut process = match launched {
            Ok(process) => process,
            Err(e) => {
                self.log(LogStream::Stderr, format!("Could not start the judge: {e}"));
                self.end(Outcome::from_core(
                    Verdict::HarnessError,
                    false,
                    Some(launch_failure(&e)),
                ));
                return;
            }
        };

        let mut open = 0;
        if let Some(stdout) = process.take_stdout() {
            open += spawn_reader(stdout, tx.clone(), Message::Stdout);
        }
        if let Some(stderr) = process.take_stderr() {
            open += spawn_reader(stderr, tx.clone(), Message::Stderr);
        }

        let interruption = self.pump(rx, open);
        let grace = self.plan.limits.exit_grace;
        match interruption {
            None => {
                process.finish(grace);
            }
            Some(Interruption::Canceled) => {
                process.finish(Duration::ZERO);
                self.end(Outcome::from_core(Verdict::Canceled, false, None));
            }
            Some(Interruption::TimedOut { limit, started }) => {
                process.finish(Duration::ZERO);
                let phase = if started {
                    "The run"
                } else {
                    "Starting the judge"
                };
                let message = format!(
                    "{phase} exceeded the overall limit of {:.1} s and was killed.",
                    limit.as_secs_f64()
                );
                self.log(LogStream::Stderr, message.clone());
                self.end(Outcome::from_core(Verdict::TimeLimit, true, Some(message)));
            }
            Some(Interruption::Exited) => {
                let how = match process.finish(grace) {
                    Some(code) => format!("exited with code {code}"),
                    None => "was killed by a signal".to_owned(),
                };
                let message = format!("The judge {how} without reporting a result.");
                self.log(LogStream::Stderr, message.clone());
                self.end(Outcome::from_core(
                    Verdict::HarnessError,
                    true,
                    Some(message),
                ));
            }
        }
    }

    /// Consumes process output until the harness reports `end` and closes its pipes, or
    /// the run is interrupted. Returns `None` when `end` was reported (and forwarded).
    fn pump(&mut self, rx: &mpsc::Receiver<Message>, mut open: usize) -> Option<Interruption> {
        let mut limit = self.plan.limits.startup;
        let mut deadline = Instant::now() + limit;
        let mut started = false;

        while open > 0 {
            let message = match rx.recv_timeout(deadline.saturating_duration_since(Instant::now()))
            {
                Ok(message) => message,
                Err(RecvTimeoutError::Timeout) if self.ended => return None,
                Err(RecvTimeoutError::Timeout) => {
                    return Some(Interruption::TimedOut { limit, started });
                }
                Err(RecvTimeoutError::Disconnected) => break,
            };
            match message {
                Message::Closed => open -= 1,
                Message::Cancel if self.ended => {}
                Message::Cancel => return Some(Interruption::Canceled),
                Message::Stderr(line) => self.log(LogStream::Stderr, line),
                Message::Stdout(line) => match parse_stdout_line(&line) {
                    StdoutLine::Log(text) => self.log(LogStream::Stdout, text),
                    StdoutLine::Event(HarnessEvent::Start {
                        cases,
                        env,
                        requires,
                        requires_met,
                    }) => {
                        self.announced.clone_from(&cases);
                        if !started {
                            started = true;
                            limit = self
                                .plan
                                .limits
                                .case_budget(cases.len(), self.plan.time_limit_s);
                            deadline = Instant::now().checked_add(limit).unwrap_or(deadline);
                        }
                        self.emit(RunEvent::Start {
                            cases,
                            env: stringify_env(env),
                            requires,
                            requires_met,
                        });
                    }
                    StdoutLine::Event(HarnessEvent::CaseStart { id }) => {
                        self.emit(RunEvent::CaseStart { id });
                    }
                    StdoutLine::Event(HarnessEvent::End {
                        status,
                        passed,
                        total,
                        duration_ms,
                        metrics,
                        skipped,
                        missing_assets,
                        missing_hardware,
                        message,
                        detail,
                    }) => {
                        if !self.ended {
                            self.end(Outcome {
                                verdict: status,
                                metrics,
                                reported: Some((passed, total, duration_ms)),
                                skipped,
                                missing_assets,
                                missing_hardware,
                                launched: true,
                                message,
                                detail,
                            });
                            // Give the process a moment to exit by itself, then stop waiting.
                            deadline = Instant::now() + self.plan.limits.exit_grace;
                        }
                    }
                    StdoutLine::Event(event) => {
                        if let Some(result) = event.into_case_result() {
                            // A skipped case arrives with no `case_start`; either way
                            // the name comes from the announcement.
                            if !self.ended {
                                let info = self.announced.iter().find(|c| c.id == result.id);
                                self.results.push(StoredCase::new(result.clone(), info));
                            }
                            self.emit(RunEvent::CaseEnd { result });
                        }
                    }
                },
            }
        }
        if self.ended {
            None
        } else {
            Some(Interruption::Exited)
        }
    }

    /// Stores the submission if appropriate and emits the single `end` event.
    fn end(&mut self, outcome: Outcome) {
        if self.ended {
            return;
        }
        let Outcome {
            verdict,
            metrics,
            reported,
            skipped,
            missing_assets,
            missing_hardware,
            launched,
            message,
            detail,
        } = outcome;
        let skipped = skipped.unwrap_or_else(|| {
            let count = self
                .results
                .iter()
                .filter(|r| r.result.status == CaseStatus::Skipped)
                .count();
            u32::try_from(count).unwrap_or(u32::MAX)
        });
        let (passed, total, duration_ms) = reported.unwrap_or_else(|| {
            let passed = self
                .results
                .iter()
                .filter(|r| r.result.status == CaseStatus::Pass)
                .count();
            let total = self.announced.len().max(self.results.len());
            (
                u32::try_from(passed).unwrap_or(u32::MAX),
                u32::try_from(total).unwrap_or(u32::MAX),
                self.began.elapsed().as_secs_f64() * 1000.0,
            )
        });

        // Canceled runs and runs whose judge never started are not attempts.
        let store =
            self.plan.job.mode == RunMode::Submit && launched && verdict != Verdict::Canceled;
        let submission_id = if store {
            let stored = self.plan.store.insert_submission(&NewSubmission {
                problem_id: &self.plan.problem_id,
                framework: self.plan.job.framework,
                mode: self.plan.job.mode,
                verdict,
                passed,
                total,
                skipped,
                duration_ms,
                target: self.plan.target.name(),
                created_at: &crate::now_iso8601(),
                code: &self.plan.code,
                cases: &self.results,
                metrics: &metrics,
            });
            match stored {
                Ok(id) => Some(id),
                Err(e) => {
                    self.log(
                        LogStream::Stderr,
                        format!("Could not store the submission: {e}"),
                    );
                    None
                }
            }
        } else {
            None
        };

        tracing::info!(
            problem = %self.plan.problem_id,
            framework = %self.plan.job.framework,
            mode = %self.plan.job.mode,
            %verdict,
            passed,
            total,
            skipped,
            submission = ?submission_id,
            "run finished"
        );
        self.emit(RunEvent::End {
            verdict,
            passed,
            total,
            duration_ms,
            metrics,
            submission_id,
            skipped,
            missing_assets,
            missing_hardware,
            message,
            detail,
        });
        self.ended = true;
    }
}
