//! The runtime documentation helper (`docs/ARCHITECTURE.md` section 4.4): a long-lived
//! Python process (`python -m learnai.editor.docs`) that answers "what is the signature
//! and docstring of this dotted name" for names the language server has no docstring
//! for. It speaks line-delimited JSON; requests and responses are matched by `id`.
//!
//! The helper is started on the first lookup. Nothing depends on it: a helper that
//! cannot start, crashes, or stops answering makes lookups return `unavailable`, is
//! killed, and is started again by a later lookup after a pause.

use std::io::Write;
use std::process::{ChildStdin, Stdio};
use std::sync::mpsc::{self, Receiver};
use std::sync::{Mutex, PoisonError};
use std::thread;
use std::time::{Duration, Instant};

use serde_json::{Value, json};

use crate::harness::CommandSpec;
use crate::model::EditorDocs;
use crate::stream::for_each_line;
use crate::target::{LocalProcess, Process};

/// The `reason` of an answer the helper could not give.
const UNAVAILABLE: &str = "unavailable";

struct Running {
    stdin: ChildStdin,
    /// The helper's stdout, one message per line; disconnected once it exits.
    lines: Receiver<String>,
}

#[derive(Default)]
struct State {
    running: Option<Running>,
    next_id: u64,
    /// Set when the helper failed: no new one is started before this time.
    retry_at: Option<Instant>,
}

/// The app's one docs helper process.
pub(crate) struct DocsHelper {
    spec: CommandSpec,
    timeout: Duration,
    retry_pause: Duration,
    /// The helper process; killed (with its process group) when taken out and dropped.
    /// Apart from `state`, so that `stop` can kill a helper a lookup is waiting on.
    process: Mutex<Option<LocalProcess>>,
    state: Mutex<State>,
}

impl DocsHelper {
    /// Creates the accessor. Nothing is started until the first lookup.
    pub fn new(spec: CommandSpec, timeout: Duration, retry_pause: Duration) -> Self {
        Self {
            spec,
            timeout,
            retry_pause,
            process: Mutex::new(None),
            state: Mutex::new(State::default()),
        }
    }

    /// Looks up `expr` (or its attribute `member`) as written in a file whose text is
    /// `source`. Never fails: when the helper cannot answer, the result is not found
    /// with reason `unavailable`.
    pub fn lookup(&self, source: &str, expr: &str, member: Option<&str>) -> EditorDocs {
        let mut state = self.state.lock().unwrap_or_else(PoisonError::into_inner);
        if state.running.is_none() {
            if state.retry_at.is_some_and(|at| Instant::now() < at) {
                return EditorDocs::not_found(UNAVAILABLE);
            }
            match self.spawn() {
                Ok((process, running)) => {
                    self.set_process(Some(process));
                    state.running = Some(running);
                }
                Err(error) => {
                    tracing::warn!("the docs helper could not be started: {error}");
                    state.retry_at = Some(Instant::now() + self.retry_pause);
                    return EditorDocs::not_found(UNAVAILABLE);
                }
            }
        }
        state.next_id += 1;
        let id = state.next_id;
        let request = json!({ "id": id, "source": source, "expr": expr, "member": member });
        let answer = state
            .running
            .as_mut()
            .ok_or_else(|| "it is not running".to_owned())
            .and_then(|running| ask(running, id, &request, self.timeout));
        match answer {
            Ok(docs) => {
                state.retry_at = None;
                docs
            }
            Err(problem) => {
                tracing::warn!("the docs helper was stopped: {problem}");
                self.set_process(None);
                state.running = None;
                state.retry_at = Some(Instant::now() + self.retry_pause);
                EditorDocs::not_found(UNAVAILABLE)
            }
        }
    }

    /// Kills the helper, if it is running. A later lookup starts a new one.
    pub fn stop(&self) {
        // Kill first: a lookup that is waiting for an answer then fails and lets go of
        // the state.
        self.set_process(None);
        let mut state = self.state.lock().unwrap_or_else(PoisonError::into_inner);
        state.running = None;
        state.retry_at = None;
    }

    fn set_process(&self, process: Option<LocalProcess>) {
        *self.process.lock().unwrap_or_else(PoisonError::into_inner) = process;
    }

    fn spawn(&self) -> std::io::Result<(LocalProcess, Running)> {
        let mut process =
            LocalProcess::spawn_io(&self.spec, Stdio::piped(), Stdio::piped(), Stdio::piped())?;
        let stdin = process
            .take_stdin()
            .ok_or_else(|| std::io::Error::other("the helper has no standard input"))?;
        let (tx, lines) = mpsc::channel();
        if let Some(stdout) = process.take_stdout() {
            thread::Builder::new()
                .name("docs-helper-reader".to_owned())
                .spawn(move || for_each_line(stdout, |line| tx.send(line).is_ok()))?;
        }
        if let Some(stderr) = process.take_stderr() {
            // Libraries the helper imports print warnings; keep the pipe drained.
            thread::Builder::new()
                .name("docs-helper-stderr".to_owned())
                .spawn(move || {
                    for_each_line(stderr, |line| {
                        tracing::debug!(target: "learnai_core::docs::helper", "{line}");
                        true
                    });
                })?;
        }
        tracing::info!("docs helper started");
        Ok((process, Running { stdin, lines }))
    }
}

/// Sends one request and waits for the response with the same id.
fn ask(
    running: &mut Running,
    id: u64,
    request: &Value,
    timeout: Duration,
) -> Result<EditorDocs, String> {
    let mut line = request.to_string();
    line.push('\n');
    running
        .stdin
        .write_all(line.as_bytes())
        .and_then(|()| running.stdin.flush())
        .map_err(|error| format!("it stopped reading requests ({error})"))?;

    let deadline = Instant::now() + timeout;
    loop {
        let line = running
            .lines
            .recv_timeout(deadline.saturating_duration_since(Instant::now()))
            .map_err(|error| match error {
                mpsc::RecvTimeoutError::Timeout => {
                    format!("it did not answer within {:.0} s", timeout.as_secs_f64())
                }
                mpsc::RecvTimeoutError::Disconnected => "it exited".to_owned(),
            })?;
        let Ok(value) = serde_json::from_str::<Value>(&line) else {
            return Err("it printed something that is not a response".to_owned());
        };
        if value.get("id").and_then(Value::as_u64) != Some(id) {
            // The answer to a request that was given up on earlier.
            continue;
        }
        return serde_json::from_value(value)
            .map_err(|error| format!("its response has the wrong shape ({error})"));
    }
}
