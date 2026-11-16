//! The language server behind editor intelligence (`docs/ARCHITECTURE.md` section 4.4).
//!
//! One server (ty) runs for the whole app, rooted at the workspace directory. The
//! editor's client speaks bare JSON-RPC strings; this module adds and removes the
//! `Content-Length` framing the server's stdio needs, rewrites the client's
//! `initialize` request so that core owns the server's root and configuration, and
//! keeps the process alive: a server that dies is started again a bounded number of
//! times, with a growing delay.
//!
//! Guarantees: every session's stream ends with exactly one terminal state
//! (`notInstalled`, `failed`, or `stopped`), and when that state has been emitted the
//! server's whole process group is gone.

use std::borrow::Cow;
use std::collections::VecDeque;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::process::{ChildStdin, Stdio};
use std::sync::mpsc::{self, Receiver, RecvTimeoutError, Sender};
use std::sync::{Arc, Mutex, MutexGuard, PoisonError};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use serde_json::{Value, json};

use crate::env::{capture, parse_tool_version};
use crate::error::{CoreError, Result};
use crate::harness::CommandSpec;
use crate::model::{EditorEvent, EditorState, EditorStopReason};
use crate::sink::SharedSink;
use crate::stream::for_each_line;
use crate::target::{LocalProcess, Process};
use crate::uv::venv_program;

/// The setting that turns editor intelligence on and off (default on).
pub const ENABLED_SETTING: &str = "editor_intelligence";

/// The language server's executable in the virtual environment, and its name.
const SERVER: &str = "ty";

/// The largest message body accepted from the server.
const MAX_FRAME_BYTES: usize = 64 * 1024 * 1024;

/// The largest header block accepted from the server.
const MAX_HEADER_BYTES: usize = 8 * 1024;

/// How long a server that closed its output may take to exit before it is killed.
const EXIT_GRACE: Duration = Duration::from_millis(200);

/// How often a running server is checked for having exited. Its output closing is
/// noticed at once; this catches a server whose output is held open by something it
/// started.
const EXIT_POLL: Duration = Duration::from_millis(250);

/// How many of the server's last stderr lines are kept for failure messages.
const STDERR_TAIL: usize = 20;

/// The diagnostics the editor shows: everything is off except the rules that catch a
/// mistyped or misused name. Measured against the catalog: no reports in any
/// reference solution. ty's defaults are too noisy for code written against
/// partially typed tensor libraries.
const RULES: &[&str] = &[
    "unresolved-reference",
    "import-not-found",
    "missing-attribute",
    "missing-argument",
    "too-many-positional-arguments",
    "unknown-argument",
    "parameter-already-assigned",
];

/// How to run the editor's helper processes. The defaults are what the app uses; tests
/// substitute small scripts.
#[derive(Debug, Clone)]
pub struct EditorOptions {
    /// The virtual environment holding the language server and the Python packages it
    /// resolves imports against; `None` uses `<root>/python/.venv`.
    pub venv: Option<PathBuf>,
    /// Overrides the language server command; `None` runs `<venv>/bin/ty server`. A
    /// command given here counts as installed.
    pub server: Option<CommandSpec>,
    /// Overrides the runtime docs helper command; `None` runs
    /// `<venv>/bin/python -I -m learnai.editor.docs`.
    pub docs: Option<CommandSpec>,
    /// When a language server that died is started again.
    pub restart: RestartPolicy,
    /// How long one docs lookup may take before the helper is given up on.
    pub docs_timeout: Duration,
    /// How long a docs helper that failed is left alone before a lookup starts a new
    /// one.
    pub docs_retry_pause: Duration,
}

impl Default for EditorOptions {
    fn default() -> Self {
        Self {
            venv: None,
            server: None,
            docs: None,
            restart: RestartPolicy::default(),
            docs_timeout: Duration::from_secs(20),
            docs_retry_pause: Duration::from_secs(10),
        }
    }
}

/// When a language server that died is started again.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RestartPolicy {
    /// How many restarts in a row are tried before the session fails.
    pub max_restarts: u32,
    /// The wait before the first restart; it doubles with each further one.
    pub base_delay: Duration,
    /// The longest wait.
    pub max_delay: Duration,
    /// A server that ran at least this long was healthy: the count starts over.
    pub stable_after: Duration,
}

impl Default for RestartPolicy {
    fn default() -> Self {
        Self {
            max_restarts: 4,
            base_delay: Duration::from_millis(500),
            max_delay: Duration::from_secs(8),
            stable_after: Duration::from_secs(60),
        }
    }
}

impl RestartPolicy {
    /// The wait before restart number `attempt` (from 1).
    fn delay(&self, attempt: u32) -> Duration {
        let doublings = attempt.saturating_sub(1).min(16);
        self.base_delay
            .saturating_mul(1 << doublings)
            .min(self.max_delay)
    }
}

/// Counts consecutive restarts against a [`RestartPolicy`].
#[derive(Debug)]
struct Restarts {
    policy: RestartPolicy,
    consecutive: u32,
}

impl Restarts {
    fn new(policy: RestartPolicy) -> Self {
        Self {
            policy,
            consecutive: 0,
        }
    }

    /// Records that the server died after running for `uptime`. Returns the number of
    /// the restart to make and the wait before it, or `None` when the limit is spent.
    fn next(&mut self, uptime: Duration) -> Option<(u32, Duration)> {
        if uptime >= self.policy.stable_after {
            self.consecutive = 0;
        }
        if self.consecutive >= self.policy.max_restarts {
            return None;
        }
        self.consecutive += 1;
        Some((self.consecutive, self.policy.delay(self.consecutive)))
    }
}

/// Why the server's output stopped being a stream of framed messages.
#[derive(Debug, PartialEq, Eq, thiserror::Error)]
pub(crate) enum FrameError {
    /// A header line has no colon.
    #[error("a header is not `Name: value`: `{0}`")]
    BadHeader(String),
    /// The header block has no `Content-Length`.
    #[error("a message has no Content-Length header")]
    MissingLength,
    /// `Content-Length` is not a non-negative integer.
    #[error("Content-Length is not a number: `{0}`")]
    BadLength(String),
    /// The body is larger than core accepts.
    #[error("a message of {0} bytes is over the limit of {max} bytes", max = MAX_FRAME_BYTES)]
    TooLarge(usize),
    /// The header block does not end.
    #[error("the headers of a message are longer than {max} bytes", max = MAX_HEADER_BYTES)]
    HeadersTooLong,
}

/// Wraps one JSON-RPC message in the base protocol's framing.
pub(crate) fn frame(message: &str) -> Vec<u8> {
    let mut framed = format!("Content-Length: {}\r\n\r\n", message.len()).into_bytes();
    framed.extend_from_slice(message.as_bytes());
    framed
}

/// Splits a byte stream in the base protocol's framing into messages. Bytes arrive in
/// whatever pieces the pipe delivers: half a header, or several messages at once.
#[derive(Debug, Default)]
pub(crate) struct FrameDecoder {
    buffer: Vec<u8>,
    /// The body length of the message whose headers have been consumed.
    body: Option<usize>,
}

impl FrameDecoder {
    /// Adds bytes read from the server.
    pub fn push(&mut self, bytes: &[u8]) {
        self.buffer.extend_from_slice(bytes);
    }

    /// The next whole message, `None` when more bytes are needed. After an error the
    /// stream cannot be resynchronized and the decoder must be dropped.
    pub fn next_message(&mut self) -> Result<Option<String>, FrameError> {
        let length = match self.body {
            Some(length) => length,
            None => {
                let Some(end) = find(&self.buffer, b"\r\n\r\n") else {
                    if self.buffer.len() > MAX_HEADER_BYTES {
                        return Err(FrameError::HeadersTooLong);
                    }
                    return Ok(None);
                };
                if end > MAX_HEADER_BYTES {
                    return Err(FrameError::HeadersTooLong);
                }
                let length = content_length(&self.buffer[..end])?;
                self.buffer.drain(..end + 4);
                self.body = Some(length);
                length
            }
        };
        if self.buffer.len() < length {
            return Ok(None);
        }
        let body: Vec<u8> = self.buffer.drain(..length).collect();
        self.body = None;
        Ok(Some(match String::from_utf8(body) {
            Ok(text) => text,
            Err(error) => String::from_utf8_lossy(error.as_bytes()).into_owned(),
        }))
    }
}

fn find(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    haystack
        .windows(needle.len())
        .position(|window| window == needle)
}

/// Reads `Content-Length` out of a header block (without its terminating blank line).
/// Other headers (`Content-Type`) are allowed and ignored.
fn content_length(headers: &[u8]) -> Result<usize, FrameError> {
    let text = String::from_utf8_lossy(headers);
    let mut length = None;
    for line in text.split("\r\n") {
        let Some((name, value)) = line.split_once(':') else {
            return Err(FrameError::BadHeader(line.chars().take(80).collect()));
        };
        if name.trim().eq_ignore_ascii_case("content-length") {
            let value = value.trim();
            let parsed: usize = value
                .parse()
                .map_err(|_| FrameError::BadLength(value.chars().take(80).collect()))?;
            if parsed > MAX_FRAME_BYTES {
                return Err(FrameError::TooLarge(parsed));
            }
            length = Some(parsed);
        }
    }
    length.ok_or(FrameError::MissingLength)
}

/// The `file://` URI of an absolute path, percent-encoded as RFC 3986 asks.
pub(crate) fn file_uri(path: &Path) -> String {
    let text = path.to_string_lossy();
    let mut uri = String::from("file://");
    // A Windows path (`C:\dir`) becomes `/C:/dir`.
    if !text.starts_with('/') {
        uri.push('/');
    }
    for &byte in text.as_bytes() {
        match byte {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'.' | b'_' | b'~' | b'/' => {
                uri.push(char::from(byte));
            }
            b'\\' if cfg!(windows) => uri.push('/'),
            b':' if cfg!(windows) => uri.push(':'),
            other => uri.push_str(&format!("%{other:02X}")),
        }
    }
    uri
}

/// What core puts into every `initialize` request.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ServerContext {
    /// The workspace directory as a `file://` URI.
    pub root_uri: String,
    /// The virtual environment imports are resolved against.
    pub venv: PathBuf,
}

impl ServerContext {
    /// The server settings core owns, sent as `initializationOptions`.
    ///
    /// ty reads its settings from the initialization options when the client does not
    /// support `workspace/configuration`. `configuration` is ty's inline configuration
    /// (the content of a `ty.toml`); it takes precedence over any configuration file,
    /// so nothing has to be written into the user's workspace.
    ///
    /// Auto-import completions are off: they offer every name in the environment
    /// (`mlx_lm` internals in a torch solution) and only work in a client that applies
    /// the import edit that comes with the item.
    fn initialization_options(&self) -> Value {
        let mut rules = serde_json::Map::new();
        rules.insert("all".to_owned(), json!("ignore"));
        for rule in RULES {
            rules.insert((*rule).to_owned(), json!("error"));
        }
        json!({
            "diagnosticMode": "openFilesOnly",
            "completions": { "autoImport": false },
            "configuration": {
                "environment": { "python": self.venv.to_string_lossy() },
                "rules": rules,
            },
        })
    }

    /// Checks one message from the editor's client and returns what to send to the
    /// server. Only `initialize` is changed:
    ///
    /// - `processId` is this process, so the server can notice the app dying.
    /// - `rootUri` and `workspaceFolders` are the workspace directory and `rootPath` is
    ///   removed, whatever the client sent: the server must index the learner's
    ///   solutions and nothing else (never the reference solutions under `problems/`).
    /// - `initializationOptions` are replaced by the settings core owns.
    /// - `capabilities.textDocument.diagnostic` is removed. The CodeMirror client
    ///   advertises pull diagnostics but never pulls; without the capability the
    ///   server pushes `textDocument/publishDiagnostics`.
    /// - `capabilities.workspace.configuration` is removed, so the server takes its
    ///   settings from the initialization options instead of asking a client that
    ///   answers every request with "Method not implemented".
    pub fn prepare<'a>(&self, message: &'a str) -> Result<Cow<'a, str>> {
        let mut value: Value = serde_json::from_str(message).map_err(|error| {
            CoreError::InvalidArgument(format!("the message is not JSON: {error}"))
        })?;
        let Value::Object(object) = &mut value else {
            return Err(CoreError::InvalidArgument(
                "the message is not a JSON-RPC object".to_owned(),
            ));
        };
        if object.get("method").and_then(Value::as_str) != Some("initialize") {
            return Ok(Cow::Borrowed(message));
        }
        let params = object.entry("params").or_insert_with(|| json!({}));
        let Value::Object(params) = params else {
            return Err(CoreError::InvalidArgument(
                "the parameters of `initialize` are not an object".to_owned(),
            ));
        };
        params.insert("processId".to_owned(), json!(std::process::id()));
        params.insert("rootUri".to_owned(), json!(self.root_uri));
        params.remove("rootPath");
        params.insert(
            "workspaceFolders".to_owned(),
            json!([{ "uri": self.root_uri, "name": "workspace" }]),
        );
        params.insert(
            "initializationOptions".to_owned(),
            self.initialization_options(),
        );
        if let Some(Value::Object(capabilities)) = params.get_mut("capabilities") {
            if let Some(Value::Object(text_document)) = capabilities.get_mut("textDocument") {
                text_document.remove("diagnostic");
            }
            if let Some(Value::Object(workspace)) = capabilities.get_mut("workspace") {
                workspace.remove("configuration");
            }
        }
        Ok(Cow::Owned(value.to_string()))
    }
}

/// What the supervisor thread is told.
enum Control {
    /// End the session.
    Stop(EditorStopReason),
    /// The server's output ended (it exited) or stopped making sense (`problem`).
    Exited {
        generation: u32,
        problem: Option<String>,
    },
}

/// The server's stdin; `None` while no server is running (between restarts).
type Writer = Arc<Mutex<Option<ChildStdin>>>;

struct Session {
    id: String,
    control: Sender<Control>,
    writer: Writer,
    context: Arc<ServerContext>,
    thread: Option<JoinHandle<()>>,
}

#[derive(Default)]
struct Shared {
    session: Mutex<Option<Session>>,
    /// The latest state of the latest session.
    state: Mutex<Option<EditorState>>,
}

impl Shared {
    fn session(&self) -> MutexGuard<'_, Option<Session>> {
        self.session.lock().unwrap_or_else(PoisonError::into_inner)
    }

    /// Records `state` and delivers it.
    fn emit(&self, sink: &SharedSink<EditorEvent>, state: EditorState) {
        *self.state.lock().unwrap_or_else(PoisonError::into_inner) = Some(state.clone());
        sink.emit(EditorEvent::State { state });
    }
}

/// How the server is started, once it is known to be installed.
struct Launch {
    spec: CommandSpec,
    name: String,
    version: Option<String>,
}

/// The app's one language server.
pub(crate) struct LanguageServer {
    workspace: PathBuf,
    venv: PathBuf,
    server: Option<CommandSpec>,
    policy: RestartPolicy,
    shared: Arc<Shared>,
    /// Serializes `start`, so two starts cannot both believe they replaced the other.
    starting: Mutex<()>,
}

impl LanguageServer {
    /// Creates the manager. Nothing is started until [`LanguageServer::start`].
    pub fn new(workspace: &Path, venv: PathBuf, options: &EditorOptions) -> Self {
        Self {
            workspace: workspace.to_owned(),
            venv,
            server: options.server.clone(),
            policy: options.restart,
            shared: Arc::new(Shared::default()),
            starting: Mutex::new(()),
        }
    }

    fn program(&self) -> PathBuf {
        venv_program(&self.venv, SERVER)
    }

    /// Whether the language server can be started without installing anything.
    pub fn installed(&self) -> bool {
        self.server.is_some() || self.program().is_file()
    }

    /// The version of the installed server, when it is the real one.
    pub fn version(&self) -> Option<String> {
        if self.server.is_some() || !self.program().is_file() {
            return None;
        }
        let spec = CommandSpec {
            program: self.program().into(),
            args: vec!["--version".into()],
            ..CommandSpec::default()
        };
        capture(&spec).and_then(|output| parse_tool_version(&output, SERVER))
    }

    /// The latest state of the latest session.
    pub fn state(&self) -> Option<EditorState> {
        self.shared
            .state
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .clone()
    }

    fn launch(&self, workspace: &Path) -> Option<Launch> {
        let (mut spec, name, version) = match &self.server {
            Some(spec) => {
                let name = Path::new(&spec.program)
                    .file_stem()
                    .map_or_else(String::new, |stem| stem.to_string_lossy().into_owned());
                (spec.clone(), name, None)
            }
            None if self.program().is_file() => (
                CommandSpec {
                    program: self.program().into(),
                    args: vec!["server".into()],
                    ..CommandSpec::default()
                },
                SERVER.to_owned(),
                self.version(),
            ),
            None => return None,
        };
        // What `uv run` would set; the inline configuration names the environment too.
        spec.envs
            .push(("VIRTUAL_ENV".into(), self.venv.clone().into()));
        if spec.cwd.is_none() {
            spec.cwd = Some(workspace.to_owned());
        }
        Some(Launch {
            spec,
            name,
            version,
        })
    }

    /// Starts a new session and returns its id. A session that is running is stopped
    /// first (its stream ends with `stopped`, reason `replaced`): a language server
    /// accepts `initialize` once, so a new client needs a new server.
    ///
    /// Events flow to `sink` from background threads and end with exactly one terminal
    /// state. When the server is not installed that state is `notInstalled`, emitted
    /// before this returns, and nothing is started or downloaded.
    pub fn start(&self, sink: SharedSink<EditorEvent>) -> Result<String> {
        let _starting = self.starting.lock().unwrap_or_else(PoisonError::into_inner);
        self.stop(None, EditorStopReason::Replaced);

        std::fs::create_dir_all(&self.workspace)
            .map_err(|e| CoreError::io("create", &self.workspace, e))?;
        // The server reports paths with symlinks resolved; URIs must match its view.
        let workspace = std::fs::canonicalize(&self.workspace)
            .map_err(|e| CoreError::io("resolve", &self.workspace, e))?;
        let id = uuid::Uuid::new_v4().to_string();

        let Some(launch) = self.launch(&workspace) else {
            self.shared.emit(
                &sink,
                EditorState::NotInstalled {
                    message: "The language server (ty) is not installed in the Python \
                              environment. Install editor intelligence from Settings."
                        .to_owned(),
                },
            );
            return Ok(id);
        };

        let context = Arc::new(ServerContext {
            root_uri: file_uri(&workspace),
            venv: self.venv.clone(),
        });
        let (tx, rx) = mpsc::channel();
        let writer: Writer = Arc::new(Mutex::new(None));
        let supervisor = Supervisor {
            shared: Arc::clone(&self.shared),
            session_id: id.clone(),
            launch,
            context: Arc::clone(&context),
            workspace,
            restarts: Restarts::new(self.policy),
            sink,
            writer: Arc::clone(&writer),
            tx: tx.clone(),
        };
        // Registered before the thread runs, so a `send` that follows `ready` finds it.
        let mut session = self.shared.session();
        *session = Some(Session {
            id: id.clone(),
            control: tx,
            writer,
            context,
            thread: None,
        });
        let spawned = thread::Builder::new()
            .name("language-server".to_owned())
            .spawn(move || supervisor.run(&rx));
        match spawned {
            Ok(handle) => {
                if let Some(session) = session.as_mut() {
                    session.thread = Some(handle);
                }
                Ok(id)
            }
            Err(source) => {
                *session = None;
                Err(CoreError::Io {
                    context: "failed to start the language server thread".to_owned(),
                    source,
                })
            }
        }
    }

    /// Sends one JSON-RPC message from the editor's client to the server of session
    /// `session_id`. Fails when that session is not the current one or its server is
    /// not running right now.
    pub fn send(&self, session_id: &str, message: &str) -> Result<()> {
        let (writer, context) = match &*self.shared.session() {
            Some(session) if session.id == session_id => {
                (Arc::clone(&session.writer), Arc::clone(&session.context))
            }
            _ => return Err(CoreError::EditorNotRunning),
        };
        let message = context.prepare(message)?;
        // The session lock is released: a server that stopped reading must not keep
        // `stop` from killing it.
        let mut stdin = writer.lock().unwrap_or_else(PoisonError::into_inner);
        let Some(pipe) = stdin.as_mut() else {
            return Err(CoreError::EditorNotRunning);
        };
        pipe.write_all(&frame(&message))
            .and_then(|()| pipe.flush())
            .map_err(|_| CoreError::EditorNotRunning)
    }

    /// Stops the session and waits until its server is gone. With `session_id`, only
    /// that session is stopped: a stop that names a session which was already replaced
    /// does nothing. Returns whether a session was stopped.
    pub fn stop(&self, session_id: Option<&str>, reason: EditorStopReason) -> bool {
        let session = {
            let mut current = self.shared.session();
            match (&*current, session_id) {
                (Some(session), Some(id)) if session.id != id => return false,
                _ => current.take(),
            }
        };
        let Some(mut session) = session else {
            return false;
        };
        // The receiver is gone only if the supervisor already finished.
        let _ = session.control.send(Control::Stop(reason));
        if let Some(handle) = session.thread.take() {
            let _ = handle.join();
        }
        true
    }
}

impl Drop for LanguageServer {
    fn drop(&mut self) {
        self.stop(None, EditorStopReason::Shutdown);
    }
}

/// Owns the server process of one session: starts it, forwards its messages, and
/// starts it again when it dies.
struct Supervisor {
    shared: Arc<Shared>,
    session_id: String,
    launch: Launch,
    context: Arc<ServerContext>,
    workspace: PathBuf,
    restarts: Restarts,
    sink: SharedSink<EditorEvent>,
    writer: Writer,
    tx: Sender<Control>,
}

impl Supervisor {
    fn emit(&self, state: EditorState) {
        self.shared.emit(&self.sink, state);
    }

    fn set_writer(&self, stdin: Option<ChildStdin>) {
        *self.writer.lock().unwrap_or_else(PoisonError::into_inner) = stdin;
    }

    /// Ends the session with a terminal state that was not asked for.
    fn give_up(&self, state: EditorState) {
        {
            let mut current = self.shared.session();
            if current
                .as_ref()
                .is_some_and(|session| session.id == self.session_id)
            {
                // Dropping the session detaches this thread's own handle.
                *current = None;
            }
        }
        self.emit(state);
    }

    fn run(mut self, rx: &Receiver<Control>) {
        let mut generation = 0;
        loop {
            self.emit(EditorState::Starting);
            let spawned = LocalProcess::spawn_io(
                &self.launch.spec,
                Stdio::piped(),
                Stdio::piped(),
                Stdio::piped(),
            );
            let mut process = match spawned {
                Ok(process) => process,
                Err(error) => {
                    tracing::warn!("the language server could not be started: {error}");
                    self.give_up(EditorState::Failed {
                        message: format!("The language server could not be started: {error}"),
                    });
                    return;
                }
            };
            generation += 1;
            let started = Instant::now();
            let stderr_tail = Arc::new(Mutex::new(VecDeque::new()));
            let mut readers = Vec::new();
            if let Some(stdout) = process.take_stdout() {
                let (sink, tx) = (Arc::clone(&self.sink), self.tx.clone());
                let reader =
                    thread::Builder::new()
                        .name("lsp-reader".to_owned())
                        .spawn(move || {
                            let problem = forward_messages(stdout, &sink);
                            let _ = tx.send(Control::Exited {
                                generation,
                                problem,
                            });
                        });
                if let Err(error) = &reader {
                    // Nobody would notice this server dying; treat it as dead now.
                    let _ = self.tx.send(Control::Exited {
                        generation,
                        problem: Some(format!("its output could not be read ({error})")),
                    });
                }
                readers.push(reader);
            }
            if let Some(stderr) = process.take_stderr() {
                let tail = Arc::clone(&stderr_tail);
                readers.push(thread::Builder::new().name("lsp-stderr".to_owned()).spawn(
                    move || {
                        for_each_line(stderr, |line| {
                            tracing::debug!(target: "learnai_core::lsp::server", "{line}");
                            let mut tail = tail.lock().unwrap_or_else(PoisonError::into_inner);
                            if tail.len() == STDERR_TAIL {
                                tail.pop_front();
                            }
                            tail.push_back(line);
                            true
                        });
                    },
                ));
            }
            self.set_writer(process.take_stdin());
            tracing::info!(
                server = %self.launch.name,
                version = ?self.launch.version,
                generation,
                "language server started"
            );
            self.emit(EditorState::Ready {
                session_id: self.session_id.clone(),
                server_name: self.launch.name.clone(),
                server_version: self.launch.version.clone(),
                root_uri: self.context.root_uri.clone(),
                workspace_path: self.workspace.to_string_lossy().into_owned(),
                generation,
            });

            // Wait for the server to die or the session to be stopped.
            let ended = loop {
                match rx.recv_timeout(EXIT_POLL) {
                    Ok(Control::Exited { generation: g, .. }) if g != generation => {}
                    Ok(control) => break Some(control),
                    Err(RecvTimeoutError::Timeout) => {
                        if process.wait_exit(Duration::ZERO) {
                            break Some(Control::Exited {
                                generation,
                                problem: None,
                            });
                        }
                    }
                    Err(RecvTimeoutError::Disconnected) => break None,
                }
            };
            // Whatever happened, this server is finished: kill its process group and
            // let the readers drain before saying anything more. The kill comes before
            // taking the writer: a `send` blocked on a server that stopped reading
            // holds the writer until the pipe breaks.
            let grace = match &ended {
                Some(Control::Exited { problem: None, .. }) => EXIT_GRACE,
                _ => Duration::ZERO,
            };
            let code = process.finish(grace);
            self.set_writer(None);
            for reader in readers.into_iter().flatten() {
                let _ = reader.join();
            }

            let problem = match ended {
                Some(Control::Exited { problem, .. }) => problem,
                Some(Control::Stop(reason)) => {
                    tracing::info!(%reason, "language server stopped");
                    self.emit(EditorState::Stopped { reason });
                    return;
                }
                None => {
                    self.emit(EditorState::Stopped {
                        reason: EditorStopReason::Shutdown,
                    });
                    return;
                }
            };
            let last_line = stderr_tail
                .lock()
                .unwrap_or_else(PoisonError::into_inner)
                .back()
                .cloned();
            let message = exit_message(problem, code, last_line.as_deref());
            tracing::warn!(generation, "{message}");

            let Some((attempt, delay)) = self.restarts.next(started.elapsed()) else {
                self.give_up(EditorState::Failed {
                    message: format!(
                        "{message} It was restarted {} times in a row and stopped each \
                         time, so it was not started again. Turn editor intelligence off \
                         and on in Settings to try again; the app log has the details.",
                        self.restarts.policy.max_restarts
                    ),
                });
                return;
            };
            self.emit(EditorState::Restarting {
                attempt,
                max_attempts: self.restarts.policy.max_restarts,
                delay_ms: u64::try_from(delay.as_millis()).unwrap_or(u64::MAX),
                message,
            });
            let deadline = Instant::now() + delay;
            loop {
                match rx.recv_timeout(deadline.saturating_duration_since(Instant::now())) {
                    Ok(Control::Stop(reason)) => {
                        self.emit(EditorState::Stopped { reason });
                        return;
                    }
                    // A late note from a reader of the server that is already gone.
                    Ok(Control::Exited { .. }) => {}
                    Err(RecvTimeoutError::Timeout) => break,
                    Err(RecvTimeoutError::Disconnected) => {
                        self.emit(EditorState::Stopped {
                            reason: EditorStopReason::Shutdown,
                        });
                        return;
                    }
                }
            }
        }
    }
}

/// Reads framed messages from the server's stdout and delivers each one. Returns what
/// went wrong, or `None` when the output simply ended (the server exited).
fn forward_messages(
    mut pipe: Box<dyn Read + Send>,
    sink: &SharedSink<EditorEvent>,
) -> Option<String> {
    let mut decoder = FrameDecoder::default();
    let mut chunk = vec![0; 64 * 1024];
    loop {
        match pipe.read(&mut chunk) {
            Ok(0) => return None,
            Ok(read) => decoder.push(&chunk[..read]),
            Err(error) if error.kind() == std::io::ErrorKind::Interrupted => continue,
            Err(error) => return Some(format!("its output could not be read ({error})")),
        }
        loop {
            match decoder.next_message() {
                Ok(Some(message)) => sink.emit(EditorEvent::Message { message }),
                Ok(None) => break,
                Err(error) => {
                    return Some(format!("it sent output that is not a message ({error})"));
                }
            }
        }
    }
}

/// One sentence saying how the server ended, with its last word when it left one.
fn exit_message(problem: Option<String>, code: Option<i32>, last_line: Option<&str>) -> String {
    let how = problem.unwrap_or_else(|| match code {
        Some(code) => format!("it exited with code {code}"),
        None => "it was killed by a signal".to_owned(),
    });
    match last_line.map(str::trim).filter(|line| !line.is_empty()) {
        Some(line) => format!("The language server stopped: {how}. Its last output: {line}"),
        None => format!("The language server stopped: {how}."),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn decode_all(decoder: &mut FrameDecoder) -> Vec<String> {
        let mut messages = Vec::new();
        while let Some(message) = decoder.next_message().expect("well-formed stream") {
            messages.push(message);
        }
        messages
    }

    #[test]
    fn frames_carry_the_byte_length() {
        assert_eq!(frame("{}"), b"Content-Length: 2\r\n\r\n{}");
        // Length is counted in bytes, not characters.
        assert_eq!(
            frame(r#"{"a":"é"}"#),
            "Content-Length: 10\r\n\r\n{\"a\":\"é\"}".as_bytes()
        );
    }

    #[test]
    fn a_message_split_at_every_byte_is_reassembled() {
        let stream = frame(r#"{"jsonrpc":"2.0","id":1,"result":"é"}"#);
        let mut decoder = FrameDecoder::default();
        let mut messages = Vec::new();
        for byte in &stream {
            decoder.push(std::slice::from_ref(byte));
            messages.extend(decode_all(&mut decoder));
        }
        assert_eq!(messages, [r#"{"jsonrpc":"2.0","id":1,"result":"é"}"#]);
    }

    #[test]
    fn messages_that_arrive_together_are_separated() {
        let mut stream = frame("{\"id\":1}");
        stream.extend(frame("{\"id\":2}"));
        stream.extend(frame(""));
        stream.extend(frame("{\"id\":3}"));
        let (first, second) = stream.split_at(stream.len() - 5);

        let mut decoder = FrameDecoder::default();
        decoder.push(first);
        assert_eq!(decode_all(&mut decoder), ["{\"id\":1}", "{\"id\":2}", ""]);
        decoder.push(second);
        assert_eq!(decode_all(&mut decoder), ["{\"id\":3}"]);
        assert_eq!(decoder.next_message(), Ok(None));
    }

    #[test]
    fn large_bodies_arrive_whole() {
        let body = format!("{{\"text\":\"{}\"}}", "x".repeat(5 * 1024 * 1024));
        let stream = frame(&body);
        let mut decoder = FrameDecoder::default();
        let mut messages = Vec::new();
        for piece in stream.chunks(64 * 1024) {
            decoder.push(piece);
            messages.extend(decode_all(&mut decoder));
        }
        assert_eq!(messages.len(), 1);
        assert_eq!(messages[0], body);
    }

    #[test]
    fn other_headers_and_header_case_are_tolerated() {
        let mut decoder = FrameDecoder::default();
        decoder.push(
            b"Content-Type: application/vscode-jsonrpc; charset=utf-8\r\ncontent-length:  2 \r\n\r\n{}",
        );
        assert_eq!(decode_all(&mut decoder), ["{}"]);
    }

    #[test]
    fn bad_headers_are_errors() {
        let decode = |bytes: &[u8]| {
            let mut decoder = FrameDecoder::default();
            decoder.push(bytes);
            decoder.next_message()
        };
        assert_eq!(
            decode(b"Content-Type: x\r\n\r\n{}"),
            Err(FrameError::MissingLength)
        );
        assert_eq!(
            decode(b"Content-Length: two\r\n\r\n{}"),
            Err(FrameError::BadLength("two".to_owned()))
        );
        assert_eq!(
            decode(b"Content-Length: -1\r\n\r\n{}"),
            Err(FrameError::BadLength("-1".to_owned()))
        );
        assert_eq!(
            decode(b"a log line the server printed to stdout\r\n\r\n"),
            Err(FrameError::BadHeader(
                "a log line the server printed to stdout".to_owned()
            ))
        );
        assert_eq!(
            decode(b"Content-Length: 99999999999\r\n\r\n"),
            Err(FrameError::TooLarge(99_999_999_999))
        );
        // Output with no header terminator at all (a server printing plain text).
        assert_eq!(decode(&[b'x'; 100]), Ok(None));
        assert_eq!(
            decode(&vec![b'x'; MAX_HEADER_BYTES + 1]),
            Err(FrameError::HeadersTooLong)
        );
    }

    #[test]
    fn file_uris_are_percent_encoded() {
        assert_eq!(
            file_uri(Path::new("/Users/ada/learnai/workspace")),
            "file:///Users/ada/learnai/workspace"
        );
        assert_eq!(
            file_uri(Path::new("/tmp/my files/caf\u{e9} #1/w+x")),
            "file:///tmp/my%20files/caf%C3%A9%20%231/w%2Bx"
        );
    }

    fn context() -> ServerContext {
        ServerContext {
            root_uri: "file:///r/workspace".to_owned(),
            venv: PathBuf::from("/r/python/.venv"),
        }
    }

    #[test]
    fn initialize_is_rewritten() {
        // What @codemirror/lsp-client sends, plus things a client must not decide.
        let sent = json!({
            "jsonrpc": "2.0", "id": 0, "method": "initialize",
            "params": {
                "processId": null,
                "clientInfo": {"name": "@codemirror/lsp-client"},
                "rootUri": "file:///r/problems",
                "rootPath": "/r/problems",
                "workspaceFolders": [{"uri": "file:///r/problems", "name": "p"}],
                "initializationOptions": {"configuration": {"rules": {"all": "error"}}},
                "capabilities": {
                    "general": {"markdown": {"parser": "marked"}},
                    "textDocument": {
                        "hover": {"contentFormat": ["markdown", "plaintext"]},
                        "diagnostic": {},
                        "publishDiagnostics": {"versionSupport": true}
                    },
                    "workspace": {"configuration": true, "workspaceFolders": true},
                    "window": {"showMessage": {}}
                }
            }
        });
        let sent = sent.to_string();
        let prepared = context().prepare(&sent).expect("prepare");
        let value: Value = serde_json::from_str(&prepared).expect("JSON");
        assert_eq!(
            value,
            json!({
                "jsonrpc": "2.0", "id": 0, "method": "initialize",
                "params": {
                    "processId": std::process::id(),
                    "clientInfo": {"name": "@codemirror/lsp-client"},
                    "rootUri": "file:///r/workspace",
                    "workspaceFolders": [{"uri": "file:///r/workspace", "name": "workspace"}],
                    "initializationOptions": {
                        "diagnosticMode": "openFilesOnly",
                        "completions": {"autoImport": false},
                        "configuration": {
                            "environment": {"python": "/r/python/.venv"},
                            "rules": {
                                "all": "ignore",
                                "unresolved-reference": "error",
                                "import-not-found": "error",
                                "missing-attribute": "error",
                                "missing-argument": "error",
                                "too-many-positional-arguments": "error",
                                "unknown-argument": "error",
                                "parameter-already-assigned": "error"
                            }
                        }
                    },
                    "capabilities": {
                        "general": {"markdown": {"parser": "marked"}},
                        "textDocument": {
                            "hover": {"contentFormat": ["markdown", "plaintext"]},
                            "publishDiagnostics": {"versionSupport": true}
                        },
                        "workspace": {"workspaceFolders": true},
                        "window": {"showMessage": {}}
                    }
                }
            })
        );
    }

    #[test]
    fn initialize_without_parameters_gets_them() {
        let prepared = context()
            .prepare(r#"{"jsonrpc":"2.0","id":0,"method":"initialize"}"#)
            .expect("prepare");
        let value: Value = serde_json::from_str(&prepared).expect("JSON");
        assert_eq!(value["params"]["rootUri"], "file:///r/workspace");
        assert_eq!(
            value["params"]["initializationOptions"]["configuration"]["rules"]["all"],
            "ignore"
        );
    }

    #[test]
    fn other_messages_pass_through_byte_for_byte() {
        // Odd spacing and a `diagnostic` key outside `initialize` are left alone.
        let message = r#"{ "jsonrpc":"2.0",  "method":"textDocument/didOpen", "params":{"capabilities":{"textDocument":{"diagnostic":{}}}, "n": 1.50} }"#;
        let prepared = context().prepare(message).expect("prepare");
        assert!(matches!(prepared, Cow::Borrowed(_)));
        assert_eq!(prepared, message);
        // A response to a server request has no method.
        let response = r#"{"jsonrpc":"2.0","id":3,"error":{"code":-32601,"message":"Method not implemented"}}"#;
        assert_eq!(context().prepare(response).expect("prepare"), response);
    }

    #[test]
    fn messages_that_are_not_json_rpc_objects_are_refused() {
        for bad in [
            "",
            "Content-Length: 2\r\n\r\n{}",
            "[1, 2]",
            "\"initialize\"",
            "{",
        ] {
            let error = context().prepare(bad).expect_err(bad);
            assert!(
                matches!(error, CoreError::InvalidArgument(_)),
                "{bad}: {error}"
            );
        }
        let error = context()
            .prepare(r#"{"method":"initialize","params":[1]}"#)
            .expect_err("array parameters");
        assert!(matches!(error, CoreError::InvalidArgument(_)));
    }

    #[test]
    fn restart_delays_double_up_to_the_cap() {
        let policy = RestartPolicy::default();
        let delays: Vec<u64> = (1..=6)
            .map(|n| u64::try_from(policy.delay(n).as_millis()).expect("small"))
            .collect();
        assert_eq!(delays, [500, 1000, 2000, 4000, 8000, 8000]);
        assert_eq!(policy.delay(u32::MAX), Duration::from_secs(8));
    }

    #[test]
    fn restarts_are_bounded_and_the_count_starts_over_after_a_stable_run() {
        let policy = RestartPolicy {
            max_restarts: 3,
            base_delay: Duration::from_millis(100),
            max_delay: Duration::from_millis(250),
            stable_after: Duration::from_secs(60),
        };
        let quick = Duration::from_secs(1);
        let mut restarts = Restarts::new(policy);
        assert_eq!(restarts.next(quick), Some((1, Duration::from_millis(100))));
        assert_eq!(restarts.next(quick), Some((2, Duration::from_millis(200))));
        assert_eq!(restarts.next(quick), Some((3, Duration::from_millis(250))));
        assert_eq!(restarts.next(quick), None);
        assert_eq!(restarts.next(quick), None, "the limit stays spent");

        // A server that then ran for a while earns a fresh count.
        assert_eq!(
            restarts.next(Duration::from_secs(61)),
            Some((1, Duration::from_millis(100)))
        );
        assert_eq!(restarts.next(quick), Some((2, Duration::from_millis(200))));

        let mut never = Restarts::new(RestartPolicy {
            max_restarts: 0,
            ..policy
        });
        assert_eq!(never.next(Duration::from_secs(3600)), None);
    }

    #[test]
    fn exit_messages_say_what_happened() {
        assert_eq!(
            exit_message(None, Some(3), None),
            "The language server stopped: it exited with code 3."
        );
        assert_eq!(
            exit_message(None, None, Some("  panicked at main.rs  ")),
            "The language server stopped: it was killed by a signal. Its last output: \
             panicked at main.rs"
        );
        assert_eq!(
            exit_message(
                Some("it sent output that is not a message (x)".into()),
                Some(0),
                Some("")
            ),
            "The language server stopped: it sent output that is not a message (x)."
        );
    }
}
