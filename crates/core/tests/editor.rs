//! Editor intelligence against a fake language server and a fake docs helper
//! (`fixtures/fake_lsp.pl`, `fixtures/fake_docs.sh`): the session lifecycle, message
//! framing in both directions, the `initialize` rewrite as the server sees it,
//! restarts, the on/off preference, installing, and process cleanup.

// The fakes these tests drive are shell and Perl scripts started as programs, which
// Windows cannot do. Windows needs its own process tests with a fake that is not a script.
#![cfg(unix)]

mod common;

use std::fs;
use std::path::{Path, PathBuf};
use std::time::Duration;

use common::{Collector, TestRoot, fixtures, process_alive, wait_for_exit, wait_for_file};
use learnai_core::{
    CommandSpec, Core, CoreError, CoreOptions, EditorEvent, EditorOptions, EditorState,
    EditorStopReason, Extra, Framework, InstallEvent, RestartPolicy,
};
use serde_json::{Value, json};

/// A content root with a state directory for the fakes.
struct Fixture {
    root: TestRoot,
}

impl Fixture {
    fn new() -> Self {
        let root = TestRoot::new();
        fs::create_dir_all(root.path().join("fake")).expect("state directory");
        Self { root }
    }

    fn state(&self) -> PathBuf {
        self.root.path().join("fake")
    }

    fn server(&self, scenario: &str) -> CommandSpec {
        CommandSpec {
            program: "perl".into(),
            args: vec![fixtures().join("fake_lsp.pl").into()],
            envs: vec![
                ("FAKE_LSP".into(), scenario.into()),
                ("FAKE_LSP_DIR".into(), self.state().into()),
            ],
            cwd: None,
        }
    }

    fn helper(&self, scenario: &str) -> CommandSpec {
        CommandSpec {
            program: "sh".into(),
            args: vec![fixtures().join("fake_docs.sh").into()],
            envs: vec![
                ("FAKE_DOCS".into(), scenario.into()),
                ("FAKE_DOCS_DIR".into(), self.state().into()),
            ],
            cwd: None,
        }
    }

    fn options(&self, server: &str, helper: &str) -> EditorOptions {
        EditorOptions {
            server: Some(self.server(server)),
            docs: Some(self.helper(helper)),
            restart: RestartPolicy {
                max_restarts: 2,
                base_delay: Duration::from_millis(20),
                max_delay: Duration::from_millis(40),
                stable_after: Duration::from_secs(60),
            },
            docs_timeout: Duration::from_millis(1500),
            docs_retry_pause: Duration::from_millis(300),
            ..EditorOptions::default()
        }
    }

    fn core_with(&self, editor: EditorOptions) -> Core {
        Core::open(CoreOptions {
            uv_program: Some(fixtures().join("fake_uv.sh")),
            editor,
            ..CoreOptions::new(self.root.path())
        })
        .expect("open core")
    }

    /// A core whose language server and docs helper are the fakes in these scenarios.
    fn core(&self, server: &str, helper: &str) -> Core {
        self.core_with(self.options(server, helper))
    }

    /// The workspace directory as the server sees it.
    fn workspace(&self) -> PathBuf {
        fs::canonicalize(self.root.path().join("workspace")).expect("workspace exists")
    }

    fn lines(&self, name: &str) -> Vec<String> {
        fs::read_to_string(self.state().join(name))
            .unwrap_or_default()
            .lines()
            .map(str::to_owned)
            .collect()
    }

    fn server_pid(&self) -> String {
        wait_for_file(&self.state().join("server.pid"))
    }
}

fn states(events: &[EditorEvent]) -> Vec<EditorState> {
    events
        .iter()
        .filter_map(|event| match event {
            EditorEvent::State { state } => Some(state.clone()),
            EditorEvent::Message { .. } => None,
        })
        .collect()
}

/// The server's messages, parsed.
fn messages(events: &[EditorEvent]) -> Vec<Value> {
    events
        .iter()
        .filter_map(|event| match event {
            EditorEvent::Message { message } => {
                Some(serde_json::from_str(message).expect("the server sends JSON"))
            }
            EditorEvent::State { .. } => None,
        })
        .collect()
}

fn ready_count(events: &[EditorEvent]) -> usize {
    states(events)
        .iter()
        .filter(|state| matches!(state, EditorState::Ready { .. }))
        .count()
}

fn wait_ready(collector: &Collector<EditorEvent>, count: usize) -> Vec<EditorEvent> {
    collector.wait_for("the ready state", |events| ready_count(events) >= count)
}

fn wait_last(collector: &Collector<EditorEvent>) -> Vec<EditorEvent> {
    let events = collector.wait_for("the terminal state", |events| {
        events.iter().any(EditorEvent::is_last)
    });
    assert_well_formed(&events);
    events
}

fn wait_message(
    collector: &Collector<EditorEvent>,
    what: &str,
    found: impl Fn(&Value) -> bool,
) -> Value {
    let events = collector.wait_for(what, |events| messages(events).iter().any(&found));
    messages(&events)
        .into_iter()
        .find(|m| found(m))
        .expect("message")
}

/// Exactly one terminal state, and it is the last event.
fn assert_well_formed(events: &[EditorEvent]) {
    assert_eq!(
        events.iter().filter(|e| e.is_last()).count(),
        1,
        "exactly one terminal state: {events:?}"
    );
    assert!(
        events.last().is_some_and(EditorEvent::is_last),
        "the terminal state is last: {events:?}"
    );
}

fn request(id: u64, method: &str, params: Value) -> String {
    json!({"jsonrpc": "2.0", "id": id, "method": method, "params": params}).to_string()
}

fn notification(method: &str, params: Value) -> String {
    json!({"jsonrpc": "2.0", "method": method, "params": params}).to_string()
}

#[test]
fn a_session_starts_round_trips_messages_and_stops_cleanly() {
    let fixture = Fixture::new();
    let core = fixture.core("", "");
    assert_eq!(core.editor_status().expect("status").state, None);

    let collector = Collector::new();
    let session = core.editor_start(collector.sink()).expect("start");
    let events = wait_ready(&collector, 1);
    let workspace = fixture.workspace();
    let root_uri = format!("file://{}", workspace.display());
    assert_eq!(
        states(&events),
        [
            EditorState::Starting,
            EditorState::Ready {
                session_id: session.clone(),
                server_name: "perl".into(),
                server_version: None,
                root_uri: root_uri.clone(),
                workspace_path: workspace.to_string_lossy().into_owned(),
                generation: 1,
            }
        ]
    );
    let status = core.editor_status().expect("status");
    assert!(status.enabled && status.installed && !status.installing);
    assert!(matches!(status.state, Some(EditorState::Ready { .. })));

    // `initialize` reaches the server rewritten: core's root and settings, no pull
    // diagnostics, whatever the client asked for.
    let initialize = request(
        0,
        "initialize",
        json!({
            "processId": null,
            "rootUri": "file:///somewhere/problems",
            "capabilities": {"textDocument": {"diagnostic": {}, "hover": {}}}
        }),
    );
    core.editor_send(&session, &initialize).expect("send");
    let reply = wait_message(&collector, "the initialize reply", |m| m["id"] == 0);
    let seen = &reply["result"]["params"];
    assert_eq!(seen["rootUri"], root_uri.as_str());
    assert_eq!(seen["processId"], std::process::id());
    assert_eq!(seen["capabilities"]["textDocument"], json!({"hover": {}}));
    let venv = fixture.root.path().join("python/.venv");
    let configuration = &seen["initializationOptions"]["configuration"];
    assert_eq!(
        configuration["environment"]["python"],
        venv.to_string_lossy().as_ref()
    );
    assert_eq!(configuration["rules"]["all"], "ignore");
    assert_eq!(configuration["rules"]["missing-attribute"], "error");
    // The server runs in the workspace with the project environment selected.
    assert_eq!(
        reply["result"]["env"],
        json!({"VIRTUAL_ENV": venv.to_string_lossy(), "cwd": workspace.to_string_lossy()})
    );

    // Other messages pass through as sent; diagnostics are pushed back.
    let uri = format!("{root_uri}/add-two/torch.py");
    let open = notification(
        "textDocument/didOpen",
        json!({"textDocument": {"uri": uri, "languageId": "python", "version": 1,
               "text": "import torch\n# caf\u{e9} \u{1f600}\n"}}),
    );
    core.editor_send(&session, &open).expect("send");
    let diagnostics = wait_message(&collector, "pushed diagnostics", |m| {
        m["method"] == "textDocument/publishDiagnostics"
    });
    assert_eq!(diagnostics["params"]["uri"], uri.as_str());
    core.editor_send(
        &session,
        &request(1, "textDocument/hover", json!({"text": "caf\u{e9}"})),
    )
    .expect("send");
    let hover = wait_message(&collector, "the hover reply", |m| m["id"] == 1);
    assert_eq!(hover["result"]["params"]["text"], "caf\u{e9}");

    let pid = fixture.server_pid();
    assert!(process_alive(&pid));
    core.editor_stop(Some(&session));
    // `editor_stop` returns once the process is gone and the stream has ended.
    assert!(!process_alive(&pid));
    let events = collector.snapshot();
    assert_well_formed(&events);
    assert_eq!(
        states(&events).last(),
        Some(&EditorState::Stopped {
            reason: EditorStopReason::Requested
        })
    );
    assert_eq!(fixture.lines("starts").len(), 1);
    assert!(matches!(
        core.editor_send(&session, &request(2, "x", json!(null))),
        Err(CoreError::EditorNotRunning)
    ));
    // Stopping again, or stopping with nothing running, does nothing.
    core.editor_stop(Some(&session));
    core.editor_stop(None);
}

#[test]
fn server_output_in_pieces_in_bulk_and_at_size_arrives_as_whole_messages() {
    let fixture = Fixture::new();
    let core = fixture.core("", "");
    let collector = Collector::new();
    let session = core.editor_start(collector.sink()).expect("start");
    wait_ready(&collector, 1);

    core.editor_send(&session, &notification("test/split", json!({})))
        .expect("send");
    let events = collector.wait_for("three parts", |events| messages(events).len() >= 3);
    let parts: Vec<Value> = messages(&events)
        .iter()
        .map(|m| m["params"]["n"].clone())
        .collect();
    assert_eq!(parts, [json!(1), json!(2), json!(3)]);

    core.editor_send(&session, &notification("test/big", json!({})))
        .expect("send");
    let big = wait_message(&collector, "the big message", |m| m["method"] == "test/big");
    assert_eq!(
        big["params"]["text"].as_str().map(str::len),
        Some(3 * 1024 * 1024)
    );

    // A big message toward the server arrives whole as well.
    let text = "y".repeat(2 * 1024 * 1024);
    core.editor_send(&session, &request(7, "test/echo", json!({"text": text})))
        .expect("send");
    let echoed = wait_message(&collector, "the echo", |m| m["id"] == 7);
    assert_eq!(echoed["result"]["params"]["text"], text.as_str());
}

#[test]
fn server_requests_and_the_clients_answers_pass_through() {
    let fixture = Fixture::new();
    let core = fixture.core("", "");
    let collector = Collector::new();
    let session = core.editor_start(collector.sink()).expect("start");
    wait_ready(&collector, 1);

    core.editor_send(&session, &notification("test/request", json!({})))
        .expect("send");
    let asked = wait_message(&collector, "the server's request", |m| m["id"] == "srv-1");
    assert_eq!(asked["method"], "workspace/configuration");
    // What @codemirror/lsp-client answers to every server request.
    let answer = json!({"jsonrpc": "2.0", "id": "srv-1",
                        "error": {"code": -32601, "message": "Method not implemented"}});
    core.editor_send(&session, &answer.to_string())
        .expect("send");
    let seen = wait_message(&collector, "the answer", |m| m["method"] == "test/response");
    assert_eq!(seen["params"], answer);

    // Text that is not one JSON-RPC object never reaches the server.
    for bad in ["not json", "[]", "Content-Length: 2\r\n\r\n{}"] {
        assert!(matches!(
            core.editor_send(&session, bad),
            Err(CoreError::InvalidArgument(_))
        ));
    }
    core.editor_send(&session, &request(9, "ping", json!(null)))
        .expect("the session is unharmed");
    wait_message(&collector, "the ping reply", |m| m["id"] == 9);
}

#[test]
fn a_server_that_dies_is_started_again() {
    let fixture = Fixture::new();
    let core = fixture.core("die-once", "");
    let collector = Collector::new();
    let session = core.editor_start(collector.sink()).expect("start");

    let events = wait_ready(&collector, 2);
    let states = states(&events);
    assert!(
        matches!(
            &states[..],
            [
                EditorState::Starting,
                EditorState::Ready { generation: 1, .. },
                EditorState::Restarting {
                    attempt: 1,
                    max_attempts: 2,
                    delay_ms: 20,
                    ..
                },
                EditorState::Starting,
                EditorState::Ready { generation: 2, .. },
            ]
        ),
        "{states:?}"
    );
    let EditorState::Restarting { message, .. } = &states[2] else {
        unreachable!()
    };
    assert_eq!(
        message,
        "The language server stopped: it exited with code 3. Its last output: fatal: the \
         fake server was told to die"
    );

    // The new server is a new process: it needs `initialize` again and gets it.
    core.editor_send(&session, &request(0, "initialize", json!({})))
        .expect("send to the second server");
    wait_message(&collector, "the reply", |m| m["id"] == 0);
    assert_eq!(fixture.lines("starts").len(), 2);
    core.editor_stop(Some(&session));
    assert_well_formed(&collector.snapshot());
}

#[test]
fn a_dead_server_is_noticed_even_when_its_child_holds_its_output_open() {
    let fixture = Fixture::new();
    let mut options = fixture.options("", "");
    options
        .server
        .as_mut()
        .expect("server")
        .envs
        .push(("FAKE_LSP_CHILD".into(), "1".into()));
    let core = fixture.core_with(options);
    let collector = Collector::new();
    let session = core.editor_start(collector.sink()).expect("start");
    wait_ready(&collector, 1);
    let first_child = wait_for_file(&fixture.state().join("child.pid"));
    fs::remove_file(fixture.state().join("child.pid")).expect("remove pid file");

    core.editor_send(&session, &notification("test/exit", json!({})))
        .expect("send");
    wait_ready(&collector, 2);
    // What the dead server left running went with it.
    assert!(wait_for_exit(&first_child));

    let second_child = wait_for_file(&fixture.state().join("child.pid"));
    let server = fixture.server_pid();
    core.editor_stop(None);
    assert!(!process_alive(&server));
    assert!(
        wait_for_exit(&second_child),
        "the whole process group is killed"
    );
}

#[test]
fn restarts_are_bounded_and_end_in_failed() {
    let fixture = Fixture::new();
    let core = fixture.core("die", "");
    let collector = Collector::new();
    let session = core.editor_start(collector.sink()).expect("start");

    let events = wait_last(&collector);
    let states = states(&events);
    let attempts: Vec<(u32, u64)> = states
        .iter()
        .filter_map(|state| match state {
            EditorState::Restarting {
                attempt, delay_ms, ..
            } => Some((*attempt, *delay_ms)),
            _ => None,
        })
        .collect();
    assert_eq!(attempts, [(1, 20), (2, 40)]);
    assert_eq!(ready_count(&events), 3, "the first start and two restarts");
    let Some(EditorState::Failed { message }) = states.last() else {
        panic!("the session fails: {states:?}");
    };
    assert!(message.contains("it exited with code 3"), "{message}");
    assert!(message.contains("restarted 2 times in a row"), "{message}");
    assert!(message.contains("Settings"), "{message}");
    assert_eq!(
        fixture.lines("starts").len(),
        3,
        "nothing is started after that"
    );

    assert!(matches!(
        core.editor_status().expect("status").state,
        Some(EditorState::Failed { .. })
    ));
    assert!(matches!(
        core.editor_send(&session, &request(0, "initialize", json!({}))),
        Err(CoreError::EditorNotRunning)
    ));
    // A failed session leaves nothing to stop, and a new start tries afresh.
    core.editor_stop(Some(&session));
    let again = Collector::new();
    core.editor_start(again.sink()).expect("start again");
    wait_last(&again);
    assert_eq!(fixture.lines("starts").len(), 6);
}

#[test]
fn output_that_is_not_framed_messages_ends_that_server() {
    let fixture = Fixture::new();
    let core = fixture.core("garbage", "");
    let collector = Collector::new();
    core.editor_start(collector.sink()).expect("start");

    let events = wait_last(&collector);
    assert!(messages(&events).is_empty());
    let restarting = states(&events)
        .into_iter()
        .find_map(|state| match state {
            EditorState::Restarting { message, .. } => Some(message),
            _ => None,
        })
        .expect("a restart");
    assert!(
        restarting.starts_with(
            "The language server stopped: it sent output that is not a message (a message \
             has no Content-Length header)"
        ),
        "{restarting}"
    );
    // Each of those servers was idling, not exiting: it had to be killed.
    for pid in fixture.lines("starts") {
        assert!(wait_for_exit(&pid));
    }
}

#[test]
fn a_send_blocked_on_a_server_that_stopped_reading_does_not_block_stop() {
    let fixture = Fixture::new();
    let core = fixture.core("deaf", "");
    let collector = Collector::new();
    let session = core.editor_start(collector.sink()).expect("start");
    wait_ready(&collector, 1);
    let pid = fixture.server_pid();

    // Far more than a pipe holds: the write blocks until the server is killed.
    let big = request(1, "test/echo", json!({"text": "z".repeat(8 * 1024 * 1024)}));
    let sender = {
        let (core, session) = (core.clone(), session.clone());
        std::thread::spawn(move || core.editor_send(&session, &big))
    };
    std::thread::sleep(Duration::from_millis(200));
    assert!(
        !sender.is_finished(),
        "the send is blocked on the full pipe"
    );

    core.editor_stop(Some(&session));
    assert!(!process_alive(&pid));
    let sent = sender.join().expect("sender thread");
    assert!(matches!(sent, Err(CoreError::EditorNotRunning)), "{sent:?}");
    assert_well_formed(&collector.snapshot());
}

#[test]
fn a_new_start_replaces_the_running_session() {
    let fixture = Fixture::new();
    let core = fixture.core("", "");
    let first = Collector::new();
    let old = core.editor_start(first.sink()).expect("start");
    wait_ready(&first, 1);
    let old_pid = fixture.server_pid();

    let second = Collector::new();
    let new = core.editor_start(second.sink()).expect("start");
    assert_ne!(old, new);
    // The first stream has ended and its server is gone before the call returned.
    let events = first.snapshot();
    assert_well_formed(&events);
    assert_eq!(
        states(&events).last(),
        Some(&EditorState::Stopped {
            reason: EditorStopReason::Replaced
        })
    );
    assert!(!process_alive(&old_pid));

    wait_ready(&second, 1);
    assert!(matches!(
        core.editor_send(&old, &request(0, "initialize", json!({}))),
        Err(CoreError::EditorNotRunning)
    ));
    // A late stop from the page that started the first session leaves the second alone.
    core.editor_stop(Some(&old));
    core.editor_send(&new, &request(0, "initialize", json!({})))
        .expect("the second session is running");
    wait_message(&second, "the reply", |m| m["id"] == 0);
    assert_eq!(fixture.lines("starts").len(), 2);
}

#[test]
fn a_missing_server_is_reported_and_nothing_is_started_or_installed() {
    let fixture = Fixture::new();
    // No server override: the real location in the (nonexistent) environment is used.
    let core = fixture.core_with(EditorOptions {
        docs: Some(fixture.helper("")),
        ..EditorOptions::default()
    });
    let status = core.editor_status().expect("status");
    assert!(status.enabled && !status.installed);
    assert_eq!(status.server_version, None);

    let collector = Collector::new();
    let session = core.editor_start(collector.sink()).expect("start");
    let events = collector.snapshot();
    assert!(
        matches!(
            &events[..],
            [EditorEvent::State {
                state: EditorState::NotInstalled { message }
            }] if message.contains("not installed") && message.contains("Settings")
        ),
        "{events:?}"
    );
    assert!(matches!(
        core.editor_send(&session, &request(0, "initialize", json!({}))),
        Err(CoreError::EditorNotRunning)
    ));
    // Starting ran no uv command at all.
    assert!(!fixture.root.path().join("python/.fake-uv.log").exists());
}

fn install(core: &Core) -> Vec<InstallEvent> {
    let collector = Collector::new();
    core.editor_install(collector.sink())
        .expect("start install");
    collector.wait_for("the done event", |events| {
        events
            .iter()
            .any(|e| matches!(e, InstallEvent::Done { .. }))
    })
}

fn sync_log(root: &Path) -> Vec<String> {
    fs::read_to_string(root.join("python/.fake-uv.log"))
        .unwrap_or_default()
        .lines()
        .filter(|line| line.starts_with("sync"))
        .map(str::to_owned)
        .collect()
}

#[test]
fn installing_the_editor_extra_keeps_every_other_extra() {
    let fixture = Fixture::new();
    let core = fixture.core("", "");
    let root = fixture.root.path();
    // JAX was installed outside the app; nothing in the settings knows about it.
    fs::write(root.join("python/.fake-venv"), " jax\n").expect("seed environment");

    let events = install(&core);
    assert_eq!(
        events.last(),
        Some(&InstallEvent::Done {
            ok: true,
            message: None
        })
    );
    assert!(matches!(
        events.first(),
        Some(InstallEvent::Log { text }) if text.ends_with("--extra editor --extra jax")
    ));
    assert_eq!(sync_log(root), ["sync editor jax"]);
    assert_eq!(
        core.enabled_extras().expect("extras"),
        [Extra::Jax, Extra::Editor]
    );

    // A framework installed afterwards keeps the language server.
    let collector = Collector::new();
    core.env_install(Framework::Mlx, collector.sink())
        .expect("start install");
    collector.wait_for("the done event", |events| {
        events
            .iter()
            .any(|e| matches!(e, InstallEvent::Done { .. }))
    });
    assert_eq!(sync_log(root), ["sync editor jax", "sync editor jax mlx"]);

    // A failed install records nothing.
    let fixture = Fixture::new();
    let core = fixture.core("", "");
    fs::write(fixture.root.path().join("python/.fail-sync"), b"").expect("marker");
    assert!(matches!(
        install(&core).last(),
        Some(InstallEvent::Done { ok: false, .. })
    ));
    assert!(core.enabled_extras().expect("extras").is_empty());
}

#[test]
fn turning_the_preference_off_stops_everything_and_keeps_it_stopped() {
    let fixture = Fixture::new();
    let core = fixture.core("", "");
    let settings = core.get_settings().expect("settings");
    assert!(settings.editor_intelligence, "on by default");

    let collector = Collector::new();
    let session = core.editor_start(collector.sink()).expect("start");
    wait_ready(&collector, 1);
    assert!(core.editor_docs("", "F.softmax", None).expect("docs").found);
    let server = fixture.server_pid();
    let helper = fixture.lines("pids")[0].clone();
    assert!(process_alive(&server) && process_alive(&helper));

    let settings = core.set_editor_intelligence(false).expect("turn off");
    assert!(!settings.editor_intelligence);
    assert!(settings.reviews_enabled, "other settings are untouched");
    assert!(!process_alive(&server));
    assert!(wait_for_exit(&helper));
    let events = collector.snapshot();
    assert_well_formed(&events);
    assert_eq!(
        states(&events).last(),
        Some(&EditorState::Stopped {
            reason: EditorStopReason::Disabled
        })
    );

    // While it is off nothing starts.
    let refused = Collector::<EditorEvent>::new();
    assert!(matches!(
        core.editor_start(refused.sink()),
        Err(CoreError::EditorDisabled)
    ));
    assert!(refused.snapshot().is_empty());
    assert!(matches!(
        core.editor_docs("", "F.softmax", None),
        Err(CoreError::EditorDisabled)
    ));
    assert!(matches!(
        core.editor_send(&session, &request(0, "initialize", json!({}))),
        Err(CoreError::EditorNotRunning)
    ));
    assert_eq!(fixture.lines("starts").len(), 1);
    assert_eq!(fixture.lines("pids").len(), 1);
    assert!(!core.editor_status().expect("status").enabled);

    // The preference is stored: a new core for the same root still has it off.
    drop(core);
    let core = fixture.core("", "");
    assert!(!core.get_settings().expect("settings").editor_intelligence);
    assert!(
        core.set_editor_intelligence(true)
            .expect("turn on")
            .editor_intelligence
    );
    let collector = Collector::new();
    core.editor_start(collector.sink()).expect("start");
    wait_ready(&collector, 1);
}

#[test]
fn shutdown_and_dropping_the_core_leave_no_process_behind() {
    let fixture = Fixture::new();
    let core = fixture.core("", "");
    let collector = Collector::new();
    core.editor_start(collector.sink()).expect("start");
    wait_ready(&collector, 1);
    core.editor_docs("", "F.softmax", None).expect("docs");
    let (server, helper) = (fixture.server_pid(), fixture.lines("pids")[0].clone());

    core.shutdown();
    assert!(!process_alive(&server));
    assert!(wait_for_exit(&helper));
    assert_eq!(
        states(&collector.snapshot()).last(),
        Some(&EditorState::Stopped {
            reason: EditorStopReason::Shutdown
        })
    );

    // Without an explicit shutdown, dropping the last handle does the same.
    fs::remove_file(fixture.state().join("server.pid")).expect("remove pid file");
    let collector = Collector::new();
    core.editor_start(collector.sink()).expect("start");
    wait_ready(&collector, 1);
    core.editor_docs("", "F.softmax", None).expect("docs");
    let (server, helper) = (fixture.server_pid(), fixture.lines("pids")[1].clone());
    assert!(process_alive(&server) && process_alive(&helper));
    drop(core);
    assert!(wait_for_exit(&server));
    assert!(wait_for_exit(&helper));
    assert_well_formed(&collector.snapshot());
}

#[test]
fn docs_lookups_start_the_helper_on_first_use_and_reuse_it() {
    let fixture = Fixture::new();
    let core = fixture.core("", "");
    assert!(
        fixture.lines("pids").is_empty(),
        "nothing runs before the first lookup"
    );

    let docs = core
        .editor_docs("import torch.nn.functional as F\n", "F.softmax", None)
        .expect("docs");
    let pid = fixture.lines("pids")[0].clone();
    assert_eq!(
        json!(docs),
        json!({"found": true, "name": "fake.F.softmax", "kind": "function",
               "signature": "f()", "doc": format!("from process {pid}"),
               "truncated": false, "reason": null})
    );
    // The member form, and a source with quotes, newlines, and non-ASCII text.
    let docs = core
        .editor_docs("x = \"caf\u{e9}\"\n'''\n", "F", Some("relu"))
        .expect("docs");
    assert_eq!(docs.name.as_deref(), Some("fake.F.relu"));
    assert_eq!(
        fixture.lines("pids").len(),
        1,
        "one helper serves every lookup"
    );

    // The helper works without a language server session, and stops with `editor_stop`.
    core.editor_stop(None);
    assert!(wait_for_exit(&pid));
    assert!(core.editor_docs("", "F.gelu", None).expect("docs").found);
    assert_eq!(fixture.lines("pids").len(), 2);

    let big = "#".repeat(1024 * 1024 + 1);
    assert!(matches!(
        core.editor_docs(&big, "F.softmax", None),
        Err(CoreError::InvalidArgument(_))
    ));
}

#[test]
fn a_helper_that_crashes_hangs_or_babbles_is_harmless() {
    for scenario in ["crash", "hang", "garbage"] {
        let fixture = Fixture::new();
        let core = fixture.core("", scenario);
        let docs = core.editor_docs("", "F.softmax", None).expect(scenario);
        assert_eq!(
            json!(docs),
            json!({"found": false, "name": null, "kind": null, "signature": null,
                   "doc": null, "truncated": false, "reason": "unavailable"}),
            "{scenario}"
        );
        let pids = fixture.lines("pids");
        assert_eq!(pids.len(), 1);
        assert!(wait_for_exit(&pids[0]), "{scenario}: the helper is killed");
        // No new helper is started right away.
        let again = core.editor_docs("", "F.softmax", None).expect(scenario);
        assert_eq!(again.reason.as_deref(), Some("unavailable"));
        assert_eq!(fixture.lines("pids").len(), 1, "{scenario}");
    }

    // Stopping does not wait for a lookup that a hung helper would hold for the whole
    // timeout.
    let fixture = Fixture::new();
    let mut options = fixture.options("", "hang");
    options.docs_timeout = Duration::from_secs(30);
    let core = fixture.core_with(options);
    let lookup = {
        let core = core.clone();
        std::thread::spawn(move || core.editor_docs("", "F.softmax", None))
    };
    let pid = wait_for_file(&fixture.state().join("pids"));
    let began = std::time::Instant::now();
    core.editor_stop(None);
    assert!(began.elapsed() < Duration::from_secs(10));
    assert!(wait_for_exit(&pid));
    let docs = lookup.join().expect("lookup thread").expect("docs");
    assert_eq!(docs.reason.as_deref(), Some("unavailable"));

    // After the pause a new helper is started, and it may well work.
    let fixture = Fixture::new();
    let core = fixture.core("", "crash-once");
    assert!(!core.editor_docs("", "F.softmax", None).expect("docs").found);
    std::thread::sleep(Duration::from_millis(350));
    let docs = core.editor_docs("", "F.softmax", None).expect("docs");
    assert!(docs.found, "{docs:?}");
    assert_eq!(fixture.lines("pids").len(), 2);
}
