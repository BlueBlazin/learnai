//! Editor intelligence against the real language server (ty) and the real docs helper,
//! in the repository's own Python environment. Skipped when ty is not installed there
//! (`uv sync --project python --extra editor`, plus the extras already in use).
//!
//! The content root is a temporary one; only the virtual environment is the real one.

mod common;

use std::fs;
use std::path::{Path, PathBuf};

use common::{Collector, TestRoot, fixtures};
use learnai_core::{Core, CoreOptions, EditorEvent, EditorOptions, EditorState};
use serde_json::{Value, json};

/// The repository's virtual environment, if it holds ty.
fn real_venv() -> Option<PathBuf> {
    let venv = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../python/.venv");
    let venv = fs::canonicalize(venv).ok()?;
    venv.join("bin/ty").is_file().then_some(venv)
}

struct Client {
    core: Core,
    session: String,
    collector: Collector<EditorEvent>,
    next_id: u64,
}

impl Client {
    fn messages(events: &[EditorEvent]) -> Vec<Value> {
        events
            .iter()
            .filter_map(|event| match event {
                EditorEvent::Message { message } => serde_json::from_str(message).ok(),
                EditorEvent::State { .. } => None,
            })
            .collect()
    }

    fn wait(&self, what: &str, found: impl Fn(&Value) -> bool) -> Value {
        let events = self
            .collector
            .wait_for(what, |events| Self::messages(events).iter().any(&found));
        Self::messages(&events)
            .into_iter()
            .rev()
            .find(|m| found(m))
            .expect("message")
    }

    fn notify(&self, method: &str, params: Value) {
        let message = json!({"jsonrpc": "2.0", "method": method, "params": params});
        self.core
            .editor_send(&self.session, &message.to_string())
            .expect("send");
    }

    fn request(&mut self, method: &str, params: Value) -> Value {
        self.next_id += 1;
        let id = self.next_id;
        let message = json!({"jsonrpc": "2.0", "id": id, "method": method, "params": params});
        self.core
            .editor_send(&self.session, &message.to_string())
            .expect("send");
        let reply = self.wait(method, |m| m["id"] == id && m.get("method").is_none());
        assert!(reply.get("error").is_none(), "{method}: {reply}");
        reply["result"].clone()
    }

    /// The latest pushed diagnostics for `uri` that satisfy `found`.
    fn diagnostics(&self, uri: &str, found: impl Fn(&[Value]) -> bool) -> Vec<Value> {
        let published = self.wait("pushed diagnostics", |m| {
            m["method"] == "textDocument/publishDiagnostics"
                && m["params"]["uri"] == uri
                && m["params"]["diagnostics"]
                    .as_array()
                    .is_some_and(|d| found(d))
        });
        published["params"]["diagnostics"]
            .as_array()
            .expect("diagnostics")
            .clone()
    }
}

#[test]
fn ty_completes_hovers_and_pushes_diagnostics_for_a_file_named_torch_py() {
    let Some(venv) = real_venv() else {
        eprintln!("skipped: ty is not installed in python/.venv");
        return;
    };
    let root = TestRoot::new();
    // A reference solution next to the workspace, and another saved solution in it.
    let reference = root.path().join("problems/softmax/reference");
    fs::create_dir_all(&reference).expect("create directory");
    fs::write(
        reference.join("torch.py"),
        "def reference_only_secret_function():\n    return 1\n",
    )
    .expect("write reference");
    let other = root.path().join("workspace/other");
    fs::create_dir_all(&other).expect("create directory");
    fs::write(
        other.join("torch.py"),
        "def learner_visible_function():\n    return 1\n",
    )
    .expect("write solution");

    let core = Core::open(CoreOptions {
        uv_program: Some(fixtures().join("fake_uv.sh")),
        editor: EditorOptions {
            venv: Some(venv),
            ..EditorOptions::default()
        },
        ..CoreOptions::new(root.path())
    })
    .expect("open core");
    let status = core.editor_status().expect("status");
    assert!(status.installed);
    let version = status.server_version.expect("ty reports its version");
    assert!(version.starts_with("0."), "{version}");

    let collector = Collector::new();
    let session = core.editor_start(collector.sink()).expect("start");
    let events = collector.wait_for("the ready state", |events| {
        events.iter().any(|e| {
            matches!(
                e,
                EditorEvent::State {
                    state: EditorState::Ready { .. }
                }
            )
        })
    });
    let Some(EditorEvent::State {
        state:
            EditorState::Ready {
                server_name,
                server_version,
                root_uri,
                ..
            },
    }) = events.last().cloned()
    else {
        panic!("ready is the latest event: {events:?}");
    };
    assert_eq!(server_name, "ty");
    assert_eq!(server_version.as_deref(), Some(version.as_str()));
    let mut client = Client {
        core,
        session,
        collector,
        next_id: 0,
    };

    // What @codemirror/lsp-client 6.3.0 sends: pull diagnostics advertised (and never
    // pulled), and a root the client chose. Core rewrites both.
    let initialized = client.request(
        "initialize",
        json!({
            "processId": null,
            "clientInfo": {"name": "@codemirror/lsp-client"},
            "rootUri": root_uri,
            "capabilities": {
                "general": {"markdown": {"parser": "marked"}},
                "textDocument": {
                    "completion": {
                        "completionItem": {
                            "snippetSupport": true,
                            "documentationFormat": ["markdown", "plaintext"],
                            "insertReplaceSupport": false
                        },
                        "completionList": {
                            "itemDefaults": ["commitCharacters", "editRange", "insertTextFormat"]
                        },
                        "completionItemKind": {"valueSet": []},
                        "contextSupport": true
                    },
                    "hover": {"contentFormat": ["markdown", "plaintext"]},
                    "signatureHelp": {
                        "contextSupport": true,
                        "signatureInformation": {
                            "documentationFormat": ["markdown", "plaintext"],
                            "parameterInformation": {"labelOffsetSupport": true},
                            "activeParameterSupport": true
                        }
                    },
                    "diagnostic": {},
                    "publishDiagnostics": {"versionSupport": true}
                },
                "window": {"showMessage": {}}
            }
        }),
    );
    assert_eq!(initialized["serverInfo"]["name"], "ty");
    client.notify("initialized", json!({}));

    // The solution file is named after its framework and is not on disk yet.
    let uri = format!("{root_uri}/softmax/torch.py");
    client.notify(
        "textDocument/didOpen",
        json!({"textDocument": {"uri": uri, "languageId": "python", "version": 1,
               "text": "import torch\ntorch.ex"}}),
    );
    let completion = client.request(
        "textDocument/completion",
        json!({"textDocument": {"uri": uri}, "position": {"line": 1, "character": 8}}),
    );
    let items = completion
        .get("items")
        .unwrap_or(&completion)
        .as_array()
        .expect("completion items")
        .clone();
    let labels: Vec<&str> = items.iter().filter_map(|i| i["label"].as_str()).collect();
    assert!(
        labels.contains(&"exp"),
        "the real torch is completed: {labels:?}"
    );

    // A planted mistake is reported by push; a rule outside the allowlist is not.
    client.notify(
        "textDocument/didChange",
        json!({"textDocument": {"uri": uri, "version": 2},
               "contentChanges": [{"text":
                   "import torch\ntorch.exp\ny = torch.exxp(1)\nq: int = 'not an int'\n"}]}),
    );
    let hover = client.request(
        "textDocument/hover",
        json!({"textDocument": {"uri": uri}, "position": {"line": 1, "character": 7}}),
    );
    let text = hover["contents"]["value"].as_str().expect("markdown hover");
    assert!(text.contains("def exp("), "a signature: {text}");
    assert!(
        text.contains("exponential of the elements"),
        "a docstring: {text}"
    );
    let diagnostics = client.diagnostics(&uri, |d| !d.is_empty());
    let codes: Vec<&str> = diagnostics
        .iter()
        .filter_map(|d| d["code"].as_str())
        .collect();
    assert_eq!(codes, ["missing-attribute"], "{diagnostics:?}");
    assert!(
        diagnostics[0]["message"]
            .as_str()
            .is_some_and(|m| m.contains("exxp")),
        "{diagnostics:?}"
    );
    assert_eq!(diagnostics[0]["range"]["start"]["line"], 2);

    // Only the workspace is indexed: the reference solution is unknown to the server.
    let symbols = |client: &mut Client, query: &str| -> usize {
        client
            .request("workspace/symbol", json!({"query": query}))
            .as_array()
            .map_or(0, Vec::len)
    };
    assert_eq!(symbols(&mut client, "learner_visible_function"), 1);
    assert_eq!(symbols(&mut client, "reference_only_secret_function"), 0);

    // The runtime docs helper fills the gap ty leaves for runtime-only docstrings.
    let source = "import torch\nimport torch.nn.functional as F\nx = F.softmax(\n";
    let docs = client
        .core
        .editor_docs(source, "F.scaled_dot_product_attention", None)
        .expect("docs");
    assert!(docs.found, "{docs:?}");
    assert_eq!(
        docs.name.as_deref(),
        Some("torch.nn.functional.scaled_dot_product_attention")
    );
    assert!(
        docs.signature
            .as_deref()
            .is_some_and(|s| s.starts_with("scaled_dot_product_attention(query, key, value")),
        "{docs:?}"
    );
    assert!(
        docs.doc
            .as_deref()
            .is_some_and(|d| d.starts_with("Computes scaled dot product attention")),
        "{docs:?}"
    );
    let member = client
        .core
        .editor_docs(source, "torch.linalg", Some("svd"))
        .expect("docs");
    assert_eq!(member.name.as_deref(), Some("torch.linalg.svd"));
    assert!(member.doc.is_some());

    client.core.editor_stop(None);
    let events = client.collector.snapshot();
    assert!(
        events.last().is_some_and(EditorEvent::is_last),
        "{events:?}"
    );
}
