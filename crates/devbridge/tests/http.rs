//! Drives the bridge over real HTTP on an ephemeral port, with the fake harness and
//! fake uv from the core fixtures.

use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use reqwest::{Client, StatusCode};
use serde_json::{Value, json};
use tempfile::TempDir;
use tokio::net::TcpListener;
use whetstone_core::{Core, CoreOptions, ProgramHarness};

fn fixtures() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../core/tests/fixtures")
}

fn copy_tree(from: &Path, to: &Path) {
    fs::create_dir_all(to).expect("create directory");
    for entry in fs::read_dir(from).expect("read directory") {
        let entry = entry.expect("directory entry");
        let target = to.join(entry.file_name());
        if entry.path().is_dir() {
            copy_tree(&entry.path(), &target);
        } else {
            fs::copy(entry.path(), &target).expect("copy file");
        }
    }
}

/// A bridge serving a private copy of the fixture root.
struct Bridge {
    base: String,
    client: Client,
    root: TempDir,
}

impl Bridge {
    /// Starts a bridge whose fake harness picks its scenario from the solution file.
    async fn start() -> Self {
        let root = tempfile::tempdir().expect("temp dir");
        copy_tree(&fixtures().join("root"), root.path());
        let core = Core::open(CoreOptions {
            harness: Some(Arc::new(
                ProgramHarness::new("sh").arg(fixtures().join("fake_harness.sh")),
            )),
            uv_program: Some(fixtures().join("fake_uv.sh")),
            ..CoreOptions::new(root.path())
        })
        .expect("open core");

        let listener = TcpListener::bind("127.0.0.1:0").await.expect("bind");
        let base = format!("http://{}", listener.local_addr().expect("address"));
        let origins = vec!["http://localhost:1420".to_owned()];
        tokio::spawn(async move { whetstone_devbridge::serve(listener, core, &origins).await });
        Self {
            base,
            client: Client::new(),
            root,
        }
    }

    async fn post(&self, command: &str, body: Value) -> (StatusCode, Value) {
        let response = self
            .client
            .post(format!("{}/api/{command}", self.base))
            .json(&body)
            .send()
            .await
            .expect("request");
        let status = response.status();
        (status, response.json().await.expect("JSON body"))
    }

    async fn call(&self, command: &str, body: Value) -> Value {
        let (status, value) = self.post(command, body).await;
        assert_eq!(status, StatusCode::OK, "{command}: {value}");
        value
    }

    /// Reads a server-sent event stream until the server closes it.
    async fn events(&self, path: &str) -> Vec<Value> {
        let response = self
            .client
            .get(format!("{}{path}", self.base))
            .send()
            .await
            .expect("request");
        assert_eq!(response.status(), StatusCode::OK);
        assert_eq!(response.headers()["content-type"], "text/event-stream");
        let body = response.text().await.expect("stream body");
        body.lines()
            .filter_map(|line| line.strip_prefix("data: "))
            .map(|data| serde_json::from_str(data).expect("event JSON"))
            .collect()
    }
}

fn types(events: &[Value]) -> Vec<&str> {
    events
        .iter()
        .map(|e| e["type"].as_str().expect("type"))
        .collect()
}

#[tokio::test(flavor = "multi_thread")]
async fn full_run_over_http_and_sse() {
    let bridge = Bridge::start().await;

    // Catalog.
    let tracks = bridge.call("list_tracks", json!({})).await;
    assert_eq!(tracks[0]["id"], "basics");
    assert_eq!(tracks[0]["problemCount"], 2);
    let problems = bridge.call("list_problems", json!({})).await;
    assert_eq!(problems.as_array().expect("array").len(), 3);
    assert_eq!(problems[0]["id"], "add-two");
    assert_eq!(problems[0]["status"], "todo");
    let detail = bridge.call("get_problem", json!({"id": "add-two"})).await;
    assert_eq!(detail["hints"].as_array().expect("hints").len(), 3);
    assert_eq!(detail["editorial"], Value::Null);

    // Solution round trip.
    let starter = bridge
        .call(
            "read_solution",
            json!({"id": "add-two", "framework": "torch"}),
        )
        .await;
    assert_eq!(starter["isStarter"], true);
    let code = "def add(a, b):\n    return a + b\n";
    let written = bridge
        .call(
            "write_solution",
            json!({"id": "add-two", "framework": "torch", "code": code}),
        )
        .await;
    assert_eq!(written, Value::Null);
    let saved = bridge
        .call(
            "read_solution",
            json!({"id": "add-two", "framework": "torch"}),
        )
        .await;
    assert_eq!(
        saved,
        json!({"code": code, "isStarter": false, "path": saved["path"]})
    );
    assert_eq!(
        fs::read_to_string(saved["path"].as_str().expect("path")).expect("file"),
        code
    );

    // Submit and follow the event stream to its end.
    let started = bridge
        .call("start_run", json!({"problemId": "add-two", "framework": "torch", "mode": "submit", "target": "local"}))
        .await;
    let run_id = started["runId"].as_str().expect("runId");
    let events = bridge.events(&format!("/api/runs/{run_id}/events")).await;
    let kinds = types(&events);
    assert_eq!(kinds[0], "preparing");
    let rest: Vec<_> = kinds
        .iter()
        .copied()
        .filter(|k| *k != "preparing")
        .collect();
    assert_eq!(
        rest,
        [
            "start",
            "caseStart",
            "caseEnd",
            "caseStart",
            "caseEnd",
            "caseStart",
            "caseEnd",
            "end"
        ]
    );
    let end = events.last().expect("end");
    assert_eq!(end["verdict"], "accepted");
    assert_eq!(
        (end["passed"].as_u64(), end["total"].as_u64()),
        (Some(3), Some(3))
    );
    assert_eq!(end["durationMs"], 42.0);
    assert_eq!(end["message"], Value::Null);
    assert_eq!(end["detail"], Value::Null);
    let submission_id = end["submissionId"].as_i64().expect("submissionId");
    let case_end = events
        .iter()
        .find(|e| e["type"] == "caseEnd")
        .expect("caseEnd");
    assert_eq!(
        case_end["result"],
        json!({
            "id": "c1", "status": "pass", "durationMs": 1.5, "message": null, "detail": null,
            "stdout": "", "metrics": []
        })
    );

    // A late subscriber gets the same stream replayed from the first event.
    assert_eq!(
        bridge.events(&format!("/api/runs/{run_id}/events")).await,
        events
    );

    // The submission and the progress it implies.
    let submissions = bridge
        .call("list_submissions", json!({"problemId": "add-two"}))
        .await;
    assert_eq!(submissions[0]["id"], submission_id);
    assert_eq!(
        bridge
            .call("list_submissions", json!({}))
            .await
            .as_array()
            .expect("array")
            .len(),
        1
    );
    let submission = bridge
        .call("get_submission", json!({"id": submission_id}))
        .await;
    assert_eq!(submission["code"], code);
    assert_eq!(submission["cases"].as_array().expect("cases").len(), 3);
    assert_eq!(
        submission["cases"][2],
        json!({
            "id": "h1", "status": "pass", "durationMs": 1.5, "message": null, "detail": null,
            "stdout": "", "metrics": [], "name": "Hidden case 1", "public": false
        })
    );
    let detail = bridge.call("get_problem", json!({"id": "add-two"})).await;
    assert_eq!(detail["status"], "solved");
    assert_eq!(detail["solvedFrameworks"], json!(["torch"]));
    assert_eq!(detail["editorialUnlocked"], true);
    let reference = bridge
        .call(
            "get_reference",
            json!({"id": "add-two", "framework": "torch"}),
        )
        .await;
    assert_eq!(reference, "def add(a, b):\n    return a + b\n");
    let stats = bridge.call("get_stats", json!({})).await;
    assert_eq!(
        stats,
        json!({
            "solved": 1, "total": 3, "streakDays": 1,
            "byDifficulty": {"easy": [1, 1], "medium": [0, 1], "hard": [0, 1]}
        })
    );

    // Malformed problems in the fixture catalog are listed, as path and message.
    let issues = bridge.call("list_catalog_issues", json!({})).await;
    let issues = issues.as_array().expect("issues");
    assert!(!issues.is_empty());
    assert!(issues.iter().all(|i| {
        i.as_object().is_some_and(|o| o.len() == 2)
            && i["path"].is_string()
            && i["message"].is_string()
    }));

    // Reset.
    let reset = bridge
        .call(
            "reset_solution",
            json!({"id": "add-two", "framework": "torch"}),
        )
        .await;
    assert_eq!(reset["isStarter"], true);
}

#[tokio::test(flavor = "multi_thread")]
async fn failing_run_and_cancel_over_http() {
    let bridge = Bridge::start().await;
    let run = |code: &'static str| {
        let bridge = &bridge;
        async move {
            bridge
                .call(
                    "write_solution",
                    json!({"id": "add-two", "framework": "jax", "code": code}),
                )
                .await;
            let started = bridge
                .call("start_run", json!({"problemId": "add-two", "framework": "jax", "mode": "run", "target": "local"}))
                .await;
            started["runId"].as_str().expect("runId").to_owned()
        }
    };

    let failing = run("# scenario: fail\n").await;
    let events = bridge.events(&format!("/api/runs/{failing}/events")).await;
    let end = events.last().expect("end");
    assert_eq!(end["verdict"], "wrong_answer");
    assert_eq!(end["submissionId"], Value::Null);
    let failed = events
        .iter()
        .find(|e| e["type"] == "caseEnd" && e["result"]["status"] == "fail")
        .expect("fail");
    assert_eq!(
        failed["result"]["metrics"],
        json!([{"name": "abs_err", "value": 1.0, "limit": 0.001}])
    );

    let hanging = run("# scenario: hang\n").await;
    let canceled = bridge.call("cancel_run", json!({"runId": hanging})).await;
    assert_eq!(canceled, Value::Null);
    let events = bridge.events(&format!("/api/runs/{hanging}/events")).await;
    assert_eq!(events.last().expect("end")["verdict"], "canceled");
    assert_eq!(types(&events).iter().filter(|k| **k == "end").count(), 1);
}

#[tokio::test(flavor = "multi_thread")]
async fn environment_status_and_install_job() {
    let bridge = Bridge::start().await;
    let status = bridge.call("env_status", json!({})).await;
    assert_eq!(status["uv"], "0.13.8");
    assert_eq!(status["python"], "3.15.2");
    assert_eq!(
        status["frameworks"][1],
        json!({"framework": "jax", "installed": false, "version": null})
    );
    assert_eq!(
        status["root"],
        bridge.root.path().to_string_lossy().as_ref()
    );

    let job = bridge
        .call("env_install", json!({"framework": "jax"}))
        .await;
    let job_id = job["jobId"].as_str().expect("jobId");
    let events = bridge.events(&format!("/api/jobs/{job_id}/events")).await;
    assert!(events.len() >= 2);
    assert!(
        events[..events.len() - 1]
            .iter()
            .all(|e| e["type"] == "log" && e["text"].is_string())
    );
    assert_eq!(
        events.last(),
        Some(&json!({"type": "done", "ok": true, "message": null}))
    );
    assert_eq!(
        bridge.events(&format!("/api/jobs/{job_id}/events")).await,
        events,
        "replay"
    );

    let status = bridge.call("env_status", json!({})).await;
    assert_eq!(
        status["frameworks"][1],
        json!({"framework": "jax", "installed": true, "version": "0.12.0.1"})
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn errors_are_json_with_meaningful_statuses() {
    let bridge = Bridge::start().await;
    let cases = [
        (
            "get_problem",
            json!({"id": "nope"}),
            StatusCode::NOT_FOUND,
            "unknown problem `nope`",
        ),
        (
            "no_such_command",
            json!({}),
            StatusCode::NOT_FOUND,
            "unknown command `no_such_command`",
        ),
        (
            "get_submission",
            json!({"id": 12345}),
            StatusCode::NOT_FOUND,
            "unknown submission 12345",
        ),
        (
            "get_reference",
            json!({"id": "add-two", "framework": "torch"}),
            StatusCode::FORBIDDEN,
            "locked",
        ),
        (
            "read_solution",
            json!({"id": "add-two", "framework": "mlx"}),
            StatusCode::BAD_REQUEST,
            "does not support",
        ),
        (
            "read_solution",
            json!({"id": "add-two", "framework": "cobol"}),
            StatusCode::BAD_REQUEST,
            "invalid arguments",
        ),
        (
            "read_solution",
            json!({"id": "add-two"}),
            StatusCode::BAD_REQUEST,
            "missing field `framework`",
        ),
        (
            "start_run",
            json!({"problemId": "add-two", "framework": "torch", "mode": "run", "target": "moon"}),
            StatusCode::NOT_FOUND,
            "unknown target `moon`",
        ),
    ];
    for (command, body, status, needle) in cases {
        let (got, value) = bridge.post(command, body).await;
        assert_eq!(got, status, "{command}: {value}");
        let message = value["error"]
            .as_str()
            .unwrap_or_else(|| panic!("{command}: no error in {value}"));
        assert!(message.contains(needle), "{command}: {message}");
    }

    // Malformed JSON and an empty body.
    let raw = |body: &'static str| {
        bridge
            .client
            .post(format!("{}/api/list_tracks", bridge.base))
            .body(body)
            .send()
    };
    let response = raw("{not json").await.expect("request");
    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    assert!(response.json::<Value>().await.expect("json")["error"].is_string());
    assert_eq!(raw("").await.expect("request").status(), StatusCode::OK);

    // Unknown streams and paths.
    for path in [
        "/api/runs/nope/events",
        "/api/jobs/nope/events",
        "/elsewhere",
    ] {
        let response = bridge
            .client
            .get(format!("{}{path}", bridge.base))
            .send()
            .await
            .expect("request");
        assert_eq!(response.status(), StatusCode::NOT_FOUND, "{path}");
        assert!(response.json::<Value>().await.expect("json")["error"].is_string());
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn cors_allows_only_the_dev_origin() {
    let bridge = Bridge::start().await;
    let preflight = |origin: &'static str| {
        bridge
            .client
            .request(
                reqwest::Method::OPTIONS,
                format!("{}/api/list_tracks", bridge.base),
            )
            .header("origin", origin)
            .header("access-control-request-method", "POST")
            .header("access-control-request-headers", "content-type")
            .send()
    };
    let allowed = preflight("http://localhost:1420").await.expect("request");
    assert_eq!(allowed.status(), StatusCode::OK);
    assert_eq!(
        allowed.headers()["access-control-allow-origin"],
        "http://localhost:1420"
    );
    assert!(
        allowed.headers()["access-control-allow-headers"]
            .to_str()
            .expect("header")
            .contains("content-type")
    );

    let denied = preflight("http://evil.example").await.expect("request");
    assert!(
        denied
            .headers()
            .get("access-control-allow-origin")
            .is_none()
    );

    let response = bridge
        .client
        .post(format!("{}/api/list_tracks", bridge.base))
        .header("origin", "http://localhost:1420")
        .json(&json!({}))
        .send()
        .await
        .expect("request");
    assert_eq!(
        response.headers()["access-control-allow-origin"],
        "http://localhost:1420"
    );
}
