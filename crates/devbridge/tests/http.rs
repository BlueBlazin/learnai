//! Drives the bridge over real HTTP on an ephemeral port, with the fake harness and
//! fake uv from the core fixtures.

use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use learnai_core::{Core, CoreOptions, ProgramHarness, SshOptions};
use reqwest::{Client, StatusCode};
use serde_json::{Value, json};
use tempfile::TempDir;
use tokio::net::TcpListener;

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
            // A private SSH configuration, and an ssh that does not exist: the registry
            // is exercised without ever reaching a host.
            ssh: SshOptions {
                ssh_program: Some(root.path().join("no-ssh")),
                rsync_program: Some(root.path().join("no-rsync")),
                config: Some(root.path().join("ssh_config")),
                control_dir: Some(root.path().join("ctl")),
            },
            ..CoreOptions::new(root.path())
        })
        .expect("open core");
        std::fs::write(
            root.path().join("ssh_config"),
            "Host gpu-box\n  HostName 10.0.0.5\n  User ada\nHost *\n  User nobody\n",
        )
        .expect("write ssh config");

        let listener = TcpListener::bind("127.0.0.1:0").await.expect("bind");
        let base = format!("http://{}", listener.local_addr().expect("address"));
        let origins = vec!["http://localhost:1420".to_owned()];
        tokio::spawn(async move { learnai_devbridge::serve(listener, core, &origins).await });
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
            "solved": 1, "total": 3, "streakDays": 1, "reviewsDue": 0,
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
async fn paths_and_assets_over_http() {
    let bridge = Bridge::start().await;
    let root = bridge.root.path();

    // No paths directory: an empty list, not an error.
    assert_eq!(bridge.call("list_paths", json!({})).await, json!([]));

    fs::create_dir_all(root.join("paths")).expect("mkdir");
    fs::write(
        root.join("paths/starter.toml"),
        "id = \"starter\"\ntitle = \"Starter Path\"\ndescription = \"A short path.\"\n\n\
         [[stages]]\ntitle = \"Warm-up\"\ndescription = \"First steps.\"\n\
         problems = [\"add-two\", \"missing\", \"hard-one\"]\n",
    )
    .expect("write");
    assert_eq!(
        bridge.call("list_paths", json!({})).await,
        json!([{
            "id": "starter", "title": "Starter Path", "description": "A short path.",
            "stages": [{"title": "Warm-up", "description": "First steps.",
                        "problems": ["add-two", "hard-one"]}],
            "problemCount": 2, "solvedCount": 0
        }])
    );
    let issues = bridge.call("list_catalog_issues", json!({})).await;
    assert!(issues.as_array().expect("array").iter().any(|i| {
        i["path"]
            .as_str()
            .is_some_and(|p| p.ends_with("starter.toml"))
            && i["message"]
                .as_str()
                .is_some_and(|m| m.contains("`missing`"))
    }));

    // Progress follows an accepted submit.
    let started = bridge
        .call("start_run", json!({"problemId": "add-two", "framework": "torch", "mode": "submit", "target": "local"}))
        .await;
    let run_id = started["runId"].as_str().expect("runId");
    bridge.events(&format!("/api/runs/{run_id}/events")).await;
    assert_eq!(
        bridge.call("list_paths", json!({})).await[0]["solvedCount"],
        1
    );

    // Assets.
    let assets = root.join("problems/basics/add-two/assets");
    fs::create_dir_all(&assets).expect("mkdir");
    fs::write(assets.join("figure.svg"), "<svg/>").expect("write");
    fs::write(assets.join("notes.txt"), "x").expect("write");
    fs::write(assets.join("huge.png"), vec![0u8; 1024 * 1024 + 1]).expect("write");
    assert_eq!(
        bridge
            .call(
                "get_problem_asset",
                json!({"id": "add-two", "path": "figure.svg"})
            )
            .await,
        json!("data:image/svg+xml;base64,PHN2Zy8+")
    );
    let cases = [
        (
            json!({"id": "add-two", "path": "../statement.md"}),
            StatusCode::BAD_REQUEST,
            "must stay inside",
        ),
        (
            json!({"id": "add-two", "path": "notes.txt"}),
            StatusCode::BAD_REQUEST,
            "only .svg and .png",
        ),
        (
            json!({"id": "add-two", "path": "huge.png"}),
            StatusCode::BAD_REQUEST,
            "the limit is",
        ),
        (
            json!({"id": "add-two", "path": "nope.svg"}),
            StatusCode::NOT_FOUND,
            "missing file",
        ),
        (
            json!({"id": "nope", "path": "figure.svg"}),
            StatusCode::NOT_FOUND,
            "unknown problem `nope`",
        ),
        (
            json!({"id": "add-two"}),
            StatusCode::BAD_REQUEST,
            "missing field `path`",
        ),
    ];
    for (body, status, needle) in cases {
        let (got, value) = bridge.post("get_problem_asset", body.clone()).await;
        assert_eq!(got, status, "{body}: {value}");
        let message = value["error"].as_str().expect("error message");
        assert!(message.contains(needle), "{body}: {message}");
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

#[tokio::test(flavor = "multi_thread")]
async fn model_assets_over_http() {
    let bridge = Bridge::start().await;

    let listing = bridge.call("list_assets", json!({})).await;
    let models_dir = bridge.root.path().join(".learnai/models");
    assert_eq!(listing["modelsDir"], models_dir.to_string_lossy().as_ref());
    assert_eq!(
        listing["assets"][1],
        json!({
            "name": "tiny", "repo": "fake/tiny",
            "revision": "0123456789abcdef0123456789abcdef01234567",
            "files": ["config.json"], "sizeMb": 0.5, "state": "missing", "reason": null,
            "path": models_dir.join("tiny").to_string_lossy(), "bytesOnDisk": 0,
            "problems": ["add-two"]
        })
    );
    let problem = bridge.call("get_problem", json!({"id": "add-two"})).await;
    assert_eq!(problem["assets"], json!(["tiny"]));

    // A fetch returns its job id at once; the events replay on the jobs endpoint.
    let job = bridge
        .call("fetch_assets", json!({"names": ["tiny"]}))
        .await;
    let job_id = job["jobId"].as_str().expect("jobId");
    let events = bridge.events(&format!("/api/jobs/{job_id}/events")).await;
    assert_eq!(events[0]["type"], "preparing");
    let protocol: Vec<&Value> = events
        .iter()
        .filter(|e| e["type"] != "log" && e["type"] != "preparing")
        .collect();
    assert_eq!(
        protocol,
        [
            &json!({"type": "start", "asset": "tiny", "totalBytes": 2048}),
            &json!({"type": "progress", "asset": "tiny", "file": "config.json",
                    "doneBytes": 1024, "totalBytes": 2048}),
            &json!({"type": "progress", "asset": "tiny", "file": "config.json",
                    "doneBytes": 2048, "totalBytes": 2048}),
            &json!({"type": "assetEnd", "asset": "tiny", "ok": true, "message": null}),
            &json!({"type": "done", "ok": true, "canceled": false, "message": null}),
        ]
    );
    assert_eq!(
        bridge.events(&format!("/api/jobs/{job_id}/events")).await,
        events,
        "replay"
    );
    // Canceling a finished or unknown job is a no-op.
    assert_eq!(
        bridge
            .call("cancel_asset_fetch", json!({"jobId": job_id}))
            .await,
        Value::Null
    );
    assert_eq!(
        bridge
            .call("cancel_asset_fetch", json!({"jobId": "nope"}))
            .await,
        Value::Null
    );

    let listing = bridge.call("list_assets", json!({})).await;
    assert_eq!(listing["assets"][1]["state"], "present");

    // Bad requests are 400s and delete or start nothing.
    for (command, body) in [
        ("fetch_assets", json!({"names": []})),
        ("fetch_assets", json!({"names": ["--all"]})),
        ("fetch_assets", json!({})),
        ("remove_asset", json!({"name": "../state.db"})),
        ("remove_asset", json!({"name": "state.db"})),
        ("cancel_asset_fetch", json!({})),
    ] {
        let (status, value) = bridge.post(command, body.clone()).await;
        assert_eq!(status, StatusCode::BAD_REQUEST, "{command} {body}: {value}");
        assert!(value["error"].is_string());
    }
    assert!(models_dir.join("tiny").is_dir());

    let listing = bridge.call("remove_asset", json!({"name": "tiny"})).await;
    assert_eq!(listing["assets"][1]["state"], "missing");
    assert!(!models_dir.join("tiny").exists());

    // One fetch at a time: a second one is a 409 while the first hangs.
    fs::write(bridge.root.path().join("python/.fetch-scenario"), "hang").expect("write");
    let job = bridge
        .call("fetch_assets", json!({"names": ["tiny"]}))
        .await;
    let job_id = job["jobId"].as_str().expect("jobId").to_owned();
    let (status, value) = bridge
        .post("fetch_assets", json!({"names": ["other"]}))
        .await;
    assert_eq!(status, StatusCode::CONFLICT, "{value}");
    let (status, _) = bridge.post("remove_asset", json!({"name": "tiny"})).await;
    assert_eq!(status, StatusCode::CONFLICT);
    bridge
        .call("cancel_asset_fetch", json!({"jobId": job_id}))
        .await;
    let events = bridge.events(&format!("/api/jobs/{job_id}/events")).await;
    assert_eq!(
        events.last(),
        Some(&json!({"type": "done", "ok": false, "canceled": true, "message": null}))
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn a_not_judged_submit_over_http() {
    let bridge = Bridge::start().await;
    bridge
        .call(
            "write_solution",
            json!({"id": "add-two", "framework": "torch", "code": "# scenario: assets-missing\n"}),
        )
        .await;
    let run = bridge
        .call(
            "start_run",
            json!({"problemId": "add-two", "framework": "torch", "mode": "submit", "target": "local"}),
        )
        .await;
    let run_id = run["runId"].as_str().expect("runId");
    let events = bridge.events(&format!("/api/runs/{run_id}/events")).await;
    let start = events.iter().find(|e| e["type"] == "start").expect("start");
    assert_eq!(start["requires"], "cpu");
    assert_eq!(start["requiresMet"], true);
    let skipped = events
        .iter()
        .find(|e| e["type"] == "caseEnd" && e["result"]["status"] == "skipped")
        .expect("skipped case");
    assert_eq!(
        skipped["result"]["skip"],
        json!({"kind": "asset", "asset": "tiny"})
    );
    let end = events.last().expect("end");
    assert_eq!(end["verdict"], "assets_missing");
    assert_eq!(end["skipped"], 1);
    assert_eq!(
        end["missingAssets"],
        json!([{"name": "tiny", "repo": "fake/tiny", "sizeMb": 0.5}])
    );
    assert_eq!(end["metrics"][0]["higherIsBetter"], true);
    assert!(end.get("missingHardware").is_none());

    let submission = bridge
        .call("get_submission", json!({"id": end["submissionId"]}))
        .await;
    assert_eq!(submission["verdict"], "assets_missing");
    assert_eq!(submission["skipped"], 1);
    let problem = bridge.call("get_problem", json!({"id": "add-two"})).await;
    assert_eq!(problem["status"], "attempted");
    assert_eq!(bridge.call("get_stats", json!({})).await["solved"], 0);
}

#[tokio::test(flavor = "multi_thread")]
async fn reviews_and_case_stages_over_http() {
    let bridge = Bridge::start().await;
    let submit = async |code: &str, review: bool| {
        let command = if review {
            "write_review_solution"
        } else {
            "write_solution"
        };
        bridge
            .call(
                command,
                json!({"id": "add-two", "framework": "torch", "code": code}),
            )
            .await;
        let run = bridge
            .call(
                "start_run",
                json!({"problemId": "add-two", "framework": "torch", "mode": "submit",
                       "target": "local", "review": review}),
            )
            .await;
        let run_id = run["runId"].as_str().expect("runId").to_owned();
        bridge.events(&format!("/api/runs/{run_id}/events")).await
    };

    assert_eq!(
        bridge.call("get_settings", json!({})).await,
        json!({"reviewsEnabled": true})
    );
    assert_eq!(bridge.call("list_reviews", json!({})).await, json!([]));
    // Without a schedule there is nothing to review.
    let (status, body) = bridge
        .post(
            "start_review",
            json!({"id": "add-two", "framework": "torch"}),
        )
        .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert_eq!(body["error"], "no review is scheduled for `add-two`");

    // A staged, failing submit: stages arrive with the announcement and are stored.
    let events = submit("# scenario: staged\n", false).await;
    let start = events.iter().find(|e| e["type"] == "start").expect("start");
    assert_eq!(start["cases"][0]["stage"], "1: Tokenizer");
    assert_eq!(start["cases"][2]["stage"], "Stage 2: Sampler");
    let end = events.last().expect("end");
    let stored = bridge
        .call("get_submission", json!({"id": end["submissionId"]}))
        .await;
    assert_eq!(stored["cases"][1]["stage"], "1: Tokenizer");
    assert_eq!(stored["review"], false);

    // Solving the problem schedules its first review, three days out.
    let events = submit("# scenario: happy\n", false).await;
    assert_eq!(events.last().expect("end")["verdict"], "accepted");
    let reviews = bridge.call("list_reviews", json!({})).await;
    assert_eq!(reviews.as_array().expect("array").len(), 1);
    assert_eq!(reviews[0]["problemId"], "add-two");
    assert_eq!(reviews[0]["stage"], 0);
    assert_eq!(reviews[0]["intervalDays"], 3);
    assert_eq!(reviews[0]["due"], false);
    assert_eq!(reviews[0]["lastReviewedAt"], Value::Null);
    assert_eq!(bridge.call("get_stats", json!({})).await["reviewsDue"], 0);

    // The review file starts as the starter and is separate from the saved solution.
    let blank = bridge
        .call(
            "read_review_solution",
            json!({"id": "add-two", "framework": "torch"}),
        )
        .await;
    assert_eq!(blank["isStarter"], true);
    assert!(
        blank["path"]
            .as_str()
            .expect("path")
            .ends_with("workspace/add-two/review/torch.py")
    );

    // A failing attempt is judged (the saved solution would pass) and stored as a review.
    let events = submit("# scenario: fail\n", true).await;
    let end = events.last().expect("end");
    assert_eq!(end["verdict"], "wrong_answer");
    let stored = bridge
        .call("get_submission", json!({"id": end["submissionId"]}))
        .await;
    assert_eq!(stored["review"], true);
    assert_eq!(stored["code"], "# scenario: fail\n");
    let reset = bridge
        .call(
            "start_review",
            json!({"id": "add-two", "framework": "torch"}),
        )
        .await;
    assert_eq!(reset["isStarter"], true);

    // An accepted review submit advances the schedule.
    let events = submit("# scenario: happy\n", true).await;
    assert_eq!(events.last().expect("end")["verdict"], "accepted");
    let reviews = bridge.call("list_reviews", json!({})).await;
    assert_eq!(reviews[0]["stage"], 1);
    assert_eq!(reviews[0]["intervalDays"], 7);
    let listed = bridge
        .call("list_submissions", json!({"problemId": "add-two"}))
        .await;
    assert_eq!(listed[0]["review"], true);

    // Turning reviews off hides the schedule; turning them on brings it back.
    assert_eq!(
        bridge
            .call("set_reviews_enabled", json!({"enabled": false}))
            .await,
        json!({"reviewsEnabled": false})
    );
    assert_eq!(bridge.call("list_reviews", json!({})).await, json!([]));
    let (status, _) = bridge
        .post(
            "resolve_review",
            json!({"id": "add-two", "action": "snooze"}),
        )
        .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    bridge
        .call("set_reviews_enabled", json!({"enabled": true}))
        .await;

    let snoozed = bridge
        .call(
            "resolve_review",
            json!({"id": "add-two", "action": "snooze"}),
        )
        .await;
    assert_eq!(snoozed[0]["stage"], 1);
    let forgot = bridge
        .call(
            "resolve_review",
            json!({"id": "add-two", "action": "forgot"}),
        )
        .await;
    assert_eq!(forgot[0]["stage"], 0);
    let (status, _) = bridge
        .post(
            "resolve_review",
            json!({"id": "add-two", "action": "later"}),
        )
        .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert_eq!(
        bridge
            .call(
                "resolve_review",
                json!({"id": "add-two", "action": "retire"})
            )
            .await,
        json!([])
    );
}

#[tokio::test]
async fn run_targets_over_http() {
    let bridge = Bridge::start().await;

    let hosts = bridge.call("list_ssh_hosts", json!({})).await;
    assert_eq!(
        hosts,
        json!([{"alias": "gpu-box", "hostName": "10.0.0.5", "user": "ada", "port": null}])
    );

    let targets = bridge.call("list_targets", json!({})).await;
    assert_eq!(targets.as_array().expect("array").len(), 1);
    assert_eq!(targets[0]["name"], "local");
    assert_eq!(targets[0]["kind"], "local");
    assert_eq!(targets[0]["lastCheck"], Value::Null);

    let targets = bridge.call("add_target", json!({"host": "gpu-box"})).await;
    assert_eq!(
        targets[1],
        json!({"name": "ssh:gpu-box", "kind": "ssh", "label": "gpu-box", "host": "gpu-box",
               "remoteDir": "~/.learnai", "lastCheck": null})
    );

    // The local check streams steps and ends with `done`, like any other job.
    let started = bridge.call("check_target", json!({"name": "local"})).await;
    let job = started["jobId"].as_str().expect("jobId");
    let events = bridge.events(&format!("/api/jobs/{job}/events")).await;
    assert_eq!(types(&events), ["step", "log", "step", "done"]);
    assert_eq!(events[0]["step"]["id"], "uv");
    let done = events.last().expect("done");
    assert_eq!(done["check"]["ok"], true);
    assert_eq!(done["check"]["doctor"]["hardware"], json!(["cpu"]));
    let targets = bridge.call("list_targets", json!({})).await;
    assert_eq!(targets[0]["lastCheck"]["ok"], true);

    // A remote check with no ssh fails at the first step, with the reason.
    let started = bridge
        .call("check_target", json!({"name": "ssh:gpu-box"}))
        .await;
    let job = started["jobId"].as_str().expect("jobId");
    let events = bridge.events(&format!("/api/jobs/{job}/events")).await;
    assert_eq!(types(&events), ["step", "done"]);
    assert_eq!(events[0]["step"]["id"], "connect");
    assert_eq!(events[0]["step"]["ok"], false);

    for (command, body, status) in [
        (
            "add_target",
            json!({"host": "stranger"}),
            StatusCode::BAD_REQUEST,
        ),
        (
            "add_target",
            json!({"host": "gpu-box"}),
            StatusCode::BAD_REQUEST,
        ),
        ("add_target", json!({}), StatusCode::BAD_REQUEST),
        (
            "remove_target",
            json!({"name": "local"}),
            StatusCode::BAD_REQUEST,
        ),
        (
            "remove_target",
            json!({"name": "ssh:stranger"}),
            StatusCode::NOT_FOUND,
        ),
        (
            "check_target",
            json!({"name": "ssh:stranger"}),
            StatusCode::NOT_FOUND,
        ),
        (
            "fetch_assets",
            json!({"names": ["tiny"], "target": "ssh:stranger"}),
            StatusCode::NOT_FOUND,
        ),
        (
            "start_run",
            json!({"problemId": "add-two", "framework": "torch", "mode": "run", "target": "ssh:stranger"}),
            StatusCode::NOT_FOUND,
        ),
    ] {
        let (got, reply) = bridge.post(command, body.clone()).await;
        assert_eq!(got, status, "{command} {body}: {reply}");
        assert!(reply["error"].is_string(), "{command} {body}: {reply}");
    }

    let targets = bridge
        .call("remove_target", json!({"name": "ssh:gpu-box"}))
        .await;
    assert_eq!(targets.as_array().expect("array").len(), 1);
}
