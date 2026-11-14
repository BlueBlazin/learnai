//! Dev-only HTTP bridge exposing whetstone-core to a plain browser.
//!
//! Implements the devbridge transport in `docs/ARCHITECTURE.md` section 4.1:
//! `POST /api/<command>` with a JSON object body, and server-sent event streams for run
//! and install events with buffering and replay. Never bundled with the app.

mod events;

use std::sync::Arc;

use axum::Router;
use axum::body::Bytes;
use axum::extract::{Path, State};
use axum::http::header::CONTENT_TYPE;
use axum::http::{HeaderValue, Method, StatusCode};
use axum::response::sse::{KeepAlive, Sse};
use axum::response::{IntoResponse, Json, Response};
use axum::routing::{get, post};
use serde::Deserialize;
use serde::de::DeserializeOwned;
use serde_json::{Value, json};
use tokio::net::TcpListener;
use tower_http::cors::CorsLayer;
use whetstone_core::{Core, CoreError, Framework, InstallEvent, RunEvent, RunMode};

use crate::events::{EventLog, LogRegistry};

/// The port the bridge listens on unless told otherwise.
pub const DEFAULT_PORT: u16 = 1421;

/// Origins of the Vite dev server (see `devUrl` in `src-tauri/tauri.conf.json`).
pub const DEFAULT_ORIGINS: &[&str] = &["http://localhost:1420", "http://127.0.0.1:1420"];

struct Bridge {
    core: Core,
    runs: LogRegistry,
    jobs: LogRegistry,
}

type Shared = Arc<Bridge>;

/// An error response: a status and a message, rendered as `{"error": "<message>"}`.
#[derive(Debug)]
struct ApiError {
    status: StatusCode,
    message: String,
}

impl ApiError {
    fn new(status: StatusCode, message: impl Into<String>) -> Self {
        Self {
            status,
            message: message.into(),
        }
    }
}

impl From<CoreError> for ApiError {
    fn from(error: CoreError) -> Self {
        let status = match &error {
            CoreError::EditorialLocked(_) => StatusCode::FORBIDDEN,
            CoreError::InstallInProgress => StatusCode::CONFLICT,
            e if e.is_not_found() => StatusCode::NOT_FOUND,
            e if e.is_invalid_request() => StatusCode::BAD_REQUEST,
            _ => StatusCode::INTERNAL_SERVER_ERROR,
        };
        Self::new(status, error.to_string())
    }
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        (self.status, Json(json!({ "error": self.message }))).into_response()
    }
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct IdArgs {
    id: String,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct AssetArgs {
    id: String,
    path: String,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct SolutionArgs {
    id: String,
    framework: Framework,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct WriteSolutionArgs {
    id: String,
    framework: Framework,
    code: String,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct StartRunArgs {
    problem_id: String,
    framework: Framework,
    mode: RunMode,
    target: String,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct RunIdArgs {
    run_id: String,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct ListSubmissionsArgs {
    #[serde(default)]
    problem_id: Option<String>,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct SubmissionIdArgs {
    id: i64,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct FrameworkArgs {
    framework: Framework,
}

/// Parses the request body: a JSON object of camelCase arguments. An empty body counts
/// as an empty object.
fn parse_body(body: &[u8]) -> Result<Value, ApiError> {
    if body.iter().all(u8::is_ascii_whitespace) {
        return Ok(json!({}));
    }
    match serde_json::from_slice(body) {
        Ok(value @ Value::Object(_)) => Ok(value),
        Ok(_) => Err(ApiError::new(
            StatusCode::BAD_REQUEST,
            "invalid arguments: expected a JSON object",
        )),
        Err(e) => Err(ApiError::new(
            StatusCode::BAD_REQUEST,
            format!("invalid arguments: {e}"),
        )),
    }
}

/// Extracts the typed arguments of a command from the request object.
fn args<T: DeserializeOwned>(body: Value) -> Result<T, ApiError> {
    serde_json::from_value(body)
        .map_err(|e| ApiError::new(StatusCode::BAD_REQUEST, format!("invalid arguments: {e}")))
}

fn reply<T: serde::Serialize>(value: T) -> Result<Value, ApiError> {
    serde_json::to_value(value)
        .map_err(|e| ApiError::new(StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))
}

/// Runs one command. Blocking: touches the file system, SQLite, and subprocesses.
fn dispatch(bridge: &Bridge, command: &str, body: &[u8]) -> Result<Value, ApiError> {
    let core = &bridge.core;
    let body = parse_body(body)?;
    match command {
        "list_tracks" => reply(core.list_tracks()?),
        "list_problems" => reply(core.list_problems()?),
        "get_problem" => reply(core.get_problem(&args::<IdArgs>(body)?.id)?),
        "read_solution" => {
            let a: SolutionArgs = args(body)?;
            reply(core.read_solution(&a.id, a.framework)?)
        }
        "write_solution" => {
            let a: WriteSolutionArgs = args(body)?;
            reply(core.write_solution(&a.id, a.framework, &a.code)?)
        }
        "reset_solution" => {
            let a: SolutionArgs = args(body)?;
            reply(core.reset_solution(&a.id, a.framework)?)
        }
        "start_run" => {
            let a: StartRunArgs = args(body)?;
            let log = EventLog::new();
            let sink = Arc::clone(&log);
            let run_id = core.start_run(
                &a.problem_id,
                a.framework,
                a.mode,
                &a.target,
                move |event: RunEvent| {
                    sink.push(&event, matches!(event, RunEvent::End { .. }));
                },
            )?;
            bridge.runs.insert(run_id.clone(), log);
            Ok(json!({ "runId": run_id }))
        }
        "cancel_run" => {
            core.cancel_run(&args::<RunIdArgs>(body)?.run_id);
            Ok(Value::Null)
        }
        "list_submissions" => {
            reply(core.list_submissions(args::<ListSubmissionsArgs>(body)?.problem_id.as_deref())?)
        }
        "get_submission" => reply(core.get_submission(args::<SubmissionIdArgs>(body)?.id)?),
        "unlock_editorial" => reply(core.unlock_editorial(&args::<IdArgs>(body)?.id)?),
        "get_reference" => {
            let a: SolutionArgs = args(body)?;
            reply(core.get_reference(&a.id, a.framework)?)
        }
        "env_status" => reply(core.env_status()?),
        "env_install" => {
            let a: FrameworkArgs = args(body)?;
            let log = EventLog::new();
            let sink = Arc::clone(&log);
            core.env_install(a.framework, move |event: InstallEvent| {
                sink.push(&event, matches!(event, InstallEvent::Done { .. }));
            })?;
            let job_id = uuid::Uuid::new_v4().to_string();
            bridge.jobs.insert(job_id.clone(), log);
            Ok(json!({ "jobId": job_id }))
        }
        "get_stats" => reply(core.get_stats()?),
        "list_catalog_issues" => reply(core.list_catalog_issues()),
        "list_paths" => reply(core.list_paths()?),
        "get_problem_asset" => {
            let a: AssetArgs = args(body)?;
            reply(core.get_problem_asset(&a.id, &a.path)?)
        }
        other => Err(ApiError::new(
            StatusCode::NOT_FOUND,
            format!("unknown command `{other}`"),
        )),
    }
}

async fn command(
    State(bridge): State<Shared>,
    Path(name): Path<String>,
    body: Bytes,
) -> Result<Json<Value>, ApiError> {
    tokio::task::spawn_blocking(move || dispatch(&bridge, &name, &body))
        .await
        .map_err(|e| {
            ApiError::new(
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("command panicked: {e}"),
            )
        })?
        .map(Json)
}

fn event_stream(registry: &LogRegistry, kind: &str, id: &str) -> Result<Response, ApiError> {
    let log = registry
        .get(id)
        .ok_or_else(|| ApiError::new(StatusCode::NOT_FOUND, format!("unknown {kind} `{id}`")))?;
    Ok(Sse::new(log.subscribe())
        .keep_alive(KeepAlive::default())
        .into_response())
}

async fn run_events(
    State(bridge): State<Shared>,
    Path(id): Path<String>,
) -> Result<Response, ApiError> {
    event_stream(&bridge.runs, "run", &id)
}

async fn job_events(
    State(bridge): State<Shared>,
    Path(id): Path<String>,
) -> Result<Response, ApiError> {
    event_stream(&bridge.jobs, "job", &id)
}

async fn not_found() -> ApiError {
    ApiError::new(StatusCode::NOT_FOUND, "not found")
}

/// Builds the bridge router for `core`, with CORS open to `origins`.
pub fn router(core: Core, origins: &[String]) -> Router {
    let origins: Vec<HeaderValue> = origins
        .iter()
        .filter_map(|origin| origin.parse().ok())
        .collect();
    let cors = CorsLayer::new()
        .allow_origin(origins)
        .allow_methods([Method::GET, Method::POST])
        .allow_headers([CONTENT_TYPE]);
    Router::new()
        .route("/api/runs/{id}/events", get(run_events))
        .route("/api/jobs/{id}/events", get(job_events))
        .route("/api/{command}", post(command))
        .fallback(not_found)
        .layer(cors)
        .with_state(Arc::new(Bridge {
            core,
            runs: LogRegistry::default(),
            jobs: LogRegistry::default(),
        }))
}

/// Serves the bridge on `listener` until the task is dropped or the process exits.
pub async fn serve(listener: TcpListener, core: Core, origins: &[String]) -> std::io::Result<()> {
    axum::serve(listener, router(core, origins)).await
}
