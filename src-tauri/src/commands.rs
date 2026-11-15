//! One Tauri command per row of the command table in `docs/ARCHITECTURE.md` section 4.
//!
//! Each command only maps its arguments onto a [`Core`] method and converts the error to
//! a string. Core calls block (files, SQLite, subprocesses), so they run on the blocking
//! thread pool rather than on the main thread or an async worker.

use tauri::State;
use tauri::ipc::Channel;
use whetstone_core::{
    AssetEvent, AssetList, CatalogIssue, Core, EnvStatus, Framework, InstallEvent, ProblemDetail,
    ProblemSummary, RunEvent, RunMode, SolutionFile, SshHost, Stats, StudyPath, Submission,
    SubmissionDetail, TargetCheckEvent, TargetInfo, Track,
};

/// Shared application state: the core, or the reason it could not be opened.
pub struct AppState(pub Result<Core, String>);

/// What every command returns: errors cross the IPC boundary as a message.
type CommandResult<T> = Result<T, String>;

/// Runs `call` against the core on the blocking thread pool. Every command is logged
/// at debug level (`WHETSTONE_LOG=debug`) and every failure as a warning.
async fn with_core<T, F>(
    state: State<'_, AppState>,
    command: &'static str,
    call: F,
) -> CommandResult<T>
where
    T: Send + 'static,
    F: FnOnce(Core) -> whetstone_core::Result<T> + Send + 'static,
{
    tracing::debug!(command, "command");
    let result = match state.0.clone() {
        Ok(core) => tauri::async_runtime::spawn_blocking(move || call(core))
            .await
            .map_err(|e| e.to_string())
            .and_then(|result| result.map_err(|e| e.to_string())),
        Err(message) => Err(message),
    };
    if let Err(message) = &result {
        tracing::warn!(command, "command failed: {message}");
    }
    result
}

#[tauri::command]
pub async fn list_tracks(state: State<'_, AppState>) -> CommandResult<Vec<Track>> {
    with_core(state, "list_tracks", |core| core.list_tracks()).await
}

#[tauri::command]
pub async fn list_problems(state: State<'_, AppState>) -> CommandResult<Vec<ProblemSummary>> {
    with_core(state, "list_problems", |core| core.list_problems()).await
}

#[tauri::command]
pub async fn get_problem(state: State<'_, AppState>, id: String) -> CommandResult<ProblemDetail> {
    with_core(state, "get_problem", move |core| core.get_problem(&id)).await
}

#[tauri::command]
pub async fn read_solution(
    state: State<'_, AppState>,
    id: String,
    framework: Framework,
) -> CommandResult<SolutionFile> {
    with_core(state, "read_solution", move |core| {
        core.read_solution(&id, framework)
    })
    .await
}

#[tauri::command]
pub async fn write_solution(
    state: State<'_, AppState>,
    id: String,
    framework: Framework,
    code: String,
) -> CommandResult<()> {
    with_core(state, "write_solution", move |core| {
        core.write_solution(&id, framework, &code)
    })
    .await
}

#[tauri::command]
pub async fn reset_solution(
    state: State<'_, AppState>,
    id: String,
    framework: Framework,
) -> CommandResult<SolutionFile> {
    with_core(state, "reset_solution", move |core| {
        core.reset_solution(&id, framework)
    })
    .await
}

#[tauri::command]
pub async fn start_run(
    state: State<'_, AppState>,
    problem_id: String,
    framework: Framework,
    mode: RunMode,
    target: String,
    on_event: Channel<RunEvent>,
) -> CommandResult<String> {
    with_core(state, "start_run", move |core| {
        core.start_run(
            &problem_id,
            framework,
            mode,
            &target,
            move |event: RunEvent| {
                // The webview may be gone (window closed mid-run); there is no one to tell.
                let _ = on_event.send(event);
            },
        )
    })
    .await
}

#[tauri::command]
pub async fn cancel_run(state: State<'_, AppState>, run_id: String) -> CommandResult<()> {
    with_core(state, "cancel_run", move |core| {
        core.cancel_run(&run_id);
        Ok(())
    })
    .await
}

#[tauri::command]
pub async fn list_submissions(
    state: State<'_, AppState>,
    problem_id: Option<String>,
) -> CommandResult<Vec<Submission>> {
    with_core(state, "list_submissions", move |core| {
        core.list_submissions(problem_id.as_deref())
    })
    .await
}

#[tauri::command]
pub async fn get_submission(
    state: State<'_, AppState>,
    id: i64,
) -> CommandResult<SubmissionDetail> {
    with_core(state, "get_submission", move |core| core.get_submission(id)).await
}

#[tauri::command]
pub async fn unlock_editorial(
    state: State<'_, AppState>,
    id: String,
) -> CommandResult<ProblemDetail> {
    with_core(state, "unlock_editorial", move |core| {
        core.unlock_editorial(&id)
    })
    .await
}

#[tauri::command]
pub async fn get_reference(
    state: State<'_, AppState>,
    id: String,
    framework: Framework,
) -> CommandResult<String> {
    with_core(state, "get_reference", move |core| {
        core.get_reference(&id, framework)
    })
    .await
}

#[tauri::command]
pub async fn env_status(state: State<'_, AppState>) -> CommandResult<EnvStatus> {
    with_core(state, "env_status", |core| core.env_status()).await
}

/// Resolves once the install has finished, after the `done` event (section 4.1).
#[tauri::command]
pub async fn env_install(
    state: State<'_, AppState>,
    framework: Framework,
    on_event: Channel<InstallEvent>,
) -> CommandResult<()> {
    with_core(state, "env_install", move |core| {
        let (finished, wait) = std::sync::mpsc::channel();
        core.env_install(framework, move |event: InstallEvent| {
            let done = matches!(event, InstallEvent::Done { .. });
            let _ = on_event.send(event);
            if done {
                let _ = finished.send(());
            }
        })?;
        // An error here means the sink was dropped without `done`; either way it is over.
        let _ = wait.recv();
        Ok(())
    })
    .await
}

#[tauri::command]
pub async fn get_stats(state: State<'_, AppState>) -> CommandResult<Stats> {
    with_core(state, "get_stats", |core| core.get_stats()).await
}

#[tauri::command]
pub async fn list_catalog_issues(state: State<'_, AppState>) -> CommandResult<Vec<CatalogIssue>> {
    with_core(state, "list_catalog_issues", |core| {
        Ok(core.list_catalog_issues())
    })
    .await
}

#[tauri::command]
pub async fn list_paths(state: State<'_, AppState>) -> CommandResult<Vec<StudyPath>> {
    with_core(state, "list_paths", |core| core.list_paths()).await
}

#[tauri::command]
pub async fn get_problem_asset(
    state: State<'_, AppState>,
    id: String,
    path: String,
) -> CommandResult<String> {
    with_core(state, "get_problem_asset", move |core| {
        core.get_problem_asset(&id, &path)
    })
    .await
}

#[tauri::command]
pub async fn list_assets(state: State<'_, AppState>) -> CommandResult<AssetList> {
    with_core(state, "list_assets", |core| core.list_assets()).await
}

/// Returns the job id at once; events flow over the channel until `done` (section 4.1).
#[tauri::command]
pub async fn fetch_assets(
    state: State<'_, AppState>,
    names: Vec<String>,
    target: Option<String>,
    on_event: Channel<AssetEvent>,
) -> CommandResult<String> {
    with_core(state, "fetch_assets", move |core| {
        core.fetch_assets(&names, target.as_deref(), move |event: AssetEvent| {
            // The webview may be gone (window closed mid-download); there is no one to tell.
            let _ = on_event.send(event);
        })
    })
    .await
}

#[tauri::command]
pub async fn cancel_asset_fetch(state: State<'_, AppState>, job_id: String) -> CommandResult<()> {
    with_core(state, "cancel_asset_fetch", move |core| {
        core.cancel_asset_fetch(&job_id);
        Ok(())
    })
    .await
}

#[tauri::command]
pub async fn remove_asset(state: State<'_, AppState>, name: String) -> CommandResult<AssetList> {
    with_core(state, "remove_asset", move |core| core.remove_asset(&name)).await
}

#[tauri::command]
pub async fn list_ssh_hosts(state: State<'_, AppState>) -> CommandResult<Vec<SshHost>> {
    with_core(state, "list_ssh_hosts", |core| Ok(core.list_ssh_hosts())).await
}

#[tauri::command]
pub async fn list_targets(state: State<'_, AppState>) -> CommandResult<Vec<TargetInfo>> {
    with_core(state, "list_targets", |core| core.list_targets()).await
}

#[tauri::command]
pub async fn add_target(
    state: State<'_, AppState>,
    host: String,
    remote_dir: Option<String>,
) -> CommandResult<Vec<TargetInfo>> {
    with_core(state, "add_target", move |core| {
        core.add_target(&host, remote_dir.as_deref())
    })
    .await
}

#[tauri::command]
pub async fn remove_target(
    state: State<'_, AppState>,
    name: String,
) -> CommandResult<Vec<TargetInfo>> {
    with_core(state, "remove_target", move |core| {
        core.remove_target(&name)
    })
    .await
}

/// Returns the job id at once; events flow over the channel until `done` (section 4.1).
#[tauri::command]
pub async fn check_target(
    state: State<'_, AppState>,
    name: String,
    on_event: Channel<TargetCheckEvent>,
) -> CommandResult<String> {
    with_core(state, "check_target", move |core| {
        core.check_target(&name, move |event: TargetCheckEvent| {
            // The webview may be gone (window closed mid-check); there is no one to tell.
            let _ = on_event.send(event);
        })
    })
    .await
}
