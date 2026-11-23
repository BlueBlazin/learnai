//! One Tauri command per row of the command table in `docs/ARCHITECTURE.md` section 4.
//!
//! Each command only maps its arguments onto a [`Core`] method and converts the error to
//! a string. Core calls block (files, SQLite, subprocesses), so they run on the blocking
//! thread pool rather than on the main thread or an async worker.

use learnai_core::{
    Area, AssetEvent, AssetList, CatalogIssue, Core, EditorDocs, EditorEvent, EditorStatus,
    EnvStatus, Framework, InstallEvent, ProblemDetail, ProblemSummary, Review, ReviewAction,
    RunEvent, RunMode, Settings, Skill, SolutionFile, SshHost, Stats, Submission, SubmissionDetail,
    TargetCheckEvent, TargetInfo, Track,
};
use tauri::State;
use tauri::ipc::Channel;

/// Shared application state: the core, or the reason it could not be opened.
pub struct AppState(pub Result<Core, String>);

/// What every command returns: errors cross the IPC boundary as a message.
type CommandResult<T> = Result<T, String>;

/// Runs `call` against the core on the blocking thread pool. Every command is logged
/// at debug level (`LEARNAI_LOG=debug`) and every failure as a warning.
async fn with_core<T, F>(
    state: State<'_, AppState>,
    command: &'static str,
    call: F,
) -> CommandResult<T>
where
    T: Send + 'static,
    F: FnOnce(Core) -> learnai_core::Result<T> + Send + 'static,
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

/// `review: true` judges the review attempt instead of the saved solution (section 4.2).
#[tauri::command]
pub async fn start_run(
    state: State<'_, AppState>,
    problem_id: String,
    framework: Framework,
    mode: RunMode,
    target: String,
    review: Option<bool>,
    on_event: Channel<RunEvent>,
) -> CommandResult<String> {
    with_core(state, "start_run", move |core| {
        let sink = move |event: RunEvent| {
            // The webview may be gone (window closed mid-run); there is no one to tell.
            let _ = on_event.send(event);
        };
        if review.unwrap_or(false) {
            core.start_review_run(&problem_id, framework, mode, &target, sink)
        } else {
            core.start_run(&problem_id, framework, mode, &target, sink)
        }
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
pub async fn list_areas(state: State<'_, AppState>) -> CommandResult<Vec<Area>> {
    with_core(state, "list_areas", |core| core.list_areas()).await
}

#[tauri::command]
pub async fn list_skills(state: State<'_, AppState>) -> CommandResult<Vec<Skill>> {
    with_core(state, "list_skills", |core| core.list_skills()).await
}

#[tauri::command]
pub async fn skills_earned_by(
    state: State<'_, AppState>,
    submission_id: i64,
) -> CommandResult<Vec<Skill>> {
    with_core(state, "skills_earned_by", move |core| {
        core.skills_earned_by(submission_id)
    })
    .await
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

#[tauri::command]
pub async fn list_reviews(state: State<'_, AppState>) -> CommandResult<Vec<Review>> {
    with_core(state, "list_reviews", |core| core.list_reviews()).await
}

#[tauri::command]
pub async fn start_review(
    state: State<'_, AppState>,
    id: String,
    framework: Framework,
) -> CommandResult<SolutionFile> {
    with_core(state, "start_review", move |core| {
        core.start_review(&id, framework)
    })
    .await
}

#[tauri::command]
pub async fn read_review_solution(
    state: State<'_, AppState>,
    id: String,
    framework: Framework,
) -> CommandResult<SolutionFile> {
    with_core(state, "read_review_solution", move |core| {
        core.read_review_solution(&id, framework)
    })
    .await
}

#[tauri::command]
pub async fn write_review_solution(
    state: State<'_, AppState>,
    id: String,
    framework: Framework,
    code: String,
) -> CommandResult<()> {
    with_core(state, "write_review_solution", move |core| {
        core.write_review_solution(&id, framework, &code)
    })
    .await
}

#[tauri::command]
pub async fn resolve_review(
    state: State<'_, AppState>,
    id: String,
    action: ReviewAction,
) -> CommandResult<Vec<Review>> {
    with_core(state, "resolve_review", move |core| {
        core.resolve_review(&id, action)
    })
    .await
}

#[tauri::command]
pub async fn get_settings(state: State<'_, AppState>) -> CommandResult<Settings> {
    with_core(state, "get_settings", |core| core.get_settings()).await
}

#[tauri::command]
pub async fn set_reviews_enabled(
    state: State<'_, AppState>,
    enabled: bool,
) -> CommandResult<Settings> {
    with_core(state, "set_reviews_enabled", move |core| {
        core.set_reviews_enabled(enabled)
    })
    .await
}

#[tauri::command]
pub async fn set_editor_intelligence(
    state: State<'_, AppState>,
    enabled: bool,
) -> CommandResult<Settings> {
    with_core(state, "set_editor_intelligence", move |core| {
        core.set_editor_intelligence(enabled)
    })
    .await
}

#[tauri::command]
pub async fn editor_status(state: State<'_, AppState>) -> CommandResult<EditorStatus> {
    with_core(state, "editor_status", |core| core.editor_status()).await
}

/// Returns the session id at once; events flow over the channel until a terminal state
/// (section 4.4).
#[tauri::command]
pub async fn editor_start(
    state: State<'_, AppState>,
    on_event: Channel<EditorEvent>,
) -> CommandResult<String> {
    with_core(state, "editor_start", move |core| {
        core.editor_start(move |event: EditorEvent| {
            // The webview may be gone (window closed); there is no one to tell.
            let _ = on_event.send(event);
        })
    })
    .await
}

#[tauri::command]
pub async fn editor_send(
    state: State<'_, AppState>,
    session_id: String,
    message: String,
) -> CommandResult<()> {
    with_core(state, "editor_send", move |core| {
        core.editor_send(&session_id, &message)
    })
    .await
}

#[tauri::command]
pub async fn editor_stop(
    state: State<'_, AppState>,
    session_id: Option<String>,
) -> CommandResult<()> {
    with_core(state, "editor_stop", move |core| {
        core.editor_stop(session_id.as_deref());
        Ok(())
    })
    .await
}

/// Resolves once the install has finished, after the `done` event, like `env_install`.
#[tauri::command]
pub async fn editor_install(
    state: State<'_, AppState>,
    on_event: Channel<InstallEvent>,
) -> CommandResult<()> {
    with_core(state, "editor_install", move |core| {
        let (finished, wait) = std::sync::mpsc::channel();
        core.editor_install(move |event: InstallEvent| {
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
pub async fn editor_docs(
    state: State<'_, AppState>,
    source: String,
    expr: String,
    member: Option<String>,
) -> CommandResult<EditorDocs> {
    with_core(state, "editor_docs", move |core| {
        core.editor_docs(&source, &expr, member.as_deref())
    })
    .await
}
