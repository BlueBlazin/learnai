//! Thin shell: maps Tauri commands to whetstone-core functions. No logic lives here.

mod commands;

use std::fs::{self, OpenOptions};
use std::sync::Mutex;

use tauri::Manager;
use tracing_subscriber::layer::SubscriberExt;
use tracing_subscriber::util::SubscriberInitExt;
use whetstone_core::{Core, CoreOptions, RootInputs, resolve_root};

use crate::commands::AppState;

/// File name of the app log inside the platform log directory
/// (`~/Library/Logs/dev.whetstone.app/` on macOS).
const LOG_FILE: &str = "whetstone.log";

/// The log level: `info` unless `WHETSTONE_LOG` names another (`debug` logs every
/// command the window invokes). The windowing libraries trace every event, so the
/// default keeps the log to what a person reads.
fn log_level() -> tracing_subscriber::filter::LevelFilter {
    std::env::var("WHETSTONE_LOG")
        .ok()
        .and_then(|level| level.parse().ok())
        .unwrap_or(tracing_subscriber::filter::LevelFilter::INFO)
}

/// Sends core's diagnostics (catalog issues, for one) to stderr and, when the log
/// directory is usable, to the app's log file. Without this, `tracing` events are dropped.
fn init_logging(app: &tauri::App) {
    let file = app.path().app_log_dir().ok().and_then(|dir| {
        fs::create_dir_all(&dir).ok()?;
        OpenOptions::new()
            .create(true)
            .append(true)
            .open(dir.join(LOG_FILE))
            .ok()
    });
    let file_layer = file.map(|file| {
        tracing_subscriber::fmt::layer()
            .with_ansi(false)
            .with_writer(Mutex::new(file))
    });
    // Fails only if a subscriber is already installed, which is fine.
    let _ = tracing_subscriber::registry()
        .with(log_level())
        .with(tracing_subscriber::fmt::layer().with_writer(std::io::stderr))
        .with(file_layer)
        .try_init();
}

/// Opens the core for the resolved content root. The saved root setting lives in the
/// app's configuration directory.
fn open_core(app: &tauri::App) -> Result<Core, String> {
    let inputs = RootInputs {
        config_dir: app.path().app_config_dir().ok(),
        ..RootInputs::default()
    };
    let root = resolve_root(&inputs).map_err(|e| e.to_string())?;
    tracing::info!(root = %root.path.display(), source = ?root.source, "content root");
    let core = Core::open(CoreOptions::new(root.path)).map_err(|e| e.to_string())?;
    // Scan the catalog once at startup so its issues reach the log before any window asks.
    let issues = core.catalog_issues().len();
    let problems = core.list_problems().map_or(0, |p| p.len());
    tracing::info!(problems, issues, "catalog loaded");
    Ok(core)
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_opener::init())
        .setup(|app| {
            // A core that fails to open is reported by every command instead of
            // preventing the window from appearing.
            init_logging(app);
            let core = open_core(app);
            if let Err(message) = &core {
                tracing::error!("the core could not be opened: {message}");
            }
            app.manage(AppState(core));
            Ok(())
        })
        .invoke_handler(tauri::generate_handler![
            commands::list_tracks,
            commands::list_problems,
            commands::get_problem,
            commands::read_solution,
            commands::write_solution,
            commands::reset_solution,
            commands::start_run,
            commands::cancel_run,
            commands::list_submissions,
            commands::get_submission,
            commands::unlock_editorial,
            commands::get_reference,
            commands::env_status,
            commands::env_install,
            commands::get_stats,
            commands::list_catalog_issues,
            commands::list_paths,
            commands::get_problem_asset,
            commands::list_assets,
            commands::fetch_assets,
            commands::cancel_asset_fetch,
            commands::remove_asset,
            commands::list_ssh_hosts,
            commands::list_targets,
            commands::add_target,
            commands::remove_target,
            commands::check_target,
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
