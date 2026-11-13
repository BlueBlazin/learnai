//! Thin shell: maps Tauri commands to whetstone-core functions. No logic lives here.

mod commands;

use tauri::Manager;
use whetstone_core::{Core, CoreOptions, RootInputs, resolve_root};

use crate::commands::AppState;

/// Opens the core for the resolved content root. The saved root setting lives in the
/// app's configuration directory.
fn open_core(app: &tauri::App) -> Result<Core, String> {
    let inputs = RootInputs {
        config_dir: app.path().app_config_dir().ok(),
        ..RootInputs::default()
    };
    let root = resolve_root(&inputs).map_err(|e| e.to_string())?;
    Core::open(CoreOptions::new(root.path)).map_err(|e| e.to_string())
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_opener::init())
        .setup(|app| {
            // A core that fails to open is reported by every command instead of
            // preventing the window from appearing.
            app.manage(AppState(open_core(app)));
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
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
