/// Every command the front end may invoke. Listing them makes Tauri generate an
/// `allow-<command>` permission for each, which `capabilities/default.json` grants to
/// the main window; a command missing from either place is rejected at runtime.
const COMMANDS: &[&str] = &[
    "list_tracks",
    "list_problems",
    "get_problem",
    "read_solution",
    "write_solution",
    "reset_solution",
    "start_run",
    "cancel_run",
    "list_submissions",
    "get_submission",
    "unlock_editorial",
    "get_reference",
    "env_status",
    "env_install",
    "get_stats",
    "list_catalog_issues",
    "list_paths",
    "get_problem_asset",
    "list_assets",
    "fetch_assets",
    "cancel_asset_fetch",
    "remove_asset",
    "list_ssh_hosts",
    "list_targets",
    "add_target",
    "remove_target",
    "check_target",
    "list_reviews",
    "start_review",
    "read_review_solution",
    "write_review_solution",
    "resolve_review",
    "get_settings",
    "set_reviews_enabled",
];

fn main() {
    let attributes = tauri_build::Attributes::new()
        .app_manifest(tauri_build::AppManifest::new().commands(COMMANDS));
    tauri_build::try_build(attributes).expect("failed to run the Tauri build script");
}
