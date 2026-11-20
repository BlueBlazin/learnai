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
    "list_areas",
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
    "set_editor_intelligence",
    "editor_status",
    "editor_start",
    "editor_send",
    "editor_stop",
    "editor_install",
    "editor_docs",
];

/// Fails the build when a command has no grant in `capabilities/default.json`. Without
/// this, a new command works in the browser preview (which talks to the development
/// bridge) and is silently refused in the real app.
fn check_capabilities() {
    println!("cargo:rerun-if-changed=capabilities/default.json");
    let granted = std::fs::read_to_string("capabilities/default.json")
        .expect("capabilities/default.json is readable");
    let missing: Vec<String> = COMMANDS
        .iter()
        .map(|command| format!("allow-{}", command.replace('_', "-")))
        .filter(|permission| !granted.contains(&format!("\"{permission}\"")))
        .collect();
    assert!(
        missing.is_empty(),
        "capabilities/default.json does not grant: {missing:?}"
    );
}

fn main() {
    check_capabilities();
    let attributes = tauri_build::Attributes::new()
        .app_manifest(tauri_build::AppManifest::new().commands(COMMANDS));
    tauri_build::try_build(attributes).expect("failed to run the Tauri build script");
}
