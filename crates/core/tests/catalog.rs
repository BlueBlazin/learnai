//! Catalog loading, validation, and reload.

mod common;

use std::fs;
use std::time::Instant;

use common::TestRoot;
use whetstone_core::catalog::CatalogLoader;
use whetstone_core::{CoreError, Difficulty, Framework, JudgeKind, ProblemStatus, Requires};

#[test]
fn lists_valid_problems_in_track_then_problem_order() {
    let root = TestRoot::new();
    let core = root.core("happy");

    let problems = core.list_problems().expect("list problems");
    let ids: Vec<_> = problems.iter().map(|p| p.id.as_str()).collect();
    assert_eq!(ids, ["add-two", "echo-server", "hard-one"]);

    let add = &problems[0];
    assert_eq!(add.title, "Add Two Tensors");
    assert_eq!(add.track, "basics");
    assert_eq!(add.difficulty, Difficulty::Easy);
    assert_eq!(add.frameworks, [Framework::Torch, Framework::Jax]);
    assert_eq!(add.judges, [JudgeKind::Numeric]);
    assert_eq!(add.tags, ["warmup"]);
    assert_eq!(add.status, ProblemStatus::Todo);
    assert!(add.solved_frameworks.is_empty());

    let echo = &problems[1];
    assert_eq!(echo.requires, Requires::MlxGpu);
    assert_eq!(echo.prerequisites, ["add-two"]);
    assert_eq!(echo.judges, [JudgeKind::Service, JudgeKind::Simclock]);
    // Defaults for omitted optional keys.
    assert_eq!(problems[2].requires, Requires::Cpu);
    assert!(problems[2].tags.is_empty());

    let tracks = core.list_tracks().expect("list tracks");
    let summary: Vec<_> = tracks
        .iter()
        .map(|t| (t.id.as_str(), t.problem_count, t.solved_count))
        .collect();
    assert_eq!(summary, [("basics", 2, 0), ("advanced", 1, 0)]);
    assert_eq!(tracks[0].description, "Warm-up problems.");
}

#[test]
fn malformed_problems_are_reported_not_fatal() {
    let root = TestRoot::new();
    let core = root.core("happy");
    let issues = core.catalog_issues();
    let find = |needle: &str| {
        issues
            .iter()
            .find(|i| i.path.to_string_lossy().contains(needle))
            .unwrap_or_else(|| panic!("no issue for {needle}: {issues:?}"))
    };

    assert!(!find("broken-toml").message.is_empty());
    assert!(
        find("wrong-id")
            .message
            .contains("does not equal the directory name")
    );
    assert_eq!(find("no-starter").message, "missing starter/torch.py");
    assert_eq!(find("empty-dir").message, "missing problem.toml");
    assert_eq!(find("orphan").message, "missing track.toml");
    assert!(
        find("hard-one")
            .message
            .contains("prerequisite `does-not-exist`")
    );
    assert!(
        !issues
            .iter()
            .any(|i| i.path.to_string_lossy().contains("_drafts"))
    );
    assert_eq!(issues.len(), 6, "{issues:?}");

    // Skipped problems are unknown to every command.
    assert!(matches!(
        core.get_problem("wrong-id"),
        Err(CoreError::ProblemNotFound(_))
    ));
    assert!(matches!(
        core.get_problem("lost"),
        Err(CoreError::ProblemNotFound(_))
    ));
    assert!(matches!(
        core.get_problem("../basics"),
        Err(CoreError::ProblemNotFound(_))
    ));
}

#[test]
fn problem_detail_has_statement_hints_and_a_locked_editorial() {
    let root = TestRoot::new();
    let core = root.core("happy");

    let detail = core.get_problem("add-two").expect("detail");
    assert!(detail.statement.starts_with("# Add Two Tensors"));
    assert_eq!(
        detail.hints,
        [
            "Think about broadcasting.",
            "Use the `+` operator.",
            "Really, just `a + b`."
        ]
    );
    assert_eq!(detail.entry, ["add"]);
    assert_eq!(detail.time_limit_s, 10.0);
    assert!(!detail.editorial_unlocked);
    assert_eq!(detail.editorial, None);
    assert!(matches!(
        core.get_reference("add-two", Framework::Torch),
        Err(CoreError::EditorialLocked(_))
    ));

    let unlocked = core.unlock_editorial("add-two").expect("unlock");
    assert!(unlocked.editorial_unlocked);
    assert_eq!(
        unlocked.editorial.as_deref(),
        Some("# Editorial\n\nAdd them.\n")
    );
    assert!(
        core.get_problem("add-two")
            .expect("detail")
            .editorial_unlocked
    );
    assert_eq!(
        core.get_reference("add-two", Framework::Jax)
            .expect("reference"),
        "def add(a, b):\n    return a + b  # jax\n"
    );
    assert!(matches!(
        core.get_reference("add-two", Framework::Mlx),
        Err(CoreError::FrameworkNotSupported { .. })
    ));

    // A problem without hints or editorial files still loads.
    let echo = core.unlock_editorial("echo-server").expect("unlock");
    assert!(echo.hints.is_empty());
    assert_eq!(echo.editorial.as_deref(), Some(""));
    assert_eq!(echo.time_limit_s, 0.05);
}

#[test]
fn detail_serializes_flat_and_camel_case() {
    let root = TestRoot::new();
    let detail = root
        .core("happy")
        .get_problem("echo-server")
        .expect("detail");
    let json = serde_json::to_value(&detail).expect("json");
    assert_eq!(json["id"], "echo-server");
    assert_eq!(json["requires"], "mlx-gpu");
    assert_eq!(json["solvedFrameworks"], serde_json::json!([]));
    assert_eq!(json["editorialUnlocked"], false);
    assert_eq!(json["editorial"], serde_json::Value::Null);
    assert_eq!(json["timeLimitS"], 0.05);
    assert!(json.get("summary").is_none());
}

#[test]
fn reload_picks_up_fixes_edits_additions_and_removals() {
    let root = TestRoot::new();
    let core = root.core("happy");
    let basics = root.path().join("problems").join("basics");
    assert_eq!(core.list_problems().expect("list").len(), 3);

    // Fix a broken problem by adding the missing file (problem.toml untouched).
    fs::create_dir_all(basics.join("no-starter").join("starter")).expect("mkdir");
    fs::write(
        basics.join("no-starter").join("starter").join("torch.py"),
        "x = 1\n",
    )
    .expect("write");
    // Edit an existing problem.
    let toml = basics.join("add-two").join("problem.toml");
    let text = fs::read_to_string(&toml)
        .expect("read")
        .replace("Add Two Tensors", "Add Two (Renamed)");
    fs::write(&toml, text).expect("write");
    // Remove one.
    fs::remove_dir_all(basics.join("echo-server")).expect("remove");

    let problems = core.list_problems().expect("list");
    let ids: Vec<_> = problems.iter().map(|p| p.id.as_str()).collect();
    assert_eq!(ids, ["no-starter", "add-two", "hard-one"]);
    assert_eq!(problems[1].title, "Add Two (Renamed)");
}

#[test]
fn duplicate_ids_across_tracks_keep_the_first() {
    let root = TestRoot::new();
    let problems = root.path().join("problems");
    let copy = problems.join("advanced").join("add-two");
    fs::create_dir_all(copy.join("starter")).expect("mkdir");
    let text = fs::read_to_string(problems.join("basics/add-two/problem.toml"))
        .expect("read")
        .replace("track = \"basics\"", "track = \"advanced\"");
    fs::write(copy.join("problem.toml"), text).expect("write");
    fs::write(copy.join("statement.md"), "# dup\n").expect("write");
    for framework in ["torch", "jax"] {
        fs::write(
            copy.join("starter").join(format!("{framework}.py")),
            "x = 1\n",
        )
        .expect("write");
    }

    let core = root.core("happy");
    assert_eq!(
        core.get_problem("add-two").expect("detail").summary.track,
        "basics"
    );
    assert!(
        core.catalog_issues()
            .iter()
            .any(|i| i.message.contains("duplicate problem id"))
    );
}

#[test]
fn reloading_a_large_catalog_is_cheap() {
    let root = TestRoot::new();
    let track = root.path().join("problems").join("bulk");
    fs::create_dir_all(&track).expect("mkdir");
    fs::write(
        track.join("track.toml"),
        "id = \"bulk\"\ntitle = \"Bulk\"\norder = 99\n",
    )
    .expect("write");
    for n in 0..200 {
        let id = format!("bulk-{n:03}");
        let dir = track.join(&id);
        fs::create_dir_all(dir.join("starter")).expect("mkdir");
        fs::write(
            dir.join("problem.toml"),
            format!(
                "id = \"{id}\"\ntitle = \"Bulk {n}\"\ntrack = \"bulk\"\norder = {n}\ndifficulty = \"medium\"\n\
                 frameworks = [\"torch\", \"jax\", \"mlx\"]\njudges = [\"numeric\"]\nentry = [\"f\"]\n"
            ),
        )
        .expect("write");
        fs::write(dir.join("statement.md"), "# bulk\n").expect("write");
        for framework in ["torch", "jax", "mlx"] {
            fs::write(
                dir.join("starter").join(format!("{framework}.py")),
                "x = 1\n",
            )
            .expect("write");
        }
    }

    let mut loader = CatalogLoader::new(root.path());
    assert_eq!(loader.load().problems().len(), 203);
    let began = Instant::now();
    for _ in 0..10 {
        assert_eq!(loader.load().problems().len(), 203);
    }
    let per_reload = began.elapsed() / 10;
    // Generous bound: a warm reload is a directory walk plus stats, a few milliseconds.
    assert!(per_reload.as_millis() < 250, "reload took {per_reload:?}");
}
