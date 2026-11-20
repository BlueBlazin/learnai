//! Study paths: loading, validation, progress, and catalog issues.

mod common;

use std::fs;
use std::path::Path;

use common::{Collector, TestRoot};
use learnai_core::{Framework, RunEvent, RunMode};

const STARTER: &str = r#"
id = "starter"
title = "Starter Path"
description = "A short path."

[[stages]]
title = "Warm-up"
description = "First steps."
problems = ["add-two", "not-a-problem", "echo-server"]

[[stages]]
title = "Harder"
description = "Then this."
problems = ["hard-one", "add-two"]

[[stages]]
title = "Nothing usable"
problems = ["wrong-id", "also-missing"]
"#;

fn write_path(root: &Path, name: &str, text: &str) {
    let dir = root.join("paths");
    fs::create_dir_all(&dir).expect("mkdir");
    fs::write(dir.join(name), text).expect("write");
}

#[test]
fn no_paths_directory_means_no_paths_and_no_issues() {
    let root = TestRoot::new();
    let core = root.core("happy");
    let before = core.catalog_issues().len();
    assert!(core.list_paths().expect("paths").is_empty());
    assert_eq!(core.catalog_issues().len(), before);
}

#[test]
fn unknown_and_repeated_problems_are_skipped_and_reported() {
    let root = TestRoot::new();
    let core = root.core("happy");
    let before = core.catalog_issues().len();
    write_path(root.path(), "starter.toml", STARTER);

    let paths = core.list_paths().expect("paths");
    assert_eq!(paths.len(), 1);
    let path = &paths[0];
    assert_eq!(
        (path.id.as_str(), path.title.as_str()),
        ("starter", "Starter Path")
    );
    assert_eq!(path.description, "A short path.");
    // The stage whose problems were all skipped is dropped; order is preserved.
    let stages: Vec<_> = path
        .stages
        .iter()
        .map(|s| (s.title.as_str(), s.problems.clone()))
        .collect();
    assert_eq!(
        stages,
        [
            (
                "Warm-up",
                vec!["add-two".to_owned(), "echo-server".to_owned()]
            ),
            ("Harder", vec!["hard-one".to_owned()]),
        ]
    );
    assert_eq!(path.stages[0].description, "First steps.");
    assert_eq!((path.problem_count, path.solved_count), (3, 0));

    let issues = core.list_catalog_issues();
    let mine: Vec<_> = issues
        .iter()
        .filter(|i| i.path.ends_with("starter.toml"))
        .map(|i| i.message.as_str())
        .collect();
    assert_eq!(issues.len(), before + 4, "{issues:?}");
    assert_eq!(mine.len(), 4);
    assert!(mine[0].contains("`not-a-problem`") && mine[0].contains("not a known problem"));
    assert!(mine[1].contains("`add-two` again"));
    // A problem the catalog skipped as malformed is unknown to paths too.
    assert!(mine[2].contains("`wrong-id`"));
    assert!(mine[3].contains("`also-missing`"));
}

#[test]
fn malformed_path_files_are_reported_and_never_break_the_others() {
    let root = TestRoot::new();
    let core = root.core("happy");
    write_path(root.path(), "starter.toml", STARTER);
    write_path(root.path(), "broken.toml", "id = \"broken\"\ntitle = ");
    write_path(
        root.path(),
        "misnamed.toml",
        "id = \"other\"\ntitle = \"T\"\n[[stages]]\ntitle = \"S\"\nproblems = [\"add-two\"]\n",
    );
    write_path(
        root.path(),
        "untitled.toml",
        "id = \"untitled\"\ntitle = \" \"\n[[stages]]\ntitle = \"S\"\nproblems = []\n",
    );
    write_path(
        root.path(),
        "no-stages.toml",
        "id = \"no-stages\"\ntitle = \"T\"\n",
    );
    write_path(
        root.path(),
        "blank-stage.toml",
        "id = \"blank-stage\"\ntitle = \"T\"\n[[stages]]\ntitle = \"\"\nproblems = [\"add-two\"]\n",
    );
    write_path(
        root.path(),
        "Bad Name.toml",
        "id = \"Bad Name\"\ntitle = \"T\"\n[[stages]]\ntitle = \"S\"\n",
    );
    // Not path files: ignored without comment.
    write_path(root.path(), "README.md", "notes\n");
    write_path(root.path(), "_draft.toml", "not even toml [");
    fs::create_dir_all(root.path().join("paths").join("nested.toml")).expect("mkdir");

    let paths = core.list_paths().expect("paths");
    assert_eq!(
        paths.iter().map(|p| p.id.as_str()).collect::<Vec<_>>(),
        ["starter"]
    );

    let issues = core.list_catalog_issues();
    let message = |file: &str| {
        issues
            .iter()
            .find(|i| i.path.ends_with(file))
            .unwrap_or_else(|| panic!("no issue for {file}: {issues:?}"))
            .message
            .clone()
    };
    assert!(!message("broken.toml").is_empty());
    assert!(message("misnamed.toml").contains("does not equal the file name"));
    assert_eq!(message("untitled.toml"), "title is empty");
    assert!(message("no-stages.toml").contains("at least one [[stages]]"));
    assert_eq!(message("blank-stage.toml"), "stage 1 has an empty title");
    assert!(message("Bad Name.toml").contains("must use only"));
    assert!(!issues.iter().any(|i| i.path.contains("_draft")
        || i.path.contains("README")
        || i.path.contains("nested")));
}

fn one_stage(extra: &str) -> String {
    format!(
        "id = \"deco\"\ntitle = \"T\"\n[[stages]]\ntitle = \"S\"\nproblems = [\"add-two\"]\n{extra}"
    )
}

fn stage_fields(core: &learnai_core::Core) -> (Option<String>, Option<String>) {
    let paths = core.list_paths().expect("paths");
    assert_eq!(paths.len(), 1);
    assert_eq!(paths[0].stages.len(), 1);
    let stage = &paths[0].stages[0];
    (stage.scene.clone(), stage.color.clone())
}

fn deco_issues(core: &learnai_core::Core) -> Vec<String> {
    core.list_catalog_issues()
        .into_iter()
        .filter(|i| i.path.ends_with("deco.toml"))
        .map(|i| i.message)
        .collect()
}

#[test]
fn stage_display_fields_round_trip() {
    let root = TestRoot::new();
    let core = root.core("happy");
    let text = one_stage("scene = \"stage-01-trailhead\"\ncolor = \"#ed9b11\"\n");
    write_path(root.path(), "deco.toml", &text);
    assert_eq!(
        stage_fields(&core),
        (
            Some("stage-01-trailhead".to_owned()),
            Some("#ed9b11".to_owned())
        )
    );
    assert!(deco_issues(&core).is_empty());
}

#[test]
fn stage_color_is_lowercased() {
    let root = TestRoot::new();
    let core = root.core("happy");
    write_path(
        root.path(),
        "deco.toml",
        &one_stage("color = \"#ED9B1F\"\n"),
    );
    assert_eq!(stage_fields(&core), (None, Some("#ed9b1f".to_owned())));
    assert!(deco_issues(&core).is_empty());
}

#[test]
fn a_bad_scene_or_color_is_dropped_with_one_issue_and_the_stage_stays() {
    let root = TestRoot::new();
    let core = root.core("happy");
    write_path(
        root.path(),
        "deco.toml",
        &one_stage("scene = \"Not A Slug\"\ncolor = \"#12345g\"\n"),
    );
    assert_eq!(stage_fields(&core), (None, None));
    let issues = deco_issues(&core);
    assert_eq!(issues.len(), 2, "{issues:?}");
    assert!(issues[0].contains("stage `S`") && issues[0].contains("`Not A Slug`"));
    assert!(issues[1].contains("stage `S`") && issues[1].contains("`#12345g`"));

    // A short color is bad too; issues are recomputed on each load.
    write_path(root.path(), "deco.toml", &one_stage("color = \"#fff\"\n"));
    assert_eq!(stage_fields(&core), (None, None));
    assert_eq!(deco_issues(&core).len(), 1);
}

#[test]
fn a_stage_without_display_fields_is_unchanged() {
    let root = TestRoot::new();
    let core = root.core("happy");
    write_path(root.path(), "deco.toml", &one_stage(""));
    assert_eq!(stage_fields(&core), (None, None));
    assert!(deco_issues(&core).is_empty());
}

#[test]
fn paths_are_sorted_by_file_name_and_reloaded_on_every_call() {
    let root = TestRoot::new();
    let core = root.core("happy");
    let one = |id: &str| {
        format!(
            "id = \"{id}\"\ntitle = \"{id}\"\n[[stages]]\ntitle = \"S\"\nproblems = [\"add-two\"]\n"
        )
    };
    write_path(root.path(), "zeta.toml", &one("zeta"));
    write_path(root.path(), "alpha.toml", &one("alpha"));
    let ids = |core: &learnai_core::Core| -> Vec<String> {
        core.list_paths()
            .expect("paths")
            .into_iter()
            .map(|p| p.id)
            .collect()
    };
    assert_eq!(ids(&core), ["alpha", "zeta"]);

    fs::remove_file(root.path().join("paths").join("alpha.toml")).expect("remove");
    write_path(root.path(), "mid.toml", &one("mid"));
    assert_eq!(ids(&core), ["mid", "zeta"]);
}

#[test]
fn progress_counts_solved_problems_on_the_path() {
    let root = TestRoot::new();
    let core = root.core("happy");
    write_path(root.path(), "starter.toml", STARTER);

    // An attempt is not progress.
    core.write_solution("hard-one", Framework::Torch, "x = 2\n")
        .expect("write");
    assert_eq!(core.list_paths().expect("paths")[0].solved_count, 0);

    let events = Collector::<RunEvent>::new();
    core.start_run(
        "add-two",
        Framework::Torch,
        RunMode::Submit,
        "local",
        events.sink(),
    )
    .expect("start");
    events.wait_for_end();

    let path = &core.list_paths().expect("paths")[0];
    assert_eq!((path.problem_count, path.solved_count), (3, 1));
}

#[test]
fn the_shipped_starter_path_is_valid_against_the_real_catalog() {
    let repo = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let mut loader = learnai_core::catalog::CatalogLoader::new(&repo);
    let catalog = loader.load();
    let (paths, issues) = learnai_core::paths::load_paths(&repo, &catalog);
    assert!(issues.is_empty(), "{issues:?}");
    let path = paths
        .iter()
        .find(|p| p.id == "inference-75")
        .expect("inference-75");
    // The path grows with the catalog; it has always had at least three stages.
    assert!(path.stages.len() >= 3);
    assert!(path.stages.iter().all(|s| !s.description.is_empty()));
    // Nothing is taught before its prerequisites.
    let order: Vec<&str> = path.problem_ids().collect();
    for (position, id) in order.iter().enumerate() {
        let problem = catalog.problem(id).expect("problem");
        for prerequisite in &problem.prerequisites {
            if let Some(at) = order.iter().position(|other| other == prerequisite) {
                assert!(at < position, "{prerequisite} must come before {id}");
            }
        }
    }
}
