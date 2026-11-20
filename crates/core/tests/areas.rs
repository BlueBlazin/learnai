//! Areas: loading, validation, and catalog issues. Also the track fields that modules
//! own (`scene`, `color`, `core`), and the shipped content.

mod common;

use std::fs;
use std::path::Path;

use common::TestRoot;
use learnai_core::areas::load_areas;
use learnai_core::catalog::CatalogLoader;

fn write_area(root: &Path, name: &str, text: &str) {
    let dir = root.join("areas");
    fs::create_dir_all(&dir).expect("mkdir");
    fs::write(dir.join(name), text).expect("write");
}

fn write_track(root: &Path, track: &str, extra: &str) {
    let (title, order) = if track == "basics" {
        ("Basics", 10)
    } else {
        ("Advanced", 20)
    };
    fs::write(
        root.join("problems").join(track).join("track.toml"),
        format!("id = \"{track}\"\ntitle = \"{title}\"\norder = {order}\n{extra}"),
    )
    .expect("write");
}

const VALID: &str = r#"
id = "starter"
title = "Starter Area"
description = "A short area."
modules = ["basics", "advanced"]
"#;

#[test]
fn no_areas_directory_means_no_areas_and_no_issues() {
    let root = TestRoot::new();
    let core = root.core("happy");
    let before = core.catalog_issues().len();
    assert!(core.list_areas().expect("areas").is_empty());
    assert_eq!(core.catalog_issues().len(), before);
}

#[test]
fn a_valid_area_lists_its_modules_in_order() {
    let root = TestRoot::new();
    let core = root.core("happy");
    let before = core.catalog_issues().len();
    write_area(root.path(), "starter.toml", VALID);

    let areas = core.list_areas().expect("areas");
    assert_eq!(areas.len(), 1);
    assert_eq!(areas[0].id, "starter");
    assert_eq!(areas[0].title, "Starter Area");
    assert_eq!(areas[0].description, "A short area.");
    assert_eq!(areas[0].modules, ["basics", "advanced"]);
    assert_eq!(core.catalog_issues().len(), before);
}

#[test]
fn areas_are_sorted_by_file_name() {
    let root = TestRoot::new();
    let core = root.core("happy");
    write_area(
        root.path(),
        "b-second.toml",
        "id = \"b-second\"\ntitle = \"B\"\nmodules = [\"advanced\"]\n",
    );
    write_area(
        root.path(),
        "a-first.toml",
        "id = \"a-first\"\ntitle = \"A\"\nmodules = [\"basics\"]\n",
    );
    let ids: Vec<_> = core
        .list_areas()
        .expect("areas")
        .into_iter()
        .map(|a| a.id)
        .collect();
    assert_eq!(ids, ["a-first", "b-second"]);
}

#[test]
fn an_unknown_module_is_dropped_and_reported() {
    let root = TestRoot::new();
    let core = root.core("happy");
    let before = core.catalog_issues().len();
    write_area(
        root.path(),
        "starter.toml",
        "id = \"starter\"\ntitle = \"S\"\nmodules = [\"basics\", \"nope\", \"advanced\", \"basics\"]\n",
    );

    let areas = core.list_areas().expect("areas");
    assert_eq!(areas[0].modules, ["basics", "advanced"]);

    let issues = core.list_catalog_issues();
    assert_eq!(issues.len(), before + 2, "{issues:?}");
    let mine: Vec<_> = issues
        .iter()
        .filter(|i| i.path.ends_with("starter.toml"))
        .map(|i| i.message.as_str())
        .collect();
    assert!(mine[0].contains("`nope`") && mine[0].contains("not a known track"));
    assert!(mine[1].contains("`basics` again"));
}

#[test]
fn a_track_in_two_areas_stays_with_the_first_area() {
    let root = TestRoot::new();
    let core = root.core("happy");
    write_area(
        root.path(),
        "a-one.toml",
        "id = \"a-one\"\ntitle = \"One\"\nmodules = [\"basics\"]\n",
    );
    write_area(
        root.path(),
        "b-two.toml",
        "id = \"b-two\"\ntitle = \"Two\"\nmodules = [\"basics\", \"advanced\"]\n",
    );

    let areas = core.list_areas().expect("areas");
    assert_eq!(areas[0].modules, ["basics"]);
    assert_eq!(areas[1].modules, ["advanced"]);
    let issues = core.list_catalog_issues();
    assert!(issues.iter().any(|i| i.path.ends_with("b-two.toml")
        && i.message.contains("`basics`")
        && i.message.contains("`a-one`")));
}

#[test]
fn an_area_with_no_usable_modules_is_dropped_and_its_claims_are_released() {
    let root = TestRoot::new();
    let core = root.core("happy");
    write_area(
        root.path(),
        "a-empty.toml",
        "id = \"a-empty\"\ntitle = \"Empty\"\nmodules = [\"ghost\"]\n",
    );
    write_area(
        root.path(),
        "b-later.toml",
        "id = \"b-later\"\ntitle = \"Later\"\nmodules = [\"basics\"]\n",
    );
    let areas = core.list_areas().expect("areas");
    assert_eq!(
        areas.iter().map(|a| a.id.as_str()).collect::<Vec<_>>(),
        ["b-later"]
    );
    let issues = core.list_catalog_issues();
    assert!(
        issues
            .iter()
            .any(|i| i.path.ends_with("a-empty.toml") && i.message.contains("`ghost`"))
    );
    assert!(
        issues
            .iter()
            .any(|i| i.path.ends_with("a-empty.toml") && i.message.contains("no usable modules"))
    );
}

#[test]
fn malformed_area_files_are_reported_and_never_break_the_others() {
    let root = TestRoot::new();
    let core = root.core("happy");
    write_area(root.path(), "starter.toml", VALID);
    write_area(root.path(), "broken.toml", "id = \"broken\"\ntitle = ");
    write_area(
        root.path(),
        "misnamed.toml",
        "id = \"other\"\ntitle = \"T\"\nmodules = [\"basics\"]\n",
    );
    write_area(
        root.path(),
        "untitled.toml",
        "id = \"untitled\"\ntitle = \" \"\nmodules = [\"basics\"]\n",
    );
    write_area(root.path(), "README.md", "notes\n");
    write_area(
        root.path(),
        "Bad Name.toml",
        "id = \"Bad Name\"\ntitle = \"T\"\nmodules = [\"basics\"]\n",
    );
    write_area(root.path(), "_draft.toml", "not even toml [");
    // A directory named like a file is not an area file.
    fs::create_dir_all(root.path().join("areas").join("nested.toml")).expect("mkdir");

    let areas = core.list_areas().expect("areas");
    assert_eq!(
        areas.iter().map(|a| a.id.as_str()).collect::<Vec<_>>(),
        ["starter"]
    );
    let issues = core.list_catalog_issues();
    let message = |file: &str| {
        issues
            .iter()
            .find(|i| i.path.ends_with(file))
            .unwrap_or_else(|| panic!("no issue for {file}: {issues:?}"))
            .message
            .as_str()
    };
    assert!(!message("broken.toml").is_empty());
    assert!(message("misnamed.toml").contains("does not equal the file name"));
    assert!(message("untitled.toml").contains("title is empty"));
    assert!(message("Bad Name.toml").contains("must use only"));
    assert!(!issues.iter().any(|i| i.path.ends_with("README.md")
        || i.path.ends_with("_draft.toml")
        || i.path.ends_with("nested.toml")));
}

#[test]
fn a_valid_module_scene_color_and_core_load() {
    let root = TestRoot::new();
    write_track(
        root.path(),
        "basics",
        "scene = \"trailhead\"\ncolor = \"#ED9B11\"\ncore = [\"echo-server\", \"add-two\"]\n",
    );
    let core = root.core("happy");
    let before = core.catalog_issues().len();
    let tracks = core.list_tracks().expect("tracks");
    assert_eq!(tracks[0].scene.as_deref(), Some("trailhead"));
    // Stored lowercase.
    assert_eq!(tracks[0].color.as_deref(), Some("#ed9b11"));
    assert_eq!(tracks[0].core, ["echo-server", "add-two"]);
    assert_eq!(core.catalog_issues().len(), before);
}

#[test]
fn a_track_without_the_new_fields_has_none_and_an_empty_core() {
    let root = TestRoot::new();
    let core = root.core("happy");
    for track in core.list_tracks().expect("tracks") {
        assert_eq!((track.scene, track.color), (None, None));
        assert!(track.core.is_empty());
    }
}

#[test]
fn a_bad_scene_or_color_is_dropped_and_reported() {
    let root = TestRoot::new();
    write_track(
        root.path(),
        "basics",
        "scene = \"Not A Slug\"\ncolor = \"red\"\n",
    );
    write_track(root.path(), "advanced", "color = \"#12345\"\n");
    let core = root.core("happy");

    let tracks = core.list_tracks().expect("tracks");
    assert_eq!(tracks.len(), 2);
    for track in &tracks {
        assert_eq!((&track.scene, &track.color), (&None, &None));
    }
    let issues = core.list_catalog_issues();
    let mine = |track: &str| -> Vec<&str> {
        issues
            .iter()
            .filter(|i| i.path.ends_with(&format!("{track}/track.toml")))
            .map(|i| i.message.as_str())
            .collect()
    };
    let basics = mine("basics");
    assert_eq!(basics.len(), 2, "{issues:?}");
    assert!(basics[0].contains("scene `Not A Slug`"));
    assert!(basics[1].contains("color `red`"));
    assert!(mine("advanced")[0].contains("color `#12345`"));
}

#[test]
fn core_ids_from_another_track_or_repeated_are_dropped_and_reported() {
    let root = TestRoot::new();
    write_track(
        root.path(),
        "basics",
        "core = [\"add-two\", \"hard-one\", \"ghost\", \"echo-server\", \"add-two\"]\n",
    );
    let core = root.core("happy");

    let tracks = core.list_tracks().expect("tracks");
    assert_eq!(tracks[0].core, ["add-two", "echo-server"]);
    let issues = core.list_catalog_issues();
    let mine: Vec<_> = issues
        .iter()
        .filter(|i| i.path.ends_with("basics/track.toml"))
        .map(|i| i.message.as_str())
        .collect();
    assert_eq!(mine.len(), 3, "{issues:?}");
    assert!(mine[0].contains("`hard-one`") && mine[0].contains("not a problem of track"));
    assert!(mine[1].contains("`ghost`"));
    assert!(mine[2].contains("`add-two` again"));
}

#[test]
fn track_field_issues_survive_the_parse_cache() {
    let root = TestRoot::new();
    write_track(root.path(), "basics", "scene = \"Bad Scene\"\n");
    let mut loader = CatalogLoader::new(root.path());
    let first = loader.load().issues().len();
    assert!(first >= 1);
    // The second load answers from the cache and reports the same issues.
    assert_eq!(loader.load().issues().len(), first);
}

#[test]
fn the_shipped_content_has_one_area_of_fifteen_modules_and_no_issues() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let catalog = CatalogLoader::new(&root).load();
    assert!(catalog.issues().is_empty(), "{:?}", catalog.issues());

    let (areas, issues) = load_areas(&root, &catalog);
    assert!(issues.is_empty(), "{issues:?}");
    assert_eq!(areas.len(), 1);
    assert_eq!(areas[0].id, "inference-and-serving");
    assert_eq!(areas[0].modules.len(), 15);

    let core_total: usize = catalog.tracks().iter().map(|t| t.core.len()).sum();
    assert_eq!(core_total, 84);
    for track in catalog.tracks() {
        assert!(track.scene.is_some(), "{} has no scene", track.id);
        assert!(track.color.is_some(), "{} has no color", track.id);
    }
}
