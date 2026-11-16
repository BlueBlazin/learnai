//! Statement assets: data URLs, path traversal, file types, and the size cap.

mod common;

use std::fs;
use std::path::PathBuf;

use common::TestRoot;
use learnai_core::assets::MAX_ASSET_BYTES;
use learnai_core::{Core, CoreError};

const SVG: &str =
    "<svg xmlns=\"http://www.w3.org/2000/svg\" viewBox=\"0 0 4 4\"><path d=\"M0 0h4\"/></svg>\n";
const PNG: &[u8] = b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR";

fn setup() -> (TestRoot, Core, PathBuf) {
    let root = TestRoot::new();
    let core = root.core("happy");
    let assets = root.path().join("problems/basics/add-two/assets");
    fs::create_dir_all(assets.join("sub")).expect("mkdir");
    fs::write(assets.join("figure.svg"), SVG).expect("write");
    fs::write(assets.join("sub").join("Plot.PNG"), PNG).expect("write");
    fs::write(assets.join("notes.txt"), "secret\n").expect("write");
    (root, core, assets)
}

fn invalid(result: learnai_core::Result<String>) -> String {
    match result {
        Err(CoreError::InvalidArgument(message)) => message,
        other => panic!("expected InvalidArgument, got {other:?}"),
    }
}

#[test]
fn svg_and_png_come_back_as_data_urls() {
    let (_root, core, _assets) = setup();
    assert_eq!(
        core.get_problem_asset("add-two", "figure.svg")
            .expect("svg"),
        // base64 of SVG, checked against `base64 < figure.svg`.
        "data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCA0IDQiPjxwYXRoIGQ9Ik0wIDBoNCIvPjwvc3ZnPgo="
    );
    assert_eq!(
        core.get_problem_asset("add-two", "sub/Plot.PNG")
            .expect("png"),
        "data:image/png;base64,iVBORw0KGgoAAAANSUhEUg=="
    );
}

#[test]
fn traversal_is_rejected_in_every_spelling() {
    let (root, core, _assets) = setup();
    // The targets exist, so only the path check stands between the caller and the file.
    assert!(
        root.path()
            .join("problems/basics/add-two/statement.md")
            .is_file()
    );
    for path in [
        "../statement.md",
        "../problem.toml",
        "sub/../../statement.md",
        "../assets/figure.svg",
        "./figure.svg",
        "..\\statement.md",
        "sub\\Plot.PNG",
        "",
    ] {
        let message = invalid(core.get_problem_asset("add-two", path));
        assert!(
            message.contains("invalid asset path"),
            "{path:?}: {message}"
        );
    }
    let absolute = root
        .path()
        .join("problems/basics/add-two/assets/figure.svg");
    invalid(core.get_problem_asset("add-two", &absolute.to_string_lossy()));
    // The problem id is looked up in the catalog, never joined into a path.
    assert!(matches!(
        core.get_problem_asset("../basics/add-two", "figure.svg"),
        Err(CoreError::ProblemNotFound(_))
    ));
}

#[cfg(unix)]
#[test]
fn a_symlink_cannot_lead_out_of_the_assets_directory() {
    let (root, core, assets) = setup();
    let outside = root.path().join("outside.svg");
    fs::write(&outside, SVG).expect("write");
    std::os::unix::fs::symlink(&outside, assets.join("escape.svg")).expect("symlink");
    std::os::unix::fs::symlink(root.path(), assets.join("up")).expect("symlink");
    std::os::unix::fs::symlink(assets.join("figure.svg"), assets.join("alias.svg"))
        .expect("symlink");

    invalid(core.get_problem_asset("add-two", "escape.svg"));
    invalid(core.get_problem_asset("add-two", "up/outside.svg"));
    // A link that stays inside is fine.
    assert!(core.get_problem_asset("add-two", "alias.svg").is_ok());
}

#[test]
fn only_svg_and_png_are_served() {
    let (_root, core, assets) = setup();
    for name in [
        "notes.txt",
        "figure.svg.txt",
        "figure",
        "page.html",
        "a.jpg",
    ] {
        fs::write(assets.join(name), "x").expect("write");
        let message = invalid(core.get_problem_asset("add-two", name));
        assert!(message.contains("only .svg and .png"), "{name}: {message}");
    }
    // The extension alone is not trusted.
    fs::write(assets.join("fake.png"), "<html>").expect("write");
    assert!(invalid(core.get_problem_asset("add-two", "fake.png")).contains("not a PNG"));
    fs::write(assets.join("binary.svg"), [0xff, 0xfe, 0x00]).expect("write");
    assert!(invalid(core.get_problem_asset("add-two", "binary.svg")).contains("not UTF-8"));
}

#[test]
fn missing_files_and_directories_are_not_found() {
    let (_root, core, assets) = setup();
    fs::create_dir_all(assets.join("dir.svg")).expect("mkdir");
    for path in ["nope.svg", "sub/nope.png", "dir.svg"] {
        let error = core.get_problem_asset("add-two", path).expect_err(path);
        assert!(
            matches!(error, CoreError::MissingFile(_)),
            "{path}: {error}"
        );
        assert!(error.is_not_found());
    }
    // A problem with no assets directory at all.
    assert!(matches!(
        core.get_problem_asset("hard-one", "figure.svg"),
        Err(CoreError::MissingFile(_))
    ));
}

#[test]
fn files_over_the_size_cap_are_refused() {
    let (_root, core, assets) = setup();
    let limit = usize::try_from(MAX_ASSET_BYTES).expect("limit fits");
    fs::write(assets.join("exact.svg"), " ".repeat(limit)).expect("write");
    fs::write(assets.join("huge.svg"), " ".repeat(limit + 1)).expect("write");

    assert!(core.get_problem_asset("add-two", "exact.svg").is_ok());
    let error = core
        .get_problem_asset("add-two", "huge.svg")
        .expect_err("too large");
    assert!(matches!(
        error,
        CoreError::AssetTooLarge { size, limit, .. } if size == limit + 1
    ));
    assert!(error.is_invalid_request());
    assert!(error.to_string().contains("the limit is 1048576 bytes"));
}
