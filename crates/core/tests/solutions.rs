//! Solution files: read (starter fallback), write, reset, and the status they imply.

mod common;

use std::fs;
use std::path::Path;

use common::TestRoot;
use learnai_core::{CoreError, Framework, ProblemStatus};

const STARTER: &str = "def add(a, b):\n    raise NotImplementedError\n";

#[test]
fn read_falls_back_to_the_starter() {
    let root = TestRoot::new();
    let core = root.core("happy");
    let file = core
        .read_solution("add-two", Framework::Torch)
        .expect("read");
    assert_eq!(file.code, STARTER);
    assert!(file.is_starter);
    assert!(
        Path::new(&file.path).ends_with("workspace/add-two/torch.py"),
        "{}",
        file.path
    );
    assert!(
        !root.path().join("workspace").exists(),
        "reading must not create files"
    );
}

#[test]
fn write_then_read_then_reset() {
    let root = TestRoot::new();
    let core = root.core("happy");
    let path = root.path().join("workspace/add-two/torch.py");

    core.write_solution(
        "add-two",
        Framework::Torch,
        "def add(a, b):\n    return a + b\n",
    )
    .expect("write");
    assert_eq!(
        fs::read_to_string(&path).expect("file"),
        "def add(a, b):\n    return a + b\n"
    );
    let file = core
        .read_solution("add-two", Framework::Torch)
        .expect("read");
    assert!(!file.is_starter);
    assert_eq!(file.code, "def add(a, b):\n    return a + b\n");
    // The other framework is untouched.
    assert!(
        core.read_solution("add-two", Framework::Jax)
            .expect("read")
            .is_starter
    );

    // Edits made in an external editor are picked up.
    fs::write(&path, "# edited elsewhere\n").expect("external edit");
    assert_eq!(
        core.read_solution("add-two", Framework::Torch)
            .expect("read")
            .code,
        "# edited elsewhere\n"
    );

    let reset = core
        .reset_solution("add-two", Framework::Torch)
        .expect("reset");
    assert_eq!(reset.code, STARTER);
    assert!(reset.is_starter);
    assert!(!path.exists());
    // Resetting again is fine.
    assert!(
        core.reset_solution("add-two", Framework::Torch)
            .expect("reset")
            .is_starter
    );
}

#[test]
fn status_becomes_attempted_only_when_the_file_differs_from_the_starter() {
    let root = TestRoot::new();
    let core = root.core("happy");
    let status = || core.get_problem("add-two").expect("detail").summary.status;

    core.write_solution(
        "add-two",
        Framework::Jax,
        "def add(a, b):\n    raise NotImplementedError  # jax\n",
    )
    .expect("write");
    assert!(
        core.read_solution("add-two", Framework::Jax)
            .expect("read")
            .is_starter
    );
    assert_eq!(status(), ProblemStatus::Todo);

    core.write_solution(
        "add-two",
        Framework::Jax,
        "def add(a, b):\n    return b + a\n",
    )
    .expect("write");
    assert_eq!(status(), ProblemStatus::Attempted);

    core.reset_solution("add-two", Framework::Jax)
        .expect("reset");
    assert_eq!(status(), ProblemStatus::Todo);
}

#[test]
fn unsupported_frameworks_and_unknown_problems_are_rejected() {
    let root = TestRoot::new();
    let core = root.core("happy");
    assert!(matches!(
        core.read_solution("add-two", Framework::Mlx),
        Err(CoreError::FrameworkNotSupported { .. })
    ));
    assert!(matches!(
        core.write_solution("add-two", Framework::Python, "x"),
        Err(CoreError::FrameworkNotSupported { .. })
    ));
    assert!(matches!(
        core.reset_solution("nope", Framework::Torch),
        Err(CoreError::ProblemNotFound(_))
    ));
    assert!(!root.path().join("workspace").exists());
}
