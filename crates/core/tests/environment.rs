//! Environment status and installs against the fake uv.

// The fakes these tests drive are shell and Perl scripts started as programs, which
// Windows cannot do. Windows needs its own process tests with a fake that is not a script.
#![cfg(unix)]

mod common;

use std::fs;
use std::sync::Arc;

use common::{Collector, TestRoot, fixtures, test_limits};
use learnai_core::{
    Core, CoreError, CoreOptions, Extra, Framework, FrameworkEnv, InstallEvent, LOCAL_TARGET,
    RunMode,
};

fn install(core: &Core, framework: Framework) -> Vec<InstallEvent> {
    let collector = Collector::new();
    core.env_install(framework, collector.sink())
        .expect("start install");
    let events = collector.wait_for("the done event", |events| {
        events
            .iter()
            .any(|e| matches!(e, InstallEvent::Done { .. }))
    });
    assert_eq!(
        events
            .iter()
            .filter(|e| matches!(e, InstallEvent::Done { .. }))
            .count(),
        1
    );
    assert!(
        matches!(events.last(), Some(InstallEvent::Done { .. })),
        "done is last: {events:?}"
    );
    events
}

fn uv_log(root: &TestRoot) -> Vec<String> {
    fs::read_to_string(root.path().join("python/.fake-uv.log"))
        .unwrap_or_default()
        .lines()
        .filter(|line| !line.starts_with("run"))
        .map(str::to_owned)
        .collect()
}

fn env_of(core: &Core, framework: Framework) -> FrameworkEnv {
    let status = core.env_status().expect("status");
    status
        .frameworks
        .into_iter()
        .find(|f| f.framework == framework)
        .expect("framework entry")
}

#[test]
fn status_reports_uv_python_and_base_frameworks() {
    let root = TestRoot::new();
    let core = root.core_with_fake_uv();
    let status = core.env_status().expect("status");

    assert_eq!(status.uv.as_deref(), Some("0.13.8"));
    assert_eq!(status.python.as_deref(), Some("3.15.2"));
    assert_eq!(status.root, root.path().to_string_lossy());
    let frameworks: Vec<_> = status
        .frameworks
        .iter()
        .map(|f| (f.framework, f.installed, f.version.as_deref()))
        .collect();
    assert_eq!(
        frameworks,
        [
            (Framework::Torch, true, Some("2.15.0")),
            (Framework::Jax, false, None),
            (Framework::Mlx, false, None),
            (Framework::Python, true, Some("3.15.2")),
        ]
    );
    assert!(core.enabled_extras().expect("extras").is_empty());
}

#[test]
fn status_without_uv_reports_nothing_installed() {
    let root = TestRoot::new();
    let core = Core::open(CoreOptions {
        uv_program: Some("/definitely/not/uv".into()),
        ..CoreOptions::new(root.path())
    })
    .expect("open core");
    let status = core.env_status().expect("status");
    assert_eq!(status.uv, None);
    assert_eq!(status.python, None);
    assert!(
        status
            .frameworks
            .iter()
            .all(|f| !f.installed && f.version.is_none())
    );
    assert_eq!(status.frameworks.len(), 4);
}

#[test]
fn installs_accumulate_extras_instead_of_replacing_them() {
    let root = TestRoot::new();
    let core = root.core_with_fake_uv();

    let events = install(&core, Framework::Jax);
    assert_eq!(
        events.last(),
        Some(&InstallEvent::Done {
            ok: true,
            message: None
        })
    );
    let text: Vec<_> = events
        .iter()
        .filter_map(|e| match e {
            InstallEvent::Log { text } => Some(text.as_str()),
            _ => None,
        })
        .collect();
    assert!(
        text[0].starts_with("$ ") && text[0].ends_with("--extra jax"),
        "{text:?}"
    );
    assert!(
        text.contains(&"Resolved 42 packages in 1ms"),
        "stdout is streamed: {text:?}"
    );
    assert!(
        text.contains(&" + jax==9.9.9"),
        "stderr is streamed: {text:?}"
    );
    assert_eq!(core.enabled_extras().expect("extras"), [Extra::Jax]);
    assert!(env_of(&core, Framework::Jax).installed);

    install(&core, Framework::Mlx);
    assert_eq!(
        uv_log(&root),
        ["sync jax", "sync jax mlx"],
        "the second sync passes the full set"
    );
    assert_eq!(
        core.enabled_extras().expect("extras"),
        [Extra::Jax, Extra::Mlx]
    );
    assert_eq!(
        env_of(&core, Framework::Jax).version.as_deref(),
        Some("0.12.0.1")
    );
    assert_eq!(
        env_of(&core, Framework::Mlx).version.as_deref(),
        Some("0.32.4")
    );

    // Base frameworks sync the same full set and enable nothing new.
    install(&core, Framework::Torch);
    assert_eq!(
        uv_log(&root).last().map(String::as_str),
        Some("sync jax mlx")
    );
    assert_eq!(
        core.enabled_extras().expect("extras"),
        [Extra::Jax, Extra::Mlx]
    );
}

#[test]
fn enabled_extras_are_passed_to_every_run_and_survive_a_restart() {
    let root = TestRoot::new();
    install(&root.core_with_fake_uv(), Framework::Mlx);

    // A new core over the same root reads the setting back from the database.
    let core = root.core_with_fake_uv();
    assert_eq!(core.enabled_extras().expect("extras"), [Extra::Mlx]);
    let collector = Collector::new();
    core.start_run(
        "add-two",
        Framework::Torch,
        RunMode::Run,
        LOCAL_TARGET,
        collector.sink(),
    )
    .expect("start run");
    collector.wait_for_end();
    let log = fs::read_to_string(root.path().join("python/.fake-uv.log")).expect("uv log");
    assert_eq!(log.lines().last(), Some("run mlx"));
}

#[test]
fn extras_installed_outside_the_app_are_adopted_not_stripped() {
    let root = TestRoot::new();
    fs::write(root.path().join("python/.fake-venv"), " mlx\n")
        .expect("simulate a manual uv sync --extra mlx");
    let core = root.core_with_fake_uv();

    // Even without a status call first, installing jax must keep mlx.
    install(&core, Framework::Jax);
    assert_eq!(uv_log(&root), ["sync jax mlx"]);
    assert!(env_of(&core, Framework::Mlx).installed);
    assert!(env_of(&core, Framework::Jax).installed);
}

#[test]
fn a_failed_install_reports_done_false_and_enables_nothing() {
    let root = TestRoot::new();
    fs::write(root.path().join("python/.fail-sync"), "").expect("write");
    let core = root.core_with_fake_uv();

    let events = install(&core, Framework::Jax);
    assert_eq!(
        events.last(),
        Some(&InstallEvent::Done {
            ok: false,
            message: Some("uv sync exited with code 1".into())
        })
    );
    assert!(events.contains(&InstallEvent::Log {
        text: "error: No solution found when resolving dependencies".into()
    }));
    assert!(core.enabled_extras().expect("extras").is_empty());

    // The failure does not leave the install lock held.
    fs::remove_file(root.path().join("python/.fail-sync")).expect("remove");
    assert_eq!(
        install(&core, Framework::Jax).last(),
        Some(&InstallEvent::Done {
            ok: true,
            message: None
        })
    );
}

#[test]
fn a_missing_uv_fails_the_install_with_done_false() {
    let root = TestRoot::new();
    let core = Core::open(CoreOptions {
        uv_program: Some("/definitely/not/uv".into()),
        harness: Some(Arc::new(root.harness("happy"))),
        limits: test_limits(),
        ..CoreOptions::new(root.path())
    })
    .expect("open core");
    let events = install(&core, Framework::Jax);
    assert!(matches!(
        events.last(),
        Some(InstallEvent::Done {
            ok: false,
            message: Some(_)
        })
    ));
}

#[test]
fn only_one_install_runs_at_a_time() {
    let root = TestRoot::new();
    // A uv whose sync blocks until released.
    let slow_uv = root.path().join("slow_uv.sh");
    let gate = root.path().join("release");
    fs::write(
        &slow_uv,
        format!(
            "#!/bin/sh\nif [ \"$1\" = sync ]; then while [ ! -e '{}' ]; do sleep 0.02; done; fi\nexec sh '{}' \"$@\"\n",
            gate.display(),
            fixtures().join("fake_uv.sh").display()
        ),
    )
    .expect("write");
    let core = Core::open(CoreOptions {
        uv_program: Some(make_executable(&slow_uv)),
        ..CoreOptions::new(root.path())
    })
    .expect("open core");

    let first = Collector::new();
    core.env_install(Framework::Jax, first.sink())
        .expect("start install");
    let second: Collector<InstallEvent> = Collector::new();
    assert!(matches!(
        core.env_install(Framework::Mlx, second.sink()),
        Err(CoreError::InstallInProgress)
    ));

    fs::write(&gate, "").expect("release");
    let events = first.wait_for("done", |e| {
        e.iter().any(|e| matches!(e, InstallEvent::Done { .. }))
    });
    assert_eq!(
        events.last(),
        Some(&InstallEvent::Done {
            ok: true,
            message: None
        })
    );
    assert!(second.snapshot().is_empty());
}

fn make_executable(path: &std::path::Path) -> std::path::PathBuf {
    use std::os::unix::fs::PermissionsExt;
    fs::set_permissions(path, fs::Permissions::from_mode(0o755)).expect("chmod");
    path.to_path_buf()
}
