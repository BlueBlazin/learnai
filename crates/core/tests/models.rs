//! Model assets against the fake uv: listing, fetching (events, failure, cancel, one
//! job at a time), removal, and the extras every uv command passes.

mod common;

use std::fs;
use std::process::Command;
use std::time::{Duration, Instant};

use common::{Collector, TestRoot};
use learnai_core::{
    AssetEvent, AssetState, Core, CoreError, Extra, Framework, InstallEvent, LOCAL_TARGET,
    RunEvent, RunMode,
};

fn is_done(event: &AssetEvent) -> bool {
    matches!(event, AssetEvent::Done { .. })
}

/// Fetches `names` and returns the whole stream, checking its shape: `preparing` first,
/// exactly one `done`, and `done` last.
fn fetch(core: &Core, names: &[&str]) -> Vec<AssetEvent> {
    let collector = Collector::new();
    let names: Vec<String> = names.iter().map(|name| (*name).to_owned()).collect();
    let job_id = core
        .fetch_assets(&names, None, collector.sink())
        .expect("start fetch");
    assert!(!job_id.is_empty());
    let events = collector.wait_for("the done event", |events| events.iter().any(is_done));
    std::thread::sleep(Duration::from_millis(50));
    assert_eq!(
        collector.snapshot().len(),
        events.len(),
        "events after done"
    );
    assert_well_formed(&events);
    events
}

fn assert_well_formed(events: &[AssetEvent]) {
    assert!(
        matches!(events.first(), Some(AssetEvent::Preparing { .. })),
        "first event: {events:?}"
    );
    assert_eq!(events.iter().filter(|e| is_done(e)).count(), 1);
    assert!(is_done(events.last().expect("events")), "done is last");
}

/// The uv invocations so far, without the environment probes.
fn uv_log(root: &TestRoot) -> Vec<String> {
    fs::read_to_string(root.path().join("python/.fake-uv.log"))
        .unwrap_or_default()
        .lines()
        .map(str::to_owned)
        .collect()
}

fn state_of(core: &Core, name: &str) -> AssetState {
    core.list_assets()
        .expect("list")
        .assets
        .into_iter()
        .find(|asset| asset.name == name)
        .expect("asset")
        .state
}

fn process_alive(pid: &str) -> bool {
    Command::new("kill")
        .args(["-0", pid])
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
}

#[test]
fn listing_reports_declared_assets_under_the_content_root() {
    let root = TestRoot::new();
    let core = root.core_with_fake_uv();
    let listing = core.list_assets().expect("list");

    let models_dir = root.path().join(".learnai/models");
    assert_eq!(listing.models_dir, models_dir.to_string_lossy());
    assert_eq!(
        listing
            .assets
            .iter()
            .map(|a| (a.name.as_str(), a.state, a.size_mb))
            .collect::<Vec<_>>(),
        [
            ("other", AssetState::Missing, 3900.0),
            ("tiny", AssetState::Missing, 0.5)
        ]
    );
    let tiny = &listing.assets[1];
    assert_eq!(tiny.repo, "fake/tiny");
    assert_eq!(tiny.problems, ["add-two"]);
    assert_eq!(tiny.path, models_dir.join("tiny").to_string_lossy());
    assert_eq!(tiny.bytes_on_disk, 0);

    // Listing needs no extra and installs none.
    assert_eq!(uv_log(&root), ["run"]);
    assert!(core.enabled_extras().expect("extras").is_empty());

    // The catalog side agrees about which problem declares it.
    let problem = core.get_problem("add-two").expect("problem");
    assert_eq!(problem.summary.assets, ["tiny"]);
    assert!(
        core.get_problem("echo-server")
            .expect("problem")
            .summary
            .assets
            .is_empty()
    );
}

#[test]
fn a_fetch_streams_progress_and_makes_the_asset_present() {
    let root = TestRoot::new();
    let core = root.core_with_fake_uv();
    let events = fetch(&core, &["tiny"]);

    let kinds: Vec<_> = events
        .iter()
        .filter(|e| !matches!(e, AssetEvent::Log { .. } | AssetEvent::Preparing { .. }))
        .cloned()
        .collect();
    assert_eq!(
        kinds,
        [
            AssetEvent::Start {
                asset: "tiny".into(),
                total_bytes: 2048
            },
            AssetEvent::Progress {
                asset: "tiny".into(),
                file: "config.json".into(),
                done_bytes: 1024,
                total_bytes: 2048
            },
            AssetEvent::Progress {
                asset: "tiny".into(),
                file: "config.json".into(),
                done_bytes: 2048,
                total_bytes: 2048
            },
            AssetEvent::AssetEnd {
                asset: "tiny".into(),
                ok: true,
                message: None
            },
            AssetEvent::Done {
                ok: true,
                canceled: false,
                message: None
            },
        ]
    );
    // Non-protocol stdout and all of stderr arrive as log lines.
    let logs: Vec<_> = events
        .iter()
        .filter_map(|e| match e {
            AssetEvent::Log { text } => Some(text.as_str()),
            _ => None,
        })
        .collect();
    assert!(logs.contains(&"a plain progress line"), "{logs:?}");
    assert!(logs.contains(&"Fetching 1 files"), "{logs:?}");
    assert!(logs.iter().any(|l| l.starts_with("$ ")), "{logs:?}");

    assert_eq!(state_of(&core, "tiny"), AssetState::Present);
    assert_eq!(state_of(&core, "other"), AssetState::Missing);
}

#[test]
fn every_uv_command_passes_the_same_extras_once_models_is_installed() {
    let root = TestRoot::new();
    let core = root.core_with_fake_uv();

    let run = |core: &Core| {
        let collector: Collector<RunEvent> = Collector::new();
        core.start_run(
            "add-two",
            Framework::Torch,
            RunMode::Run,
            LOCAL_TARGET,
            collector.sink(),
        )
        .expect("start run");
        collector.wait_for_end();
    };
    let install = |core: &Core, framework: Framework| {
        let collector = Collector::new();
        core.env_install(framework, collector.sink())
            .expect("start install");
        collector.wait_for("the done event", |events| {
            events
                .iter()
                .any(|e| matches!(e, InstallEvent::Done { .. }))
        });
    };

    run(&core);
    fetch(&core, &["tiny"]);
    assert_eq!(core.enabled_extras().expect("extras"), [Extra::Models]);
    run(&core);
    core.list_assets().expect("list");
    fetch(&core, &["tiny"]);
    install(&core, Framework::Jax);
    run(&core);
    core.list_assets().expect("list");
    fetch(&core, &["other"]);

    // Probes (`run` with --no-sync) aside: the fetch adds `models`, and from then on
    // every run, listing, fetch, and sync carries it, so nothing is ever removed.
    let commands: Vec<String> = uv_log(&root);
    let meaningful: Vec<&str> = {
        // Drop the probes: they are the `run` lines directly before a fetch or a sync.
        let mut kept = Vec::new();
        let mut index = 0;
        while index < commands.len() {
            let line = commands[index].as_str();
            let next = commands.get(index + 1).map(String::as_str);
            let is_probe = line == "run"
                && matches!(next, Some(n) if n.starts_with("sync") || n.contains("models"))
                && index != 0;
            if !is_probe {
                kept.push(line);
            }
            index += 1;
        }
        kept
    };
    assert_eq!(
        meaningful,
        [
            "run",             // the first run: nothing installed yet
            "run models",      // the first fetch installs the extra
            "run models",      // run
            "run models",      // list
            "run models",      // fetch again
            "sync jax models", // installing JAX keeps models
            "run jax models",  // run
            "run jax models",  // list
            "run jax models",  // fetch
        ],
        "all commands: {commands:?}"
    );
    // The environment itself agrees: models was installed once and never removed.
    assert_eq!(
        fs::read_to_string(root.path().join("python/.fake-venv"))
            .expect("venv")
            .split_whitespace()
            .collect::<Vec<_>>(),
        ["jax", "models"]
    );
    assert_eq!(
        core.enabled_extras().expect("extras"),
        [Extra::Jax, Extra::Models]
    );

    // A new core over the same root remembers.
    let reopened = root.core_with_fake_uv();
    assert_eq!(
        reopened.enabled_extras().expect("extras"),
        [Extra::Jax, Extra::Models]
    );
}

#[test]
fn models_installed_outside_the_app_survive_a_framework_install() {
    let root = TestRoot::new();
    fs::write(root.path().join("python/.fake-venv"), " models\n")
        .expect("simulate `uv run --extra models learnai assets fetch` in a terminal");
    let core = root.core_with_fake_uv();

    let collector = Collector::new();
    core.env_install(Framework::Mlx, collector.sink())
        .expect("start install");
    collector.wait_for("the done event", |events| {
        events
            .iter()
            .any(|e| matches!(e, InstallEvent::Done { .. }))
    });
    assert!(
        uv_log(&root).contains(&"sync mlx models".to_owned()),
        "{:?}",
        uv_log(&root)
    );
    assert_eq!(
        core.enabled_extras().expect("extras"),
        [Extra::Mlx, Extra::Models]
    );
}

#[test]
fn a_failed_asset_ends_with_its_one_line_message_and_can_be_retried() {
    let root = TestRoot::new();
    let core = root.core_with_fake_uv();
    fs::write(root.path().join("python/.fetch-scenario"), "fail").expect("write");

    let events = fetch(&core, &["tiny"]);
    assert!(events.contains(&AssetEvent::AssetEnd {
        asset: "tiny".into(),
        ok: false,
        message: Some("the connection was reset".into())
    }));
    assert_eq!(
        events.last(),
        Some(&AssetEvent::Done {
            ok: false,
            canceled: false,
            message: Some("tiny: the connection was reset".into())
        })
    );
    assert_eq!(state_of(&core, "tiny"), AssetState::Missing);

    // The failure does not leave the job slot held.
    fs::remove_file(root.path().join("python/.fetch-scenario")).expect("remove");
    assert!(matches!(
        fetch(&core, &["tiny"]).last(),
        Some(AssetEvent::Done { ok: true, .. })
    ));
}

#[test]
fn an_unknown_asset_fails_with_what_the_cli_said() {
    let root = TestRoot::new();
    let core = root.core_with_fake_uv();
    let events = fetch(&core, &["nope"]);
    assert_eq!(
        events.last(),
        Some(&AssetEvent::Done {
            ok: false,
            canceled: false,
            message: Some(
                "learnai assets fetch: no asset named nope in the catalog (known: other, tiny)"
                    .into()
            )
        })
    );
}

#[test]
fn without_uv_a_fetch_ends_with_a_clear_message_and_listing_fails() {
    let root = TestRoot::new();
    let core = Core::open(learnai_core::CoreOptions {
        uv_program: Some("/definitely/not/uv".into()),
        ..learnai_core::CoreOptions::new(root.path())
    })
    .expect("open core");

    let error = core.list_assets().expect_err("no uv");
    assert!(error.to_string().contains("uv was not found"), "{error}");

    let events = fetch(&core, &["tiny"]);
    let Some(AssetEvent::Done {
        ok: false,
        canceled: false,
        message: Some(message),
    }) = events.last()
    else {
        panic!("unexpected end: {events:?}")
    };
    assert!(message.contains("uv was not found"), "{message}");
}

#[test]
fn bad_requests_start_nothing() {
    let root = TestRoot::new();
    let core = root.core_with_fake_uv();
    for names in [
        vec![],
        vec!["--all".to_owned()],
        vec!["tiny".to_owned(), "../x".to_owned()],
        vec!["a b".to_owned()],
    ] {
        let collector: Collector<AssetEvent> = Collector::new();
        let error = core
            .fetch_assets(&names, None, collector.sink())
            .expect_err("rejected");
        assert!(
            matches!(error, CoreError::InvalidArgument(_)),
            "{names:?}: {error}"
        );
        assert!(error.is_invalid_request());
        assert!(collector.snapshot().is_empty());
    }
    assert!(uv_log(&root).is_empty(), "nothing was run");
}

#[test]
fn cancel_kills_the_process_group_keeps_files_and_frees_the_job() {
    let root = TestRoot::new();
    let core = root.core_with_fake_uv();
    fs::write(root.path().join("python/.fetch-scenario"), "hang").expect("write");

    let collector = Collector::new();
    let job_id = core
        .fetch_assets(&["tiny".to_owned()], None, collector.sink())
        .expect("start fetch");
    collector.wait_for("progress", |events| {
        events
            .iter()
            .any(|e| matches!(e, AssetEvent::Progress { .. }))
    });

    // One job at a time: a second fetch, and a removal, are refused while it runs.
    let second: Collector<AssetEvent> = Collector::new();
    let error = core
        .fetch_assets(&["other".to_owned()], None, second.sink())
        .expect_err("second fetch");
    assert!(matches!(error, CoreError::FetchInProgress), "{error}");
    assert!(error.to_string().contains("already running"));
    assert!(matches!(
        core.remove_asset("tiny"),
        Err(CoreError::FetchInProgress)
    ));
    let install: Collector<InstallEvent> = Collector::new();
    assert!(matches!(
        core.env_install(Framework::Jax, install.sink()),
        Err(CoreError::FetchInProgress)
    ));

    let pidfile = root.path().join("python/.fetch-pid");
    let deadline = Instant::now() + Duration::from_secs(10);
    while !pidfile.is_file() && Instant::now() < deadline {
        std::thread::sleep(Duration::from_millis(10));
    }
    let pid = fs::read_to_string(&pidfile).expect("pid file");
    let pid = pid.trim();
    assert!(process_alive(pid));

    // Canceling something else does nothing; canceling the job ends it.
    core.cancel_asset_fetch("not-a-job");
    std::thread::sleep(Duration::from_millis(100));
    assert!(!collector.snapshot().iter().any(is_done));
    core.cancel_asset_fetch(&job_id);
    let events = collector.wait_for("the done event", |events| events.iter().any(is_done));
    assert_well_formed(&events);
    assert_eq!(
        events.last(),
        Some(&AssetEvent::Done {
            ok: false,
            canceled: true,
            message: None
        })
    );
    let deadline = Instant::now() + Duration::from_secs(5);
    while process_alive(pid) && Instant::now() < deadline {
        std::thread::sleep(Duration::from_millis(20));
    }
    assert!(!process_alive(pid), "the child of the fetch was killed");

    // What was on disk stays (a later fetch resumes), and the job slot is free.
    assert_eq!(state_of(&core, "tiny"), AssetState::Incomplete);
    core.cancel_asset_fetch(&job_id); // finished: a no-op
    fs::remove_file(root.path().join("python/.fetch-scenario")).expect("remove");
    assert!(matches!(
        fetch(&core, &["tiny"]).last(),
        Some(AssetEvent::Done { ok: true, .. })
    ));
    assert_eq!(state_of(&core, "tiny"), AssetState::Present);
}

#[test]
fn removal_deletes_only_the_declared_asset_directory() {
    let root = TestRoot::new();
    let core = root.core_with_fake_uv();
    fetch(&core, &["tiny"]);
    fetch(&core, &["other"]);
    let models = root.path().join(".learnai/models");
    let database = root.path().join(".learnai/state.db");
    assert!(models.join("tiny/learnai-asset.json").is_file());
    assert!(database.is_file());

    // Path tricks and undeclared names are refused, and delete nothing.
    for name in [
        "..",
        "../state.db",
        "tiny/..",
        "/",
        "",
        ".",
        "state.db",
        "--all",
    ] {
        let error = core.remove_asset(name).expect_err(name);
        assert!(
            matches!(error, CoreError::InvalidArgument(_)),
            "{name:?}: {error}"
        );
    }
    assert!(models.join("tiny").is_dir());
    assert!(models.join("other").is_dir());
    assert!(database.is_file());

    let listing = core.remove_asset("tiny").expect("remove");
    assert!(!models.join("tiny").exists());
    assert!(
        models.join("other").is_dir(),
        "the other asset is untouched"
    );
    assert_eq!(
        listing
            .assets
            .iter()
            .map(|a| (a.name.as_str(), a.state))
            .collect::<Vec<_>>(),
        [
            ("other", AssetState::Present),
            ("tiny", AssetState::Missing)
        ]
    );
    // Removing what is not there succeeds.
    core.remove_asset("tiny").expect("remove again");
}
