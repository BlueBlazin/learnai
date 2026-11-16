//! SSH run targets against fake `ssh` and `rsync` executables: the registry, the remote
//! sync, event streaming, stopping the remote process tree, connection failures, the
//! target check and each of its failing steps, and remote model downloads.
#![cfg(unix)]

mod common;

use std::fs;
use std::io::Read;
use std::process::{Command, Stdio};
use std::time::Duration;

use common::{
    Collector, FakeRemote, TestRoot, assert_well_formed, logs, process_alive, test_limits,
    wait_for_exit, wait_for_file,
};
use learnai_core::ssh::SUPERVISOR_SCRIPT;
use learnai_core::{
    AssetEvent, CheckStepId, Core, CoreError, Framework, LOCAL_TARGET, LogStream, Requires,
    RunEvent, RunLimits, RunMode, TargetCheck, TargetCheckEvent, TargetKind, Verdict,
};

const BOX: &str = "ssh:box";

fn preparing(events: &[RunEvent]) -> Vec<String> {
    events
        .iter()
        .filter_map(|event| match event {
            RunEvent::Preparing { message } => Some(message.clone()),
            _ => None,
        })
        .collect()
}

fn end_of(events: &[RunEvent]) -> (Verdict, Option<i64>, Option<String>) {
    match assert_well_formed(events) {
        RunEvent::End {
            verdict,
            submission_id,
            message,
            ..
        } => (*verdict, *submission_id, message.clone()),
        _ => unreachable!(),
    }
}

/// Writes a solution that selects a fake harness scenario and returns the core with
/// `box` added as a target.
fn remote_core(root: &TestRoot, remote: &FakeRemote, scenario: &str, limits: RunLimits) -> Core {
    let core = remote.core_with(root, limits);
    core.add_target("box", None).expect("add target");
    core.write_solution(
        "add-two",
        Framework::Torch,
        &format!("# scenario: {scenario}\ndef add(a, b):\n    return a + b\n"),
    )
    .expect("write solution");
    core
}

fn run(core: &Core, mode: RunMode, target: &str) -> Vec<RunEvent> {
    let collector = Collector::new();
    core.start_run("add-two", Framework::Torch, mode, target, collector.sink())
        .expect("start run");
    collector.wait_for_end()
}

/// Checks a target and returns the stream, verifying its shape: `done` exactly once
/// and last, carrying the steps that were streamed.
fn check(core: &Core, name: &str) -> (Vec<TargetCheckEvent>, TargetCheck) {
    let collector = Collector::new();
    let job = core
        .check_target(name, collector.sink())
        .expect("start check");
    assert!(!job.is_empty());
    let events = collector.wait_for("the done event", |events| {
        events
            .iter()
            .any(|event| matches!(event, TargetCheckEvent::Done { .. }))
    });
    let dones = events
        .iter()
        .filter(|event| matches!(event, TargetCheckEvent::Done { .. }))
        .count();
    assert_eq!(dones, 1, "{events:?}");
    let Some(TargetCheckEvent::Done { check }) = events.last() else {
        panic!("done is last: {events:?}");
    };
    let streamed: Vec<_> = events
        .iter()
        .filter_map(|event| match event {
            TargetCheckEvent::Step { step } => Some(step.clone()),
            _ => None,
        })
        .collect();
    assert_eq!(streamed, check.steps);
    (events.clone(), check.clone())
}

fn step_ids(check: &TargetCheck) -> Vec<(CheckStepId, bool)> {
    check.steps.iter().map(|step| (step.id, step.ok)).collect()
}

/// Asserts the check stopped at `id` and returns that step's message and fix.
fn failed_at(check: &TargetCheck, id: CheckStepId) -> (String, String) {
    assert!(!check.ok, "{check:?}");
    assert_eq!(check.doctor, None);
    let last = check.steps.last().expect("a step");
    assert_eq!((last.id, last.ok), (id, false), "{check:?}");
    assert!(
        check.steps[..check.steps.len() - 1]
            .iter()
            .all(|step| step.ok),
        "only the last step fails: {check:?}"
    );
    (last.message.clone(), last.fix.clone().unwrap_or_default())
}

#[test]
fn hosts_come_from_the_ssh_configuration_without_patterns() {
    let (root, remote) = (TestRoot::new(), FakeRemote::new());
    let core = remote.core(&root);
    let hosts = core.list_ssh_hosts();
    let aliases: Vec<&str> = hosts.iter().map(|host| host.alias.as_str()).collect();
    assert_eq!(
        aliases,
        ["box", "denied", "newhost", "changed", "down", "nowhere"]
    );
    assert_eq!(hosts[0].host_name.as_deref(), Some("10.0.0.5"));
    assert_eq!(hosts[0].user.as_deref(), Some("ada"));
    assert_eq!(hosts[0].port, Some(2222));
}

#[test]
fn the_registry_validates_persists_and_keeps_local_first() {
    let (root, remote) = (TestRoot::new(), FakeRemote::new());
    {
        let core = remote.core(&root);
        let targets = core.list_targets().expect("list");
        assert_eq!(targets.len(), 1);
        assert_eq!(targets[0].name, LOCAL_TARGET);
        assert_eq!(targets[0].kind, TargetKind::Local);
        assert_eq!(
            (targets[0].host.clone(), targets[0].remote_dir.clone()),
            (None, None)
        );

        // Only aliases from the configuration, and only plain remote directories.
        for host in [
            "stranger",
            "",
            "-oProxyCommand=x",
            "box; rm -rf ~",
            "*.internal",
        ] {
            let error = core.add_target(host, None).expect_err(host);
            assert!(
                matches!(error, CoreError::InvalidArgument(_)),
                "{host:?}: {error}"
            );
        }
        for dir in ["relative", "~/a b", "/tmp/$(id)", "~/../x", "/a;b"] {
            let error = core.add_target("box", Some(dir)).expect_err(dir);
            assert!(
                error.to_string().contains("remote directory"),
                "{dir:?}: {error}"
            );
        }

        let targets = core.add_target("box", None).expect("add");
        assert_eq!(targets.len(), 2);
        assert_eq!(targets[1].name, BOX);
        assert_eq!(targets[1].kind, TargetKind::Ssh);
        assert_eq!(targets[1].label, "box");
        assert_eq!(targets[1].host.as_deref(), Some("box"));
        assert_eq!(targets[1].remote_dir.as_deref(), Some("~/.learnai"));
        assert_eq!(targets[1].last_check, None);
        let targets = core.add_target("down", Some("/srv/ws/")).expect("add");
        assert_eq!(targets[2].remote_dir.as_deref(), Some("/srv/ws"));

        let duplicate = core.add_target("box", None).expect_err("duplicate");
        assert!(duplicate.to_string().contains("already"), "{duplicate}");
    }

    let core = remote.core(&root);
    let names: Vec<String> = core
        .list_targets()
        .expect("list")
        .into_iter()
        .map(|target| target.name)
        .collect();
    assert_eq!(names, [LOCAL_TARGET, BOX, "ssh:down"]);

    assert!(matches!(
        core.remove_target(LOCAL_TARGET),
        Err(CoreError::InvalidArgument(_))
    ));
    assert!(matches!(
        core.remove_target("ssh:stranger"),
        Err(CoreError::UnknownTarget(_))
    ));
    let targets = core.remove_target("ssh:down").expect("remove");
    assert_eq!(targets.len(), 2);

    // A run on a target that does not exist starts nothing.
    let collector = Collector::<RunEvent>::new();
    let error = core
        .start_run(
            "add-two",
            Framework::Torch,
            RunMode::Run,
            "ssh:down",
            collector.sink(),
        )
        .expect_err("unknown target");
    assert!(matches!(error, CoreError::UnknownTarget(_)));
    assert!(error.to_string().contains("ssh:down"));
    assert!(collector.snapshot().is_empty());
}

#[test]
fn a_remote_submit_syncs_runs_and_is_stored_with_its_target() {
    let (root, remote) = (TestRoot::new(), FakeRemote::new());
    // Things that must not travel: the local virtual environment and caches.
    fs::create_dir_all(root.path().join("python/.venv/lib")).expect("mkdir");
    fs::write(root.path().join("python/.venv/lib/big.so"), b"x").expect("write");
    fs::create_dir_all(root.path().join("python/src/__pycache__")).expect("mkdir");
    fs::write(root.path().join("python/src/__pycache__/m.pyc"), b"x").expect("write");
    fs::write(root.path().join("python/pyproject.toml"), b"[project]\n").expect("write");

    let core = remote_core(&root, &remote, "happy", test_limits());
    let events = run(&core, RunMode::Submit, BOX);
    let (verdict, submission, _) = end_of(&events);
    assert_eq!(verdict, Verdict::Accepted, "{events:?}");

    let steps = preparing(&events);
    assert!(steps[0].contains("ssh:box"), "{steps:?}");
    assert_eq!(
        steps[1..],
        [
            "Connecting to box",
            "Copying the judge to box",
            "Copying the problem and your solution to box",
            "Syncing the Python environment on box and starting the judge",
        ]
    );
    let cases = events
        .iter()
        .filter(|event| matches!(event, RunEvent::CaseEnd { .. }))
        .count();
    assert_eq!(cases, 3);

    // Stored with the target's name.
    let stored = core
        .get_submission(submission.expect("stored"))
        .expect("submission");
    assert_eq!(stored.submission.target, BOX);
    assert_eq!(stored.submission.verdict, Verdict::Accepted);

    // The remote root mirrors a content root.
    let remote_root = remote.root();
    assert!(remote_root.join("python/pyproject.toml").is_file());
    assert!(
        remote_root
            .join("problems/basics/add-two/problem.toml")
            .is_file()
    );
    assert!(remote_root.join("problems/basics/track.toml").is_file());
    let solution =
        fs::read_to_string(remote_root.join("workspace/add-two/torch.py")).expect("solution");
    assert!(solution.starts_with("# scenario: happy"));
    assert!(!remote_root.join("python/.venv").exists());
    assert!(!remote_root.join("python/src/__pycache__").exists());
    assert!(
        !remote_root.join("problems/basics/echo-server").exists(),
        "only the problem being run"
    );

    // The harness ran there, through uv, with the content root set.
    let uv_log = fs::read_to_string(remote_root.join("python/.fake-uv.log")).expect("uv log");
    assert_eq!(uv_log.trim(), "run");

    // A second run keeps what the remote host built for itself and picks up edits.
    fs::create_dir_all(remote_root.join("python/.venv")).expect("mkdir");
    fs::write(remote_root.join("python/.venv/marker"), b"x").expect("write");
    fs::write(remote_root.join("python/stale.py"), b"x").expect("write");
    core.write_solution("add-two", Framework::Torch, "# scenario: fail\n")
        .expect("write");
    let events = run(&core, RunMode::Run, BOX);
    assert_eq!(end_of(&events).0, Verdict::WrongAnswer);
    assert!(remote_root.join("python/.venv/marker").is_file());
    assert!(
        !remote_root.join("python/stale.py").exists(),
        "deleted files are deleted there"
    );

    // The local target still works beside it.
    let local = Collector::new();
    remote
        .core(&root)
        .start_run(
            "add-two",
            Framework::Torch,
            RunMode::Run,
            LOCAL_TARGET,
            local.sink(),
        )
        .expect("local run");
    assert_eq!(end_of(&local.wait_for_end()).0, Verdict::WrongAnswer);
}

#[test]
fn every_invocation_is_non_interactive_and_nothing_is_spliced_unquoted() {
    let (root, remote) = (TestRoot::new(), FakeRemote::new());
    let core = remote_core(&root, &remote, "happy", test_limits());
    run(&core, RunMode::Run, BOX);

    let ssh = remote.ssh_log();
    let invocations: Vec<&String> = ssh.iter().filter(|line| line.starts_with("-o ")).collect();
    assert!(invocations.len() >= 3, "{ssh:?}");
    for line in &invocations {
        assert!(
            line.contains("-o BatchMode=yes -o ConnectTimeout=10"),
            "{line}"
        );
        assert!(line.contains("-o ControlPath="), "{line}");
        assert!(line.contains(" -T -- box "), "{line}");
    }
    // The first invocation opens the shared connection; the rest only use it.
    assert!(invocations[0].contains("ControlMaster=auto"));
    assert!(invocations[0].contains("ControlPersist=60"));
    assert!(
        invocations[1..]
            .iter()
            .all(|line| line.contains("ControlMaster=no"))
    );

    let whole = ssh.join("\n");
    let root_dir = remote.root().display().to_string();
    assert!(
        whole.contains(&format!("'mkdir' '-p' '--' '{root_dir}/python'")),
        "{whole}"
    );
    assert!(
        whole.contains(&format!(
            "'python' '-m' 'learnai.harness' '--problem' '{root_dir}/problems/basics/add-two' \
             '--solution' '{root_dir}/workspace/add-two/torch.py' '--framework' 'torch' '--mode' 'run'"
        )),
        "{whole}"
    );

    let rsync = remote.rsync_log();
    assert_eq!(rsync.len(), 4, "{rsync:?}");
    for line in &rsync {
        assert!(line.contains("-e "), "{line}");
        assert!(line.contains("-o BatchMode=yes"), "{line}");
        assert!(line.contains("ControlMaster=no"), "{line}");
    }
    assert!(rsync[0].contains("--delete --exclude=.venv --exclude=__pycache__"));
    assert!(rsync[0].ends_with(&format!("python/ box:{root_dir}/python/")));
}

#[test]
fn stdout_and_stderr_stay_separate_streams() {
    let (root, remote) = (TestRoot::new(), FakeRemote::new());
    let core = remote_core(&root, &remote, "garbage", test_limits());
    let events = run(&core, RunMode::Run, BOX);
    assert_eq!(end_of(&events).0, Verdict::Accepted, "{events:?}");
    let stream_of = |needle: &str| {
        events.iter().find_map(|event| match event {
            RunEvent::Log { stream, text } if text.contains(needle) => Some(*stream),
            _ => None,
        })
    };
    assert_eq!(stream_of("warning on stderr"), Some(LogStream::Stderr));
    assert_eq!(
        stream_of("plain line before start"),
        Some(LogStream::Stdout)
    );
    assert_eq!(stream_of("between cases"), Some(LogStream::Stdout));
}

fn cancel_kills_the_remote_tree(remote: &FakeRemote) {
    let root = TestRoot::new();
    let core = remote_core(&root, remote, "hang", test_limits());
    let collector = Collector::new();
    let run_id = core
        .start_run(
            "add-two",
            Framework::Torch,
            RunMode::Submit,
            BOX,
            collector.sink(),
        )
        .expect("start run");
    collector.wait_for("the first case", |events| {
        events
            .iter()
            .any(|event| matches!(event, RunEvent::CaseStart { .. }))
    });
    let pid = wait_for_file(&remote.path().join("hang.pid"));
    assert!(process_alive(&pid), "the remote child is running");

    core.cancel_run(&run_id);
    let events = collector.wait_for_end();
    let (verdict, submission, _) = end_of(&events);
    assert_eq!(verdict, Verdict::Canceled);
    assert_eq!(submission, None, "a canceled submit is not stored");
    assert!(wait_for_exit(&pid), "the remote process tree was killed");
    assert!(core.list_submissions(None).expect("list").is_empty());
}

#[test]
fn cancel_kills_the_remote_process_tree() {
    cancel_kills_the_remote_tree(&FakeRemote::new());
}

#[test]
fn cancel_kills_the_remote_process_tree_on_a_dash_and_setsid_host() {
    let remote = FakeRemote::new();
    if remote.use_dash_with_setsid() {
        cancel_kills_the_remote_tree(&remote);
    }
}

#[test]
fn a_timeout_kills_the_remote_process_tree() {
    let (root, remote) = (TestRoot::new(), FakeRemote::new());
    // The harness never reports `start`, so the startup limit is the one that fires.
    let limits = RunLimits {
        startup: Duration::from_millis(1500),
        ..test_limits()
    };
    let core = remote_core(&root, &remote, "slow-start", limits);
    let events = run(&core, RunMode::Submit, BOX);
    let (verdict, _, message) = end_of(&events);
    assert_eq!(verdict, Verdict::TimeLimit, "{events:?}");
    assert!(
        message
            .expect("message")
            .contains("exceeded the overall limit")
    );

    assert_nothing_left_running(&remote);
}

/// Asserts that no process still mentions the fake remote's root: the harness and
/// everything under it are gone.
fn assert_nothing_left_running(remote: &FakeRemote) {
    let marker = remote.root().display().to_string();
    let deadline = std::time::Instant::now() + Duration::from_secs(10);
    loop {
        let listing = Command::new("ps")
            .args(["-A", "-o", "pid=,command="])
            .output()
            .expect("ps");
        let listing = String::from_utf8_lossy(&listing.stdout).into_owned();
        let left: Vec<&str> = listing
            .lines()
            .filter(|line| line.contains(&marker))
            .collect();
        if left.is_empty() {
            return;
        }
        assert!(
            std::time::Instant::now() < deadline,
            "left running: {left:?}"
        );
        std::thread::sleep(Duration::from_millis(50));
    }
}

#[test]
fn closing_the_channel_alone_stops_the_remote_command() {
    // What happens when the app exits or crashes: nobody calls kill, the operating
    // system just closes the pipe to ssh. The supervisor must do the rest.
    let tmp = tempfile::tempdir().expect("temp dir");
    let pidfile = tmp.path().join("pid");
    let mut child = Command::new("sh")
        .args(["-c", SUPERVISOR_SCRIPT, "learnai", "/tmp/root", "sh", "-c"])
        .arg(format!(
            "echo \"root=$LEARNAI_ROOT\"; sh -c 'sleep 600 & echo $! > {}; wait' & wait",
            pidfile.display()
        ))
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .expect("spawn supervisor");
    let pid = wait_for_file(&pidfile);
    assert!(process_alive(&pid));

    drop(child.stdin.take());
    let status = child.wait().expect("wait");
    assert_eq!(status.code(), Some(143));
    assert!(
        wait_for_exit(&pid),
        "a grandchild of the command was killed"
    );
    let mut stdout = String::new();
    child
        .stdout
        .take()
        .expect("stdout")
        .read_to_string(&mut stdout)
        .expect("read");
    assert_eq!(
        stdout.trim(),
        "root=/tmp/root",
        "the content root is exported"
    );
}

#[test]
fn the_supervisor_passes_the_exit_code_on_and_sweeps_leftovers() {
    let tmp = tempfile::tempdir().expect("temp dir");
    let pidfile = tmp.path().join("pid");
    let mut child = Command::new("sh")
        .args(["-c", SUPERVISOR_SCRIPT, "learnai", "/tmp/root", "sh", "-c"])
        .arg(format!(
            "sleep 600 & echo $! > {}; exit 7",
            pidfile.display()
        ))
        .stdin(Stdio::piped())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .expect("spawn supervisor");
    // Keep the channel open: `Child::wait` would close it otherwise.
    let _channel = child.stdin.take();
    let status = child.wait().expect("wait");
    assert_eq!(status.code(), Some(7));
    let pid = wait_for_file(&pidfile);
    assert!(
        wait_for_exit(&pid),
        "what the command left running was killed"
    );
}

#[test]
fn a_host_that_cannot_be_reached_ends_the_run_with_a_fix() {
    let (root, remote) = (TestRoot::new(), FakeRemote::new());
    let core = remote.core(&root);
    let cases = [
        ("denied", "ssh-add"),
        ("newhost", "confirm the host key"),
        ("changed", "ssh-keygen -R"),
        ("down", "is on and reachable"),
        ("nowhere", "HostName"),
    ];
    for (host, fix) in cases {
        core.add_target(host, None).expect("add");
        let events = run(&core, RunMode::Submit, &format!("ssh:{host}"));
        let (verdict, submission, message) = end_of(&events);
        assert_eq!(verdict, Verdict::HarnessError, "{host}");
        assert_eq!(
            submission, None,
            "{host}: a run that never launched is not stored"
        );
        let message = message.expect("message");
        assert!(message.contains(host), "{host}: {message}");
        assert!(message.contains(fix), "{host}: {message}");
        assert!(
            logs(&events)
                .iter()
                .any(|line| line.contains("Could not start the judge")),
            "{host}"
        );
    }
    assert!(core.list_submissions(None).expect("list").is_empty());
}

#[test]
fn missing_uv_or_rsync_ends_the_run_with_the_install_instruction() {
    let (root, remote) = (TestRoot::new(), FakeRemote::new());
    let core = remote_core(&root, &remote, "happy", test_limits());
    remote.remove_tool("rsync");
    let (verdict, _, message) = end_of(&run(&core, RunMode::Run, BOX));
    assert_eq!(verdict, Verdict::HarnessError);
    assert!(
        message
            .expect("message")
            .contains("rsync is not installed on `box`")
    );

    remote.remove_tool("uv");
    let (verdict, _, message) = end_of(&run(&core, RunMode::Run, BOX));
    assert_eq!(verdict, Verdict::HarnessError);
    let message = message.expect("message");
    assert!(
        message.contains("uv is not installed on `box`"),
        "{message}"
    );
    assert!(
        message.contains("ssh box 'curl -LsSf https://astral.sh/uv/install.sh | sh'"),
        "{message}"
    );
}

#[test]
fn a_dropped_connection_ends_the_run_saying_so() {
    let (root, remote) = (TestRoot::new(), FakeRemote::new());
    let core = remote_core(&root, &remote, "hang", test_limits());
    remote.mark("drop");
    let events = run(&core, RunMode::Run, BOX);
    let (verdict, _, message) = end_of(&events);
    assert_eq!(verdict, Verdict::HarnessError, "{events:?}");
    let message = message.expect("message");
    assert!(message.contains("connection to `box` dropped"), "{message}");
    // The remote side saw its channel close and cleaned up after itself.
    assert_nothing_left_running(&remote);
}

#[test]
fn a_healthy_target_passes_every_step_and_saves_the_report() {
    let (root, remote) = (TestRoot::new(), FakeRemote::new());
    let core = remote.core(&root);
    core.add_target("box", None).expect("add");
    remote.mark_root(".fake-cuda");

    let (events, result) = check(&core, BOX);
    assert!(result.ok, "{result:?}");
    assert_eq!(
        step_ids(&result),
        [
            (CheckStepId::Connect, true),
            (CheckStepId::Uv, true),
            (CheckStepId::Sync, true),
            (CheckStepId::Environment, true),
            (CheckStepId::Doctor, true),
        ]
    );
    assert!(result.steps.iter().all(|step| step.fix.is_none()));
    assert!(result.steps[0].message.contains("Connected to box"));
    let doctor = result.doctor.clone().expect("doctor");
    assert_eq!(doctor.python, "3.15.2");
    assert_eq!(
        (doctor.system.as_str(), doctor.machine.as_str()),
        ("Linux", "x86_64")
    );
    assert_eq!(doctor.hardware, [Requires::Cpu, Requires::Cuda]);
    assert_eq!(doctor.gpu_name.as_deref(), Some("NVIDIA GeForce RTX 4090"));
    assert_eq!(doctor.gpu_memory_bytes, Some(25_757_220_864));
    assert_eq!(
        doctor.frameworks[&Framework::Torch].as_deref(),
        Some("2.15.0")
    );
    assert_eq!(doctor.frameworks[&Framework::Mlx], None);

    // The environment build's output is streamed as log lines.
    assert!(events.iter().any(|event| matches!(
        event,
        TargetCheckEvent::Log { text } if text.contains("Resolved 42 packages")
    )));

    // Saved as the target's last check, and across a reopen.
    let saved = remote.core(&root).list_targets().expect("list")[1]
        .last_check
        .clone();
    assert_eq!(saved, Some(result));

    // Without the GPU marker the same host reports a CPU only.
    fs::remove_file(remote.root().join(".fake-cuda")).expect("remove marker");
    let (_, result) = check(&core, BOX);
    let doctor = result.doctor.expect("doctor");
    assert_eq!(doctor.hardware, [Requires::Cpu]);
    assert_eq!(doctor.gpu_name, None);
}

#[test]
fn each_failing_check_step_stops_the_check_with_a_message() {
    let root = TestRoot::new();

    // connect: one target per way of failing.
    let remote = FakeRemote::new();
    let core = remote.core(&root);
    for (host, fix) in [
        ("denied", "ssh-add"),
        ("newhost", "ssh newhost"),
        ("down", "ssh down"),
    ] {
        core.add_target(host, None).expect("add");
        let (_, result) = check(&core, &format!("ssh:{host}"));
        let (message, step_fix) = failed_at(&result, CheckStepId::Connect);
        assert!(message.contains(host), "{message}");
        assert!(step_fix.contains(fix), "{host}: {step_fix}");
        assert_eq!(result.steps.len(), 1);
    }
    let saved = core.list_targets().expect("list");
    assert!(
        saved[1..]
            .iter()
            .all(|target| target.last_check.as_ref().is_some_and(|c| !c.ok))
    );

    // uv: missing on the remote host; the fix is the official installer, over ssh.
    let remote = FakeRemote::new();
    let local = TestRoot::new();
    let core = remote.core(&local);
    core.add_target("box", None).expect("add");
    remote.remove_tool("uv");
    let (message, fix) = failed_at(&check(&core, BOX).1, CheckStepId::Uv);
    assert!(
        message.contains("uv is not installed on `box`"),
        "{message}"
    );
    assert_eq!(
        fix,
        "ssh box 'curl -LsSf https://astral.sh/uv/install.sh | sh'"
    );

    // sync: rsync missing on the remote host, then rsync failing.
    let remote = FakeRemote::new();
    let local = TestRoot::new();
    let core = remote.core(&local);
    core.add_target("box", None).expect("add");
    remote.remove_tool("rsync");
    let (message, fix) = failed_at(&check(&core, BOX).1, CheckStepId::Sync);
    assert!(
        message.contains("rsync is not installed on `box`"),
        "{message}"
    );
    assert!(fix.contains("apt install rsync"), "{fix}");

    let remote = FakeRemote::new();
    let local = TestRoot::new();
    let core = remote.core(&local);
    core.add_target("box", None).expect("add");
    remote.mark("fail-rsync");
    let (message, _) = failed_at(&check(&core, BOX).1, CheckStepId::Sync);
    assert!(message.contains("Copying the judge failed"), "{message}");

    // environment: the build fails; its last error line is the message.
    let remote = FakeRemote::new();
    let local = TestRoot::new();
    let core = remote.core(&local);
    core.add_target("box", None).expect("add");
    remote.mark_root(".fail-sync");
    let (events, result) = check(&core, BOX);
    let (message, fix) = failed_at(&result, CheckStepId::Environment);
    assert!(
        message.contains("No solution found when resolving dependencies"),
        "{message}"
    );
    assert!(fix.contains("ssh box"), "{fix}");
    assert!(events.iter().any(|event| matches!(
        event,
        TargetCheckEvent::Log { text } if text.contains("No solution found")
    )));

    // doctor: the report cannot be produced.
    let remote = FakeRemote::new();
    let local = TestRoot::new();
    let core = remote.core(&local);
    core.add_target("box", None).expect("add");
    remote.mark_root(".fail-doctor");
    let (message, _) = failed_at(&check(&core, BOX).1, CheckStepId::Doctor);
    assert!(message.contains("No module named 'learnai'"), "{message}");
}

#[test]
fn a_local_rsync_that_is_missing_is_reported_as_such() {
    let (root, remote) = (TestRoot::new(), FakeRemote::new());
    let mut options = remote.ssh_options();
    options.rsync_program = Some(remote.path().join("bin/no-such-rsync"));
    let core = Core::open(learnai_core::CoreOptions {
        uv_program: Some(common::fixtures().join("fake_uv.sh")),
        ssh: options,
        ..learnai_core::CoreOptions::new(root.path())
    })
    .expect("open core");
    core.add_target("box", None).expect("add");
    let (message, fix) = failed_at(&check(&core, BOX).1, CheckStepId::Sync);
    assert!(
        message.contains("was not found on this computer"),
        "{message}"
    );
    assert!(fix.contains("Install"), "{fix}");
}

#[test]
fn checks_cover_the_local_target_and_refuse_unknown_ones() {
    let (root, remote) = (TestRoot::new(), FakeRemote::new());
    let core = remote.core(&root);
    let (_, result) = check(&core, LOCAL_TARGET);
    assert!(result.ok, "{result:?}");
    assert_eq!(
        step_ids(&result),
        [(CheckStepId::Uv, true), (CheckStepId::Doctor, true)]
    );
    assert!(result.steps[0].message.contains("uv 0.13.8"));
    assert_eq!(result.doctor.as_ref().expect("doctor").python, "3.15.2");
    assert_eq!(
        core.list_targets().expect("list")[0].last_check,
        Some(result)
    );
    // The local check installs nothing: the probe ran without a sync.
    let uv_log = fs::read_to_string(root.path().join("python/.fake-uv.log")).expect("log");
    assert_eq!(uv_log.trim(), "run");

    let collector = Collector::<TargetCheckEvent>::new();
    assert!(matches!(
        core.check_target("ssh:stranger", collector.sink()),
        Err(CoreError::UnknownTarget(_))
    ));
    assert!(collector.snapshot().is_empty());
}

fn fetch(core: &Core, target: Option<&str>) -> (String, Collector<AssetEvent>) {
    let collector = Collector::new();
    let job = core
        .fetch_assets(&["tiny".to_owned()], target, collector.sink())
        .expect("start fetch");
    (job, collector)
}

fn wait_done(collector: &Collector<AssetEvent>) -> Vec<AssetEvent> {
    let events = collector.wait_for("the done event", |events| {
        events
            .iter()
            .any(|event| matches!(event, AssetEvent::Done { .. }))
    });
    assert!(
        matches!(events.first(), Some(AssetEvent::Preparing { .. })),
        "{events:?}"
    );
    assert!(
        matches!(events.last(), Some(AssetEvent::Done { .. })),
        "{events:?}"
    );
    events
}

#[test]
fn a_remote_fetch_streams_the_same_events_and_lands_on_the_remote_host() {
    let (root, remote) = (TestRoot::new(), FakeRemote::new());
    let core = remote.core(&root);
    core.add_target("box", None).expect("add");
    let (_, collector) = fetch(&core, Some(BOX));
    let events = wait_done(&collector);
    assert_eq!(
        events.last(),
        Some(&AssetEvent::Done {
            ok: true,
            canceled: false,
            message: None
        }),
        "{events:?}"
    );
    assert!(events.contains(&AssetEvent::Start {
        asset: "tiny".into(),
        total_bytes: 2048
    }));
    assert!(events.contains(&AssetEvent::AssetEnd {
        asset: "tiny".into(),
        ok: true,
        message: None
    }));
    assert!(events.iter().any(|event| matches!(
        event,
        AssetEvent::Preparing { message } if message == "Connecting to box"
    )));

    // On the remote host, not here; and the problem that declares it traveled too.
    let remote_root = remote.root();
    assert!(
        remote_root
            .join(".learnai/models/tiny/learnai-asset.json")
            .is_file()
    );
    assert!(
        remote_root
            .join("problems/basics/add-two/problem.toml")
            .is_file()
    );
    assert!(!root.path().join(".learnai/models/tiny").exists());
    // The remote environment got the download tools; the local one was not touched.
    let uv_log = fs::read_to_string(remote_root.join("python/.fake-uv.log")).expect("log");
    assert_eq!(uv_log.trim(), "run models");
    assert!(!root.path().join("python/.fake-uv.log").exists());

    // An unknown target starts nothing; `local` is the default.
    let nothing = Collector::<AssetEvent>::new();
    assert!(matches!(
        core.fetch_assets(&["tiny".to_owned()], Some("ssh:stranger"), nothing.sink()),
        Err(CoreError::UnknownTarget(_))
    ));
    let (_, local) = fetch(&core, Some(LOCAL_TARGET));
    wait_done(&local);
    assert!(
        root.path()
            .join(".learnai/models/tiny/learnai-asset.json")
            .is_file()
    );
}

#[test]
fn canceling_a_remote_fetch_kills_it_on_the_remote_host() {
    let (root, remote) = (TestRoot::new(), FakeRemote::new());
    let core = remote.core(&root);
    core.add_target("box", None).expect("add");
    remote.mark_root(".fetch-scenario");
    fs::write(remote.root().join(".fetch-scenario"), "hang").expect("scenario");

    let (job, collector) = fetch(&core, Some(BOX));
    collector.wait_for("progress", |events| {
        events
            .iter()
            .any(|event| matches!(event, AssetEvent::Progress { .. }))
    });
    let pid = wait_for_file(&remote.root().join("python/.fetch-pid"));
    assert!(process_alive(&pid));

    // One fetch at a time, wherever it runs.
    let second = Collector::<AssetEvent>::new();
    assert!(matches!(
        core.fetch_assets(&["tiny".to_owned()], None, second.sink()),
        Err(CoreError::FetchInProgress)
    ));

    core.cancel_asset_fetch(&job);
    let events = wait_done(&collector);
    assert_eq!(
        events.last(),
        Some(&AssetEvent::Done {
            ok: false,
            canceled: true,
            message: None
        })
    );
    assert!(wait_for_exit(&pid), "the remote download was killed");
}

#[test]
fn a_remote_fetch_that_cannot_connect_says_what_to_do() {
    let (root, remote) = (TestRoot::new(), FakeRemote::new());
    let core = remote.core(&root);
    core.add_target("denied", None).expect("add");
    let (_, collector) = fetch(&core, Some("ssh:denied"));
    let events = wait_done(&collector);
    let Some(AssetEvent::Done {
        ok: false,
        canceled: false,
        message: Some(message),
    }) = events.last()
    else {
        panic!("{events:?}");
    };
    assert!(message.contains("ssh-add"), "{message}");
}
