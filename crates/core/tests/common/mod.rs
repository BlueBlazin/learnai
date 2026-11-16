//! Shared helpers for the integration tests: a throwaway content root copied from
//! `tests/fixtures/root`, fake harness and uv scripts, and an event collector.
#![allow(dead_code)]

use std::fs;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Condvar, Mutex};
use std::time::{Duration, Instant};

use learnai_core::{Core, CoreOptions, EventSink, ProgramHarness, RunEvent, RunLimits, SshOptions};
use tempfile::TempDir;

/// The fixtures directory.
pub fn fixtures() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
}

fn copy_tree(from: &Path, to: &Path) {
    fs::create_dir_all(to).expect("create directory");
    for entry in fs::read_dir(from).expect("read directory") {
        let entry = entry.expect("directory entry");
        let target = to.join(entry.file_name());
        if entry.path().is_dir() {
            copy_tree(&entry.path(), &target);
        } else {
            fs::copy(entry.path(), &target).expect("copy file");
        }
    }
}

/// Limits that keep failing tests fast while leaving healthy runs plenty of room.
pub fn test_limits() -> RunLimits {
    RunLimits {
        startup: Duration::from_secs(20),
        slack: Duration::from_secs(20),
        exit_grace: Duration::from_millis(300),
    }
}

/// A private copy of the fixture content root.
pub struct TestRoot {
    dir: TempDir,
}

impl TestRoot {
    pub fn new() -> Self {
        let dir = tempfile::tempdir().expect("temp dir");
        copy_tree(&fixtures().join("root"), dir.path());
        Self { dir }
    }

    pub fn path(&self) -> &Path {
        self.dir.path()
    }

    /// A harness stand-in that runs the fake harness script in `scenario`.
    pub fn harness(&self, scenario: &str) -> ProgramHarness {
        ProgramHarness::new("sh")
            .arg(fixtures().join("fake_harness.sh"))
            .env("FAKE_SCENARIO", scenario)
            .env("FAKE_PIDFILE", self.pidfile())
    }

    /// Where the `hang` scenario records its child's pid.
    pub fn pidfile(&self) -> PathBuf {
        self.path().join("hang.pid")
    }

    /// A core whose harness is the fake script in `scenario`, with the fake uv.
    pub fn core(&self, scenario: &str) -> Core {
        self.core_with(scenario, test_limits())
    }

    pub fn core_with(&self, scenario: &str, limits: RunLimits) -> Core {
        Core::open(CoreOptions {
            harness: Some(Arc::new(self.harness(scenario))),
            uv_program: Some(fixtures().join("fake_uv.sh")),
            limits,
            ..CoreOptions::new(self.path())
        })
        .expect("open core")
    }

    /// A core that uses the real `uv run` command line against the fake uv.
    pub fn core_with_fake_uv(&self) -> Core {
        Core::open(CoreOptions {
            uv_program: Some(fixtures().join("fake_uv.sh")),
            limits: test_limits(),
            ..CoreOptions::new(self.path())
        })
        .expect("open core")
    }
}

/// Collects events and lets a test wait for a condition on them.
pub struct Collector<E> {
    state: Arc<(Mutex<Vec<E>>, Condvar)>,
}

impl<E> Clone for Collector<E> {
    fn clone(&self) -> Self {
        Self {
            state: Arc::clone(&self.state),
        }
    }
}

impl<E: Clone + Send + 'static> Collector<E> {
    pub fn new() -> Self {
        Self {
            state: Arc::new((Mutex::new(Vec::new()), Condvar::new())),
        }
    }

    pub fn sink(&self) -> impl EventSink<E> + use<E> {
        let state = Arc::clone(&self.state);
        move |event: E| {
            state.0.lock().expect("collector lock").push(event);
            state.1.notify_all();
        }
    }

    /// Waits until `done` holds for the collected events, then returns them.
    pub fn wait_for(&self, what: &str, done: impl Fn(&[E]) -> bool) -> Vec<E> {
        let deadline = Instant::now() + Duration::from_secs(30);
        let mut events = self.state.0.lock().expect("collector lock");
        while !done(&events) {
            let left = deadline.saturating_duration_since(Instant::now());
            assert!(!left.is_zero(), "timed out waiting for {what}");
            events = self
                .state
                .1
                .wait_timeout(events, left)
                .expect("collector lock")
                .0;
        }
        events.clone()
    }

    pub fn snapshot(&self) -> Vec<E> {
        self.state.0.lock().expect("collector lock").clone()
    }
}

impl Collector<RunEvent> {
    /// Waits for the `end` event and returns the whole stream.
    pub fn wait_for_end(&self) -> Vec<RunEvent> {
        self.wait_for("the end event", |events| events.iter().any(is_end))
    }
}

pub fn is_end(event: &RunEvent) -> bool {
    matches!(event, RunEvent::End { .. })
}

/// Asserts the stream shape every run must have: at least one `preparing` first, exactly
/// one `end`, and `end` last. Returns the `end` event.
pub fn assert_well_formed(events: &[RunEvent]) -> &RunEvent {
    assert!(
        matches!(events.first(), Some(RunEvent::Preparing { .. })),
        "first event: {events:?}"
    );
    assert_eq!(
        events.iter().filter(|e| is_end(e)).count(),
        1,
        "exactly one end: {events:?}"
    );
    let last = events.last().expect("events");
    assert!(is_end(last), "end is last: {events:?}");
    last
}

/// The log lines of a stream.
pub fn logs(events: &[RunEvent]) -> Vec<String> {
    events
        .iter()
        .filter_map(|e| match e {
            RunEvent::Log { text, .. } => Some(text.clone()),
            _ => None,
        })
        .collect()
}

/// A fake remote host: fake `ssh` and `rsync` executables, a "remote" home directory,
/// and an SSH configuration, all in one temporary directory. See
/// `fixtures/fake_ssh.sh` for how the pieces behave.
pub struct FakeRemote {
    dir: TempDir,
}

/// The tools the fake remote host has on its `PATH`, besides uv.
const REMOTE_TOOLS: &[&str] = &[
    "sh", "awk", "ps", "uname", "sleep", "cat", "mkdir", "sed", "head", "basename", "dirname",
    "wc", "tr", "rsync", "perl",
];

fn find_tool(name: &str) -> Option<PathBuf> {
    ["/bin", "/usr/bin", "/usr/local/bin", "/opt/homebrew/bin"]
        .iter()
        .map(|dir| Path::new(dir).join(name))
        .find(|path| path.is_file())
}

fn install_script(from: &Path, to: &Path) {
    use std::os::unix::fs::PermissionsExt;
    fs::copy(from, to).expect("copy script");
    fs::set_permissions(to, fs::Permissions::from_mode(0o755)).expect("chmod");
}

impl FakeRemote {
    /// A healthy host whose SSH configuration defines `box` (and the aliases the fake
    /// ssh refuses: `denied`, `newhost`, `changed`, `down`, `nowhere`).
    pub fn new() -> Self {
        let dir = tempfile::tempdir().expect("temp dir");
        let path = dir.path();
        for sub in ["bin", "home", "remote-bin", "ctl"] {
            fs::create_dir_all(path.join(sub)).expect("create directory");
        }
        install_script(&fixtures().join("fake_ssh.sh"), &path.join("bin/ssh"));
        install_script(&fixtures().join("fake_rsync.sh"), &path.join("bin/rsync"));
        install_script(&fixtures().join("fake_uv.sh"), &path.join("remote-bin/uv"));
        install_script(
            &fixtures().join("fake_harness.sh"),
            &path.join("remote-bin/fake_harness.sh"),
        );
        for tool in REMOTE_TOOLS {
            if let Some(real) = find_tool(tool) {
                std::os::unix::fs::symlink(real, path.join("remote-bin").join(tool))
                    .expect("link tool");
            }
        }
        fs::write(
            path.join("ssh_config"),
            "Host *\n  ServerAliveInterval 30\n\n\
             Host box\n  HostName 10.0.0.5\n  User ada\n  Port 2222\n\n\
             Host denied newhost changed down nowhere\n  User ada\n\n\
             Host *.internal\n  User nobody\n",
        )
        .expect("write config");
        Self { dir }
    }

    pub fn path(&self) -> &Path {
        self.dir.path()
    }

    /// The remote home directory.
    pub fn home(&self) -> PathBuf {
        self.path().join("home")
    }

    /// The remote content root of a target added with the default directory.
    pub fn root(&self) -> PathBuf {
        self.home().join(".learnai/root")
    }

    pub fn ssh_options(&self) -> SshOptions {
        SshOptions {
            ssh_program: Some(self.path().join("bin/ssh")),
            rsync_program: Some(self.path().join("bin/rsync")),
            config: Some(self.path().join("ssh_config")),
            control_dir: Some(self.path().join("ctl")),
        }
    }

    /// A core for `root` whose SSH tooling is this fake.
    pub fn core(&self, root: &TestRoot) -> Core {
        self.core_with(root, test_limits())
    }

    pub fn core_with(&self, root: &TestRoot, limits: RunLimits) -> Core {
        Core::open(CoreOptions {
            uv_program: Some(fixtures().join("fake_uv.sh")),
            limits,
            ssh: self.ssh_options(),
            ..CoreOptions::new(root.path())
        })
        .expect("open core")
    }

    /// Creates a marker file under the fake's state directory.
    pub fn mark(&self, name: &str) {
        fs::write(self.path().join(name), b"").expect("write marker");
    }

    /// Creates a marker file in the remote content root.
    pub fn mark_root(&self, name: &str) {
        fs::create_dir_all(self.root()).expect("create root");
        fs::write(self.root().join(name), b"").expect("write marker");
    }

    /// Removes a tool from the remote `PATH`.
    pub fn remove_tool(&self, name: &str) {
        fs::remove_file(self.path().join("remote-bin").join(name)).expect("remove tool");
    }

    /// Makes the remote `sh` dash and gives the host a `setsid`, as on a typical Linux
    /// machine. Returns false when this machine has no dash or perl to build that from.
    pub fn use_dash_with_setsid(&self) -> bool {
        let (Some(dash), Some(_perl)) = (find_tool("dash"), find_tool("perl")) else {
            return false;
        };
        let bin = self.path().join("remote-bin");
        fs::remove_file(bin.join("sh")).expect("remove sh");
        std::os::unix::fs::symlink(dash, bin.join("sh")).expect("link dash");
        let setsid = bin.join("setsid");
        fs::write(
            &setsid,
            "#!/bin/sh\nexec perl -e 'use POSIX; POSIX::setsid(); exec @ARGV or die' -- \"$@\"\n",
        )
        .expect("write setsid");
        use std::os::unix::fs::PermissionsExt;
        fs::set_permissions(&setsid, fs::Permissions::from_mode(0o755)).expect("chmod");
        true
    }

    fn log(&self, name: &str) -> Vec<String> {
        fs::read_to_string(self.path().join(name))
            .unwrap_or_default()
            .lines()
            .map(str::to_owned)
            .collect()
    }

    /// Every fake ssh invocation so far, one line of arguments each.
    pub fn ssh_log(&self) -> Vec<String> {
        self.log("ssh.log")
    }

    /// Every fake rsync invocation so far.
    pub fn rsync_log(&self) -> Vec<String> {
        self.log("rsync.log")
    }
}

/// Whether a process is still alive.
pub fn process_alive(pid: &str) -> bool {
    std::process::Command::new("kill")
        .args(["-0", pid.trim()])
        .output()
        .map(|output| output.status.success())
        .unwrap_or(false)
}

/// Waits (up to ten seconds) for a process to be gone.
pub fn wait_for_exit(pid: &str) -> bool {
    let deadline = Instant::now() + Duration::from_secs(10);
    while process_alive(pid) {
        if Instant::now() >= deadline {
            return false;
        }
        std::thread::sleep(Duration::from_millis(20));
    }
    true
}

/// Waits (up to ten seconds) for a file to exist and be non-empty, and returns it.
pub fn wait_for_file(path: &Path) -> String {
    let deadline = Instant::now() + Duration::from_secs(10);
    loop {
        if let Ok(text) = fs::read_to_string(path)
            && !text.trim().is_empty()
        {
            return text.trim().to_owned();
        }
        assert!(
            Instant::now() < deadline,
            "timed out waiting for {}",
            path.display()
        );
        std::thread::sleep(Duration::from_millis(20));
    }
}
