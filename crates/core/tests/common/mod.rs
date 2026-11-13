//! Shared helpers for the integration tests: a throwaway content root copied from
//! `tests/fixtures/root`, fake harness and uv scripts, and an event collector.
#![allow(dead_code)]

use std::fs;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Condvar, Mutex};
use std::time::{Duration, Instant};

use tempfile::TempDir;
use whetstone_core::{Core, CoreOptions, EventSink, ProgramHarness, RunEvent, RunLimits};

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
