//! The run target registry and the target check (`docs/ARCHITECTURE.md` section 4.1).
//!
//! `local` always exists. SSH targets are stored in the settings table as the host
//! alias, the remote directory, and the latest check; no key, password, or host name
//! is stored, because `ssh` reads those from the user's own configuration every time.
//!
//! A check is the bootstrap of a target, one step at a time: connect, find uv, copy
//! `python/`, build the environment, and run `learnai doctor --json`. It streams
//! [`TargetCheckEvent`]s, always finishes with exactly one `done` event, and saves the
//! result as the target's `lastCheck`.

use std::collections::{BTreeMap, HashSet};
use std::path::PathBuf;
use std::process::{Command, Stdio};
use std::sync::atomic::AtomicBool;
use std::sync::mpsc::{self, RecvTimeoutError};
use std::sync::{Arc, Mutex, MutexGuard, PoisonError};
use std::thread;
use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};

use crate::env::Environment;
use crate::error::{CoreError, Result};
use crate::model::{
    CheckStep, CheckStepId, Framework, RemoteDoctor, Requires, SshHost, TargetCheck,
    TargetCheckEvent, TargetInfo, TargetKind,
};
use crate::remote::{Session, SshTarget, StepFailure, target_name};
use crate::sink::SharedSink;
use crate::ssh::{DEFAULT_REMOTE_DIR, Ssh, is_host_alias, is_remote_dir, last_line};
use crate::store::Store;
use crate::stream::{Message, spawn_reader};
use crate::target::{LOCAL_TARGET, LocalProcess, Process, Target};

/// Settings key holding the SSH targets as a JSON array.
const TARGETS_SETTING: &str = "run_targets";
/// Settings key holding the latest check of the local target.
const LOCAL_CHECK_SETTING: &str = "local_check";

/// How long building the remote environment may take (a first build downloads Python
/// and PyTorch).
const ENVIRONMENT_LIMIT: Duration = Duration::from_secs(1800);
/// How long `learnai doctor` may take.
const DOCTOR_LIMIT: Duration = Duration::from_secs(180);

/// An SSH target as stored in the settings table.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
struct StoredTarget {
    host: String,
    remote_dir: String,
    #[serde(default)]
    last_check: Option<TargetCheck>,
}

impl StoredTarget {
    fn info(&self) -> TargetInfo {
        TargetInfo {
            name: target_name(&self.host),
            kind: TargetKind::Ssh,
            label: self.host.clone(),
            host: Some(self.host.clone()),
            remote_dir: Some(self.remote_dir.clone()),
            last_check: self.last_check.clone(),
        }
    }
}

/// `learnai doctor --json` as the CLI prints it (PROBLEM_AUTHORING section 19). Keys core
/// does not use, and keys added later, are ignored.
#[derive(Debug, Deserialize)]
struct WireDoctor {
    python: String,
    #[serde(default)]
    platform: WirePlatform,
    #[serde(default)]
    frameworks: BTreeMap<String, Option<String>>,
    #[serde(default)]
    hardware: BTreeMap<String, WireHardware>,
}

#[derive(Debug, Default, Deserialize)]
struct WirePlatform {
    #[serde(default)]
    system: String,
    #[serde(default)]
    machine: String,
}

#[derive(Debug, Default, Deserialize)]
struct WireHardware {
    #[serde(default)]
    present: bool,
    #[serde(default)]
    name: Option<String>,
    #[serde(default)]
    memory_bytes: Option<u64>,
}

/// Parses the document `learnai doctor --json` prints (indented over several lines).
/// Anything printed before it is skipped.
pub(crate) fn parse_doctor(stdout: &str) -> Option<RemoteDoctor> {
    // The document starts at a `{` that opens a line and runs to the last `}`.
    let end = stdout.rfind('}')?;
    let wire: WireDoctor = stdout
        .match_indices('{')
        .filter(|(index, _)| *index == 0 || stdout[..*index].ends_with('\n'))
        .find_map(|(index, _)| serde_json::from_str(stdout.get(index..=end)?).ok())?;
    let mut hardware = vec![Requires::Cpu];
    let mut gpu: Option<&WireHardware> = None;
    // CUDA first: on a machine with both, the NVIDIA GPU is the one worth naming.
    for requires in [Requires::Cuda, Requires::MlxGpu] {
        if let Some(entry) = wire.hardware.get(requires.as_str()).filter(|h| h.present) {
            hardware.push(requires);
            gpu = gpu.or(Some(entry));
        }
    }
    hardware.sort_unstable();
    let frameworks = [Framework::Torch, Framework::Jax, Framework::Mlx]
        .into_iter()
        .map(|framework| {
            let version = wire.frameworks.get(framework.as_str()).cloned().flatten();
            (framework, version)
        })
        .collect();
    Some(RemoteDoctor {
        python: wire.python,
        system: wire.platform.system,
        machine: wire.platform.machine,
        frameworks,
        hardware,
        gpu_name: gpu.and_then(|g| g.name.clone()),
        gpu_memory_bytes: gpu.and_then(|g| g.memory_bytes),
    })
}

/// The run targets of one content root.
#[derive(Clone)]
pub(crate) struct Targets {
    root: PathBuf,
    store: Arc<Store>,
    env: Environment,
    ssh: Ssh,
    local: Arc<dyn Target>,
    /// Names of the targets being checked.
    checking: Arc<Mutex<HashSet<String>>>,
}

impl Targets {
    pub fn new(
        root: impl Into<PathBuf>,
        store: Arc<Store>,
        env: Environment,
        ssh: Ssh,
        local: Arc<dyn Target>,
    ) -> Self {
        Self {
            root: root.into(),
            store,
            env,
            ssh,
            local,
            checking: Arc::new(Mutex::new(HashSet::new())),
        }
    }

    /// The SSH tooling.
    pub fn ssh(&self) -> &Ssh {
        &self.ssh
    }

    fn stored(&self) -> Result<Vec<StoredTarget>> {
        Ok(self.store.setting(TARGETS_SETTING)?.unwrap_or_default())
    }

    fn checking(&self) -> MutexGuard<'_, HashSet<String>> {
        self.checking.lock().unwrap_or_else(PoisonError::into_inner)
    }

    /// `list_ssh_hosts`: the host aliases in the user's SSH configuration.
    pub fn hosts(&self) -> Vec<SshHost> {
        self.ssh.hosts()
    }

    /// `list_targets`: `local` first, then the SSH targets in the order they were added.
    pub fn list(&self) -> Result<Vec<TargetInfo>> {
        let local = TargetInfo {
            name: LOCAL_TARGET.to_owned(),
            kind: TargetKind::Local,
            label: if cfg!(target_os = "macos") {
                "This Mac".to_owned()
            } else {
                "This computer".to_owned()
            },
            host: None,
            remote_dir: None,
            last_check: self.store.setting(LOCAL_CHECK_SETTING)?,
        };
        Ok(std::iter::once(local)
            .chain(self.stored()?.iter().map(StoredTarget::info))
            .collect())
    }

    /// `add_target`: registers the SSH host `host`, which must be an alias from the
    /// user's SSH configuration, working in `remote_dir` (default `~/.learnai`).
    pub fn add(&self, host: &str, remote_dir: Option<&str>) -> Result<Vec<TargetInfo>> {
        if !is_host_alias(host) || !self.hosts().iter().any(|known| known.alias == host) {
            return Err(CoreError::InvalidArgument(format!(
                "`{host}` is not a host in your SSH configuration. Add a `Host {host}` entry to \
                 ~/.ssh/config first."
            )));
        }
        let remote_dir = remote_dir
            .map(|dir| dir.trim().trim_end_matches('/'))
            .filter(|dir| !dir.is_empty())
            .unwrap_or(DEFAULT_REMOTE_DIR);
        if !is_remote_dir(remote_dir) {
            return Err(CoreError::InvalidArgument(format!(
                "`{remote_dir}` cannot be used as the remote directory. Use `~/<name>` or an \
                 absolute path made of letters, digits, `.`, `_`, and `-`."
            )));
        }
        let mut stored = self.stored()?;
        if stored.iter().any(|target| target.host == host) {
            return Err(CoreError::InvalidArgument(format!(
                "`{host}` is already a run target"
            )));
        }
        stored.push(StoredTarget {
            host: host.to_owned(),
            remote_dir: remote_dir.to_owned(),
            last_check: None,
        });
        self.store.set_setting(TARGETS_SETTING, &stored)?;
        tracing::info!(host, remote_dir, "added an SSH run target");
        self.list()
    }

    /// `remove_target`: forgets an SSH target. Nothing is deleted on the remote host.
    pub fn remove(&self, name: &str) -> Result<Vec<TargetInfo>> {
        if name == LOCAL_TARGET {
            return Err(CoreError::InvalidArgument(
                "the local target cannot be removed".to_owned(),
            ));
        }
        if self.checking().contains(name) {
            return Err(CoreError::InvalidArgument(format!(
                "`{name}` is being checked; wait for the check to finish"
            )));
        }
        let mut stored = self.stored()?;
        let before = stored.len();
        stored.retain(|target| target_name(&target.host) != name);
        if stored.len() == before {
            return Err(CoreError::UnknownTarget(name.to_owned()));
        }
        self.store.set_setting(TARGETS_SETTING, &stored)?;
        tracing::info!(name, "removed an SSH run target");
        self.list()
    }

    /// The stored SSH target called `name`: its host alias and remote directory.
    pub fn remote(&self, name: &str) -> Result<(String, String)> {
        self.stored()?
            .into_iter()
            .find(|target| target_name(&target.host) == name)
            .map(|target| (target.host, target.remote_dir))
            .ok_or_else(|| CoreError::UnknownTarget(name.to_owned()))
    }

    /// The target called `name`, ready to launch a run.
    pub fn resolve(&self, name: &str) -> Result<Arc<dyn Target>> {
        if name == LOCAL_TARGET {
            return Ok(Arc::clone(&self.local));
        }
        let (host, remote_dir) = self.remote(name)?;
        Ok(Arc::new(SshTarget::new(
            host,
            remote_dir,
            &self.root,
            self.ssh.clone(),
        )))
    }

    /// `check_target`: starts checking `name` and returns the job id at once. Events
    /// flow to `sink` from a background thread and always finish with exactly one
    /// `done` event. One check per target runs at a time.
    pub fn check(&self, name: &str, sink: SharedSink<TargetCheckEvent>) -> Result<String> {
        let remote = if name == LOCAL_TARGET {
            None
        } else {
            Some(self.remote(name)?)
        };
        if !self.checking().insert(name.to_owned()) {
            return Err(CoreError::InvalidArgument(format!(
                "`{name}` is already being checked"
            )));
        }
        let job_id = uuid::Uuid::new_v4().to_string();
        let this = self.clone();
        let name = name.to_owned();
        let owned = name.clone();
        let spawned = thread::Builder::new()
            .name(format!("target-check-{job_id}"))
            .spawn(move || {
                let mut run = CheckRun::new(&sink);
                match &remote {
                    None => this.check_local(&mut run),
                    Some((host, remote_dir)) => this.check_remote(host, remote_dir, &mut run),
                }
                let check = run.finish();
                if let Err(error) = this.save_check(&owned, &check) {
                    tracing::warn!("could not save the check of `{owned}`: {error}");
                }
                tracing::info!(target = %owned, ok = check.ok, "target check finished");
                // Free the slot first, so a retry started from the `done` handler works.
                this.checking().remove(&owned);
                sink.emit(TargetCheckEvent::Done { check });
            });
        match spawned {
            Ok(_) => Ok(job_id),
            Err(source) => {
                self.checking().remove(&name);
                Err(CoreError::Io {
                    context: "failed to start the check thread".to_owned(),
                    source,
                })
            }
        }
    }

    fn save_check(&self, name: &str, check: &TargetCheck) -> Result<()> {
        if name == LOCAL_TARGET {
            return self.store.set_setting(LOCAL_CHECK_SETTING, check);
        }
        let mut stored = self.stored()?;
        // The target may have been removed while it was being checked.
        if let Some(target) = stored
            .iter_mut()
            .find(|target| target_name(&target.host) == name)
        {
            target.last_check = Some(check.clone());
            self.store.set_setting(TARGETS_SETTING, &stored)?;
        }
        Ok(())
    }

    /// The local check: uv, then the doctor report of the environment as it is. It
    /// never installs anything.
    fn check_local(&self, run: &mut CheckRun<'_>) {
        let uv = self.env.uv();
        let version = Command::new(uv.program())
            .arg("--version")
            .stdin(Stdio::null())
            .stderr(Stdio::null())
            .output()
            .ok()
            .filter(|output| output.status.success())
            .map(|output| String::from_utf8_lossy(&output.stdout).trim().to_owned());
        let Some(version) = version else {
            run.fail(
                CheckStepId::Uv,
                "uv was not found on this computer.",
                Some(format!(
                    "Install it with uv's official installer, then reopen Learn AI: {}",
                    crate::ssh::UV_INSTALL_COMMAND
                )),
            );
            return;
        };
        run.pass(CheckStepId::Uv, format!("Found {version}."));

        let mut spec = uv.run_no_sync();
        spec.args
            .extend(["learnai".into(), "doctor".into(), "--json".into()]);
        spec.envs
            .push(("LEARNAI_ROOT".into(), self.root.clone().into()));
        let process = match LocalProcess::spawn(&spec) {
            Ok(process) => process,
            Err(error) => {
                run.fail(
                    CheckStepId::Doctor,
                    format!("The environment report could not be started: {error}."),
                    None,
                );
                return;
            }
        };
        run.doctor(
            process,
            Some(
                "Set up the Python environment under Settings, Environment, then check again."
                    .to_owned(),
            ),
        );
    }

    fn check_remote(&self, host: &str, remote_dir: &str, run: &mut CheckRun<'_>) {
        let never = AtomicBool::new(false);
        let session = match Session::connect(&self.ssh, host, remote_dir, &never) {
            Ok(session) => session,
            Err(failure) => return run.fail_with(CheckStepId::Connect, failure),
        };
        let machine = [session.host.system.as_str(), session.host.machine.as_str()]
            .into_iter()
            .filter(|part| !part.is_empty())
            .collect::<Vec<_>>()
            .join(" ");
        run.pass(
            CheckStepId::Connect,
            if machine.is_empty() {
                format!("Connected to {host}.")
            } else {
                format!("Connected to {host} ({machine}).")
            },
        );

        match session.uv() {
            Ok(uv) => run.pass(CheckStepId::Uv, format!("Found uv at {uv}.")),
            Err(failure) => return run.fail_with(CheckStepId::Uv, failure),
        }

        if let Err(failure) = session.sync_python(&self.root) {
            return run.fail_with(CheckStepId::Sync, failure);
        }
        run.pass(
            CheckStepId::Sync,
            format!("Copied the judge to {}/python.", session.root),
        );

        // An inexact sync: build what is missing, remove nothing an earlier run added.
        let build = session.uv().map(|uv| {
            vec![
                uv.to_owned(),
                "sync".to_owned(),
                "--inexact".to_owned(),
                "--project".to_owned(),
                format!("{}/python", session.root),
            ]
        });
        let process = match build.and_then(|command| {
            run.log(format!("$ {}", command.join(" ")));
            session.spawn(&command)
        }) {
            Ok(process) => process,
            Err(failure) => return run.fail_with(CheckStepId::Environment, failure),
        };
        let output = stream(process, ENVIRONMENT_LIMIT, true, |line| {
            run.log(line.to_owned())
        });
        match output.code {
            Some(0) => run.pass(CheckStepId::Environment, "The Python environment is ready."),
            _ => {
                return run.fail(
                    CheckStepId::Environment,
                    output.failure("Building the Python environment"),
                    Some(format!(
                        "Read the log above. To see the error on the machine itself, run \
                         `ssh {host}` and then `uv sync --project {}/python`.",
                        session.root
                    )),
                );
            }
        }

        let process = match session.uv_run(&[], true).and_then(|mut command| {
            command.extend(["learnai", "doctor", "--json"].map(str::to_owned));
            session.spawn(&command)
        }) {
            Ok(process) => process,
            Err(failure) => return run.fail_with(CheckStepId::Doctor, failure),
        };
        run.doctor(process, None);
    }
}

/// The output of a streamed command.
struct Streamed {
    code: Option<i32>,
    stdout: String,
    last_stderr: Option<String>,
    timed_out: bool,
}

impl Streamed {
    /// One line saying why the command did not succeed.
    fn failure(&self, what: &str) -> String {
        if self.timed_out {
            return format!("{what} took too long and was stopped.");
        }
        let reason = self
            .last_stderr
            .as_deref()
            .and_then(last_line)
            .map(str::to_owned)
            .unwrap_or_else(|| match self.code {
                Some(255) => "the connection dropped".to_owned(),
                Some(code) => format!("it exited with code {code}"),
                None => "it was killed by a signal".to_owned(),
            });
        format!("{what} failed: {reason}.")
    }
}

/// Runs `process` to completion, passing every stderr line (and, with `log_stdout`,
/// every stdout line) to `on_log` and collecting stdout. Stops the process at `limit`.
fn stream(
    mut process: impl Process,
    limit: Duration,
    log_stdout: bool,
    mut on_log: impl FnMut(&str),
) -> Streamed {
    let (tx, rx) = mpsc::channel();
    let mut open = 0;
    if let Some(stdout) = process.take_stdout() {
        open += spawn_reader(stdout, tx.clone(), Message::Stdout);
    }
    if let Some(stderr) = process.take_stderr() {
        open += spawn_reader(stderr, tx.clone(), Message::Stderr);
    }
    drop(tx);
    let deadline = Instant::now() + limit;
    let mut output = Streamed {
        code: None,
        stdout: String::new(),
        last_stderr: None,
        timed_out: false,
    };
    while open > 0 {
        match rx.recv_timeout(deadline.saturating_duration_since(Instant::now())) {
            Ok(Message::Closed) => open -= 1,
            Ok(Message::Stdout(line)) => {
                if log_stdout {
                    on_log(&line);
                }
                output.stdout.push_str(&line);
                output.stdout.push('\n');
            }
            Ok(Message::Stderr(line)) => {
                on_log(&line);
                if !line.trim().is_empty() {
                    output.last_stderr = Some(line);
                }
            }
            Ok(Message::Cancel) => {}
            Err(RecvTimeoutError::Timeout) => {
                output.timed_out = true;
                break;
            }
            Err(RecvTimeoutError::Disconnected) => break,
        }
    }
    output.code = process.finish(if output.timed_out {
        Duration::ZERO
    } else {
        Duration::from_secs(5)
    });
    output
}

/// A check in progress: emits each step as it ends and builds the final result.
struct CheckRun<'a> {
    sink: &'a SharedSink<TargetCheckEvent>,
    steps: Vec<CheckStep>,
    doctor: Option<RemoteDoctor>,
}

impl<'a> CheckRun<'a> {
    fn new(sink: &'a SharedSink<TargetCheckEvent>) -> Self {
        Self {
            sink,
            steps: Vec::new(),
            doctor: None,
        }
    }

    fn log(&self, text: String) {
        self.sink.emit(TargetCheckEvent::Log { text });
    }

    fn step(&mut self, step: CheckStep) {
        self.sink
            .emit(TargetCheckEvent::Step { step: step.clone() });
        self.steps.push(step);
    }

    fn pass(&mut self, id: CheckStepId, message: impl Into<String>) {
        self.step(CheckStep {
            id,
            ok: true,
            message: message.into(),
            fix: None,
        });
    }

    fn fail(&mut self, id: CheckStepId, message: impl Into<String>, fix: Option<String>) {
        self.step(CheckStep {
            id,
            ok: false,
            message: message.into(),
            fix,
        });
    }

    fn fail_with(&mut self, id: CheckStepId, failure: StepFailure) {
        self.fail(id, failure.message, failure.fix);
    }

    /// Runs `learnai doctor --json` (already started as `process`) and records its report.
    fn doctor(&mut self, process: impl Process, fix: Option<String>) {
        let output = stream(process, DOCTOR_LIMIT, false, |line| {
            self.log(line.to_owned())
        });
        let report = match output.code {
            Some(0) => parse_doctor(&output.stdout),
            _ => None,
        };
        match report {
            Some(report) => {
                let hardware = match (&report.gpu_name, report.hardware.len()) {
                    (Some(gpu), _) => gpu.clone(),
                    (None, 1) => "no GPU".to_owned(),
                    (None, _) => "a GPU".to_owned(),
                };
                self.pass(
                    CheckStepId::Doctor,
                    format!("Python {}, {hardware}.", report.python),
                );
                self.doctor = Some(report);
            }
            None if output.code == Some(0) => self.fail(
                CheckStepId::Doctor,
                "The environment report could not be read.",
                fix,
            ),
            None => self.fail(
                CheckStepId::Doctor,
                output.failure("The environment report"),
                fix,
            ),
        }
    }

    fn finish(self) -> TargetCheck {
        TargetCheck {
            ok: !self.steps.is_empty() && self.steps.iter().all(|step| step.ok),
            checked_at: crate::now_iso8601(),
            steps: self.steps,
            doctor: self.doctor,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const DOCTOR: &str = r#"warning: something on stdout
{
  "schema": 1,
  "learnai": "0.1.0",
  "python": "3.15.2",
  "executable": "/abs/path/python/.venv/bin/python3",
  "platform": {"system": "Linux", "machine": "x86_64", "release": "6.8.0"},
  "frameworks": {"torch": "2.15.0", "jax": "0.12.0.1", "mlx": null},
  "packages": {"numpy": "2.5.4"},
  "hardware": {
    "cpu": {"present": true, "cores": 32},
    "mlx-gpu": {"present": false, "name": null, "memory_bytes": null},
    "cuda": {"present": true, "name": "NVIDIA GeForce RTX 4090", "memory_bytes": 25757220864,
             "devices": [{"index": 0, "name": "NVIDIA GeForce RTX 4090", "memory_bytes": 25757220864}],
             "torch_cuda_version": "13.0"}
  },
  "hardware_override": null,
  "root": "/abs/content/root",
  "models_dir": "/abs/content/root/.learnai/models"
}
"#;

    #[test]
    fn the_documented_doctor_report_maps_to_the_wire_type() {
        let doctor = parse_doctor(DOCTOR).expect("doctor");
        assert_eq!(doctor.python, "3.15.2");
        assert_eq!(doctor.system, "Linux");
        assert_eq!(doctor.machine, "x86_64");
        assert_eq!(
            doctor.frameworks,
            BTreeMap::from([
                (Framework::Torch, Some("2.15.0".to_owned())),
                (Framework::Jax, Some("0.12.0.1".to_owned())),
                (Framework::Mlx, None),
            ])
        );
        assert_eq!(doctor.hardware, [Requires::Cpu, Requires::Cuda]);
        assert_eq!(doctor.gpu_name.as_deref(), Some("NVIDIA GeForce RTX 4090"));
        assert_eq!(doctor.gpu_memory_bytes, Some(25_757_220_864));
    }

    #[test]
    fn a_mac_and_a_bare_machine_report_what_they_have() {
        let mac = parse_doctor(
            r#"{"python":"3.15.2","platform":{"system":"Darwin","machine":"arm64"},
                "frameworks":{"torch":"2.15.0","mlx":"0.32.4"},
                "hardware":{"cpu":{"present":true},"mlx-gpu":{"present":true,"name":"Apple M1 Max","memory_bytes":68719476736},
                            "cuda":{"present":false,"name":null,"memory_bytes":null}}}"#,
        )
        .expect("doctor");
        assert_eq!(mac.hardware, [Requires::Cpu, Requires::MlxGpu]);
        assert_eq!(mac.gpu_name.as_deref(), Some("Apple M1 Max"));
        assert_eq!(mac.frameworks[&Framework::Jax], None);

        let bare = parse_doctor(r#"{"python":"3.15.2"}"#).expect("doctor");
        assert_eq!(bare.hardware, [Requires::Cpu]);
        assert_eq!(bare.gpu_name, None);
        assert_eq!(bare.gpu_memory_bytes, None);

        assert_eq!(parse_doctor("Traceback (most recent call last):"), None);
        assert_eq!(parse_doctor(""), None);
        assert_eq!(parse_doctor("{not json}"), None);
    }
}
