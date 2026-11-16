//! Running things on a remote host over SSH: the SSH run target and the pieces the
//! target check and remote model downloads share with it.
//!
//! A [`Session`] is one host reached and probed: it knows the remote home directory,
//! where uv is, and the remote content root `<remoteDir>/root`. It syncs files there
//! with rsync and starts supervised commands.
//!
//! **Stopping a remote command.** Killing the local `ssh` client does not stop what it
//! started: without a terminal the remote process keeps running. Every long-running
//! remote command therefore runs under [`crate::ssh::SUPERVISOR_SCRIPT`], which puts it
//! in its own process group and kills that group (and every descendant) as soon as its
//! standard input, the SSH channel, closes. [`SshProcess`] holds that channel open
//! through a pipe to the local `ssh`. Closing the pipe is how a run is stopped, and the
//! operating system closes it for us if the app exits or crashes, so nothing outlives
//! the app on the remote host.

use std::ffi::OsString;
use std::fs::{self, File};
use std::io::{self, Read};
use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

use crate::harness::{CommandSpec, HarnessJob};
use crate::ssh::{
    PROBE_SCRIPT, SUPERVISOR_SCRIPT, Ssh, UV_INSTALL_COMMAND, diagnose_connection, is_path_segment,
    is_plain_absolute_path, last_line, resolve_remote_dir, sh_join, sh_script,
};
use crate::target::{LaunchFailure, LocalProcess, Process, Target};
use crate::uv::extra_names;

/// How long a connection attempt may take in all (the SSH connect timeout plus the
/// probe).
const CONNECT_LIMIT: Duration = Duration::from_secs(30);
/// How long one file sync may take.
const SYNC_LIMIT: Duration = Duration::from_secs(300);
/// How long the remote side gets to kill its process tree after the channel closes,
/// before the local `ssh` is killed too.
const REMOTE_STOP_GRACE: Duration = Duration::from_secs(3);

/// What is never copied from `python/`: the virtual environment (each host builds its
/// own) and caches. rsync does not delete excluded files on the receiving side, so the
/// remote environment survives every sync.
const PYTHON_EXCLUDES: &[&str] = &[
    ".venv",
    "__pycache__",
    "*.pyc",
    ".pytest_cache",
    ".ruff_cache",
    ".mypy_cache",
    "*.egg-info",
];

/// Why a remote step did not succeed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct StepFailure {
    /// One line saying what went wrong.
    pub message: String,
    /// A command or instruction for the user.
    pub fix: Option<String>,
    /// The step was abandoned because the job was canceled.
    pub canceled: bool,
}

impl StepFailure {
    pub(crate) fn new(message: impl Into<String>, fix: Option<String>) -> Self {
        Self {
            message: message.into(),
            fix,
            canceled: false,
        }
    }

    fn canceled() -> Self {
        Self {
            message: "Canceled.".to_owned(),
            fix: None,
            canceled: true,
        }
    }

    /// The failure as one paragraph: what went wrong, then what to do.
    pub(crate) fn text(&self) -> String {
        match &self.fix {
            Some(fix) => format!("{} {fix}", self.message),
            None => self.message.clone(),
        }
    }
}

/// The output of a command that ran to completion.
struct Captured {
    code: Option<i32>,
    stdout: String,
    stderr: String,
}

/// A file for a command's output that removes itself.
struct Scratch(PathBuf);

impl Scratch {
    fn create(dir: &Path, suffix: &str) -> io::Result<(Self, File)> {
        let path = dir.join(format!("learnai-{}.{suffix}", uuid::Uuid::new_v4()));
        let mut options = fs::OpenOptions::new();
        options.write(true).create_new(true);
        #[cfg(unix)]
        {
            use std::os::unix::fs::OpenOptionsExt;
            options.mode(0o600);
        }
        let file = options.open(&path)?;
        Ok((Self(path), file))
    }

    fn read(&self) -> String {
        let mut text = Vec::new();
        if let Ok(mut file) = File::open(&self.0) {
            let _ = file.read_to_end(&mut text);
        }
        String::from_utf8_lossy(&text).into_owned()
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = fs::remove_file(&self.0);
    }
}

/// Runs `spec` to completion and returns its output, giving up at `limit` or when
/// `canceled` becomes true.
///
/// Output goes to files rather than pipes: the `ssh` that opens a shared connection
/// leaves a background process behind that would hold a pipe open long after the
/// command itself has finished.
fn run_captured(
    spec: &CommandSpec,
    scratch: &Path,
    limit: Duration,
    canceled: &AtomicBool,
) -> Result<Captured, StepFailure> {
    let failed = |error: io::Error| StepFailure::new(format!("{error}."), None);
    let (out, out_file) = Scratch::create(scratch, "out").map_err(failed)?;
    let (err, err_file) = Scratch::create(scratch, "err").map_err(failed)?;
    let mut process = LocalProcess::spawn_io(spec, Stdio::null(), out_file.into(), err_file.into())
        .map_err(|error| {
            let program = spec.program.to_string_lossy().into_owned();
            if error.kind() == io::ErrorKind::NotFound {
                StepFailure::new(
                    format!("`{program}` was not found on this computer."),
                    Some(format!("Install `{program}`, then try again.")),
                )
            } else {
                StepFailure::new(format!("`{program}` could not be started: {error}."), None)
            }
        })?;
    let deadline = Instant::now() + limit;
    loop {
        if process.wait_exit(Duration::from_millis(25)) {
            break;
        }
        if canceled.load(Ordering::SeqCst) {
            process.finish(Duration::ZERO);
            return Err(StepFailure::canceled());
        }
        if Instant::now() >= deadline {
            process.finish(Duration::ZERO);
            return Err(StepFailure::new(
                format!(
                    "`{}` did not finish within {} s.",
                    spec.program.to_string_lossy(),
                    limit.as_secs()
                ),
                None,
            ));
        }
    }
    let code = process.finish(Duration::ZERO);
    Ok(Captured {
        code,
        stdout: out.read(),
        stderr: err.read(),
    })
}

/// What the probe found on the remote host.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub(crate) struct RemoteHost {
    /// The remote home directory.
    pub home: String,
    /// The uv executable, when there is one.
    pub uv: Option<String>,
    /// Whether rsync is installed there.
    pub rsync: bool,
    /// `uname -s`.
    pub system: String,
    /// `uname -m`.
    pub machine: String,
}

/// Reads the probe's `learnai-<key>=<value>` lines; anything else is ignored.
pub(crate) fn parse_probe(stdout: &str) -> Option<RemoteHost> {
    let mut host = RemoteHost::default();
    let mut found_home = false;
    for line in stdout.lines() {
        let Some((key, value)) = line
            .trim()
            .strip_prefix("learnai-")
            .and_then(|l| l.split_once('='))
        else {
            continue;
        };
        let value = value.trim();
        match key {
            "home" => {
                host.home = value.to_owned();
                found_home = true;
            }
            "uv" => host.uv = (!value.is_empty()).then(|| value.to_owned()),
            "rsync" => host.rsync = !value.is_empty(),
            "system" => host.system = value.to_owned(),
            "machine" => host.machine = value.to_owned(),
            _ => {}
        }
    }
    found_home.then_some(host)
}

/// A remote host that has been reached and probed.
pub(crate) struct Session<'a> {
    ssh: &'a Ssh,
    alias: String,
    /// What the probe found.
    pub host: RemoteHost,
    /// The remote content root, absolute: `<remoteDir>/root`.
    pub root: String,
    canceled: &'a AtomicBool,
}

impl<'a> Session<'a> {
    /// Connects to `alias` (opening the shared connection the following commands
    /// reuse) and probes the host.
    pub fn connect(
        ssh: &'a Ssh,
        alias: &str,
        remote_dir: &str,
        canceled: &'a AtomicBool,
    ) -> Result<Self, StepFailure> {
        let spec = ssh
            .command(alias, &sh_script(PROBE_SCRIPT, &[]), true)
            .map_err(|message| StepFailure::new(format!("{message}."), None))?;
        let output = run_captured(&spec, &ssh.scratch_dir(), CONNECT_LIMIT, canceled)?;
        let host = match output.code {
            Some(0) => parse_probe(&output.stdout),
            _ => None,
        };
        let Some(host) = host else {
            // 255 is ssh's own failure; anything else means the host was reached but
            // could not run a plain `sh` script.
            return Err(if matches!(output.code, Some(255) | None) {
                let diagnosis = diagnose_connection(alias, &output.stderr);
                StepFailure::new(diagnosis.message, Some(diagnosis.fix))
            } else {
                StepFailure::new(
                    format!(
                        "`{alias}` was reached but could not run a shell command: {}.",
                        last_line(&output.stderr).unwrap_or("it gave no reason")
                    ),
                    Some(format!(
                        "Learn AI needs a POSIX `sh` on the remote machine. Check that \
                         `ssh {alias} sh -c true` works in a terminal."
                    )),
                )
            });
        };
        // The remote shell parses rsync's destination, so the root must be a path with
        // nothing to expand or split. The home directory only matters when the remote
        // directory is relative to it.
        let root = format!("{}/root", resolve_remote_dir(remote_dir, &host.home));
        if !is_plain_absolute_path(&root) {
            return Err(StepFailure::new(
                format!(
                    "The remote directory on `{alias}` resolves to `{root}`, which contains \
                     characters Learn AI cannot pass to rsync safely."
                ),
                Some(
                    "Remove the target and add it again with an absolute remote directory made \
                     of letters, digits, `.`, `_`, and `-`."
                        .to_owned(),
                ),
            ));
        }
        Ok(Self {
            ssh,
            alias: alias.to_owned(),
            host,
            root,
            canceled,
        })
    }

    /// The uv executable on the remote host, or how to get one.
    pub fn uv(&self) -> Result<&str, StepFailure> {
        self.host.uv.as_deref().ok_or_else(|| {
            StepFailure::new(
                format!(
                    "uv is not installed on `{}`. Install it with uv's official installer, then \
                     try again:",
                    self.alias
                ),
                Some(format!("ssh {} '{UV_INSTALL_COMMAND}'", self.alias)),
            )
        })
    }

    /// Confirms rsync exists on the remote host, or says how to get it.
    pub fn require_rsync(&self) -> Result<(), StepFailure> {
        if self.host.rsync {
            return Ok(());
        }
        Err(StepFailure::new(
            format!("rsync is not installed on `{}`.", self.alias),
            Some(format!(
                "Install rsync there with its package manager, for example \
                 `ssh -t {} sudo apt install rsync`, then try again.",
                self.alias
            )),
        ))
    }

    fn run(&self, spec: &CommandSpec, limit: Duration, what: &str) -> Result<(), StepFailure> {
        let output = run_captured(spec, &self.ssh.scratch_dir(), limit, self.canceled)?;
        match output.code {
            Some(0) => Ok(()),
            Some(255) => {
                let diagnosis = diagnose_connection(&self.alias, &output.stderr);
                Err(StepFailure::new(diagnosis.message, Some(diagnosis.fix)))
            }
            _ => Err(StepFailure::new(
                format!(
                    "{what} failed: {}.",
                    last_line(&output.stderr).unwrap_or("the command gave no reason")
                ),
                None,
            )),
        }
    }

    fn mkdir(&self, dirs: &[String]) -> Result<(), StepFailure> {
        let mut words = vec!["mkdir".to_owned(), "-p".to_owned(), "--".to_owned()];
        words.extend(dirs.iter().cloned());
        let spec = self
            .ssh
            .command(&self.alias, &sh_join(&words), false)
            .map_err(|message| StepFailure::new(format!("{message}."), None))?;
        self.run(&spec, CONNECT_LIMIT, "Creating the remote directories")
    }

    fn rsync(
        &self,
        source: OsString,
        dest: &str,
        delete: bool,
        excludes: &[&str],
        what: &str,
    ) -> Result<(), StepFailure> {
        let spec = self
            .ssh
            .rsync(&self.alias, &[source], dest, delete, excludes)
            .map_err(|message| StepFailure::new(format!("{message}."), None))?;
        self.run(&spec, SYNC_LIMIT, what)
    }

    /// Copies `<local root>/python/` to the remote root, without the virtual
    /// environment and caches. Only what changed is sent.
    pub fn sync_python(&self, local_root: &Path) -> Result<(), StepFailure> {
        self.require_rsync()?;
        let dest = format!("{}/python", self.root);
        self.mkdir(std::slice::from_ref(&dest))?;
        self.rsync(
            directory_contents(&local_root.join("python")),
            &format!("{dest}/"),
            true,
            PYTHON_EXCLUDES,
            "Copying the judge",
        )
    }

    /// Copies a problem directory to `<root>/problems/<track>/<id>/` and returns the
    /// remote directory.
    pub fn sync_problem(&self, problem_dir: &Path) -> Result<String, StepFailure> {
        let (track, id) = track_and_id(problem_dir)?;
        let dest = format!("{}/problems/{track}/{id}", self.root);
        self.mkdir(std::slice::from_ref(&dest))?;
        // The track file is not needed to judge, but it keeps the remote catalog valid
        // for `learnai` commands that scan it.
        let track_file = problem_dir.parent().map(|dir| dir.join("track.toml"));
        if let Some(track_file) = track_file.filter(|file| file.is_file()) {
            self.rsync(
                track_file.into(),
                &format!("{}/problems/{track}/track.toml", self.root),
                false,
                &[],
                "Copying the problem",
            )?;
        }
        self.rsync(
            directory_contents(problem_dir),
            &format!("{dest}/"),
            true,
            &["__pycache__", "*.pyc"],
            "Copying the problem",
        )?;
        Ok(dest)
    }

    /// Copies the solution file to `<root>/workspace/<id>/<framework>.py` (a review
    /// attempt to `<root>/workspace/<id>/review/<framework>.py`, so it never replaces
    /// the saved solution there either) and returns the remote path.
    pub fn sync_solution(&self, job: &HarnessJob) -> Result<String, StepFailure> {
        let (_track, id) = track_and_id(&job.problem_dir)?;
        let dir = if job.review {
            format!("{}/workspace/{id}/review", self.root)
        } else {
            format!("{}/workspace/{id}", self.root)
        };
        self.mkdir(std::slice::from_ref(&dir))?;
        let dest = format!("{dir}/{}.py", job.framework);
        self.rsync(
            job.solution.clone().into(),
            &dest,
            false,
            &[],
            "Copying the solution",
        )?;
        Ok(dest)
    }

    /// The local `ssh` command that runs `command` on the remote host under the
    /// supervisor, with `LEARNAI_ROOT` set to the remote root.
    pub fn supervised(&self, command: &[String]) -> Result<CommandSpec, StepFailure> {
        let mut args = vec![self.root.clone()];
        args.extend(command.iter().cloned());
        self.ssh
            .command(&self.alias, &sh_script(SUPERVISOR_SCRIPT, &args), false)
            .map_err(|message| StepFailure::new(format!("{message}."), None))
    }

    /// Starts `command` on the remote host under the supervisor. stdout and stderr are
    /// separate pipes.
    pub fn spawn(&self, command: &[String]) -> Result<SshProcess, StepFailure> {
        let spec = self.supervised(command)?;
        SshProcess::spawn(&spec, &self.alias).map_err(|error| {
            StepFailure::new(format!("`ssh` could not be started: {error}."), None)
        })
    }

    /// `<uv> run --project <root>/python [--extra ...] [--no-sync] --`: the prefix of
    /// every Python command on the remote host.
    pub fn uv_run(&self, extras: &[&str], no_sync: bool) -> Result<Vec<String>, StepFailure> {
        let mut command = vec![
            self.uv()?.to_owned(),
            "run".to_owned(),
            "--project".to_owned(),
            format!("{}/python", self.root),
        ];
        for extra in extras {
            command.extend(["--extra".to_owned(), (*extra).to_owned()]);
        }
        if no_sync {
            command.push("--no-sync".to_owned());
        }
        command.push("--".to_owned());
        Ok(command)
    }
}

/// `dir/`: for rsync, the contents of the directory rather than the directory.
fn directory_contents(dir: &Path) -> OsString {
    let mut source = dir.as_os_str().to_owned();
    source.push("/");
    source
}

/// The track and problem id a problem directory stands for
/// (`.../problems/<track>/<id>`), checked to be plain path components.
fn track_and_id(problem_dir: &Path) -> Result<(String, String), StepFailure> {
    let name = |path: Option<&Path>| {
        path.and_then(Path::file_name)
            .and_then(|name| name.to_str())
            .filter(|name| is_path_segment(name))
            .map(str::to_owned)
    };
    match (name(problem_dir.parent()), name(Some(problem_dir))) {
        (Some(track), Some(id)) => Ok((track, id)),
        _ => Err(StepFailure::new(
            format!(
                "The problem directory {} cannot be mirrored on a remote host: its track and \
                 id must be made of letters, digits, `.`, `_`, and `-`.",
                problem_dir.display()
            ),
            None,
        )),
    }
}

/// A command running on a remote host under the supervisor, seen through the local
/// `ssh` client.
pub struct SshProcess {
    inner: LocalProcess,
    alias: String,
    stopped: bool,
}

impl SshProcess {
    /// Spawns the local `ssh` for a supervised remote command. Its standard input is a
    /// pipe this process holds open: the remote supervisor kills the command when it
    /// closes.
    pub fn spawn(spec: &CommandSpec, alias: &str) -> io::Result<Self> {
        Ok(Self {
            inner: LocalProcess::spawn_io(spec, Stdio::piped(), Stdio::piped(), Stdio::piped())?,
            alias: alias.to_owned(),
            stopped: false,
        })
    }
}

impl Process for SshProcess {
    fn take_stdout(&mut self) -> Option<Box<dyn Read + Send>> {
        self.inner.take_stdout()
    }

    fn take_stderr(&mut self) -> Option<Box<dyn Read + Send>> {
        self.inner.take_stderr()
    }

    /// Stops the remote command, then the local client. Closing the channel makes the
    /// remote supervisor kill the command's whole process tree and exit, which ends
    /// the `ssh` session by itself; the local `ssh` is killed only if that takes too
    /// long (a stalled network), in which case the remote side cleans up as soon as its
    /// SSH daemon notices the dead connection.
    fn kill(&mut self) {
        if self.stopped {
            return;
        }
        self.stopped = true;
        self.inner.close_stdin();
        self.inner.wait_exit(REMOTE_STOP_GRACE);
        self.inner.kill();
    }

    fn finish(&mut self, grace: Duration) -> Option<i32> {
        self.inner.wait_exit(grace);
        self.kill();
        self.inner.finish(Duration::ZERO)
    }

    fn exit_note(&self, code: Option<i32>) -> Option<String> {
        // 255 is how ssh reports its own failure, as opposed to the remote command's.
        (code == Some(255)).then(|| {
            format!(
                "The connection to `{}` dropped before the judge reported a result. Check the \
                 network and that the machine is still on, then run again.",
                self.alias
            )
        })
    }
}

impl Drop for SshProcess {
    fn drop(&mut self) {
        self.kill();
    }
}

/// Runs harness jobs on a remote host over SSH.
pub struct SshTarget {
    name: String,
    alias: String,
    remote_dir: String,
    local_root: PathBuf,
    ssh: Ssh,
}

impl SshTarget {
    /// The target `ssh:<alias>`, working in `remote_dir` on the host and mirroring
    /// files from the content root at `local_root`.
    pub fn new(
        alias: impl Into<String>,
        remote_dir: impl Into<String>,
        local_root: impl Into<PathBuf>,
        ssh: Ssh,
    ) -> Self {
        let alias = alias.into();
        Self {
            name: target_name(&alias),
            alias,
            remote_dir: remote_dir.into(),
            local_root: local_root.into(),
            ssh,
        }
    }

    fn prepare(
        &self,
        job: &HarnessJob,
        progress: &dyn Fn(String),
        canceled: &AtomicBool,
    ) -> Result<SshProcess, StepFailure> {
        let alias = &self.alias;
        progress(format!("Connecting to {alias}"));
        let session = Session::connect(&self.ssh, alias, &self.remote_dir, canceled)?;
        session.uv()?;
        progress(format!("Copying the judge to {alias}"));
        session.sync_python(&self.local_root)?;
        progress(format!("Copying the problem and your solution to {alias}"));
        let problem = session.sync_problem(&job.problem_dir)?;
        let solution = session.sync_solution(job)?;

        // The framework's own extra is all a run needs. `uv run` only adds packages, so
        // this never removes what an earlier run or check installed there.
        let extras: Vec<_> = job.framework.extra().into_iter().collect();
        let mut command = session.uv_run(&extra_names(&extras), false)?;
        command.extend(["python", "-m", "learnai.harness"].map(str::to_owned));
        command.extend([
            "--problem".to_owned(),
            problem,
            "--solution".to_owned(),
            solution,
            "--framework".to_owned(),
            job.framework.to_string(),
            "--mode".to_owned(),
            job.mode.to_string(),
        ]);
        if let Some(case) = &job.case {
            command.extend(["--case".to_owned(), case.clone()]);
        }
        progress(format!(
            "Syncing the Python environment on {alias} and starting the judge"
        ));
        session.spawn(&command)
    }
}

/// The target name of an SSH host alias: `ssh:<alias>`.
pub fn target_name(alias: &str) -> String {
    format!("ssh:{alias}")
}

impl Target for SshTarget {
    fn name(&self) -> &str {
        &self.name
    }

    fn launch(
        &self,
        job: &HarnessJob,
        progress: &dyn Fn(String),
        canceled: &AtomicBool,
    ) -> io::Result<Box<dyn Process>> {
        match self.prepare(job, progress, canceled) {
            Ok(process) => Ok(Box::new(process)),
            Err(failure) => Err(LaunchFailure::into_io(failure.text())),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_probe_is_read_through_noise() {
        let host = parse_probe(
            "Welcome to Ubuntu\nlearnai-home=/home/ada\nlearnai-uv=/home/ada/.local/bin/uv\n\
             learnai-rsync=/usr/bin/rsync\nlearnai-system=Linux\nlearnai-machine=x86_64\nlearnai-future=1\n",
        )
        .expect("probe");
        assert_eq!(
            host,
            RemoteHost {
                home: "/home/ada".into(),
                uv: Some("/home/ada/.local/bin/uv".into()),
                rsync: true,
                system: "Linux".into(),
                machine: "x86_64".into(),
            }
        );
        let bare = parse_probe("learnai-home=/root\nlearnai-uv=\nlearnai-rsync=\n").expect("probe");
        assert_eq!(bare.uv, None);
        assert!(!bare.rsync);
        assert_eq!(parse_probe("command not found: sh\n"), None);
        assert_eq!(parse_probe(""), None);
    }

    #[test]
    fn problem_directories_map_to_track_and_id() {
        assert_eq!(
            track_and_id(Path::new("/r/problems/foundations/stable-softmax")).expect("names"),
            ("foundations".to_owned(), "stable-softmax".to_owned())
        );
        assert!(track_and_id(Path::new("/r/problems/my track/x")).is_err());
        assert!(track_and_id(Path::new("/")).is_err());
    }

    #[test]
    fn a_failure_reads_as_one_paragraph() {
        assert_eq!(
            StepFailure::new("uv is missing.", Some("Install it.".into())).text(),
            "uv is missing. Install it."
        );
        assert_eq!(StepFailure::new("Broken.", None).text(), "Broken.");
    }
}
