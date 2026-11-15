//! Run targets: where a harness process executes.
//!
//! The run pipeline only knows the [`Target`] and [`Process`] traits. `local` spawns the
//! harness on this machine. The SSH target ([`crate::remote::SshTarget`]) implements the
//! same two traits: it uploads the problem and solution in `launch`, reports progress,
//! and returns a process whose pipes carry the remote harness output.

use std::io::{self, Read};
use std::process::{Child, ChildStdin, Command, Stdio};
use std::sync::Arc;
use std::sync::atomic::AtomicBool;
use std::thread;
use std::time::{Duration, Instant};

use crate::harness::{CommandSpec, HarnessCommand, HarnessJob};

/// Name of the built-in local target.
pub const LOCAL_TARGET: &str = "local";

/// A place harness jobs can run.
pub trait Target: Send + Sync {
    /// Stable name, stored with submissions and used to select the target.
    fn name(&self) -> &str;

    /// Starts the harness for `job`. `progress` receives human-readable preparation
    /// steps (shown as `preparing` events). A target whose preparation takes time (a
    /// remote sync) watches `canceled` and gives up early when it becomes true.
    fn launch(
        &self,
        job: &HarnessJob,
        progress: &dyn Fn(String),
        canceled: &AtomicBool,
    ) -> io::Result<Box<dyn Process>>;
}

/// A launch failure whose text is already written for the user (what went wrong and
/// what to do). The run reports it verbatim instead of wrapping it.
#[derive(Debug)]
pub struct LaunchFailure(pub String);

impl std::fmt::Display for LaunchFailure {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for LaunchFailure {}

impl LaunchFailure {
    /// Wraps the message in an [`io::Error`], as [`Target::launch`] returns.
    pub fn into_io(message: impl Into<String>) -> io::Error {
        io::Error::other(LaunchFailure(message.into()))
    }

    /// The user-facing message carried by `error`, if it is one of these.
    pub fn message_of(error: &io::Error) -> Option<&str> {
        error
            .get_ref()
            .and_then(|inner| inner.downcast_ref::<LaunchFailure>())
            .map(|failure| failure.0.as_str())
    }
}

/// A running harness.
pub trait Process: Send {
    /// Takes the stdout pipe. Returns `None` after the first call.
    fn take_stdout(&mut self) -> Option<Box<dyn Read + Send>>;

    /// Takes the stderr pipe. Returns `None` after the first call.
    fn take_stderr(&mut self) -> Option<Box<dyn Read + Send>>;

    /// Kills the process and everything it spawned. Idempotent; never fails.
    fn kill(&mut self);

    /// Waits up to `grace` for the process to exit by itself, then kills it. Either way,
    /// anything it left running is killed too. Returns the exit code, or `None` when the
    /// process was killed by a signal.
    fn finish(&mut self, grace: Duration) -> Option<i32>;

    /// What to tell the user when the process ended with `code` before the harness
    /// reported a result, when the target knows better than "exited with code N" (a
    /// dropped connection, say).
    fn exit_note(&self, _code: Option<i32>) -> Option<String> {
        None
    }
}

/// Runs harness jobs on this machine.
pub struct LocalTarget {
    harness: Arc<dyn HarnessCommand>,
}

impl LocalTarget {
    /// Creates the local target with the given harness command builder.
    pub fn new(harness: Arc<dyn HarnessCommand>) -> Self {
        Self { harness }
    }
}

impl Target for LocalTarget {
    fn name(&self) -> &str {
        LOCAL_TARGET
    }

    fn launch(
        &self,
        job: &HarnessJob,
        progress: &dyn Fn(String),
        _canceled: &AtomicBool,
    ) -> io::Result<Box<dyn Process>> {
        let spec = self.harness.command(job);
        progress("Syncing the Python environment and starting the judge".to_owned());
        Ok(Box::new(LocalProcess::spawn(&spec)?))
    }
}

/// A local child process that leads its own process group.
pub struct LocalProcess {
    child: Child,
    stdin: Option<ChildStdin>,
    reaped: bool,
}

impl LocalProcess {
    /// Spawns `spec` with piped stdout and stderr, no stdin, in a new process group so
    /// the whole tree (uv, Python, anything the solution starts) can be killed at once.
    pub fn spawn(spec: &CommandSpec) -> io::Result<Self> {
        Self::spawn_io(spec, Stdio::null(), Stdio::piped(), Stdio::piped())
    }

    /// Like [`LocalProcess::spawn`], with the three standard streams chosen by the
    /// caller. A piped stdin is kept open until [`LocalProcess::close_stdin`], a kill,
    /// or drop.
    pub fn spawn_io(
        spec: &CommandSpec,
        stdin: Stdio,
        stdout: Stdio,
        stderr: Stdio,
    ) -> io::Result<Self> {
        let mut command = Command::new(&spec.program);
        command
            .args(&spec.args)
            .envs(spec.envs.iter().map(|(key, value)| (key, value)))
            .stdin(stdin)
            .stdout(stdout)
            .stderr(stderr);
        if let Some(cwd) = &spec.cwd {
            command.current_dir(cwd);
        }
        #[cfg(unix)]
        {
            use std::os::unix::process::CommandExt;
            command.process_group(0);
        }
        let mut child = command.spawn().map_err(|e| {
            io::Error::new(
                e.kind(),
                format!("failed to start `{}`: {e}", spec.program.to_string_lossy()),
            )
        })?;
        Ok(Self {
            stdin: child.stdin.take(),
            child,
            reaped: false,
        })
    }

    /// Closes the child's standard input, if it was piped.
    pub fn close_stdin(&mut self) {
        self.stdin = None;
    }

    /// Waits up to `limit` for the child to exit by itself, without reaping it. Returns
    /// whether it has exited.
    pub fn wait_exit(&mut self, limit: Duration) -> bool {
        let deadline = Instant::now() + limit;
        loop {
            if self.has_exited() {
                return true;
            }
            if Instant::now() >= deadline {
                return false;
            }
            thread::sleep(Duration::from_millis(5));
        }
    }
}

impl Process for LocalProcess {
    fn take_stdout(&mut self) -> Option<Box<dyn Read + Send>> {
        self.child
            .stdout
            .take()
            .map(|pipe| Box::new(pipe) as Box<dyn Read + Send>)
    }

    fn take_stderr(&mut self) -> Option<Box<dyn Read + Send>> {
        self.child
            .stderr
            .take()
            .map(|pipe| Box::new(pipe) as Box<dyn Read + Send>)
    }

    fn kill(&mut self) {
        if !self.reaped {
            self.kill_group();
        }
    }

    fn finish(&mut self, grace: Duration) -> Option<i32> {
        self.wait_exit(grace);
        // Kill the group while the leader is still unreaped, so the group id cannot have
        // been recycled. If the leader already exited this only sweeps what it left behind.
        self.kill();
        self.stdin = None;
        let status = self.child.wait();
        self.reaped = true;
        status.ok().and_then(|status| status.code())
    }
}

impl LocalProcess {
    /// Whether the child has exited, without reaping it.
    #[cfg(unix)]
    fn has_exited(&mut self) -> bool {
        if self.reaped {
            return true;
        }
        // SAFETY: `siginfo_t` is plain data for which all-zero bytes are a valid value.
        let mut info: libc::siginfo_t = unsafe { std::mem::zeroed() };
        // SAFETY: `info` is a valid, writable `siginfo_t`. `WNOWAIT` leaves the child
        // waitable, so the later `Child::wait` still observes its status.
        let rc = unsafe {
            libc::waitid(
                libc::P_PID,
                self.child.id() as libc::id_t,
                &mut info,
                libc::WEXITED | libc::WNOHANG | libc::WNOWAIT,
            )
        };
        // SAFETY: after a successful `waitid`, `si_pid` is initialized: it is the child's
        // pid when a status is available and stays zero (from the zeroed struct) otherwise.
        rc != 0 || unsafe { info.si_pid() } != 0
    }

    #[cfg(not(unix))]
    fn has_exited(&mut self) -> bool {
        self.reaped || !matches!(self.child.try_wait(), Ok(None))
    }

    #[cfg(unix)]
    fn kill_group(&mut self) {
        if let Ok(pgid) = libc::pid_t::try_from(self.child.id()) {
            // SAFETY: `killpg` has no memory-safety preconditions. The child leads its own
            // process group and has not been reaped (callers check `reaped`), so the
            // group id still refers to this group and nothing else.
            unsafe {
                libc::killpg(pgid, libc::SIGKILL);
            }
        }
    }

    #[cfg(not(unix))]
    fn kill_group(&mut self) {
        let _ = self.child.kill();
    }
}

impl Drop for LocalProcess {
    fn drop(&mut self) {
        if !self.reaped {
            self.finish(Duration::ZERO);
        }
    }
}
