//! Building the harness command line (`docs/ARCHITECTURE.md` section 3.5).
//!
//! The command is injectable through [`HarnessCommand`], so tests and the devbridge can
//! substitute a fake harness for `uv run ... python -m whetstone.harness`.

use std::ffi::OsString;
use std::path::PathBuf;

use crate::model::{Extra, Framework, RunMode};
use crate::uv::Uv;

/// Everything needed to judge one solution once.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HarnessJob {
    /// Absolute problem directory.
    pub problem_dir: PathBuf,
    /// Absolute solution file.
    pub solution: PathBuf,
    /// Framework the solution is written in.
    pub framework: Framework,
    /// Public cases only, or all cases.
    pub mode: RunMode,
    /// Restrict the run to one case.
    pub case: Option<String>,
    /// Every installed extra; passed as `--extra` to every `uv run`.
    pub extras: Vec<Extra>,
    /// Whether `solution` is a review attempt (`workspace/<id>/review/<framework>.py`).
    /// The harness judges it like any other file; a remote target keeps it apart from
    /// the saved solution.
    pub review: bool,
}

impl HarnessJob {
    /// The arguments understood by `python -m whetstone.harness`.
    pub fn harness_args(&self) -> Vec<OsString> {
        let mut args: Vec<OsString> = vec![
            "--problem".into(),
            self.problem_dir.clone().into(),
            "--solution".into(),
            self.solution.clone().into(),
            "--framework".into(),
            self.framework.as_str().into(),
            "--mode".into(),
            self.mode.as_str().into(),
        ];
        if let Some(case) = &self.case {
            args.push("--case".into());
            args.push(case.into());
        }
        args
    }
}

/// A fully specified command, independent of how it is spawned.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct CommandSpec {
    /// Program to execute.
    pub program: OsString,
    /// Arguments.
    pub args: Vec<OsString>,
    /// Extra environment variables.
    pub envs: Vec<(OsString, OsString)>,
    /// Working directory; inherited when `None`.
    pub cwd: Option<PathBuf>,
}

impl CommandSpec {
    /// Renders the command for display in logs (no shell quoting).
    pub fn display(&self) -> String {
        std::iter::once(&self.program)
            .chain(&self.args)
            .map(|part| part.to_string_lossy())
            .collect::<Vec<_>>()
            .join(" ")
    }
}

/// Builds the command that runs the harness for a job.
pub trait HarnessCommand: Send + Sync {
    /// Returns the command to spawn.
    fn command(&self, job: &HarnessJob) -> CommandSpec;
}

/// The real harness:
/// `uv run --project <root>/python [--extra ...] -- python -m whetstone.harness <args>`.
#[derive(Debug, Clone)]
pub struct UvHarness {
    uv: Uv,
}

impl UvHarness {
    /// Creates the builder for a uv installation and project.
    pub fn new(uv: Uv) -> Self {
        Self { uv }
    }
}

impl HarnessCommand for UvHarness {
    fn command(&self, job: &HarnessJob) -> CommandSpec {
        let mut spec = self.uv.run(&job.extras);
        spec.args
            .extend(["python".into(), "-m".into(), "whetstone.harness".into()]);
        spec.args.extend(job.harness_args());
        spec
    }
}

/// A stand-in harness: `<program> <leading args> <harness args>`.
///
/// The program must accept the harness arguments and speak the event protocol. Used by
/// tests and by the devbridge `--harness` flag.
#[derive(Debug, Clone, Default)]
pub struct ProgramHarness {
    /// Program to execute.
    pub program: OsString,
    /// Arguments placed before the harness arguments.
    pub args: Vec<OsString>,
    /// Extra environment variables.
    pub envs: Vec<(OsString, OsString)>,
}

impl ProgramHarness {
    /// Creates a stand-in that runs `program` with only the harness arguments.
    pub fn new(program: impl Into<OsString>) -> Self {
        Self {
            program: program.into(),
            ..Self::default()
        }
    }

    /// Adds an argument before the harness arguments.
    pub fn arg(mut self, arg: impl Into<OsString>) -> Self {
        self.args.push(arg.into());
        self
    }

    /// Adds an environment variable.
    pub fn env(mut self, key: impl Into<OsString>, value: impl Into<OsString>) -> Self {
        self.envs.push((key.into(), value.into()));
        self
    }
}

impl HarnessCommand for ProgramHarness {
    fn command(&self, job: &HarnessJob) -> CommandSpec {
        let mut args = self.args.clone();
        args.extend(job.harness_args());
        CommandSpec {
            program: self.program.clone(),
            args,
            envs: self.envs.clone(),
            cwd: None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn job(extras: Vec<Extra>) -> HarnessJob {
        HarnessJob {
            problem_dir: "/r/problems/t/p".into(),
            solution: "/r/workspace/p/torch.py".into(),
            framework: Framework::Torch,
            mode: RunMode::Submit,
            case: None,
            extras,
            review: false,
        }
    }

    #[test]
    fn uv_harness_matches_the_contract() {
        let harness = UvHarness::new(Uv::new("uv", "/r/python"));
        let spec = harness.command(&job(vec![Extra::Jax, Extra::Mlx]));
        assert_eq!(
            spec.display(),
            "uv run --project /r/python --extra jax --extra mlx -- python -m whetstone.harness \
             --problem /r/problems/t/p --solution /r/workspace/p/torch.py --framework torch --mode submit"
        );
        let bare = harness.command(&HarnessJob {
            case: Some("c2".into()),
            mode: RunMode::Run,
            ..job(vec![])
        });
        assert_eq!(
            bare.display(),
            "uv run --project /r/python -- python -m whetstone.harness \
             --problem /r/problems/t/p --solution /r/workspace/p/torch.py --framework torch --mode run --case c2"
        );
    }

    #[test]
    fn program_harness_prepends_its_arguments() {
        let spec = ProgramHarness::new("sh")
            .arg("fake.sh")
            .env("K", "V")
            .command(&job(vec![]));
        assert!(
            spec.display()
                .starts_with("sh fake.sh --problem /r/problems/t/p")
        );
        assert_eq!(spec.envs, vec![("K".into(), "V".into())]);
    }
}
