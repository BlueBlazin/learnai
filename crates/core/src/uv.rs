//! Locating uv and building its command lines.
//!
//! Every invocation passes the full set of enabled extras (optional frameworks, and
//! `models` once it has been installed), so the result never depends on uv's
//! exact/inexact defaults (see the uv notes in `docs/VERSIONS.md`) and no command
//! removes what another one installed.

use std::ffi::OsString;
use std::path::{Path, PathBuf};

use crate::harness::CommandSpec;
use crate::model::Extra;

/// Environment variable that overrides the uv executable.
pub const UV_ENV: &str = "LEARNAI_UV";

/// A uv executable bound to the `python/` project.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Uv {
    program: PathBuf,
    project: PathBuf,
}

impl Uv {
    /// Binds `program` to the uv project at `project`.
    pub fn new(program: impl Into<PathBuf>, project: impl Into<PathBuf>) -> Self {
        Self {
            program: program.into(),
            project: project.into(),
        }
    }

    /// Finds uv and binds it to `<root>/python`.
    pub fn locate(root: &Path) -> Self {
        Self::new(find_uv(), root.join("python"))
    }

    /// The uv executable.
    pub fn program(&self) -> &Path {
        &self.program
    }

    /// The project's virtual environment, where uv creates it.
    pub fn venv(&self) -> PathBuf {
        self.project.join(".venv")
    }

    fn spec(&self, subcommand: &str, extras: &[Extra], tail: &[&str]) -> CommandSpec {
        let mut args: Vec<OsString> = vec![
            subcommand.into(),
            "--project".into(),
            self.project.clone().into(),
        ];
        for extra in extra_names(extras) {
            args.push("--extra".into());
            args.push(extra.into());
        }
        args.extend(tail.iter().map(OsString::from));
        CommandSpec {
            program: self.program.clone().into(),
            args,
            ..CommandSpec::default()
        }
    }

    /// `uv run --project <project> [--extra ...] --`; append the command to run.
    pub fn run(&self, extras: &[Extra]) -> CommandSpec {
        self.spec("run", extras, &["--"])
    }

    /// `uv run --project <project> --no-sync --`: runs in the environment as it is,
    /// without installing or removing anything.
    pub fn run_no_sync(&self) -> CommandSpec {
        self.spec("run", &[], &["--no-sync", "--"])
    }

    /// `uv sync --project <project> [--extra ...]` with the full enabled set.
    pub fn sync(&self, extras: &[Extra]) -> CommandSpec {
        self.spec("sync", extras, &[])
    }

    /// `uv --version`.
    pub fn version(&self) -> CommandSpec {
        CommandSpec {
            program: self.program.clone().into(),
            args: vec!["--version".into()],
            ..CommandSpec::default()
        }
    }
}

/// The path of an executable inside a virtual environment.
pub fn venv_program(venv: &Path, name: &str) -> PathBuf {
    if cfg!(windows) {
        venv.join("Scripts").join(format!("{name}.exe"))
    } else {
        venv.join("bin").join(name)
    }
}

/// The uv extra names for a set of extras: sorted and deduplicated.
pub fn extra_names(extras: &[Extra]) -> Vec<&'static str> {
    let mut names: Vec<_> = extras.iter().map(|extra| extra.as_str()).collect();
    names.sort_unstable();
    names.dedup();
    names
}

/// Finds the uv executable.
///
/// Order: `LEARNAI_UV`, then `PATH`, then the usual install locations. The fallback
/// locations matter for the desktop app: a macOS app launched from Finder inherits a
/// minimal `PATH` that contains neither `~/.local/bin` nor Homebrew.
pub fn find_uv() -> PathBuf {
    if let Some(path) = std::env::var_os(UV_ENV).filter(|v| !v.is_empty()) {
        return PathBuf::from(path);
    }
    let on_path = std::env::var_os("PATH")
        .map(|path| std::env::split_paths(&path).collect::<Vec<_>>())
        .unwrap_or_default();
    let home = std::env::var_os("HOME").map(PathBuf::from);
    let fallbacks = [
        home.as_ref().map(|h| h.join(".local/bin")),
        home.as_ref().map(|h| h.join(".cargo/bin")),
        Some(PathBuf::from("/opt/homebrew/bin")),
        Some(PathBuf::from("/usr/local/bin")),
    ];
    on_path
        .into_iter()
        .chain(fallbacks.into_iter().flatten())
        .map(|dir| dir.join("uv"))
        .find(|candidate| candidate.is_file())
        .unwrap_or_else(|| PathBuf::from("uv"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn commands_carry_the_full_extra_set_in_a_stable_order() {
        let uv = Uv::new("uv", "/r/python");
        let extras = [Extra::Mlx, Extra::Jax, Extra::Mlx];
        assert_eq!(
            uv.sync(&extras).display(),
            "uv sync --project /r/python --extra jax --extra mlx"
        );
        assert_eq!(
            uv.run(&extras).display(),
            "uv run --project /r/python --extra jax --extra mlx --"
        );
        assert_eq!(uv.sync(&[]).display(), "uv sync --project /r/python");
        assert_eq!(
            uv.run(&[Extra::Models, Extra::Jax]).display(),
            "uv run --project /r/python --extra jax --extra models --"
        );
        assert_eq!(
            uv.run_no_sync().display(),
            "uv run --project /r/python --no-sync --"
        );
    }
}
