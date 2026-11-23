//! Locating uv and building its command lines.
//!
//! Every invocation passes the full set of enabled extras (optional frameworks, and
//! `models` once it has been installed), so the result never depends on uv's
//! exact/inexact defaults (see the uv notes in `docs/VERSIONS.md`) and no command
//! removes what another one installed.
//!
//! An installed folder (see `root.rs`) carries its own uv in `<root>/bin`. Commands made
//! with it keep everything they download inside the folder, so an install neither uses
//! nor changes anything else on the machine and is removed by deleting the folder.

use std::ffi::OsString;
use std::path::{Path, PathBuf};

use crate::harness::CommandSpec;
use crate::model::Extra;

/// Environment variable that overrides the uv executable.
pub const UV_ENV: &str = "LEARNAI_UV";

/// File name of the uv executable.
const UV_FILE: &str = if cfg!(windows) { "uv.exe" } else { "uv" };

/// A uv executable bound to the `python/` project.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Uv {
    program: PathBuf,
    project: PathBuf,
    envs: Vec<(OsString, OsString)>,
}

impl Uv {
    /// Binds `program` to the uv project at `project`.
    pub fn new(program: impl Into<PathBuf>, project: impl Into<PathBuf>) -> Self {
        Self {
            program: program.into(),
            project: project.into(),
            envs: Vec::new(),
        }
    }

    /// Finds uv and binds it to `<root>/python`.
    ///
    /// The uv of an installed folder, `<root>/bin/uv`, wins over any other on the
    /// machine and runs with [`private_envs`]. `LEARNAI_UV` still overrides it.
    pub fn locate(root: &Path) -> Self {
        let named = std::env::var_os(UV_ENV).is_some_and(|v| !v.is_empty());
        let private = root.join("bin").join(UV_FILE);
        if !named && private.is_file() {
            return Self {
                program: private,
                project: root.join("python"),
                envs: private_envs(root),
            };
        }
        Self::new(find_uv(), root.join("python"))
    }

    /// The environment variables every command carries.
    pub fn envs(&self) -> &[(OsString, OsString)] {
        &self.envs
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
            envs: self.envs.clone(),
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
            envs: self.envs.clone(),
            ..CommandSpec::default()
        }
    }
}

/// The environment of the uv that belongs to an installed folder.
///
/// It keeps uv's cache and the Pythons uv downloads under `<root>/uv`, uses no Python
/// of the system, and stops uv from announcing its Python to the rest of the machine (a
/// link in a `bin` directory, an entry in the Windows registry). What Hugging Face and
/// PyTorch cache goes under `<root>/.learnai/cache`. The installer runs its first
/// `uv sync` with the same uv variables (`install.sh`, `install.ps1`); keep the three
/// in step.
pub fn private_envs(root: &Path) -> Vec<(OsString, OsString)> {
    let uv = root.join("uv");
    let cache = root.join(".learnai").join("cache");
    vec![
        ("UV_CACHE_DIR".into(), uv.join("cache").into()),
        ("UV_PYTHON_INSTALL_DIR".into(), uv.join("python").into()),
        ("UV_MANAGED_PYTHON".into(), "1".into()),
        ("UV_PYTHON_INSTALL_BIN".into(), "0".into()),
        ("UV_PYTHON_INSTALL_REGISTRY".into(), "0".into()),
        ("HF_HOME".into(), cache.join("huggingface").into()),
        ("TORCH_HOME".into(), cache.join("torch").into()),
    ]
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
/// minimal `PATH` that contains neither `~/.local/bin` nor Homebrew. On Windows the
/// file is `uv.exe` and the home directory is `USERPROFILE`.
pub fn find_uv() -> PathBuf {
    if let Some(path) = std::env::var_os(UV_ENV).filter(|v| !v.is_empty()) {
        return PathBuf::from(path);
    }
    let on_path = std::env::var_os("PATH")
        .map(|path| std::env::split_paths(&path).collect::<Vec<_>>())
        .unwrap_or_default();
    let home = std::env::var_os("HOME")
        .or_else(|| std::env::var_os("USERPROFILE"))
        .map(PathBuf::from);
    let fallbacks = [
        home.as_ref().map(|h| h.join(".local/bin")),
        home.as_ref().map(|h| h.join(".cargo/bin")),
        Some(PathBuf::from("/opt/homebrew/bin")),
        Some(PathBuf::from("/usr/local/bin")),
    ];
    on_path
        .into_iter()
        .chain(fallbacks.into_iter().flatten())
        .map(|dir| dir.join(UV_FILE))
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

    #[test]
    fn the_uv_of_an_installed_folder_is_used_and_keeps_its_files_inside_it() {
        let root = tempfile::tempdir().expect("a temporary directory");
        let root = root.path();
        assert!(Uv::locate(root).envs().is_empty());

        let private = root.join("bin").join(UV_FILE);
        std::fs::create_dir_all(root.join("bin")).expect("create bin");
        std::fs::write(&private, "").expect("write uv");
        let uv = Uv::locate(root);
        // `LEARNAI_UV` names another uv on purpose; then this one is not chosen.
        if std::env::var_os(UV_ENV).is_some_and(|v| !v.is_empty()) {
            return;
        }
        assert_eq!(uv.program(), private);
        for spec in [uv.sync(&[]), uv.run(&[]), uv.run_no_sync(), uv.version()] {
            let value = |name: &str| {
                spec.envs
                    .iter()
                    .find(|(key, _)| key == name)
                    .map(|(_, value)| PathBuf::from(value))
            };
            for name in ["UV_CACHE_DIR", "UV_PYTHON_INSTALL_DIR", "HF_HOME", "TORCH_HOME"] {
                let path = value(name).unwrap_or_else(|| panic!("{name} is set"));
                assert!(path.starts_with(root), "{name} = {}", path.display());
            }
            assert_eq!(value("UV_MANAGED_PYTHON"), Some(PathBuf::from("1")));
            assert_eq!(value("UV_PYTHON_INSTALL_BIN"), Some(PathBuf::from("0")));
            assert_eq!(value("UV_PYTHON_INSTALL_REGISTRY"), Some(PathBuf::from("0")));
        }
    }
}
