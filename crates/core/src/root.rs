//! Content root resolution.
//!
//! The content root is the directory holding `python/`, `problems/`, and `workspace/`.
//! Resolution order: an explicit override (for example the devbridge `--root` flag), the
//! `LEARNAI_ROOT` environment variable, the saved setting, then where this build keeps
//! its content. A build from a checkout keeps it in the repository it was compiled in.
//! A build made for the installer (compiled with `LEARNAI_DIST` set) was compiled on
//! another machine, so it keeps it in the installed folder, `~/.learnai`.
//!
//! The saved setting cannot live in the state database (which sits under the root), so it
//! is a small JSON file, `root.json`, in a configuration directory supplied by the shell.

use std::fs;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::error::{CoreError, Result};

/// Environment variable that overrides the content root.
pub const ROOT_ENV: &str = "LEARNAI_ROOT";

/// File name of the saved root setting inside the configuration directory.
pub const ROOT_SETTING_FILE: &str = "root.json";

/// Name of the installed folder inside the home directory.
pub const INSTALL_DIR: &str = ".learnai";

/// Whether this build was made for the installer.
const DIST: bool = option_env!("LEARNAI_DIST").is_some();

/// Where a resolved root came from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RootSource {
    /// Passed explicitly by the caller.
    Explicit,
    /// The `LEARNAI_ROOT` environment variable.
    Env,
    /// The saved setting.
    Saved,
    /// The installed folder.
    Installed,
    /// The repository path captured at compile time.
    Compiled,
}

/// A resolved content root.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResolvedRoot {
    /// Absolute path of the root.
    pub path: PathBuf,
    /// Which source supplied it.
    pub source: RootSource,
}

/// Inputs to [`resolve_root`]. `Default` reads the real environment and compile-time path.
#[derive(Debug, Clone)]
pub struct RootInputs {
    /// Explicit override; wins over everything.
    pub explicit: Option<PathBuf>,
    /// Value of `LEARNAI_ROOT`.
    pub env: Option<PathBuf>,
    /// Directory holding the saved setting file, if the shell has one.
    pub config_dir: Option<PathBuf>,
    /// The installed folder; only a build made for the installer has one.
    pub installed: Option<PathBuf>,
    /// Repository path captured at compile time; only a build from a checkout has one.
    pub compiled: Option<PathBuf>,
}

impl Default for RootInputs {
    fn default() -> Self {
        Self {
            explicit: None,
            env: std::env::var_os(ROOT_ENV)
                .filter(|v| !v.is_empty())
                .map(PathBuf::from),
            config_dir: None,
            installed: if DIST { installed_root() } else { None },
            compiled: (!DIST).then(compiled_root),
        }
    }
}

/// The folder the installer fills: `.learnai` in the home directory (`HOME`, or
/// `USERPROFILE` on Windows).
pub fn installed_root() -> Option<PathBuf> {
    std::env::var_os("HOME")
        .filter(|v| !v.is_empty())
        .or_else(|| std::env::var_os("USERPROFILE").filter(|v| !v.is_empty()))
        .map(|home| PathBuf::from(home).join(INSTALL_DIR))
}

/// The repository root captured when this crate was compiled.
pub fn compiled_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("..").join("..")
}

#[derive(Serialize, Deserialize)]
struct SavedRoot {
    root: PathBuf,
}

/// Reads the saved root setting from `config_dir`, if present and well formed.
pub fn read_saved_root(config_dir: &Path) -> Option<PathBuf> {
    let text = fs::read_to_string(config_dir.join(ROOT_SETTING_FILE)).ok()?;
    serde_json::from_str::<SavedRoot>(&text)
        .ok()
        .map(|saved| saved.root)
}

/// Saves `root` as the root setting in `config_dir`.
pub fn save_root_setting(config_dir: &Path, root: &Path) -> Result<()> {
    fs::create_dir_all(config_dir).map_err(|e| CoreError::io("create", config_dir, e))?;
    let path = config_dir.join(ROOT_SETTING_FILE);
    let text = serde_json::to_string_pretty(&SavedRoot {
        root: root.to_path_buf(),
    })?;
    fs::write(&path, text).map_err(|e| CoreError::io("write", &path, e))
}

/// Resolves the content root.
///
/// An explicit path or environment value that is not a directory is an error rather than
/// a silent fall-through, because the user asked for it by name. A stale saved setting
/// falls through to the installed folder or the compile-time path.
pub fn resolve_root(inputs: &RootInputs) -> Result<ResolvedRoot> {
    if let Some(path) = &inputs.explicit {
        return named_root(path, RootSource::Explicit, "the explicit root");
    }
    if let Some(path) = &inputs.env {
        return named_root(path, RootSource::Env, ROOT_ENV);
    }
    let saved = inputs.config_dir.as_deref().and_then(read_saved_root);
    if let Some(path) = saved.and_then(|p| canonical_dir(&p)) {
        return Ok(ResolvedRoot {
            path,
            source: RootSource::Saved,
        });
    }
    if let Some(path) = inputs.installed.as_deref().and_then(canonical_dir) {
        return Ok(ResolvedRoot {
            path,
            source: RootSource::Installed,
        });
    }
    if let Some(path) = inputs.compiled.as_deref().and_then(canonical_dir) {
        return Ok(ResolvedRoot {
            path,
            source: RootSource::Compiled,
        });
    }
    if let Some(path) = &inputs.installed {
        return Err(CoreError::Root(format!(
            "Learn AI is not installed in `{}`: run the installer again, or set \
             {ROOT_ENV} to the directory that holds python/, problems/, and workspace/",
            path.display()
        )));
    }
    Err(CoreError::Root(format!(
        "set {ROOT_ENV} to the directory that holds python/, problems/, and workspace/"
    )))
}

fn named_root(path: &Path, source: RootSource, what: &str) -> Result<ResolvedRoot> {
    canonical_dir(path)
        .map(|path| ResolvedRoot { path, source })
        .ok_or_else(|| CoreError::Root(format!("{what} `{}` is not a directory", path.display())))
}

fn canonical_dir(path: &Path) -> Option<PathBuf> {
    let path = fs::canonicalize(path).ok()?;
    path.is_dir().then_some(path)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn inputs() -> RootInputs {
        RootInputs {
            explicit: None,
            env: None,
            config_dir: None,
            installed: None,
            compiled: None,
        }
    }

    #[test]
    fn precedence_is_explicit_env_saved_installed_compiled() -> Result<()> {
        let tmp = tempfile::tempdir().map_err(|e| CoreError::io("create", Path::new("tmp"), e))?;
        let mk = |name: &str| -> Result<PathBuf> {
            let dir = tmp.path().join(name);
            fs::create_dir_all(&dir).map_err(|e| CoreError::io("create", &dir, e))?;
            fs::canonicalize(&dir).map_err(|e| CoreError::io("resolve", &dir, e))
        };
        let (explicit, env, saved, compiled) =
            (mk("explicit")?, mk("env")?, mk("saved")?, mk("compiled")?);
        let installed = mk("installed")?;
        let config = mk("config")?;
        save_root_setting(&config, &saved)?;

        let mut all = RootInputs {
            explicit: Some(explicit.clone()),
            env: Some(env.clone()),
            config_dir: Some(config.clone()),
            installed: Some(installed.clone()),
            compiled: Some(compiled.clone()),
        };
        assert_eq!(
            resolve_root(&all)?,
            ResolvedRoot {
                path: explicit,
                source: RootSource::Explicit
            }
        );
        all.explicit = None;
        assert_eq!(
            resolve_root(&all)?,
            ResolvedRoot {
                path: env,
                source: RootSource::Env
            }
        );
        all.env = None;
        assert_eq!(
            resolve_root(&all)?,
            ResolvedRoot {
                path: saved,
                source: RootSource::Saved
            }
        );

        // A stale saved setting falls through to the installed folder, then to the
        // compiled path.
        save_root_setting(&config, &tmp.path().join("gone"))?;
        assert_eq!(
            resolve_root(&all)?,
            ResolvedRoot {
                path: installed,
                source: RootSource::Installed
            }
        );
        all.installed = None;
        assert_eq!(
            resolve_root(&all)?,
            ResolvedRoot {
                path: compiled,
                source: RootSource::Compiled
            }
        );
        Ok(())
    }

    #[test]
    fn a_missing_installed_folder_is_named_in_the_error() {
        let missing = RootInputs {
            installed: Some(PathBuf::from("/definitely/not/here/.learnai")),
            ..inputs()
        };
        match resolve_root(&missing) {
            Err(CoreError::Root(message)) => {
                assert!(message.contains("run the installer again"), "{message}");
                assert!(message.contains("/definitely/not/here/.learnai"), "{message}");
            }
            other => panic!("expected a root error, got {other:?}"),
        }
    }

    #[test]
    fn a_bad_env_root_is_an_error_not_a_fall_through() {
        let bad = RootInputs {
            env: Some(PathBuf::from("/definitely/not/here")),
            compiled: Some(compiled_root()),
            ..inputs()
        };
        assert!(matches!(resolve_root(&bad), Err(CoreError::Root(_))));
        assert!(matches!(resolve_root(&inputs()), Err(CoreError::Root(_))));
    }

    #[test]
    fn compiled_root_is_the_repository() -> Result<()> {
        let root = resolve_root(&RootInputs {
            env: None,
            ..RootInputs::default()
        })?;
        assert!(root.path.join("crates").join("core").is_dir());
        Ok(())
    }
}
