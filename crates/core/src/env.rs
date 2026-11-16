//! The Python environment: status probing and framework installation.
//!
//! The set of enabled extras lives in the settings table: the optional frameworks the
//! user installed, and `models` once a model download has installed it. It is passed in
//! full to every `uv sync` and `uv run` core makes (runs, installs, asset listing and
//! fetching), because a `uv sync` with a partial set would silently remove the rest.

use std::collections::BTreeMap;
use std::process::{Command, Stdio};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc;
use std::thread;
use std::time::Duration;

use serde::Deserialize;

use crate::error::{CoreError, Result};
use crate::harness::CommandSpec;
use crate::model::{EnvStatus, Extra, Framework, FrameworkEnv, InstallEvent};
use crate::sink::SharedSink;
use crate::store::Store;
use crate::stream::for_each_line;
use crate::target::{LocalProcess, Process};
use crate::uv::Uv;

/// Settings key holding the enabled extras as a JSON array of extra names.
const EXTRAS_SETTING: &str = "enabled_extras";

/// Prints the Python version and installed framework versions as one JSON object.
/// Uses distribution metadata, so nothing heavy is imported. `models` reports the
/// package that marks the `models` extra (`huggingface_hub`).
const PROBE_SCRIPT: &str = "\
import json, platform
from importlib import metadata

def version(name):
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None

print(json.dumps({'python': platform.python_version(), 'torch': version('torch'), \
'jax': version('jax'), 'mlx': version('mlx'), 'models': version('huggingface_hub')}))
";

#[derive(Debug, Deserialize)]
struct Probe {
    python: String,
    #[serde(flatten)]
    packages: BTreeMap<String, Option<String>>,
}

impl Probe {
    fn version(&self, package: &str) -> Option<&str> {
        self.packages.get(package)?.as_deref()
    }

    /// The extras whose packages are in the environment.
    fn installed_extras(&self) -> Vec<Extra> {
        Extra::ALL
            .iter()
            .copied()
            .filter(|extra| self.version(extra.as_str()).is_some())
            .collect()
    }
}

/// Access to the uv-managed Python environment of a content root.
#[derive(Clone)]
pub struct Environment {
    uv: Uv,
    store: Arc<Store>,
    installing: Arc<AtomicBool>,
}

impl Environment {
    /// Creates the environment accessor.
    pub fn new(uv: Uv, store: Arc<Store>) -> Self {
        Self {
            uv,
            store,
            installing: Arc::new(AtomicBool::new(false)),
        }
    }

    /// The uv binding used for every command.
    pub fn uv(&self) -> &Uv {
        &self.uv
    }

    /// Whether an install is running.
    pub fn is_installing(&self) -> bool {
        self.installing.load(Ordering::SeqCst)
    }

    /// The extras known to be installed, sorted. Names this build does not know (from
    /// a newer version of the app) are ignored.
    pub fn enabled_extras(&self) -> Result<Vec<Extra>> {
        let names: Vec<String> = self.store.setting(EXTRAS_SETTING)?.unwrap_or_default();
        let mut extras: Vec<Extra> = names.iter().filter_map(|name| name.parse().ok()).collect();
        extras.sort_unstable();
        extras.dedup();
        Ok(extras)
    }

    /// Records that `add` are installed, so every later uv command keeps them.
    pub(crate) fn enable_extras(&self, add: impl IntoIterator<Item = Extra>) -> Result<Vec<Extra>> {
        let current = self.enabled_extras()?;
        let mut extras = current.clone();
        extras.extend(add);
        extras.sort_unstable();
        extras.dedup();
        if extras != current {
            self.store.set_setting(EXTRAS_SETTING, &extras)?;
        }
        Ok(extras)
    }

    /// Probes uv and the project environment.
    ///
    /// Extras found installed are adopted into the enabled set, so extras installed
    /// outside the app (for example with `uv sync --extra mlx` or
    /// `uv run --extra models learnai assets fetch ...` in a terminal) are not removed by a
    /// later install.
    pub fn status(&self, root: &std::path::Path) -> Result<EnvStatus> {
        let uv = capture(&self.uv.version()).and_then(|out| parse_uv_version(&out));
        let probe = uv.as_ref().and_then(|_| self.probe());

        let frameworks: Vec<FrameworkEnv> = Framework::ALL
            .iter()
            .map(|&framework| {
                let version = probe.as_ref().and_then(|probe| match framework {
                    Framework::Python => Some(probe.python.clone()),
                    other => probe.version(other.as_str()).map(str::to_owned),
                });
                FrameworkEnv {
                    framework,
                    installed: version.is_some(),
                    version,
                }
            })
            .collect();
        if let Some(probe) = &probe {
            self.enable_extras(probe.installed_extras())?;
        }

        Ok(EnvStatus {
            uv,
            python: probe.map(|probe| probe.python),
            frameworks,
            root: root.to_string_lossy().into_owned(),
        })
    }

    /// Adopts whatever extras are installed right now and returns the enabled set: the
    /// extras a uv command must pass so that it removes nothing.
    pub(crate) fn current_extras(&self) -> Result<Vec<Extra>> {
        let installed = self
            .probe()
            .map(|probe| probe.installed_extras())
            .unwrap_or_default();
        self.enable_extras(installed)
    }

    /// Runs the probe script in the project environment without syncing it.
    fn probe(&self) -> Option<Probe> {
        let mut spec = self.uv.run_no_sync();
        spec.args
            .extend(["python".into(), "-c".into(), PROBE_SCRIPT.into()]);
        let output = capture(&spec)?;
        output
            .lines()
            .rev()
            .find_map(|line| serde_json::from_str(line.trim()).ok())
    }

    /// Installs `framework` by syncing the environment with the full enabled set plus
    /// that framework's extra. Returns once the install has started; progress and the
    /// final `done` event arrive through `sink`. Only one install runs at a time.
    pub fn install(&self, framework: Framework, sink: SharedSink<InstallEvent>) -> Result<()> {
        if self.installing.swap(true, Ordering::SeqCst) {
            return Err(CoreError::InstallInProgress);
        }
        let this = self.clone();
        let spawned = thread::Builder::new()
            .name("env-install".to_owned())
            .spawn(move || {
                let outcome = this.sync_with(framework, &sink);
                this.installing.store(false, Ordering::SeqCst);
                sink.emit(match outcome {
                    Ok(()) => InstallEvent::Done {
                        ok: true,
                        message: None,
                    },
                    Err(message) => InstallEvent::Done {
                        ok: false,
                        message: Some(message),
                    },
                });
            });
        spawned.map(drop).map_err(|source| {
            self.installing.store(false, Ordering::SeqCst);
            CoreError::Io {
                context: "failed to start the install thread".to_owned(),
                source,
            }
        })
    }

    fn sync_with(
        &self,
        framework: Framework,
        sink: &SharedSink<InstallEvent>,
    ) -> std::result::Result<(), String> {
        // Adopt anything already installed, so the sync below cannot strip it.
        let mut extras = self.current_extras().map_err(|e| e.to_string())?;
        extras.extend(framework.extra());

        let spec = self.uv.sync(&extras);
        sink.emit(InstallEvent::Log {
            text: format!("$ {}", spec.display()),
        });
        let mut process = LocalProcess::spawn(&spec).map_err(|e| e.to_string())?;

        let (tx, rx) = mpsc::channel();
        for pipe in [process.take_stdout(), process.take_stderr()]
            .into_iter()
            .flatten()
        {
            let tx = tx.clone();
            let reader = thread::Builder::new()
                .name("env-install-reader".to_owned())
                .spawn(move || for_each_line(pipe, |line| tx.send(line).is_ok()));
            reader.map_err(|e| e.to_string())?;
        }
        drop(tx);
        for text in rx {
            sink.emit(InstallEvent::Log { text });
        }

        match process.finish(Duration::from_secs(5)) {
            Some(0) => {
                // Remember the extra only once it is really installed.
                self.enable_extras(framework.extra())
                    .map_err(|e| e.to_string())?;
                Ok(())
            }
            Some(code) => Err(format!("uv sync exited with code {code}")),
            None => Err("uv sync was killed by a signal".to_owned()),
        }
    }
}

/// Runs a command to completion and returns its stdout if it succeeded.
fn capture(spec: &CommandSpec) -> Option<String> {
    let output = Command::new(&spec.program)
        .args(&spec.args)
        .envs(spec.envs.iter().map(|(key, value)| (key, value)))
        .stdin(Stdio::null())
        .stderr(Stdio::null())
        .output()
        .ok()?;
    output
        .status
        .success()
        .then(|| String::from_utf8_lossy(&output.stdout).into_owned())
}

/// Extracts `0.9.26` from `uv 0.9.26 (ee4f00362 2026-01-15)`.
fn parse_uv_version(output: &str) -> Option<String> {
    let mut words = output.split_whitespace();
    match (words.next(), words.next()) {
        (Some("uv"), Some(version)) => Some(version.to_owned()),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn uv_version_is_the_second_word() {
        assert_eq!(
            parse_uv_version("uv 0.9.26 (ee4f00362 2026-01-15)\n").as_deref(),
            Some("0.9.26")
        );
        assert_eq!(parse_uv_version("uv 0.13.8").as_deref(), Some("0.13.8"));
        assert_eq!(parse_uv_version("command not found"), None);
    }

    #[test]
    fn probe_output_parses() {
        let probe: Probe = serde_json::from_str(
            r#"{"python":"3.15.2","torch":"2.15.0","jax":null,"mlx":"0.32.4"}"#,
        )
        .expect("probe JSON");
        assert_eq!(probe.python, "3.15.2");
        assert_eq!(probe.packages["torch"].as_deref(), Some("2.15.0"));
        assert_eq!(probe.packages["jax"], None);
        assert_eq!(probe.installed_extras(), [Extra::Mlx]);

        // The `models` extra is recognized by its marker package.
        let probe: Probe = serde_json::from_str(
            r#"{"python":"3.15.2","torch":"2.15.0","jax":"0.12.0.1","mlx":null,"models":"2.4.0"}"#,
        )
        .expect("probe JSON");
        assert_eq!(probe.installed_extras(), [Extra::Jax, Extra::Models]);
    }

    #[test]
    fn enabled_extras_ignore_names_from_other_versions() -> Result<()> {
        let store = Arc::new(Store::open_in_memory()?);
        store.set_setting(
            EXTRAS_SETTING,
            &["mlx", "torch", "quantum", "models", "mlx"],
        )?;
        let env = Environment::new(crate::uv::Uv::new("uv", "/r/python"), Arc::clone(&store));
        assert_eq!(env.enabled_extras()?, [Extra::Mlx, Extra::Models]);
        assert_eq!(
            env.enable_extras([Extra::Jax])?,
            [Extra::Jax, Extra::Mlx, Extra::Models]
        );
        assert_eq!(
            store.setting::<Vec<String>>(EXTRAS_SETTING)?,
            Some(vec!["jax".into(), "mlx".into(), "models".into()])
        );
        Ok(())
    }
}
