//! Model assets: the weights real-model problems need (`docs/ARCHITECTURE.md` sections
//! 3.3, 3.7, and 4).
//!
//! The Python side owns what an asset is and whether it is present; core only runs
//! `ws assets list --json` and `ws assets fetch <names> --json` in the project
//! environment and maps their output onto the wire types. A fetch streams
//! [`AssetEvent`]s and always finishes with exactly one `done` event, whatever happens
//! (launch failure, a failed asset, cancel).
//!
//! One fetch runs at a time. Canceling kills the fetch's process group; files that
//! were complete stay on disk, so a later fetch resumes. Removing an asset deletes its
//! directory under the models directory and nothing else.

use std::ffi::OsString;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::mpsc::{self, Sender};
use std::sync::{Arc, Mutex, MutexGuard, PoisonError};
use std::thread;
use std::time::Duration;

use serde::Deserialize;

use crate::env::Environment;
use crate::error::{CoreError, Result};
use crate::harness::CommandSpec;
use crate::model::{AssetEvent, AssetList, AssetState, Extra, ModelAsset};
use crate::protocol::PREFIX;
use crate::sink::SharedSink;
use crate::stream::{Message, spawn_reader};
use crate::target::{LocalProcess, Process};

/// Environment variable that tells the Python side where the content root is.
const ROOT_ENV: &str = "WHETSTONE_ROOT";

/// `ws assets list --json` as the CLI prints it (snake_case; see PROBLEM_AUTHORING
/// section 15). Keys core does not use (`present`, `fetch_command`, `issues`) and keys
/// added later are ignored.
#[derive(Debug, Deserialize)]
struct WireListing {
    models_dir: String,
    #[serde(default)]
    assets: Vec<WireAsset>,
}

#[derive(Debug, Deserialize)]
struct WireAsset {
    name: String,
    #[serde(default)]
    repo: String,
    #[serde(default)]
    revision: String,
    #[serde(default)]
    files: Vec<String>,
    #[serde(default)]
    size_mb: f64,
    state: AssetState,
    #[serde(default)]
    reason: Option<String>,
    #[serde(default)]
    path: String,
    #[serde(default)]
    bytes_on_disk: u64,
    #[serde(default)]
    problems: Vec<String>,
}

impl From<WireListing> for AssetList {
    fn from(listing: WireListing) -> Self {
        Self {
            models_dir: listing.models_dir,
            assets: listing
                .assets
                .into_iter()
                .map(|asset| ModelAsset {
                    name: asset.name,
                    repo: asset.repo,
                    revision: asset.revision,
                    files: asset.files,
                    size_mb: asset.size_mb,
                    state: asset.state,
                    reason: asset.reason,
                    path: asset.path,
                    bytes_on_disk: asset.bytes_on_disk,
                    problems: asset.problems,
                })
                .collect(),
        }
    }
}

/// Parses the document printed by `ws assets list --json`. Anything printed before it
/// (a warning, say) is skipped: the document is the last line that is a JSON object.
pub(crate) fn parse_listing(stdout: &str) -> Result<AssetList> {
    let whole = serde_json::from_str::<WireListing>(stdout.trim());
    let listing = match whole {
        Ok(listing) => listing,
        Err(error) => stdout
            .lines()
            .rev()
            .find_map(|line| serde_json::from_str::<WireListing>(line.trim()).ok())
            .ok_or_else(|| {
                CoreError::Assets(format!(
                    "`ws assets list --json` printed something unexpected: {error}"
                ))
            })?,
    };
    Ok(listing.into())
}

/// An event of `ws assets fetch --json` as the CLI writes it.
#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(tag = "event")]
enum FetchEvent {
    #[serde(rename = "asset_start")]
    Start {
        asset: String,
        #[serde(default)]
        total_bytes: u64,
    },
    #[serde(rename = "asset_progress")]
    Progress {
        asset: String,
        #[serde(default)]
        file: String,
        #[serde(default)]
        done_bytes: u64,
        #[serde(default)]
        total_bytes: u64,
    },
    /// One file is complete. The front end follows bytes, not files, so this is not
    /// forwarded.
    #[serde(rename = "asset_file")]
    File {},
    #[serde(rename = "asset_end")]
    End {
        asset: String,
        ok: bool,
        #[serde(default)]
        message: Option<String>,
    },
}

/// Maps one stdout line of the fetch onto the event to forward, if any. A line without
/// the protocol prefix, or with a payload this build does not understand, is a log
/// line, so nothing the CLI prints is lost.
pub(crate) fn map_fetch_line(line: &str) -> Option<AssetEvent> {
    let event = line
        .strip_prefix(PREFIX)
        .and_then(|payload| serde_json::from_str::<FetchEvent>(payload).ok());
    match event {
        Some(FetchEvent::Start { asset, total_bytes }) => {
            Some(AssetEvent::Start { asset, total_bytes })
        }
        Some(FetchEvent::Progress {
            asset,
            file,
            done_bytes,
            total_bytes,
        }) => Some(AssetEvent::Progress {
            asset,
            file,
            done_bytes,
            total_bytes,
        }),
        Some(FetchEvent::File {}) => None,
        Some(FetchEvent::End { asset, ok, message }) => {
            Some(AssetEvent::AssetEnd { asset, ok, message })
        }
        None => Some(AssetEvent::Log {
            text: line.to_owned(),
        }),
    }
}

/// Whether `name` has the shape of an asset name: lowercase letters and digits with
/// single `.`, `_`, or `-` between them. This is also what keeps a name from being a
/// path (`..`, `a/b`, an absolute path) or a command-line option (`--all`).
pub fn is_asset_name(name: &str) -> bool {
    let alphanumeric = |b: u8| b.is_ascii_lowercase() || b.is_ascii_digit();
    let bytes = name.as_bytes();
    !bytes.is_empty()
        && bytes.len() <= 128
        && alphanumeric(bytes[0])
        && alphanumeric(bytes[bytes.len() - 1])
        && bytes
            .windows(2)
            .all(|pair| alphanumeric(pair[0]) || alphanumeric(pair[1]))
        && bytes
            .iter()
            .all(|&b| alphanumeric(b) || matches!(b, b'.' | b'_' | b'-'))
}

fn check_name(name: &str) -> Result<()> {
    if is_asset_name(name) {
        Ok(())
    } else {
        Err(CoreError::InvalidArgument(format!(
            "invalid asset name `{name}`: a name is lowercase letters and digits with `.`, `_`, or `-` between them"
        )))
    }
}

/// Decides which directory removing `name` deletes: `<models dir>/<name>`, and only
/// when `name` is an asset the catalog declares whose directory is exactly that.
pub(crate) fn removal_target(listing: &AssetList, name: &str) -> Result<PathBuf> {
    check_name(name)?;
    let asset = listing
        .assets
        .iter()
        .find(|asset| asset.name == name)
        .ok_or_else(|| {
            CoreError::InvalidArgument(format!(
                "`{name}` is not a model asset any problem declares"
            ))
        })?;
    let target = Path::new(&listing.models_dir).join(name);
    if !asset.path.is_empty() && Path::new(&asset.path) != target {
        return Err(CoreError::InvalidArgument(format!(
            "asset `{name}` is stored at {}, which is not inside the models directory",
            asset.path
        )));
    }
    Ok(target)
}

/// Deletes an asset directory. A missing directory is not an error; a symbolic link is
/// unlinked, never followed.
fn remove_directory(target: &Path) -> Result<()> {
    let metadata = match fs::symlink_metadata(target) {
        Ok(metadata) => metadata,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(()),
        Err(error) => return Err(CoreError::io("inspect", target, error)),
    };
    let removed = if metadata.is_dir() {
        fs::remove_dir_all(target)
    } else {
        fs::remove_file(target)
    };
    removed.map_err(|error| CoreError::io("remove", target, error))
}

/// The fetch in flight, if any.
struct Job {
    id: String,
    cancel: Sender<Message>,
}

/// Lists, fetches, and removes model assets for one content root.
#[derive(Clone)]
pub struct ModelAssets {
    root: PathBuf,
    env: Environment,
    job: Arc<Mutex<Option<Job>>>,
}

impl ModelAssets {
    /// Creates the accessor for the content root at `root`.
    pub fn new(root: impl Into<PathBuf>, env: Environment) -> Self {
        Self {
            root: root.into(),
            env,
            job: Arc::new(Mutex::new(None)),
        }
    }

    fn job(&self) -> MutexGuard<'_, Option<Job>> {
        self.job.lock().unwrap_or_else(PoisonError::into_inner)
    }

    /// Whether a fetch is running.
    pub fn is_fetching(&self) -> bool {
        self.job().is_some()
    }

    /// `uv run --project <root>/python [--extra ...] -- ws assets <args>`, told where
    /// the content root is.
    fn command(&self, extras: &[Extra], args: &[&str]) -> CommandSpec {
        let mut spec = self.env.uv().run(extras);
        spec.args.extend(["ws".into(), "assets".into()]);
        spec.args.extend(args.iter().map(OsString::from));
        spec.envs.push((ROOT_ENV.into(), self.root.clone().into()));
        spec
    }

    /// `list_assets`: every asset the catalog declares, with its state on disk.
    pub fn list(&self) -> Result<AssetList> {
        let spec = self.command(&self.env.enabled_extras()?, &["list", "--json"]);
        let output = Command::new(&spec.program)
            .args(&spec.args)
            .envs(spec.envs.iter().map(|(key, value)| (key, value)))
            .stdin(Stdio::null())
            .output()
            .map_err(|error| {
                CoreError::Assets(if error.kind() == std::io::ErrorKind::NotFound {
                    "uv was not found, so model assets cannot be listed. Install uv \
                     (https://docs.astral.sh/uv/), then reopen Whetstone."
                        .to_owned()
                } else {
                    format!("`{}` could not be started: {error}", spec.display())
                })
            })?;
        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            let reason = last_line(&stderr).unwrap_or("it printed nothing");
            return Err(CoreError::Assets(format!(
                "`ws assets list` failed: {reason}"
            )));
        }
        parse_listing(&String::from_utf8_lossy(&output.stdout))
    }

    /// `fetch_assets`: starts downloading `names` and returns the job id at once.
    ///
    /// Events flow to `sink` from a background thread and always finish with exactly
    /// one `done` event. An error returned here (no names, a malformed name, a fetch or
    /// an install already running) means nothing was started and no events follow.
    pub fn fetch(&self, names: &[String], sink: SharedSink<AssetEvent>) -> Result<String> {
        if names.is_empty() {
            return Err(CoreError::InvalidArgument(
                "name at least one model asset to download".to_owned(),
            ));
        }
        names.iter().try_for_each(|name| check_name(name))?;
        if self.env.is_installing() {
            return Err(CoreError::InstallInProgress);
        }

        let job_id = uuid::Uuid::new_v4().to_string();
        let (tx, rx) = mpsc::channel();
        {
            let mut job = self.job();
            if job.is_some() {
                return Err(CoreError::FetchInProgress);
            }
            *job = Some(Job {
                id: job_id.clone(),
                cancel: tx.clone(),
            });
        }

        let this = self.clone();
        let names = names.to_vec();
        let spawned = thread::Builder::new()
            .name(format!("asset-fetch-{job_id}"))
            .spawn(move || {
                let done = this.run_fetch(&names, &sink, &tx, &rx);
                // Free the slot first, so a retry started from the `done` handler works.
                *this.job() = None;
                sink.emit(done);
            });
        match spawned {
            Ok(_) => Ok(job_id),
            Err(source) => {
                *self.job() = None;
                Err(CoreError::Io {
                    context: "failed to start the fetch thread".to_owned(),
                    source,
                })
            }
        }
    }

    /// `cancel_asset_fetch`: kills the fetch's process group. The stream ends with
    /// `done` (`canceled: true`). An unknown or finished job is a no-op.
    pub fn cancel(&self, job_id: &str) {
        if let Some(job) = self.job().as_ref().filter(|job| job.id == job_id) {
            // The receiver is gone only if the fetch just finished.
            let _ = job.cancel.send(Message::Cancel);
        }
    }

    /// `remove_asset`: deletes `<models dir>/<name>` and returns the new listing.
    /// Refused while a fetch is running, and for any name the catalog does not declare.
    pub fn remove(&self, name: &str) -> Result<AssetList> {
        check_name(name)?;
        if self.is_fetching() {
            return Err(CoreError::FetchInProgress);
        }
        let target = removal_target(&self.list()?, name)?;
        tracing::info!(asset = name, path = %target.display(), "removing a model asset");
        remove_directory(&target)?;
        self.list()
    }

    /// Runs the fetch to completion and returns its `done` event.
    fn run_fetch(
        &self,
        names: &[String],
        sink: &SharedSink<AssetEvent>,
        tx: &Sender<Message>,
        rx: &mpsc::Receiver<Message>,
    ) -> AssetEvent {
        let failed = |message: String| AssetEvent::Done {
            ok: false,
            canceled: false,
            message: Some(message),
        };
        sink.emit(AssetEvent::Preparing {
            message: "Checking the Python environment".to_owned(),
        });

        // Whatever is installed stays installed; `models` is what the fetch itself needs.
        let mut extras = match self.env.current_extras() {
            Ok(extras) => extras,
            Err(error) => return failed(error.to_string()),
        };
        let had_models = extras.contains(&Extra::Models);
        if !had_models {
            extras.push(Extra::Models);
            sink.emit(AssetEvent::Preparing {
                message: "Installing the download tools (first download only)".to_owned(),
            });
        }

        let mut args = vec!["fetch"];
        args.extend(names.iter().map(String::as_str));
        args.push("--json");
        let spec = self.command(&extras, &args);
        sink.emit(AssetEvent::Log {
            text: format!("$ {}", spec.display()),
        });
        let mut process = match LocalProcess::spawn(&spec) {
            Ok(process) => process,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
                return failed(
                    "uv was not found, so nothing can be downloaded. Install uv \
                     (https://docs.astral.sh/uv/), then reopen Whetstone."
                        .to_owned(),
                );
            }
            Err(error) => return failed(format!("The download could not be started: {error}")),
        };

        let mut open = 0;
        if let Some(stdout) = process.take_stdout() {
            open += spawn_reader(stdout, tx.clone(), Message::Stdout);
        }
        if let Some(stderr) = process.take_stderr() {
            open += spawn_reader(stderr, tx.clone(), Message::Stderr);
        }

        // The CLI reached the point of reporting, which means the environment has the
        // `models` extra: remember it, so no later command removes it again.
        let mut reported = false;
        let mut failures: Vec<String> = Vec::new();
        let mut last_stderr: Option<String> = None;
        let mut canceled = false;
        while open > 0 {
            match rx.recv() {
                Ok(Message::Closed) => open -= 1,
                Ok(Message::Cancel) => {
                    canceled = true;
                    break;
                }
                Ok(Message::Stderr(text)) => {
                    if !text.trim().is_empty() {
                        last_stderr = Some(text.trim().to_owned());
                    }
                    sink.emit(AssetEvent::Log { text });
                }
                Ok(Message::Stdout(line)) => {
                    let Some(event) = map_fetch_line(&line) else {
                        reported = true;
                        continue;
                    };
                    if !matches!(event, AssetEvent::Log { .. }) {
                        reported = true;
                    }
                    if let AssetEvent::AssetEnd {
                        asset,
                        ok: false,
                        message,
                    } = &event
                    {
                        failures.push(match message {
                            Some(message) => format!("{asset}: {message}"),
                            None => format!("{asset}: the download failed"),
                        });
                    }
                    sink.emit(event);
                }
                Err(_) => break,
            }
        }

        let code = process.finish(if canceled {
            Duration::ZERO
        } else {
            Duration::from_secs(5)
        });
        if (reported || code == Some(0))
            && !had_models
            && let Err(error) = self.env.enable_extras([Extra::Models])
        {
            tracing::warn!("could not record the models extra: {error}");
        }
        tracing::info!(assets = ?names, ?code, canceled, "asset fetch finished");

        if canceled {
            return AssetEvent::Done {
                ok: false,
                canceled: true,
                message: None,
            };
        }
        match code {
            Some(0) if failures.is_empty() => AssetEvent::Done {
                ok: true,
                canceled: false,
                message: None,
            },
            _ if !failures.is_empty() => failed(failures.join("; ")),
            Some(code) => failed(
                last_stderr.unwrap_or_else(|| format!("The download exited with code {code}.")),
            ),
            None => failed("The download was killed by a signal.".to_owned()),
        }
    }
}

/// The last non-empty line of `text`.
fn last_line(text: &str) -> Option<&str> {
    text.lines()
        .rev()
        .map(str::trim)
        .find(|line| !line.is_empty())
}

#[cfg(test)]
mod tests {
    use super::*;

    const LISTING: &str = r#"{
  "models_dir": "/abs/learnai/.whetstone/models",
  "assets": [
    {
      "name": "granite-4.0-350m",
      "repo": "ibm-granite/granite-4.0-350m",
      "revision": "bd8a1497065c0d6ba1ef19af6b0d2b14bacf71c2",
      "files": ["config.json", "tokenizer.json", "model.safetensors"],
      "size_mb": 712,
      "state": "missing",
      "present": false,
      "reason": null,
      "path": "/abs/learnai/.whetstone/models/granite-4.0-350m",
      "bytes_on_disk": 0,
      "problems": ["real-model-block", "real-model-logits"],
      "fetch_command": "ws assets fetch granite-4.0-350m",
      "a_key_from_the_future": {"x": 1}
    },
    {
      "name": "tiny",
      "repo": "o/tiny",
      "revision": "0123456789abcdef0123456789abcdef01234567",
      "files": ["config.json"],
      "size_mb": 0.01,
      "state": "incomplete",
      "present": false,
      "reason": "config.json is missing",
      "path": "/abs/learnai/.whetstone/models/tiny",
      "bytes_on_disk": 12,
      "problems": ["p"],
      "fetch_command": "ws assets fetch tiny"
    }
  ],
  "issues": [{"problem": "some-problem", "message": "asset `x`: `revision` must be ..."}]
}"#;

    #[test]
    fn the_documented_listing_parses() {
        let listing = parse_listing(LISTING).expect("listing");
        assert_eq!(listing.models_dir, "/abs/learnai/.whetstone/models");
        assert_eq!(listing.assets.len(), 2);
        let granite = &listing.assets[0];
        assert_eq!(granite.name, "granite-4.0-350m");
        assert_eq!(granite.size_mb, 712.0);
        assert_eq!(granite.state, AssetState::Missing);
        assert_eq!(granite.reason, None);
        assert_eq!(granite.problems, ["real-model-block", "real-model-logits"]);
        let tiny = &listing.assets[1];
        assert_eq!(tiny.state, AssetState::Incomplete);
        assert_eq!(tiny.reason.as_deref(), Some("config.json is missing"));
        assert_eq!(tiny.bytes_on_disk, 12);
    }

    #[test]
    fn a_listing_survives_noise_and_rejects_garbage() {
        let compact = serde_json::to_string(
            &serde_json::from_str::<serde_json::Value>(LISTING).expect("json"),
        )
        .expect("compact");
        let noisy = format!("warning: something\n{compact}\n");
        assert_eq!(parse_listing(&noisy).expect("listing").assets.len(), 2);
        assert_eq!(
            parse_listing(r#"{"models_dir": "/m"}"#)
                .expect("empty")
                .assets,
            vec![]
        );
        assert!(matches!(
            parse_listing("Traceback (most recent call last):"),
            Err(CoreError::Assets(_))
        ));
        assert!(matches!(parse_listing(""), Err(CoreError::Assets(_))));
    }

    #[test]
    fn fetch_lines_map_to_asset_events() {
        let start = r#"@@WS@@ {"event":"asset_start","asset":"granite-4.0-350m","repo":"ibm-granite/granite-4.0-350m","revision":"bd8a","path":"/abs/granite-4.0-350m","files":[{"path":"config.json","size":1764}],"total_bytes":711966212}"#;
        assert_eq!(
            map_fetch_line(start),
            Some(AssetEvent::Start {
                asset: "granite-4.0-350m".into(),
                total_bytes: 711_966_212
            })
        );
        let progress = r#"@@WS@@ {"event":"asset_progress","asset":"granite-4.0-350m","file":"model.safetensors","file_bytes":150763002,"file_total":704786224,"done_bytes":150771331,"total_bytes":711966212}"#;
        assert_eq!(
            map_fetch_line(progress),
            Some(AssetEvent::Progress {
                asset: "granite-4.0-350m".into(),
                file: "model.safetensors".into(),
                done_bytes: 150_771_331,
                total_bytes: 711_966_212
            })
        );
        let file = r#"@@WS@@ {"event":"asset_file","asset":"granite-4.0-350m","file":"model.safetensors","size":704786224,"cached":false}"#;
        assert_eq!(map_fetch_line(file), None);
        let end = r#"@@WS@@ {"event":"asset_end","asset":"granite-4.0-350m","ok":true,"path":"/abs/granite-4.0-350m","bytes":711966212,"message":null}"#;
        assert_eq!(
            map_fetch_line(end),
            Some(AssetEvent::AssetEnd {
                asset: "granite-4.0-350m".into(),
                ok: true,
                message: None
            })
        );
        let failed = r#"@@WS@@ {"event":"asset_end","asset":"tiny","ok":false,"path":"/abs/tiny","bytes":0,"message":"no network"}"#;
        assert_eq!(
            map_fetch_line(failed),
            Some(AssetEvent::AssetEnd {
                asset: "tiny".into(),
                ok: false,
                message: Some("no network".into())
            })
        );
    }

    #[test]
    fn anything_else_from_a_fetch_is_a_log_line() {
        for line in [
            "Downloading huggingface_hub",
            "@@WS@@ not json",
            r#"@@WS@@ {"event":"asset_verified","asset":"x"}"#,
            r#"@@WS@@ {"event":"asset_end","asset":"x"}"#,
            "",
        ] {
            assert_eq!(
                map_fetch_line(line),
                Some(AssetEvent::Log {
                    text: line.to_owned()
                })
            );
        }
    }

    #[test]
    fn asset_names_cannot_be_paths_or_options() {
        for name in [
            "granite-4.0-350m",
            "granite-4.2-3b-mlx-8bit",
            "tiny_test.model",
            "a",
            "7b",
        ] {
            assert!(is_asset_name(name), "{name}");
        }
        for name in [
            "",
            ".",
            "..",
            "../state.db",
            "a/b",
            "/etc",
            "a\\b",
            "-a",
            "--all",
            "a-",
            ".hidden",
            "a..b",
            "a--b",
            "Granite",
            "a b",
            "a\0b",
            "caf\u{e9}",
            "~",
        ] {
            assert!(!is_asset_name(name), "{name:?}");
        }
        assert!(!is_asset_name(&"a".repeat(129)));
    }

    #[test]
    fn only_declared_assets_inside_the_models_directory_are_removed() {
        let listing = parse_listing(LISTING).expect("listing");
        assert_eq!(
            removal_target(&listing, "tiny").expect("target"),
            Path::new("/abs/learnai/.whetstone/models/tiny")
        );
        for name in ["..", "../state.db", "tiny/..", "/abs", "", "TINY"] {
            assert!(
                matches!(
                    removal_target(&listing, name),
                    Err(CoreError::InvalidArgument(_))
                ),
                "{name:?}"
            );
        }
        // Well-formed, but not something the catalog declares.
        let error = removal_target(&listing, "state.db").expect_err("undeclared");
        assert!(error.to_string().contains("not a model asset"), "{error}");

        // A listing whose path disagrees with the models directory is not trusted.
        let mut moved = listing;
        moved.assets[1].path = "/somewhere/else/tiny".into();
        assert!(removal_target(&moved, "tiny").is_err());
    }

    #[test]
    fn removal_deletes_the_directory_and_never_follows_a_link() {
        let tmp = tempfile::tempdir().expect("temp dir");
        let models = tmp.path().join("models");
        let asset = models.join("tiny");
        fs::create_dir_all(asset.join("nested")).expect("create");
        fs::write(asset.join("nested/model.safetensors"), b"weights").expect("write");
        remove_directory(&asset).expect("remove");
        assert!(!asset.exists());
        assert!(models.is_dir());
        // Removing what is not there is fine.
        remove_directory(&asset).expect("remove again");

        #[cfg(unix)]
        {
            let precious = tmp.path().join("precious");
            fs::create_dir_all(&precious).expect("create");
            fs::write(precious.join("keep.txt"), b"keep").expect("write");
            std::os::unix::fs::symlink(&precious, &asset).expect("symlink");
            remove_directory(&asset).expect("unlink");
            assert!(fs::symlink_metadata(&asset).is_err());
            assert!(precious.join("keep.txt").is_file());
        }
    }
}
