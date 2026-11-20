//! The [`Core`] facade: one method per command in `docs/ARCHITECTURE.md` section 4.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, PoisonError};

use crate::areas::load_areas;
use crate::catalog::{Catalog, CatalogIssue, CatalogLoader, ProblemMeta};
use crate::docs::DocsHelper;
use crate::env::Environment;
use crate::error::{CoreError, Result};
use crate::harness::{CommandSpec, HarnessCommand, HarnessJob, UvHarness};
use crate::lsp::{self, EditorOptions, LanguageServer};
use crate::model::{
    Area, AssetEvent, AssetList, Difficulty, EditorDocs, EditorEvent, EditorStatus,
    EditorStopReason, EnvStatus, Extra, Framework, InstallEvent, ProblemDetail, ProblemStatus,
    ProblemSummary, Review, ReviewAction, RunEvent, RunMode, Settings, SolutionFile, SshHost,
    Stats, Submission, SubmissionDetail, TargetCheckEvent, TargetInfo, Track,
};
use crate::models::{ModelAssets, RemoteFetch};
use crate::progress::{current_streak, derive_status};
use crate::review::{self, Clock, SystemClock};
use crate::run::{RunLimits, RunPlan, RunRegistry};
use crate::sink::EventSink;
use crate::solutions::Workspace;
use crate::ssh::{Ssh, SshOptions};
use crate::store::{Store, SubmissionProgress};
use crate::target::{LOCAL_TARGET, LocalTarget, Target};
use crate::targets::Targets;
use crate::uv::{Uv, venv_program};

/// The largest document `editor_docs` accepts, in bytes.
const MAX_DOCS_SOURCE_BYTES: usize = 1024 * 1024;

/// How to construct a [`Core`].
pub struct CoreOptions {
    /// The content root (see [`crate::root::resolve_root`]).
    pub root: PathBuf,
    /// Overrides the harness command; `None` uses `uv run ... python -m learnai.harness`.
    pub harness: Option<Arc<dyn HarnessCommand>>,
    /// Overrides the uv executable; `None` searches for it.
    pub uv_program: Option<PathBuf>,
    /// Wall-clock limits for runs.
    pub limits: RunLimits,
    /// Overrides for the SSH tooling used by remote targets.
    pub ssh: SshOptions,
    /// Overrides the clock that dates submissions and review schedules; `None` uses
    /// the system clock. Tests move time through this instead of sleeping.
    pub clock: Option<Arc<dyn Clock>>,
    /// How the editor's helper processes (language server, docs helper) are run.
    pub editor: EditorOptions,
}

impl CoreOptions {
    /// Default options for a content root.
    pub fn new(root: impl Into<PathBuf>) -> Self {
        Self {
            root: root.into(),
            harness: None,
            uv_program: None,
            limits: RunLimits::default(),
            ssh: SshOptions::default(),
            clock: None,
            editor: EditorOptions::default(),
        }
    }
}

struct Inner {
    root: PathBuf,
    catalog: Mutex<CatalogLoader>,
    workspace: Workspace,
    store: Arc<Store>,
    env: Environment,
    models: ModelAssets,
    targets: Targets,
    runs: Arc<RunRegistry>,
    limits: RunLimits,
    clock: Arc<dyn Clock>,
    editor: LanguageServer,
    docs: DocsHelper,
}

/// The application core. Cheap to clone; all clones share state.
#[derive(Clone)]
pub struct Core {
    inner: Arc<Inner>,
}

impl Core {
    /// Opens the core for a content root, creating `.learnai/state.db` if needed.
    pub fn open(options: CoreOptions) -> Result<Self> {
        let root = options.root;
        let store = Arc::new(Store::open(&root.join(".learnai").join("state.db"))?);
        let uv = match options.uv_program {
            Some(program) => Uv::new(program, root.join("python")),
            None => Uv::locate(&root),
        };
        let harness = options
            .harness
            .unwrap_or_else(|| Arc::new(UvHarness::new(uv.clone())) as Arc<dyn HarnessCommand>);
        let local: Arc<dyn Target> = Arc::new(LocalTarget::new(harness));

        let workspace = Workspace::new(&root);
        let venv = options.editor.venv.clone().unwrap_or_else(|| uv.venv());
        let docs = options.editor.docs.clone().unwrap_or_else(|| CommandSpec {
            // Isolated mode: the working directory and the user's environment never
            // reach `sys.path`, so nothing the learner wrote can be imported.
            program: venv_program(&venv, "python").into(),
            args: vec!["-I".into(), "-m".into(), "learnai.editor.docs".into()],
            cwd: Some(root.join("python")),
            ..CommandSpec::default()
        });
        let editor = LanguageServer::new(workspace.dir(), venv, &options.editor);
        let docs = DocsHelper::new(
            docs,
            options.editor.docs_timeout,
            options.editor.docs_retry_pause,
        );

        let env = Environment::new(uv, Arc::clone(&store));
        let targets = Targets::new(
            &root,
            Arc::clone(&store),
            env.clone(),
            Ssh::new(options.ssh),
            local,
        );
        Ok(Self {
            inner: Arc::new(Inner {
                catalog: Mutex::new(CatalogLoader::new(&root)),
                workspace,
                editor,
                docs,
                models: ModelAssets::new(&root, env.clone()),
                env,
                store,
                targets,
                runs: Arc::new(RunRegistry::default()),
                limits: options.limits,
                clock: options
                    .clock
                    .unwrap_or_else(|| Arc::new(SystemClock) as Arc<dyn Clock>),
                root,
            }),
        })
    }

    /// The content root.
    pub fn root(&self) -> &Path {
        &self.inner.root
    }

    fn catalog(&self) -> Catalog {
        self.inner
            .catalog
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .load()
    }

    /// Malformed tracks, problems, and areas found by the latest scan.
    pub fn catalog_issues(&self) -> Vec<CatalogIssue> {
        let catalog = self.catalog();
        let mut issues = catalog.issues().to_vec();
        issues.extend(load_areas(&self.inner.root, &catalog).1);
        issues
    }

    /// `list_catalog_issues`: the same issues in wire form.
    pub fn list_catalog_issues(&self) -> Vec<crate::model::CatalogIssue> {
        self.catalog_issues()
            .into_iter()
            .map(|issue| crate::model::CatalogIssue {
                path: issue.path.to_string_lossy().into_owned(),
                message: issue.message,
            })
            .collect()
    }

    /// `get_problem_asset`: a statement figure as a `data:` URL. `path` is relative to
    /// the problem's `assets/` directory and may not escape it; only `.svg` and `.png`.
    pub fn get_problem_asset(&self, id: &str, path: &str) -> Result<String> {
        crate::assets::read_asset(&self.catalog().problem(id)?.dir, path)
    }

    fn summarize(
        &self,
        problem: &ProblemMeta,
        progress: Option<&SubmissionProgress>,
    ) -> ProblemSummary {
        let solved_frameworks: Vec<Framework> = progress
            .map(|p| {
                p.solved_frameworks
                    .iter()
                    .copied()
                    .filter(|f| problem.frameworks.contains(f))
                    .collect()
            })
            .unwrap_or_default();
        let status = derive_status(
            !solved_frameworks.is_empty(),
            progress.is_some(),
            // Only touch the file system when the answer is still open.
            solved_frameworks.is_empty()
                && progress.is_none()
                && self.inner.workspace.is_modified(problem),
        );
        ProblemSummary {
            id: problem.id.clone(),
            title: problem.title.clone(),
            track: problem.track.clone(),
            order: problem.order,
            difficulty: problem.difficulty,
            frameworks: problem.frameworks.clone(),
            judges: problem.judges.clone(),
            tags: problem.tags.clone(),
            prerequisites: problem.prerequisites.clone(),
            requires: problem.requires,
            status,
            solved_frameworks,
            assets: problem.assets.clone(),
        }
    }

    fn summaries(&self, catalog: &Catalog) -> Result<Vec<ProblemSummary>> {
        let progress = self.inner.store.submission_progress()?;
        Ok(catalog
            .problems()
            .iter()
            .map(|p| self.summarize(p, progress.get(&p.id)))
            .collect())
    }

    fn detail(&self, problem: &ProblemMeta) -> Result<ProblemDetail> {
        let progress = self.inner.store.submission_progress()?;
        let summary = self.summarize(problem, progress.get(&problem.id));
        let editorial_unlocked = summary.status == ProblemStatus::Solved
            || self.inner.store.editorial_revealed(&problem.id)?;
        Ok(ProblemDetail {
            statement: problem.statement()?,
            hints: problem.hints()?,
            editorial: if editorial_unlocked {
                Some(problem.editorial()?)
            } else {
                None
            },
            editorial_unlocked,
            entry: problem.entry.clone(),
            time_limit_s: problem.time_limit_s,
            summary,
        })
    }

    /// `list_tracks`: tracks in order, with problem and solved counts.
    pub fn list_tracks(&self) -> Result<Vec<Track>> {
        let catalog = self.catalog();
        let summaries = self.summaries(&catalog)?;
        Ok(catalog
            .tracks()
            .iter()
            .map(|track| {
                let (mut problem_count, mut solved_count) = (0, 0);
                for summary in summaries.iter().filter(|s| s.track == track.id) {
                    problem_count += 1;
                    solved_count += u32::from(summary.status == ProblemStatus::Solved);
                }
                Track {
                    id: track.id.clone(),
                    title: track.title.clone(),
                    description: track.description.clone(),
                    order: track.order,
                    problem_count,
                    solved_count,
                    scene: track.scene.clone(),
                    color: track.color.clone(),
                    core: track.core.clone(),
                }
            })
            .collect())
    }

    /// `list_areas`: every valid area, sorted by file name.
    pub fn list_areas(&self) -> Result<Vec<Area>> {
        let catalog = self.catalog();
        let (areas, _issues) = load_areas(&self.inner.root, &catalog);
        Ok(areas
            .into_iter()
            .map(|area| Area {
                id: area.id,
                title: area.title,
                description: area.description,
                modules: area.modules,
            })
            .collect())
    }

    /// `list_problems`: every valid problem, in track order then problem order.
    pub fn list_problems(&self) -> Result<Vec<ProblemSummary>> {
        self.summaries(&self.catalog())
    }

    /// `get_problem`: statement, hints, and (when unlocked) the editorial.
    pub fn get_problem(&self, id: &str) -> Result<ProblemDetail> {
        self.detail(self.catalog().problem(id)?)
    }

    /// `read_solution`: the saved solution, or the starter if none is saved.
    pub fn read_solution(&self, id: &str, framework: Framework) -> Result<SolutionFile> {
        self.inner
            .workspace
            .read(self.catalog().problem(id)?, framework)
    }

    /// `write_solution`: saves the solution to `workspace/<id>/<framework>.py`.
    pub fn write_solution(&self, id: &str, framework: Framework, code: &str) -> Result<()> {
        self.inner
            .workspace
            .write(self.catalog().problem(id)?, framework, code)
    }

    /// `reset_solution`: discards the saved solution and returns the starter.
    pub fn reset_solution(&self, id: &str, framework: Framework) -> Result<SolutionFile> {
        self.inner
            .workspace
            .reset(self.catalog().problem(id)?, framework)
    }

    /// `start_run`: starts judging the current solution and returns the run id at once.
    ///
    /// Events flow to `sink` from a background thread and always finish with exactly one
    /// `end` event. Errors returned here (unknown problem, framework, or target) mean no
    /// run was started and no events will be emitted.
    pub fn start_run(
        &self,
        problem_id: &str,
        framework: Framework,
        mode: RunMode,
        target: &str,
        sink: impl EventSink<RunEvent>,
    ) -> Result<String> {
        self.launch_run(problem_id, framework, mode, target, false, sink)
    }

    /// `start_run` with `review: true`: judges the review attempt
    /// (`workspace/<id>/review/<framework>.py`) instead of the saved solution. An
    /// accepted review submit advances the problem's review schedule. Fails when the
    /// problem has no active review or reviews are turned off.
    pub fn start_review_run(
        &self,
        problem_id: &str,
        framework: Framework,
        mode: RunMode,
        target: &str,
        sink: impl EventSink<RunEvent>,
    ) -> Result<String> {
        self.launch_run(problem_id, framework, mode, target, true, sink)
    }

    fn launch_run(
        &self,
        problem_id: &str,
        framework: Framework,
        mode: RunMode,
        target: &str,
        is_review: bool,
        sink: impl EventSink<RunEvent>,
    ) -> Result<String> {
        let catalog = self.catalog();
        let problem = catalog.problem(problem_id)?;
        let workspace = &self.inner.workspace;
        let (solution, saved) = if is_review {
            self.active_review(&problem.id)?;
            (
                workspace.read_review(problem, framework)?,
                workspace.review_path(&problem.id, framework),
            )
        } else {
            (
                workspace.read(problem, framework)?,
                workspace.path(&problem.id, framework),
            )
        };
        let target = self.inner.targets.resolve(target)?;

        // Judge the saved file when there is one, else the starter in place.
        let solution_path = if saved.is_file() {
            saved
        } else {
            problem.starter_path(framework)
        };

        let plan = RunPlan {
            problem_id: problem.id.clone(),
            time_limit_s: problem.time_limit_s,
            job: HarnessJob {
                problem_dir: problem.dir.clone(),
                solution: solution_path,
                framework,
                mode,
                case: None,
                extras: self.inner.env.enabled_extras()?,
                review: is_review,
            },
            code: solution.code,
            target,
            limits: self.inner.limits,
            store: Arc::clone(&self.inner.store),
            sink: Arc::new(sink),
            clock: Arc::clone(&self.inner.clock),
            review_dir: workspace.review_dir(&problem.id),
        };
        self.inner.runs.start(plan).map_err(|source| CoreError::Io {
            context: "failed to start the run thread".to_owned(),
            source,
        })
    }

    /// The active review schedule of a problem; an error when reviews are off or the
    /// problem has none.
    fn active_review(&self, problem_id: &str) -> Result<crate::store::ReviewRow> {
        if !review::enabled(&self.inner.store)? {
            return Err(CoreError::ReviewsDisabled);
        }
        self.inner
            .store
            .review(problem_id)?
            .filter(|review| !review.retired)
            .ok_or_else(|| CoreError::NoActiveReview(problem_id.to_owned()))
    }

    /// `list_reviews`: the active review schedules of problems in the catalog, soonest
    /// due first. Empty while reviews are turned off.
    pub fn list_reviews(&self) -> Result<Vec<Review>> {
        if !review::enabled(&self.inner.store)? {
            return Ok(Vec::new());
        }
        let catalog = self.catalog();
        let now = self.inner.clock.now();
        Ok(self
            .inner
            .store
            .active_reviews()?
            .into_iter()
            .filter(|row| catalog.problem(&row.problem_id).is_ok())
            .map(|row| review::to_wire(row, now))
            .collect())
    }

    /// `start_review`: discards any review attempt for the framework and returns the
    /// starter. The saved solution is not touched.
    pub fn start_review(&self, id: &str, framework: Framework) -> Result<SolutionFile> {
        let catalog = self.catalog();
        let problem = catalog.problem(id)?;
        self.active_review(&problem.id)?;
        self.inner.workspace.start_review(problem, framework)
    }

    /// `read_review_solution`: the review attempt, or the starter if there is none.
    pub fn read_review_solution(&self, id: &str, framework: Framework) -> Result<SolutionFile> {
        self.inner
            .workspace
            .read_review(self.catalog().problem(id)?, framework)
    }

    /// `write_review_solution`: saves the review attempt to
    /// `workspace/<id>/review/<framework>.py`.
    pub fn write_review_solution(&self, id: &str, framework: Framework, code: &str) -> Result<()> {
        let catalog = self.catalog();
        let problem = catalog.problem(id)?;
        self.active_review(&problem.id)?;
        self.inner.workspace.write_review(problem, framework, code)
    }

    /// `resolve_review`: ends a review without solving it (`forgot`, `snooze`, or
    /// `retire`) and returns the remaining schedules. `forgot` and `retire` remove the
    /// review attempt; `snooze` keeps it for the next day.
    pub fn resolve_review(&self, id: &str, action: ReviewAction) -> Result<Vec<Review>> {
        let current = self.active_review(id)?;
        let next = review::resolved(&current, action, self.inner.clock.now());
        self.inner.store.save_review(&next)?;
        if action != ReviewAction::Snooze {
            self.inner.workspace.clear_review(id)?;
        }
        self.list_reviews()
    }

    /// `get_settings`: the app settings stored in the state database.
    pub fn get_settings(&self) -> Result<Settings> {
        Ok(Settings {
            reviews_enabled: review::enabled(&self.inner.store)?,
            editor_intelligence: self.editor_enabled()?,
        })
    }

    fn editor_enabled(&self) -> Result<bool> {
        Ok(self
            .inner
            .store
            .setting::<bool>(lsp::ENABLED_SETTING)?
            .unwrap_or(true))
    }

    /// `set_editor_intelligence`: turns editor intelligence on or off. Turning it off
    /// stops the language server and the docs helper; nothing starts while it is off.
    pub fn set_editor_intelligence(&self, enabled: bool) -> Result<Settings> {
        self.inner
            .store
            .set_setting(lsp::ENABLED_SETTING, &enabled)?;
        if !enabled {
            self.inner.editor.stop(None, EditorStopReason::Disabled);
            self.inner.docs.stop();
        }
        self.get_settings()
    }

    /// `editor_status`: the preference, whether the language server is installed, and
    /// the state of the latest session. Starts nothing.
    pub fn editor_status(&self) -> Result<EditorStatus> {
        let editor = &self.inner.editor;
        Ok(EditorStatus {
            enabled: self.editor_enabled()?,
            installed: editor.installed(),
            installing: self.inner.env.is_installing(),
            server_version: editor.version(),
            state: editor.state(),
        })
    }

    /// `editor_start`: starts a language server session and returns its id at once.
    /// A session that is already running is stopped first.
    ///
    /// Events flow to `sink` from background threads and end with exactly one terminal
    /// state (`notInstalled`, `failed`, or `stopped`). Nothing is installed or
    /// downloaded: when the server is missing the stream is a single `notInstalled`.
    /// Fails, with no events, while editor intelligence is turned off.
    pub fn editor_start(&self, sink: impl EventSink<EditorEvent>) -> Result<String> {
        if !self.editor_enabled()? {
            return Err(CoreError::EditorDisabled);
        }
        self.inner.editor.start(Arc::new(sink))
    }

    /// `editor_send`: passes one JSON-RPC message (bare JSON text) from the editor's
    /// client to the language server of session `session_id`. Core adds the framing
    /// and rewrites `initialize` (see [`crate::lsp`]).
    pub fn editor_send(&self, session_id: &str, message: &str) -> Result<()> {
        self.inner.editor.send(session_id, message)
    }

    /// `editor_stop`: stops the session (any session when `session_id` is `None`) and
    /// the docs helper, and returns once their processes are gone. Stopping a session
    /// that was already replaced or has ended does nothing.
    pub fn editor_stop(&self, session_id: Option<&str>) {
        if self
            .inner
            .editor
            .stop(session_id, EditorStopReason::Requested)
            || session_id.is_none()
        {
            self.inner.docs.stop();
        }
    }

    /// `editor_install`: installs the language server (the `editor` extra) the way
    /// `env_install` installs a framework: one `uv sync` with every installed extra
    /// plus `editor`, so nothing else is removed. Returns once the install has started;
    /// events flow to `sink` and always finish with exactly one `done` event.
    pub fn editor_install(&self, sink: impl EventSink<InstallEvent>) -> Result<()> {
        if self.inner.models.is_fetching() {
            return Err(CoreError::FetchInProgress);
        }
        self.inner.env.install(Some(Extra::Editor), Arc::new(sink))
    }

    /// `editor_docs`: runtime documentation for `expr` (or its attribute `member`) as
    /// written in a file whose text is `source`. The file is never executed; see
    /// `learnai/editor/docs.py`. A helper that cannot answer yields a result with
    /// reason `unavailable`, not an error.
    pub fn editor_docs(
        &self,
        source: &str,
        expr: &str,
        member: Option<&str>,
    ) -> Result<EditorDocs> {
        if !self.editor_enabled()? {
            return Err(CoreError::EditorDisabled);
        }
        if source.len() > MAX_DOCS_SOURCE_BYTES {
            return Err(CoreError::InvalidArgument(format!(
                "the document is larger than {MAX_DOCS_SOURCE_BYTES} bytes"
            )));
        }
        Ok(self.inner.docs.lookup(source, expr, member))
    }

    /// Stops every helper process core keeps running (the language server and the
    /// docs helper). Shells call this when the app exits; dropping the last clone of
    /// the core does the same.
    pub fn shutdown(&self) {
        self.inner.editor.stop(None, EditorStopReason::Shutdown);
        self.inner.docs.stop();
    }

    /// `set_reviews_enabled`: turns reviews on or off. Turning them off keeps every
    /// schedule; turning them on schedules the problems that were solved meanwhile,
    /// dated from their first accepted submit.
    pub fn set_reviews_enabled(&self, enabled: bool) -> Result<Settings> {
        self.inner
            .store
            .set_setting(review::ENABLED_SETTING, &enabled)?;
        if enabled {
            self.inner
                .store
                .backfill_reviews(review::INTERVALS_DAYS[0])?;
        }
        self.get_settings()
    }

    /// `cancel_run`: kills the run's whole process group. The run ends with verdict
    /// `canceled`. Canceling a finished or unknown run does nothing.
    pub fn cancel_run(&self, run_id: &str) {
        self.inner.runs.cancel(run_id);
    }

    /// `list_submissions`: stored submissions, newest first, optionally for one problem.
    pub fn list_submissions(&self, problem_id: Option<&str>) -> Result<Vec<Submission>> {
        self.inner.store.list_submissions(problem_id)
    }

    /// `get_submission`: one submission with its code snapshot and per-case results.
    pub fn get_submission(&self, id: i64) -> Result<SubmissionDetail> {
        self.inner.store.get_submission(id)
    }

    /// `unlock_editorial`: reveals the editorial and returns the updated detail.
    pub fn unlock_editorial(&self, id: &str) -> Result<ProblemDetail> {
        let catalog = self.catalog();
        let problem = catalog.problem(id)?;
        self.inner
            .store
            .unlock_editorial(&problem.id, &review::format_time(self.inner.clock.now()))?;
        self.detail(problem)
    }

    /// `get_reference`: the reference solution; only once the editorial is unlocked.
    pub fn get_reference(&self, id: &str, framework: Framework) -> Result<String> {
        let catalog = self.catalog();
        let problem = catalog.problem(id)?;
        problem.require_framework(framework)?;
        let solved = self
            .inner
            .store
            .submission_progress()?
            .get(&problem.id)
            .is_some_and(|p| !p.solved_frameworks.is_empty());
        if !solved && !self.inner.store.editorial_revealed(&problem.id)? {
            return Err(CoreError::EditorialLocked(problem.id.clone()));
        }
        problem.reference(framework)
    }

    /// `env_status`: uv, Python, and framework versions.
    pub fn env_status(&self) -> Result<EnvStatus> {
        self.inner.env.status(&self.inner.root)
    }

    /// `env_install`: installs a framework. Returns once the install has started; events
    /// flow to `sink` and always finish with exactly one `done` event.
    pub fn env_install(
        &self,
        framework: Framework,
        sink: impl EventSink<InstallEvent>,
    ) -> Result<()> {
        // An install is an exact sync; it must not run under a download that is
        // installing the `models` extra at the same moment.
        if self.inner.models.is_fetching() {
            return Err(CoreError::FetchInProgress);
        }
        self.inner.env.install(framework.extra(), Arc::new(sink))
    }

    /// The extras known to be installed (passed as `--extra` to every uv command).
    pub fn enabled_extras(&self) -> Result<Vec<Extra>> {
        self.inner.env.enabled_extras()
    }

    /// `list_assets`: every model asset the catalog declares, with its state on disk.
    pub fn list_assets(&self) -> Result<AssetList> {
        self.inner.models.list()
    }

    /// `fetch_assets`: starts downloading the named assets and returns the job id at
    /// once. Events flow to `sink` and always finish with exactly one `done` event.
    /// One fetch runs at a time; nothing is ever downloaded without this call.
    ///
    /// `target` is where the files should land: `None` or `local` is this machine; an
    /// SSH target runs the same fetch on that host.
    pub fn fetch_assets(
        &self,
        names: &[String],
        target: Option<&str>,
        sink: impl EventSink<AssetEvent>,
    ) -> Result<String> {
        let remote = match target.filter(|name| *name != LOCAL_TARGET) {
            None => None,
            Some(name) => {
                let (host, remote_dir) = self.inner.targets.remote(name)?;
                let catalog = self.catalog();
                let problem_dirs = catalog
                    .problems()
                    .iter()
                    .filter(|problem| problem.assets.iter().any(|asset| names.contains(asset)))
                    .map(|problem| problem.dir.clone())
                    .collect();
                Some(RemoteFetch {
                    ssh: self.inner.targets.ssh().clone(),
                    host,
                    remote_dir,
                    problem_dirs,
                })
            }
        };
        self.inner.models.fetch(names, remote, Arc::new(sink))
    }

    /// `list_ssh_hosts`: the host aliases in the user's SSH configuration.
    pub fn list_ssh_hosts(&self) -> Vec<SshHost> {
        self.inner.targets.hosts()
    }

    /// `list_targets`: `local` first, then the SSH targets.
    pub fn list_targets(&self) -> Result<Vec<TargetInfo>> {
        self.inner.targets.list()
    }

    /// `add_target`: registers an SSH host from the user's configuration as a target.
    pub fn add_target(&self, host: &str, remote_dir: Option<&str>) -> Result<Vec<TargetInfo>> {
        self.inner.targets.add(host, remote_dir)
    }

    /// `remove_target`: forgets an SSH target; nothing is deleted on the remote host.
    pub fn remove_target(&self, name: &str) -> Result<Vec<TargetInfo>> {
        self.inner.targets.remove(name)
    }

    /// `check_target`: starts checking a target and returns the job id at once. Events
    /// flow to `sink` and always finish with exactly one `done` event; the result is
    /// saved as the target's `lastCheck`.
    pub fn check_target(
        &self,
        name: &str,
        sink: impl EventSink<TargetCheckEvent>,
    ) -> Result<String> {
        self.inner.targets.check(name, Arc::new(sink))
    }

    /// `cancel_asset_fetch`: kills the fetch's process group. Completed files are
    /// kept, so a later fetch resumes. An unknown or finished job is a no-op.
    pub fn cancel_asset_fetch(&self, job_id: &str) {
        self.inner.models.cancel(job_id);
    }

    /// `remove_asset`: deletes a declared asset's directory and returns the new
    /// listing. Fails while a fetch is running.
    pub fn remove_asset(&self, name: &str) -> Result<AssetList> {
        self.inner.models.remove(name)
    }

    /// `get_stats`: solved counts and the day streak.
    pub fn get_stats(&self) -> Result<Stats> {
        let summaries = self.list_problems()?;
        let mut by_difficulty: BTreeMap<Difficulty, (u32, u32)> =
            Difficulty::ALL.iter().map(|&d| (d, (0, 0))).collect();
        let mut solved = 0;
        for summary in &summaries {
            let is_solved = u32::from(summary.status == ProblemStatus::Solved);
            solved += is_solved;
            let entry = by_difficulty.entry(summary.difficulty).or_default();
            entry.0 += is_solved;
            entry.1 += 1;
        }
        let accepted = self.inner.store.accepted_times()?;
        let reviews_due = self.list_reviews()?.iter().filter(|r| r.due).count();
        Ok(Stats {
            solved,
            total: u32::try_from(summaries.len()).unwrap_or(u32::MAX),
            by_difficulty,
            streak_days: current_streak(accepted.iter().map(String::as_str)),
            reviews_due: u32::try_from(reviews_due).unwrap_or(u32::MAX),
        })
    }
}
