//! The [`Core`] facade: one method per command in `docs/ARCHITECTURE.md` section 4.

use std::collections::{BTreeMap, HashMap};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, PoisonError};

use crate::catalog::{Catalog, CatalogIssue, CatalogLoader, ProblemMeta};
use crate::env::Environment;
use crate::error::{CoreError, Result};
use crate::harness::{HarnessCommand, HarnessJob, UvHarness};
use crate::model::{
    Difficulty, EnvStatus, Framework, InstallEvent, ProblemDetail, ProblemStatus, ProblemSummary,
    RunEvent, RunMode, SolutionFile, Stats, Submission, SubmissionDetail, Track,
};
use crate::progress::{current_streak, derive_status};
use crate::run::{RunLimits, RunPlan, RunRegistry};
use crate::sink::EventSink;
use crate::solutions::Workspace;
use crate::store::{Store, SubmissionProgress};
use crate::target::{LocalTarget, Target};
use crate::uv::Uv;

/// How to construct a [`Core`].
pub struct CoreOptions {
    /// The content root (see [`crate::root::resolve_root`]).
    pub root: PathBuf,
    /// Overrides the harness command; `None` uses `uv run ... python -m whetstone.harness`.
    pub harness: Option<Arc<dyn HarnessCommand>>,
    /// Overrides the uv executable; `None` searches for it.
    pub uv_program: Option<PathBuf>,
    /// Wall-clock limits for runs.
    pub limits: RunLimits,
}

impl CoreOptions {
    /// Default options for a content root.
    pub fn new(root: impl Into<PathBuf>) -> Self {
        Self {
            root: root.into(),
            harness: None,
            uv_program: None,
            limits: RunLimits::default(),
        }
    }
}

struct Inner {
    root: PathBuf,
    catalog: Mutex<CatalogLoader>,
    workspace: Workspace,
    store: Arc<Store>,
    env: Environment,
    targets: HashMap<String, Arc<dyn Target>>,
    runs: Arc<RunRegistry>,
    limits: RunLimits,
}

/// The application core. Cheap to clone; all clones share state.
#[derive(Clone)]
pub struct Core {
    inner: Arc<Inner>,
}

impl Core {
    /// Opens the core for a content root, creating `.whetstone/state.db` if needed.
    pub fn open(options: CoreOptions) -> Result<Self> {
        let root = options.root;
        let store = Arc::new(Store::open(&root.join(".whetstone").join("state.db"))?);
        let uv = match options.uv_program {
            Some(program) => Uv::new(program, root.join("python")),
            None => Uv::locate(&root),
        };
        let harness = options
            .harness
            .unwrap_or_else(|| Arc::new(UvHarness::new(uv.clone())) as Arc<dyn HarnessCommand>);
        let local: Arc<dyn Target> = Arc::new(LocalTarget::new(harness));
        let targets = HashMap::from([(local.name().to_owned(), local)]);

        Ok(Self {
            inner: Arc::new(Inner {
                catalog: Mutex::new(CatalogLoader::new(&root)),
                workspace: Workspace::new(&root),
                env: Environment::new(uv, Arc::clone(&store)),
                store,
                targets,
                runs: Arc::new(RunRegistry::default()),
                limits: options.limits,
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

    /// Malformed tracks and problems found by the latest catalog scan.
    pub fn catalog_issues(&self) -> Vec<CatalogIssue> {
        self.catalog().issues().to_vec()
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
                }
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
        let catalog = self.catalog();
        let problem = catalog.problem(problem_id)?;
        let solution = self.inner.workspace.read(problem, framework)?;
        let target = self
            .inner
            .targets
            .get(target)
            .cloned()
            .ok_or_else(|| CoreError::UnknownTarget(target.to_owned()))?;

        // Judge the saved file when there is one, else the starter in place.
        let saved = self.inner.workspace.path(&problem.id, framework);
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
            },
            code: solution.code,
            target,
            limits: self.inner.limits,
            store: Arc::clone(&self.inner.store),
            sink: Arc::new(sink),
        };
        self.inner.runs.start(plan).map_err(|source| CoreError::Io {
            context: "failed to start the run thread".to_owned(),
            source,
        })
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
            .unlock_editorial(&problem.id, &crate::now_iso8601())?;
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
        self.inner.env.install(framework, Arc::new(sink))
    }

    /// The optional frameworks currently enabled (passed as `--extra` to every uv command).
    pub fn enabled_extras(&self) -> Result<Vec<Framework>> {
        self.inner.env.enabled_extras()
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
        Ok(Stats {
            solved,
            total: u32::try_from(summaries.len()).unwrap_or(u32::MAX),
            by_difficulty,
            streak_days: current_streak(accepted.iter().map(String::as_str)),
        })
    }
}
