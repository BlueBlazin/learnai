//! The problem catalog: `problems/<track>/track.toml` and `problems/<track>/<id>/`.
//!
//! Loading never fails as a whole. A malformed track or problem is skipped and reported
//! as a [`CatalogIssue`]; everything else stays usable.
//!
//! Reloading is cheap. [`CatalogLoader`] caches each parsed TOML file keyed by its
//! modification time and size, so a refresh costs one directory walk plus a few `stat`
//! calls per problem, and only files that changed are read and parsed again. Statements,
//! hints, and editorials are read on demand and never cached.

use std::collections::{HashMap, HashSet};
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::SystemTime;

use serde::Deserialize;
use serde::de::DeserializeOwned;

use crate::error::{CoreError, Result};
use crate::model::{Difficulty, Framework, JudgeKind, Requires};

/// Static metadata of a track.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TrackMeta {
    /// Track id; equals the directory name.
    pub id: String,
    /// Display title.
    pub title: String,
    /// Short description.
    pub description: String,
    /// Position among tracks.
    pub order: i64,
    /// Slug of the bundled illustration, if any. The core does not know the images.
    pub scene: Option<String>,
    /// The module's color as lowercase `#rrggbb`, if any.
    pub color: Option<String>,
    /// Ids of this track's core problems, in teaching order. Every id is a valid
    /// problem of this track and appears once.
    pub core: Vec<String>,
}

/// Static metadata of a valid problem.
#[derive(Debug, Clone, PartialEq)]
pub struct ProblemMeta {
    /// Globally unique id; equals the directory name.
    pub id: String,
    /// Display title.
    pub title: String,
    /// Id of the owning track.
    pub track: String,
    /// Position within the track.
    pub order: i64,
    /// Difficulty.
    pub difficulty: Difficulty,
    /// Level within the difficulty, 1 to 3.
    pub level: u8,
    /// Frameworks a solution may be written in.
    pub frameworks: Vec<Framework>,
    /// Judges the problem uses.
    pub judges: Vec<JudgeKind>,
    /// Free-form tags.
    pub tags: Vec<String>,
    /// Ids of prerequisite problems.
    pub prerequisites: Vec<String>,
    /// Hardware requirement.
    pub requires: Requires,
    /// Names the solution must define.
    pub entry: Vec<String>,
    /// Default per-case time limit in seconds.
    pub time_limit_s: f64,
    /// Names of the model assets the problem declares (`[[assets]]`), in file order.
    pub assets: Vec<String>,
    /// Absolute problem directory.
    pub dir: PathBuf,
}

impl ProblemMeta {
    /// Path of the starter for `framework`.
    pub fn starter_path(&self, framework: Framework) -> PathBuf {
        self.dir.join("starter").join(format!("{framework}.py"))
    }

    /// Path of the reference solution for `framework`.
    pub fn reference_path(&self, framework: Framework) -> PathBuf {
        self.dir.join("reference").join(format!("{framework}.py"))
    }

    /// Fails unless the problem supports `framework`.
    pub fn require_framework(&self, framework: Framework) -> Result<()> {
        if self.frameworks.contains(&framework) {
            Ok(())
        } else {
            Err(CoreError::FrameworkNotSupported {
                problem: self.id.clone(),
                framework,
            })
        }
    }

    /// Reads the statement Markdown.
    pub fn statement(&self) -> Result<String> {
        read_required(&self.dir.join("statement.md"))
    }

    /// Reads the hints, split on lines containing only `---`. Missing file: no hints.
    pub fn hints(&self) -> Result<Vec<String>> {
        Ok(read_optional(&self.dir.join("hints.md"))?
            .as_deref()
            .map(split_hints)
            .unwrap_or_default())
    }

    /// Reads the editorial Markdown. Missing file: empty text.
    pub fn editorial(&self) -> Result<String> {
        Ok(read_optional(&self.dir.join("editorial.md"))?.unwrap_or_default())
    }

    /// Reads the starter for `framework`.
    pub fn starter(&self, framework: Framework) -> Result<String> {
        self.require_framework(framework)?;
        read_required(&self.starter_path(framework))
    }

    /// Reads the reference solution for `framework`.
    pub fn reference(&self, framework: Framework) -> Result<String> {
        self.require_framework(framework)?;
        read_required(&self.reference_path(framework))
    }
}

/// Splits hints Markdown on lines containing only `---`, dropping empty hints.
pub fn split_hints(text: &str) -> Vec<String> {
    let mut hints = Vec::new();
    let mut current = String::new();
    for line in text.lines() {
        if line.trim() == "---" {
            hints.push(std::mem::take(&mut current));
        } else {
            current.push_str(line);
            current.push('\n');
        }
    }
    hints.push(current);
    hints
        .into_iter()
        .map(|h| h.trim().to_owned())
        .filter(|h| !h.is_empty())
        .collect()
}

/// Something wrong in the catalog that caused a track or problem to be skipped, or a
/// non-fatal inconsistency worth showing to an author.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CatalogIssue {
    /// The offending file or directory.
    pub path: PathBuf,
    /// What is wrong.
    pub message: String,
}

/// An immutable snapshot of the catalog.
#[derive(Debug, Default)]
pub struct Catalog {
    tracks: Vec<TrackMeta>,
    problems: Vec<Arc<ProblemMeta>>,
    index: HashMap<String, usize>,
    issues: Vec<CatalogIssue>,
}

impl Catalog {
    /// Tracks sorted by `order`, then id.
    pub fn tracks(&self) -> &[TrackMeta] {
        &self.tracks
    }

    /// Valid problems sorted by track order, then `order`, then id.
    pub fn problems(&self) -> &[Arc<ProblemMeta>] {
        &self.problems
    }

    /// Looks up a problem by id.
    pub fn problem(&self, id: &str) -> Result<&Arc<ProblemMeta>> {
        self.index
            .get(id)
            .and_then(|&i| self.problems.get(i))
            .ok_or_else(|| CoreError::ProblemNotFound(id.to_owned()))
    }

    /// Everything that was skipped or looked inconsistent.
    pub fn issues(&self) -> &[CatalogIssue] {
        &self.issues
    }
}

#[derive(Deserialize)]
struct TrackToml {
    id: String,
    title: String,
    #[serde(default)]
    description: String,
    #[serde(default)]
    order: i64,
    #[serde(default)]
    scene: Option<String>,
    #[serde(default)]
    color: Option<String>,
    #[serde(default)]
    core: Vec<String>,
}

/// A parsed track plus the non-fatal problems found while parsing it. Cached together
/// so a cached track reports the same issues as a fresh one.
#[derive(Clone)]
struct ParsedTrack {
    meta: TrackMeta,
    warnings: Vec<String>,
}

#[derive(Deserialize)]
struct ProblemToml {
    id: String,
    title: String,
    track: String,
    #[serde(default)]
    order: i64,
    difficulty: Difficulty,
    level: u8,
    frameworks: Vec<Framework>,
    judges: Vec<JudgeKind>,
    #[serde(default)]
    tags: Vec<String>,
    #[serde(default)]
    prerequisites: Vec<String>,
    #[serde(default = "default_requires")]
    requires: Requires,
    entry: Vec<String>,
    #[serde(default = "default_time_limit")]
    time_limit_s: f64,
    #[serde(default)]
    assets: Vec<AssetToml>,
}

/// One `[[assets]]` table. Core needs only the name; the Python side owns the rest
/// (repository, revision, files) and reports what is wrong with it.
#[derive(Deserialize)]
struct AssetToml {
    name: String,
}

fn default_requires() -> Requires {
    Requires::Cpu
}

fn default_time_limit() -> f64 {
    10.0
}

/// Identity of a file's contents, cheap to obtain.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Stamp {
    modified: Option<SystemTime>,
    len: u64,
}

struct Cached<T> {
    stamp: Stamp,
    parsed: std::result::Result<T, String>,
}

/// Loads the catalog and keeps per-file parse results between loads.
pub struct CatalogLoader {
    problems_dir: PathBuf,
    tracks: HashMap<PathBuf, Cached<ParsedTrack>>,
    problems: HashMap<PathBuf, Cached<Arc<ProblemMeta>>>,
    reported: HashSet<(PathBuf, String)>,
}

impl CatalogLoader {
    /// Creates a loader for `<root>/problems`.
    pub fn new(root: &Path) -> Self {
        Self {
            problems_dir: root.join("problems"),
            tracks: HashMap::new(),
            problems: HashMap::new(),
            reported: HashSet::new(),
        }
    }

    /// Returns a fresh snapshot, reparsing only files that changed since the last call.
    pub fn load(&mut self) -> Catalog {
        let mut catalog = Catalog::default();
        let mut seen_tracks = HashSet::new();
        let mut seen_problems = HashSet::new();
        let mut track_order = HashMap::new();

        for track_dir in subdirectories(&self.problems_dir, &mut catalog.issues) {
            let toml_path = track_dir.join("track.toml");
            seen_tracks.insert(toml_path.clone());
            let parsed = cached(&mut self.tracks, &toml_path, |text| {
                parse_track(text, &track_dir)
            });
            let track = match parsed {
                Ok(ParsedTrack { meta, warnings }) => {
                    for message in warnings {
                        catalog.issues.push(CatalogIssue {
                            path: toml_path.clone(),
                            message,
                        });
                    }
                    meta
                }
                Err(message) => {
                    catalog.issues.push(CatalogIssue {
                        path: toml_path,
                        message,
                    });
                    continue;
                }
            };
            track_order.insert(track.id.clone(), track.order);
            let track_index = catalog.tracks.len();
            catalog.tracks.push(track);
            let mut track_problems = HashSet::new();

            for problem_dir in subdirectories(&track_dir, &mut catalog.issues) {
                let toml_path = problem_dir.join("problem.toml");
                seen_problems.insert(toml_path.clone());
                let parsed = cached(&mut self.problems, &toml_path, |text| {
                    parse_problem(text, &problem_dir).map(Arc::new)
                })
                .and_then(|problem| check_files(&problem).map(|()| problem));
                match parsed {
                    Ok(problem) => {
                        track_problems.insert(problem.id.clone());
                        catalog.problems.push(problem);
                    }
                    Err(message) => catalog.issues.push(CatalogIssue {
                        path: toml_path,
                        message,
                    }),
                }
            }

            let track = &mut catalog.tracks[track_index];
            let track_toml = track_dir.join("track.toml");
            let mut listed = HashSet::new();
            let core = std::mem::take(&mut track.core);
            for id in core {
                let message = if !track_problems.contains(&id) {
                    format!(
                        "core lists `{id}`, which is not a problem of track `{}`; skipped",
                        track.id
                    )
                } else if !listed.insert(id.clone()) {
                    format!("core lists `{id}` again; a problem appears once in a core")
                } else {
                    track.core.push(id);
                    continue;
                };
                catalog.issues.push(CatalogIssue {
                    path: track_toml.clone(),
                    message,
                });
            }
        }

        self.tracks.retain(|path, _| seen_tracks.contains(path));
        self.problems.retain(|path, _| seen_problems.contains(path));

        finish(&mut catalog, &track_order);
        self.report_new_issues(&catalog.issues);
        catalog
    }

    /// Logs each distinct issue once, so a polling UI does not flood the log.
    fn report_new_issues(&mut self, issues: &[CatalogIssue]) {
        let current: HashSet<_> = issues
            .iter()
            .map(|i| (i.path.clone(), i.message.clone()))
            .collect();
        for (path, message) in current.difference(&self.reported) {
            tracing::warn!(path = %path.display(), "catalog issue: {message}");
        }
        self.reported = current;
    }
}

/// Sorts, removes duplicate ids, builds the index, and checks cross-references.
fn finish(catalog: &mut Catalog, track_order: &HashMap<String, i64>) {
    catalog
        .tracks
        .sort_by(|a, b| (a.order, &a.id).cmp(&(b.order, &b.id)));
    catalog.problems.sort_by(|a, b| {
        let key = |p: &ProblemMeta| {
            (
                track_order.get(&p.track).copied().unwrap_or(0),
                p.track.clone(),
            )
        };
        (key(a), a.order, &a.id).cmp(&(key(b), b.order, &b.id))
    });

    let mut index = HashMap::new();
    let mut unique = Vec::with_capacity(catalog.problems.len());
    for problem in catalog.problems.drain(..) {
        if index.contains_key(&problem.id) {
            catalog.issues.push(CatalogIssue {
                path: problem.dir.join("problem.toml"),
                message: format!(
                    "duplicate problem id `{}`; ids are globally unique",
                    problem.id
                ),
            });
        } else {
            index.insert(problem.id.clone(), unique.len());
            unique.push(problem);
        }
    }

    for problem in &unique {
        for prerequisite in &problem.prerequisites {
            if !index.contains_key(prerequisite) {
                catalog.issues.push(CatalogIssue {
                    path: problem.dir.join("problem.toml"),
                    message: format!("prerequisite `{prerequisite}` is not a known problem"),
                });
            }
        }
    }
    catalog.problems = unique;
    catalog.index = index;
}

/// Parses `path` through the cache.
fn cached<T: Clone>(
    cache: &mut HashMap<PathBuf, Cached<T>>,
    path: &Path,
    parse: impl FnOnce(&str) -> std::result::Result<T, String>,
) -> std::result::Result<T, String> {
    let stamp = match fs::metadata(path) {
        Ok(meta) => Stamp {
            modified: meta.modified().ok(),
            len: meta.len(),
        },
        Err(_) => {
            cache.remove(path);
            let name = path.file_name().and_then(|n| n.to_str()).unwrap_or("file");
            return Err(format!("missing {name}"));
        }
    };
    if let Some(entry) = cache.get(path)
        && entry.stamp == stamp
    {
        return entry.parsed.clone();
    }
    let parsed = fs::read_to_string(path)
        .map_err(|e| format!("unreadable: {e}"))
        .and_then(|t| parse(&t));
    cache.insert(
        path.to_path_buf(),
        Cached {
            stamp,
            parsed: parsed.clone(),
        },
    );
    parsed
}

fn parse_toml<T: DeserializeOwned>(text: &str) -> std::result::Result<T, String> {
    toml::from_str(text).map_err(|e| e.message().to_owned())
}

fn dir_name(dir: &Path) -> &str {
    dir.file_name().and_then(|n| n.to_str()).unwrap_or_default()
}

/// Parses a track. A bad `scene` or `color` is dropped with a warning; `core` is kept
/// as written and checked against the track's problems once those are loaded.
fn parse_track(text: &str, dir: &Path) -> std::result::Result<ParsedTrack, String> {
    let raw: TrackToml = parse_toml(text)?;
    if raw.id != dir_name(dir) {
        return Err(format!(
            "id `{}` does not equal the directory name `{}`",
            raw.id,
            dir_name(dir)
        ));
    }
    if raw.title.trim().is_empty() {
        return Err("title is empty".to_owned());
    }
    let mut warnings = Vec::new();
    let scene = raw.scene.filter(|scene| {
        let ok = is_slug(scene);
        if !ok {
            warnings.push(format!(
                "scene `{scene}` must use only a-z, 0-9, `-`, and `_`; ignored"
            ));
        }
        ok
    });
    let color = raw.color.and_then(|color| {
        let normalized = normalize_hex_color(&color);
        if normalized.is_none() {
            warnings.push(format!(
                "color `{color}` must be `#` and six hex digits; ignored"
            ));
        }
        normalized
    });
    Ok(ParsedTrack {
        meta: TrackMeta {
            id: raw.id,
            title: raw.title,
            description: raw.description,
            order: raw.order,
            scene,
            color,
            core: raw.core,
        },
        warnings,
    })
}

/// `#` followed by exactly six hex digits, in either case, returned in lowercase.
pub(crate) fn normalize_hex_color(value: &str) -> Option<String> {
    value
        .strip_prefix('#')
        .is_some_and(|hex| hex.len() == 6 && hex.bytes().all(|b| b.is_ascii_hexdigit()))
        .then(|| value.to_ascii_lowercase())
}

/// Whether `id` is a valid identifier: only a-z, 0-9, `-`, and `_`.
pub(crate) fn is_slug(id: &str) -> bool {
    !id.is_empty()
        && id
            .bytes()
            .all(|b| b.is_ascii_lowercase() || b.is_ascii_digit() || b == b'-' || b == b'_')
}

/// A [`CatalogIssue`] for `path`.
pub(crate) fn issue(path: &Path, message: String) -> CatalogIssue {
    CatalogIssue {
        path: path.to_path_buf(),
        message,
    }
}

/// Lists the visible `.toml` files of `dir`, sorted by name. A missing `dir` yields nothing.
pub(crate) fn toml_files(dir: &Path, issues: &mut Vec<CatalogIssue>) -> Vec<PathBuf> {
    let entries = match fs::read_dir(dir) {
        Ok(entries) => entries,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Vec::new(),
        Err(e) => {
            issues.push(issue(dir, format!("unreadable: {e}")));
            return Vec::new();
        }
    };
    let mut files: Vec<PathBuf> = entries
        .filter_map(|entry| entry.ok())
        .map(|entry| entry.path())
        .filter(|path| {
            path.is_file()
                && path.extension().is_some_and(|ext| ext == "toml")
                && !file_stem(path).starts_with(['.', '_'])
        })
        .collect();
    files.sort();
    files
}

pub(crate) fn file_stem(path: &Path) -> &str {
    path.file_stem()
        .and_then(|stem| stem.to_str())
        .unwrap_or_default()
}

fn parse_problem(text: &str, dir: &Path) -> std::result::Result<ProblemMeta, String> {
    let raw: ProblemToml = parse_toml(text)?;
    let track = dir.parent().map(dir_name).unwrap_or_default();
    if raw.id != dir_name(dir) {
        return Err(format!(
            "id `{}` does not equal the directory name `{}`",
            raw.id,
            dir_name(dir)
        ));
    }
    if !is_slug(&raw.id) {
        return Err(format!(
            "id `{}` must use only a-z, 0-9, `-`, and `_`",
            raw.id
        ));
    }
    if raw.track != track {
        return Err(format!(
            "track `{}` does not equal the parent directory name `{track}`",
            raw.track
        ));
    }
    if raw.title.trim().is_empty() {
        return Err("title is empty".to_owned());
    }
    if !(1..=3).contains(&raw.level) {
        return Err(format!("level is {}; it must be 1, 2, or 3", raw.level));
    }
    if raw.frameworks.is_empty() {
        return Err("frameworks is empty".to_owned());
    }
    if raw.judges.is_empty() {
        return Err("judges is empty".to_owned());
    }
    if raw.entry.is_empty() {
        return Err("entry is empty".to_owned());
    }
    if !(raw.time_limit_s.is_finite() && raw.time_limit_s > 0.0) {
        return Err("time_limit_s must be a positive number".to_owned());
    }
    Ok(ProblemMeta {
        id: raw.id,
        title: raw.title,
        track: raw.track,
        order: raw.order,
        difficulty: raw.difficulty,
        level: raw.level,
        frameworks: dedup(raw.frameworks),
        judges: dedup(raw.judges),
        tags: raw.tags,
        prerequisites: raw.prerequisites,
        requires: raw.requires,
        entry: raw.entry,
        time_limit_s: raw.time_limit_s,
        assets: dedup(raw.assets.into_iter().map(|asset| asset.name).collect()),
        dir: dir.to_path_buf(),
    })
}

fn dedup<T: PartialEq>(items: Vec<T>) -> Vec<T> {
    let mut out = Vec::with_capacity(items.len());
    for item in items {
        if !out.contains(&item) {
            out.push(item);
        }
    }
    out
}

/// Checks the files core itself depends on. Not cached: authors add files without
/// touching `problem.toml`.
fn check_files(problem: &ProblemMeta) -> std::result::Result<(), String> {
    if !problem.dir.join("statement.md").is_file() {
        return Err("missing statement.md".to_owned());
    }
    for &framework in &problem.frameworks {
        if !problem.starter_path(framework).is_file() {
            return Err(format!("missing starter/{framework}.py"));
        }
    }
    Ok(())
}

/// Lists the visible subdirectories of `dir`, sorted by name. Hidden and underscore
/// directories are ignored. A missing `dir` yields nothing.
fn subdirectories(dir: &Path, issues: &mut Vec<CatalogIssue>) -> Vec<PathBuf> {
    let entries = match fs::read_dir(dir) {
        Ok(entries) => entries,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Vec::new(),
        Err(e) => {
            issues.push(CatalogIssue {
                path: dir.to_path_buf(),
                message: format!("unreadable: {e}"),
            });
            return Vec::new();
        }
    };
    let mut dirs: Vec<PathBuf> = entries
        .filter_map(|entry| entry.ok())
        .map(|entry| entry.path())
        .filter(|path| path.is_dir() && !dir_name(path).starts_with(['.', '_']))
        .collect();
    dirs.sort();
    dirs
}

fn read_required(path: &Path) -> Result<String> {
    read_optional(path)?.ok_or_else(|| CoreError::MissingFile(path.to_path_buf()))
}

fn read_optional(path: &Path) -> Result<Option<String>> {
    match fs::read_to_string(path) {
        Ok(text) => Ok(Some(text)),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(None),
        Err(e) => Err(CoreError::io("read", path, e)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hints_split_on_rule_lines_only() {
        let text = "First hint.\n\n---\nSecond --- still second\nline two\n  ---  \n\n---\nThird\n";
        assert_eq!(
            split_hints(text),
            vec!["First hint.", "Second --- still second\nline two", "Third"]
        );
        assert!(split_hints("\n---\n").is_empty());
    }

    const BASE: &str = r#"
id = "asset-vocab"
title = "Read a Vocabulary Size"
track = "real"
difficulty = "easy"
level = 1
frameworks = ["python"]
judges = ["numeric"]
entry = ["vocab_size"]
"#;

    #[test]
    fn level_must_be_one_two_or_three() {
        let dir = Path::new("/r/problems/real/asset-vocab");
        assert_eq!(parse_problem(BASE, dir).expect("problem").level, 1);
        for bad in ["level = 0", "level = 4"] {
            let text = BASE.replace("level = 1", bad);
            let error = parse_problem(&text, dir).expect_err(bad);
            assert!(error.contains("it must be 1, 2, or 3"), "{error}");
        }
        let missing = BASE.replace("level = 1\n", "");
        let error = parse_problem(&missing, dir).expect_err("missing level");
        assert!(error.contains("level"), "{error}");
    }

    #[test]
    fn declared_assets_are_read_by_name() {
        let dir = Path::new("/r/problems/real/asset-vocab");
        assert!(parse_problem(BASE, dir).expect("problem").assets.is_empty());

        let text = format!(
            r#"{BASE}
[[assets]]
name = "granite-4.0-350m"
repo = "ibm-granite/granite-4.0-350m"
revision = "bd8a1497065c0d6ba1ef19af6b0d2b14bacf71c2"
files = ["config.json", "*.safetensors"]
size_mb = 712
a_key_from_the_future = true

[[assets]]
name = "tiny-test-model"

[[assets]]
name = "granite-4.0-350m"
"#
        );
        assert_eq!(
            parse_problem(&text, dir).expect("problem").assets,
            ["granite-4.0-350m", "tiny-test-model"]
        );
    }
}
