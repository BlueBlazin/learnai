//! Study paths: curated, ordered study plans at `paths/<id>.toml` in the content root.
//!
//! Loading follows the catalog's rule: it never fails as a whole. A malformed path file
//! is skipped and reported as a [`CatalogIssue`]. A problem id that the catalog does not
//! know (or that is listed twice in one path) is dropped from its stage and reported; it
//! never breaks the path.
//!
//! Path files are few and small, so they are read on every call rather than cached.

use std::collections::HashSet;
use std::fs;
use std::path::{Path, PathBuf};

use serde::Deserialize;

use crate::catalog::{Catalog, CatalogIssue, is_slug, normalize_hex_color};

/// One stage of a path: a titled group of problems in study order.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StageMeta {
    /// Display title.
    pub title: String,
    /// One-sentence description.
    pub description: String,
    /// Ids of problems that exist in the catalog, in study order.
    pub problems: Vec<String>,
    /// Slug of the bundled illustration, if any. The core does not know the images.
    pub scene: Option<String>,
    /// The stage's color as lowercase `#rrggbb`, if any.
    pub color: Option<String>,
}

/// A validated path.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PathMeta {
    /// Path id; equals the file stem.
    pub id: String,
    /// Display title.
    pub title: String,
    /// One-sentence description.
    pub description: String,
    /// Stages in order. A stage whose problems were all skipped is dropped.
    pub stages: Vec<StageMeta>,
}

impl PathMeta {
    /// Every problem id on the path, in study order.
    pub fn problem_ids(&self) -> impl Iterator<Item = &str> {
        self.stages
            .iter()
            .flat_map(|stage| stage.problems.iter().map(String::as_str))
    }
}

#[derive(Deserialize)]
struct PathToml {
    id: String,
    title: String,
    #[serde(default)]
    description: String,
    #[serde(default)]
    stages: Vec<StageToml>,
}

#[derive(Deserialize)]
struct StageToml {
    title: String,
    #[serde(default)]
    description: String,
    #[serde(default)]
    problems: Vec<String>,
    #[serde(default)]
    scene: Option<String>,
    #[serde(default)]
    color: Option<String>,
}

/// Loads every `paths/*.toml` under `root`, sorted by file name, validated against
/// `catalog`. Returns the usable paths and everything that was skipped.
pub fn load_paths(root: &Path, catalog: &Catalog) -> (Vec<PathMeta>, Vec<CatalogIssue>) {
    let mut paths = Vec::new();
    let mut issues = Vec::new();
    let mut seen_ids = HashSet::new();

    for file in path_files(&root.join("paths"), &mut issues) {
        let parsed = fs::read_to_string(&file)
            .map_err(|e| format!("unreadable: {e}"))
            .and_then(|text| parse_path(&text, &file, catalog, &mut issues));
        match parsed {
            Ok(path) => {
                if seen_ids.insert(path.id.clone()) {
                    paths.push(path);
                } else {
                    issues.push(issue(&file, format!("duplicate path id `{}`", path.id)));
                }
            }
            Err(message) => issues.push(issue(&file, message)),
        }
    }
    (paths, issues)
}

pub(crate) fn issue(path: &Path, message: String) -> CatalogIssue {
    CatalogIssue {
        path: path.to_path_buf(),
        message,
    }
}

/// Lists the visible `.toml` files of `dir`, sorted by name. A missing `dir` yields nothing.
pub(crate) fn path_files(dir: &Path, issues: &mut Vec<CatalogIssue>) -> Vec<PathBuf> {
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

/// Parses and validates one path file. Problem ids that cannot be used are dropped and
/// reported through `issues`; structural problems reject the whole file.
fn parse_path(
    text: &str,
    file: &Path,
    catalog: &Catalog,
    issues: &mut Vec<CatalogIssue>,
) -> Result<PathMeta, String> {
    let raw: PathToml = toml::from_str(text).map_err(|e| e.message().to_owned())?;
    if raw.id != file_stem(file) {
        return Err(format!(
            "id `{}` does not equal the file name `{}`",
            raw.id,
            file_stem(file)
        ));
    }
    if !is_slug(&raw.id) {
        return Err(format!(
            "id `{}` must use only a-z, 0-9, `-`, and `_`",
            raw.id
        ));
    }
    if raw.title.trim().is_empty() {
        return Err("title is empty".to_owned());
    }
    if raw.stages.is_empty() {
        return Err("a path needs at least one [[stages]] table".to_owned());
    }

    let mut listed = HashSet::new();
    let mut stages = Vec::with_capacity(raw.stages.len());
    for (index, stage) in raw.stages.into_iter().enumerate() {
        if stage.title.trim().is_empty() {
            return Err(format!("stage {} has an empty title", index + 1));
        }
        let mut problems = Vec::with_capacity(stage.problems.len());
        for id in stage.problems {
            if catalog.problem(&id).is_err() {
                issues.push(issue(
                    file,
                    format!(
                        "stage `{}` lists `{id}`, which is not a known problem; skipped",
                        stage.title
                    ),
                ));
            } else if !listed.insert(id.clone()) {
                issues.push(issue(
                    file,
                    format!(
                        "stage `{}` lists `{id}` again; a problem appears once in a path",
                        stage.title
                    ),
                ));
            } else {
                problems.push(id);
            }
        }
        let scene = stage.scene.filter(|scene| {
            let ok = is_slug(scene);
            if !ok {
                issues.push(issue(
                    file,
                    format!(
                        "stage `{}` has scene `{scene}`, which must use only a-z, 0-9, `-`, and `_`; ignored",
                        stage.title
                    ),
                ));
            }
            ok
        });
        let color = stage.color.and_then(|color| {
            let normalized = normalize_hex_color(&color);
            if normalized.is_none() {
                issues.push(issue(
                    file,
                    format!(
                        "stage `{}` has color `{color}`, which must be `#` and six hex digits; ignored",
                        stage.title
                    ),
                ));
            }
            normalized
        });
        if !problems.is_empty() {
            stages.push(StageMeta {
                title: stage.title,
                description: stage.description,
                problems,
                scene,
                color,
            });
        }
    }

    Ok(PathMeta {
        id: raw.id,
        title: raw.title,
        description: raw.description,
        stages,
    })
}
