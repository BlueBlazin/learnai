//! Areas: subjects, each an ordered list of modules (tracks), at `areas/<id>.toml` in the
//! content root.
//!
//! Loading follows the catalog's rule: it never fails as a whole. A malformed area file
//! is skipped and reported as a [`CatalogIssue`]. A module id that is not a known track
//! is dropped and reported. A track listed by several areas stays with the first area
//! (file-name order); the later listings are dropped and reported. An area left with no
//! modules is dropped and reported.
//!
//! Area files are few and small, so they are read on every call rather than cached.

use std::collections::{HashMap, HashSet};
use std::fs;
use std::path::Path;

use serde::Deserialize;

use crate::catalog::{Catalog, CatalogIssue, is_slug};
use crate::paths::{file_stem, issue, path_files};

/// A validated area.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AreaMeta {
    /// Area id; equals the file stem.
    pub id: String,
    /// Display title.
    pub title: String,
    /// One-sentence description.
    pub description: String,
    /// Ids of existing tracks, in order. Never empty; no track is in two areas.
    pub modules: Vec<String>,
}

#[derive(Deserialize)]
struct AreaToml {
    id: String,
    title: String,
    #[serde(default)]
    description: String,
    #[serde(default)]
    modules: Vec<String>,
}

/// Loads every `areas/*.toml` under `root`, sorted by file name, validated against
/// `catalog`. Returns the usable areas and everything that was skipped.
pub fn load_areas(root: &Path, catalog: &Catalog) -> (Vec<AreaMeta>, Vec<CatalogIssue>) {
    let mut areas = Vec::new();
    let mut issues = Vec::new();
    let mut seen_ids = HashSet::new();
    // Track id to the area that owns it.
    let mut owners: HashMap<String, String> = HashMap::new();

    for file in path_files(&root.join("areas"), &mut issues) {
        let parsed = fs::read_to_string(&file)
            .map_err(|e| format!("unreadable: {e}"))
            .and_then(|text| parse_area(&text, &file, catalog, &mut owners, &mut issues));
        match parsed {
            Ok(area) => {
                if seen_ids.insert(area.id.clone()) {
                    areas.push(area);
                } else {
                    issues.push(issue(&file, format!("duplicate area id `{}`", area.id)));
                }
            }
            Err(message) => issues.push(issue(&file, message)),
        }
    }
    (areas, issues)
}

/// Parses and validates one area file. Modules that cannot be used are dropped and
/// reported through `issues`; structural problems reject the whole file.
fn parse_area(
    text: &str,
    file: &Path,
    catalog: &Catalog,
    owners: &mut HashMap<String, String>,
    issues: &mut Vec<CatalogIssue>,
) -> Result<AreaMeta, String> {
    let raw: AreaToml = toml::from_str(text).map_err(|e| e.message().to_owned())?;
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

    let mut modules = Vec::with_capacity(raw.modules.len());
    let mut claimed = Vec::new();
    for id in raw.modules {
        if !catalog.tracks().iter().any(|track| track.id == id) {
            issues.push(issue(
                file,
                format!("lists module `{id}`, which is not a known track; skipped"),
            ));
        } else if let Some(owner) = owners.get(&id) {
            issues.push(issue(
                file,
                format!("lists module `{id}`, which already belongs to area `{owner}`; skipped"),
            ));
        } else if modules.contains(&id) {
            issues.push(issue(
                file,
                format!("lists module `{id}` again; a module appears once in an area"),
            ));
        } else {
            claimed.push(id.clone());
            modules.push(id);
        }
    }
    if modules.is_empty() {
        return Err("no usable modules; area dropped".to_owned());
    }
    // Only an area that survives claims its tracks.
    for id in claimed {
        owners.insert(id, raw.id.clone());
    }

    Ok(AreaMeta {
        id: raw.id,
        title: raw.title,
        description: raw.description,
        modules,
    })
}
