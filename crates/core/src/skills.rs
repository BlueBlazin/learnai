//! Skills: curated groups of problems, each optionally building on other skills, at
//! `skills/<id>.toml` in the content root. A skill is earned once all of its problems
//! are solved and every skill it requires is earned.
//!
//! Loading follows the catalog's rule: it never fails as a whole. A malformed skill file
//! is skipped and reported as a [`CatalogIssue`]. A problem id that is not a known
//! problem is dropped and reported, and a skill left with no problems is dropped and
//! reported. A requirement that names no usable skill, repeats, or would close a cycle
//! is dropped and reported, so the skills that load always form a DAG.
//!
//! Skill files are few and small, so they are read on every call rather than cached.

use std::collections::{HashMap, HashSet};
use std::fs;
use std::path::{Path, PathBuf};

use chrono::{DateTime, FixedOffset};
use serde::Deserialize;

use crate::catalog::{
    Catalog, CatalogIssue, file_stem, is_slug, issue, normalize_hex_color, toml_files,
};

/// A validated skill.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SkillMeta {
    /// Skill id; equals the file stem.
    pub id: String,
    /// The name the skill goes by.
    pub title: String,
    /// The plain name of what it covers.
    pub name: String,
    /// One-sentence description.
    pub summary: String,
    /// Slug of the bundled graphic, if any.
    pub art: Option<String>,
    /// The skill's color as lowercase `#rrggbb`, if any.
    pub color: Option<String>,
    /// Ids of the usable skills this one builds on. Never part of a cycle.
    pub requires: Vec<String>,
    /// Ids of existing problems, in order. Never empty.
    pub problems: Vec<String>,
    /// Row on the Accomplishments tree, top to bottom from 0.
    pub row: u32,
    /// Horizontal center in px on a 1000 px wide canvas.
    pub x: u32,
    /// `[row, x]` points where this skill's trails pass a row on the way down.
    pub lanes: Vec<[u32; 2]>,
}

#[derive(Deserialize)]
struct SkillToml {
    id: String,
    title: String,
    name: String,
    #[serde(default)]
    summary: String,
    #[serde(default)]
    art: Option<String>,
    #[serde(default)]
    color: Option<String>,
    #[serde(default)]
    requires: Vec<String>,
    #[serde(default)]
    problems: Vec<String>,
    row: u32,
    x: u32,
    #[serde(default)]
    lanes: Vec<toml::Value>,
}

/// A skill whose own file is valid, before its requirements are checked against the
/// other skills.
struct Candidate {
    file: PathBuf,
    skill: SkillMeta,
    /// `requires` as written.
    requires: Vec<String>,
}

/// Loads every `skills/*.toml` under `root`, sorted by file name, validated against
/// `catalog`. Returns the usable skills and everything that was skipped.
pub fn load_skills(root: &Path, catalog: &Catalog) -> (Vec<SkillMeta>, Vec<CatalogIssue>) {
    let mut candidates: Vec<Candidate> = Vec::new();
    let mut issues = Vec::new();
    let mut seen_ids = HashSet::new();

    for file in toml_files(&root.join("skills"), &mut issues) {
        let parsed = fs::read_to_string(&file)
            .map_err(|e| format!("unreadable: {e}"))
            .and_then(|text| parse_skill(&text, &file, catalog, &mut issues));
        match parsed {
            Ok(candidate) => {
                if seen_ids.insert(candidate.skill.id.clone()) {
                    candidates.push(candidate);
                } else {
                    issues.push(issue(
                        &file,
                        format!("duplicate skill id `{}`", candidate.skill.id),
                    ));
                }
            }
            Err(message) => issues.push(issue(&file, message)),
        }
    }

    // Requirements are checked once every skill is known: a skill may build on one
    // whose file sorts after its own.
    let rows: HashMap<String, u32> = candidates
        .iter()
        .map(|candidate| (candidate.skill.id.clone(), candidate.skill.row))
        .collect();
    // Skill id to the requirements accepted for it so far.
    let mut accepted: HashMap<String, Vec<String>> = HashMap::new();
    for candidate in &mut candidates {
        let (file, id, row) = (&candidate.file, &candidate.skill.id, candidate.skill.row);
        let mut requires: Vec<String> = Vec::with_capacity(candidate.requires.len());
        for required in std::mem::take(&mut candidate.requires) {
            if required == *id {
                issues.push(issue(file, "requires itself; skipped".to_owned()));
            } else if !rows.contains_key(&required) {
                issues.push(issue(
                    file,
                    format!("requires `{required}`, which is not a usable skill; skipped"),
                ));
            } else if requires.contains(&required) {
                issues.push(issue(
                    file,
                    format!("requires `{required}` again; a skill is required once"),
                ));
            } else if depends_on(&accepted, &required, id) {
                issues.push(issue(
                    file,
                    format!("requires `{required}`, which already builds on `{id}`; skipped"),
                ));
            } else {
                // Reported and kept: earning must not depend on the layout.
                if rows[&required] >= row {
                    issues.push(issue(
                        file,
                        format!(
                            "requires `{required}`, which is not on a higher row (row {} against row {row})",
                            rows[&required]
                        ),
                    ));
                }
                requires.push(required);
            }
        }
        accepted.insert(id.clone(), requires.clone());
        candidate.skill.requires = requires;
    }

    let skills = candidates
        .into_iter()
        .map(|candidate| candidate.skill)
        .collect();
    (skills, issues)
}

/// Whether `from` builds on `target`, directly or through other skills, by the
/// requirements in `edges`.
fn depends_on(edges: &HashMap<String, Vec<String>>, from: &str, target: &str) -> bool {
    let mut seen = HashSet::new();
    let mut stack = vec![from];
    while let Some(id) = stack.pop() {
        if id == target {
            return true;
        }
        if seen.insert(id) {
            stack.extend(edges.get(id).into_iter().flatten().map(String::as_str));
        }
    }
    false
}

/// Parses and validates one skill file, except for its requirements. Fields that cannot
/// be used are dropped and reported through `issues`; structural problems reject the
/// whole file.
fn parse_skill(
    text: &str,
    file: &Path,
    catalog: &Catalog,
    issues: &mut Vec<CatalogIssue>,
) -> Result<Candidate, String> {
    let raw: SkillToml = toml::from_str(text).map_err(|e| e.message().to_owned())?;
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
    if raw.name.trim().is_empty() {
        return Err("name is empty".to_owned());
    }

    let mut problems = Vec::with_capacity(raw.problems.len());
    for id in raw.problems {
        if catalog.problem(&id).is_err() {
            issues.push(issue(
                file,
                format!("lists problem `{id}`, which is not a known problem; skipped"),
            ));
        } else if problems.contains(&id) {
            issues.push(issue(
                file,
                format!("lists problem `{id}` again; a problem appears once in a skill"),
            ));
        } else {
            problems.push(id);
        }
    }
    if problems.is_empty() {
        return Err("no usable problems; skill dropped".to_owned());
    }

    let art = raw.art.filter(|art| {
        let ok = is_slug(art);
        if !ok {
            issues.push(issue(
                file,
                format!("art `{art}` must use only a-z, 0-9, `-`, and `_`; ignored"),
            ));
        }
        ok
    });
    let color = raw.color.and_then(|color| {
        let normalized = normalize_hex_color(&color);
        if normalized.is_none() {
            issues.push(issue(
                file,
                format!("color `{color}` must be `#` and six hex digits; ignored"),
            ));
        }
        normalized
    });
    let mut lanes = Vec::with_capacity(raw.lanes.len());
    for lane in raw.lanes {
        match parse_lane(&lane) {
            Some(point) => lanes.push(point),
            None => issues.push(issue(
                file,
                format!("lane `{lane}` must be `[row, x]` of non-negative integers; skipped"),
            )),
        }
    }

    Ok(Candidate {
        file: file.to_path_buf(),
        skill: SkillMeta {
            id: raw.id,
            title: raw.title,
            name: raw.name,
            summary: raw.summary,
            art,
            color,
            requires: Vec::new(),
            problems,
            row: raw.row,
            x: raw.x,
            lanes,
        },
        requires: raw.requires,
    })
}

/// A lane entry: exactly `[row, x]`, both non-negative integers.
fn parse_lane(value: &toml::Value) -> Option<[u32; 2]> {
    let number = |value: &toml::Value| u32::try_from(value.as_integer()?).ok();
    match value.as_array()?.as_slice() {
        [row, x] => Some([number(row)?, number(x)?]),
        _ => None,
    }
}

/// When each skill was earned, by skill id; `None` while it is not.
///
/// `first_accepted` maps a problem id to the time of its first accepted submit. A skill
/// is unearned while any of its problems has none or any skill it requires is unearned.
/// Otherwise it was earned at the latest of its problems' first accepted submits and
/// its required skills' earned times. Times are compared as parsed ISO 8601 and
/// returned as the stored string.
pub fn earned_times(
    skills: &[SkillMeta],
    first_accepted: &HashMap<String, String>,
) -> HashMap<String, Option<String>> {
    let by_id: HashMap<&str, &SkillMeta> = skills
        .iter()
        .map(|skill| (skill.id.as_str(), skill))
        .collect();
    let mut earned = HashMap::with_capacity(skills.len());
    for skill in skills {
        resolve_earned(skill, &by_id, first_accepted, &mut earned);
    }
    earned
}

fn resolve_earned(
    skill: &SkillMeta,
    by_id: &HashMap<&str, &SkillMeta>,
    first_accepted: &HashMap<String, String>,
    earned: &mut HashMap<String, Option<String>>,
) -> Option<String> {
    if let Some(known) = earned.get(&skill.id) {
        return known.clone();
    }
    // Unearned while it is being resolved, so that a cycle (which `load_skills` never
    // produces) ends instead of recursing forever.
    earned.insert(skill.id.clone(), None);
    let mut times = Vec::with_capacity(skill.problems.len() + skill.requires.len());
    for problem in &skill.problems {
        times.push(first_accepted.get(problem)?.clone());
    }
    for required in &skill.requires {
        times.push(resolve_earned(
            by_id.get(required.as_str())?,
            by_id,
            first_accepted,
            earned,
        )?);
    }
    let latest = times.into_iter().max_by_key(|time| parse_time(time))?;
    earned.insert(skill.id.clone(), Some(latest.clone()));
    Some(latest)
}

/// A stored time as a comparable value; an unparsable one sorts before every time.
fn parse_time(time: &str) -> Option<DateTime<FixedOffset>> {
    DateTime::parse_from_rfc3339(time).ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn skill(id: &str, requires: &[&str], problems: &[&str]) -> SkillMeta {
        SkillMeta {
            id: id.to_owned(),
            title: id.to_owned(),
            name: id.to_owned(),
            summary: String::new(),
            art: None,
            color: None,
            requires: requires.iter().map(|&s| s.to_owned()).collect(),
            problems: problems.iter().map(|&s| s.to_owned()).collect(),
            row: 0,
            x: 0,
            lanes: Vec::new(),
        }
    }

    fn accepted(times: &[(&str, &str)]) -> HashMap<String, String> {
        times
            .iter()
            .map(|&(problem, time)| (problem.to_owned(), time.to_owned()))
            .collect()
    }

    #[test]
    fn a_skill_is_earned_at_its_last_first_solve() {
        let skills = [skill("a", &[], &["p1", "p2"])];
        let earned = earned_times(
            &skills,
            &accepted(&[
                ("p1", "2026-11-03T10:00:00.000Z"),
                ("p2", "2026-11-02T10:00:00.000Z"),
            ]),
        );
        assert_eq!(earned["a"].as_deref(), Some("2026-11-03T10:00:00.000Z"));
    }

    #[test]
    fn a_skill_with_an_unsolved_problem_is_unearned() {
        let skills = [skill("a", &[], &["p1", "p2"])];
        let earned = earned_times(&skills, &accepted(&[("p1", "2026-11-03T10:00:00.000Z")]));
        assert_eq!(earned["a"], None);
        assert_eq!(earned_times(&skills, &HashMap::new())["a"], None);
    }

    #[test]
    fn a_skill_waits_on_the_skills_it_requires() {
        // Listed before what it requires: the order of `skills` does not matter.
        let skills = [
            skill("c", &["b"], &["p3"]),
            skill("b", &["a"], &["p2"]),
            skill("a", &[], &["p1"]),
        ];
        let mut times = accepted(&[
            ("p2", "2026-11-02T10:00:00.000Z"),
            ("p3", "2026-11-01T10:00:00.000Z"),
        ]);
        let earned = earned_times(&skills, &times);
        assert_eq!(
            (&earned["a"], &earned["b"], &earned["c"]),
            (&None, &None, &None)
        );

        // Solving the one problem of `a` earns all three at that moment.
        times.insert("p1".to_owned(), "2026-11-05T10:00:00.000Z".to_owned());
        let earned = earned_times(&skills, &times);
        for id in ["a", "b", "c"] {
            assert_eq!(earned[id].as_deref(), Some("2026-11-05T10:00:00.000Z"));
        }
    }

    #[test]
    fn the_earned_time_is_the_latest_of_problems_and_required_skills() {
        let skills = [skill("a", &[], &["p1"]), skill("b", &["a"], &["p2"])];
        let earned = earned_times(
            &skills,
            &accepted(&[
                ("p1", "2026-11-01T10:00:00.000Z"),
                ("p2", "2026-11-04T10:00:00.000Z"),
            ]),
        );
        assert_eq!(earned["a"].as_deref(), Some("2026-11-01T10:00:00.000Z"));
        assert_eq!(earned["b"].as_deref(), Some("2026-11-04T10:00:00.000Z"));
    }

    #[test]
    fn times_are_compared_as_times_and_returned_as_stored() {
        // As strings the first sorts last; as times it is two hours earlier.
        let skills = [skill("a", &[], &["p1", "p2"])];
        let earned = earned_times(
            &skills,
            &accepted(&[
                ("p1", "2026-11-03T09:00:00+02:00"),
                ("p2", "2026-11-03T08:00:00.000Z"),
            ]),
        );
        assert_eq!(earned["a"].as_deref(), Some("2026-11-03T08:00:00.000Z"));
    }

    #[test]
    fn an_unknown_requirement_or_a_cycle_leaves_a_skill_unearned() {
        let times = accepted(&[
            ("p1", "2026-11-01T10:00:00.000Z"),
            ("p2", "2026-11-01T10:00:00.000Z"),
        ]);
        let earned = earned_times(&[skill("a", &["ghost"], &["p1"])], &times);
        assert_eq!(earned["a"], None);
        let earned = earned_times(
            &[skill("a", &["b"], &["p1"]), skill("b", &["a"], &["p2"])],
            &times,
        );
        assert_eq!((&earned["a"], &earned["b"]), (&None, &None));
    }

    #[test]
    fn lanes_are_pairs_of_non_negative_integers() {
        let lanes: toml::Table =
            toml::from_str("ok = [2, 600]\nshort = [2]\nlong = [1, 2, 3]\nneg = [-1, 5]\nfloat = [1.5, 2]\nword = \"x\"")
                .expect("toml");
        assert_eq!(parse_lane(&lanes["ok"]), Some([2, 600]));
        for bad in ["short", "long", "neg", "float", "word"] {
            assert_eq!(parse_lane(&lanes[bad]), None, "{bad}");
        }
    }
}
