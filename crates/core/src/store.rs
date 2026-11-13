//! Persistent state in SQLite at `<root>/.whetstone/state.db`.
//!
//! Holds submissions (with the code snapshot and per-case results), editorial unlocks,
//! and settings. The schema is versioned with `PRAGMA user_version`; each entry of
//! [`MIGRATIONS`] upgrades the database by one version.

use std::collections::HashMap;
use std::fs;
use std::path::Path;
use std::sync::{Mutex, MutexGuard, PoisonError};

use rusqlite::types::Type;
use rusqlite::{Connection, OptionalExtension, Row, params};
use serde::Serialize;
use serde::de::DeserializeOwned;

use crate::error::{CoreError, Result};
use crate::model::{Framework, Metric, RunMode, StoredCase, Submission, SubmissionDetail, Verdict};

/// Schema migrations, applied in order. Never edit an entry that has shipped; append.
const MIGRATIONS: &[&str] = &[
    // Version 1.
    "CREATE TABLE submissions (
         id          INTEGER PRIMARY KEY AUTOINCREMENT,
         problem_id  TEXT NOT NULL,
         framework   TEXT NOT NULL,
         mode        TEXT NOT NULL,
         verdict     TEXT NOT NULL,
         passed      INTEGER NOT NULL,
         total       INTEGER NOT NULL,
         duration_ms REAL NOT NULL,
         target      TEXT NOT NULL,
         created_at  TEXT NOT NULL,
         code        TEXT NOT NULL,
         cases       TEXT NOT NULL,
         metrics     TEXT NOT NULL
     );
     CREATE INDEX submissions_by_problem ON submissions (problem_id, id);
     CREATE TABLE editorial_unlocks (
         problem_id  TEXT PRIMARY KEY,
         unlocked_at TEXT NOT NULL
     );
     CREATE TABLE settings (
         key   TEXT PRIMARY KEY,
         value TEXT NOT NULL
     );",
];

const SUBMISSION_COLUMNS: &str =
    "id, problem_id, framework, mode, verdict, passed, total, duration_ms, target, created_at";

/// A submission about to be stored.
#[derive(Debug, Clone)]
pub struct NewSubmission<'a> {
    /// Problem id.
    pub problem_id: &'a str,
    /// Framework of the solution.
    pub framework: Framework,
    /// Run mode (always `submit` in practice).
    pub mode: RunMode,
    /// Final verdict.
    pub verdict: Verdict,
    /// Number of passed cases.
    pub passed: u32,
    /// Number of cases.
    pub total: u32,
    /// Duration in milliseconds.
    pub duration_ms: f64,
    /// Target name.
    pub target: &'a str,
    /// Creation time, ISO 8601 in UTC.
    pub created_at: &'a str,
    /// The submitted code.
    pub code: &'a str,
    /// Per-case results, with names.
    pub cases: &'a [StoredCase],
    /// Run-level metrics.
    pub metrics: &'a [Metric],
}

/// What the submissions table says about one problem.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct SubmissionProgress {
    /// Frameworks with an accepted submit, in first-accepted order.
    pub solved_frameworks: Vec<Framework>,
}

/// The state database.
pub struct Store {
    conn: Mutex<Connection>,
}

impl Store {
    /// Opens (creating if needed) the database at `path` and migrates it.
    pub fn open(path: &Path) -> Result<Self> {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent).map_err(|e| CoreError::io("create", parent, e))?;
        }
        Self::from_connection(Connection::open(path)?)
    }

    /// Opens a private in-memory database, for tests.
    pub fn open_in_memory() -> Result<Self> {
        Self::from_connection(Connection::open_in_memory()?)
    }

    fn from_connection(mut conn: Connection) -> Result<Self> {
        conn.pragma_update(None, "journal_mode", "WAL")?;
        conn.pragma_update(None, "foreign_keys", true)?;
        migrate(&mut conn)?;
        Ok(Self {
            conn: Mutex::new(conn),
        })
    }

    fn conn(&self) -> MutexGuard<'_, Connection> {
        // A panic while holding the lock cannot leave SQLite in a torn state.
        self.conn.lock().unwrap_or_else(PoisonError::into_inner)
    }

    /// Stores a submission and returns its id.
    pub fn insert_submission(&self, new: &NewSubmission<'_>) -> Result<i64> {
        let conn = self.conn();
        conn.execute(
            "INSERT INTO submissions (problem_id, framework, mode, verdict, passed, total,
                                      duration_ms, target, created_at, code, cases, metrics)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12)",
            params![
                new.problem_id,
                new.framework.as_str(),
                new.mode.as_str(),
                new.verdict.as_str(),
                new.passed,
                new.total,
                new.duration_ms,
                new.target,
                new.created_at,
                new.code,
                serde_json::to_string(new.cases)?,
                serde_json::to_string(new.metrics)?,
            ],
        )?;
        Ok(conn.last_insert_rowid())
    }

    /// Lists submissions newest first, optionally for one problem.
    pub fn list_submissions(&self, problem_id: Option<&str>) -> Result<Vec<Submission>> {
        let conn = self.conn();
        let mut statement = conn.prepare(&format!(
            "SELECT {SUBMISSION_COLUMNS} FROM submissions
             WHERE ?1 IS NULL OR problem_id = ?1
             ORDER BY id DESC"
        ))?;
        let rows = statement.query_map(params![problem_id], submission_from_row)?;
        Ok(rows.collect::<rusqlite::Result<_>>()?)
    }

    /// Loads one submission with its code and per-case results.
    pub fn get_submission(&self, id: i64) -> Result<SubmissionDetail> {
        let conn = self.conn();
        conn.query_row(
            &format!(
                "SELECT {SUBMISSION_COLUMNS}, code, cases, metrics FROM submissions WHERE id = ?1"
            ),
            params![id],
            |row| {
                let mut cases: Vec<StoredCase> = json_column(row, 11)?;
                // Rows stored before names were kept: show the id rather than nothing.
                for case in cases.iter_mut().filter(|c| c.name.is_empty()) {
                    case.name.clone_from(&case.result.id);
                }
                Ok(SubmissionDetail {
                    submission: submission_from_row(row)?,
                    code: row.get(10)?,
                    cases,
                    metrics: json_column(row, 12)?,
                })
            },
        )
        .optional()?
        .ok_or(CoreError::SubmissionNotFound(id))
    }

    /// Summarizes submissions per problem. A problem appears here once it has any submit.
    pub fn submission_progress(&self) -> Result<HashMap<String, SubmissionProgress>> {
        let conn = self.conn();
        let mut statement = conn.prepare(
            "SELECT problem_id, framework, MAX(verdict = 'accepted')
             FROM submissions GROUP BY problem_id, framework ORDER BY MIN(id)",
        )?;
        let mut rows = statement.query([])?;
        let mut progress: HashMap<String, SubmissionProgress> = HashMap::new();
        while let Some(row) = rows.next()? {
            let entry = progress.entry(row.get(0)?).or_default();
            if row.get::<_, bool>(2)? {
                entry.solved_frameworks.push(parse_column(row, 1)?);
            }
        }
        Ok(progress)
    }

    /// Creation times of every accepted submission.
    pub fn accepted_times(&self) -> Result<Vec<String>> {
        let conn = self.conn();
        let mut statement =
            conn.prepare("SELECT created_at FROM submissions WHERE verdict = 'accepted'")?;
        let rows = statement.query_map([], |row| row.get(0))?;
        Ok(rows.collect::<rusqlite::Result<_>>()?)
    }

    /// Records that the editorial of `problem_id` was explicitly revealed.
    pub fn unlock_editorial(&self, problem_id: &str, now: &str) -> Result<()> {
        self.conn().execute(
            "INSERT OR IGNORE INTO editorial_unlocks (problem_id, unlocked_at) VALUES (?1, ?2)",
            params![problem_id, now],
        )?;
        Ok(())
    }

    /// Whether the editorial of `problem_id` was explicitly revealed.
    pub fn editorial_revealed(&self, problem_id: &str) -> Result<bool> {
        Ok(self.conn().query_row(
            "SELECT EXISTS (SELECT 1 FROM editorial_unlocks WHERE problem_id = ?1)",
            params![problem_id],
            |row| row.get(0),
        )?)
    }

    /// Reads a JSON-encoded setting.
    pub fn setting<T: DeserializeOwned>(&self, key: &str) -> Result<Option<T>> {
        let text: Option<String> = self
            .conn()
            .query_row(
                "SELECT value FROM settings WHERE key = ?1",
                params![key],
                |row| row.get(0),
            )
            .optional()?;
        Ok(text.map(|t| serde_json::from_str(&t)).transpose()?)
    }

    /// Writes a JSON-encoded setting.
    pub fn set_setting<T: Serialize + ?Sized>(&self, key: &str, value: &T) -> Result<()> {
        self.conn().execute(
            "INSERT INTO settings (key, value) VALUES (?1, ?2)
             ON CONFLICT (key) DO UPDATE SET value = excluded.value",
            params![key, serde_json::to_string(value)?],
        )?;
        Ok(())
    }
}

fn migrate(conn: &mut Connection) -> Result<()> {
    let found: i64 = conn.pragma_query_value(None, "user_version", |row| row.get(0))?;
    let supported = MIGRATIONS.len() as i64;
    if found > supported {
        return Err(CoreError::SchemaTooNew { found, supported });
    }
    for (index, sql) in MIGRATIONS.iter().enumerate().skip(found.max(0) as usize) {
        let tx = conn.transaction()?;
        tx.execute_batch(sql)?;
        tx.pragma_update(None, "user_version", index as i64 + 1)?;
        tx.commit()?;
    }
    Ok(())
}

fn submission_from_row(row: &Row<'_>) -> rusqlite::Result<Submission> {
    Ok(Submission {
        id: row.get(0)?,
        problem_id: row.get(1)?,
        framework: parse_column(row, 2)?,
        mode: parse_column(row, 3)?,
        verdict: parse_column(row, 4)?,
        passed: row.get(5)?,
        total: row.get(6)?,
        duration_ms: row.get(7)?,
        target: row.get(8)?,
        created_at: row.get(9)?,
    })
}

/// Reads a text column and parses it with `FromStr`.
fn parse_column<T>(row: &Row<'_>, index: usize) -> rusqlite::Result<T>
where
    T: std::str::FromStr<Err = String>,
{
    let text: String = row.get(index)?;
    text.parse()
        .map_err(|e: String| rusqlite::Error::FromSqlConversionFailure(index, Type::Text, e.into()))
}

/// Reads a text column holding JSON.
fn json_column<T: DeserializeOwned>(row: &Row<'_>, index: usize) -> rusqlite::Result<T> {
    let text: String = row.get(index)?;
    serde_json::from_str(&text)
        .map_err(|e| rusqlite::Error::FromSqlConversionFailure(index, Type::Text, Box::new(e)))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::{CaseInfo, CaseResult, CaseStatus};

    fn submission<'a>(
        problem: &'a str,
        framework: Framework,
        verdict: Verdict,
    ) -> NewSubmission<'a> {
        NewSubmission {
            problem_id: problem,
            framework,
            mode: RunMode::Submit,
            verdict,
            passed: 1,
            total: 2,
            duration_ms: 12.5,
            target: "local",
            created_at: "2026-11-14T10:00:00.000Z",
            code: "def f(): ...\n",
            cases: &[],
            metrics: &[],
        }
    }

    #[test]
    fn submissions_round_trip_newest_first() -> Result<()> {
        let store = Store::open_in_memory()?;
        let cases = vec![StoredCase::new(
            CaseResult {
                id: "c1".into(),
                status: CaseStatus::Fail,
                duration_ms: 3.0,
                message: Some("off by one".into()),
                detail: None,
                stdout: "hi\n".into(),
                metrics: vec![],
            },
            Some(&CaseInfo {
                id: "c1".into(),
                name: "Hidden case 1".into(),
                public: false,
            }),
        )];
        let metrics = vec![Metric {
            name: "tokens".into(),
            value: 37.0,
            unit: Some("tokens".into()),
            limit: None,
        }];
        let first = store.insert_submission(&NewSubmission {
            cases: &cases,
            metrics: &metrics,
            ..submission("a", Framework::Torch, Verdict::WrongAnswer)
        })?;
        let second =
            store.insert_submission(&submission("b", Framework::Jax, Verdict::Accepted))?;

        let all = store.list_submissions(None)?;
        assert_eq!(
            all.iter().map(|s| s.id).collect::<Vec<_>>(),
            vec![second, first]
        );
        assert_eq!(store.list_submissions(Some("a"))?.len(), 1);
        assert!(store.list_submissions(Some("zzz"))?.is_empty());

        let detail = store.get_submission(first)?;
        assert_eq!(detail.submission.verdict, Verdict::WrongAnswer);
        assert_eq!(detail.code, "def f(): ...\n");
        assert_eq!(detail.cases, cases);
        assert_eq!(detail.metrics, metrics);
        assert!(matches!(
            store.get_submission(999),
            Err(CoreError::SubmissionNotFound(999))
        ));
        Ok(())
    }

    #[test]
    fn cases_stored_without_names_fall_back_to_the_id() -> Result<()> {
        let store = Store::open_in_memory()?;
        let id = store.insert_submission(&submission("a", Framework::Torch, Verdict::Accepted))?;
        store.conn().execute(
            "UPDATE submissions SET cases = ?1 WHERE id = ?2",
            params![
                r#"[{"id":"c1","status":"pass","durationMs":1.0,"message":null,"detail":null,"stdout":"","metrics":[]}]"#,
                id
            ],
        )?;
        let detail = store.get_submission(id)?;
        assert_eq!(detail.cases[0].name, "c1");
        assert!(detail.cases[0].public);
        Ok(())
    }

    #[test]
    fn progress_tracks_solved_frameworks() -> Result<()> {
        let store = Store::open_in_memory()?;
        store.insert_submission(&submission("a", Framework::Torch, Verdict::WrongAnswer))?;
        store.insert_submission(&submission("a", Framework::Mlx, Verdict::Accepted))?;
        store.insert_submission(&submission("a", Framework::Torch, Verdict::Accepted))?;
        store.insert_submission(&submission("b", Framework::Torch, Verdict::RuntimeError))?;
        let progress = store.submission_progress()?;
        assert_eq!(
            progress["a"].solved_frameworks,
            vec![Framework::Torch, Framework::Mlx]
        );
        assert!(progress["b"].solved_frameworks.is_empty());
        assert!(!progress.contains_key("c"));
        assert_eq!(store.accepted_times()?.len(), 2);
        Ok(())
    }

    #[test]
    fn unlocks_and_settings_persist_across_reopen() -> Result<()> {
        let tmp = tempfile::tempdir().map_err(|e| CoreError::io("create", Path::new("tmp"), e))?;
        let path = tmp.path().join(".whetstone").join("state.db");
        {
            let store = Store::open(&path)?;
            assert!(!store.editorial_revealed("a")?);
            store.unlock_editorial("a", "2026-11-14T10:00:00Z")?;
            store.unlock_editorial("a", "2026-11-15T10:00:00Z")?;
            store.set_setting("extras", &vec!["jax"])?;
            store.set_setting("extras", &vec!["jax", "mlx"])?;
        }
        let store = Store::open(&path)?;
        assert!(store.editorial_revealed("a")?);
        assert_eq!(
            store.setting::<Vec<String>>("extras")?,
            Some(vec!["jax".into(), "mlx".into()])
        );
        assert_eq!(store.setting::<Vec<String>>("missing")?, None);
        let version: i64 = store
            .conn()
            .pragma_query_value(None, "user_version", |r| r.get(0))?;
        assert_eq!(version, MIGRATIONS.len() as i64);
        Ok(())
    }

    #[test]
    fn a_newer_schema_is_rejected() -> Result<()> {
        let tmp = tempfile::tempdir().map_err(|e| CoreError::io("create", Path::new("tmp"), e))?;
        let path = tmp.path().join("state.db");
        Connection::open(&path)?.pragma_update(None, "user_version", 99)?;
        assert!(matches!(
            Store::open(&path),
            Err(CoreError::SchemaTooNew { found: 99, .. })
        ));
        Ok(())
    }
}
