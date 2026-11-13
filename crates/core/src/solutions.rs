//! The user's solution files at `workspace/<problem id>/<framework>.py`.

use std::fs;
use std::io::ErrorKind;
use std::path::{Path, PathBuf};

use crate::catalog::ProblemMeta;
use crate::error::{CoreError, Result};
use crate::model::{Framework, SolutionFile};

/// Reads and writes solution files under a content root.
#[derive(Debug, Clone)]
pub struct Workspace {
    dir: PathBuf,
}

impl Workspace {
    /// Creates the workspace accessor for `<root>/workspace`.
    pub fn new(root: &Path) -> Self {
        Self {
            dir: root.join("workspace"),
        }
    }

    /// Path of the solution file for a problem and framework (it may not exist).
    pub fn path(&self, problem_id: &str, framework: Framework) -> PathBuf {
        self.dir.join(problem_id).join(format!("{framework}.py"))
    }

    /// Reads the saved solution, if there is one.
    pub fn saved(&self, problem_id: &str, framework: Framework) -> Result<Option<String>> {
        let path = self.path(problem_id, framework);
        match fs::read_to_string(&path) {
            Ok(code) => Ok(Some(code)),
            Err(e) if e.kind() == ErrorKind::NotFound => Ok(None),
            Err(e) => Err(CoreError::io("read", &path, e)),
        }
    }

    /// Reads the solution, falling back to the starter when nothing has been saved.
    pub fn read(&self, problem: &ProblemMeta, framework: Framework) -> Result<SolutionFile> {
        let starter = problem.starter(framework)?;
        let saved = self.saved(&problem.id, framework)?;
        let is_starter = saved.as_ref().is_none_or(|code| *code == starter);
        Ok(SolutionFile {
            code: saved.unwrap_or(starter),
            is_starter,
            path: self
                .path(&problem.id, framework)
                .to_string_lossy()
                .into_owned(),
        })
    }

    /// Saves the solution. The write is atomic so an external editor never sees half a file.
    pub fn write(&self, problem: &ProblemMeta, framework: Framework, code: &str) -> Result<()> {
        problem.require_framework(framework)?;
        let path = self.path(&problem.id, framework);
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent).map_err(|e| CoreError::io("create", parent, e))?;
        }
        let temp = path.with_extension("py.tmp");
        fs::write(&temp, code).map_err(|e| CoreError::io("write", &temp, e))?;
        fs::rename(&temp, &path).map_err(|e| CoreError::io("write", &path, e))
    }

    /// Discards the saved solution and returns the starter.
    pub fn reset(&self, problem: &ProblemMeta, framework: Framework) -> Result<SolutionFile> {
        problem.require_framework(framework)?;
        let path = self.path(&problem.id, framework);
        match fs::remove_file(&path) {
            Ok(()) => {}
            Err(e) if e.kind() == ErrorKind::NotFound => {}
            Err(e) => return Err(CoreError::io("remove", &path, e)),
        }
        if let Some(parent) = path.parent() {
            // Tidy up the problem directory when this was its last file; ignore "not empty".
            let _ = fs::remove_dir(parent);
        }
        self.read(problem, framework)
    }

    /// Whether any framework's saved solution differs from its starter.
    pub fn is_modified(&self, problem: &ProblemMeta) -> bool {
        problem
            .frameworks
            .iter()
            .any(|&framework| match self.saved(&problem.id, framework) {
                // A saved file with an unreadable starter still counts as the user's work.
                Ok(Some(saved)) => problem
                    .starter(framework)
                    .map_or(true, |starter| starter != saved),
                _ => false,
            })
    }
}
