//! SSH plumbing for remote run targets (`docs/ARCHITECTURE.md` section 4.1).
//!
//! The app drives the system `ssh` and `rsync` with the user's own configuration and
//! agent. It never reads, stores, or asks for a key or a password: every invocation uses
//! `BatchMode=yes`, so a host that would prompt fails instead, and the failure is turned
//! into an instruction the user can follow in a terminal.
//!
//! This module holds the parts with no process handling: reading host aliases from
//! `~/.ssh/config`, validating what goes on a command line, building `ssh` and `rsync`
//! invocations (with connection multiplexing, so a run pays one handshake), quoting for
//! the remote shell, and the two scripts that run on the remote host.
//!
//! Safety rules: a host alias and a remote directory are checked against strict patterns
//! before they are used; everything is passed to `ssh` and `rsync` as separate
//! arguments; and each word of the one string the remote login shell parses is
//! single-quoted by [`sh_quote`].

use std::ffi::OsString;
use std::fs;
use std::path::{Path, PathBuf};

use crate::harness::CommandSpec;
use crate::model::SshHost;

/// Where the app works on a remote host unless the user chose another directory.
pub const DEFAULT_REMOTE_DIR: &str = "~/.whetstone";

/// Seconds `ssh` waits for a connection before giving up.
const CONNECT_TIMEOUT_S: u32 = 10;
/// Seconds an idle multiplexed connection stays open for the next command.
const CONTROL_PERSIST_S: u32 = 60;
/// Seconds of silence after which `rsync` gives up on a transfer.
const RSYNC_IO_TIMEOUT_S: u32 = 60;
/// How deep `Include` directives are followed.
const MAX_INCLUDE_DEPTH: usize = 8;

/// The official uv installer for macOS and Linux, from
/// <https://docs.astral.sh/uv/getting-started/installation/>.
pub const UV_INSTALL_COMMAND: &str = "curl -LsSf https://astral.sh/uv/install.sh | sh";

/// Runs on the remote host before anything else: reports the home directory, where uv
/// and rsync are, and what the machine is. uv's installer puts it in `~/.local/bin`
/// (older versions: `~/.cargo/bin`), which a non-interactive SSH session does not have
/// on its `PATH`, so both are added. Lines other than `ws-<key>=<value>` (a login
/// banner, say) are ignored by the reader.
pub const PROBE_SCRIPT: &str = r#"PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
echo "ws-home=$HOME"
echo "ws-uv=$(command -v uv 2>/dev/null)"
echo "ws-rsync=$(command -v rsync 2>/dev/null)"
echo "ws-system=$(uname -s 2>/dev/null)"
echo "ws-machine=$(uname -m 2>/dev/null)"
"#;

/// Runs a command on the remote host so that it can always be stopped from here.
///
/// Usage: `sh -c <this> whetstone <WHETSTONE_ROOT> <program> <args>...`.
///
/// Without a terminal, a remote command does not die when the SSH client goes away: it
/// keeps running until it happens to write to the closed connection. So the command is
/// started in its own process group (through `setsid` where it exists, otherwise through
/// the shell's job control) with its input detached, and this script keeps watching its
/// own standard input, which is the SSH channel. The local side holds that channel open
/// for as long as the run should live. When it closes (cancel, timeout, the app exiting
/// or crashing, the connection dropping), or when this script receives a hangup, it
/// kills the command's process group and every descendant it can still find, then
/// exits. When the command ends by itself, whatever it left running is killed the same
/// way and its exit code is passed on. The script's own standard error is discarded
/// (the command keeps the real one), so a shell's job-control notices never mix into
/// the command's output.
///
/// The script is POSIX `sh` and contains no backslash, so it survives being quoted for
/// any login shell.
pub const SUPERVISOR_SCRIPT: &str = r#"PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
WHETSTONE_ROOT=$1
export PATH WHETSTONE_ROOT
shift
exec 3<&0 4>&2 2>/dev/null
if command -v setsid >/dev/null 2>&1; then
  setsid "$@" </dev/null 2>&4 3<&- 4>&- &
else
  set -m
  "$@" </dev/null 2>&4 3<&- 4>&- &
fi
child=$!
tree() {
  for pid in $(ps -A -o pid= -o ppid= 2>/dev/null | awk -v parent="$1" '$2 == parent { print $1 }'); do
    tree "$pid"
  done
  echo "$1"
}
stop() {
  pids=$(tree "$child")
  kill -s KILL -- "-$child"
  kill -s KILL $pids
}
trap 'stop; exit 143' HUP INT TERM
(while read -r line; do :; done; kill -s HUP "$$") <&3 3<&- 4>&- &
watcher=$!
exec 3<&- 4>&-
wait "$child"
code=$?
trap - HUP INT TERM
kill "$watcher"
stop
exit "$code"
"#;

/// Whether `alias` is usable as an SSH destination on a command line: letters and
/// digits with `.`, `_`, or `-` inside. This excludes anything `ssh` would read as an
/// option (`-oProxyCommand=...`), a `user@host`, a wildcard pattern, and white space.
pub fn is_host_alias(alias: &str) -> bool {
    let bytes = alias.as_bytes();
    !bytes.is_empty()
        && bytes.len() <= 128
        && bytes[0].is_ascii_alphanumeric()
        && bytes
            .iter()
            .all(|&b| b.is_ascii_alphanumeric() || matches!(b, b'.' | b'_' | b'-'))
}

/// Whether `segment` is one plain path component: letters, digits, and `.`, `_`, `-`,
/// `+`, `@`; not `.` or `..`; not starting with `-`.
pub fn is_path_segment(segment: &str) -> bool {
    !segment.is_empty()
        && segment.len() <= 255
        && segment != "."
        && segment != ".."
        && !segment.starts_with('-')
        && segment
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || matches!(b, b'.' | b'_' | b'-' | b'+' | b'@'))
}

/// Whether `path` is an absolute path made of plain components (see
/// [`is_path_segment`]): nothing a shell would expand or split.
pub fn is_plain_absolute_path(path: &str) -> bool {
    path.len() <= 1024
        && path
            .strip_prefix('/')
            .is_some_and(|rest| !rest.is_empty() && rest.split('/').all(is_path_segment))
}

/// Whether `dir` may be used as a remote working directory: `~`, `~/<components>`, or
/// an absolute path, made of plain components only.
pub fn is_remote_dir(dir: &str) -> bool {
    dir == "~"
        || dir
            .strip_prefix("~/")
            .is_some_and(|rest| rest.len() <= 1024 && rest.split('/').all(is_path_segment))
        || is_plain_absolute_path(dir)
}

/// Resolves a validated remote directory against the remote home directory.
pub fn resolve_remote_dir(dir: &str, home: &str) -> String {
    match dir.strip_prefix('~') {
        Some(rest) => format!("{}{rest}", home.trim_end_matches('/')),
        None => dir.to_owned(),
    }
}

/// Quotes one word for a POSIX shell: wrapped in single quotes, with each single quote
/// written as `'\''`.
pub fn sh_quote(word: &str) -> String {
    let mut quoted = String::with_capacity(word.len() + 2);
    quoted.push('\'');
    for ch in word.chars() {
        if ch == '\'' {
            quoted.push_str("'\\''");
        } else {
            quoted.push(ch);
        }
    }
    quoted.push('\'');
    quoted
}

/// Joins words into one command line for the remote shell, quoting each.
pub fn sh_join<S: AsRef<str>>(words: impl IntoIterator<Item = S>) -> String {
    words
        .into_iter()
        .map(|word| sh_quote(word.as_ref()))
        .collect::<Vec<_>>()
        .join(" ")
}

/// The remote command line that runs `script` with `args` under `sh`, whatever the
/// user's login shell is.
pub fn sh_script(script: &str, args: &[String]) -> String {
    let mut words = vec!["-c".to_owned(), script.to_owned(), "whetstone".to_owned()];
    words.extend(args.iter().cloned());
    format!("exec sh {}", sh_join(&words))
}

/// The user's SSH configuration file.
pub fn default_config_path() -> Option<PathBuf> {
    std::env::var_os("HOME")
        .filter(|home| !home.is_empty())
        .map(|home| PathBuf::from(home).join(".ssh").join("config"))
}

/// Reads the host aliases an SSH configuration file defines, in file order.
///
/// A `Host` line may name several aliases; patterns (`*`, `?`, `!`) are skipped, as is
/// anything that is not a plain alias. `HostName`, `User`, and `Port` are taken from
/// the alias's own blocks (the first value wins, as in `ssh`); values that only come
/// from a wildcard block or a `Match` block are not resolved, because they are shown
/// for orientation only: `ssh` itself always reads the real configuration.
///
/// `Include` is followed (relative to the configuration's directory, `~` expanded,
/// `*` and `?` in the file name). Wildcards in directory components of an `Include`
/// are not supported. A missing or unreadable file yields no hosts.
pub fn read_hosts(config: &Path) -> Vec<SshHost> {
    let mut hosts = Vec::new();
    let base = config.parent().map(Path::to_path_buf).unwrap_or_default();
    read_into(config, &base, &mut Vec::new(), &mut hosts, 0);
    hosts
}

fn read_into(
    file: &Path,
    base: &Path,
    current: &mut Vec<usize>,
    hosts: &mut Vec<SshHost>,
    depth: usize,
) {
    let Ok(text) = fs::read_to_string(file) else {
        return;
    };
    parse_into(&text, base, current, hosts, depth);
}

/// Parses configuration text. `current` holds the indices of the hosts the directives
/// being read apply to.
fn parse_into(
    text: &str,
    base: &Path,
    current: &mut Vec<usize>,
    hosts: &mut Vec<SshHost>,
    depth: usize,
) {
    for line in text.lines() {
        let Some((keyword, values)) = split_directive(line) else {
            continue;
        };
        match keyword.as_str() {
            "host" => {
                current.clear();
                for alias in values {
                    if alias.contains(['*', '?', '!']) || !is_host_alias(&alias) {
                        continue;
                    }
                    let index = hosts
                        .iter()
                        .position(|host| host.alias == alias)
                        .unwrap_or_else(|| {
                            hosts.push(SshHost {
                                alias,
                                host_name: None,
                                user: None,
                                port: None,
                            });
                            hosts.len() - 1
                        });
                    current.push(index);
                }
            }
            "match" => current.clear(),
            "hostname" | "user" | "port" => {
                let Some(value) = values.into_iter().next() else {
                    continue;
                };
                for &index in current.iter() {
                    let host = &mut hosts[index];
                    match keyword.as_str() {
                        "hostname" => {
                            host.host_name.get_or_insert_with(|| value.clone());
                        }
                        "user" => {
                            host.user.get_or_insert_with(|| value.clone());
                        }
                        _ => {
                            if host.port.is_none() {
                                host.port = value.parse().ok();
                            }
                        }
                    }
                }
            }
            "include" if depth < MAX_INCLUDE_DEPTH => {
                for pattern in values {
                    for file in expand_include(&pattern, base) {
                        // Directives in the included file apply to the block that
                        // includes it; a `Host` line inside starts its own block, which
                        // ends with the file.
                        let mut inner = current.clone();
                        read_into(&file, base, &mut inner, hosts, depth + 1);
                    }
                }
            }
            _ => {}
        }
    }
}

/// Splits a configuration line into its lowercased keyword and its values. Handles
/// `Keyword value`, `Keyword=value`, double-quoted values, and comments.
fn split_directive(line: &str) -> Option<(String, Vec<String>)> {
    let line = line.trim();
    if line.is_empty() || line.starts_with('#') {
        return None;
    }
    let end = line
        .find(|c: char| c.is_whitespace() || c == '=')
        .unwrap_or(line.len());
    let keyword = line[..end].to_ascii_lowercase();
    let rest = line[end..].trim_start();
    let rest = rest.strip_prefix('=').unwrap_or(rest).trim_start();

    let mut values = Vec::new();
    let mut word = String::new();
    let mut quoted = false;
    let mut started = false;
    for ch in rest.chars() {
        match ch {
            '"' => {
                quoted = !quoted;
                started = true;
            }
            '#' if !quoted && !started => break,
            c if c.is_whitespace() && !quoted => {
                if started {
                    values.push(std::mem::take(&mut word));
                    started = false;
                }
            }
            c => {
                word.push(c);
                started = true;
            }
        }
    }
    if started {
        values.push(word);
    }
    Some((keyword, values))
}

/// The files an `Include` value names, sorted. `~/` is the home directory; a relative
/// path is relative to the configuration's directory.
fn expand_include(pattern: &str, base: &Path) -> Vec<PathBuf> {
    let path = if let Some(rest) = pattern.strip_prefix("~/") {
        match std::env::var_os("HOME") {
            Some(home) => PathBuf::from(home).join(rest),
            None => return Vec::new(),
        }
    } else if Path::new(pattern).is_absolute() {
        PathBuf::from(pattern)
    } else {
        base.join(pattern)
    };
    let Some(name) = path.file_name().and_then(|name| name.to_str()) else {
        return Vec::new();
    };
    if !name.contains(['*', '?']) {
        return vec![path];
    }
    let Some(dir) = path.parent() else {
        return Vec::new();
    };
    let mut files: Vec<PathBuf> = fs::read_dir(dir)
        .into_iter()
        .flatten()
        .filter_map(|entry| entry.ok())
        .filter(|entry| {
            entry
                .file_name()
                .to_str()
                .is_some_and(|candidate| wildcard_match(name, candidate))
        })
        .map(|entry| entry.path())
        .filter(|path| path.is_file())
        .collect();
    files.sort();
    files
}

/// Matches `text` against a pattern where `*` is any run of characters and `?` is any
/// one character.
fn wildcard_match(pattern: &str, text: &str) -> bool {
    let pattern: Vec<char> = pattern.chars().collect();
    let text: Vec<char> = text.chars().collect();
    let (mut p, mut t) = (0, 0);
    let mut star: Option<(usize, usize)> = None;
    while t < text.len() {
        if p < pattern.len() && (pattern[p] == '?' || pattern[p] == text[t]) {
            p += 1;
            t += 1;
        } else if p < pattern.len() && pattern[p] == '*' {
            star = Some((p, t));
            p += 1;
        } else if let Some((star_p, star_t)) = star {
            p = star_p + 1;
            t = star_t + 1;
            star = Some((star_p, star_t + 1));
        } else {
            return false;
        }
    }
    pattern[p..].iter().all(|&c| c == '*')
}

/// Overrides for the SSH tooling; everything defaults to the system's.
#[derive(Debug, Clone, Default)]
pub struct SshOptions {
    /// The `ssh` executable; `None` finds `ssh` on `PATH`.
    pub ssh_program: Option<PathBuf>,
    /// The `rsync` executable; `None` finds `rsync` on `PATH`.
    pub rsync_program: Option<PathBuf>,
    /// The configuration file to read host aliases from (and to pass to `ssh -F`);
    /// `None` is `~/.ssh/config`, read by `ssh` itself.
    pub config: Option<PathBuf>,
    /// The directory for connection-sharing sockets; `None` is an app-owned directory
    /// under `/tmp`.
    pub control_dir: Option<PathBuf>,
}

/// Builds `ssh` and `rsync` command lines.
#[derive(Debug, Clone)]
pub struct Ssh {
    ssh: PathBuf,
    rsync: PathBuf,
    config: Option<PathBuf>,
    control_dir: Option<PathBuf>,
}

impl Ssh {
    /// The system tooling with `options` applied. Prepares the directory for
    /// connection-sharing sockets; when that is not possible, connections are simply
    /// not shared.
    pub fn new(options: SshOptions) -> Self {
        let control_dir = match options.control_dir {
            Some(dir) => prepare_private_dir(&dir).then_some(dir),
            None => default_control_dir(),
        };
        Self {
            ssh: options.ssh_program.unwrap_or_else(|| PathBuf::from("ssh")),
            rsync: options
                .rsync_program
                .unwrap_or_else(|| PathBuf::from("rsync")),
            config: options.config,
            control_dir,
        }
    }

    /// The configuration file host aliases are read from.
    pub fn config_path(&self) -> Option<PathBuf> {
        self.config.clone().or_else(default_config_path)
    }

    /// The host aliases the user's configuration defines.
    pub fn hosts(&self) -> Vec<SshHost> {
        self.config_path()
            .map(|path| read_hosts(&path))
            .unwrap_or_default()
    }

    /// A private directory for short-lived files (captured command output).
    pub fn scratch_dir(&self) -> PathBuf {
        self.control_dir.clone().unwrap_or_else(std::env::temp_dir)
    }

    /// The options every `ssh` invocation carries: never prompt, give up on a dead
    /// host, notice a dead connection, and share one connection per host. `master`
    /// lets this invocation open the shared connection; otherwise it only uses one
    /// that exists (and connects directly when none does).
    fn options(&self, master: bool) -> Vec<String> {
        let mut options = vec![
            "-o".to_owned(),
            "BatchMode=yes".to_owned(),
            "-o".to_owned(),
            format!("ConnectTimeout={CONNECT_TIMEOUT_S}"),
            "-o".to_owned(),
            "ServerAliveInterval=15".to_owned(),
            "-o".to_owned(),
            "ServerAliveCountMax=3".to_owned(),
        ];
        if let Some(dir) = &self.control_dir {
            options.extend([
                "-o".to_owned(),
                format!("ControlMaster={}", if master { "auto" } else { "no" }),
                "-o".to_owned(),
                format!("ControlPath={}/%C", dir.display()),
            ]);
            if master {
                options.extend([
                    "-o".to_owned(),
                    format!("ControlPersist={CONTROL_PERSIST_S}"),
                ]);
            }
        }
        if let Some(config) = &self.config {
            options.extend(["-F".to_owned(), config.display().to_string()]);
        }
        options
    }

    /// `ssh <options> -T -- <alias> <remote command>`. The remote command is one
    /// string, parsed by the remote login shell; build it with [`sh_join`] or
    /// [`sh_script`]. Fails when `alias` is not a plain host alias.
    pub fn command(
        &self,
        alias: &str,
        remote_command: &str,
        master: bool,
    ) -> Result<CommandSpec, String> {
        check_alias(alias)?;
        let mut args: Vec<OsString> = self.options(master).into_iter().map(Into::into).collect();
        args.extend([
            "-T".into(),
            "--".into(),
            alias.into(),
            remote_command.into(),
        ]);
        Ok(CommandSpec {
            program: self.ssh.clone().into(),
            args,
            ..CommandSpec::default()
        })
    }

    /// `rsync -az [--delete] [--exclude=...] -e "<ssh with options>" <sources>
    /// <alias>:<dest>`. `dest` must be a plain absolute path, because the remote shell
    /// parses it. A source ending in `/` copies the directory's contents.
    pub fn rsync(
        &self,
        alias: &str,
        sources: &[OsString],
        dest: &str,
        delete: bool,
        excludes: &[&str],
    ) -> Result<CommandSpec, String> {
        check_alias(alias)?;
        let plain = dest.strip_suffix('/').unwrap_or(dest);
        if !is_plain_absolute_path(plain) {
            return Err(format!("`{dest}` is not a usable remote path"));
        }
        let mut shell = vec![rsh_word(&self.ssh.display().to_string())?];
        for option in self.options(false) {
            shell.push(rsh_word(&option)?);
        }
        let mut args: Vec<OsString> = vec![
            "-az".into(),
            format!("--timeout={RSYNC_IO_TIMEOUT_S}").into(),
        ];
        if delete {
            args.push("--delete".into());
        }
        args.extend(
            excludes
                .iter()
                .map(|pattern| format!("--exclude={pattern}").into()),
        );
        args.extend(["-e".into(), shell.join(" ").into()]);
        args.extend(sources.iter().cloned());
        args.push(format!("{alias}:{dest}").into());
        Ok(CommandSpec {
            program: self.rsync.clone().into(),
            args,
            ..CommandSpec::default()
        })
    }
}

fn check_alias(alias: &str) -> Result<(), String> {
    if is_host_alias(alias) {
        Ok(())
    } else {
        Err(format!("`{alias}` is not a usable SSH host alias"))
    }
}

/// One word of the remote-shell command rsync is given with `-e`, which rsync splits
/// on spaces: as is when it has nothing to split or expand, single-quoted when that is
/// enough, refused otherwise.
fn rsh_word(word: &str) -> Result<String, String> {
    let plain = !word.is_empty()
        && word
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || b"_@%+=:,./-".contains(&b));
    if plain {
        Ok(word.to_owned())
    } else if !word.contains(['\'', '\\']) {
        Ok(format!("'{word}'"))
    } else {
        Err(format!("`{word}` cannot be passed to rsync"))
    }
}

/// The default directory for connection-sharing sockets: `/tmp/whetstone-ssh-<uid>`,
/// private to the user. Kept short because a socket path is limited to about 100
/// bytes, and the system temporary directory on macOS already uses half of that.
#[cfg(unix)]
fn default_control_dir() -> Option<PathBuf> {
    // SAFETY: `geteuid` has no preconditions and cannot fail.
    let uid = unsafe { libc::geteuid() };
    let dir = PathBuf::from(format!("/tmp/whetstone-ssh-{uid}"));
    prepare_private_dir(&dir).then_some(dir)
}

#[cfg(not(unix))]
fn default_control_dir() -> Option<PathBuf> {
    None
}

/// Creates `dir` if needed and confirms it is a real directory owned by this user that
/// nobody else can enter. `/tmp` is shared, so an existing entry is not trusted
/// otherwise.
#[cfg(unix)]
fn prepare_private_dir(dir: &Path) -> bool {
    use std::os::unix::fs::{DirBuilderExt, MetadataExt, PermissionsExt};
    let _ = fs::DirBuilder::new()
        .recursive(true)
        .mode(0o700)
        .create(dir);
    let Ok(metadata) = fs::symlink_metadata(dir) else {
        return false;
    };
    // SAFETY: `geteuid` has no preconditions and cannot fail.
    let uid = unsafe { libc::geteuid() };
    if !metadata.is_dir() || metadata.uid() != uid {
        return false;
    }
    if metadata.permissions().mode() & 0o077 != 0
        && fs::set_permissions(dir, fs::Permissions::from_mode(0o700)).is_err()
    {
        return false;
    }
    true
}

#[cfg(not(unix))]
fn prepare_private_dir(_dir: &Path) -> bool {
    false
}

/// What a failed `ssh` connection means and what the user should do about it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Diagnosis {
    /// One line saying what went wrong.
    pub message: String,
    /// A command or instruction for the user.
    pub fix: String,
}

/// Reads the error output of a failed `ssh` invocation.
pub fn diagnose_connection(alias: &str, stderr: &str) -> Diagnosis {
    let lower = stderr.to_ascii_lowercase();
    let has = |needle: &str| lower.contains(needle);
    if has("remote host identification has changed") {
        Diagnosis {
            message: format!("The host key of `{alias}` has changed since you last connected."),
            fix: format!(
                "Run `ssh {alias}` in a terminal and read its warning. If the machine was \
                 reinstalled, remove the old key with `ssh-keygen -R <its host name>`, connect \
                 once to accept the new one, then check again."
            ),
        }
    } else if has("host key verification failed") {
        Diagnosis {
            message: format!("The host key of `{alias}` has not been accepted yet."),
            fix: format!(
                "Run `ssh {alias}` in a terminal once, confirm the host key, then check again."
            ),
        }
    } else if has("permission denied") {
        Diagnosis {
            message: format!(
                "`{alias}` did not accept a key, and Whetstone never asks for a password."
            ),
            fix: format!(
                "Load your key into the SSH agent with `ssh-add`, confirm that \
                 `ssh {alias} true` works in a terminal without a prompt, then check again."
            ),
        }
    } else if has("could not resolve hostname") {
        Diagnosis {
            message: format!("The host name of `{alias}` could not be resolved."),
            fix: format!("Check the `HostName` of `{alias}` in ~/.ssh/config and your network."),
        }
    } else if has("connection refused")
        || has("timed out")
        || has("no route to host")
        || has("network is unreachable")
        || has("connection closed")
        || has("connection reset")
    {
        Diagnosis {
            message: format!(
                "`{alias}` could not be reached: {}.",
                last_line(stderr).unwrap_or("the connection failed")
            ),
            fix: format!(
                "Check that the machine is on and reachable, and that `ssh {alias}` works in a terminal."
            ),
        }
    } else {
        Diagnosis {
            message: format!(
                "SSH could not connect to `{alias}`: {}.",
                last_line(stderr).unwrap_or("it gave no reason")
            ),
            fix: format!("Run `ssh {alias}` in a terminal to see the full error."),
        }
    }
}

/// The last non-empty line of `text`, without a trailing period.
pub(crate) fn last_line(text: &str) -> Option<&str> {
    text.lines()
        .rev()
        .map(|line| line.trim().trim_end_matches('.'))
        .find(|line| !line.is_empty())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn hosts_of(text: &str) -> Vec<SshHost> {
        let mut hosts = Vec::new();
        parse_into(
            text,
            Path::new("/nonexistent"),
            &mut Vec::new(),
            &mut hosts,
            0,
        );
        hosts
    }

    fn host(
        alias: &str,
        host_name: Option<&str>,
        user: Option<&str>,
        port: Option<u16>,
    ) -> SshHost {
        SshHost {
            alias: alias.into(),
            host_name: host_name.map(Into::into),
            user: user.map(Into::into),
            port,
        }
    }

    #[test]
    fn hosts_are_read_in_order_without_patterns() {
        let hosts = hosts_of(
            "# my machines\n\
             Host *\n  ServerAliveInterval 30\n  User everyone\n\n\
             Host gpu-box\n  HostName 10.0.0.5\n  User ada\n  Port 2222\n  IdentityFile ~/.ssh/id_ed25519\n\n\
             Host=studio  mini\n\tHostName=studio.local\n\
             Host *.example.com !bad ?x\n  User nobody\n\
             Host \"quoted\" # trailing comment\n  hostname \"h.example.com\"\n\
             Match host gpu-box\n  User ignored\n\
             Host gpu-box\n  User later\n  HostName later.example.com\n",
        );
        assert_eq!(
            hosts,
            vec![
                host("gpu-box", Some("10.0.0.5"), Some("ada"), Some(2222)),
                host("studio", Some("studio.local"), None, None),
                host("mini", Some("studio.local"), None, None),
                host("quoted", Some("h.example.com"), None, None),
            ]
        );
    }

    #[test]
    fn unusable_aliases_are_skipped() {
        let hosts = hosts_of(
            "Host -oProxyCommand=evil\nHost user@host\nHost a;b\nHost ok.example\nHost $(x)\n",
        );
        assert_eq!(hosts, vec![host("ok.example", None, None, None)]);
        assert!(hosts_of("").is_empty());
        assert!(hosts_of("Host\nHostName x\nPort 22\n").is_empty());
        assert_eq!(
            hosts_of("Host a\n Port many\n Port 22\n"),
            vec![host("a", None, None, Some(22))]
        );
    }

    #[test]
    fn includes_are_followed_with_wildcards() {
        let tmp = tempfile::tempdir().expect("temp dir");
        let dir = tmp.path();
        fs::create_dir_all(dir.join("conf.d")).expect("mkdir");
        fs::write(
            dir.join("config"),
            format!(
                "Include conf.d/*.conf\nInclude {}\nInclude missing\nHost main\n  Include user.inc\n",
                dir.join("extra").display()
            ),
        )
        .expect("write");
        fs::write(dir.join("conf.d/b.conf"), "Host beta\n HostName b\n").expect("write");
        fs::write(dir.join("conf.d/a.conf"), "Host alpha\n").expect("write");
        fs::write(dir.join("conf.d/ignored.txt"), "Host nope\n").expect("write");
        fs::write(dir.join("extra"), "Host extra\nInclude config\n").expect("write");
        fs::write(dir.join("user.inc"), "User ada\n").expect("write");
        let hosts = read_hosts(&dir.join("config"));
        let aliases: Vec<&str> = hosts.iter().map(|host| host.alias.as_str()).collect();
        assert_eq!(aliases, ["alpha", "beta", "extra", "main"]);
        assert_eq!(hosts[1].host_name.as_deref(), Some("b"));
        assert_eq!(hosts[3].user.as_deref(), Some("ada"));
        assert!(read_hosts(&dir.join("absent")).is_empty());
    }

    #[test]
    fn wildcards_match_file_names() {
        assert!(wildcard_match("*.conf", "a.conf"));
        assert!(wildcard_match("a?c*", "abcdef"));
        assert!(wildcard_match("*", ""));
        assert!(!wildcard_match("*.conf", "a.txt"));
        assert!(!wildcard_match("a?c", "ac"));
    }

    #[test]
    fn aliases_and_remote_directories_are_strict() {
        for alias in ["gpu-box", "a", "studio.local", "h_1", "10.0.0.5"] {
            assert!(is_host_alias(alias), "{alias}");
        }
        for alias in [
            "",
            "-o",
            "-oProxyCommand=x",
            "a b",
            "a;b",
            "user@host",
            "a/b",
            "*",
            "a*",
            "$(x)",
            "a'b",
            "caf\u{e9}",
            ".hidden",
        ] {
            assert!(!is_host_alias(alias), "{alias:?}");
        }
        assert!(!is_host_alias(&"a".repeat(129)));

        for dir in [
            "~",
            "~/.whetstone",
            "~/a/b-c/d_e",
            "/srv/whetstone",
            "/data/u@h/x+y",
        ] {
            assert!(is_remote_dir(dir), "{dir}");
        }
        for dir in [
            "",
            "/",
            "~/",
            "relative",
            "~user/x",
            "~/a b",
            "~/../etc",
            "/a/./b",
            "/a//b",
            "~/a/",
            "/a;rm -rf",
            "~/$HOME",
            "/a/`x`",
            "/a/'b'",
            "/a/-rf",
            "~/a\nb",
            "/a\\b",
        ] {
            assert!(!is_remote_dir(dir), "{dir:?}");
        }
        assert_eq!(
            resolve_remote_dir("~/.whetstone", "/home/ada"),
            "/home/ada/.whetstone"
        );
        assert_eq!(resolve_remote_dir("~", "/home/ada/"), "/home/ada");
        assert_eq!(resolve_remote_dir("/srv/ws", "/home/ada"), "/srv/ws");
    }

    #[test]
    fn words_are_quoted_for_the_remote_shell() {
        assert_eq!(sh_quote("plain"), "'plain'");
        assert_eq!(sh_quote(""), "''");
        assert_eq!(
            sh_quote("a b; rm -rf ~ $(x) `y`"),
            "'a b; rm -rf ~ $(x) `y`'"
        );
        assert_eq!(sh_quote("it's"), "'it'\\''s'");
        assert_eq!(sh_join(["mkdir", "-p", "/a b"]), "'mkdir' '-p' '/a b'");
        let line = sh_script("echo \"$1\"", &["x y".to_owned()]);
        assert_eq!(line, "exec sh '-c' 'echo \"$1\"' 'whetstone' 'x y'");
    }

    #[test]
    fn the_remote_scripts_survive_any_login_shell() {
        // No backslash: nothing for a login shell's quoting rules to reinterpret.
        assert!(!SUPERVISOR_SCRIPT.contains('\\'));
        assert!(!PROBE_SCRIPT.contains('\\'));
    }

    fn ssh(control: Option<&str>) -> Ssh {
        Ssh {
            ssh: "ssh".into(),
            rsync: "rsync".into(),
            config: None,
            control_dir: control.map(PathBuf::from),
        }
    }

    #[test]
    fn ssh_commands_never_prompt_and_share_a_connection() {
        let spec = ssh(Some("/tmp/ws-ctl"))
            .command("gpu-box", "'uname' '-a'", true)
            .expect("command");
        assert_eq!(
            spec.display(),
            "ssh -o BatchMode=yes -o ConnectTimeout=10 -o ServerAliveInterval=15 \
             -o ServerAliveCountMax=3 -o ControlMaster=auto -o ControlPath=/tmp/ws-ctl/%C \
             -o ControlPersist=60 -T -- gpu-box 'uname' '-a'"
        );
        // The remote command is one argument, after `--` and the alias.
        assert_eq!(spec.args.last().expect("args"), "'uname' '-a'");
        assert_eq!(spec.args[spec.args.len() - 3], "--");

        let client = ssh(Some("/tmp/ws-ctl"))
            .command("gpu-box", "true", false)
            .expect("command");
        assert!(
            client
                .display()
                .contains("-o ControlMaster=no -o ControlPath=/tmp/ws-ctl/%C -T")
        );
        assert!(!client.display().contains("ControlPersist"));

        let bare = ssh(None).command("gpu-box", "true", true).expect("command");
        assert!(!bare.display().contains("Control"));
        assert!(bare.display().contains("BatchMode=yes"));

        for alias in ["-oProxyCommand=x", "a b", "", "x;y"] {
            assert!(
                ssh(None).command(alias, "true", false).is_err(),
                "{alias:?}"
            );
        }
    }

    #[test]
    fn rsync_commands_carry_the_same_ssh_options() {
        let spec = ssh(Some("/tmp/ws-ctl"))
            .rsync(
                "gpu-box",
                &["/r/python/".into()],
                "/home/ada/.whetstone/root/python/",
                true,
                &[".venv", "__pycache__"],
            )
            .expect("rsync");
        let args: Vec<String> = spec
            .args
            .iter()
            .map(|arg| arg.to_string_lossy().into_owned())
            .collect();
        assert_eq!(
            args,
            [
                "-az",
                "--timeout=60",
                "--delete",
                "--exclude=.venv",
                "--exclude=__pycache__",
                "-e",
                "ssh -o BatchMode=yes -o ConnectTimeout=10 -o ServerAliveInterval=15 \
                 -o ServerAliveCountMax=3 -o ControlMaster=no -o ControlPath=/tmp/ws-ctl/%C",
                "/r/python/",
                "gpu-box:/home/ada/.whetstone/root/python/",
            ]
        );
        for dest in ["relative", "/a b", "/a/$(x)", "/a;b", "~/x", "/a/../b"] {
            assert!(
                ssh(None)
                    .rsync("h", &["/s".into()], dest, false, &[])
                    .is_err(),
                "{dest:?}"
            );
        }
        assert!(
            ssh(None)
                .rsync("-e", &["/s".into()], "/d", false, &[])
                .is_err()
        );

        // A program path with a space is quoted for rsync's own splitting.
        let spaced = Ssh {
            ssh: "/opt/my tools/ssh".into(),
            ..ssh(None)
        };
        let spec = spaced
            .rsync("h", &["/s".into()], "/d", false, &[])
            .expect("rsync");
        assert!(
            spec.args[3]
                .to_string_lossy()
                .starts_with("'/opt/my tools/ssh' -o BatchMode=yes")
        );
    }

    #[test]
    fn connection_failures_come_with_a_fix() {
        let cases = [
            (
                "No ED25519 host key is known for box and you have requested strict checking.\nHost key verification failed.\n",
                "not been accepted",
                "ssh box",
            ),
            (
                "@@@ WARNING: REMOTE HOST IDENTIFICATION HAS CHANGED! @@@\nHost key verification failed.\n",
                "has changed",
                "ssh-keygen -R",
            ),
            (
                "ada@box: Permission denied (publickey,password).\n",
                "never asks for a password",
                "ssh-add",
            ),
            (
                "ssh: Could not resolve hostname box: nodename nor servname provided, or not known\n",
                "could not be resolved",
                "HostName",
            ),
            (
                "ssh: connect to host box port 22: Connection refused\n",
                "Connection refused",
                "is on and reachable",
            ),
            (
                "ssh: connect to host box port 22: Operation timed out\n",
                "timed out",
                "is on and reachable",
            ),
            ("something unexpected\n", "something unexpected", "ssh box"),
            ("", "gave no reason", "ssh box"),
        ];
        for (stderr, message, fix) in cases {
            let diagnosis = diagnose_connection("box", stderr);
            assert!(
                diagnosis.message.contains(message),
                "{stderr:?}: {diagnosis:?}"
            );
            assert!(diagnosis.fix.contains(fix), "{stderr:?}: {diagnosis:?}");
        }
    }

    #[cfg(unix)]
    #[test]
    fn the_control_directory_must_be_private() {
        use std::os::unix::fs::PermissionsExt;
        let tmp = tempfile::tempdir().expect("temp dir");
        let dir = tmp.path().join("ctl");
        assert!(prepare_private_dir(&dir));
        assert_eq!(
            fs::metadata(&dir).expect("metadata").permissions().mode() & 0o777,
            0o700
        );
        // Too-open permissions are tightened; a file or a link in its place is refused.
        fs::set_permissions(&dir, fs::Permissions::from_mode(0o755)).expect("chmod");
        assert!(prepare_private_dir(&dir));
        assert_eq!(
            fs::metadata(&dir).expect("metadata").permissions().mode() & 0o777,
            0o700
        );
        let file = tmp.path().join("file");
        fs::write(&file, b"x").expect("write");
        assert!(!prepare_private_dir(&file));
        let link = tmp.path().join("link");
        std::os::unix::fs::symlink(&dir, &link).expect("symlink");
        assert!(!prepare_private_dir(&link));
    }
}
