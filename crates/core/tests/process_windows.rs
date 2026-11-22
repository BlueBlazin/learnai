//! Windows: killing a local process takes its descendants with it, as the process
//! group does on Unix (`runs.rs` covers that side with the fake harness).
#![cfg(windows)]

use std::fs;
use std::process::Command;
use std::time::{Duration, Instant};

use learnai_core::CommandSpec;
use learnai_core::target::{LocalProcess, Process};

/// Whether a process with this id exists. `tasklist` prints one CSV row for it, with
/// the id as a quoted field, and an information line when there is none.
fn alive(pid: &str) -> bool {
    let output = Command::new("tasklist")
        .args(["/FI", &format!("PID eq {pid}"), "/FO", "CSV", "/NH"])
        .output()
        .expect("tasklist");
    String::from_utf8_lossy(&output.stdout).contains(&format!("\"{pid}\""))
}

fn wait_until(what: &str, mut done: impl FnMut() -> bool) {
    let deadline = Instant::now() + Duration::from_secs(30);
    while !done() {
        assert!(Instant::now() < deadline, "timed out waiting for {what}");
        std::thread::sleep(Duration::from_millis(50));
    }
}

#[test]
fn kill_ends_the_process_and_its_descendants() {
    let dir = tempfile::tempdir().expect("temp dir");
    let pidfile = dir.path().join("child.pid");
    // PowerShell starts a ten-minute ping, records its id, and waits for it: a parent
    // with a child, like uv with the harness under it.
    let script = format!(
        "$p = Start-Process ping -ArgumentList '-n','600','127.0.0.1' -PassThru \
         -WindowStyle Hidden; Set-Content -Path '{}' -Value $p.Id; $p.WaitForExit()",
        pidfile.display()
    );
    let spec = CommandSpec {
        program: "powershell".into(),
        args: vec![
            "-NoProfile".into(),
            "-NonInteractive".into(),
            "-Command".into(),
            script.into(),
        ],
        ..CommandSpec::default()
    };
    let mut process = LocalProcess::spawn(&spec).expect("spawn");

    let mut pid = String::new();
    wait_until("the child's id", || {
        pid = fs::read_to_string(&pidfile)
            .unwrap_or_default()
            .trim()
            .trim_start_matches('\u{feff}')
            .to_owned();
        !pid.is_empty()
    });
    assert!(alive(&pid), "the child should be running before the kill");

    process.kill();
    process.finish(Duration::from_secs(5));
    wait_until("the child to be gone", || !alive(&pid));
}
