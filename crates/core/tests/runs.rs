//! Run orchestration against the fake harness: event mapping, failure modes, cancel,
//! timeouts, submission persistence, and the progress derived from submissions.

// The fakes these tests drive are shell and Perl scripts started as programs, which
// Windows cannot do. Windows needs its own process tests with a fake that is not a script.
#![cfg(unix)]

mod common;

use std::fs;
use std::process::Command;
use std::sync::Arc;
use std::time::{Duration, Instant};

use common::{Collector, TestRoot, assert_well_formed, logs, test_limits, wait_for_file};
use learnai_core::{
    CaseStatus, Core, CoreError, CoreOptions, Framework, LOCAL_TARGET, LogStream, MissingAsset,
    ProblemStatus, ProgramHarness, Requires, RunEvent, RunLimits, RunMode, Skip, Verdict,
};

/// Starts a run of `add-two` (torch) and returns its complete event stream.
fn run(core: &Core, mode: RunMode) -> Vec<RunEvent> {
    run_problem(core, "add-two", Framework::Torch, mode)
}

fn run_problem(core: &Core, problem: &str, framework: Framework, mode: RunMode) -> Vec<RunEvent> {
    let collector = Collector::new();
    let run_id = core
        .start_run(problem, framework, mode, LOCAL_TARGET, collector.sink())
        .expect("start run");
    assert!(!run_id.is_empty());
    let events = collector.wait_for_end();
    // Nothing may follow `end`.
    std::thread::sleep(Duration::from_millis(50));
    assert_eq!(collector.snapshot().len(), events.len(), "events after end");
    events
}

fn end_of(events: &[RunEvent]) -> (Verdict, u32, u32, Option<i64>) {
    match assert_well_formed(events) {
        RunEvent::End {
            verdict,
            passed,
            total,
            submission_id,
            ..
        } => (*verdict, *passed, *total, *submission_id),
        other => panic!("not an end event: {other:?}"),
    }
}

fn end_message(events: &[RunEvent]) -> (Option<String>, Option<String>) {
    match events.last() {
        Some(RunEvent::End {
            message, detail, ..
        }) => (message.clone(), detail.clone()),
        other => panic!("not an end event: {other:?}"),
    }
}

fn process_alive(pid: &str) -> bool {
    Command::new("kill")
        .args(["-0", pid])
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
}

#[test]
fn happy_run_maps_every_event_and_stores_nothing() {
    let root = TestRoot::new();
    let core = root.core("happy");
    let events = run(&core, RunMode::Run);

    assert_eq!(end_of(&events), (Verdict::Accepted, 2, 2, None));
    let kinds: Vec<_> = events
        .iter()
        .filter(|e| !matches!(e, RunEvent::Preparing { .. }))
        .map(|e| {
            serde_json::to_value(e).expect("json")["type"]
                .as_str()
                .expect("type")
                .to_owned()
        })
        .collect();
    assert_eq!(
        kinds,
        [
            "start",
            "caseStart",
            "caseEnd",
            "caseStart",
            "caseEnd",
            "end"
        ]
    );

    let RunEvent::Start {
        cases,
        env,
        requires,
        requires_met,
    } = events
        .iter()
        .find(|e| matches!(e, RunEvent::Start { .. }))
        .expect("start")
    else {
        unreachable!()
    };
    assert_eq!(
        cases
            .iter()
            .map(|c| (c.id.as_str(), c.public))
            .collect::<Vec<_>>(),
        [("c1", true), ("c2", true)]
    );
    assert_eq!((*requires, *requires_met), (Requires::Cpu, true));
    assert_eq!(env["python"], "3.15.2");
    assert_eq!(env["fake"], "true", "non-string env values are stringified");

    let RunEvent::End {
        duration_ms,
        metrics,
        ..
    } = events.last().expect("end")
    else {
        unreachable!()
    };
    assert_eq!(*duration_ms, 42.0);
    assert_eq!(metrics[0].name, "peak_mb");
    assert_eq!(metrics[0].unit.as_deref(), Some("MB"));

    // `run` results are ephemeral and do not count as an attempt.
    assert!(core.list_submissions(None).expect("list").is_empty());
    assert_eq!(
        core.get_problem("add-two").expect("detail").summary.status,
        ProblemStatus::Todo
    );
}

#[test]
fn accepted_submit_is_stored_and_solves_the_problem() {
    let root = TestRoot::new();
    let core = root.core("happy");
    core.write_solution(
        "add-two",
        Framework::Torch,
        "def add(a, b):\n    return a + b\n",
    )
    .expect("write");
    let events = run(&core, RunMode::Submit);

    let (verdict, passed, total, submission_id) = end_of(&events);
    assert_eq!((verdict, passed, total), (Verdict::Accepted, 3, 3));
    let id = submission_id.expect("submission id");

    let listed = core.list_submissions(Some("add-two")).expect("list");
    assert_eq!(listed.len(), 1);
    let submission = &listed[0];
    assert_eq!(submission.id, id);
    assert_eq!(submission.framework, Framework::Torch);
    assert_eq!(submission.mode, RunMode::Submit);
    assert_eq!(submission.verdict, Verdict::Accepted);
    assert_eq!(submission.target, "local");
    assert_eq!((submission.passed, submission.total), (3, 3));
    assert!(
        chrono::DateTime::parse_from_rfc3339(&submission.created_at).is_ok(),
        "{}",
        submission.created_at
    );

    let detail = core.get_submission(id).expect("detail");
    assert_eq!(detail.code, "def add(a, b):\n    return a + b\n");
    assert_eq!(
        detail
            .cases
            .iter()
            .map(|c| (c.result.id.as_str(), c.name.as_str(), c.public))
            .collect::<Vec<_>>(),
        [
            ("c1", "Small input", true),
            ("c2", "Large input", true),
            ("h1", "Hidden case 1", false)
        ]
    );
    let json = serde_json::to_value(&detail).expect("json");
    assert_eq!(json["cases"][2]["id"], "h1");
    assert_eq!(json["cases"][2]["name"], "Hidden case 1");
    assert_eq!(json["cases"][2]["public"], false);
    assert_eq!(detail.metrics[0].value, 12.5);
    let json = serde_json::to_value(&detail).expect("json");
    assert_eq!(json["problemId"], "add-two");
    assert_eq!(json["durationMs"], 42.0);
    assert!(json["createdAt"].is_string());

    // Solving unlocks the editorial and the reference, and shows up everywhere.
    let problem = core.get_problem("add-two").expect("detail");
    assert_eq!(problem.summary.status, ProblemStatus::Solved);
    assert_eq!(problem.summary.solved_frameworks, [Framework::Torch]);
    assert!(problem.editorial_unlocked);
    assert!(problem.editorial.is_some());
    assert!(core.get_reference("add-two", Framework::Jax).is_ok());
    assert_eq!(core.list_tracks().expect("tracks")[0].solved_count, 1);

    let stats = core.get_stats().expect("stats");
    assert_eq!((stats.solved, stats.total), (1, 3));
    assert_eq!(stats.by_difficulty[&learnai_core::Difficulty::Easy], (1, 1));
    assert_eq!(
        stats.by_difficulty[&learnai_core::Difficulty::Medium],
        (0, 1)
    );
    assert_eq!(stats.by_difficulty[&learnai_core::Difficulty::Hard], (0, 1));
    assert_eq!(stats.streak_days, 1);
}

#[test]
fn failing_submit_is_stored_with_case_details_and_marks_attempted() {
    let root = TestRoot::new();
    let core = root.core("fail");
    let events = run(&core, RunMode::Submit);

    let (verdict, passed, total, submission_id) = end_of(&events);
    assert_eq!((verdict, passed, total), (Verdict::WrongAnswer, 2, 3));

    let failed = events
        .iter()
        .find_map(|e| match e {
            RunEvent::CaseEnd { result } if result.status == CaseStatus::Fail => {
                Some(result.clone())
            }
            _ => None,
        })
        .expect("failing case");
    assert_eq!(failed.id, "c2");
    assert_eq!(failed.message.as_deref(), Some("expected 4, got 5"));
    assert_eq!(failed.detail.as_deref(), Some("expected: 4\ngot: 5"));
    assert_eq!(failed.stdout, "debug print\n");
    assert_eq!(failed.metrics[0].limit, Some(0.001));

    // The snapshot is the starter, because nothing was saved.
    let detail = core
        .get_submission(submission_id.expect("id"))
        .expect("detail");
    assert!(detail.code.contains("NotImplementedError"));
    assert_eq!(detail.cases[1].result, failed);
    assert_eq!(detail.cases[1].name, "Large input");

    let problem = core.get_problem("add-two").expect("detail");
    assert_eq!(problem.summary.status, ProblemStatus::Attempted);
    assert!(problem.summary.solved_frameworks.is_empty());
    assert!(!problem.editorial_unlocked);
    assert_eq!(core.get_stats().expect("stats").streak_days, 0);
}

#[test]
fn submissions_list_newest_first_across_problems_and_frameworks() {
    let root = TestRoot::new();
    let failing = root.core("fail");
    let passing = root.core("happy");
    run_problem(&failing, "add-two", Framework::Torch, RunMode::Submit);
    run_problem(&passing, "add-two", Framework::Jax, RunMode::Submit);
    run_problem(&passing, "hard-one", Framework::Torch, RunMode::Submit);

    let all = passing.list_submissions(None).expect("list");
    let summary: Vec<_> = all
        .iter()
        .map(|s| (s.problem_id.as_str(), s.framework, s.verdict))
        .collect();
    assert_eq!(
        summary,
        [
            ("hard-one", Framework::Torch, Verdict::Accepted),
            ("add-two", Framework::Jax, Verdict::Accepted),
            ("add-two", Framework::Torch, Verdict::WrongAnswer),
        ]
    );
    assert_eq!(
        passing
            .list_submissions(Some("add-two"))
            .expect("list")
            .len(),
        2
    );
    assert_eq!(
        passing
            .get_problem("add-two")
            .expect("detail")
            .summary
            .solved_frameworks,
        [Framework::Jax]
    );
    assert_eq!(passing.get_stats().expect("stats").solved, 2);
}

#[test]
fn garbage_lines_become_logs_and_do_not_disturb_the_protocol() {
    let root = TestRoot::new();
    let events = run(&root.core("garbage"), RunMode::Run);
    assert_eq!(end_of(&events), (Verdict::Accepted, 2, 2, None));

    let stdout_logs: Vec<_> = events
        .iter()
        .filter_map(|e| match e {
            RunEvent::Log {
                stream: LogStream::Stdout,
                text,
            } => Some(text.as_str()),
            _ => None,
        })
        .collect();
    assert_eq!(
        stdout_logs,
        [
            "plain line before start",
            "@@WS@@ this is not json",
            r#"@@WS@@ {"event":"mystery","x":1}"#,
            r#"   @@WS@@ {"event":"case_start","id":"indented"}"#,
            "invalid utf-8: \u{fffd}\u{fffd}",
            "",
            "between cases",
            "between cases",
        ]
    );
    assert!(events.contains(&RunEvent::Log {
        stream: LogStream::Stderr,
        text: "warning on stderr".into()
    }));
    let case_ends = events
        .iter()
        .filter(|e| matches!(e, RunEvent::CaseEnd { .. }))
        .count();
    assert_eq!(case_ends, 2);
}

#[test]
fn a_harness_that_dies_mid_run_ends_with_harness_error() {
    let root = TestRoot::new();
    let core = root.core("die");
    let events = run(&core, RunMode::Submit);

    let (verdict, passed, total, submission_id) = end_of(&events);
    assert_eq!((verdict, passed, total), (Verdict::HarnessError, 1, 3));
    let logs = logs(&events);
    assert!(
        logs.iter().any(|l| l == "Segmentation fault (pretend)"),
        "{logs:?}"
    );
    assert!(
        logs.iter()
            .any(|l| l.contains("exited with code 3 without reporting a result")),
        "{logs:?}"
    );

    assert!(
        end_message(&events)
            .0
            .is_some_and(|m| m.contains("exited with code 3 without reporting a result"))
    );

    // The attempt is recorded with the cases that did finish.
    let detail = core
        .get_submission(submission_id.expect("id"))
        .expect("detail");
    assert_eq!(detail.submission.verdict, Verdict::HarnessError);
    assert_eq!(detail.cases.len(), 1);
}

#[test]
fn a_harness_that_cannot_be_spawned_ends_with_harness_error() {
    let root = TestRoot::new();
    let core = Core::open(CoreOptions {
        harness: Some(Arc::new(ProgramHarness::new("/definitely/not/a/program"))),
        limits: test_limits(),
        ..CoreOptions::new(root.path())
    })
    .expect("open core");
    let events = run(&core, RunMode::Submit);

    assert_eq!(end_of(&events), (Verdict::HarnessError, 0, 0, None));
    assert!(
        logs(&events)
            .iter()
            .any(|l| l.contains("Could not start the judge"))
    );
    let (message, detail) = end_message(&events);
    assert!(
        message.is_some_and(|m| m.contains("uv was not found") && m.contains("Install uv")),
        "{events:?}"
    );
    assert_eq!(detail, None);
    assert!(
        core.list_submissions(None).expect("list").is_empty(),
        "a run that never launched is not an attempt"
    );
}

#[test]
fn an_import_error_forwards_message_and_detail_and_names_skipped_cases() {
    let root = TestRoot::new();
    let core = root.core("import-error");
    let events = run(&core, RunMode::Submit);

    let (verdict, passed, total, submission_id) = end_of(&events);
    assert_eq!((verdict, passed, total), (Verdict::ImportError, 0, 3));
    let (message, detail) = end_message(&events);
    assert_eq!(
        message.as_deref(),
        Some("the solution failed to import: SyntaxError: invalid syntax")
    );
    assert_eq!(
        detail.as_deref(),
        Some("  File \"torch.py\", line 1\nSyntaxError: invalid syntax\n")
    );

    // Skipped cases report `case_end` with no `case_start`.
    assert!(
        !events
            .iter()
            .any(|e| matches!(e, RunEvent::CaseStart { .. }))
    );
    let skipped = events
        .iter()
        .filter(
            |e| matches!(e, RunEvent::CaseEnd { result } if result.status == CaseStatus::Skipped),
        )
        .count();
    assert_eq!(skipped, 3);

    // They are stored with the names announced at the start.
    let detail = core
        .get_submission(submission_id.expect("id"))
        .expect("detail");
    assert_eq!(
        detail
            .cases
            .iter()
            .map(|c| (c.name.as_str(), c.public, c.result.status))
            .collect::<Vec<_>>(),
        [
            ("Small input", true, CaseStatus::Skipped),
            ("Large input", true, CaseStatus::Skipped),
            ("Hidden case 1", false, CaseStatus::Skipped)
        ]
    );
}

#[test]
fn cancel_kills_the_whole_process_group_and_ends_with_canceled() {
    let root = TestRoot::new();
    let core = root.core("hang");
    let collector = Collector::new();
    let run_id = core
        .start_run(
            "add-two",
            Framework::Torch,
            RunMode::Submit,
            LOCAL_TARGET,
            collector.sink(),
        )
        .expect("start run");

    collector.wait_for("the first case to start", |events| {
        events
            .iter()
            .any(|e| matches!(e, RunEvent::CaseStart { .. }))
    });
    // The script creates the file before it writes the pid into it: wait for the pid.
    let pid = wait_for_file(&root.pidfile());
    assert!(
        process_alive(&pid),
        "grandchild should be running before cancel"
    );

    let began = Instant::now();
    core.cancel_run(&run_id);
    let events = collector.wait_for_end();
    assert!(
        began.elapsed() < Duration::from_secs(5),
        "cancel took {:?}",
        began.elapsed()
    );

    assert_eq!(end_of(&events), (Verdict::Canceled, 0, 3, None));
    assert!(
        !process_alive(&pid),
        "the harness's child must be killed with it"
    );
    assert!(
        core.list_submissions(None).expect("list").is_empty(),
        "canceled runs are not stored"
    );

    // Canceling again, or canceling nonsense, is harmless.
    core.cancel_run(&run_id);
    core.cancel_run("no-such-run");
    std::thread::sleep(Duration::from_millis(50));
    assert_eq!(collector.snapshot().len(), events.len());
}

#[test]
fn a_run_that_exceeds_the_overall_limit_is_killed_with_time_limit() {
    let root = TestRoot::new();
    // echo-server: time_limit_s = 0.05, so the budget is 200 ms slack + 2 x 50 ms.
    let limits = RunLimits {
        slack: Duration::from_millis(200),
        ..test_limits()
    };
    let core = root.core_with("hang", limits);

    let began = Instant::now();
    let events = run_problem(&core, "echo-server", Framework::Python, RunMode::Run);
    let elapsed = began.elapsed();
    assert!(
        elapsed >= Duration::from_millis(300),
        "killed too early: {elapsed:?}"
    );
    assert!(
        elapsed < Duration::from_secs(10),
        "killed too late: {elapsed:?}"
    );

    assert_eq!(end_of(&events), (Verdict::TimeLimit, 0, 2, None));
    assert!(
        logs(&events)
            .iter()
            .any(|l| l.contains("exceeded the overall limit of 0.3 s")),
        "{events:?}"
    );
    let pid = fs::read_to_string(root.pidfile())
        .expect("pid file")
        .trim()
        .to_owned();
    assert!(!process_alive(&pid));
}

#[test]
fn a_harness_that_never_starts_hits_the_startup_limit() {
    let root = TestRoot::new();
    let limits = RunLimits {
        startup: Duration::from_millis(300),
        ..test_limits()
    };
    let core = root.core_with("slow-start", limits);
    let events = run(&core, RunMode::Submit);

    let (verdict, passed, total, submission_id) = end_of(&events);
    assert_eq!((verdict, passed, total), (Verdict::TimeLimit, 0, 0));
    assert!(
        submission_id.is_some(),
        "a timed-out submit is still an attempt"
    );
    assert!(logs(&events).iter().any(|l| l == "resolving environment"));
    assert!(
        logs(&events)
            .iter()
            .any(|l| l.starts_with("Starting the judge exceeded"))
    );
}

#[test]
fn a_harness_that_lingers_after_end_is_killed_without_changing_the_verdict() {
    let root = TestRoot::new();
    let core = root.core("linger");
    let began = Instant::now();
    let events = run(&core, RunMode::Run);
    assert_eq!(end_of(&events), (Verdict::Accepted, 2, 2, None));
    assert!(
        began.elapsed() < Duration::from_secs(5),
        "end must not wait for the process to exit"
    );
    // The lingering process is killed shortly after; wait for that, so the test proves
    // it and leaves nothing running.
    let marker = root.path().display().to_string();
    let deadline = Instant::now() + Duration::from_secs(10);
    loop {
        let listing = std::process::Command::new("ps")
            .args(["-A", "-o", "command="])
            .output()
            .expect("ps");
        let listing = String::from_utf8_lossy(&listing.stdout).into_owned();
        if !listing.lines().any(|line| line.contains(&marker)) {
            break;
        }
        assert!(
            Instant::now() < deadline,
            "the lingering harness was not killed"
        );
        std::thread::sleep(Duration::from_millis(20));
    }
}

#[test]
fn start_run_rejects_bad_requests_without_emitting_events() {
    let root = TestRoot::new();
    let core = root.core("happy");
    let collector: Collector<RunEvent> = Collector::new();

    assert!(matches!(
        core.start_run(
            "nope",
            Framework::Torch,
            RunMode::Run,
            LOCAL_TARGET,
            collector.sink()
        ),
        Err(CoreError::ProblemNotFound(_))
    ));
    assert!(matches!(
        core.start_run(
            "add-two",
            Framework::Mlx,
            RunMode::Run,
            LOCAL_TARGET,
            collector.sink()
        ),
        Err(CoreError::FrameworkNotSupported { .. })
    ));
    assert!(matches!(
        core.start_run(
            "add-two",
            Framework::Torch,
            RunMode::Run,
            "gpu-box",
            collector.sink()
        ),
        Err(CoreError::UnknownTarget(_))
    ));
    std::thread::sleep(Duration::from_millis(50));
    assert!(collector.snapshot().is_empty());
}

#[test]
fn concurrent_runs_are_independent() {
    let root = TestRoot::new();
    let core = root.core("happy");
    let collectors: Vec<Collector<RunEvent>> = (0..4).map(|_| Collector::new()).collect();
    let ids: Vec<String> = collectors
        .iter()
        .map(|c| {
            core.start_run(
                "add-two",
                Framework::Torch,
                RunMode::Submit,
                LOCAL_TARGET,
                c.sink(),
            )
            .expect("start")
        })
        .collect();
    let unique: std::collections::HashSet<_> = ids.iter().collect();
    assert_eq!(unique.len(), 4);
    for collector in &collectors {
        assert_eq!(end_of(&collector.wait_for_end()).0, Verdict::Accepted);
    }
    assert_eq!(core.list_submissions(None).expect("list").len(), 4);
}

#[test]
fn the_real_uv_command_line_reaches_the_harness_with_the_saved_solution() {
    let root = TestRoot::new();
    let core = root.core_with_fake_uv();
    // The fake harness reads its scenario from the solution file it is given.
    core.write_solution("add-two", Framework::Torch, "# scenario: fail\n")
        .expect("write");
    let events = run(&core, RunMode::Run);
    assert_eq!(end_of(&events), (Verdict::WrongAnswer, 1, 2, None));

    let log = fs::read_to_string(root.path().join("python/.fake-uv.log")).expect("uv log");
    assert_eq!(log.lines().last(), Some("run"), "no extras are enabled yet");
}

#[test]
fn a_hardware_missing_submit_is_stored_with_its_skip_and_solves_nothing() {
    let root = TestRoot::new();
    let core = root.core("hardware-missing");
    let events = run(&core, RunMode::Submit);

    let RunEvent::Start {
        cases,
        requires,
        requires_met,
        ..
    } = events
        .iter()
        .find(|e| matches!(e, RunEvent::Start { .. }))
        .expect("start")
    else {
        unreachable!()
    };
    assert_eq!((*requires, *requires_met), (Requires::Cuda, false));
    assert_eq!(cases[0].requires, None);
    assert_eq!(cases[1].requires, Some(Requires::Cuda));

    let Some(RunEvent::End {
        verdict,
        passed,
        total,
        skipped,
        missing_hardware,
        missing_assets,
        message,
        submission_id,
        ..
    }) = events.last()
    else {
        panic!("no end event")
    };
    assert_eq!(
        (*verdict, *passed, *total, *skipped),
        (Verdict::HardwareMissing, 1, 2, 1)
    );
    assert_eq!(*missing_hardware, Some(Requires::Cuda));
    assert_eq!(*missing_assets, None);
    assert!(message.as_deref().expect("message").contains("NVIDIA GPU"));

    // Stored like any other submit, with the count and the reason for the skip.
    let detail = core
        .get_submission(submission_id.expect("stored"))
        .expect("detail");
    assert_eq!(detail.submission.verdict, Verdict::HardwareMissing);
    assert_eq!(detail.submission.skipped, 1);
    let skipped_case = &detail.cases[1];
    assert_eq!(skipped_case.name, "On the GPU");
    assert_eq!(skipped_case.result.status, CaseStatus::Skipped);
    assert_eq!(
        skipped_case.result.skip,
        Some(Skip::Hardware {
            requires: Requires::Cuda
        })
    );

    // Attempted, never solved; no stats, no streak.
    let problem = core.get_problem("add-two").expect("problem");
    assert_eq!(problem.summary.status, ProblemStatus::Attempted);
    assert!(problem.summary.solved_frameworks.is_empty());
    assert!(!problem.editorial_unlocked);
    let stats = core.get_stats().expect("stats");
    assert_eq!((stats.solved, stats.streak_days), (0, 0));
    assert_eq!(core.list_tracks().expect("tracks")[0].solved_count, 0);
}

#[test]
fn an_assets_missing_run_names_the_asset_and_a_submit_never_counts_as_solved() {
    let root = TestRoot::new();
    let core = root.core("assets-missing");

    // A run is not stored, but reports the same outcome.
    let events = run(&core, RunMode::Run);
    let Some(RunEvent::End {
        verdict,
        passed,
        total,
        skipped,
        missing_assets,
        metrics,
        submission_id,
        ..
    }) = events.last()
    else {
        panic!("no end event")
    };
    assert_eq!(
        (*verdict, *passed, *total, *skipped, *submission_id),
        (Verdict::AssetsMissing, 1, 2, 1, None)
    );
    assert_eq!(
        missing_assets.as_deref(),
        Some(
            &[MissingAsset {
                name: "tiny".into(),
                repo: "fake/tiny".into(),
                size_mb: 0.5
            }][..]
        )
    );
    assert!(metrics[0].higher_is_better);
    let skip = events.iter().find_map(|e| match e {
        RunEvent::CaseEnd { result } => result.skip.clone(),
        _ => None,
    });
    assert_eq!(
        skip,
        Some(Skip::Asset {
            asset: "tiny".into()
        })
    );

    let events = run(&core, RunMode::Submit);
    let (verdict, passed, total, submission_id) = end_of(&events);
    assert_eq!((verdict, passed, total), (Verdict::AssetsMissing, 2, 3));
    let listed = core.list_submissions(Some("add-two")).expect("list");
    assert_eq!(listed.len(), 1);
    assert_eq!(listed[0].id, submission_id.expect("stored"));
    assert_eq!(listed[0].skipped, 1);
    let summary = &core.list_problems().expect("problems")[0];
    assert_eq!(summary.status, ProblemStatus::Attempted);
    assert_eq!(core.get_stats().expect("stats").solved, 0);

    // The problem declares the asset the run asked for.
    assert_eq!(
        core.get_problem("add-two").expect("problem").summary.assets,
        ["tiny"]
    );
}

#[test]
fn skipped_cases_are_counted_when_core_ends_the_run_itself() {
    // The import-error scenario reports no `skipped` count; core counts the results.
    let root = TestRoot::new();
    let core = root.core("import-error");
    let events = run(&core, RunMode::Run);
    let Some(RunEvent::End { skipped, .. }) = events.last() else {
        panic!("no end event")
    };
    assert_eq!(*skipped, 2);
}
