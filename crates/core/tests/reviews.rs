//! Spaced re-solve end to end against the fake harness: scheduling from accepted
//! submits, the review file and its lifecycle, review runs (local and on an SSH
//! target), resolving a review, the on/off setting, and case stages.
//!
//! Time comes from a test clock that the tests move; nothing here sleeps to wait for a
//! due date.

mod common;

use std::fs;
use std::sync::{Arc, Mutex};

use chrono::{DateTime, Duration, Utc};
use common::{Collector, FakeRemote, TestRoot, fixtures, test_limits};
use whetstone_core::{
    Clock, Core, CoreError, CoreOptions, Framework, LOCAL_TARGET, ProgramHarness, ReviewAction,
    RunEvent, RunMode, Verdict,
};

const PROBLEM: &str = "add-two";
const TORCH: Framework = Framework::Torch;

/// A clock the test sets and advances.
#[derive(Clone)]
struct TestClock(Arc<Mutex<DateTime<Utc>>>);

impl TestClock {
    fn new() -> Self {
        Self(Arc::new(Mutex::new(
            "2026-11-01T10:00:00Z".parse().expect("valid time"),
        )))
    }

    fn advance_days(&self, days: i64) {
        *self.0.lock().expect("clock lock") += Duration::days(days);
    }

    fn iso(&self) -> String {
        self.now()
            .to_rfc3339_opts(chrono::SecondsFormat::Millis, true)
    }
}

impl Clock for TestClock {
    fn now(&self) -> DateTime<Utc> {
        *self.0.lock().expect("clock lock")
    }
}

/// A core whose fake harness picks its scenario from the solution file it is given, so
/// a test can tell which file was judged.
fn core(root: &TestRoot, clock: &TestClock) -> Core {
    Core::open(CoreOptions {
        harness: Some(Arc::new(
            ProgramHarness::new("sh").arg(fixtures().join("fake_harness.sh")),
        )),
        uv_program: Some(fixtures().join("fake_uv.sh")),
        limits: test_limits(),
        clock: Some(Arc::new(clock.clone())),
        ..CoreOptions::new(root.path())
    })
    .expect("open core")
}

fn solution(scenario: &str) -> String {
    format!("# scenario: {scenario}\ndef add(a, b):\n    return a + b\n")
}

fn finish(collector: &Collector<RunEvent>) -> (Verdict, Option<i64>) {
    match collector.wait_for_end().last() {
        Some(RunEvent::End {
            verdict,
            submission_id,
            ..
        }) => (*verdict, *submission_id),
        other => panic!("not an end event: {other:?}"),
    }
}

fn submit(core: &Core, mode: RunMode) -> (Verdict, Option<i64>) {
    let collector = Collector::new();
    core.start_run(PROBLEM, TORCH, mode, LOCAL_TARGET, collector.sink())
        .expect("start run");
    finish(&collector)
}

fn review_run(core: &Core, mode: RunMode, target: &str) -> (Verdict, Option<i64>) {
    let collector = Collector::new();
    core.start_review_run(PROBLEM, TORCH, mode, target, collector.sink())
        .expect("start review run");
    finish(&collector)
}

/// Solves the problem with a normal accepted submit.
fn solve(core: &Core) {
    core.write_solution(PROBLEM, TORCH, &solution("happy"))
        .expect("write solution");
    assert_eq!(submit(core, RunMode::Submit).0, Verdict::Accepted);
}

fn review_file(root: &TestRoot) -> std::path::PathBuf {
    root.path().join("workspace/add-two/review/torch.py")
}

#[test]
fn the_first_accepted_submit_schedules_a_review_three_days_later() {
    let (root, clock) = (TestRoot::new(), TestClock::new());
    let core = core(&root, &clock);
    assert!(core.list_reviews().expect("list").is_empty());

    // A failed submit and an accepted `run` schedule nothing.
    core.write_solution(PROBLEM, TORCH, &solution("fail"))
        .expect("write");
    assert_eq!(submit(&core, RunMode::Submit).0, Verdict::WrongAnswer);
    core.write_solution(PROBLEM, TORCH, &solution("happy"))
        .expect("write");
    assert_eq!(submit(&core, RunMode::Run).0, Verdict::Accepted);
    assert!(core.list_reviews().expect("list").is_empty());

    assert_eq!(submit(&core, RunMode::Submit).0, Verdict::Accepted);
    let reviews = core.list_reviews().expect("list");
    assert_eq!(reviews.len(), 1);
    let review = &reviews[0];
    assert_eq!(
        (
            review.problem_id.as_str(),
            review.stage,
            review.interval_days
        ),
        (PROBLEM, 0, 3)
    );
    assert_eq!(review.due_at, "2026-11-04T10:00:00.000Z");
    assert_eq!(review.last_reviewed_at, None);
    assert!(!review.due);
    assert_eq!(core.get_stats().expect("stats").reviews_due, 0);

    // The submission is dated by the same clock and is not a review.
    let stored = &core.list_submissions(Some(PROBLEM)).expect("list")[0];
    assert_eq!(stored.created_at, "2026-11-01T10:00:00.000Z");
    assert!(!stored.review);

    clock.advance_days(3);
    assert!(core.list_reviews().expect("list")[0].due);
    assert_eq!(core.get_stats().expect("stats").reviews_due, 1);

    // A normal accepted submit while the review is due does not advance it.
    assert_eq!(submit(&core, RunMode::Submit).0, Verdict::Accepted);
    let after = core.list_reviews().expect("list");
    assert_eq!((after[0].stage, after[0].due), (0, true));
    assert_eq!(after[0].due_at, "2026-11-04T10:00:00.000Z");
}

#[test]
fn the_review_file_is_separate_and_review_runs_judge_it() {
    let (root, clock) = (TestRoot::new(), TestClock::new());
    let core = core(&root, &clock);
    solve(&core);
    clock.advance_days(4);
    let saved = root.path().join("workspace/add-two/torch.py");
    let saved_code = fs::read_to_string(&saved).expect("saved solution");

    // With no review file the review starts from the starter.
    let starter = fs::read_to_string(root.path().join("problems/basics/add-two/starter/torch.py"))
        .expect("starter");
    let blank = core.read_review_solution(PROBLEM, TORCH).expect("read");
    assert_eq!(
        (blank.code.as_str(), blank.is_starter),
        (starter.as_str(), true)
    );
    assert_eq!(blank.path, review_file(&root).to_string_lossy());
    assert!(!review_file(&root).exists());

    // A failing attempt: the review run judges it, not the accepted saved solution.
    core.write_review_solution(PROBLEM, TORCH, &solution("fail"))
        .expect("write review");
    assert_eq!(fs::read_to_string(&saved).expect("saved"), saved_code);
    assert_eq!(
        review_run(&core, RunMode::Run, LOCAL_TARGET).0,
        Verdict::WrongAnswer
    );
    let (verdict, id) = review_run(&core, RunMode::Submit, LOCAL_TARGET);
    assert_eq!(verdict, Verdict::WrongAnswer);
    let stored = core
        .get_submission(id.expect("stored"))
        .expect("submission");
    assert!(stored.submission.review);
    assert_eq!(stored.code, solution("fail"));
    // A non-accepted review submit changes nothing.
    let review = &core.list_reviews().expect("list")[0];
    assert_eq!((review.stage, review.due), (0, true));
    assert!(review_file(&root).is_file());

    // `start_review` is the explicit reset; the saved solution is still untouched.
    let reset = core.start_review(PROBLEM, TORCH).expect("start review");
    assert_eq!(
        (reset.code.as_str(), reset.is_starter),
        (starter.as_str(), true)
    );
    assert!(!review_file(&root).exists());
    assert_eq!(fs::read_to_string(&saved).expect("saved"), saved_code);

    // An accepted review `run` is ephemeral and advances nothing.
    core.write_review_solution(PROBLEM, TORCH, &solution("happy"))
        .expect("write review");
    assert_eq!(
        review_run(&core, RunMode::Run, LOCAL_TARGET),
        (Verdict::Accepted, None)
    );
    assert_eq!(core.list_reviews().expect("list")[0].stage, 0);

    // An accepted review submit advances, dates the next review from now, and removes
    // the attempt so the next review starts blank.
    let (verdict, id) = review_run(&core, RunMode::Submit, LOCAL_TARGET);
    assert_eq!(verdict, Verdict::Accepted);
    assert!(
        core.get_submission(id.expect("stored"))
            .expect("submission")
            .submission
            .review
    );
    let review = &core.list_reviews().expect("list")[0];
    assert_eq!(
        (review.stage, review.interval_days, review.due),
        (1, 7, false)
    );
    assert_eq!(review.due_at, "2026-11-12T10:00:00.000Z");
    assert_eq!(
        review.last_reviewed_at.as_deref(),
        Some(clock.iso().as_str())
    );
    assert!(!review_file(&root).exists());
    assert!(!root.path().join("workspace/add-two/review").exists());
    assert_eq!(fs::read_to_string(&saved).expect("saved"), saved_code);
    assert!(
        core.read_review_solution(PROBLEM, TORCH)
            .expect("read")
            .is_starter
    );
}

#[test]
fn five_accepted_reviews_master_the_problem() {
    let (root, clock) = (TestRoot::new(), TestClock::new());
    let core = core(&root, &clock);
    solve(&core);

    let mut gaps = Vec::new();
    for stage in 0..5 {
        let review = core.list_reviews().expect("list").remove(0);
        assert_eq!(review.stage, stage);
        let due: DateTime<Utc> = review.due_at.parse().expect("due time");
        gaps.push((due - clock.now()).num_days());
        clock.advance_days(gaps[stage as usize]);
        assert!(core.list_reviews().expect("list")[0].due);
        core.write_review_solution(PROBLEM, TORCH, &solution("happy"))
            .expect("write review");
        assert_eq!(
            review_run(&core, RunMode::Submit, LOCAL_TARGET).0,
            Verdict::Accepted
        );
    }
    assert_eq!(gaps, [3, 7, 21, 60, 180]);

    // Mastered: no active review, nothing due, and no further review run.
    assert!(core.list_reviews().expect("list").is_empty());
    assert_eq!(core.get_stats().expect("stats").reviews_due, 0);
    let collector: Collector<RunEvent> = Collector::new();
    assert!(matches!(
        core.start_review_run(
            PROBLEM,
            TORCH,
            RunMode::Submit,
            LOCAL_TARGET,
            collector.sink()
        ),
        Err(CoreError::NoActiveReview(_))
    ));
    // Solving it again does not start the schedule over.
    assert_eq!(submit(&core, RunMode::Submit).0, Verdict::Accepted);
    assert!(core.list_reviews().expect("list").is_empty());
    // Six accepted submits in all; the five reviews are marked.
    let marks: Vec<bool> = core
        .list_submissions(Some(PROBLEM))
        .expect("list")
        .iter()
        .map(|s| s.review)
        .collect();
    assert_eq!(marks, [false, true, true, true, true, true, false]);
}

#[test]
fn forgot_snooze_and_retire_resolve_a_review() {
    let (root, clock) = (TestRoot::new(), TestClock::new());
    let core = core(&root, &clock);
    solve(&core);
    clock.advance_days(3);
    core.write_review_solution(PROBLEM, TORCH, &solution("happy"))
        .expect("write review");
    assert_eq!(
        review_run(&core, RunMode::Submit, LOCAL_TARGET).0,
        Verdict::Accepted
    );
    clock.advance_days(10);

    // Snooze: one day from now (it was overdue), same stage, and the attempt is kept.
    core.write_review_solution(PROBLEM, TORCH, "attempt\n")
        .expect("write review");
    let reviews = core
        .resolve_review(PROBLEM, ReviewAction::Snooze)
        .expect("snooze");
    assert_eq!((reviews[0].stage, reviews[0].due), (1, false));
    assert_eq!(reviews[0].due_at, "2026-11-15T10:00:00.000Z");
    assert_eq!(
        fs::read_to_string(review_file(&root)).expect("attempt"),
        "attempt\n"
    );

    // Forgot: back to the first interval, due tomorrow, and the attempt is removed.
    clock.advance_days(1);
    let reviews = core
        .resolve_review(PROBLEM, ReviewAction::Forgot)
        .expect("forgot");
    assert_eq!(
        (reviews[0].stage, reviews[0].interval_days, reviews[0].due),
        (0, 3, false)
    );
    assert_eq!(reviews[0].due_at, "2026-11-16T10:00:00.000Z");
    assert!(!review_file(&root).exists());

    // Retire: reviews end; the saved solution is still there.
    core.write_review_solution(PROBLEM, TORCH, "attempt\n")
        .expect("write review");
    assert!(
        core.resolve_review(PROBLEM, ReviewAction::Retire)
            .expect("retire")
            .is_empty()
    );
    assert!(!review_file(&root).exists());
    assert!(root.path().join("workspace/add-two/torch.py").is_file());

    // Nothing is left to resolve, write, or restart.
    for result in [
        core.resolve_review(PROBLEM, ReviewAction::Snooze)
            .map(|_| ()),
        core.write_review_solution(PROBLEM, TORCH, "x"),
        core.start_review(PROBLEM, TORCH).map(|_| ()),
        core.resolve_review("hard-one", ReviewAction::Forgot)
            .map(|_| ()),
    ] {
        assert!(
            matches!(result, Err(CoreError::NoActiveReview(_))),
            "{result:?}"
        );
    }
}

#[test]
fn turning_reviews_off_hides_and_keeps_schedules() {
    let (root, clock) = (TestRoot::new(), TestClock::new());
    let core = core(&root, &clock);
    assert!(core.get_settings().expect("settings").reviews_enabled);
    solve(&core);
    clock.advance_days(5);
    assert_eq!(core.get_stats().expect("stats").reviews_due, 1);

    assert!(
        !core
            .set_reviews_enabled(false)
            .expect("disable")
            .reviews_enabled
    );
    assert!(core.list_reviews().expect("list").is_empty());
    assert_eq!(core.get_stats().expect("stats").reviews_due, 0);
    let collector: Collector<RunEvent> = Collector::new();
    assert!(matches!(
        core.start_review_run(PROBLEM, TORCH, RunMode::Run, LOCAL_TARGET, collector.sink()),
        Err(CoreError::ReviewsDisabled)
    ));
    assert!(matches!(
        core.resolve_review(PROBLEM, ReviewAction::Retire),
        Err(CoreError::ReviewsDisabled)
    ));

    // A problem solved while reviews are off is not scheduled...
    core.write_solution("hard-one", TORCH, &solution("happy"))
        .expect("write");
    let collector = Collector::new();
    core.start_run(
        "hard-one",
        TORCH,
        RunMode::Submit,
        LOCAL_TARGET,
        collector.sink(),
    )
    .expect("start run");
    assert_eq!(finish(&collector).0, Verdict::Accepted);

    // ...until they are turned on again: the old schedule is as it was, and the new
    // problem is scheduled from its first accepted submit. The setting persists.
    drop(core);
    let core = self::core(&root, &clock);
    assert!(!core.get_settings().expect("settings").reviews_enabled);
    assert!(
        core.set_reviews_enabled(true)
            .expect("enable")
            .reviews_enabled
    );
    let reviews = core.list_reviews().expect("list");
    let summary: Vec<_> = reviews
        .iter()
        .map(|r| (r.problem_id.as_str(), r.due_at.as_str(), r.due))
        .collect();
    assert_eq!(
        summary,
        [
            (PROBLEM, "2026-11-04T10:00:00.000Z", true),
            ("hard-one", "2026-11-09T10:00:00.000Z", false),
        ]
    );
}

#[test]
fn a_review_run_on_an_ssh_target_syncs_and_judges_the_review_file() {
    let (root, remote, clock) = (TestRoot::new(), FakeRemote::new(), TestClock::new());
    let core = Core::open(CoreOptions {
        uv_program: Some(fixtures().join("fake_uv.sh")),
        limits: test_limits(),
        ssh: remote.ssh_options(),
        clock: Some(Arc::new(clock.clone())),
        ..CoreOptions::new(root.path())
    })
    .expect("open core");
    core.add_target("box", None).expect("add target");
    const BOX: &str = "ssh:box";

    // Solve it remotely, so the saved solution is on the remote host.
    core.write_solution(PROBLEM, TORCH, &solution("happy"))
        .expect("write");
    let collector = Collector::new();
    core.start_run(PROBLEM, TORCH, RunMode::Submit, BOX, collector.sink())
        .expect("start run");
    assert_eq!(finish(&collector).0, Verdict::Accepted);
    clock.advance_days(3);

    // The review attempt fails where the saved solution passes: the verdict shows
    // which file the remote harness judged.
    core.write_review_solution(PROBLEM, TORCH, &solution("fail"))
        .expect("write review");
    let (verdict, id) = review_run(&core, RunMode::Submit, BOX);
    assert_eq!(verdict, Verdict::WrongAnswer);
    let stored = core
        .get_submission(id.expect("stored"))
        .expect("submission");
    assert!(stored.submission.review);
    assert_eq!(stored.submission.target, BOX);

    let remote_root = remote.root();
    assert_eq!(
        fs::read_to_string(remote_root.join("workspace/add-two/review/torch.py")).expect("review"),
        solution("fail")
    );
    // The saved solution on the remote host was not replaced by the attempt.
    assert_eq!(
        fs::read_to_string(remote_root.join("workspace/add-two/torch.py")).expect("saved"),
        solution("happy")
    );
    let root_dir = remote_root.display().to_string();
    assert!(
        remote.ssh_log().iter().any(|line| line.contains(&format!(
            "'--solution' '{root_dir}/workspace/add-two/review/torch.py'"
        ))),
        "{:?}",
        remote.ssh_log()
    );

    // An accepted remote review submit advances the schedule like a local one.
    core.write_review_solution(PROBLEM, TORCH, &solution("happy"))
        .expect("write review");
    assert_eq!(review_run(&core, RunMode::Submit, BOX).0, Verdict::Accepted);
    assert_eq!(core.list_reviews().expect("list")[0].stage, 1);
    assert!(!review_file(&root).exists());
}

#[test]
fn case_stages_travel_from_the_harness_to_the_stored_submission() {
    let (root, clock) = (TestRoot::new(), TestClock::new());
    let core = core(&root, &clock);
    core.write_solution(PROBLEM, TORCH, &solution("staged"))
        .expect("write");
    let collector = Collector::new();
    core.start_run(
        PROBLEM,
        TORCH,
        RunMode::Submit,
        LOCAL_TARGET,
        collector.sink(),
    )
    .expect("start run");
    let events = collector.wait_for_end();

    let Some(RunEvent::Start { cases, .. }) =
        events.iter().find(|e| matches!(e, RunEvent::Start { .. }))
    else {
        panic!("no start event: {events:?}")
    };
    let announced: Vec<_> = cases
        .iter()
        .map(|c| (c.id.as_str(), c.public, c.stage.as_deref()))
        .collect();
    assert_eq!(
        announced,
        [
            ("t1", true, Some("1: Tokenizer")),
            // A stage label is public even for a hidden case.
            ("hidden-2", false, Some("1: Tokenizer")),
            ("s1", true, Some("Stage 2: Sampler")),
        ]
    );
    // On the wire the field is `stage`, next to the other case fields.
    let wire = serde_json::to_value(&events).expect("json");
    let start = wire
        .as_array()
        .expect("array")
        .iter()
        .find(|e| e["type"] == "start")
        .expect("start");
    assert_eq!(start["cases"][2]["stage"], "Stage 2: Sampler");

    // Stages gate nothing: every case ran, and the verdict comes from all of them.
    let Some(RunEvent::End {
        verdict,
        passed,
        total,
        submission_id,
        ..
    }) = events.last()
    else {
        panic!("no end event")
    };
    assert_eq!((*verdict, *passed, *total), (Verdict::WrongAnswer, 2, 3));

    let stored = core
        .get_submission(submission_id.expect("stored"))
        .expect("submission");
    let stages: Vec<_> = stored
        .cases
        .iter()
        .map(|c| (c.result.id.as_str(), c.stage.as_deref()))
        .collect();
    assert_eq!(
        stages,
        [
            ("t1", Some("1: Tokenizer")),
            ("hidden-2", Some("1: Tokenizer")),
            ("s1", Some("Stage 2: Sampler")),
        ]
    );

    // An unstaged run stores no stage at all.
    core.write_solution(PROBLEM, TORCH, &solution("happy"))
        .expect("write");
    let (_, id) = submit(&core, RunMode::Submit);
    let plain = core
        .get_submission(id.expect("stored"))
        .expect("submission");
    assert!(plain.cases.iter().all(|c| c.stage.is_none()));
    let json = serde_json::to_value(&plain.cases).expect("json");
    assert!(json[0].get("stage").is_none());
}
