//! Skills: loading, validation, catalog issues, when a skill is earned, which skills a
//! submit earned, and the shipped content.

mod common;

use std::fs;
use std::path::Path;

use common::TestRoot;
use learnai_core::Core;
use learnai_core::catalog::CatalogLoader;
use learnai_core::skills::load_skills;

fn write_skill(root: &Path, name: &str, text: &str) {
    let dir = root.join("skills");
    fs::create_dir_all(&dir).expect("mkdir");
    fs::write(dir.join(name), text).expect("write");
}

/// A skill file with the required keys; `extra` adds or follows them.
fn skill(id: &str, problems: &[&str], row: u32, x: u32, extra: &str) -> String {
    format!(
        "id = \"{id}\"\ntitle = \"Title of {id}\"\nname = \"Name of {id}\"\n\
         problems = {problems:?}\nrow = {row}\nx = {x}\n{extra}"
    )
}

/// Three skills in a chain, one fixture problem each: `first`, then `second`, then
/// `third`. Only the `earning` tests use it, and they are built on Unix alone.
#[cfg(unix)]
fn write_chain(root: &Path) {
    write_skill(
        root,
        "first.toml",
        &skill("first", &["add-two"], 0, 500, ""),
    );
    write_skill(
        root,
        "second.toml",
        &skill(
            "second",
            &["echo-server"],
            1,
            500,
            "requires = [\"first\"]\n",
        ),
    );
    write_skill(
        root,
        "third.toml",
        &skill("third", &["hard-one"], 2, 500, "requires = [\"second\"]\n"),
    );
}

/// The messages of the issues reported for one skill file, in order.
fn issues_of(core: &Core, file: &str) -> Vec<String> {
    core.list_catalog_issues()
        .into_iter()
        .filter(|i| Path::new(&i.path).ends_with(format!("skills/{file}")))
        .map(|i| i.message)
        .collect()
}

fn ids(core: &Core) -> Vec<String> {
    core.list_skills()
        .expect("skills")
        .into_iter()
        .map(|s| s.id)
        .collect()
}

const VALID: &str = r##"
id = "starter"
title = "Sure Footing"
name = "Stable Numerics"
summary = "A short skill."
art = "trail-start"
color = "#ED9B11"
requires = []
problems = ["add-two", "echo-server"]
row = 2
x = 490
lanes = [[3, 480], [4, 470]]
"##;

#[test]
fn no_skills_directory_means_no_skills_and_no_issues() {
    let root = TestRoot::new();
    let core = root.core("happy");
    let before = core.catalog_issues().len();
    assert!(core.list_skills().expect("skills").is_empty());
    assert_eq!(core.catalog_issues().len(), before);
}

#[test]
fn a_valid_skill_loads_with_every_field() {
    let root = TestRoot::new();
    let core = root.core("happy");
    let before = core.catalog_issues().len();
    write_skill(root.path(), "starter.toml", VALID);

    let skills = core.list_skills().expect("skills");
    assert_eq!(skills.len(), 1);
    let skill = &skills[0];
    assert_eq!(skill.id, "starter");
    assert_eq!(skill.title, "Sure Footing");
    assert_eq!(skill.name, "Stable Numerics");
    assert_eq!(skill.summary, "A short skill.");
    assert_eq!(skill.art.as_deref(), Some("trail-start"));
    // Stored lowercase.
    assert_eq!(skill.color.as_deref(), Some("#ed9b11"));
    assert!(skill.requires.is_empty());
    assert_eq!(skill.problems, ["add-two", "echo-server"]);
    assert_eq!((skill.row, skill.x), (2, 490));
    assert_eq!(skill.lanes, [[3, 480], [4, 470]]);
    assert_eq!(skill.earned_at, None);
    assert_eq!(core.catalog_issues().len(), before);
}

#[test]
fn the_optional_keys_may_be_left_out() {
    let root = TestRoot::new();
    let core = root.core("happy");
    let before = core.catalog_issues().len();
    write_skill(
        root.path(),
        "bare.toml",
        &skill("bare", &["add-two"], 0, 10, ""),
    );

    let skills = core.list_skills().expect("skills");
    let skill = &skills[0];
    assert_eq!((&skill.art, &skill.color), (&None, &None));
    assert_eq!(skill.summary, "");
    assert!(skill.requires.is_empty() && skill.lanes.is_empty());
    assert_eq!(core.catalog_issues().len(), before);
}

#[test]
fn load_skills_returns_skills_in_file_name_order() {
    let root = TestRoot::new();
    write_skill(
        root.path(),
        "b-low.toml",
        &skill("b-low", &["add-two"], 0, 10, ""),
    );
    write_skill(
        root.path(),
        "a-high.toml",
        &skill("a-high", &["add-two"], 5, 10, ""),
    );
    let catalog = CatalogLoader::new(root.path()).load();
    let (skills, issues) = load_skills(root.path(), &catalog);
    assert!(issues.is_empty(), "{issues:?}");
    assert_eq!(
        skills.iter().map(|s| s.id.as_str()).collect::<Vec<_>>(),
        ["a-high", "b-low"]
    );
}

#[test]
fn list_skills_is_sorted_by_row_then_x() {
    let root = TestRoot::new();
    let core = root.core("happy");
    for (id, row, x) in [("a", 1, 300), ("b", 0, 900), ("c", 1, 100), ("d", 0, 200)] {
        write_skill(
            root.path(),
            &format!("{id}.toml"),
            &skill(id, &["add-two"], row, x, ""),
        );
    }
    assert_eq!(ids(&core), ["d", "b", "c", "a"]);
}

#[test]
fn malformed_skill_files_are_reported_and_never_break_the_others() {
    let root = TestRoot::new();
    let core = root.core("happy");
    write_skill(root.path(), "starter.toml", VALID);
    write_skill(root.path(), "broken.toml", "id = \"broken\"\ntitle = ");
    write_skill(
        root.path(),
        "misnamed.toml",
        &skill("other", &["add-two"], 0, 0, ""),
    );
    write_skill(
        root.path(),
        "untitled.toml",
        "id = \"untitled\"\ntitle = \" \"\nname = \"N\"\nproblems = [\"add-two\"]\nrow = 0\nx = 0\n",
    );
    write_skill(
        root.path(),
        "unnamed.toml",
        "id = \"unnamed\"\ntitle = \"T\"\nname = \"\"\nproblems = [\"add-two\"]\nrow = 0\nx = 0\n",
    );
    write_skill(
        root.path(),
        "Bad Name.toml",
        &skill("Bad Name", &["add-two"], 0, 0, ""),
    );
    // `row` and `x` are required, and are non-negative integers.
    write_skill(
        root.path(),
        "unplaced.toml",
        "id = \"unplaced\"\ntitle = \"T\"\nname = \"N\"\nproblems = [\"add-two\"]\nx = 0\n",
    );
    write_skill(
        root.path(),
        "negative.toml",
        "id = \"negative\"\ntitle = \"T\"\nname = \"N\"\nproblems = [\"add-two\"]\nrow = 0\nx = -5\n",
    );
    write_skill(
        root.path(),
        "fraction.toml",
        "id = \"fraction\"\ntitle = \"T\"\nname = \"N\"\nproblems = [\"add-two\"]\nrow = 1.5\nx = 0\n",
    );
    write_skill(root.path(), "README.md", "notes\n");
    write_skill(root.path(), "_draft.toml", "not even toml [");
    // A directory named like a file is not a skill file.
    fs::create_dir_all(root.path().join("skills").join("nested.toml")).expect("mkdir");

    assert_eq!(ids(&core), ["starter"]);
    let issues = core.list_catalog_issues();
    let message = |file: &str| {
        issues
            .iter()
            .find(|i| i.path.ends_with(file))
            .unwrap_or_else(|| panic!("no issue for {file}: {issues:?}"))
            .message
            .as_str()
    };
    assert!(!message("broken.toml").is_empty());
    assert!(message("misnamed.toml").contains("does not equal the file name"));
    assert!(message("untitled.toml").contains("title is empty"));
    assert!(message("unnamed.toml").contains("name is empty"));
    assert!(message("Bad Name.toml").contains("must use only"));
    assert!(message("unplaced.toml").contains("row"));
    assert!(!message("negative.toml").is_empty());
    assert!(!message("fraction.toml").is_empty());
    assert!(!issues.iter().any(|i| i.path.ends_with("README.md")
        || i.path.ends_with("_draft.toml")
        || i.path.ends_with("nested.toml")));
}

#[test]
fn unknown_and_repeated_problems_are_dropped_and_reported() {
    let root = TestRoot::new();
    let core = root.core("happy");
    let before = core.catalog_issues().len();
    write_skill(
        root.path(),
        "starter.toml",
        &skill(
            "starter",
            &["add-two", "nope", "echo-server", "add-two"],
            0,
            0,
            "",
        ),
    );

    let skills = core.list_skills().expect("skills");
    assert_eq!(skills[0].problems, ["add-two", "echo-server"]);
    assert_eq!(core.catalog_issues().len(), before + 2);
    let mine = issues_of(&core, "starter.toml");
    assert!(mine[0].contains("`nope`") && mine[0].contains("not a known problem"));
    assert!(mine[1].contains("`add-two` again"));
}

#[test]
fn a_skill_with_no_usable_problems_is_dropped_and_cannot_be_required() {
    let root = TestRoot::new();
    let core = root.core("happy");
    write_skill(
        root.path(),
        "a-empty.toml",
        &skill("a-empty", &["ghost"], 0, 0, ""),
    );
    write_skill(
        root.path(),
        "b-none.toml",
        &skill("b-none", &[], 0, 100, ""),
    );
    write_skill(
        root.path(),
        "c-later.toml",
        &skill("c-later", &["add-two"], 1, 0, "requires = [\"a-empty\"]\n"),
    );

    let skills = core.list_skills().expect("skills");
    assert_eq!(skills.len(), 1);
    assert_eq!(skills[0].id, "c-later");
    assert!(skills[0].requires.is_empty());

    let empty = issues_of(&core, "a-empty.toml");
    assert!(empty[0].contains("`ghost`"));
    assert!(empty[1].contains("no usable problems"));
    assert!(issues_of(&core, "b-none.toml")[0].contains("no usable problems"));
    let later = issues_of(&core, "c-later.toml");
    assert_eq!(later.len(), 1, "{later:?}");
    assert!(later[0].contains("`a-empty`") && later[0].contains("not a usable skill"));
}

#[test]
fn unknown_self_and_repeated_requirements_are_dropped_and_reported() {
    let root = TestRoot::new();
    let core = root.core("happy");
    write_skill(
        root.path(),
        "base.toml",
        &skill("base", &["add-two"], 0, 0, ""),
    );
    write_skill(
        root.path(),
        "next.toml",
        &skill(
            "next",
            &["echo-server"],
            1,
            0,
            "requires = [\"base\", \"unknown\", \"next\", \"base\"]\n",
        ),
    );

    let skills = core.list_skills().expect("skills");
    assert_eq!(skills[1].id, "next");
    assert_eq!(skills[1].requires, ["base"]);
    let mine = issues_of(&core, "next.toml");
    assert_eq!(mine.len(), 3, "{mine:?}");
    assert!(mine[0].contains("`unknown`") && mine[0].contains("not a usable skill"));
    assert!(mine[1].contains("requires itself"));
    assert!(mine[2].contains("`base` again"));
}

#[test]
fn a_skill_may_require_one_whose_file_sorts_later() {
    let root = TestRoot::new();
    let core = root.core("happy");
    let before = core.catalog_issues().len();
    write_skill(
        root.path(),
        "a-below.toml",
        &skill(
            "a-below",
            &["echo-server"],
            1,
            0,
            "requires = [\"z-above\"]\n",
        ),
    );
    write_skill(
        root.path(),
        "z-above.toml",
        &skill("z-above", &["add-two"], 0, 0, ""),
    );
    let skills = core.list_skills().expect("skills");
    assert_eq!(skills[1].requires, ["z-above"]);
    assert_eq!(core.catalog_issues().len(), before);
}

#[test]
fn a_requirement_that_would_close_a_cycle_is_dropped_and_reported() {
    let root = TestRoot::new();
    let core = root.core("happy");
    // a builds on b, b on c, and c on a: the last edge, in file-name order, is refused.
    write_skill(
        root.path(),
        "a.toml",
        &skill("a", &["add-two"], 2, 0, "requires = [\"b\"]\n"),
    );
    write_skill(
        root.path(),
        "b.toml",
        &skill("b", &["echo-server"], 1, 0, "requires = [\"c\"]\n"),
    );
    write_skill(
        root.path(),
        "c.toml",
        &skill("c", &["hard-one"], 0, 0, "requires = [\"a\"]\n"),
    );

    let skills = core.list_skills().expect("skills");
    let requires = |id: &str| {
        skills
            .iter()
            .find(|s| s.id == id)
            .expect("skill")
            .requires
            .clone()
    };
    assert_eq!(requires("a"), ["b"]);
    assert_eq!(requires("b"), ["c"]);
    assert!(requires("c").is_empty());
    assert!(issues_of(&core, "a.toml").is_empty());
    assert!(issues_of(&core, "b.toml").is_empty());
    let mine = issues_of(&core, "c.toml");
    assert_eq!(mine.len(), 1, "{mine:?}");
    assert!(mine[0].contains("`a`") && mine[0].contains("already builds on `c`"));
}

#[test]
fn two_skills_that_require_each_other_keep_the_first_edge() {
    let root = TestRoot::new();
    let core = root.core("happy");
    write_skill(
        root.path(),
        "a.toml",
        &skill("a", &["add-two"], 1, 0, "requires = [\"b\"]\n"),
    );
    write_skill(
        root.path(),
        "b.toml",
        &skill("b", &["echo-server"], 0, 0, "requires = [\"a\"]\n"),
    );
    let skills = core.list_skills().expect("skills");
    assert!(skills[0].id == "b" && skills[0].requires.is_empty());
    assert_eq!(skills[1].requires, ["b"]);
    assert!(issues_of(&core, "b.toml")[0].contains("already builds on `b`"));
}

#[test]
fn a_bad_art_color_or_lane_is_dropped_and_reported() {
    let root = TestRoot::new();
    let core = root.core("happy");
    write_skill(
        root.path(),
        "dressed.toml",
        &skill(
            "dressed",
            &["add-two"],
            0,
            0,
            "art = \"Not A Slug\"\ncolor = \"red\"\n\
             lanes = [[1, 100], [2], [3, -4], [4, 5, 6], \"x\", [5, 1.5], [6, 600]]\n",
        ),
    );

    let skills = core.list_skills().expect("skills");
    assert_eq!((&skills[0].art, &skills[0].color), (&None, &None));
    assert_eq!(skills[0].lanes, [[1, 100], [6, 600]]);
    let mine = issues_of(&core, "dressed.toml");
    assert_eq!(mine.len(), 7, "{mine:?}");
    assert!(mine[0].contains("art `Not A Slug`"));
    assert!(mine[1].contains("color `red`"));
    assert!(mine[2..].iter().all(|m| m.contains("lane `")));
}

#[test]
fn a_requirement_that_is_not_on_a_higher_row_is_reported_and_kept() {
    let root = TestRoot::new();
    let core = root.core("happy");
    write_skill(
        root.path(),
        "base.toml",
        &skill("base", &["add-two"], 1, 0, ""),
    );
    write_skill(
        root.path(),
        "beside.toml",
        &skill(
            "beside",
            &["echo-server"],
            1,
            500,
            "requires = [\"base\"]\n",
        ),
    );
    write_skill(
        root.path(),
        "over.toml",
        &skill("over", &["hard-one"], 0, 0, "requires = [\"base\"]\n"),
    );

    for skill in core.list_skills().expect("skills") {
        if skill.id != "base" {
            assert_eq!(skill.requires, ["base"], "{}", skill.id);
        }
    }
    for file in ["beside.toml", "over.toml"] {
        let mine = issues_of(&core, file);
        assert_eq!(mine.len(), 1, "{mine:?}");
        assert!(mine[0].contains("`base`") && mine[0].contains("not on a higher row"));
    }
}

#[test]
fn the_shipped_skills_load_without_issues_and_build_downward() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let catalog = CatalogLoader::new(&root).load();
    let (skills, issues) = load_skills(&root, &catalog);
    assert!(issues.is_empty(), "{issues:?}");
    assert_eq!(skills.len(), 18);

    for skill in &skills {
        for required in &skill.requires {
            let above = skills
                .iter()
                .find(|s| &s.id == required)
                .expect("a skill requires only usable skills");
            assert!(
                above.row < skill.row,
                "{} (row {}) requires {} (row {})",
                skill.id,
                skill.row,
                above.id,
                above.row
            );
        }
    }
}

/// Earning, driven by real submits against the fake harness. The fake is a shell
/// script started as a program, which Windows cannot do.
#[cfg(unix)]
mod earning {
    use std::sync::{Arc, Mutex};

    use chrono::{DateTime, Duration, Utc};
    use learnai_core::{
        Clock, CoreError, CoreOptions, Framework, LOCAL_TARGET, ProgramHarness, RunEvent, RunMode,
        Skill, Verdict,
    };

    use super::common::{Collector, fixtures, test_limits};
    use super::*;

    const ADD_TWO: (&str, Framework) = ("add-two", Framework::Torch);
    const ECHO_SERVER: (&str, Framework) = ("echo-server", Framework::Python);
    const HARD_ONE: (&str, Framework) = ("hard-one", Framework::Torch);

    /// A clock the test advances.
    #[derive(Clone)]
    struct TestClock(Arc<Mutex<DateTime<Utc>>>);

    impl TestClock {
        fn new() -> Self {
            Self(Arc::new(Mutex::new(
                "2026-11-01T10:00:00Z".parse().expect("valid time"),
            )))
        }

        fn advance(&self, duration: Duration) {
            *self.0.lock().expect("clock lock") += duration;
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

    /// A core whose fake harness picks its scenario from the solution file.
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

    fn finish(collector: &Collector<RunEvent>) -> (Verdict, i64) {
        match collector.wait_for_end().last() {
            Some(RunEvent::End {
                verdict,
                submission_id,
                ..
            }) => (*verdict, submission_id.expect("a stored submission")),
            other => panic!("not an end event: {other:?}"),
        }
    }

    /// Submits a solution that ends in `scenario`, one hour after the previous submit.
    /// Returns the verdict, the submission id, and the submission's time.
    fn submit(
        core: &Core,
        clock: &TestClock,
        (problem, framework): (&str, Framework),
        scenario: &str,
    ) -> (Verdict, i64, String) {
        clock.advance(Duration::hours(1));
        core.write_solution(
            problem,
            framework,
            &format!("# scenario: {scenario}\ndef f():\n    return 0\n"),
        )
        .expect("write solution");
        let collector = Collector::new();
        core.start_run(
            problem,
            framework,
            RunMode::Submit,
            LOCAL_TARGET,
            collector.sink(),
        )
        .expect("start run");
        let (verdict, id) = finish(&collector);
        (verdict, id, clock.iso())
    }

    fn solve(core: &Core, clock: &TestClock, problem: (&str, Framework)) -> (i64, String) {
        let (verdict, id, at) = submit(core, clock, problem, "happy");
        assert_eq!(verdict, Verdict::Accepted);
        (id, at)
    }

    fn earned_ids(skills: &[Skill]) -> Vec<&str> {
        skills.iter().map(|s| s.id.as_str()).collect()
    }

    fn earned_at(core: &Core, id: &str) -> Option<String> {
        core.list_skills()
            .expect("skills")
            .into_iter()
            .find(|s| s.id == id)
            .expect("skill")
            .earned_at
    }

    #[test]
    fn the_submit_that_solves_a_skills_last_problem_earns_it() {
        let (root, clock) = (TestRoot::new(), TestClock::new());
        let core = core(&root, &clock);
        write_skill(
            root.path(),
            "pair.toml",
            &skill("pair", &["add-two", "hard-one"], 0, 0, ""),
        );

        let (first, _) = solve(&core, &clock, ADD_TWO);
        assert!(core.skills_earned_by(first).expect("earned").is_empty());
        assert_eq!(earned_at(&core, "pair"), None);

        let (second, at) = solve(&core, &clock, HARD_ONE);
        let earned = core.skills_earned_by(second).expect("earned");
        assert_eq!(earned_ids(&earned), ["pair"]);
        assert_eq!(earned[0].earned_at.as_deref(), Some(at.as_str()));
        assert_eq!(earned_at(&core, "pair"), Some(at));
        // The earlier submit still earned nothing.
        assert!(core.skills_earned_by(first).expect("earned").is_empty());
    }

    #[test]
    fn a_submit_that_earns_a_skill_also_earns_the_skills_waiting_on_it() {
        let (root, clock) = (TestRoot::new(), TestClock::new());
        let core = core(&root, &clock);
        write_chain(root.path());

        // The problems of `third` and `second` are solved first: both wait on `first`.
        let (hard, _) = solve(&core, &clock, HARD_ONE);
        let (echo, _) = solve(&core, &clock, ECHO_SERVER);
        for id in [hard, echo] {
            assert!(core.skills_earned_by(id).expect("earned").is_empty());
        }
        for id in ["first", "second", "third"] {
            assert_eq!(earned_at(&core, id), None);
        }

        // Solving the problem of `first` earns all three, at the time of that submit.
        let (add, at) = solve(&core, &clock, ADD_TWO);
        let earned = core.skills_earned_by(add).expect("earned");
        assert_eq!(earned_ids(&earned), ["first", "second", "third"]);
        for skill in &earned {
            assert_eq!(skill.earned_at.as_deref(), Some(at.as_str()));
        }
        for id in [hard, echo] {
            assert!(core.skills_earned_by(id).expect("earned").is_empty());
        }
    }

    #[test]
    fn skills_are_earned_one_by_one_down_a_chain() {
        let (root, clock) = (TestRoot::new(), TestClock::new());
        let core = core(&root, &clock);
        write_chain(root.path());

        let (add, add_at) = solve(&core, &clock, ADD_TWO);
        let (hard, _) = solve(&core, &clock, HARD_ONE);
        let (echo, echo_at) = solve(&core, &clock, ECHO_SERVER);

        let earned = |id| core.skills_earned_by(id).expect("earned");
        assert_eq!(earned_ids(&earned(add)), ["first"]);
        assert!(earned(hard).is_empty());
        // `third` was earned when `second` was, later than its own problem was solved.
        assert_eq!(earned_ids(&earned(echo)), ["second", "third"]);
        assert_eq!(earned_at(&core, "first"), Some(add_at));
        assert_eq!(earned_at(&core, "third"), Some(echo_at));
    }

    #[test]
    fn a_submit_that_is_not_accepted_earns_nothing() {
        let (root, clock) = (TestRoot::new(), TestClock::new());
        let core = core(&root, &clock);
        write_chain(root.path());

        let (verdict, failed, _) = submit(&core, &clock, ADD_TWO, "fail");
        assert_eq!(verdict, Verdict::WrongAnswer);
        assert!(core.skills_earned_by(failed).expect("earned").is_empty());
        // A submit that could not be judged in full is not accepted either.
        let (verdict, unjudged, _) = submit(&core, &clock, ADD_TWO, "assets-missing");
        assert_eq!(verdict, Verdict::AssetsMissing);
        assert!(core.skills_earned_by(unjudged).expect("earned").is_empty());
        assert_eq!(earned_at(&core, "first"), None);
    }

    #[test]
    fn a_second_accepted_submit_of_a_solved_problem_earns_nothing() {
        let (root, clock) = (TestRoot::new(), TestClock::new());
        let core = core(&root, &clock);
        write_chain(root.path());

        let (first, at) = solve(&core, &clock, ADD_TWO);
        let (again, _) = solve(&core, &clock, ADD_TWO);
        // In another framework too: the first solve in any framework is the one.
        let (jax, _) = solve(&core, &clock, ("add-two", Framework::Jax));

        assert_eq!(
            earned_ids(&core.skills_earned_by(first).expect("earned")),
            ["first"]
        );
        assert!(core.skills_earned_by(again).expect("earned").is_empty());
        assert!(core.skills_earned_by(jax).expect("earned").is_empty());
        assert_eq!(earned_at(&core, "first"), Some(at));
    }

    #[test]
    fn a_review_submit_earns_nothing_and_moves_no_earned_time() {
        let (root, clock) = (TestRoot::new(), TestClock::new());
        let core = core(&root, &clock);
        write_chain(root.path());
        let (_, at) = solve(&core, &clock, ADD_TWO);

        clock.advance(Duration::days(4));
        let (problem, framework) = ADD_TWO;
        core.start_review(problem, framework).expect("start review");
        core.write_review_solution(problem, framework, "# scenario: happy\n")
            .expect("write review");
        let collector = Collector::new();
        core.start_review_run(
            problem,
            framework,
            RunMode::Submit,
            LOCAL_TARGET,
            collector.sink(),
        )
        .expect("start review run");
        let (verdict, review) = finish(&collector);
        assert_eq!(verdict, Verdict::Accepted);

        assert!(core.skills_earned_by(review).expect("earned").is_empty());
        assert_eq!(earned_at(&core, "first"), Some(at));
    }

    #[test]
    fn an_unknown_submission_is_not_found() {
        let root = TestRoot::new();
        let core = root.core("happy");
        assert!(matches!(
            core.skills_earned_by(4242),
            Err(CoreError::SubmissionNotFound(4242))
        ));
    }
}
