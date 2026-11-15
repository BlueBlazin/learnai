//! Spaced re-solve (`docs/ARCHITECTURE.md` section 4.2): the schedule math and the
//! clock it reads.
//!
//! A solved problem is reviewed at growing intervals. The schedule of a problem is one
//! [`ReviewRow`]: a stage (the index of the interval that led to the current due date)
//! and that due date. Everything here is pure; the current time always comes in as an
//! argument, so tests move it instead of sleeping.

use chrono::{DateTime, Duration, SecondsFormat, Utc};

use crate::error::Result;
use crate::model::{Review, ReviewAction};
use crate::store::{ReviewRow, Store};

/// Days between solving a problem and its reviews: the first review comes 3 days after
/// the first accepted submit, the second 7 days after the first review, and so on.
pub const INTERVALS_DAYS: [u32; 5] = [3, 7, 21, 60, 180];

/// The setting that turns reviews on and off (default on).
pub const ENABLED_SETTING: &str = "reviews_enabled";

/// Where core reads the current time. Replaceable so tests can move it.
pub trait Clock: Send + Sync {
    /// The current time.
    fn now(&self) -> DateTime<Utc>;
}

/// The system clock.
#[derive(Debug, Clone, Copy, Default)]
pub struct SystemClock;

impl Clock for SystemClock {
    fn now(&self) -> DateTime<Utc> {
        Utc::now()
    }
}

/// Formats a time the way every stored timestamp is written: ISO 8601 in UTC with
/// millisecond precision.
pub fn format_time(time: DateTime<Utc>) -> String {
    time.to_rfc3339_opts(SecondsFormat::Millis, true)
}

fn parse_time(text: &str) -> Option<DateTime<Utc>> {
    DateTime::parse_from_rfc3339(text)
        .ok()
        .map(|time| time.with_timezone(&Utc))
}

fn days(count: u32) -> Duration {
    Duration::days(i64::from(count))
}

/// The interval of a stage, clamped to the last one for a stage that is out of range.
pub fn interval_days(stage: u32) -> u32 {
    let last = INTERVALS_DAYS.len() - 1;
    INTERVALS_DAYS[(stage as usize).min(last)]
}

/// When the first review of a problem solved at `solved_at` is due.
pub fn first_due(solved_at: DateTime<Utc>) -> String {
    format_time(solved_at + days(INTERVALS_DAYS[0]))
}

/// The schedule after an accepted review submit at `now`: the next stage, due one
/// interval later, or retired after the last interval (the problem is mastered).
pub fn advanced(review: &ReviewRow, now: DateTime<Utc>) -> ReviewRow {
    let stage = review.stage.saturating_add(1);
    let mastered = stage as usize >= INTERVALS_DAYS.len();
    ReviewRow {
        problem_id: review.problem_id.clone(),
        stage: if mastered { review.stage } else { stage },
        due_at: if mastered {
            review.due_at.clone()
        } else {
            format_time(now + days(interval_days(stage)))
        },
        last_reviewed_at: Some(format_time(now)),
        retired: mastered,
    }
}

/// The schedule after the learner resolves a review without solving it.
///
/// - `forgot`: back to stage 0, due the next day.
/// - `snooze`: one day later than the due date, or than now when it is already overdue,
///   so a snoozed review always leaves the due list; the stage is unchanged.
/// - `retire`: reviews for the problem end.
pub fn resolved(review: &ReviewRow, action: ReviewAction, now: DateTime<Utc>) -> ReviewRow {
    let mut next = review.clone();
    match action {
        ReviewAction::Forgot => {
            next.stage = 0;
            next.due_at = format_time(now + days(1));
        }
        ReviewAction::Snooze => {
            let due = parse_time(&review.due_at).map_or(now, |due| due.max(now));
            next.due_at = format_time(due + days(1));
        }
        ReviewAction::Retire => next.retired = true,
    }
    next
}

/// The wire form of a schedule as of `now`.
pub fn to_wire(review: ReviewRow, now: DateTime<Utc>) -> Review {
    // A due date that cannot be read is treated as due, so the entry stays reachable.
    let due = parse_time(&review.due_at).is_none_or(|due| due <= now);
    Review {
        interval_days: interval_days(review.stage),
        problem_id: review.problem_id,
        stage: review.stage,
        due_at: review.due_at,
        last_reviewed_at: review.last_reviewed_at,
        due,
    }
}

/// Whether reviews are turned on (the default).
pub fn enabled(store: &Store) -> Result<bool> {
    Ok(store.setting::<bool>(ENABLED_SETTING)?.unwrap_or(true))
}

/// What an accepted submit did to the review schedule of its problem.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Recorded {
    /// Nothing changed.
    Unchanged,
    /// The first accepted submit created the schedule.
    Scheduled,
    /// An accepted review submit moved the schedule on (or retired it).
    Advanced,
}

/// Applies an accepted submit at `now` to the schedule of `problem_id`.
///
/// A review submit advances an active schedule. Any other accepted submit creates the
/// schedule when the problem has none and reviews are on; it never advances one.
pub fn record_accepted(
    store: &Store,
    problem_id: &str,
    is_review: bool,
    now: DateTime<Utc>,
) -> Result<Recorded> {
    if is_review {
        return match store.review(problem_id)? {
            Some(review) if !review.retired => {
                store.save_review(&advanced(&review, now))?;
                Ok(Recorded::Advanced)
            }
            _ => Ok(Recorded::Unchanged),
        };
    }
    if enabled(store)? && store.create_review(problem_id, &first_due(now))? {
        return Ok(Recorded::Scheduled);
    }
    Ok(Recorded::Unchanged)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn at(text: &str) -> DateTime<Utc> {
        parse_time(text).expect("valid time")
    }

    fn row(stage: u32, due_at: &str) -> ReviewRow {
        ReviewRow {
            problem_id: "p".into(),
            stage,
            due_at: due_at.into(),
            last_reviewed_at: None,
            retired: false,
        }
    }

    #[test]
    fn the_first_review_is_due_three_days_after_solving() {
        assert_eq!(
            first_due(at("2026-11-01T10:00:00.000Z")),
            "2026-11-04T10:00:00.000Z"
        );
    }

    #[test]
    fn accepted_reviews_walk_the_intervals_and_then_retire() {
        let mut review = row(0, "2026-11-04T10:00:00.000Z");
        let mut now = at("2026-11-05T08:00:00.000Z");
        let mut gaps = Vec::new();
        for _ in 0..4 {
            let next = advanced(&review, now);
            assert!(!next.retired);
            assert_eq!(next.stage, review.stage + 1);
            assert_eq!(next.last_reviewed_at, Some(format_time(now)));
            let due = at(&next.due_at);
            // The next due date counts from the submit, not from the old due date.
            gaps.push((due - now).num_days());
            review = next;
            now = due + Duration::hours(5);
        }
        assert_eq!(gaps, [7, 21, 60, 180]);

        let mastered = advanced(&review, now);
        assert!(mastered.retired);
        assert_eq!(mastered.stage, 4);
        assert_eq!(mastered.last_reviewed_at, Some(format_time(now)));
    }

    #[test]
    fn forgot_restarts_and_retire_ends() {
        let now = at("2026-12-01T09:30:00.000Z");
        let review = row(3, "2026-11-20T10:00:00.000Z");

        let forgot = resolved(&review, ReviewAction::Forgot, now);
        assert_eq!(
            (forgot.stage, forgot.due_at.as_str(), forgot.retired),
            (0, "2026-12-02T09:30:00.000Z", false)
        );

        let retired = resolved(&review, ReviewAction::Retire, now);
        assert!(retired.retired);
        assert_eq!(retired.stage, 3);
    }

    #[test]
    fn snooze_moves_one_day_past_now_or_the_due_date_whichever_is_later() {
        let now = at("2026-12-01T09:30:00.000Z");
        // Overdue: one day from now, so it leaves the due list.
        let overdue = resolved(
            &row(2, "2026-11-20T10:00:00.000Z"),
            ReviewAction::Snooze,
            now,
        );
        assert_eq!(
            (overdue.stage, overdue.due_at.as_str()),
            (2, "2026-12-02T09:30:00.000Z")
        );
        // Not due yet: one day after the due date.
        let early = resolved(
            &row(2, "2026-12-05T10:00:00.000Z"),
            ReviewAction::Snooze,
            now,
        );
        assert_eq!(early.due_at, "2026-12-06T10:00:00.000Z");
    }

    #[test]
    fn a_review_is_due_from_its_due_time_on() {
        let review = row(1, "2026-11-10T10:00:00.000Z");
        let before = to_wire(review.clone(), at("2026-11-10T09:59:59.000Z"));
        assert!(!before.due);
        assert_eq!(before.interval_days, 7);
        assert!(to_wire(review.clone(), at("2026-11-10T10:00:00.000Z")).due);
        assert!(to_wire(review, at("2027-01-01T00:00:00.000Z")).due);
        assert!(to_wire(row(0, "garbage"), at("2026-11-10T10:00:00.000Z")).due);
    }

    #[test]
    fn accepted_submits_schedule_once_and_only_review_submits_advance() -> Result<()> {
        let store = Store::open_in_memory()?;
        let solved = at("2026-11-01T10:00:00.000Z");
        assert_eq!(
            record_accepted(&store, "p", false, solved)?,
            Recorded::Scheduled
        );
        let first = store.review("p")?.expect("scheduled");
        assert_eq!(
            (first.stage, first.due_at.as_str()),
            (0, "2026-11-04T10:00:00.000Z")
        );

        // Another normal accepted submit, even once the review is due, changes nothing.
        let later = at("2026-11-06T10:00:00.000Z");
        assert_eq!(
            record_accepted(&store, "p", false, later)?,
            Recorded::Unchanged
        );
        assert_eq!(store.review("p")?, Some(first));

        assert_eq!(
            record_accepted(&store, "p", true, later)?,
            Recorded::Advanced
        );
        let second = store.review("p")?.expect("advanced");
        assert_eq!(
            (second.stage, second.due_at.as_str()),
            (1, "2026-11-13T10:00:00.000Z")
        );

        // A retired schedule is final: neither kind of submit revives it.
        store.save_review(&resolved(&second, ReviewAction::Retire, later))?;
        assert_eq!(
            record_accepted(&store, "p", true, later)?,
            Recorded::Unchanged
        );
        assert_eq!(
            record_accepted(&store, "p", false, later)?,
            Recorded::Unchanged
        );
        assert!(store.active_reviews()?.is_empty());

        // A review submit never creates a schedule.
        assert_eq!(
            record_accepted(&store, "q", true, later)?,
            Recorded::Unchanged
        );
        assert_eq!(store.review("q")?, None);
        Ok(())
    }

    #[test]
    fn nothing_is_scheduled_while_reviews_are_off() -> Result<()> {
        let store = Store::open_in_memory()?;
        store.set_setting(ENABLED_SETTING, &false)?;
        let now = at("2026-11-01T10:00:00.000Z");
        assert_eq!(
            record_accepted(&store, "p", false, now)?,
            Recorded::Unchanged
        );
        assert_eq!(store.review("p")?, None);
        Ok(())
    }
}
