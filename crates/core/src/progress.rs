//! Progress derivation: problem status and the day streak.

use std::collections::BTreeSet;

use chrono::{DateTime, Local, NaiveDate, TimeZone};

use crate::model::ProblemStatus;

/// Derives a problem's status.
///
/// `solved` once any framework has an accepted submit; `attempted` once a solution file
/// differs from the starter or any submit exists; otherwise `todo`.
pub fn derive_status(
    has_accepted: bool,
    has_submission: bool,
    solution_modified: bool,
) -> ProblemStatus {
    if has_accepted {
        ProblemStatus::Solved
    } else if has_submission || solution_modified {
        ProblemStatus::Attempted
    } else {
        ProblemStatus::Todo
    }
}

/// Converts ISO 8601 timestamps to calendar days in `zone`, ignoring unparsable values.
pub fn days_in_zone<'a, Tz: TimeZone>(
    timestamps: impl IntoIterator<Item = &'a str>,
    zone: &Tz,
) -> BTreeSet<NaiveDate> {
    timestamps
        .into_iter()
        .filter_map(|t| DateTime::parse_from_rfc3339(t).ok())
        .map(|t| t.with_timezone(zone).date_naive())
        .collect()
}

/// Length of the run of consecutive active days ending today, or ending yesterday when
/// today has no activity yet (so a streak is not shown as broken before the day is over).
pub fn streak_days(active_days: &BTreeSet<NaiveDate>, today: NaiveDate) -> u32 {
    let mut day = if active_days.contains(&today) {
        today
    } else {
        match today.pred_opt() {
            Some(yesterday) => yesterday,
            None => return 0,
        }
    };
    let mut streak = 0;
    while active_days.contains(&day) {
        streak += 1;
        match day.pred_opt() {
            Some(previous) => day = previous,
            None => break,
        }
    }
    streak
}

/// The streak as of now in the local time zone, from accepted-submission timestamps.
pub fn current_streak<'a>(accepted_times: impl IntoIterator<Item = &'a str>) -> u32 {
    streak_days(
        &days_in_zone(accepted_times, &Local),
        Local::now().date_naive(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::FixedOffset;

    fn day(text: &str) -> NaiveDate {
        text.parse().expect("valid date")
    }

    fn days(texts: &[&str]) -> BTreeSet<NaiveDate> {
        texts.iter().map(|t| day(t)).collect()
    }

    #[test]
    fn status_precedence() {
        assert_eq!(derive_status(false, false, false), ProblemStatus::Todo);
        assert_eq!(derive_status(false, false, true), ProblemStatus::Attempted);
        assert_eq!(derive_status(false, true, false), ProblemStatus::Attempted);
        assert_eq!(derive_status(true, true, false), ProblemStatus::Solved);
    }

    #[test]
    fn streak_counts_consecutive_days_ending_today() {
        let active = days(&["2026-11-10", "2026-11-12", "2026-11-13", "2026-11-14"]);
        assert_eq!(streak_days(&active, day("2026-11-14")), 3);
    }

    #[test]
    fn streak_survives_until_the_end_of_the_next_day() {
        let active = days(&["2026-11-12", "2026-11-13"]);
        assert_eq!(streak_days(&active, day("2026-11-14")), 2);
        assert_eq!(streak_days(&active, day("2026-11-15")), 0);
    }

    #[test]
    fn streak_is_zero_without_activity() {
        assert_eq!(streak_days(&BTreeSet::new(), day("2026-11-14")), 0);
        assert_eq!(streak_days(&days(&["2026-11-01"]), day("2026-11-14")), 0);
    }

    #[test]
    fn streak_crosses_month_and_year_boundaries() {
        let active = days(&["2026-12-30", "2026-12-31", "2027-01-01"]);
        assert_eq!(streak_days(&active, day("2027-01-01")), 3);
    }

    #[test]
    fn days_are_taken_in_the_given_zone() {
        // 02:30 UTC on the 14th is still the 13th in UTC-8.
        let pacific = FixedOffset::west_opt(8 * 3600).expect("valid offset");
        let got = days_in_zone(
            [
                "2026-11-14T02:30:00.000Z",
                "not a time",
                "2026-11-14T20:00:00Z",
            ],
            &pacific,
        );
        assert_eq!(got, days(&["2026-11-13", "2026-11-14"]));
    }
}
