use anyhow::{Context, Result};
use jiff::civil::date;
use jiff::{Timestamp, Zoned};

use crate::config::Limits;

/// Identifies the current daily/monthly window start instants for a provider,
/// anchored at its configured reset time + timezone.
#[derive(Debug, Clone)]
pub struct Windows {
    pub day_start: Timestamp,
    pub month_start: Timestamp,
}

pub fn current_windows(limits: &Limits, now: Timestamp) -> Result<Windows> {
    let (hh, mm) = parse_reset(&limits.reset)?;
    let tz = &limits.reset_tz;
    let znow: Zoned = now.in_tz(tz).with_context(|| format!("bad timezone {tz}"))?;
    let today = znow.date();

    let reset_today = today
        .at(hh, mm, 0, 0)
        .in_tz(tz)
        .context("computing daily reset")?;
    let day_start = if znow >= reset_today {
        reset_today
    } else {
        today
            .yesterday()
            .context("date underflow")?
            .at(hh, mm, 0, 0)
            .in_tz(tz)?
    };

    let first_this_month = date(today.year(), today.month(), 1)
        .at(hh, mm, 0, 0)
        .in_tz(tz)?;
    let month_start = if znow >= first_this_month {
        first_this_month
    } else {
        // still before this month's first reset -> previous month
        let prev = today.first_of_month().yesterday().context("date underflow")?;
        date(prev.year(), prev.month(), 1).at(hh, mm, 0, 0).in_tz(tz)?
    };

    Ok(Windows {
        day_start: day_start.timestamp(),
        month_start: month_start.timestamp(),
    })
}

fn parse_reset(s: &str) -> Result<(i8, i8)> {
    // Accept "HH:MM" and legacy "HH:MMZ"
    let s = s.trim_end_matches('Z');
    let (h, m) = s
        .split_once(':')
        .with_context(|| format!("reset '{s}' must be HH:MM"))?;
    Ok((h.parse()?, m.parse()?))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::Limits;

    fn limits(reset: &str, tz: &str) -> Limits {
        Limits {
            rpm: None,
            daily_requests: None,
            daily_tokens: None,
            monthly_requests: None,
            monthly_tokens: None,
            reset: reset.into(),
            reset_tz: tz.into(),
        }
    }

    #[test]
    fn day_window_before_and_after_reset() {
        let l = limits("06:00", "UTC");
        // 05:00 UTC -> window started yesterday 06:00
        let now: Timestamp = "2026-08-24T05:00:00Z".parse().unwrap();
        let w = current_windows(&l, now).unwrap();
        assert_eq!(w.day_start.to_string(), "2026-08-23T06:00:00Z");
        // 07:00 UTC -> window started today 06:00
        let now: Timestamp = "2026-08-24T07:00:00Z".parse().unwrap();
        let w = current_windows(&l, now).unwrap();
        assert_eq!(w.day_start.to_string(), "2026-08-24T06:00:00Z");
    }

    #[test]
    fn month_window_rollover() {
        let l = limits("00:00", "UTC");
        let now: Timestamp = "2026-08-01T00:00:01Z".parse().unwrap();
        let w = current_windows(&l, now).unwrap();
        assert_eq!(w.month_start.to_string(), "2026-08-01T00:00:00Z");
        // Just before the first reset of the month -> previous month window
        let l = limits("06:00", "UTC");
        let now: Timestamp = "2026-08-01T05:59:00Z".parse().unwrap();
        let w = current_windows(&l, now).unwrap();
        assert_eq!(w.month_start.to_string(), "2026-07-01T06:00:00Z");
    }

    #[test]
    fn timezone_aware() {
        let l = limits("00:00", "Asia/Dhaka"); // UTC+6
        // 2026-08-24T20:00Z = 2026-08-25T02:00 Dhaka -> day started 2026-08-25T00:00 Dhaka
        let now: Timestamp = "2026-08-24T20:00:00Z".parse().unwrap();
        let w = current_windows(&l, now).unwrap();
        assert_eq!(w.day_start.to_string(), "2026-08-24T18:00:00Z");
    }
}
