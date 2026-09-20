//! `pxy stats`: what the proxy actually did, read back from the per-leg rows
//! `state` stores (wiki:state).
//!
//! The rows are pulled for the window and aggregated here rather than in SQL:
//! percentiles want the whole sample anyway, and at one row per upstream leg a
//! personal install's month is a few tens of thousands of rows.

use anyhow::{Context, Result};
use serde_json::{Value, json};

use crate::state::{AttemptRow, State, ToolRunRow};

/// What to report on.
pub struct Filter {
    /// Window start, epoch ms. 0 means everything kept.
    pub since_ms: i64,
    /// Report only this dimension: model, provider, agent, group, day or tool.
    pub by: Option<String>,
    pub provider: Option<String>,
    pub agent: Option<String>,
    pub model: Option<String>,
    /// List the failures themselves instead of the tables.
    pub errors: bool,
}

impl Filter {
    fn keeps(&self, r: &AttemptRow) -> bool {
        if let Some(p) = &self.provider
            && &r.provider != p
        {
            return false;
        }
        if let Some(a) = &self.agent
            && &r.agent != a
        {
            return false;
        }
        if let Some(m) = &self.model
            && &r.model != m
            && format!("{}/{}", r.provider, r.model) != *m
        {
            return false;
        }
        true
    }
}

/// Parse a window: `24h`, `90m`, `7d`, `2w`, `today`, `month`, `all`.
/// Returns the window start in epoch ms (0 for everything).
pub fn parse_since(spec: &str, now: jiff::Zoned) -> Result<i64> {
    let spec = spec.trim().to_lowercase();
    let now_ms = now.timestamp().as_millisecond();
    match spec.as_str() {
        "all" => return Ok(0),
        "today" => {
            let start = now.start_of_day().context("start of today")?;
            return Ok(start.timestamp().as_millisecond());
        }
        "month" => {
            let start = now
                .date()
                .first_of_month()
                .to_zoned(now.time_zone().clone())
                .context("start of the month")?;
            return Ok(start.timestamp().as_millisecond());
        }
        _ => {}
    }
    let (digits, unit) = spec.split_at(spec.len().saturating_sub(1));
    let n: i64 = digits
        .parse()
        .with_context(|| format!("'{spec}' is not a window: try 24h, 7d, today, month or all"))?;
    let ms = match unit {
        "m" => n * 60_000,
        "h" => n * 3_600_000,
        "d" => n * 86_400_000,
        "w" => n * 7 * 86_400_000,
        _ => anyhow::bail!("'{spec}' is not a window: try 24h, 7d, today, month or all"),
    };
    Ok((now_ms - ms).max(0))
}

/// One group's numbers.
#[derive(Default)]
struct Agg {
    legs: u64,
    errors: u64,
    input: u64,
    output: u64,
    cache_read: u64,
    /// Latencies of the legs that answered, for the percentiles.
    latencies: Vec<i64>,
    /// Generation time (leg minus its time to first byte) of answering legs.
    gen_ms: i64,
    /// Output tokens over that same generation time.
    gen_output: u64,
}

impl Agg {
    fn add(&mut self, r: &AttemptRow) {
        self.legs += 1;
        self.input += r.input;
        self.output += r.output;
        self.cache_read += r.cache_read;
        if is_error(r) {
            self.errors += 1;
            return;
        }
        self.latencies.push(r.ms);
        self.gen_ms += (r.ms - r.ttfb_ms).max(0);
        self.gen_output += r.output;
    }

    fn percentile(&self, q: f64) -> Option<i64> {
        if self.latencies.is_empty() {
            return None;
        }
        let mut sorted = self.latencies.clone();
        sorted.sort_unstable();
        // Nearest-rank: the smallest value at or above q of the sample.
        let rank = ((q * sorted.len() as f64).ceil() as usize).clamp(1, sorted.len());
        Some(sorted[rank - 1])
    }

    fn tokens_per_second(&self) -> Option<f64> {
        (self.gen_ms > 0 && self.gen_output > 0)
            .then(|| self.gen_output as f64 / (self.gen_ms as f64 / 1000.0))
    }

    fn error_rate(&self) -> f64 {
        if self.legs == 0 { 0.0 } else { self.errors as f64 / self.legs as f64 }
    }

    fn cache_share(&self) -> f64 {
        if self.input == 0 { 0.0 } else { self.cache_read as f64 / self.input as f64 }
    }
}

/// A leg that did not answer. "truncated" is not one: the client was handed a
/// short answer, which is a degraded success, and it is reported on its own.
fn is_error(r: &AttemptRow) -> bool {
    !matches!(r.outcome.as_str(), "ok" | "truncated" | "refused")
}

/// Group the window's legs by one dimension, biggest first.
fn group_by(rows: &[AttemptRow], key: impl Fn(&AttemptRow) -> String) -> Vec<(String, Agg)> {
    let mut map: std::collections::HashMap<String, Agg> = std::collections::HashMap::new();
    for r in rows {
        map.entry(key(r)).or_default().add(r);
    }
    let mut out: Vec<(String, Agg)> = map.into_iter().collect();
    out.sort_by(|a, b| b.1.legs.cmp(&a.1.legs).then(a.0.cmp(&b.0)));
    out
}

pub fn human_tokens(n: u64) -> String {
    if n >= 1_000_000 {
        format!("{:.1}M", n as f64 / 1e6)
    } else if n >= 1_000 {
        format!("{:.0}k", n as f64 / 1e3)
    } else {
        n.to_string()
    }
}

fn human_ms(ms: Option<i64>) -> String {
    match ms {
        None => "-".to_string(),
        Some(ms) if ms >= 10_000 => format!("{}s", ms / 1000),
        Some(ms) if ms >= 1_000 => format!("{:.1}s", ms as f64 / 1000.0),
        Some(ms) => format!("{ms}ms"),
    }
}

/// A rate inside a table column: zero is noise there, so it reads as blank.
fn percent(x: f64) -> String {
    if x == 0.0 { "-".to_string() } else { format!("{:.1}%", x * 100.0) }
}

/// The same rate in a sentence, where "-" would read as "unknown".
fn percent_flat(x: f64) -> String {
    format!("{:.1}%", x * 100.0)
}

fn ago(ts: i64, now_ms: i64) -> String {
    let secs = (now_ms - ts).max(0) / 1000;
    if secs >= 86_400 {
        format!("{}d ago", secs / 86_400)
    } else if secs >= 3_600 {
        format!("{}h ago", secs / 3_600)
    } else if secs >= 60 {
        format!("{}m ago", secs / 60)
    } else {
        format!("{secs}s ago")
    }
}

/// The text report.
pub fn report(state: &State, filter: &Filter, window: &str) -> Result<String> {
    let now_ms = jiff::Timestamp::now().as_millisecond();
    let rows: Vec<AttemptRow> = state
        .attempts_since(filter.since_ms)?
        .into_iter()
        .filter(|r| filter.keeps(r))
        .collect();
    let tools: Vec<ToolRunRow> = state
        .tool_runs_since(filter.since_ms)?
        .into_iter()
        .filter(|r| filter.agent.as_ref().is_none_or(|a| &r.agent == a))
        .collect();

    let mut total = Agg::default();
    for r in &rows {
        total.add(r);
    }
    let mut out = String::new();
    out.push_str(&format!(
        "pxy stats — {window} ({} legs, {} in / {} out, {} cached, {} errors)\n",
        total.legs,
        human_tokens(total.input),
        human_tokens(total.output),
        percent_flat(total.cache_share()),
        percent_flat(total.error_rate()),
    ));
    if rows.is_empty() && tools.is_empty() {
        out.push_str("\nnothing recorded in this window\n");
        return Ok(out);
    }
    out.push_str(&format!(
        "latency p50 {} · p95 {}\n",
        human_ms(total.percentile(0.5)),
        human_ms(total.percentile(0.95)),
    ));

    if filter.errors {
        out.push_str(&errors_section(&rows, now_ms));
        return Ok(out);
    }

    let dimension = filter.by.as_deref();
    let want = |name: &str| dimension.is_none_or(|d| d == name);
    if want("model") {
        out.push_str(&table(
            "model",
            group_by(&rows, |r| format!("{}/{}", r.provider, r.model)),
        ));
    }
    if want("provider") {
        out.push_str(&table("provider", group_by(&rows, |r| r.provider.clone())));
    }
    if want("agent") {
        out.push_str(&table("agent", group_by(&rows, |r| r.agent.clone())));
    }
    if want("group") {
        out.push_str(&table("requested", group_by(&rows, |r| r.requested.clone())));
    }
    if dimension == Some("day") {
        out.push_str(&table("day", group_by(&rows, |r| r.day.clone())));
    }
    if want("tool") {
        out.push_str(&tools_section(&tools));
    }
    if dimension.is_none() {
        out.push_str(&errors_section(&rows, now_ms));
    }
    Ok(out)
}

fn table(label: &str, groups: Vec<(String, Agg)>) -> String {
    if groups.is_empty() {
        return String::new();
    }
    let mut out = format!(
        "\n{:<28} {:>6} {:>8} {:>8} {:>7} {:>7} {:>7} {:>7} {:>7}\n",
        label, "legs", "in", "out", "cache", "err", "p50", "p95", "tok/s"
    );
    for (name, a) in groups {
        out.push_str(&format!(
            "{:<28} {:>6} {:>8} {:>8} {:>7} {:>7} {:>7} {:>7} {:>7}\n",
            truncate(&name, 28),
            a.legs,
            human_tokens(a.input),
            human_tokens(a.output),
            percent(a.cache_share()),
            percent(a.error_rate()),
            human_ms(a.percentile(0.5)),
            human_ms(a.percentile(0.95)),
            a.tokens_per_second().map(|t| format!("{t:.0}")).unwrap_or_else(|| "-".into()),
        ));
    }
    out
}

fn tools_section(runs: &[ToolRunRow]) -> String {
    if runs.is_empty() {
        return String::new();
    }
    let mut map: std::collections::HashMap<&str, (u64, u64, Vec<i64>)> =
        std::collections::HashMap::new();
    for r in runs {
        let e = map.entry(r.tool.as_str()).or_insert((0, 0, Vec::new()));
        e.0 += 1;
        if !r.ok {
            e.1 += 1;
        }
        e.2.push(r.ms);
    }
    let mut rows: Vec<(&str, (u64, u64, Vec<i64>))> = map.into_iter().collect();
    rows.sort_by(|a, b| b.1.0.cmp(&a.1.0).then(a.0.cmp(b.0)));
    let mut out = format!("\n{:<28} {:>6} {:>7} {:>7} {:>7}\n", "server tool", "calls", "err", "p50", "p95");
    for (tool, (calls, errs, mut lat)) in rows {
        lat.sort_unstable();
        let pick = |q: f64| -> Option<i64> {
            let rank = ((q * lat.len() as f64).ceil() as usize).clamp(1, lat.len());
            lat.get(rank - 1).copied()
        };
        out.push_str(&format!(
            "{:<28} {:>6} {:>7} {:>7} {:>7}\n",
            tool,
            calls,
            percent(if calls == 0 { 0.0 } else { errs as f64 / calls as f64 }),
            human_ms(pick(0.5)),
            human_ms(pick(0.95)),
        ));
    }
    out
}

/// What failed, newest first: the reason, how often, and when it last bit.
fn errors_section(rows: &[AttemptRow], now_ms: i64) -> String {
    let mut map: std::collections::HashMap<String, (u64, i64, String)> =
        std::collections::HashMap::new();
    for r in rows.iter().filter(|r| is_error(r)) {
        let text = first_line(&r.error);
        // Upstream reasons often open with the status themselves; saying it
        // twice reads as a typo.
        let reason = match r.status {
            0 => format!("{}: {text}", r.outcome),
            s if text.starts_with(&s.to_string()) => text,
            s => format!("{s} {text}"),
        };
        let e = map.entry(reason).or_insert((0, 0, String::new()));
        e.0 += 1;
        if r.ts > e.1 {
            e.1 = r.ts;
            e.2 = format!("{}/{}", r.provider, r.model);
        }
    }
    if map.is_empty() {
        return String::new();
    }
    let mut list: Vec<(String, (u64, i64, String))> = map.into_iter().collect();
    list.sort_by(|a, b| b.1.0.cmp(&a.1.0).then(b.1.1.cmp(&a.1.1)));
    let mut out = format!("\n{:<44} {:>5}  {:<22} {}\n", "error", "n", "last candidate", "last seen");
    for (reason, (n, ts, candidate)) in list.into_iter().take(15) {
        out.push_str(&format!(
            "{:<44} {:>5}  {:<22} {}\n",
            truncate(&reason, 44),
            n,
            truncate(&candidate, 22),
            ago(ts, now_ms)
        ));
    }
    out
}

fn first_line(s: &str) -> String {
    s.lines().next().unwrap_or("").trim().to_string()
}

fn truncate(s: &str, n: usize) -> String {
    if s.chars().count() <= n {
        return s.to_string();
    }
    s.chars().take(n.saturating_sub(1)).collect::<String>() + "…"
}

/// The same numbers as one JSON object, for `pxy stats --json` and for the
/// `stats` key of `pxy status --json`.
pub fn summary_json(state: &State, since_ms: i64) -> Value {
    let rows = state.attempts_since(since_ms).unwrap_or_default();
    let tools = state.tool_runs_since(since_ms).unwrap_or_default();
    let mut total = Agg::default();
    for r in &rows {
        total.add(r);
    }
    let dimension = |groups: Vec<(String, Agg)>| -> Value {
        Value::Array(
            groups
                .into_iter()
                .map(|(name, a)| {
                    json!({
                        "name": name,
                        "legs": a.legs,
                        "errors": a.errors,
                        "inputTokens": a.input,
                        "outputTokens": a.output,
                        "cacheReadTokens": a.cache_read,
                        "p50Ms": a.percentile(0.5),
                        "p95Ms": a.percentile(0.95),
                        "tokensPerSecond": a.tokens_per_second(),
                    })
                })
                .collect(),
        )
    };
    let mut tool_map: std::collections::BTreeMap<String, (u64, u64, i64)> = Default::default();
    for r in &tools {
        let e = tool_map.entry(r.tool.clone()).or_insert((0, 0, 0));
        e.0 += 1;
        if !r.ok {
            e.1 += 1;
        }
        e.2 += r.ms;
    }
    json!({
        "since": since_ms,
        "legs": total.legs,
        "errors": total.errors,
        "inputTokens": total.input,
        "outputTokens": total.output,
        "cacheReadTokens": total.cache_read,
        "p50Ms": total.percentile(0.5),
        "p95Ms": total.percentile(0.95),
        "models": dimension(group_by(&rows, |r| format!("{}/{}", r.provider, r.model))),
        "providers": dimension(group_by(&rows, |r| r.provider.clone())),
        "agents": dimension(group_by(&rows, |r| r.agent.clone())),
        "requested": dimension(group_by(&rows, |r| r.requested.clone())),
        "tools": tool_map.into_iter().map(|(tool, (calls, errors, ms))| json!({
            "name": tool, "calls": calls, "errors": errors, "totalMs": ms,
        })).collect::<Vec<_>>(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn row(model: &str, outcome: &str, ms: i64, output: u64) -> AttemptRow {
        AttemptRow {
            ts: 1_700_000_000_000,
            day: "2026-09-20".into(),
            kind: "chat".into(),
            agent: "claude".into(),
            requested: "aaa".into(),
            provider: "p".into(),
            model: model.into(),
            outcome: outcome.into(),
            ms,
            ttfb_ms: 100,
            input: 1000,
            cache_read: 250,
            output,
            ..AttemptRow::default()
        }
    }

    #[test]
    fn windows_parse_to_their_start() {
        let now: jiff::Zoned = "2026-09-20T15:30:00+06:00[Asia/Dhaka]".parse().unwrap();
        let now_ms = now.timestamp().as_millisecond();
        assert_eq!(parse_since("24h", now.clone()).unwrap(), now_ms - 86_400_000);
        assert_eq!(parse_since("90m", now.clone()).unwrap(), now_ms - 5_400_000);
        assert_eq!(parse_since("7d", now.clone()).unwrap(), now_ms - 7 * 86_400_000);
        assert_eq!(parse_since("all", now.clone()).unwrap(), 0);
        // today and month are LOCAL boundaries, not UTC ones.
        assert_eq!(parse_since("today", now.clone()).unwrap(), now_ms - 15 * 3_600_000 - 1_800_000);
        let month = parse_since("month", now.clone()).unwrap();
        assert_eq!(
            jiff::Timestamp::from_millisecond(month).unwrap().in_tz("Asia/Dhaka").unwrap().to_string(),
            "2026-09-01T00:00:00+06:00[Asia/Dhaka]"
        );
        assert!(parse_since("yesterday", now).is_err());
    }

    #[test]
    fn percentiles_and_rates_read_the_sample() {
        let mut a = Agg::default();
        for ms in [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000] {
            a.add(&row("m", "ok", ms, 10));
        }
        // Nearest-rank over ten samples: p50 is the 5th, p95 the 10th.
        assert_eq!(a.percentile(0.5), Some(500));
        assert_eq!(a.percentile(0.95), Some(1000));
        assert_eq!(a.legs, 10);
        assert_eq!(a.error_rate(), 0.0);
        assert_eq!(a.cache_share(), 0.25);
        // 100 output tokens over 4.5s of generation: each leg's time minus its
        // 100ms to first byte, so 0 + 100 + ... + 900 = 4500ms.
        assert_eq!(a.tokens_per_second().map(|t| (t * 100.0).round()), Some(2222.0));

        // A failed leg counts as a leg and as an error, and stays out of the
        // latency sample: a 429 that came back in 20ms is not a fast answer.
        a.add(&row("m", "http", 20, 0));
        assert_eq!((a.legs, a.errors), (11, 1));
        assert_eq!(a.percentile(0.5), Some(500), "an error must not enter the latencies");
        // A truncated turn did answer, so it is not an error.
        a.add(&row("m", "truncated", 300, 5));
        assert_eq!(a.errors, 1);
    }

    #[test]
    fn groups_are_biggest_first() {
        let rows = vec![
            row("small", "ok", 100, 1),
            row("big", "ok", 100, 1),
            row("big", "ok", 100, 1),
        ];
        let groups = group_by(&rows, |r| r.model.clone());
        assert_eq!(groups[0].0, "big");
        assert_eq!(groups[0].1.legs, 2);
        assert_eq!(groups[1].0, "small");
    }
}
