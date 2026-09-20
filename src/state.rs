use std::collections::HashMap;
use std::sync::Mutex;
use std::time::{Duration, Instant};

use anyhow::{Context, Result};
use jiff::{Timestamp, Zoned};
use rusqlite::Connection;
use std::os::unix::fs::PermissionsExt;

/// Persistent + in-memory runtime state.
///
/// sqlite holds usage counters (surviving restarts), small KV (the route pin,
/// session affinity, free-quota snapshots), and a mirror of the cooldown map —
/// a restart must not
/// forget a six-hour "monthly quota exhausted" cooldown and re-probe every
/// dead provider (deploys happen mid-day). The map stays authoritative at
/// runtime (lazy expiry on read, no background timers); sqlite is only read
/// at startup. rpm windows are memory-only: a 60s window never outlives a
/// restart meaningfully.
pub struct State {
    db: Mutex<Connection>,
    cooldowns: Mutex<HashMap<String, Cooldown>>,
    rpm: Mutex<HashMap<String, RpmWindow>>,
    /// Tokens billed per provider in the same two-bucket minute as `rpm`,
    /// fed from real usage counts as they land, so the read lags the wire by
    /// one response.
    tpm: Mutex<HashMap<String, RpmWindow>>,
    /// Per-model request/failure windows (litellm's failure-rate rule): a
    /// model that fails HALF its recent requests cools down even though no
    /// single error ever crossed the per-error cooldown ladder. In-memory
    /// only — persistent cooldowns already cover the decisive failures.
    model_health: Mutex<HashMap<String, ModelHealth>>,
    /// What `[stats]` asked for. Set once by the daemon at startup; a CLI
    /// reader never writes rows, so the default stands there.
    stats: Mutex<crate::config::StatsConfig>,
    /// Epoch ms of the last retention sweep, so it is not per leg.
    stats_swept: Mutex<u64>,
}

#[derive(Debug, Clone)]
pub struct Cooldown {
    pub until: Instant,
    pub level: u32,
    /// Whether waiting out this cooldown can plausibly fix the failure.
    /// Transient errors (429/5xx/network) are; auth/credit failures are not —
    /// a revoked key does not heal in seconds, and re-firing against it can
    /// burn quota or trip provider-side abuse limits.
    pub retryable: bool,
    pub reason: String,
}

/// Two-bucket sliding window (OmniRoute pattern): effective count =
/// prev * (1 - elapsed/window) + curr, computed on read.
#[derive(Debug, Default, Clone)]
struct RpmWindow {
    bucket_index: u64,
    prev: f64,
    curr: f64,
}

const RPM_WINDOW_MS: u64 = 60_000;

/// How long a session-affinity binding stays live. Anthropic prompt caches
/// expire in minutes; an hour is generous and bounds staleness after a
/// long-idle conversation returns.
const SESSION_TTL_SECS: u64 = 3600;

#[derive(Debug, Clone, Copy, Default)]
pub struct UsageRow {
    pub requests: u64,
    pub tokens: u64,
}

/// One upstream HTTP leg, as `pxy stats` reads it back. A leg is what
/// `record_request`/`record_tokens` already count: a server-tool continuation
/// and a re-send of a dead stream are each their own. Routing never reads
/// these rows — they answer questions, they do not steer anything.
#[derive(Debug, Clone, Default)]
pub struct AttemptRow {
    /// Epoch ms at the start of the leg.
    pub ts: i64,
    /// LOCAL calendar date, as `model_usage` writes it.
    pub day: String,
    /// "chat", "embed" or "media".
    pub kind: String,
    pub agent: String,
    /// The group, alias or model id the client asked for.
    pub requested: String,
    /// Bare wire name; a multi-account provider's account is its own column so
    /// no consumer ever sees a `#` key.
    pub provider: String,
    pub account: String,
    pub model: String,
    /// Position in the candidate walk, 0 for the first choice.
    pub step: i64,
    /// 0 for a client turn, 1 for a meta-tool's sub-request.
    pub depth: i64,
    pub stream: bool,
    /// "ok", "refused", "http", "network", "timeout", "dead" or "disconnect".
    pub outcome: String,
    /// HTTP status, 0 when there was none.
    pub status: u16,
    pub error: String,
    /// Whole-leg wall time.
    pub ms: i64,
    /// To the first committed stream event, or to response headers when the
    /// leg was not streamed. 0 when unknown.
    pub ttfb_ms: i64,
    /// Billed input, INCLUDING cache traffic — the same number the quota
    /// windows count. `cache_read` and `cache_write` break it down.
    pub input: u64,
    pub output: u64,
    pub cache_read: u64,
    pub cache_write: u64,
    pub reasoning: u64,
    /// Server-tool calls this leg closed on.
    pub tool_calls: i64,
}

/// One executed server-tool call.
#[derive(Debug, Clone, Default)]
pub struct ToolRunRow {
    pub ts: i64,
    pub day: String,
    pub agent: String,
    /// The tool's canonical name ("web_search", "memory", …).
    pub tool: String,
    pub ok: bool,
    pub ms: i64,
    pub error: String,
    /// A short tool-specific note (hit count, library id), or empty.
    pub detail: String,
}

/// How often the retention sweep may run. Deleting by an indexed timestamp is
/// cheap, but not once per upstream leg.
const STATS_PRUNE_INTERVAL_MS: u64 = 3_600_000;

#[derive(Debug, Clone)]
pub struct ModelUsageRow {
    pub day: String,
    pub agent: String,
    pub provider: String,
    pub model: String,
    pub requests: u64,
    pub input_tokens: u64,
    pub output_tokens: u64,
}

impl State {
    pub fn open(path: &std::path::Path) -> Result<Self> {
        if let Some(dir) = path.parent() {
            std::fs::create_dir_all(dir)?;
        }
        let db = Connection::open(path).context("opening state db")?;
        // The db records what was asked of which provider and when — traffic
        // metadata, not for other users on the box. sqlite creates
        // db/-wal/-shm with the ambient umask, which left them 0644 on disk;
        // tighten all three at every open (idempotent, covers files the umask
        // fix postdates).
        for suffix in ["", "-wal", "-shm"] {
            let p = std::path::PathBuf::from(format!("{}{suffix}", path.display()));
            if p.exists() {
                let _ = std::fs::set_permissions(&p, std::fs::Permissions::from_mode(0o600));
            }
        }
        // The CLI (explain/doctor/status) opens the daemon's live db; without
        // a busy timeout a concurrent daemon write turns into an instant
        // "database is locked" abort instead of a few-ms wait.
        db.busy_timeout(Duration::from_secs(5))?;
        db.pragma_update(None, "journal_mode", "WAL")?;
        db.execute_batch(
            "CREATE TABLE IF NOT EXISTS usage (
                provider TEXT NOT NULL,
                window TEXT NOT NULL,
                window_start TEXT NOT NULL,
                requests INTEGER NOT NULL DEFAULT 0,
                input_tokens INTEGER NOT NULL DEFAULT 0,
                output_tokens INTEGER NOT NULL DEFAULT 0,
                PRIMARY KEY (provider, window, window_start)
            );
            CREATE TABLE IF NOT EXISTS model_usage (
                day TEXT NOT NULL,
                agent TEXT NOT NULL,
                provider TEXT NOT NULL,
                model TEXT NOT NULL,
                requests INTEGER NOT NULL DEFAULT 0,
                input_tokens INTEGER NOT NULL DEFAULT 0,
                output_tokens INTEGER NOT NULL DEFAULT 0,
                PRIMARY KEY (day, agent, provider, model)
            );
            CREATE TABLE IF NOT EXISTS attempts (
                id INTEGER PRIMARY KEY,
                ts INTEGER NOT NULL,
                day TEXT NOT NULL,
                kind TEXT NOT NULL,
                agent TEXT NOT NULL,
                requested TEXT NOT NULL,
                provider TEXT NOT NULL,
                account TEXT NOT NULL,
                model TEXT NOT NULL,
                step INTEGER NOT NULL,
                depth INTEGER NOT NULL,
                stream INTEGER NOT NULL,
                outcome TEXT NOT NULL,
                status INTEGER NOT NULL,
                error TEXT NOT NULL,
                ms INTEGER NOT NULL,
                ttfb_ms INTEGER NOT NULL,
                input INTEGER NOT NULL,
                output INTEGER NOT NULL,
                cache_read INTEGER NOT NULL,
                cache_write INTEGER NOT NULL,
                reasoning INTEGER NOT NULL,
                tool_calls INTEGER NOT NULL
            );
            CREATE INDEX IF NOT EXISTS attempts_ts ON attempts (ts);
            CREATE TABLE IF NOT EXISTS tool_runs (
                id INTEGER PRIMARY KEY,
                ts INTEGER NOT NULL,
                day TEXT NOT NULL,
                agent TEXT NOT NULL,
                tool TEXT NOT NULL,
                ok INTEGER NOT NULL,
                ms INTEGER NOT NULL,
                error TEXT NOT NULL,
                detail TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS tool_runs_ts ON tool_runs (ts);
            CREATE TABLE IF NOT EXISTS kv (
                k TEXT PRIMARY KEY,
                v TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS cooldowns (
                key TEXT PRIMARY KEY,
                until_ms INTEGER NOT NULL,
                level INTEGER NOT NULL,
                retryable INTEGER NOT NULL,
                reason TEXT NOT NULL
            );",
        )?;

        // Rehydrate cooldowns that outlived the restart; drop the expired
        // ones (their escalation level restarting at zero is acceptable).
        let now = epoch_ms();
        db.execute("DELETE FROM cooldowns WHERE until_ms <= ?1", [now as i64])?;
        let mut cooldowns = HashMap::new();
        {
            let mut stmt =
                db.prepare("SELECT key, until_ms, level, retryable, reason FROM cooldowns")?;
            let rows = stmt.query_map([], |r| {
                Ok((
                    r.get::<_, String>(0)?,
                    r.get::<_, i64>(1)?,
                    r.get::<_, i64>(2)?,
                    r.get::<_, i64>(3)?,
                    r.get::<_, String>(4)?,
                ))
            })?;
            for row in rows {
                let (key, until_ms, level, retryable, reason) = row?;
                let remaining = Duration::from_millis((until_ms as u64).saturating_sub(now));
                cooldowns.insert(
                    key,
                    Cooldown {
                        until: Instant::now() + remaining,
                        level: level as u32,
                        retryable: retryable != 0,
                        reason,
                    },
                );
            }
        }

        Ok(Self {
            db: Mutex::new(db),
            cooldowns: Mutex::new(cooldowns),
            rpm: Mutex::new(HashMap::new()),
            tpm: Mutex::new(HashMap::new()),
            model_health: Mutex::new(HashMap::new()),
            stats: Mutex::new(crate::config::StatsConfig::default()),
            stats_swept: Mutex::new(0),
        })
    }

    // ---- stats rows (wiki:state) ----

    /// Adopt the config's `[stats]` settings. The daemon calls this once at
    /// startup; without it the defaults (on, 90 days) apply.
    pub fn configure_stats(&self, cfg: &crate::config::StatsConfig) {
        *self.stats.lock().unwrap_or_else(std::sync::PoisonError::into_inner) = cfg.clone();
    }

    /// Record one upstream leg. Best-effort in the same way capture is: a
    /// write that fails is a lost report, never a failed request.
    pub fn record_attempt(&self, row: &AttemptRow) {
        let Some(retain_days) = self.stats_retention() else { return };
        let db = self.db.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        if let Err(e) = db.execute(
            "INSERT INTO attempts (ts, day, kind, agent, requested, provider, account, model,
                 step, depth, stream, outcome, status, error, ms, ttfb_ms,
                 input, output, cache_read, cache_write, reasoning, tool_calls)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14, ?15, ?16,
                     ?17, ?18, ?19, ?20, ?21, ?22)",
            rusqlite::params![
                row.ts,
                row.day,
                row.kind,
                row.agent,
                row.requested,
                row.provider,
                row.account,
                row.model,
                row.step,
                row.depth,
                row.stream as i64,
                row.outcome,
                row.status as i64,
                row.error,
                row.ms,
                row.ttfb_ms,
                row.input as i64,
                row.output as i64,
                row.cache_read as i64,
                row.cache_write as i64,
                row.reasoning as i64,
                row.tool_calls,
            ],
        ) {
            tracing::warn!(error = %e, "recording an attempt failed");
        }
        self.sweep_stats(&db, retain_days);
    }

    /// Record one executed server-tool call. Best-effort, as above.
    pub fn record_tool_run(&self, row: &ToolRunRow) {
        let Some(retain_days) = self.stats_retention() else { return };
        let db = self.db.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        if let Err(e) = db.execute(
            "INSERT INTO tool_runs (ts, day, agent, tool, ok, ms, error, detail)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)",
            rusqlite::params![
                row.ts,
                row.day,
                row.agent,
                row.tool,
                row.ok as i64,
                row.ms,
                row.error,
                row.detail,
            ],
        ) {
            tracing::warn!(error = %e, "recording a tool run failed");
        }
        self.sweep_stats(&db, retain_days);
    }

    /// Every attempt at or after `since_ms`, oldest first.
    pub fn attempts_since(&self, since_ms: i64) -> Result<Vec<AttemptRow>> {
        let db = self.db.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        let mut stmt = db.prepare(
            "SELECT ts, day, kind, agent, requested, provider, account, model, step, depth,
                    stream, outcome, status, error, ms, ttfb_ms, input, output,
                    cache_read, cache_write, reasoning, tool_calls
             FROM attempts WHERE ts >= ?1 ORDER BY ts, id",
        )?;
        let rows = stmt.query_map([since_ms], |r| {
            Ok(AttemptRow {
                ts: r.get(0)?,
                day: r.get(1)?,
                kind: r.get(2)?,
                agent: r.get(3)?,
                requested: r.get(4)?,
                provider: r.get(5)?,
                account: r.get(6)?,
                model: r.get(7)?,
                step: r.get(8)?,
                depth: r.get(9)?,
                stream: r.get::<_, i64>(10)? != 0,
                outcome: r.get(11)?,
                status: r.get::<_, i64>(12)? as u16,
                error: r.get(13)?,
                ms: r.get(14)?,
                ttfb_ms: r.get(15)?,
                input: r.get::<_, i64>(16)? as u64,
                output: r.get::<_, i64>(17)? as u64,
                cache_read: r.get::<_, i64>(18)? as u64,
                cache_write: r.get::<_, i64>(19)? as u64,
                reasoning: r.get::<_, i64>(20)? as u64,
                tool_calls: r.get(21)?,
            })
        })?;
        Ok(rows.collect::<std::result::Result<Vec<_>, _>>()?)
    }

    /// Every tool run at or after `since_ms`, oldest first.
    pub fn tool_runs_since(&self, since_ms: i64) -> Result<Vec<ToolRunRow>> {
        let db = self.db.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        let mut stmt = db.prepare(
            "SELECT ts, day, agent, tool, ok, ms, error, detail
             FROM tool_runs WHERE ts >= ?1 ORDER BY ts, id",
        )?;
        let rows = stmt.query_map([since_ms], |r| {
            Ok(ToolRunRow {
                ts: r.get(0)?,
                day: r.get(1)?,
                agent: r.get(2)?,
                tool: r.get(3)?,
                ok: r.get::<_, i64>(4)? != 0,
                ms: r.get(5)?,
                error: r.get(6)?,
                detail: r.get(7)?,
            })
        })?;
        Ok(rows.collect::<std::result::Result<Vec<_>, _>>()?)
    }

    /// `Some(retain_days)` when stats recording is on, `None` when `[stats]
    /// enabled = false` — which stops the writes and leaves the reads alone.
    fn stats_retention(&self) -> Option<u64> {
        let cfg = self.stats.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        cfg.enabled.then_some(cfg.retain_days)
    }

    /// Drop rows past the retention window, at most once an hour. Called with
    /// the db lock already held, best-effort: a failed sweep is retried at the
    /// next interval.
    fn sweep_stats(&self, db: &Connection, retain_days: u64) {
        let now = epoch_ms();
        {
            let mut last = self.stats_swept.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
            if *last != 0 && now.saturating_sub(*last) < STATS_PRUNE_INTERVAL_MS {
                return;
            }
            *last = now;
        }
        let cutoff = now.saturating_sub(retain_days.saturating_mul(86_400_000)) as i64;
        for table in ["attempts", "tool_runs"] {
            if let Err(e) = db.execute(&format!("DELETE FROM {table} WHERE ts < ?1"), [cutoff]) {
                tracing::warn!(table, error = %e, "sweeping old stats rows failed");
            }
        }
    }

    // ---- usage ----

    pub fn record_usage(
        &self,
        provider: &str,
        day_start: Timestamp,
        month_start: Timestamp,
        input_tokens: u64,
        output_tokens: u64,
    ) -> Result<()> {
        let db = self.db.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        for (window, start) in windows_for(day_start, month_start) {
            db.execute(
                "INSERT INTO usage (provider, window, window_start, requests, input_tokens, output_tokens)
                 VALUES (?1, ?2, ?3, 1, ?4, ?5)
                 ON CONFLICT(provider, window, window_start) DO UPDATE SET
                   requests = requests + 1,
                   input_tokens = input_tokens + excluded.input_tokens,
                   output_tokens = output_tokens + excluded.output_tokens",
                rusqlite::params![provider, window, start, input_tokens as i64, output_tokens as i64],
            )?;
        }
        Ok(())
    }

    /// Record token usage learned after the request was already counted.
    pub fn add_tokens(
        &self,
        provider: &str,
        day_start: Timestamp,
        month_start: Timestamp,
        input_tokens: u64,
        output_tokens: u64,
    ) -> Result<()> {
        if input_tokens == 0 && output_tokens == 0 {
            return Ok(());
        }
        let db = self.db.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        for (window, start) in windows_for(day_start, month_start) {
            db.execute(
                "INSERT INTO usage (provider, window, window_start, requests, input_tokens, output_tokens)
                 VALUES (?1, ?2, ?3, 0, ?4, ?5)
                 ON CONFLICT(provider, window, window_start) DO UPDATE SET
                   input_tokens = input_tokens + excluded.input_tokens,
                   output_tokens = output_tokens + excluded.output_tokens",
                rusqlite::params![provider, window, start, input_tokens as i64, output_tokens as i64],
            )?;
        }
        Ok(())
    }

    /// Per-(agent, provider, model) daily counters, separate from the
    /// enforcement windows above: routing never reads these. They exist so
    /// "tokens by model" can be answered for group-routed traffic — the agents'
    /// own logs only know they asked for a group name. Day is the LOCAL calendar
    /// date, matching how the usage panel groups its days.
    pub fn record_model_usage(
        &self,
        agent: &str,
        provider: &str,
        model: &str,
        request: bool,
        input_tokens: u64,
        output_tokens: u64,
    ) -> Result<()> {
        let day = Zoned::now().date().to_string();
        let db = self.db.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        db.execute(
            "INSERT INTO model_usage (day, agent, provider, model, requests, input_tokens, output_tokens)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)
             ON CONFLICT(day, agent, provider, model) DO UPDATE SET
               requests = requests + excluded.requests,
               input_tokens = input_tokens + excluded.input_tokens,
               output_tokens = output_tokens + excluded.output_tokens",
            rusqlite::params![
                day,
                agent,
                provider,
                model,
                request as i64,
                input_tokens as i64,
                output_tokens as i64
            ],
        )?;
        Ok(())
    }

    /// Every model_usage row, oldest day first (for `pxy status --json`).
    pub fn model_usage_rows(&self) -> Result<Vec<ModelUsageRow>> {
        let db = self.db.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        let mut stmt = db.prepare(
            "SELECT day, agent, provider, model, requests, input_tokens, output_tokens
             FROM model_usage ORDER BY day, agent, provider, model",
        )?;
        let rows = stmt.query_map([], |r| {
            Ok(ModelUsageRow {
                day: r.get(0)?,
                agent: r.get(1)?,
                provider: r.get(2)?,
                model: r.get(3)?,
                requests: r.get::<_, i64>(4)? as u64,
                input_tokens: r.get::<_, i64>(5)? as u64,
                output_tokens: r.get::<_, i64>(6)? as u64,
            })
        })?;
        Ok(rows.collect::<std::result::Result<Vec<_>, _>>()?)
    }

    /// Lifetime totals (the "total" window, key independent of time).
    pub fn usage_total(&self, provider: &str) -> Result<UsageRow> {
        self.usage_keyed(provider, "total", TOTAL_WINDOW_START)
    }

    pub fn usage(&self, provider: &str, window: &str, start: Timestamp) -> Result<UsageRow> {
        self.usage_keyed(provider, window, &start.to_string())
    }

    fn usage_keyed(&self, provider: &str, window: &str, start: &str) -> Result<UsageRow> {
        let db = self.db.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        let row = db
            .query_row(
                "SELECT requests, input_tokens + output_tokens FROM usage
                 WHERE provider = ?1 AND window = ?2 AND window_start = ?3",
                rusqlite::params![provider, window, start],
                |r| {
                    Ok(UsageRow {
                        requests: r.get::<_, i64>(0)? as u64,
                        tokens: r.get::<_, i64>(1)? as u64,
                    })
                },
            )
            .unwrap_or_default();
        Ok(row)
    }

    // ---- kv ----

    pub fn kv_get(&self, k: &str) -> Result<Option<String>> {
        let db = self.db.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        match db.query_row("SELECT v FROM kv WHERE k = ?1", [k], |r| r.get(0)) {
            Ok(v) => Ok(Some(v)),
            Err(rusqlite::Error::QueryReturnedNoRows) => Ok(None),
            Err(e) => Err(e.into()),
        }
    }

    pub fn kv_set(&self, k: &str, v: &str) -> Result<()> {
        let db = self.db.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        db.execute(
            "INSERT INTO kv (k, v) VALUES (?1, ?2)
             ON CONFLICT(k) DO UPDATE SET v = excluded.v",
            [k, v],
        )?;
        Ok(())
    }

    // ---- session affinity bindings ----

    /// The candidate this conversation last won on, when the binding is still
    /// fresh. TTL is enforced on READ (no prune job — a single user generates
    /// a handful of stale rows per hour, and each read is one indexed SELECT).
    pub fn session_get(&self, key: &str) -> Option<String> {
        let k = format!("session:{key}");
        let v = self.kv_get(&k).ok().flatten()?;
        let v: serde_json::Value = serde_json::from_str(&v).ok()?;
        let candidate = v["candidate"].as_str()?.to_string();
        let seen = v["seen"].as_u64()?;
        let age_ms = epoch_ms().saturating_sub(seen);
        (age_ms / 1000 <= SESSION_TTL_SECS).then_some(candidate)
    }

    /// Record the winning candidate for a conversation (called on Done).
    pub fn session_set(&self, key: &str, candidate: &str) {
        let k = format!("session:{key}");
        let v = serde_json::json!({"candidate": candidate, "seen": epoch_ms()});
        let _ = self.kv_set(&k, &v.to_string());
    }

    pub fn kv_delete(&self, k: &str) -> Result<()> {
        let db = self.db.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        db.execute("DELETE FROM kv WHERE k = ?1", [k])?;
        Ok(())
    }

    // ---- cooldowns (lazy expiry on read) ----
    //
    // Two scopes, keyed in one map (OmniRoute's provider-cooldown vs
    // model-lockout separation). Auth/credit failures are account-wide, so they
    // cool the whole provider; rate limits and upstream errors are usually
    // per-model on aggregators, so they cool only "provider/model" — otherwise
    // one flaky model sidelines every other model on the same account.

    pub fn cooldown_key(provider: &str, model: Option<&str>) -> String {
        match model {
            Some(m) => format!("{provider}/{m}"),
            None => provider.to_string(),
        }
    }

    /// Blocked if the provider is cooled down, or this specific model is.
    pub fn cooldown(&self, provider: &str, model: &str) -> Option<Cooldown> {
        let map = self.cooldowns.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        let now = Instant::now();
        for key in [provider.to_string(), Self::cooldown_key(provider, Some(model))] {
            if let Some(cd) = map.get(&key) {
                // Expired cooldowns stay in the map so backoff level escalates
                // across repeated failures; they just stop blocking.
                if cd.until > now {
                    return Some(cd.clone());
                }
            }
        }
        None
    }

    /// Put a provider (or one of its models) in cooldown. `retry_after` from
    /// upstream overrides exponential backoff and resets the level.
    pub fn set_cooldown(
        &self,
        provider: &str,
        model: Option<&str>,
        retry_after: Option<Duration>,
        retryable: bool,
        reason: &str,
    ) {
        let key = Self::cooldown_key(provider, model);
        let mut map = self.cooldowns.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        let prev_level = map.get(&key).map(|c| c.level).unwrap_or(0);
        let (dur, level) = match retry_after {
            Some(d) => (d.min(Duration::from_secs(30 * 24 * 3600)), 0),
            None => {
                let level = prev_level.saturating_add(1);
                let base = Duration::from_secs(3);
                let dur = base * 2u32.saturating_pow(level.min(6) - 1);
                (dur.min(Duration::from_secs(120)), level)
            }
        };
        map.insert(
            key.clone(),
            Cooldown { until: Instant::now() + dur, level, retryable, reason: reason.to_string() },
        );
        drop(map); // never hold two locks

        // Mirror to sqlite so restarts don't forget it. Best-effort: a write
        // failure must never block routing.
        let until_ms = epoch_ms() + dur.as_millis() as u64;
        let db = self.db.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        if let Err(e) = db.execute(
            "INSERT INTO cooldowns (key, until_ms, level, retryable, reason)
             VALUES (?1, ?2, ?3, ?4, ?5)
             ON CONFLICT(key) DO UPDATE SET
               until_ms = excluded.until_ms,
               level = excluded.level,
               retryable = excluded.retryable,
               reason = excluded.reason",
            rusqlite::params![key, until_ms as i64, level as i64, retryable as i64, reason],
        ) {
            tracing::warn!(key, error = %e, "persisting cooldown failed");
        }
    }

    /// How long until this (provider, model) pair could become eligible again,
    /// considering BOTH scopes: eligibility needs both cooldowns expired, so
    /// the wait is the max of the two. None when nothing is cooling down — or
    /// when a non-retryable cooldown blocks the pair, because no amount of
    /// waiting fixes a revoked key or exhausted credits.
    pub fn recovery_wait(&self, provider: &str, model: &str) -> Option<Duration> {
        let map = self.cooldowns.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        let now = Instant::now();
        let mut wait: Option<Duration> = None;
        for key in [provider.to_string(), Self::cooldown_key(provider, Some(model))] {
            let Some(cd) = map.get(&key) else { continue };
            if cd.until <= now {
                continue;
            }
            if !cd.retryable {
                return None;
            }
            let rem = cd.until.saturating_duration_since(now);
            wait = Some(wait.map_or(rem, |w| w.max(rem)));
        }
        wait
    }

    /// Remaining wait until this pair's cooldowns (either scope) expire,
    /// INCLUDING non-retryable ones. Unlike `recovery_wait` this is a report,
    /// not an eligibility promise: it feeds the terminal 429's Retry-After,
    /// and a drained daily tier (non-retryable, expires at reset) is exactly
    /// what the client should be told to wait for. Even a dead key's ladder
    /// cooldown is honest here — it is when pxy itself would re-attempt.
    pub fn cooldown_remaining(&self, provider: &str, model: &str) -> Option<Duration> {
        let map = self.cooldowns.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        let now = Instant::now();
        let mut wait: Option<Duration> = None;
        for key in [provider.to_string(), Self::cooldown_key(provider, Some(model))] {
            let Some(cd) = map.get(&key) else { continue };
            if cd.until <= now {
                continue;
            }
            let rem = cd.until.saturating_duration_since(now);
            wait = Some(wait.map_or(rem, |w| w.max(rem)));
        }
        wait
    }

    /// Everything currently cooling down (for the @@usage report).
    pub fn active_cooldowns(&self) -> Vec<(String, Cooldown)> {
        let map = self.cooldowns.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        let now = Instant::now();
        let mut list: Vec<(String, Cooldown)> = map
            .iter()
            .filter(|(_, c)| c.until > now)
            .map(|(k, c)| (k.clone(), c.clone()))
            .collect();
        list.sort_by(|a, b| a.0.cmp(&b.0));
        list
    }

    /// Success clears both scopes for this model.
    pub fn clear_cooldown(&self, provider: &str, model: &str) {
        let model_key = Self::cooldown_key(provider, Some(model));
        let mut map = self.cooldowns.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        map.remove(provider);
        map.remove(&model_key);
        drop(map);
        let db = self.db.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        if let Err(e) = db.execute(
            "DELETE FROM cooldowns WHERE key IN (?1, ?2)",
            rusqlite::params![provider, model_key],
        ) {
            tracing::warn!(provider, error = %e, "clearing persisted cooldown failed");
        }
    }

    // ---- rpm sliding window ----

    pub fn rpm_effective(&self, provider: &str) -> f64 {
        let now_ms = epoch_ms();
        let idx = now_ms / RPM_WINDOW_MS;
        let elapsed = (now_ms % RPM_WINDOW_MS) as f64 / RPM_WINDOW_MS as f64;
        let mut map = self.rpm.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        let w = map.entry(provider.to_string()).or_default();
        roll(w, idx);
        w.prev * (1.0 - elapsed) + w.curr
    }

    pub fn rpm_increment(&self, provider: &str) {
        let now_ms = epoch_ms();
        let idx = now_ms / RPM_WINDOW_MS;
        let mut map = self.rpm.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        let w = map.entry(provider.to_string()).or_default();
        roll(w, idx);
        w.curr += 1.0;
    }

    // ---- tpm sliding window (same shape as rpm) ----

    pub fn tpm_effective(&self, provider: &str) -> f64 {
        let now_ms = epoch_ms();
        let idx = now_ms / RPM_WINDOW_MS;
        let elapsed = (now_ms % RPM_WINDOW_MS) as f64 / RPM_WINDOW_MS as f64;
        let mut map = self.tpm.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        let w = map.entry(provider.to_string()).or_default();
        roll(w, idx);
        w.prev * (1.0 - elapsed) + w.curr
    }

    pub fn tpm_add(&self, provider: &str, tokens: u64) {
        let now_ms = epoch_ms();
        let idx = now_ms / RPM_WINDOW_MS;
        let mut map = self.tpm.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        let w = map.entry(provider.to_string()).or_default();
        roll(w, idx);
        w.curr += tokens as f64;
    }

    // ---- per-model failure-rate window (litellm rule) ----

    /// Record one model attempt outcome for the failure-rate rule. Only real
    /// attempts are counted: pre-filters and context-window skips never reach
    /// here (the caller decides).
    pub fn model_result(&self, provider: &str, model: &str, ok: bool) {
        let now_ms = epoch_ms();
        let idx = now_ms / RPM_WINDOW_MS;
        let mut map = self
            .model_health
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let h = map
            .entry(Self::cooldown_key(provider, Some(model)))
            .or_default();
        roll(&mut h.req, idx);
        roll(&mut h.fail, idx);
        h.req.curr += 1.0;
        if !ok {
            h.fail.curr += 1.0;
        }
    }

    /// True when the model failed at least half of its recent attempts
    /// (>= MIN_FAILURE_RATE_REQUESTS in the sliding 60s window). The blended
    /// two-bucket read gives the same slop as the rpm estimate.
    pub fn model_unhealthy(&self, provider: &str, model: &str) -> bool {
        let now_ms = epoch_ms();
        let elapsed = (now_ms % RPM_WINDOW_MS) as f64 / RPM_WINDOW_MS as f64;
        let map = self
            .model_health
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let Some(h) = map.get(&Self::cooldown_key(provider, Some(model))) else {
            return false;
        };
        // Blend both buckets the way rpm_effective does.
        let reqs = h.req.prev * (1.0 - elapsed) + h.req.curr;
        let fails = h.fail.prev * (1.0 - elapsed) + h.fail.curr;
        reqs >= MIN_FAILURE_RATE_REQUESTS as f64 && fails / reqs >= FAILURE_RATE_THRESHOLD
    }
}

/// Minimum attempts in the window before the failure rate means anything.
const MIN_FAILURE_RATE_REQUESTS: u32 = 5;
/// litellm's default: half the recent attempts failing = unhealthy.
const FAILURE_RATE_THRESHOLD: f64 = 0.5;

/// Request/failure pair of two-bucket windows for one model.
#[derive(Debug, Default)]
struct ModelHealth {
    req: RpmWindow,
    fail: RpmWindow,
}

/// Fixed key for the all-time window.
const TOTAL_WINDOW_START: &str = "epoch";

fn windows_for(day_start: Timestamp, month_start: Timestamp) -> [(&'static str, String); 3] {
    [
        ("day", day_start.to_string()),
        ("month", month_start.to_string()),
        ("total", TOTAL_WINDOW_START.to_string()),
    ]
}

fn roll(w: &mut RpmWindow, idx: u64) {
    if idx == w.bucket_index {
        return;
    }
    if idx == w.bucket_index + 1 {
        w.prev = w.curr;
    } else {
        w.prev = 0.0;
    }
    w.curr = 0.0;
    w.bucket_index = idx;
}

fn epoch_ms() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Each test gets its own db path — these run in parallel and would
    /// otherwise clobber each other's sqlite file.
    fn state(name: &str) -> State {
        let dir = std::env::temp_dir().join(format!("pxy-test-{}-{name}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        State::open(&dir.join("s.sqlite")).unwrap()
    }

    fn attempt(model: &str, ts: i64) -> AttemptRow {
        AttemptRow {
            ts,
            day: "2026-09-20".into(),
            kind: "chat".into(),
            agent: "claude".into(),
            requested: "aaa".into(),
            provider: "zenmux".into(),
            model: model.into(),
            outcome: "ok".into(),
            ms: 1200,
            ttfb_ms: 300,
            input: 900,
            output: 40,
            cache_read: 700,
            ..AttemptRow::default()
        }
    }

    #[test]
    fn attempt_rows_round_trip() {
        let s = state("attempts");
        let now = epoch_ms() as i64;
        s.record_attempt(&attempt("glm-5.3", now - 1000));
        s.record_attempt(&AttemptRow {
            outcome: "http".into(),
            status: 429,
            error: "rate limited".into(),
            ..attempt("qwen3-coder", now)
        });
        let rows = s.attempts_since(0).unwrap();
        assert_eq!(rows.len(), 2);
        assert_eq!(rows[0].model, "glm-5.3", "oldest first: {rows:?}");
        assert_eq!((rows[0].input, rows[0].cache_read, rows[0].ttfb_ms), (900, 700, 300));
        assert_eq!((rows[1].status, rows[1].outcome.as_str()), (429, "http"));
        assert_eq!(s.attempts_since(now).unwrap().len(), 1, "since filters by ts");
    }

    #[test]
    fn tool_run_rows_round_trip() {
        let s = state("tool_runs");
        let now = epoch_ms() as i64;
        s.record_tool_run(&ToolRunRow {
            ts: now,
            day: "2026-09-20".into(),
            agent: "codex".into(),
            tool: "web_search".into(),
            ok: false,
            ms: 1800,
            error: "brave: 429".into(),
            detail: String::new(),
        });
        let rows = s.tool_runs_since(0).unwrap();
        assert_eq!(rows.len(), 1);
        assert_eq!((rows[0].tool.as_str(), rows[0].ok, rows[0].ms), ("web_search", false, 1800));
    }

    #[test]
    fn stats_writes_stop_when_disabled() {
        let s = state("stats_off");
        s.configure_stats(&crate::config::StatsConfig { enabled: false, retain_days: 90 });
        s.record_attempt(&attempt("m", epoch_ms() as i64));
        assert!(s.attempts_since(0).unwrap().is_empty(), "disabled means nothing is written");
    }

    #[test]
    fn retention_drops_rows_past_the_window() {
        let dir = std::env::temp_dir().join(format!("pxy-test-{}-stats_prune", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let path = dir.join("s.sqlite");
        let now = epoch_ms() as i64;
        let old = now - 3 * 86_400_000;

        // Seed both tables past the window...
        let seed = State::open(&path).unwrap();
        seed.record_attempt(&attempt("old", old));
        seed.record_tool_run(&ToolRunRow {
            ts: old,
            day: "2026-09-17".into(),
            agent: "claude".into(),
            tool: "memory".into(),
            ok: true,
            ms: 3,
            error: String::new(),
            detail: String::new(),
        });
        drop(seed);

        // ...and the next write of a State that has never swept sweeps them.
        let s = State::open(&path).unwrap();
        s.configure_stats(&crate::config::StatsConfig { enabled: true, retain_days: 1 });
        s.record_attempt(&attempt("new", now));
        let rows = s.attempts_since(0).unwrap();
        assert_eq!(rows.len(), 1, "the 3-day-old attempt is past a 1-day window: {rows:?}");
        assert_eq!(rows[0].model, "new");
        assert!(s.tool_runs_since(0).unwrap().is_empty(), "tool runs age out the same way");

        // ...and only then: the sweep is hourly, so the next write leaves an
        // old row alone rather than scanning again.
        s.record_attempt(&attempt("old-again", old));
        assert_eq!(s.attempts_since(0).unwrap().len(), 2, "the sweep must not run per write");
    }

    #[test]
    fn model_usage_accumulates_per_agent_and_model() {
        let s = state("model_usage");
        s.record_model_usage("codex", "zenmux", "glm-5.3", true, 0, 0).unwrap();
        s.record_model_usage("codex", "zenmux", "glm-5.3", false, 100, 20).unwrap();
        s.record_model_usage("opencode", "zenmux", "glm-5.3", true, 5, 1).unwrap();
        let rows = s.model_usage_rows().unwrap();
        assert_eq!(rows.len(), 2, "one row per (agent, model): {rows:?}");
        let codex = rows.iter().find(|r| r.agent == "codex").unwrap();
        assert_eq!((codex.requests, codex.input_tokens, codex.output_tokens), (1, 100, 20));
        let oc = rows.iter().find(|r| r.agent == "opencode").unwrap();
        assert_eq!((oc.requests, oc.input_tokens, oc.output_tokens), (1, 5, 1));
    }

    #[test]
    fn model_cooldown_does_not_block_sibling_models() {
        let s = state("sibling");
        s.set_cooldown("go", Some("flaky"), None, true, "503");
        assert!(s.cooldown("go", "flaky").is_some());
        assert!(s.cooldown("go", "healthy").is_none(), "sibling model must stay usable");
    }

    #[test]
    fn provider_cooldown_blocks_all_models() {
        let s = state("provider_wide");
        s.set_cooldown("acct", None, None, false, "401 auth error");
        assert!(s.cooldown("acct", "any-model").is_some());
        assert!(s.cooldown("acct", "other-model").is_some());
    }

    #[test]
    fn success_clears_both_scopes() {
        let s = state("clear_both");
        s.set_cooldown("p", None, None, false, "401");
        s.set_cooldown("p", Some("m"), None, true, "429");
        s.clear_cooldown("p", "m");
        assert!(s.cooldown("p", "m").is_none());
    }

    #[test]
    fn cooldowns_survive_restart() {
        let dir = std::env::temp_dir().join(format!("pxy-test-{}-persist", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let path = dir.join("s.sqlite");

        let s = State::open(&path).unwrap();
        s.set_cooldown("p", None, Some(Duration::from_secs(3600)), false, "monthly quota");
        s.set_cooldown("p", Some("m"), Some(Duration::from_millis(1)), true, "blip");
        drop(s);

        std::thread::sleep(Duration::from_millis(20));
        let s2 = State::open(&path).unwrap();
        let cd = s2.cooldown("p", "any").expect("hour-long cooldown must survive restart");
        assert_eq!(cd.reason, "monthly quota");
        assert!(!cd.retryable, "retryability must survive too");
        let remaining = cd.until.saturating_duration_since(Instant::now());
        assert!(remaining > Duration::from_secs(3500), "remaining wait preserved: {remaining:?}");

        // The prune at open really deleted the expired "p/m" row (lazy expiry
        // would mask an unpruned row from cooldown(), so check the table).
        let raw = Connection::open(&path).unwrap();
        let count = |key: &str| -> i64 {
            raw.query_row("SELECT COUNT(*) FROM cooldowns WHERE key = ?1", [key], |r| r.get(0))
                .unwrap()
        };
        assert_eq!(count("p/m"), 0, "expired row must be pruned at open");
        assert_eq!(count("p"), 1, "live row must survive the prune");

        // clear_cooldown removes the persisted rows as well.
        s2.clear_cooldown("p", "m");
        drop(s2);
        let s3 = State::open(&path).unwrap();
        assert!(s3.cooldown("p", "any").is_none(), "cleared cooldown must stay cleared");
    }

    #[test]
    fn recovery_wait_scopes_and_retryability() {
        // Both scopes active and retryable: eligibility needs both expired,
        // so the wait is the longer of the two.
        let s = state("recovery_max");
        s.set_cooldown("p", None, Some(Duration::from_secs(2)), true, "network error");
        s.set_cooldown("p", Some("m"), Some(Duration::from_secs(5)), true, "429");
        let w = s.recovery_wait("p", "m").expect("both retryable -> wait");
        assert!(w > Duration::from_secs(4), "must wait out the LONGER scope, got {w:?}");

        // A non-retryable cooldown anywhere in the pair kills the wait.
        let s2 = state("recovery_auth");
        s2.set_cooldown("p", None, None, false, "401 auth error");
        s2.set_cooldown("p", Some("m"), None, true, "429");
        assert_eq!(s2.recovery_wait("p", "m"), None, "revoked key does not heal by waiting");

        // Nothing cooling down: nothing to wait for.
        let s3 = state("recovery_none");
        assert_eq!(s3.recovery_wait("p", "m"), None);
    }
}
