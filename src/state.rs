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
    /// Per-model request/failure windows (litellm's failure-rate rule): a
    /// model that fails HALF its recent requests cools down even though no
    /// single error ever crossed the per-error cooldown ladder. In-memory
    /// only — persistent cooldowns already cover the decisive failures.
    model_health: Mutex<HashMap<String, ModelHealth>>,
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
            model_health: Mutex::new(HashMap::new()),
        })
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
    // Two scopes, keyed in one map (docs/03: OmniRoute's provider-cooldown vs
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
