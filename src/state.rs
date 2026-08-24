use std::collections::HashMap;
use std::sync::Mutex;
use std::time::{Duration, Instant};

use anyhow::{Context, Result};
use jiff::Timestamp;
use rusqlite::Connection;

/// Persistent + in-memory runtime state.
///
/// sqlite holds usage counters (surviving restarts) and small KV (minted
/// copilot tokens). Cooldowns and rpm windows are in-memory: they are
/// short-lived by design (lazy expiry on read, no background timers).
pub struct State {
    db: Mutex<Connection>,
    cooldowns: Mutex<HashMap<String, Cooldown>>,
    rpm: Mutex<HashMap<String, RpmWindow>>,
}

#[derive(Debug, Clone)]
pub struct Cooldown {
    pub until: Instant,
    pub level: u32,
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

#[derive(Debug, Clone, Copy, Default)]
pub struct UsageRow {
    pub requests: u64,
    pub tokens: u64,
}

impl State {
    pub fn open(path: &std::path::Path) -> Result<Self> {
        if let Some(dir) = path.parent() {
            std::fs::create_dir_all(dir)?;
        }
        let db = Connection::open(path).context("opening state db")?;
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
            CREATE TABLE IF NOT EXISTS kv (
                k TEXT PRIMARY KEY,
                v TEXT NOT NULL
            );",
        )?;
        Ok(Self {
            db: Mutex::new(db),
            cooldowns: Mutex::new(HashMap::new()),
            rpm: Mutex::new(HashMap::new()),
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
        let db = self.db.lock().unwrap();
        for (window, start) in [("day", day_start), ("month", month_start)] {
            db.execute(
                "INSERT INTO usage (provider, window, window_start, requests, input_tokens, output_tokens)
                 VALUES (?1, ?2, ?3, 1, ?4, ?5)
                 ON CONFLICT(provider, window, window_start) DO UPDATE SET
                   requests = requests + 1,
                   input_tokens = input_tokens + excluded.input_tokens,
                   output_tokens = output_tokens + excluded.output_tokens",
                rusqlite::params![provider, window, start.to_string(), input_tokens as i64, output_tokens as i64],
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
        let db = self.db.lock().unwrap();
        for (window, start) in [("day", day_start), ("month", month_start)] {
            db.execute(
                "INSERT INTO usage (provider, window, window_start, requests, input_tokens, output_tokens)
                 VALUES (?1, ?2, ?3, 0, ?4, ?5)
                 ON CONFLICT(provider, window, window_start) DO UPDATE SET
                   input_tokens = input_tokens + excluded.input_tokens,
                   output_tokens = output_tokens + excluded.output_tokens",
                rusqlite::params![provider, window, start.to_string(), input_tokens as i64, output_tokens as i64],
            )?;
        }
        Ok(())
    }

    pub fn usage(&self, provider: &str, window: &str, start: Timestamp) -> Result<UsageRow> {
        let db = self.db.lock().unwrap();
        let row = db
            .query_row(
                "SELECT requests, input_tokens + output_tokens FROM usage
                 WHERE provider = ?1 AND window = ?2 AND window_start = ?3",
                rusqlite::params![provider, window, start.to_string()],
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
        let db = self.db.lock().unwrap();
        match db.query_row("SELECT v FROM kv WHERE k = ?1", [k], |r| r.get(0)) {
            Ok(v) => Ok(Some(v)),
            Err(rusqlite::Error::QueryReturnedNoRows) => Ok(None),
            Err(e) => Err(e.into()),
        }
    }

    pub fn kv_set(&self, k: &str, v: &str) -> Result<()> {
        let db = self.db.lock().unwrap();
        db.execute(
            "INSERT INTO kv (k, v) VALUES (?1, ?2)
             ON CONFLICT(k) DO UPDATE SET v = excluded.v",
            [k, v],
        )?;
        Ok(())
    }

    // ---- cooldowns (lazy expiry on read) ----

    pub fn cooldown(&self, provider: &str) -> Option<Cooldown> {
        let map = self.cooldowns.lock().unwrap();
        if let Some(cd) = map.get(provider) {
            // Expired cooldowns stay in the map so backoff level escalates
            // across repeated failures; they just stop blocking.
            if cd.until > Instant::now() {
                return Some(cd.clone());
            }
        }
        None
    }

    /// Put a provider in cooldown. `retry_after` (from upstream) overrides
    /// exponential backoff and resets the level (the upstream told us exactly).
    pub fn set_cooldown(&self, provider: &str, retry_after: Option<Duration>, reason: &str) {
        let mut map = self.cooldowns.lock().unwrap();
        let prev_level = map.get(provider).map(|c| c.level).unwrap_or(0);
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
            provider.to_string(),
            Cooldown { until: Instant::now() + dur, level, reason: reason.to_string() },
        );
    }

    pub fn clear_cooldown(&self, provider: &str) {
        self.cooldowns.lock().unwrap().remove(provider);
    }

    // ---- rpm sliding window ----

    pub fn rpm_effective(&self, provider: &str) -> f64 {
        let now_ms = epoch_ms();
        let idx = now_ms / RPM_WINDOW_MS;
        let elapsed = (now_ms % RPM_WINDOW_MS) as f64 / RPM_WINDOW_MS as f64;
        let mut map = self.rpm.lock().unwrap();
        let w = map.entry(provider.to_string()).or_default();
        roll(w, idx);
        w.prev * (1.0 - elapsed) + w.curr
    }

    pub fn rpm_increment(&self, provider: &str) {
        let now_ms = epoch_ms();
        let idx = now_ms / RPM_WINDOW_MS;
        let mut map = self.rpm.lock().unwrap();
        let w = map.entry(provider.to_string()).or_default();
        roll(w, idx);
        w.curr += 1.0;
    }
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
