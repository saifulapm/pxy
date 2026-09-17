//! Opt-in request artifact capture: the client request and the
//! upstream request/response bodies, written to disk for translation
//! debugging. Off unless `[capture] enabled = true`.
//!
//! Everything here is best-effort and bounded: a capture failure must never
//! fail the request, no artifact is produced unless explicitly enabled, and
//! credentials are masked before the bytes ever touch disk.
//!
//! One `Txn` covers one upstream attempt, so the client request, the upstream
//! request and the upstream response of that attempt share an id and are
//! found together. With `on_error` set, artifacts are buffered and written
//! only when the attempt fails — the mode to leave on, since volume tracks
//! problems rather than traffic.

use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};

use serde_json::Value;

use crate::config::{CaptureConfig, Config, data_dir, write_atomic};

/// Orders transactions started within the same millisecond.
static SEQ: AtomicU64 = AtomicU64::new(0);
/// Millis of the last retention sweep; pruning is time-gated so it costs
/// nothing per request.
static LAST_PRUNE: AtomicU64 = AtomicU64::new(0);

pub fn capture_dir(cap: &CaptureConfig) -> PathBuf {
    cap.dir
        .as_ref()
        .map(PathBuf::from)
        .unwrap_or_else(|| data_dir().join("captures"))
}

/// One upstream attempt's capture buffer. Disabled (all no-ops) unless
/// `[capture] enabled`.
pub struct Txn {
    cap: Option<CaptureConfig>,
    id: String,
    buffered: Vec<(String, Value)>,
}

impl Txn {
    pub fn new(cfg: &Config) -> Self {
        Self {
            cap: cfg.capture.enabled.then(|| cfg.capture.clone()),
            id: if cfg.capture.enabled {
                format!(
                    "{}-{}",
                    jiff::Timestamp::now().as_millisecond(),
                    SEQ.fetch_add(1, Ordering::Relaxed)
                )
            } else {
                String::new()
            },
            buffered: Vec::new(),
        }
    }

    /// Add one artifact. Written immediately, or buffered until `finish` when
    /// the config is error-only.
    pub fn add(&mut self, label: &str, value: &Value) {
        let Some(cap) = &self.cap else {
            return;
        };
        if cap.on_error {
            self.buffered.push((label.to_string(), value.clone()));
        } else {
            write(cap, &self.id, label, value);
        }
    }

    /// Flush if this attempt failed (or immediately, when not error-only).
    pub fn finish(self, success: bool) {
        let Some(cap) = &self.cap else {
            return;
        };
        if !(success && cap.on_error) {
            for (label, value) in &self.buffered {
                write(cap, &self.id, label, value);
            }
        }
        prune(cap);
    }
}

/// One-shot artifact with a fresh id; test-only convenience.
#[cfg(test)]
pub fn record(cfg: &Config, label: &str, value: &Value) {
    let mut txn = Txn::new(cfg);
    txn.add(label, value);
    txn.finish(false);
}

fn write(cap: &CaptureConfig, id: &str, label: &str, value: &Value) {
    let mut v = value.clone();
    mask(&mut v);
    let mut bytes = match serde_json::to_vec_pretty(&v) {
        Ok(b) => b,
        Err(_) => return,
    };
    let limit = cap.max_bytes as usize;
    if bytes.len() > limit {
        bytes.truncate(limit);
        bytes.extend_from_slice(b"\n[truncated]\n");
    }
    let path = capture_dir(cap).join(format!("{id}__{label}.json"));
    if let Err(e) = write_atomic(&path, &bytes) {
        tracing::warn!(error = %e, "capture write failed");
    }
}

/// Retention: drop artifacts older than `max_age_secs`, then the oldest ones
/// beyond `max_files`. Time-gated (at most once per 30s) so the directory
/// scan is not on every request. Best-effort: an unreadable dir is ignored.
fn prune(cap: &CaptureConfig) {
    let now = jiff::Timestamp::now().as_millisecond() as u64;
    let last = LAST_PRUNE.load(Ordering::Relaxed);
    if now.saturating_sub(last) < 30_000 {
        return;
    }
    LAST_PRUNE.store(now, Ordering::Relaxed);

    let Ok(entries) = std::fs::read_dir(capture_dir(cap)) else {
        return;
    };
    let mut files: Vec<(std::time::SystemTime, PathBuf)> = entries
        .flatten()
        .filter_map(|e| {
            let name = e.file_name();
            let name = name.to_string_lossy();
            if !name.ends_with(".json") {
                return None;
            }
            let modified = e.metadata().ok()?.modified().ok()?;
            Some((modified, e.path()))
        })
        .collect();
    files.sort_by_key(|(t, _)| *t);

    let now_sys = std::time::SystemTime::now();
    let mut live: Vec<PathBuf> = Vec::new();
    for (modified, path) in files {
        let age = now_sys
            .duration_since(modified)
            .map(|d| d.as_secs())
            .unwrap_or(0);
        if age > cap.max_age_secs {
            let _ = std::fs::remove_file(&path);
        } else {
            live.push(path);
        }
    }
    if (live.len() as u64) > cap.max_files {
        let excess = live.len() as u64 - cap.max_files;
        for path in live.into_iter().take(excess as usize) {
            let _ = std::fs::remove_file(&path);
        }
    }
}

/// Mask obvious secrets in place: values under credential-ish keys, and
/// common token shapes embedded in strings (a user pasting a key into a
/// prompt). Deliberately conservative — an unmasked key defeats the point.
fn mask(v: &mut Value) {
    match v {
        Value::Object(map) => {
            for (k, val) in map.iter_mut() {
                if is_secret_key(k) {
                    *val = Value::String("[redacted]".into());
                } else {
                    mask(val);
                }
            }
        }
        Value::Array(items) => items.iter_mut().for_each(mask),
        Value::String(s) => {
            if let Some(m) = mask_string(s) {
                *s = m;
            }
        }
        _ => {}
    }
}

fn is_secret_key(k: &str) -> bool {
    matches!(
        k.to_ascii_lowercase().as_str(),
        "api_key"
            | "apikey"
            | "api-key"
            | "authorization"
            | "x-api-key"
            | "xi-api-key"
            | "password"
            | "secret"
            | "token"
            | "access_token"
            | "refresh_token"
            | "credentials"
            | "client_secret"
    )
}

const PREFIXES: &[&str] = &["sk-", "Bearer ", "AIza", "xoxb-", "ghp_", "gho_", "ghs_"];

fn mask_string(s: &str) -> Option<String> {
    if !PREFIXES.iter().any(|p| s.contains(p)) {
        return None;
    }
    let mut out = s.to_string();
    for p in PREFIXES {
        out = mask_prefix(&out, p);
    }
    Some(out)
}

/// Replace the token following each `prefix` with `[redacted]`, keeping the
/// prefix. Token chars are the base64/url-ish set real keys use.
fn mask_prefix(s: &str, prefix: &str) -> String {
    let is_token_char =
        |c: char| c.is_ascii_alphanumeric() || matches!(c, '-' | '_' | '.' | '~' | '+' | '/' | '=');
    let mut out = String::with_capacity(s.len());
    let mut rest = s;
    while let Some(pos) = rest.find(prefix) {
        out.push_str(&rest[..pos + prefix.len()]);
        let after = &rest[pos + prefix.len()..];
        let end = after.find(|c: char| !is_token_char(c)).unwrap_or(after.len());
        if end > 0 {
            out.push_str("[redacted]");
        }
        rest = &after[end..];
    }
    out.push_str(rest);
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn masks_credential_keys_and_token_shapes() {
        let mut v = json!({
            "api_key": "plain",
            "nested": {"authorization": "Bearer abc"},
            "text": "use sk-abc123DEF here and Bearer xyz789, then done",
        });
        mask(&mut v);
        assert_eq!(v["api_key"], "[redacted]");
        assert_eq!(v["nested"]["authorization"], "[redacted]");
        let t = v["text"].as_str().unwrap();
        assert!(t.contains("sk-[redacted]"), "{t}");
        assert!(t.contains("Bearer [redacted]"), "{t}");
        assert!(t.contains("then done"), "ordinary prose survives: {t}");
        assert_eq!(mask_string("just prose"), None);
    }

    fn test_cfg(dir: &std::path::Path) -> Config {
        let mut cfg: Config = toml::from_str("[server]\n").unwrap();
        cfg.capture.enabled = true;
        cfg.capture.dir = Some(dir.to_string_lossy().into_owned());
        cfg.capture.max_bytes = 64;
        cfg
    }

    fn files(dir: &std::path::Path) -> Vec<PathBuf> {
        std::fs::read_dir(dir)
            .map(|d| d.filter_map(|e| e.ok().map(|e| e.path())).collect())
            .unwrap_or_default()
    }

    #[test]
    fn record_writes_bounded_artifacts_when_enabled() {
        let dir = std::env::temp_dir().join(format!("pxy-capture-a-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let cfg = test_cfg(&dir);
        // Disabled: no-op even with a dir set.
        let mut disabled: Config = toml::from_str("[server]\n").unwrap();
        disabled.capture.dir = Some(dir.to_string_lossy().into_owned());
        let mut txn = Txn::new(&disabled);
        txn.add("x", &json!({"a": 1}));
        txn.finish(false);
        assert!(files(&dir).is_empty(), "disabled capture writes nothing");
        record(&cfg, "test", &json!({"big": "x".repeat(500)}));
        let fs = files(&dir);
        assert_eq!(fs.len(), 1);
        let body = std::fs::read_to_string(&fs[0]).unwrap();
        assert!(body.len() < 200, "bounded: {}", body.len());
        assert!(body.contains("[truncated]"));
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn error_only_mode_keeps_failures_and_drops_successes() {
        let dir = std::env::temp_dir().join(format!("pxy-capture-b-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let mut cfg = test_cfg(&dir);
        cfg.capture.on_error = true;

        let mut txn = Txn::new(&cfg);
        txn.add("upstream-request", &json!({"ok": 1}));
        txn.finish(true);
        assert!(files(&dir).is_empty(), "success must not be captured");

        let mut txn = Txn::new(&cfg);
        txn.add("client-request", &json!({"a": 1}));
        txn.add("upstream-request", &json!({"b": 2}));
        txn.finish(false);
        let fs = files(&dir);
        assert_eq!(fs.len(), 2, "every artifact of a failed attempt is written");
        // Shared correlation id: everything before the `__` separator.
        let stems: std::collections::HashSet<String> = fs
            .iter()
            .map(|p| {
                p.file_name()
                    .unwrap()
                    .to_string_lossy()
                    .split("__")
                    .next()
                    .unwrap()
                    .to_string()
            })
            .collect();
        assert_eq!(stems.len(), 1, "artifacts of one attempt share an id: {stems:?}");
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn retention_caps_file_count() {
        let dir = std::env::temp_dir().join(format!("pxy-capture-c-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let mut cfg = test_cfg(&dir);
        cfg.capture.max_files = 3;
        cfg.capture.max_age_secs = 3600;
        // Defeat the 30s prune gate between writes.
        for i in 0..6 {
            LAST_PRUNE.store(0, Ordering::Relaxed);
            record(&cfg, "x", &json!({"i": i}));
        }
        assert_eq!(files(&dir).len(), 3, "oldest pruned past max_files");
        std::fs::remove_dir_all(&dir).ok();
    }
}
