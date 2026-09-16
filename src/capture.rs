//! Opt-in request artifact capture (docs/09 §7): the client request and the
//! upstream request/response bodies, written to disk for translation
//! debugging. Off unless `[capture] enabled = true`.
//!
//! Everything here is best-effort and bounded: a capture failure must never
//! fail the request, no artifact is produced unless explicitly enabled, and
//! credentials are masked before the bytes ever touch disk.

use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};

use serde_json::Value;

use crate::config::{CaptureConfig, Config, data_dir, write_atomic};

/// Orders artifacts written within the same millisecond.
static SEQ: AtomicU64 = AtomicU64::new(0);

pub fn enabled(cfg: &Config) -> bool {
    cfg.capture.enabled
}

fn capture_dir(cap: &CaptureConfig) -> PathBuf {
    cap.dir
        .as_ref()
        .map(PathBuf::from)
        .unwrap_or_else(|| data_dir().join("captures"))
}

/// Write one bounded, secret-masked artifact. Best-effort: an IO error is
/// logged and dropped.
pub fn record(cfg: &Config, label: &str, value: &Value) {
    let cap = &cfg.capture;
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
    let millis = jiff::Timestamp::now().as_millisecond();
    let seq = SEQ.fetch_add(1, Ordering::Relaxed);
    let path = capture_dir(cap).join(format!("{millis}-{seq}-{label}.json"));
    if let Err(e) = write_atomic(&path, &bytes) {
        tracing::warn!(error = %e, "capture write failed");
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
        // A string with no token shape is untouched.
        assert_eq!(mask_string("just prose"), None);
    }

    #[test]
    fn record_writes_bounded_artifacts_when_enabled() {
        let dir = std::env::temp_dir().join(format!("pxy-capture-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let mut cfg: Config = toml::from_str("[server]\n").unwrap();
        assert!(!enabled(&cfg), "off by default");
        cfg.capture.enabled = true;
        cfg.capture.dir = Some(dir.to_string_lossy().into_owned());
        cfg.capture.max_bytes = 64;
        record(&cfg, "test", &json!({"big": "x".repeat(500)}));
        let files: Vec<_> = std::fs::read_dir(&dir).unwrap().filter_map(|e| e.ok()).collect();
        assert_eq!(files.len(), 1);
        let body = std::fs::read_to_string(files[0].path()).unwrap();
        assert!(body.len() < 200, "artifact must be bounded, got {}", body.len());
        assert!(body.contains("[truncated]"));
        std::fs::remove_dir_all(&dir).ok();
    }
}
