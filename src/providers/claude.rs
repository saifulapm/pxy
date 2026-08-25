//! Anthropic Claude subscription (Pro/Max) via the Claude Code OAuth
//! credential at `~/.claude/.credentials.json`.
//!
//! pxy does NOT implement the login flow — it borrows the credential the
//! locally logged-in Claude Code CLI already maintains, and becomes a good
//! citizen about it:
//!
//! - The file is re-read on every request, so a refresh done by Claude Code
//!   itself is picked up immediately.
//! - Anthropic OAuth REFRESH TOKENS ROTATE. When pxy refreshes, the new pair
//!   is written BACK to the file (atomic tmp+rename, 0600, `.bak` of the
//!   previous content, foreign JSON keys preserved) so Claude Code and pxy
//!   never fight over a consumed token. Refresh happens as late as possible
//!   (5-minute lead) to minimize rotation count, serialized under a mutex
//!   with a staleness re-read inside the lock.
//! - `ensure_sentinel` (called by the router for this kind) prepends the
//!   Claude Code system sentinel when a non-Claude-Code client selected this
//!   provider manually — the OAuth inference endpoint rejects requests
//!   without it. Real Claude Code traffic already carries it (no-op).

use anyhow::{bail, Context, Result};
use serde_json::{json, Value};
use tokio::sync::Mutex;
use tracing::{info, warn};

use crate::config::ProviderConfig;
use super::PreparedRequest;

const CLIENT_ID: &str = "9d1c250a-e61b-44d9-88ed-5944d1962f5e";
const TOKEN_URL: &str = "https://api.anthropic.com/v1/oauth/token";
const DEFAULT_URL: &str = "https://api.anthropic.com/v1/messages?beta=true";
/// Rotating refresh tokens: refresh as LATE as possible (fewer rotations =
/// fewer chances to race the CLI), but early enough that a long stream
/// doesn't outlive the access token.
const REFRESH_LEAD_MS: u64 = 5 * 60 * 1000;

const SENTINEL: &str = "You are Claude Code, Anthropic's official CLI for Claude.";

static REFRESH_LOCK: Mutex<()> = Mutex::const_new(());

pub async fn prepare(
    name: &str,
    cfg: &ProviderConfig,
    http: &reqwest::Client,
    mut headers: Vec<(String, String)>,
) -> Result<PreparedRequest> {
    let path = credentials_path(cfg);
    let mut creds = read_credentials(&path)
        .with_context(|| format!("provider {name}: reading {}", path.display()))?;

    if needs_refresh(&creds) {
        let _guard = REFRESH_LOCK.lock().await;
        // Staleness re-read inside the lock: Claude Code (or a parallel pxy
        // request) may have refreshed while we waited.
        creds = read_credentials(&path)?;
        if needs_refresh(&creds) {
            creds = refresh(name, http, &path, creds).await?;
        }
    }

    let access = creds["claudeAiOauth"]["accessToken"]
        .as_str()
        .with_context(|| format!("provider {name}: no accessToken in credentials file"))?;
    headers.push(("authorization".into(), format!("Bearer {access}")));
    // The OAuth beta is mandatory for subscription tokens. Client-negotiated
    // betas are forwarded separately by the router and merge with this one.
    if !headers.iter().any(|(k, _)| k == "anthropic-beta") {
        headers.push(("anthropic-beta".into(), "oauth-2025-04-20".into()));
    }

    Ok(PreparedRequest {
        url: cfg.base_url.clone().unwrap_or_else(|| DEFAULT_URL.into()),
        headers,
        body_patch: None,
    })
}

/// The OAuth inference endpoint requires the Claude Code sentinel as the
/// first real system block. Genuine Claude Code traffic already has it; a
/// manual request from another client gets it prepended (array form so the
/// client's own system prompt survives).
pub fn ensure_sentinel(body: &mut Value) {
    let has_sentinel = match &body["system"] {
        Value::String(s) => s.starts_with(SENTINEL),
        Value::Array(blocks) => blocks
            .iter()
            .any(|b| b["text"].as_str().is_some_and(|t| t.starts_with(SENTINEL))),
        _ => false,
    };
    if has_sentinel {
        return;
    }
    let mut blocks = match body["system"].take() {
        Value::String(s) => vec![json!({"type": "text", "text": s})],
        Value::Array(a) => a,
        _ => Vec::new(),
    };
    blocks.insert(0, json!({"type": "text", "text": SENTINEL}));
    body["system"] = Value::Array(blocks);
}

fn credentials_path(cfg: &ProviderConfig) -> std::path::PathBuf {
    let raw = cfg
        .credentials_file
        .clone()
        .unwrap_or_else(|| "~/.claude/.credentials.json".into());
    match raw.strip_prefix("~/") {
        Some(rest) => dirs_home().join(rest),
        None => raw.into(),
    }
}

fn dirs_home() -> std::path::PathBuf {
    std::env::var_os("HOME").map(Into::into).unwrap_or_else(|| "/".into())
}

fn read_credentials(path: &std::path::Path) -> Result<Value> {
    let text = std::fs::read_to_string(path)?;
    let v: Value = serde_json::from_str(&text).context("credentials file is not JSON")?;
    if !v["claudeAiOauth"].is_object() {
        bail!("credentials file has no claudeAiOauth entry (not logged in?)");
    }
    Ok(v)
}

fn needs_refresh(creds: &Value) -> bool {
    let expires_at = creds["claudeAiOauth"]["expiresAt"].as_u64().unwrap_or(0);
    expires_at.saturating_sub(now_ms()) < REFRESH_LEAD_MS
}

async fn refresh(
    name: &str,
    http: &reqwest::Client,
    path: &std::path::Path,
    mut creds: Value,
) -> Result<Value> {
    let refresh_token = creds["claudeAiOauth"]["refreshToken"]
        .as_str()
        .with_context(|| format!("provider {name}: no refreshToken — re-login with `claude`"))?
        .to_string();

    info!(provider = name, "refreshing anthropic oauth token");
    let resp = http
        .post(TOKEN_URL)
        .timeout(std::time::Duration::from_secs(30))
        .header("content-type", "application/x-www-form-urlencoded")
        .header("accept", "application/json")
        .body(super::kimi::form_encode(&[
            ("grant_type", "refresh_token"),
            ("refresh_token", refresh_token.as_str()),
            ("client_id", CLIENT_ID),
        ]))
        .send()
        .await
        .context("token refresh request failed")?;
    let status = resp.status().as_u16();
    let body: Value = resp.json().await.unwrap_or_default();
    if status >= 400 {
        bail!(
            "token refresh failed ({status}): {} — re-login with `claude` if this persists",
            body["error"].as_str().unwrap_or("unknown")
        );
    }
    let access = body["access_token"]
        .as_str()
        .context("refresh response missing access_token")?;
    let expires_in = body["expires_in"].as_u64().unwrap_or(3600);

    let oauth = &mut creds["claudeAiOauth"];
    oauth["accessToken"] = json!(access);
    // Rotation: a new refresh token replaces the consumed one; keep the old
    // one only when the server didn't rotate.
    if let Some(new_refresh) = body["refresh_token"].as_str() {
        oauth["refreshToken"] = json!(new_refresh);
    }
    oauth["expiresAt"] = json!(now_ms() + expires_in * 1000);

    // Write back BEFORE first use (the kimi lesson: losing a rotated token
    // kills the session), atomically, preserving every foreign key the CLI
    // stores in the same file.
    if let Err(e) = write_credentials(path, &creds) {
        // The refreshed pair only exists in memory now — surface loudly.
        warn!(provider = name, error = %e, "FAILED to persist rotated oauth token");
        return Err(e.context("persisting rotated token"));
    }
    Ok(creds)
}

fn write_credentials(path: &std::path::Path, creds: &Value) -> Result<()> {
    use std::os::unix::fs::PermissionsExt;
    // Backup the previous content once per write.
    if let Ok(old) = std::fs::read(path) {
        let _ = std::fs::write(path.with_extension("json.bak"), old);
    }
    let tmp = path.with_extension("json.tmp");
    let text = serde_json::to_string(creds)?;
    std::fs::write(&tmp, &text)?;
    std::fs::set_permissions(&tmp, std::fs::Permissions::from_mode(0o600))?;
    std::fs::rename(&tmp, path)?;
    Ok(())
}

fn now_ms() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sentinel_prepended_only_when_missing() {
        // Real Claude Code traffic: untouched.
        let mut cc = json!({"system": [
            {"type": "text", "text": SENTINEL},
            {"type": "text", "text": "project notes"},
        ]});
        let before = cc.clone();
        ensure_sentinel(&mut cc);
        assert_eq!(cc, before);

        // Foreign client with a string system prompt: sentinel first, prompt kept.
        let mut other = json!({"system": "You are a helpful assistant."});
        ensure_sentinel(&mut other);
        let blocks = other["system"].as_array().unwrap();
        assert_eq!(blocks[0]["text"], SENTINEL);
        assert_eq!(blocks[1]["text"], "You are a helpful assistant.");

        // No system at all.
        let mut none = json!({"messages": []});
        ensure_sentinel(&mut none);
        assert_eq!(none["system"][0]["text"], SENTINEL);
    }

    #[test]
    fn refresh_gate_uses_lead() {
        let fresh = json!({"claudeAiOauth": {"expiresAt": now_ms() + 3_600_000}});
        assert!(!needs_refresh(&fresh));
        let stale = json!({"claudeAiOauth": {"expiresAt": now_ms() + 60_000}});
        assert!(needs_refresh(&stale), "inside the 5-minute lead");
        let expired = json!({"claudeAiOauth": {"expiresAt": 1}});
        assert!(needs_refresh(&expired));
    }

    #[test]
    fn write_back_preserves_foreign_keys() {
        let dir = std::env::temp_dir().join(format!("pxy-claude-test-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("credentials.json");
        let creds = json!({
            "claudeAiOauth": {"accessToken": "a", "refreshToken": "r", "expiresAt": 1,
                              "subscriptionType": "max"},
            "mcpOAuth": {"some": "foreign state"},
        });
        write_credentials(&path, &creds).unwrap();
        let back = read_credentials(&path).unwrap();
        assert_eq!(back["mcpOAuth"]["some"], "foreign state");
        assert_eq!(back["claudeAiOauth"]["subscriptionType"], "max");
        use std::os::unix::fs::PermissionsExt;
        let mode = std::fs::metadata(&path).unwrap().permissions().mode() & 0o777;
        assert_eq!(mode, 0o600);
    }
}
