//! Kimi coding tier ("Kimi Code CLI"): OAuth with ROTATING refresh tokens.
//! Every refresh at auth.kimi.com/api/oauth/token returns a NEW refresh token
//! and invalidates the old one, so the rotated pair is persisted to sqlite kv
//! immediately, under a lock so concurrent requests can't double-refresh
//! (a second refresh with the consumed token would kill the session).
//! Rotated pairs are ALSO written back to the pass entry (gpg encryption needs
//! only the public key, so it works headless), keeping the durable store live:
//! losing state.sqlite then costs a refresh, not a re-login. A kv-only design
//! is how the original archived token died.
//! Chat goes to api.kimi.com/coding/v1/messages?beta=true (Anthropic format,
//! x-api-key) with a 6-header X-Msh-* device identity profile that must reuse
//! the SAME device id the login was issued against (anti-bot). All values
//! verified against OmniRoute (registry/kimi/coding/runtime.ts, executors/
//! kimi.ts, tokenRefresh/providers/kimiCoding.ts).

use anyhow::{Context, Result};
use serde_json::Value;
use tokio::sync::Mutex;

use crate::config::ProviderConfig;
use crate::secrets::Secrets;
use crate::state::State;

use super::PreparedRequest;

const TOKEN_URL: &str = "https://auth.kimi.com/api/oauth/token";
/// Public client id of the Kimi Code CLI (OmniRoute publicCreds `kimi_id`).
const CLIENT_ID: &str = "17e5f671-d194-4dfb-9706-5516cb48c098";
const DEFAULT_CHAT_URL: &str = "https://api.kimi.com/coding/v1/messages?beta=true";
const PLATFORM: &str = "kimi_code_cli";
const CLI_VERSION: &str = "0.26.0";
const USER_AGENT: &str = "kimi-code-cli/0.26.0";
const ANTHROPIC_VERSION: &str = "2023-06-01";
/// Refresh when fewer than this many seconds remain (matches OmniRoute).
const REFRESH_LEAD_SECS: u64 = 300;

/// Serializes refreshes across concurrent requests. One provider instance in
/// practice; a process-wide lock is sufficient and simplest.
static REFRESH_LOCK: Mutex<()> = Mutex::const_new(());

pub async fn prepare(
    name: &str,
    cfg: &ProviderConfig,
    secrets: &Secrets,
    state: &State,
    http: &reqwest::Client,
    mut headers: Vec<(String, String)>,
) -> Result<PreparedRequest> {
    let cred_ref = cfg
        .credentials
        .as_ref()
        .or(cfg.api_key.as_ref())
        .with_context(|| format!("provider {name}: credentials required"))?;
    let blob = secrets.resolve(cred_ref)?;
    let seed: Value = serde_json::from_str(blob.trim())
        .with_context(|| format!("provider {name}: credential is not the OAuth JSON blob"))?;
    let identity = identity_headers(&seed);

    let token = current_token(name, cred_ref, &seed, &identity, secrets, state, http).await?;

    headers.push(("x-api-key".into(), token));
    headers.push(("anthropic-version".into(), ANTHROPIC_VERSION.into()));
    headers.push(("user-agent".into(), USER_AGENT.into()));
    headers.extend(identity);

    let url = cfg
        .base_url
        .clone()
        .unwrap_or_else(|| DEFAULT_CHAT_URL.to_string());
    Ok(PreparedRequest { url, headers, body_patch: None })
}

#[allow(clippy::too_many_arguments)]
async fn current_token(
    name: &str,
    cred_ref: &crate::config::SecretRef,
    seed: &Value,
    identity: &[(String, String)],
    secrets: &Secrets,
    state: &State,
    http: &reqwest::Client,
) -> Result<String> {
    let kv_key = format!("kimi_token:{name}");
    if let Some(token) = cached_valid(state, &kv_key)? {
        return Ok(token);
    }

    let _guard = REFRESH_LOCK.lock().await;
    // Another request may have refreshed while we waited for the lock.
    if let Some(token) = cached_valid(state, &kv_key)? {
        return Ok(token);
    }

    // The kv row (rotated) wins over the pass bootstrap (login snapshot).
    let refresh_token = state
        .kv_get(&kv_key)?
        .and_then(|s| serde_json::from_str::<Value>(&s).ok())
        .and_then(|v| v["refresh_token"].as_str().map(String::from))
        .or_else(|| seed["refresh_token"].as_str().map(String::from))
        .with_context(|| format!("provider {name}: no refresh token in kv or pass"))?;

    let body = form_encode(&[
        ("grant_type", "refresh_token"),
        ("refresh_token", refresh_token.as_str()),
        ("client_id", CLIENT_ID),
    ]);
    let mut req = http
        .post(TOKEN_URL)
        .header("accept", "application/json")
        .header("content-type", "application/x-www-form-urlencoded")
        .body(body);
    for (k, v) in identity {
        req = req.header(k, v);
    }
    let resp = req.send().await.context("kimi token refresh request")?;

    let status = resp.status();
    let body = resp.text().await.unwrap_or_default();
    if !status.is_success() {
        let code = serde_json::from_str::<Value>(&body)
            .ok()
            .and_then(|v| v["error"].as_str().map(String::from))
            .unwrap_or_default();
        if code == "invalid_grant" || code == "invalid_request" {
            anyhow::bail!(
                "kimi refresh token rejected ({code}) — the rotating refresh \
                 token is dead; a fresh device-flow login is required"
            );
        }
        anyhow::bail!("kimi token refresh failed ({status}): {body}");
    }

    let v: Value = serde_json::from_str(&body).context("kimi token response")?;
    let access = v["access_token"]
        .as_str()
        .context("kimi token response missing access_token")?
        .to_string();
    // Rotation: the server may or may not return a new refresh token; keep
    // the presented one when absent (matches OmniRoute).
    let new_refresh = v["refresh_token"].as_str().unwrap_or(&refresh_token);
    let expires_at = now_secs() + v["expires_in"].as_u64().unwrap_or(3600);

    // Persist BEFORE returning — losing a rotated refresh token kills the session.
    state.kv_set(
        &kv_key,
        &serde_json::json!({
            "access_token": access,
            "refresh_token": new_refresh,
            "expires_at": expires_at,
        })
        .to_string(),
    )?;

    // Mirror into pass so the durable store never goes stale. Best-effort:
    // a pass failure (locked gpg agent, say) must not fail a request that
    // already has a valid token, and kv above is authoritative either way.
    if let crate::config::SecretRef::Pass { pass: entry } = cred_ref {
        let mut updated = seed.clone();
        updated["access_token"] = Value::String(access.clone());
        updated["refresh_token"] = Value::String(new_refresh.to_string());
        updated["expires_at"] = Value::String(iso8601(expires_at));
        match serde_json::to_string_pretty(&updated) {
            Ok(body) => {
                if let Err(e) = secrets.write_pass(entry, &body) {
                    tracing::warn!(provider = name, error = %e, "pass write-back failed (kv still authoritative)");
                }
            }
            Err(e) => tracing::warn!(provider = name, error = %e, "pass write-back serialize failed"),
        }
    }
    Ok(access)
}

fn cached_valid(state: &State, kv_key: &str) -> Result<Option<String>> {
    if let Some(cached) = state.kv_get(kv_key)? {
        if let Ok(v) = serde_json::from_str::<Value>(&cached) {
            if let (Some(token), Some(exp)) = (v["access_token"].as_str(), v["expires_at"].as_u64())
            {
                if exp > now_secs() + REFRESH_LEAD_SECS {
                    return Ok(Some(token.to_string()));
                }
            }
        }
    }
    Ok(None)
}

/// The X-Msh-* profile, from the identity persisted at login
/// (provider_specific_data in the pass blob). Values are already ASCII; the
/// upstream sanitizer (strip non-\x20-\x7e, fallback "unknown") is replicated.
fn identity_headers(seed: &Value) -> Vec<(String, String)> {
    let psd = &seed["provider_specific_data"];
    let get = |k: &str| sanitize(psd[k].as_str().unwrap_or_default());
    vec![
        ("x-msh-platform".into(), PLATFORM.into()),
        ("x-msh-version".into(), CLI_VERSION.into()),
        ("x-msh-device-name".into(), get("deviceName")),
        ("x-msh-device-model".into(), get("deviceModel")),
        ("x-msh-os-version".into(), get("osVersion")),
        ("x-msh-device-id".into(), get("deviceId")),
    ]
}

fn sanitize(s: &str) -> String {
    let cleaned: String = s
        .trim()
        .chars()
        .filter(|c| (' '..='~').contains(c))
        .collect();
    let cleaned = cleaned.trim().to_string();
    if cleaned.is_empty() {
        "unknown".into()
    } else {
        cleaned
    }
}

/// Minimal application/x-www-form-urlencoded encoder (reqwest is built
/// without the form feature). Unreserved chars pass through; space becomes
/// '+'; everything else is percent-encoded.
pub(crate) fn form_encode(pairs: &[(&str, &str)]) -> String {
    fn enc(s: &str) -> String {
        let mut out = String::with_capacity(s.len());
        for b in s.bytes() {
            match b {
                b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' => {
                    out.push(b as char)
                }
                b' ' => out.push('+'),
                _ => out.push_str(&format!("%{b:02X}")),
            }
        }
        out
    }
    pairs
        .iter()
        .map(|(k, v)| format!("{}={}", enc(k), enc(v)))
        .collect::<Vec<_>>()
        .join("&")
}

/// Epoch seconds -> "YYYY-MM-DDTHH:MM:SS.000Z" (the shape OmniRoute archived,
/// kept so the pass entry stays readable by both tools).
pub fn iso8601(epoch: u64) -> String {
    let days = epoch / 86_400;
    let secs = epoch % 86_400;
    // civil-from-days (Howard Hinnant's algorithm), epoch 1970-01-01.
    let z = days as i64 + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097);
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = if m <= 2 { y + 1 } else { y };
    format!(
        "{y:04}-{m:02}-{d:02}T{:02}:{:02}:{:02}.000Z",
        secs / 3600,
        (secs % 3600) / 60,
        secs % 60
    )
}

fn now_secs() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn iso8601_matches_known_timestamps() {
        assert_eq!(iso8601(0), "1970-01-01T00:00:00.000Z");
        assert_eq!(iso8601(1_000_000_000), "2001-09-09T01:46:40.000Z");
        // leap day
        assert_eq!(iso8601(1_709_164_800), "2024-02-29T00:00:00.000Z");
        assert_eq!(iso8601(1_787_635_200), "2026-08-25T05:20:00.000Z");
    }

    #[test]
    fn form_encode_escapes_the_device_code_urn() {
        assert_eq!(
            form_encode(&[("grant_type", "urn:ietf:params:oauth:grant-type:device_code")]),
            "grant_type=urn%3Aietf%3Aparams%3Aoauth%3Agrant-type%3Adevice_code"
        );
        assert_eq!(form_encode(&[("a", "b c")]), "a=b+c");
    }

    #[test]
    fn sanitize_strips_non_ascii_and_falls_back() {
        assert_eq!(sanitize("  fedora \u{1F600} "), "fedora");
        assert_eq!(sanitize(""), "unknown");
        assert_eq!(sanitize("Linux 7.1 arm64"), "Linux 7.1 arm64");
    }
}
