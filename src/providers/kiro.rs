//! Kiro / Amazon Q (AWS CodeWhisperer).
//!
//! Auth: the archived credential is a Kiro *social* login (Google/GitHub),
//! whose refresh endpoint is Kiro's own service, NOT AWS SSO-OIDC — social
//! refresh tokens cannot be exchanged at `oidc.<region>.amazonaws.com`.
//! Tokens rotate, so the same persist-before-use discipline as kimi applies
//! (kv is authoritative, pass is mirrored so the durable store stays live).
//!
//! Routing region comes from the profile ARN, never from the token: a profile
//! in us-east-1 must talk to codewhisperer.us-east-1.amazonaws.com, while
//! other regions use q.<region>.amazonaws.com. Sending a us-east-1 profile to
//! a q.* host yields empty limits and 502s.
//!
//! The profile ARN also has to ride in the request BODY, so prepare() returns
//! it as a body patch alongside the usual URL/headers.

use anyhow::{Context, Result};
use serde_json::{Value, json};
use tokio::sync::Mutex;

use crate::config::{ProviderConfig, SecretRef};
use crate::secrets::Secrets;
use crate::state::State;

use super::PreparedRequest;

const REFRESH_URL: &str = "https://prod.us-east-1.auth.desktop.kiro.dev/refreshToken";
const TARGET: &str = "AmazonCodeWhispererStreamingService.GenerateAssistantResponse";
const SDK_UA: &str = "AWS-SDK-JS/3.0.0 kiro-ide/1.0.0";
const AMZ_UA: &str = "aws-sdk-js/3.0.0 kiro-ide/1.0.0";
const REFRESH_LEAD_SECS: u64 = 300;

static REFRESH_LOCK: Mutex<()> = Mutex::const_new(());

pub async fn prepare(
    name: &str,
    cfg: &ProviderConfig,
    cred: Option<&SecretRef>,
    secrets: &Secrets,
    state: &State,
    http: &reqwest::Client,
    mut headers: Vec<(String, String)>,
) -> Result<PreparedRequest> {
    let cred_ref = cred
        .with_context(|| format!("provider {name}: credentials required"))?;
    let blob = secrets.resolve(cred_ref)?;
    let seed: Value = serde_json::from_str(blob.trim())
        .with_context(|| format!("provider {name}: credential is not the OAuth JSON blob"))?;

    let (token, profile_arn) = current_token(name, cred_ref, &seed, secrets, state, http).await?;
    let auth_method = seed["provider_specific_data"]["authMethod"]
        .as_str()
        .unwrap_or_default()
        .to_string();

    headers.extend([
        ("authorization".into(), format!("Bearer {token}")),
        ("accept".into(), "application/vnd.amazon.eventstream".into()),
        ("x-amz-target".into(), TARGET.into()),
        ("user-agent".into(), SDK_UA.into()),
        ("x-amz-user-agent".into(), AMZ_UA.into()),
        ("amz-sdk-request".into(), "attempt=1; max=3".into()),
        // A unique-per-request invocation id; AWS validates shape, not value.
        ("amz-sdk-invocation-id".into(), invocation_id(&token)),
        ("x-amzn-bedrock-cache-control".into(), "enable".into()),
        ("anthropic-beta".into(), "prompt-caching-2024-07-31".into()),
    ]);
    if auth_method == "api_key" {
        headers.push(("tokentype".into(), "API_KEY".into()));
    } else if auth_method == "external_idp" {
        headers.push(("TokenType".into(), "EXTERNAL_IDP".into()));
    }

    let url = cfg
        .base_url
        .clone()
        .unwrap_or_else(|| format!("{}/generateAssistantResponse", runtime_host(&profile_arn)));

    Ok(PreparedRequest {
        url,
        headers,
        body_patch: Some(json!({ "profileArn": profile_arn })),
    })
}

/// `arn:aws:codewhisperer:<region>:...` decides the host. us-east-1 keeps the
/// codewhisperer.* name; every other region lives under q.<region>. The
/// region is strictly validated: it is interpolated into the request host, so
/// a tampered ARN must not turn it into a different origin.
fn runtime_host(profile_arn: &str) -> String {
    let region = profile_arn
        .split(':')
        .nth(3)
        .filter(|r| !r.is_empty())
        .filter(|r| {
            // AWS region shape: eu-west-1, us-gov-west-1, cn-north-1 — never
            // "evil.com" or anything with dots/slashes.
            let mut parts = r.split('-');
            let first_ok = parts.next().is_some_and(|p| {
                !p.is_empty() && p.bytes().all(|b| b.is_ascii_lowercase())
            });
            first_ok && parts.all(|p| p.bytes().all(|b| b.is_ascii_lowercase() || b.is_ascii_digit()))
                && r.len() <= 20
        })
        .unwrap_or("us-east-1");
    if region == "us-east-1" {
        "https://codewhisperer.us-east-1.amazonaws.com".into()
    } else {
        format!("https://q.{region}.amazonaws.com")
    }
}

/// Cheap per-process unique id; AWS validates shape, not uniqueness.
fn invocation_id(token: &str) -> String {
    use std::sync::atomic::{AtomicU64, Ordering};
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let n = COUNTER.fetch_add(1, Ordering::Relaxed);
    let h: u64 = token
        .bytes()
        .take(16)
        .fold(0u64, |a, b| a.wrapping_mul(31).wrapping_add(b as u64));
    format!(
        "{:08x}-{:04x}-4{:03x}-8{:03x}-{:012x}",
        (h >> 32) as u32,
        (h >> 16) as u16,
        (h & 0xfff) as u16,
        (n & 0xfff) as u16,
        h & 0xffff_ffff_ffff
    )
}

async fn current_token(
    name: &str,
    cred_ref: &crate::config::SecretRef,
    seed: &Value,
    secrets: &Secrets,
    state: &State,
    http: &reqwest::Client,
) -> Result<(String, String)> {
    let kv_key = format!("kiro_token:{name}");
    let arn_from = |v: &Value| {
        v["provider_specific_data"]["profileArn"]
            .as_str()
            .or_else(|| v["profileArn"].as_str())
            .unwrap_or_default()
            .to_string()
    };

    if let Some(cached) = state.kv_get(&kv_key)? {
        if let Ok(v) = serde_json::from_str::<Value>(&cached) {
            if let (Some(t), Some(exp)) = (v["access_token"].as_str(), v["expires_at"].as_u64()) {
                if exp > now_secs() + REFRESH_LEAD_SECS {
                    let arn = match arn_from(&v) {
                        s if s.is_empty() => arn_from(seed),
                        s => s,
                    };
                    return Ok((t.to_string(), arn));
                }
            }
        }
    }

    let _guard = REFRESH_LOCK.lock().await;
    if let Some(cached) = state.kv_get(&kv_key)? {
        if let Ok(v) = serde_json::from_str::<Value>(&cached) {
            if let (Some(t), Some(exp)) = (v["access_token"].as_str(), v["expires_at"].as_u64()) {
                if exp > now_secs() + REFRESH_LEAD_SECS {
                    let arn = match arn_from(&v) {
                        s if s.is_empty() => arn_from(seed),
                        s => s,
                    };
                    return Ok((t.to_string(), arn));
                }
            }
        }
    }
    // Cross-process guard, same rationale as kimi's: a second pxy process
    // refreshing in the same seconds would race the (non-rotating, but
    // single-use-checked) refresh exchange. External CLIs keep their own
    // locking; this serializes pxy processes against each other.
    let _flock = super::RefreshLock::acquire(
        crate::config::data_dir().join(format!("refresh-{name}.lock")),
    )
    .await?;
    // Re-check under the file lock too.
    if let Some(cached) = state.kv_get(&kv_key)? {
        if let Ok(v) = serde_json::from_str::<Value>(&cached) {
            if let (Some(t), Some(exp)) = (v["access_token"].as_str(), v["expires_at"].as_u64()) {
                if exp > now_secs() + REFRESH_LEAD_SECS {
                    let arn = match arn_from(&v) {
                        s if s.is_empty() => arn_from(seed),
                        s => s,
                    };
                    return Ok((t.to_string(), arn));
                }
            }
        }
    }

    let refresh_token = state
        .kv_get(&kv_key)?
        .and_then(|s| serde_json::from_str::<Value>(&s).ok())
        .and_then(|v| v["refresh_token"].as_str().map(String::from))
        .or_else(|| seed["refresh_token"].as_str().map(String::from))
        .with_context(|| format!("provider {name}: no refresh token in kv or pass"))?;

    let resp = http
        .post(REFRESH_URL)
        .header("content-type", "application/json")
        .header("accept", "application/json")
        .json(&json!({ "refreshToken": refresh_token }))
        .send()
        .await
        .context("kiro token refresh request")?;

    let status = resp.status();
    let body = resp.text().await.unwrap_or_default();
    if !status.is_success() {
        // AWS reports OAuth failures as {"__type": "..."} rather than
        // {"error": "..."}; both spellings are checked because Kiro relays
        // upstream errors verbatim.
        let code = serde_json::from_str::<Value>(&body)
            .ok()
            .and_then(|v| {
                v["__type"]
                    .as_str()
                    .or(v["error"].as_str())
                    .map(String::from)
            })
            .unwrap_or_default();
        if matches!(
            code.as_str(),
            "InvalidGrantException" | "ExpiredTokenException" | "invalid_grant"
        ) {
            anyhow::bail!(
                "kiro refresh token rejected ({code}) — re-login required \
                 (Kiro IDE, or the social device flow)"
            );
        }
        anyhow::bail!("kiro token refresh failed ({status}): {body}");
    }

    let v: Value = serde_json::from_str(&body).context("kiro token response")?;
    let access = v["accessToken"]
        .as_str()
        .context("kiro token response missing accessToken")?
        .to_string();
    let new_refresh = v["refreshToken"].as_str().unwrap_or(&refresh_token);
    let expires_at = now_secs() + v["expiresIn"].as_u64().unwrap_or(3600);
    // The refresh response re-states the profile ARN; prefer it over the
    // archived one so a re-provisioned profile follows automatically.
    let profile_arn = match v["profileArn"].as_str() {
        Some(s) if !s.is_empty() => s.to_string(),
        _ => arn_from(seed),
    };

    state.kv_set(
        &kv_key,
        &json!({
            "access_token": access,
            "refresh_token": new_refresh,
            "expires_at": expires_at,
            "profileArn": profile_arn,
        })
        .to_string(),
    )?;

    if let crate::config::SecretRef::Pass { pass: entry } = cred_ref {
        let mut updated = seed.clone();
        updated["access_token"] = Value::String(access.clone());
        updated["refresh_token"] = Value::String(new_refresh.to_string());
        updated["expires_at"] = Value::String(super::kimi::iso8601(expires_at));
        if !profile_arn.is_empty() {
            updated["provider_specific_data"]["profileArn"] = Value::String(profile_arn.clone());
        }
        match serde_json::to_string_pretty(&updated) {
            Ok(b) => {
                if let Err(e) = secrets.write_pass(entry, &b) {
                    tracing::warn!(provider = name, error = %e, "pass write-back failed (kv still authoritative)");
                }
            }
            Err(e) => tracing::warn!(provider = name, error = %e, "pass write-back serialize failed"),
        }
    }

    Ok((access, profile_arn))
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
    fn runtime_host_follows_the_profile_region() {
        assert_eq!(
            runtime_host("arn:aws:codewhisperer:us-east-1:699475941385:profile/EHGA3GRVQMUK"),
            "https://codewhisperer.us-east-1.amazonaws.com"
        );
        assert_eq!(
            runtime_host("arn:aws:codewhisperer:eu-central-1:1234:profile/X"),
            "https://q.eu-central-1.amazonaws.com"
        );
        // Malformed ARN must not produce a bogus host.
        assert_eq!(
            runtime_host("garbage"),
            "https://codewhisperer.us-east-1.amazonaws.com"
        );
    }
}
