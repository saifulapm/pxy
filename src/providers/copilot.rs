//! GitHub Copilot: two-stage auth. The long-lived GitHub OAuth token (ghu_/gho_)
//! mints a short-lived Copilot API token via GET
//! https://api.github.com/copilot_internal/v2/token (Authorization: `token <gh>`,
//! note `token`, not `Bearer`). Chat then goes to api.githubcopilot.com with a
//! fixed VS Code header profile. All values verified against OmniRoute
//! (open-sse/config/providerHeaderProfiles.ts, executors/github.ts).

use anyhow::{Context, Result};
use serde_json::Value;

use crate::config::ProviderConfig;
use crate::secrets::Secrets;
use crate::state::State;

use super::PreparedRequest;

const TOKEN_URL: &str = "https://api.github.com/copilot_internal/v2/token";
const DEFAULT_CHAT_URL: &str = "https://api.githubcopilot.com/chat/completions";
const EDITOR_VERSION: &str = "vscode/1.126.0";
const CHAT_PLUGIN_VERSION: &str = "copilot-chat/0.54.0";
const CHAT_USER_AGENT: &str = "GitHubCopilotChat/0.54.0";
const REFRESH_PLUGIN_VERSION: &str = "copilot/1.388.0";
const REFRESH_USER_AGENT: &str = "GithubCopilot/1.0";
const API_VERSION: &str = "2026-06-01";
/// Re-mint when fewer than this many seconds remain.
const REFRESH_LEAD_SECS: u64 = 300;

pub async fn prepare(
    name: &str,
    cfg: &ProviderConfig,
    secrets: &Secrets,
    state: &State,
    http: &reqwest::Client,
    mut headers: Vec<(String, String)>,
) -> Result<PreparedRequest> {
    let copilot_token = current_token(name, cfg, secrets, state, http).await?;

    headers.extend([
        ("authorization".into(), format!("Bearer {copilot_token}")),
        ("copilot-integration-id".into(), "vscode-chat".into()),
        ("editor-version".into(), EDITOR_VERSION.into()),
        ("editor-plugin-version".into(), CHAT_PLUGIN_VERSION.into()),
        ("user-agent".into(), CHAT_USER_AGENT.into()),
        ("openai-intent".into(), "conversation-panel".into()),
        ("x-github-api-version".into(), API_VERSION.into()),
        (
            "x-vscode-user-agent-library-version".into(),
            "electron-fetch".into(),
        ),
        // Copilot bills "agent" turns (tool-call continuations) as free; the
        // server handler overwrites this with the client's x-initiator when set.
        ("x-initiator".into(), "user".into()),
    ]);

    let url = cfg
        .base_url
        .clone()
        .unwrap_or_else(|| DEFAULT_CHAT_URL.to_string());
    Ok(PreparedRequest { url, headers })
}

async fn current_token(
    name: &str,
    cfg: &ProviderConfig,
    secrets: &Secrets,
    state: &State,
    http: &reqwest::Client,
) -> Result<String> {
    let kv_key = format!("copilot_token:{name}");
    if let Some(cached) = state.kv_get(&kv_key)? {
        if let Ok(v) = serde_json::from_str::<Value>(&cached) {
            let expires_at = expires_secs(&v["expires_at"]);
            let now = now_secs();
            if let (Some(token), Some(exp)) = (v["token"].as_str(), expires_at) {
                if exp > now + REFRESH_LEAD_SECS {
                    return Ok(token.to_string());
                }
            }
        }
    }

    // Mint a fresh one from the long-lived GitHub token stored in pass.
    let cred_ref = cfg
        .credentials
        .as_ref()
        .or(cfg.api_key.as_ref())
        .with_context(|| format!("provider {name}: credentials required"))?;
    let blob = secrets.resolve(cred_ref)?;
    let gh_token = github_access_token(&blob)?;

    let resp = http
        .get(TOKEN_URL)
        .header("authorization", format!("token {gh_token}"))
        .header("accept", "application/json")
        .header("user-agent", REFRESH_USER_AGENT)
        .header("editor-version", EDITOR_VERSION)
        .header("editor-plugin-version", REFRESH_PLUGIN_VERSION)
        .send()
        .await
        .context("copilot token mint request")?;

    let status = resp.status();
    let body = resp.text().await.unwrap_or_default();
    if !status.is_success() {
        anyhow::bail!(
            "copilot token mint failed ({status}): {body}. \
             The GitHub token in pass may be expired — re-authenticate."
        );
    }
    let v: Value = serde_json::from_str(&body).context("copilot token response")?;
    let token = v["token"]
        .as_str()
        .context("copilot token response missing 'token'")?
        .to_string();
    state.kv_set(&kv_key, &v.to_string())?;
    Ok(token)
}

/// The pass entry is either a plain token line or our OAuth JSON blob.
fn github_access_token(blob: &str) -> Result<String> {
    let trimmed = blob.trim();
    if let Ok(v) = serde_json::from_str::<Value>(trimmed) {
        if let Some(t) = v["access_token"].as_str() {
            return Ok(t.to_string());
        }
    }
    let first = trimmed.lines().next().unwrap_or_default().trim();
    if first.is_empty() {
        anyhow::bail!("empty github credential");
    }
    Ok(first.to_string())
}

/// Copilot returns expires_at as epoch seconds (number); tolerate strings.
fn expires_secs(v: &Value) -> Option<u64> {
    v.as_u64()
        .or_else(|| v.as_str().and_then(|s| s.parse().ok()))
}

fn now_secs() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}
