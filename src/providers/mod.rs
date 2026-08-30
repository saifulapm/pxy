pub mod claude;
pub mod copilot;
pub mod kimi;
pub mod kiro;

use anyhow::{Context, Result};

use crate::config::{AuthHeader, ProviderConfig, ProviderKind};
use crate::secrets::Secrets;
use crate::state::State;

/// Everything needed to fire one upstream HTTP request.
pub struct PreparedRequest {
    pub url: String,
    pub headers: Vec<(String, String)>,
    /// Fields the provider must inject into the request body (kiro's
    /// profileArn), merged after format translation.
    pub body_patch: Option<serde_json::Value>,
}

/// Exclusive advisory lock for token refreshes, released when dropped (fd
/// close). Acquired on a blocking thread so a contended lock never stalls a
/// runtime worker. Shared by every OAuth provider: claude (against the Claude
/// Code CLI's own lock file), kimi and kiro (against a second pxy process —
/// an external CLI's refresh keeps its own locking, which pxy cannot join).
pub(crate) struct RefreshLock(#[allow(dead_code)] std::fs::File);

impl RefreshLock {
    pub(crate) async fn acquire(path: std::path::PathBuf) -> Result<Self> {
        tokio::task::spawn_blocking(move || {
            use std::os::fd::AsRawFd as _;
            use std::os::unix::fs::OpenOptionsExt;
            let f = std::fs::OpenOptions::new()
                .create(true)
                .write(true)
                .mode(0o600)
                .open(&path)
                .with_context(|| format!("opening lock file {}", path.display()))?;
            let rc = unsafe { libc::flock(f.as_raw_fd(), libc::LOCK_EX) };
            anyhow::ensure!(rc == 0, "flock failed: {}", std::io::Error::last_os_error());
            Ok(Self(f))
        })
        .await
        .context("lock task panicked")?
    }
}

/// Resolve URL + auth headers for a provider. Cheap for API-key providers;
/// may mint/refresh tokens for OAuth kinds (copilot). `acct` selects one
/// account of a multi-account provider (its credential + headers override the
/// provider's); None means the implicit single default.
pub async fn prepare(
    name: &str,
    cfg: &ProviderConfig,
    secrets: &Secrets,
    state: &State,
    http: &reqwest::Client,
    acct: Option<&crate::config::Account>,
) -> Result<PreparedRequest> {
    let mut headers: Vec<(String, String)> = cfg
        .headers
        .iter()
        .map(|(k, v)| (k.clone(), v.clone()))
        .collect();
    if let Some(a) = acct {
        for (k, v) in &a.headers {
            // Account headers OVERRIDE the provider's same-named ones.
            headers.retain(|(k2, _)| k2 != k);
            headers.push((k.clone(), v.clone()));
        }
    }
    // The account's credential wins; the provider's is the single-account
    // default (validation forbids both being set at once).
    let cred = acct
        .and_then(|a| a.credential())
        .or_else(|| cfg.credentials.as_ref().or(cfg.api_key.as_ref()));

    match cfg.kind {
        ProviderKind::OpenaiCompat => {
            let url = cfg
                .base_url
                .clone()
                .with_context(|| format!("provider {name}: missing base_url"))?;
            if let Some(sref) = cred {
                let key = secrets.resolve_key(sref)?;
                match cfg.auth_header {
                    AuthHeader::Bearer => {
                        headers.push(("authorization".into(), format!("Bearer {key}")));
                    }
                    AuthHeader::XApiKey => {
                        headers.push(("x-api-key".into(), key));
                    }
                    AuthHeader::XiApiKey => {
                        headers.push(("xi-api-key".into(), key));
                    }
                }
            }
            Ok(PreparedRequest { url, headers, body_patch: None })
        }
        ProviderKind::GithubCopilot => {
            copilot::prepare(name, cfg, cred, secrets, state, http, headers).await
        }
        ProviderKind::KimiCoding => {
            kimi::prepare(name, cfg, cred, secrets, state, http, headers).await
        }
        ProviderKind::Kiro => kiro::prepare(name, cfg, cred, secrets, state, http, headers).await,
        ProviderKind::ClaudeOauth => claude::prepare(name, cfg, http, headers).await,
    }
}
