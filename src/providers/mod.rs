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

/// Resolve URL + auth headers for a provider. Cheap for API-key providers;
/// may mint/refresh tokens for OAuth kinds (copilot).
pub async fn prepare(
    name: &str,
    cfg: &ProviderConfig,
    secrets: &Secrets,
    state: &State,
    http: &reqwest::Client,
) -> Result<PreparedRequest> {
    let mut headers: Vec<(String, String)> = cfg
        .headers
        .iter()
        .map(|(k, v)| (k.clone(), v.clone()))
        .collect();

    match cfg.kind {
        ProviderKind::OpenaiCompat => {
            let url = cfg
                .base_url
                .clone()
                .with_context(|| format!("provider {name}: missing base_url"))?;
            if let Some(sref) = &cfg.api_key {
                let key = secrets.resolve_key(sref)?;
                match cfg.auth_header {
                    AuthHeader::Bearer => {
                        headers.push(("authorization".into(), format!("Bearer {key}")));
                    }
                    AuthHeader::XApiKey => {
                        headers.push(("x-api-key".into(), key));
                    }
                }
            }
            Ok(PreparedRequest { url, headers, body_patch: None })
        }
        ProviderKind::GithubCopilot => copilot::prepare(name, cfg, secrets, state, http, headers).await,
        ProviderKind::KimiCoding => kimi::prepare(name, cfg, secrets, state, http, headers).await,
        ProviderKind::Kiro => kiro::prepare(name, cfg, secrets, state, http, headers).await,
    }
}
