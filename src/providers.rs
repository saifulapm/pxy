use anyhow::{Context, Result};

use crate::config::{AuthHeader, ProviderConfig};
use crate::secrets::Secrets;

/// Everything needed to fire one upstream HTTP request.
pub struct PreparedRequest {
    pub url: String,
    pub headers: Vec<(String, String)>,
}

/// Resolve URL + auth headers for a provider. `acct` selects one account of a
/// multi-account provider (its credential + headers override the provider's);
/// None means the implicit single default.
pub fn prepare(
    name: &str,
    cfg: &ProviderConfig,
    secrets: &Secrets,
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
    Ok(PreparedRequest { url, headers })
}
