//! Phase 2: non-chat capabilities — images, audio (STT/TTS), rerank, video,
//! web search and URL fetch. Simple pass-through handlers in the mould of
//! `/v1/embeddings`: resolve `provider/model` (or a bare id, or a configured
//! default), attach auth, translate the few non-OpenAI dialects, and record
//! usage under `<provider>#media` so media never counts against chat limits.

pub mod audio;
pub mod cli;
pub mod dashscope;
pub mod images;
pub mod rerank;
pub mod search;
pub mod video;

use axum::Json;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use serde_json::json;

use crate::config::{Config, MediaConfig, ProviderConfig};
use crate::router::App;

/// Which capability a request targets; selects the model list + URL.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Capability {
    Image,
    Transcription,
    Speech,
    Rerank,
    Video,
}

impl Capability {
    pub fn models<'a>(&self, m: &'a MediaConfig) -> &'a [String] {
        match self {
            Capability::Image => &m.image_models,
            Capability::Transcription => &m.transcription_models,
            Capability::Speech => &m.speech_models,
            Capability::Rerank => &m.rerank_models,
            Capability::Video => &m.video_models,
        }
    }

    /// Capability URL, falling back to the provider's `run_url` template.
    pub fn url<'a>(&self, m: &'a MediaConfig) -> Option<&'a str> {
        let specific = match self {
            Capability::Image => m.images_url.as_deref(),
            Capability::Transcription => m.transcription_url.as_deref(),
            Capability::Speech => m.speech_url.as_deref(),
            Capability::Rerank => m.rerank_url.as_deref(),
            Capability::Video => m.video_url.as_deref(),
        };
        specific.or(m.run_url.as_deref())
    }

    fn default_model<'a>(&self, cfg: &'a Config) -> Option<&'a str> {
        match self {
            Capability::Image => cfg.media.image.as_deref(),
            Capability::Transcription => cfg.media.transcription.as_deref(),
            Capability::Speech => cfg.media.speech.as_deref(),
            Capability::Rerank => cfg.media.rerank.as_deref(),
            Capability::Video => cfg.media.video.as_deref(),
        }
    }
}

pub struct Resolved<'a> {
    pub provider: String,
    pub model: String,
    pub cfg: &'a ProviderConfig,
    pub media: &'a MediaConfig,
    /// Capability URL with `{model}` substituted.
    pub url: String,
}

/// Resolve a requested model to (provider, model, url). Accepts
/// `provider/model` (first-slash split — cloudflare ids contain slashes),
/// a bare id matched against every provider's list (config order), or
/// empty/"auto" -> the `[media]` default for this capability.
pub fn resolve<'a>(cfg: &'a Config, capability: Capability, requested: &str) -> Option<Resolved<'a>> {
    let requested = if requested.is_empty() || requested == "auto" {
        capability.default_model(cfg)?
    } else {
        requested
    };

    let usable = |p: &&'a ProviderConfig| -> Option<&'a MediaConfig> {
        let m = p.media.as_ref()?;
        if p.enabled && capability.url(m).is_some() { Some(m) } else { None }
    };

    // Provider-prefix parse first; a miss falls back to the bare-id scan so
    // ids that themselves contain slashes (cloudflare's @cf/...) still work
    // without the provider prefix.
    let by_prefix = requested.split_once('/').and_then(|(prov, model)| {
        let p = cfg.providers.get(prov)?;
        let m = usable(&p)?;
        Some((prov.to_string(), model.to_string(), p, m))
    });
    let (prov, model, pcfg, media) = match by_prefix {
        Some(hit) => hit,
        None => cfg.providers.iter().find_map(|(name, p)| {
            let m = usable(&p)?;
            capability
                .models(m)
                .iter()
                .any(|id| id == requested)
                .then(|| (name.clone(), requested.to_string(), p, m))
        })?,
    };

    let url = capability.url(media)?.replace("{model}", &model);
    Some(Resolved { provider: prov, model, cfg: pcfg, media, url })
}

/// Usage/cooldown key for media traffic: isolated from the chat counters so
/// image bytes never eat a chat token budget and vice versa.
pub fn media_key(provider: &str) -> String {
    format!("{provider}#media")
}

pub fn error_response(status: StatusCode, msg: impl std::fmt::Display) -> Response {
    (
        status,
        Json(json!({"error": {"message": msg.to_string(), "type": "api_error"}})),
    )
        .into_response()
}

/// Auth headers for a media request. Media providers are all plain API-key
/// providers, so this is `prepare()` minus the OAuth kinds and the base_url
/// requirement.
pub fn auth_headers(app: &App, cfg: &ProviderConfig) -> anyhow::Result<Vec<(String, String)>> {
    let mut headers: Vec<(String, String)> =
        cfg.headers.iter().map(|(k, v)| (k.clone(), v.clone())).collect();
    if let Some(sref) = &cfg.api_key {
        let key = app.secrets.resolve_key(sref)?;
        match cfg.auth_header {
            crate::config::AuthHeader::Bearer => {
                headers.push(("authorization".into(), format!("Bearer {key}")));
            }
            crate::config::AuthHeader::XApiKey => {
                headers.push(("x-api-key".into(), key));
            }
            crate::config::AuthHeader::XiApiKey => {
                headers.push(("xi-api-key".into(), key));
            }
        }
    }
    Ok(headers)
}

/// True when the media-only daily cap is exhausted.
pub fn cap_exhausted(app: &App, provider: &str, media: &MediaConfig) -> bool {
    let Some(cap) = media.daily_requests else { return false };
    used_requests(app, &media_key(provider), "day") >= cap
}

pub fn used_requests(app: &App, key: &str, window: &str) -> u64 {
    let limits = crate::config::Limits::default();
    match crate::usage::current_windows(&limits, jiff::Timestamp::now()) {
        Ok(w) => {
            let start = if window == "day" { w.day_start } else { w.month_start };
            app.state.usage(key, window, start).map(|u| u.requests).unwrap_or(0)
        }
        Err(_) => 0,
    }
}

/// Count one request against a media/service counter. Called BEFORE the
/// upstream request: caps are billing safety, so a call the upstream billed
/// but whose response we lost must still have been counted (over-counting is
/// the safe direction).
pub fn record(app: &App, key: &str) {
    let limits = crate::config::Limits::default();
    if let Ok(w) = crate::usage::current_windows(&limits, jiff::Timestamp::now()) {
        if let Err(e) = app.state.record_usage(key, w.day_start, w.month_start, 0, 0) {
            tracing::warn!("recording media usage for {key}: {e:#}");
        }
    }
}

/// Add units (tokens, characters) learned from a successful response.
pub fn add_units(app: &App, key: &str, units: u64) {
    let limits = crate::config::Limits::default();
    if let Ok(w) = crate::usage::current_windows(&limits, jiff::Timestamp::now()) {
        if let Err(e) = app.state.add_tokens(key, w.day_start, w.month_start, units, 0) {
            tracing::warn!("recording media units for {key}: {e:#}");
        }
    }
}

/// Failure classification, mirroring chat (router's classify_error):
/// auth/credit failures cool the whole media pool of the provider; rate
/// limits, timeouts and 5xx cool just the model. Other 4xx are the CALLER's
/// fault (bad params) — no cooldown, or a bad request would sideline the
/// provider for the next valid one.
pub fn cool_on_failure(app: &App, provider: &str, model: &str, status: u16) {
    let key = media_key(provider);
    let reason = format!("http {status}");
    match status {
        401 | 402 | 403 => app.state.set_cooldown(&key, None, None, false, &reason),
        408 | 409 | 429 | 500..=599 => app.state.set_cooldown(&key, Some(model), None, true, &reason),
        _ => {}
    }
}

/// Pass an upstream error through unmodified — status, content-type and body
/// bytes. Chat clients rely on unmangled error bodies; an HTML 502 from an
/// edge proxy must not turn into `{}`.
pub async fn passthrough_error(resp: reqwest::Response) -> Response {
    let status = resp.status().as_u16();
    let content_type = resp
        .headers()
        .get("content-type")
        .and_then(|v| v.to_str().ok())
        .unwrap_or("application/json")
        .to_string();
    let bytes = resp.bytes().await.unwrap_or_default();
    (
        StatusCode::from_u16(status).unwrap_or(StatusCode::BAD_GATEWAY),
        [("content-type", content_type)],
        bytes.to_vec(),
    )
        .into_response()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::MediaKind;
    use std::collections::BTreeMap;

    fn test_config() -> Config {
        let toml = r#"
            [server]
            port = 1
            [providers.cloudflare]
            base_url = "https://cf.example/v1/chat/completions"
            [providers.cloudflare.media]
            kind = "cloudflare"
            run_url = "https://cf.example/ai/run/{model}"
            image_models = ["@cf/black-forest-labs/flux-1-schnell"]
            transcription_models = ["@cf/openai/whisper-large-v3-turbo"]
            [providers.groq]
            base_url = "https://groq.example/v1/chat/completions"
            [providers.groq.media]
            transcription_url = "https://groq.example/v1/audio/transcriptions"
            transcription_models = ["whisper-large-v3-turbo"]
            [media]
            transcription = "groq/whisper-large-v3-turbo"
        "#;
        toml::from_str(toml).unwrap()
    }

    #[test]
    fn resolves_prefixed_model_with_slashes() {
        let cfg = test_config();
        let r = resolve(&cfg, Capability::Image, "cloudflare/@cf/black-forest-labs/flux-1-schnell")
            .unwrap();
        assert_eq!(r.provider, "cloudflare");
        assert_eq!(r.model, "@cf/black-forest-labs/flux-1-schnell");
        assert_eq!(r.url, "https://cf.example/ai/run/@cf/black-forest-labs/flux-1-schnell");
        assert_eq!(r.media.kind, MediaKind::Cloudflare);
    }

    #[test]
    fn resolves_bare_id_and_default() {
        let cfg = test_config();
        let bare = resolve(&cfg, Capability::Transcription, "whisper-large-v3-turbo").unwrap();
        // Bare ids scan config order; cloudflare's list doesn't contain the
        // groq id, so it lands on groq.
        assert_eq!(bare.provider, "groq");
        let default = resolve(&cfg, Capability::Transcription, "auto").unwrap();
        assert_eq!(default.provider, "groq");
        assert_eq!(default.url, "https://groq.example/v1/audio/transcriptions");
        assert!(resolve(&cfg, Capability::Speech, "auto").is_none());
    }

    #[test]
    fn media_config_parses_full_table() {
        let m: MediaConfig = toml::from_str(
            r#"
            kind = "elevenlabs"
            transcription_url = "https://api.elevenlabs.io/v1/speech-to-text"
            transcription_models = ["scribe_v1"]
            speech_url = "https://api.elevenlabs.io/v1/text-to-speech/{voice}"
            speech_models = ["eleven_turbo_v2_5"]
            daily_requests = 50
            [voices]
            default = "EXAVITQu4vr4xnSDxMaL"
            nova = "FGY2WhTYpPnrIDTdsKH5"
            "#,
        )
        .unwrap();
        assert_eq!(m.kind, MediaKind::Elevenlabs);
        assert_eq!(m.voices["default"], "EXAVITQu4vr4xnSDxMaL");
        assert_eq!(m.daily_requests, Some(50));
        let _: &BTreeMap<String, String> = &m.voices;
    }
}
