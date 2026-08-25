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

    fn default_models<'a>(&self, cfg: &'a Config) -> &'a [String] {
        let chain = match self {
            Capability::Image => &cfg.media.image,
            Capability::Transcription => &cfg.media.transcription,
            Capability::Speech => &cfg.media.speech,
            Capability::Rerank => &cfg.media.rerank,
            Capability::Video => &cfg.media.video,
        };
        chain.as_ref().map(|c| c.as_slice()).unwrap_or_default()
    }

    pub fn label(&self) -> &'static str {
        match self {
            Capability::Image => "image",
            Capability::Transcription => "transcription",
            Capability::Speech => "speech",
            Capability::Rerank => "rerank",
            Capability::Video => "video",
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

/// Ordered candidate chain for a request. An explicit `provider/model` stays
/// a single candidate (its errors pass through raw, mirroring chat); a bare
/// id yields EVERY provider listing it; empty/"auto" walks the configured
/// `[media]` default chain in order (a single string is a chain of one).
pub fn resolve_chain<'a>(
    cfg: &'a Config,
    capability: Capability,
    requested: &str,
) -> Vec<Resolved<'a>> {
    if requested.is_empty() || requested == "auto" {
        let mut chain: Vec<Resolved<'a>> = Vec::new();
        for entry in capability.default_models(cfg) {
            for r in resolve_one(cfg, capability, entry) {
                if !chain.iter().any(|c| c.provider == r.provider && c.model == r.model) {
                    chain.push(r);
                }
            }
        }
        return chain;
    }
    resolve_one(cfg, capability, requested)
}

/// Resolve one requested id. Accepts `provider/model` (first-slash split —
/// cloudflare ids contain slashes) or a bare id matched against every
/// provider's list.
fn resolve_one<'a>(cfg: &'a Config, capability: Capability, requested: &str) -> Vec<Resolved<'a>> {
    let usable = |p: &&'a ProviderConfig| -> Option<&'a MediaConfig> {
        let m = p.media.as_ref()?;
        // Video needs the poll URL too; a provider without one can't serve
        // the job flow and must not enter a chain (it would burn a counted
        // request per attempt without ever calling upstream).
        let complete = capability != Capability::Video || m.video_status_url.is_some();
        if p.enabled && capability.url(m).is_some() && complete { Some(m) } else { None }
    };
    let build = |prov: String, model: String, pcfg: &'a ProviderConfig, media: &'a MediaConfig| {
        let url = capability.url(media)?.replace("{model}", &model);
        Some(Resolved { provider: prov, model, cfg: pcfg, media, url })
    };

    // Provider-prefix parse first; a miss falls back to the bare-id scan so
    // ids that themselves contain slashes (cloudflare's @cf/...) still work
    // without the provider prefix.
    if let Some((prov, model)) = requested.split_once('/')
        && let Some(p) = cfg.providers.get(prov)
        && let Some(m) = usable(&p)
    {
        return build(prov.to_string(), model.to_string(), p, m)
            .into_iter()
            .collect();
    }
    cfg.providers
        .iter()
        .filter_map(|(name, p)| {
            let m = usable(&p)?;
            capability
                .models(m)
                .iter()
                .any(|id| id == requested)
                .then(|| build(name.clone(), requested.to_string(), p, m))
                .flatten()
        })
        .collect()
}

/// Outcome of one candidate attempt in a media failover chain.
pub enum Attempt {
    Ok(Response),
    /// Provider-side failure: try the next candidate. The response is kept
    /// as the answer of last resort.
    Retryable(Response),
    /// The caller's fault (or a definitive answer): return immediately —
    /// every other provider would reject the same request identically.
    Fatal(Response),
}

/// Walk the chain: gate each candidate (cooldown + cap), attempt, and fail
/// over on provider-side failures. The last failure body is the client's
/// answer when the whole chain is down. (Boxed futures rather than AsyncFn:
/// axum needs Send futures, which AsyncFn bounds can't express yet.)
pub async fn run_chain<'a>(
    app: &'a App,
    capability: Capability,
    requested: &str,
    attempt: impl for<'r> Fn(&'r Resolved<'a>) -> futures_util::future::BoxFuture<'r, Attempt>,
) -> Response {
    let chain = resolve_chain(&app.cfg, capability, requested);
    if chain.is_empty() {
        return error_response(
            StatusCode::NOT_FOUND,
            format!("{} model '{requested}' not found", capability.label()),
        );
    }
    let mut last: Option<Response> = None;
    for r in &chain {
        if let Some(gate) = preflight(app, r) {
            last = Some(gate);
            continue;
        }
        match attempt(r).await {
            Attempt::Ok(resp) => return resp,
            // Multi-candidate carve-out mirroring chat's classify_error:
            // media model lists churn too, and one delisted id must not kill
            // the whole chain. The cooldown is non-retryable — a 404 doesn't
            // heal by waiting. An explicit single-candidate request still
            // gets the raw 404 back.
            Attempt::Fatal(resp)
                if chain.len() > 1 && resp.status() == StatusCode::NOT_FOUND =>
            {
                app.state.set_cooldown(
                    &media_key(&r.provider),
                    Some(&r.model),
                    None,
                    false,
                    "404 model not found upstream",
                );
                tracing::warn!(
                    candidate = %format!("{}/{}", r.provider, r.model),
                    "media failover (upstream 404)"
                );
                last = Some(resp);
            }
            Attempt::Fatal(resp) => return resp,
            Attempt::Retryable(resp) => {
                tracing::warn!(
                    candidate = %format!("{}/{}", r.provider, r.model),
                    "media failover"
                );
                last = Some(resp);
            }
        }
    }
    last.expect("chain is non-empty")
}

/// Cooldown + media-cap gate shared by all media handlers. When it passes,
/// the request is COUNTED immediately (see `record`): the upstream call is
/// about to happen, and a billed-but-lost response must not escape the cap.
pub fn preflight(app: &App, r: &Resolved<'_>) -> Option<Response> {
    let key = media_key(&r.provider);
    if let Some(cd) = app.state.cooldown(&key, &r.model) {
        return Some(error_response(
            StatusCode::TOO_MANY_REQUESTS,
            format!("{key} cooling down ({})", cd.reason),
        ));
    }
    if cap_exhausted(app, &r.provider, r.media) {
        return Some(error_response(
            StatusCode::TOO_MANY_REQUESTS,
            format!("{key} daily media cap reached"),
        ));
    }
    record(app, &key);
    None
}

/// Classify an upstream error status: cool down exactly like chat, pass the
/// body through unmodified either way. Fatal 4xx (caller's fault) stops the
/// chain; provider-side failures walk on.
pub async fn failed_attempt(app: &App, r: &Resolved<'_>, resp: reqwest::Response) -> Attempt {
    let status = resp.status().as_u16();
    cool_on_failure(app, &r.provider, &r.model, status);
    let retryable = matches!(status, 401 | 402 | 403 | 408 | 409 | 429) || status >= 500;
    let response = passthrough_error(resp).await;
    if retryable { Attempt::Retryable(response) } else { Attempt::Fatal(response) }
}

/// Transport failure before any status: cool the provider's media pool.
pub fn network_failure(app: &App, r: &Resolved<'_>, e: impl std::fmt::Display) -> Attempt {
    app.state
        .set_cooldown(&media_key(&r.provider), None, None, true, "network error");
    Attempt::Retryable(error_response(StatusCode::BAD_GATEWAY, format!("network: {e}")))
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
        let chain =
            resolve_chain(&cfg, Capability::Image, "cloudflare/@cf/black-forest-labs/flux-1-schnell");
        assert_eq!(chain.len(), 1, "explicit provider/model stays a single candidate");
        let r = &chain[0];
        assert_eq!(r.provider, "cloudflare");
        assert_eq!(r.model, "@cf/black-forest-labs/flux-1-schnell");
        assert_eq!(r.url, "https://cf.example/ai/run/@cf/black-forest-labs/flux-1-schnell");
        assert_eq!(r.media.kind, MediaKind::Cloudflare);
    }

    #[test]
    fn resolves_bare_id_and_default() {
        let cfg = test_config();
        let bare = resolve_chain(&cfg, Capability::Transcription, "whisper-large-v3-turbo");
        // Bare ids scan providers ALPHABETICALLY (BTreeMap); cloudflare's
        // list doesn't contain the groq id, so it lands on groq.
        assert_eq!(bare.len(), 1);
        assert_eq!(bare[0].provider, "groq");
        let default = resolve_chain(&cfg, Capability::Transcription, "auto");
        assert_eq!(default[0].provider, "groq");
        assert_eq!(default[0].url, "https://groq.example/v1/audio/transcriptions");
        assert!(resolve_chain(&cfg, Capability::Speech, "auto").is_empty());
    }

    #[test]
    fn default_list_becomes_ordered_chain() {
        let toml = r#"
            [server]
            port = 1
            [providers.a]
            base_url = "https://a.example/chat"
            [providers.a.media]
            images_url = "https://a.example/img"
            image_models = ["ma"]
            [providers.b]
            base_url = "https://b.example/chat"
            [providers.b.media]
            images_url = "https://b.example/img"
            image_models = ["mb"]
            [media]
            image = ["b/mb", "a/ma", "b/mb"]
        "#;
        let cfg: Config = toml::from_str(toml).unwrap();
        let chain = resolve_chain(&cfg, Capability::Image, "auto");
        // Configured order wins (not provider alphabetical), duplicates drop.
        assert_eq!(
            chain.iter().map(|r| format!("{}/{}", r.provider, r.model)).collect::<Vec<_>>(),
            ["b/mb", "a/ma"]
        );
    }

    #[test]
    fn bare_id_walks_every_provider_listing_it() {
        let toml = r#"
            [server]
            port = 1
            [providers.a]
            base_url = "https://a.example/chat"
            [providers.a.media]
            transcription_url = "https://a.example/stt"
            transcription_models = ["whisper"]
            [providers.b]
            base_url = "https://b.example/chat"
            [providers.b.media]
            transcription_url = "https://b.example/stt"
            transcription_models = ["whisper"]
        "#;
        let cfg: Config = toml::from_str(toml).unwrap();
        let chain = resolve_chain(&cfg, Capability::Transcription, "whisper");
        assert_eq!(chain.len(), 2, "same id on two providers = a failover pair");
    }

    #[tokio::test]
    async fn chain_fails_over_to_next_provider() {
        use axum::routing::post;
        // Provider a's endpoint 500s; b answers a valid OpenAI image body.
        let router = axum::Router::new()
            .route(
                "/a",
                post(|| async { (StatusCode::INTERNAL_SERVER_ERROR, "boom") }),
            )
            .route(
                "/b",
                post(|| async {
                    Json(serde_json::json!({"created": 1, "data": [{"url": "https://x/y.png"}]}))
                }),
            );
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move { axum::serve(listener, router).await.unwrap() });
        let base = format!("http://{addr}");

        let cfg: Config = toml::from_str(&format!(
            r#"
            [server]
            [providers.a]
            [providers.a.media]
            images_url = "{base}/a"
            image_models = ["ma"]
            [providers.b]
            [providers.b.media]
            images_url = "{base}/b"
            image_models = ["mb"]
            [media]
            image = ["a/ma", "b/mb"]
            "#
        ))
        .unwrap();
        let catalog = crate::catalog::Catalog::from_config(&cfg);
        let dir = std::env::temp_dir().join(format!("pxy-media-it-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let app = std::sync::Arc::new(crate::router::App {
            catalog,
            secrets: crate::secrets::Secrets::new(),
            state: crate::state::State::open(&dir.join("s.sqlite")).unwrap(),
            http: reqwest::Client::new(),
            cfg,
        });

        let resp = crate::media::images::generations(
            axum::extract::State(app.clone()),
            Json(serde_json::json!({"model": "auto", "prompt": "p"})),
        )
        .await;
        assert_eq!(resp.status(), StatusCode::OK, "must fail over past the 500");
        assert_eq!(resp.headers().get("x-pxy-provider").unwrap(), "b/mb");
        // The failed model cooled down in the media pool scope only.
        assert!(app.state.cooldown(&media_key("a"), "ma").is_some());
        assert!(app.state.cooldown("a", "ma").is_none(), "chat scope untouched");
    }

    #[tokio::test]
    async fn upstream_404_walks_the_chain_with_cooldown() {
        use axum::routing::post;
        // Provider a delisted its model (404); b still serves.
        let router = axum::Router::new()
            .route("/a", post(|| async { (StatusCode::NOT_FOUND, "model gone") }))
            .route(
                "/b",
                post(|| async {
                    Json(serde_json::json!({"created": 1, "data": [{"url": "https://x/y.png"}]}))
                }),
            );
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move { axum::serve(listener, router).await.unwrap() });
        let base = format!("http://{addr}");

        let cfg: Config = toml::from_str(&format!(
            r#"
            [server]
            [providers.a]
            [providers.a.media]
            images_url = "{base}/a"
            image_models = ["ma"]
            [providers.b]
            [providers.b.media]
            images_url = "{base}/b"
            image_models = ["mb"]
            [media]
            image = ["a/ma", "b/mb"]
            "#
        ))
        .unwrap();
        let catalog = crate::catalog::Catalog::from_config(&cfg);
        let dir = std::env::temp_dir().join(format!("pxy-media-404-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let app = std::sync::Arc::new(crate::router::App {
            catalog,
            secrets: crate::secrets::Secrets::new(),
            state: crate::state::State::open(&dir.join("s.sqlite")).unwrap(),
            http: reqwest::Client::new(),
            cfg,
        });

        let resp = crate::media::images::generations(
            axum::extract::State(app.clone()),
            Json(serde_json::json!({"model": "auto", "prompt": "p"})),
        )
        .await;
        assert_eq!(resp.status(), StatusCode::OK, "a delisted model must not kill the chain");
        assert_eq!(resp.headers().get("x-pxy-provider").unwrap(), "b/mb");
        // Non-retryable model-scoped cooldown on the delisted id.
        let cd = app.state.cooldown(&media_key("a"), "ma").expect("cooldown set");
        assert!(cd.reason.contains("404"));
        assert!(!cd.retryable);
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
