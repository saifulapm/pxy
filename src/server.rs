use std::sync::Arc;

use anyhow::{Context, Result};
use axum::extract::State;
use axum::http::{HeaderMap, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use jiff::Timestamp;
use serde_json::{json, Value};
use tracing::info;

use crate::catalog::Catalog;
use crate::config::Config;
use crate::router::{self, App, ClientContext, ClientFormat, Outcome, SharedApp};
use crate::secrets::Secrets;
use crate::state::State as PxyState;
use crate::translate::estimate_tokens;
use crate::usage::current_windows;

pub async fn serve(cfg: Config) -> Result<()> {
    // Everything this daemon writes is single-user and part of it is secret
    // (the state db carries OAuth refresh tokens): create files 0600 from the
    // start rather than trusting the ambient umask. State::open additionally
    // chmods anything that already exists.
    unsafe { libc::umask(0o077) };
    let state = PxyState::open(&crate::config::data_dir().join("state.sqlite"))?;
    let catalog = Catalog::from_config(&cfg);
    let http = reqwest::Client::builder()
        .connect_timeout(std::time::Duration::from_secs(10))
        .build()?;
    let port = cfg.server.port;
    let app = Arc::new(App { cfg, catalog, secrets: Secrets::new(), state, http });

    let router = Router::new()
        .route("/v1/chat/completions", post(chat_completions))
        .route("/v1/embeddings", post(embeddings))
        .route("/v1/responses", post(responses))
        .route("/v1/messages", post(messages))
        .route("/v1/messages/count_tokens", post(count_tokens))
        .route("/v1/models", get(models))
        .route("/v1/images/generations", post(crate::media::images::generations))
        .route("/v1/audio/transcriptions", post(crate::media::audio::transcriptions))
        .route("/v1/audio/speech", post(crate::media::audio::speech))
        .route("/v1/rerank", post(crate::media::rerank::rerank))
        .route("/v1/videos/generations", post(crate::media::video::generations))
        .route(
            "/v1/search",
            post(crate::media::search::search),
        )
        .route(
            "/v1/fetch",
            post(crate::media::search::fetch).get(crate::media::search::fetch_get),
        )
        // fx (vercel-labs/fx) impersonates its AI Gateway here: the generation
        // endpoint plus the catalog/credits GETs it makes. See translate/aisdk.
        .route("/v3/ai/language-model", post(ai_language_model))
        .route("/coding-agent/v1/models", get(fx_models))
        .route("/coding-agent/v1/credits", get(fx_credits))
        .route("/healthz", get(healthz))
        // Claude Code posts telemetry batches here; a 404 makes it retry.
        // Accept and discard (litellm ships the same stub for the same reason).
        .route(
            "/api/event_logging/batch",
            post(|| async { Json(json!({"status": "ok"})) }),
        )
        .fallback(not_found)
        // axum's default body cap is 2 MB, which rejects any real audio
        // upload before the transcription handler runs.
        .layer(axum::extract::DefaultBodyLimit::max(64 * 1024 * 1024))
        .with_state(app);

    let addr = std::net::SocketAddr::from(([127, 0, 0, 1], port));
    let listener = tokio::net::TcpListener::bind(addr)
        .await
        .with_context(|| format!("binding {addr}"))?;
    info!("pxy listening on http://{addr}");
    axum::serve(listener, router).await?;
    Ok(())
}

fn client_ctx(headers: &HeaderMap) -> ClientContext {
    ClientContext {
        initiator: headers
            .get("x-initiator")
            .and_then(|v| v.to_str().ok())
            .map(String::from),
        anthropic_beta: headers
            .get("anthropic-beta")
            .and_then(|v| v.to_str().ok())
            .map(String::from),
        agent: client_agent(headers),
    }
}

/// Which coding agent sent this request, for the per-model usage stats.
/// An explicit x-pxy-agent header wins; otherwise `pxy launch` smuggles the
/// agent name as a `:agent` suffix on the api key (uniform across agents —
/// every one of them sends the key somewhere, while only some can be taught
/// to send a custom header). The key itself is a soft gate and never checked,
/// so any suffix shape is safe to accept.
fn client_agent(headers: &HeaderMap) -> Option<String> {
    let name_ok = |s: &str| {
        !s.is_empty()
            && s.len() <= 32
            && s.chars().all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_')
    };
    if let Some(v) = headers.get("x-pxy-agent").and_then(|v| v.to_str().ok()) {
        let v = v.trim();
        if name_ok(v) {
            return Some(v.to_string());
        }
    }
    for header in ["authorization", "x-api-key"] {
        let Some(value) = headers.get(header).and_then(|v| v.to_str().ok()) else { continue };
        let token = value.strip_prefix("Bearer ").unwrap_or(value).trim();
        if let Some((_, agent)) = token.rsplit_once(':') {
            if name_ok(agent) {
                return Some(agent.to_string());
            }
        }
    }
    None
}

fn outcome_response(outcome: Outcome, client_format: ClientFormat) -> Response {
    match outcome {
        Outcome::Json { status, body, provider } => {
            let mut resp = (
                StatusCode::from_u16(status).unwrap_or(StatusCode::INTERNAL_SERVER_ERROR),
                Json(body),
            )
                .into_response();
            if let Some(p) = provider {
                if let Ok(v) = p.parse() {
                    resp.headers_mut().insert("x-pxy-provider", v);
                }
            }
            resp
        }
        Outcome::Stream { provider, body } => {
            let mut builder = Response::builder()
                .status(StatusCode::OK)
                .header("content-type", "text/event-stream")
                .header("cache-control", "no-cache")
                .header("connection", "keep-alive");
            if let Ok(v) = provider.parse::<axum::http::HeaderValue>() {
                builder = builder.header("x-pxy-provider", v);
            }
            let _ = client_format;
            builder.body(body).unwrap_or_else(|_| {
                StatusCode::INTERNAL_SERVER_ERROR.into_response()
            })
        }
    }
}

async fn chat_completions(
    State(app): State<SharedApp>,
    headers: HeaderMap,
    Json(payload): Json<Value>,
) -> Response {
    let ctx = client_ctx(&headers);
    let outcome = router::handle_chat(app, ClientFormat::Openai, payload, ctx).await;
    outcome_response(outcome, ClientFormat::Openai)
}

/// OpenAI Responses API (codex-cli): translate to a chat-completions call and
/// rewrite the outcome — streamed chat chunks become Responses SSE events;
/// JSON becomes a `response` object. Errors pass through in their chat shape
/// (codex reads `{"error": {...}}` fine).
async fn responses(
    State(app): State<SharedApp>,
    headers: HeaderMap,
    Json(payload): Json<Value>,
) -> Response {
    use futures_util::StreamExt;

    let ctx = client_ctx(&headers);
    let model = payload["model"]
        .as_str()
        .map(str::to_string)
        .unwrap_or_else(|| app.cfg.default_route());
    let chat_payload = crate::translate::responses::request(&payload);
    let outcome = router::handle_chat(app, ClientFormat::Openai, chat_payload, ctx).await;
    match outcome {
        Outcome::Json { status, body, provider } => {
            let body = if status == 200 {
                crate::translate::responses::response(&body, provider.as_deref().unwrap_or(&model))
            } else {
                body
            };
            outcome_response(Outcome::Json { status, body, provider }, ClientFormat::Openai)
        }
        Outcome::Stream { provider, body } => {
            let state = crate::translate::responses::StreamState::new(Timestamp::now().as_second());
            let parser = crate::translate::sse::SseParser::new();
            let upstream = body.into_data_stream();
            let stream = futures_util::stream::unfold(
                (upstream, parser, state, false),
                |(mut upstream, mut parser, mut state, done)| async move {
                    if done {
                        return None;
                    }
                    match upstream.next().await {
                        Some(Ok(bytes)) => {
                            let mut out = String::new();
                            for ev in parser.feed(&bytes) {
                                out.push_str(&state.on_data(&ev.data));
                            }
                            Some((
                                Ok::<_, std::io::Error>(bytes::Bytes::from(out)),
                                (upstream, parser, state, false),
                            ))
                        }
                        other => {
                            // A mid-stream transport error is NOT a clean
                            // end: finishing with "completed" would make
                            // codex render a truncated turn as a success.
                            if matches!(other, Some(Err(_))) {
                                state.fail();
                            }
                            let tail = bytes::Bytes::from(state.finish());
                            Some((Ok(tail), (upstream, parser, state, true)))
                        }
                    }
                },
            );
            outcome_response(
                Outcome::Stream { provider, body: axum::body::Body::from_stream(stream) },
                ClientFormat::Openai,
            )
        }
    }
}

/// `POST /v3/ai/language-model` — the fx generation endpoint. Model id and
/// streaming arrive as HEADERS in this dialect, not body fields.
async fn ai_language_model(
    State(app): State<SharedApp>,
    headers: HeaderMap,
    Json(payload): Json<Value>,
) -> Response {
    use futures_util::StreamExt;

    let hdr = |name: &str| headers.get(name).and_then(|v| v.to_str().ok());
    let model = hdr("ai-language-model-id")
        .map(str::to_string)
        .unwrap_or_else(|| app.cfg.default_route());
    let stream = hdr("ai-language-model-streaming") != Some("false");
    let ctx = client_ctx(&headers);

    let chat_payload = crate::translate::aisdk::request(&payload, &model, stream);
    let outcome = router::handle_chat(app, ClientFormat::Openai, chat_payload, ctx).await;
    match outcome {
        Outcome::Json { status, body, provider } => {
            let body = if status == 200 { crate::translate::aisdk::response(&body) } else { body };
            outcome_response(Outcome::Json { status, body, provider }, ClientFormat::Openai)
        }
        Outcome::Stream { provider, body } => {
            let state = crate::translate::aisdk::StreamState::new(&model);
            let parser = crate::translate::sse::SseParser::new();
            let upstream = body.into_data_stream();
            let stream = futures_util::stream::unfold(
                (upstream, parser, state, false),
                |(mut upstream, mut parser, mut state, done)| async move {
                    if done {
                        return None;
                    }
                    match upstream.next().await {
                        Some(Ok(bytes)) => {
                            let mut out = String::new();
                            for ev in parser.feed(&bytes) {
                                out.push_str(&state.on_data(&ev.data));
                            }
                            Some((
                                Ok::<_, std::io::Error>(bytes::Bytes::from(out)),
                                (upstream, parser, state, false),
                            ))
                        }
                        other => {
                            // A mid-stream transport error is NOT a clean
                            // end: finishing with "stop" would make fx render
                            // a truncated turn as a success.
                            if matches!(other, Some(Err(_))) {
                                state.fail();
                            }
                            let tail = bytes::Bytes::from(state.finish());
                            Some((Ok(tail), (upstream, parser, state, true)))
                        }
                    }
                },
            );
            outcome_response(
                Outcome::Stream { provider, body: axum::body::Body::from_stream(stream) },
                ClientFormat::Openai,
            )
        }
    }
}

/// fx's model catalog. Entries need `type: "language"` or fx drops them;
/// `tags` drive its capability display. Ids must be byte-equal to what fx
/// will send back in `ai-language-model-id`.
async fn fx_models(State(app): State<SharedApp>) -> Json<Value> {
    let mut data: Vec<Value> = Vec::new();
    let mut push = |id: String, ctx: u64, max_out: u64, tools: bool| {
        let mut tags = vec![json!("tool-use")];
        if !tools {
            tags.clear();
        }
        data.push(json!({
            "id": id,
            "type": "language",
            "owned_by": "pxy",
            "tags": tags,
            "context_window": ctx,
            "max_tokens": max_out,
        }));
    };
    for (name, group) in app.catalog.groups() {
        let (ctx, max_out) = crate::catalog::chain_limits(&group.chain);
        push(name.clone(), ctx, max_out, true);
    }
    for cand in app.catalog.models() {
        push(
            cand.full_id(),
            cand.model.context_length,
            cand.model.max_output_tokens,
            cand.model.tool_call != Some(false),
        );
    }
    Json(json!({"data": data}))
}

/// fx's credit check. All three fields must be STRINGS or fx drops them.
/// pxy is free-first and does no dollar accounting, so this is cosmetic.
async fn fx_credits() -> Json<Value> {
    Json(json!({"balance": "0", "used": "0", "plan": "pxy"}))
}

async fn messages(
    State(app): State<SharedApp>,
    headers: HeaderMap,
    Json(payload): Json<Value>,
) -> Response {
    let ctx = client_ctx(&headers);
    let outcome = router::handle_chat(app, ClientFormat::Anthropic, payload, ctx).await;
    outcome_response(outcome, ClientFormat::Anthropic)
}

/// Embeddings passthrough. Model resolution: "provider/model" explicitly, or a
/// bare id matched against providers' `embedding_models` (config order).
async fn embeddings(State(app): State<SharedApp>, Json(mut payload): Json<Value>) -> Response {
    let requested = payload["model"].as_str().unwrap_or("").to_string();
    let resolved = if let Some((prov, model)) = requested.split_once('/') {
        app.cfg
            .providers
            .get(prov)
            .filter(|p| p.enabled && p.embeddings_url.is_some())
            .map(|p| (prov.to_string(), model.to_string(), p))
    } else {
        app.cfg
            .providers
            .iter()
            .find(|(_, p)| {
                p.enabled
                    && p.embeddings_url.is_some()
                    && p.embedding_models.iter().any(|m| m == &requested)
            })
            .map(|(name, p)| (name.clone(), requested.clone(), p))
    };
    let Some((prov_name, model_id, provider_cfg)) = resolved else {
        return (
            StatusCode::NOT_FOUND,
            Json(json!({"error": {"message": format!("embedding model '{requested}' not found"),
                                  "type": "invalid_request_error"}})),
        )
            .into_response();
    };

    // Plain API-key auth: `prepare()` insists on base_url, which
    // embeddings-only providers (voyage, tencent-vl) legitimately lack.
    let headers = match crate::media::auth_headers(&app, provider_cfg) {
        Ok(h) => h,
        Err(e) => {
            return (
                StatusCode::BAD_GATEWAY,
                Json(json!({"error": {"message": format!("{e:#}"), "type": "api_error"}})),
            )
                .into_response();
        }
    };
    let url = provider_cfg.embeddings_url.clone().unwrap();

    payload["model"] = json!(model_id);
    let mut req = app
        .http
        .post(&url)
        .timeout(std::time::Duration::from_secs(provider_cfg.timeout_secs))
        .header("content-type", "application/json");
    for (k, v) in &headers {
        req = req.header(k, v);
    }
    let resp = match req.json(&payload).send().await {
        Ok(r) => r,
        Err(e) => {
            return (
                StatusCode::BAD_GATEWAY,
                Json(json!({"error": {"message": format!("network: {e}"), "type": "api_error"}})),
            )
                .into_response();
        }
    };
    let status = StatusCode::from_u16(resp.status().as_u16())
        .unwrap_or(StatusCode::INTERNAL_SERVER_ERROR);
    // A 200 with an unparseable body is an upstream fault, not an empty
    // embedding: shape it as a 502 like the network arm instead of handing
    // the client a silent `{}` success with zero tokens recorded.
    let body: Value = match resp.json().await {
        Ok(v) => v,
        Err(e) => {
            return (
                StatusCode::BAD_GATEWAY,
                Json(json!({"error": {"message": format!("bad upstream json: {e}"), "type": "api_error"}})),
            )
                .into_response();
        }
    };

    if status.is_success() {
        let tokens = body["usage"]["prompt_tokens"]
            .as_u64()
            .or_else(|| body["usage"]["total_tokens"].as_u64())
            .unwrap_or(0);
        router::record_embedding_usage(&app, &prov_name, tokens);
    }
    let mut resp = (status, Json(body)).into_response();
    if let Ok(v) = format!("{prov_name}/{model_id}").parse() {
        resp.headers_mut().insert("x-pxy-provider", v);
    }
    resp
}

/// Local estimate over EVERY block type (docs/04: counting only text broke
/// Claude Code auto-compaction).
async fn count_tokens(Json(payload): Json<Value>) -> Json<Value> {
    let tokens = estimate_tokens(&payload["messages"])
        + estimate_tokens(&payload["system"])
        + estimate_tokens(&payload["tools"]);
    Json(json!({"input_tokens": tokens.max(1)}))
}

/// codex identifies itself with an `originator` header — `codex_cli_rs` in the
/// TUI, `codex_exec` for `codex exec` — and it is the one client that reads a
/// different dialect off this path.
fn is_codex(headers: &HeaderMap) -> bool {
    headers
        .get("originator")
        .and_then(|v| v.to_str().ok())
        .is_some_and(|v| v.starts_with("codex"))
}

/// The instructions codex prepends for a pxy model. Its own models each ship
/// one; an entry without it is rejected outright, and codex's built-in
/// fallback is written for GPT-5 and its tool set.
const CODEX_BASE_INSTRUCTIONS: &str = "You are a coding agent running in the Codex CLI. \
You share a workspace with the user and help them carry out software engineering tasks. \
Use the tools you are given to read and edit files and to run commands, and prefer doing \
the work over describing it. Be concise and factual: state what you did and what you found, \
without padding.";

/// One entry of codex's model manifest. Codex validates the whole document, so
/// a missing or unknown field rejects EVERY model, not just this one — the
/// shape here was verified field by field against codex 0.150.1.
fn codex_model_entry(slug: &str, display: &str, context: u64) -> Value {
    json!({
        "slug": slug,
        "display_name": display,
        "description": format!("via pxy ({slug})"),
        "base_instructions": CODEX_BASE_INSTRUCTIONS,
        "default_reasoning_level": "medium",
        "supported_reasoning_levels": [
            {"effort": "low", "description": "Fast responses with lighter reasoning"},
            {"effort": "medium", "description": "Balances speed and reasoning depth"},
            {"effort": "high", "description": "Greater reasoning depth"},
        ],
        "shell_type": "unified_exec",
        "visibility": "list",
        "supported_in_api": true,
        "priority": 1,
        "additional_speed_tiers": [],
        "service_tiers": [],
        "availability_nux": null,
        "upgrade": null,
        "include_skills_usage_instructions": false,
        "include_plugin_usage_instructions": false,
        "include_apps_usage_instructions": false,
        "default_reasoning_summary": "none",
        "support_verbosity": true,
        "default_verbosity": "low",
        "apply_patch_tool_type": "freeform",
        "web_search_tool_type": "text_and_image",
        "truncation_policy": {"mode": "tokens", "limit": 10000},
        "supports_image_detail_original": false,
        "context_window": context,
        "max_context_window": context,
        "comp_hash": "3000",
        "effective_context_window_percent": 95,
        "experimental_supported_tools": [],
        "input_modalities": ["text"],
        // pxy runs web search itself for openai upstreams (translate/web_search)
        "supports_search_tool": true,
        "use_responses_lite": false,
        "node_repl_auto_review_required": false,
        "node_repl_disabled": true,
        "tool_mode": "default",
        "multi_agent_version": "v2",
    })
}

/// codex's model manifest. Without it codex can't see pxy's models in its
/// picker and warns that it is falling back to built-in metadata. The
/// "claude/" mirrors are left out: they exist only to get past Claude Code's
/// picker filter and would double every row here.
fn codex_models(app: &SharedApp) -> Json<Value> {
    let mut models: Vec<Value> = Vec::new();
    for (name, group) in app.catalog.groups() {
        let (ctx, _) = crate::catalog::chain_limits(&group.chain);
        models.push(codex_model_entry(name, &group.label, ctx));
    }
    for cand in app.catalog.models() {
        let id = cand.full_id();
        models.push(codex_model_entry(&id, &id, cand.model.context_length));
    }
    Json(json!({"models": models}))
}

/// Must answer fast: Claude Code's gateway discovery times out at 3s.
async fn models(State(app): State<SharedApp>, headers: HeaderMap) -> Json<Value> {
    if is_codex(&headers) {
        return codex_models(&app);
    }
    let created = Timestamp::now().as_second();
    let mut data: Vec<Value> = Vec::new();
    // Groups lead the list: they are the ids an agent is normally launched
    // with, and a picker shows them first. chain_limits() is a min() over the
    // members — a missing/zero context window disables opencode's
    // auto-compaction entirely and the session grows until history purge
    // destroys the context.
    for (name, group) in app.catalog.groups() {
        let (ctx, max_out) = crate::catalog::chain_limits(&group.chain);
        data.push(json!({
            "id": name,
            "object": "model",
            "created": created,
            "owned_by": "pxy",
            "display_name": group.label,
            "context_length": ctx,
            "max_output_tokens": max_out,
        }));
    }
    for cand in app.catalog.models() {
        data.push(json!({
            "id": cand.full_id(),
            "object": "model",
            "created": created,
            "owned_by": cand.provider,
            "display_name": cand.model.name.clone().unwrap_or_else(|| cand.full_id()),
            "context_length": cand.model.context_length,
            "max_output_tokens": cand.model.max_output_tokens,
        }));
    }
    // Claude Code's gateway model picker only lists ids CONTAINING
    // "claude"/"anthropic" (case-insensitive): mirror everything else under a
    // "claude/" prefix (catalog.resolve strips it) so /model works across
    // every provider. Ids that already carry the substring are left alone —
    // mirroring them too would list each one twice — but their 1M windows
    // still need the "[1m]" variant: CC's discovery schema strips
    // context_length, so the marker in the id is the only signal it honours
    // (first-party CC offers `sonnet` and `sonnet[1m]` the same way).
    // display_name carries the real id so the picker stays readable.
    let mut extra: Vec<Value> = Vec::new();
    for m in data.iter() {
        let id = m["id"].as_str().unwrap_or_default();
        let ctx = m["context_length"].as_u64().unwrap_or(0);
        let lower = id.to_ascii_lowercase();
        if lower.contains("claude") || lower.contains("anthropic") {
            let marker = crate::catalog::ctx_1m_marker(id, ctx);
            if !marker.is_empty() {
                let mut v = m.clone();
                v["id"] = json!(format!("{id}{marker}"));
                v["display_name"] = json!(format!("{id} (1M)"));
                extra.push(v);
            }
            continue;
        }
        let mut v = m.clone();
        v["id"] = json!(crate::catalog::claude_mirror_id(id, ctx));
        v["display_name"] = json!(id);
        extra.push(v);
    }
    data.extend(extra);
    Json(json!({"object": "list", "data": data}))
}

async fn healthz() -> Json<Value> {
    Json(json!({"status": "ok"}))
}

async fn not_found() -> Response {
    (
        StatusCode::NOT_FOUND,
        Json(json!({"error": {"message": "not found", "type": "invalid_request_error"}})),
    )
        .into_response()
}

/// `pxy status`: reads config + sqlite (works while the daemon runs — WAL).
/// With `remote`, also queries each provider's `balance_url` (concurrent).
/// With `json`, emits one machine-readable object instead of the table — the
/// desktop usage panel reads this. `only` restricts both the table and the
/// remote fetches to the named providers (remote checks cost real HTTP).
pub async fn print_status(cfg: &Config, remote: bool, json_out: bool, only: &[String]) -> Result<()> {
    use std::io::Write;
    let state = PxyState::open(&crate::config::data_dir().join("state.sqlite"))?;
    let now = Timestamp::now();
    // `--provider X` also matches the per-account synthetic keys
    // (`X#account`) that multi-account providers report under.
    let wanted = |name: &str| {
        only.is_empty()
            || only
                .iter()
                .any(|o| o == name || name.split('#').next() == Some(o.as_str()))
    };
    let stdout = std::io::stdout();
    let mut out = stdout.lock();
    let mut json_providers = serde_json::Map::new();
    if !json_out {
        let _ = writeln!(out, "{:<20} {:>10} {:>14} {:>12} {:>14} {:>16}", "provider", "day reqs", "day tokens", "month reqs", "month tokens", "total tokens");
    }
    for (name, p) in &cfg.providers {
        if !p.enabled || !wanted(name) {
            continue;
        }
        let default_limits = crate::config::Limits::default();
        let limits = p.limits.as_ref().unwrap_or(&default_limits);
        let Ok(w) = current_windows(limits, now) else { continue };
        // Multi-account providers track usage per account (`provider#account`
        // keys); the table shows the provider TOTAL summed over accounts.
        let state_keys: Vec<String> = if p.accounts.is_empty() {
            vec![name.clone()]
        } else {
            p.accounts.iter().map(|a| format!("{name}#{}", a.name)).collect()
        };
        let mut day = crate::state::UsageRow::default();
        let mut month = crate::state::UsageRow::default();
        let mut total = crate::state::UsageRow::default();
        for k in &state_keys {
            let d = state.usage(k, "day", w.day_start).unwrap_or_default();
            day.requests += d.requests;
            day.tokens += d.tokens;
            let m = state.usage(k, "month", w.month_start).unwrap_or_default();
            month.requests += m.requests;
            month.tokens += m.tokens;
            let t = state.usage_total(k).unwrap_or_default();
            total.requests += t.requests;
            total.tokens += t.tokens;
        }
        if json_out {
            json_providers.insert(
                name.clone(),
                json!({
                    "day": {"requests": day.requests, "tokens": day.tokens,
                            "requestLimit": limits.daily_requests, "tokenLimit": limits.daily_tokens},
                    "month": {"requests": month.requests, "tokens": month.tokens,
                              "requestLimit": limits.monthly_requests, "tokenLimit": limits.monthly_tokens},
                    "total": {"requests": total.requests, "tokens": total.tokens,
                              "tokenLimit": limits.total_tokens},
                }),
            );
            continue;
        }
        let fmt_limit = |used: u64, limit: Option<u64>| match limit {
            Some(l) => format!("{used}/{l}"),
            None => format!("{used}"),
        };
        if writeln!(
            out,
            "{:<20} {:>10} {:>14} {:>12} {:>14} {:>16}",
            name,
            fmt_limit(day.requests, limits.daily_requests),
            fmt_limit(day.tokens, limits.daily_tokens),
            fmt_limit(month.requests, limits.monthly_requests),
            fmt_limit(month.tokens, limits.monthly_tokens),
            fmt_limit(total.tokens, limits.total_tokens),
        )
        .is_err()
        {
            break;
        }
    }

    // Media + service pools use synthetic usage keys (provider#media,
    // search#name, fetch#name) with default UTC windows. Table only — the
    // JSON consumer (the usage panel) reads chat providers and model rows.
    let mut service_rows: Vec<(String, Option<u64>, Option<u64>)> = Vec::new();
    for (name, p) in &cfg.providers {
        if let Some(m) = p.media.as_ref().filter(|_| p.enabled && wanted(name)) {
            service_rows.push((crate::media::media_key(name), m.daily_requests, None));
        }
    }
    for (scope, svc) in [("search", &cfg.search), ("fetch", &cfg.fetch)] {
        for p in svc.providers.iter().filter(|p| p.enabled && wanted(&p.name)) {
            service_rows.push((
                format!("{scope}#{}", p.name),
                p.daily_requests,
                p.monthly_requests,
            ));
        }
    }
    if json_out {
        service_rows.clear();
    }
    let default_limits = crate::config::Limits::default();
    if let Ok(w) = current_windows(&default_limits, now) {
        for (key, daily, monthly) in service_rows {
            let day = state.usage(&key, "day", w.day_start).unwrap_or_default();
            let month = state.usage(&key, "month", w.month_start).unwrap_or_default();
            let total = state.usage_total(&key).unwrap_or_default();
            if day.requests == 0 && month.requests == 0 && total.requests == 0 {
                continue; // unused pools stay out of the table
            }
            let fmt_limit = |used: u64, limit: Option<u64>| match limit {
                Some(l) => format!("{used}/{l}"),
                None => format!("{used}"),
            };
            let _ = writeln!(
                out,
                "{:<20} {:>10} {:>14} {:>12} {:>14} {:>16}",
                key,
                fmt_limit(day.requests, daily),
                day.tokens,
                fmt_limit(month.requests, monthly),
                month.tokens,
                total.tokens,
            );
        }
    }

    let mut json_remote = serde_json::Map::new();
    if remote {
        let http = reqwest::Client::builder()
            .connect_timeout(std::time::Duration::from_secs(5))
            .timeout(std::time::Duration::from_secs(10))
            .build()?;
        let secrets = Secrets::new();
        let mut targets: Vec<(String, &crate::config::ProviderConfig, &String, Option<&crate::config::Account>)> =
            Vec::new();
        for (n, p) in cfg.providers.iter().filter(|(n, p)| p.enabled && wanted(n)) {
            let Some(url) = &p.balance_url else { continue };
            if p.accounts.is_empty() {
                targets.push((n.clone(), p, url, None));
            } else {
                // One balance fetch per ACCOUNT: the usage endpoint reads the
                // key's own windows, and the key is the account's.
                for a in &p.accounts {
                    targets.push((format!("{n}#{}", a.name), p, url, Some(a)));
                }
            }
        }
        // Providers that report their allowance in RESPONSE HEADERS instead of
        // at a URL (tokenharbor's rolling 7x24h free tier): the router's last
        // snapshot is the only readout there is. It costs no HTTP, so it can
        // never be fresher than the last call pxy routed — the age is part of
        // the line, and a snapshot from before an out-of-band session
        // (tokenharbor's own CLI, their web chat) will read low.
        let snapshots: Vec<(String, String, Value)> = cfg
            .providers
            .iter()
            .filter(|(n, p)| p.enabled && wanted(n))
            .filter_map(|(n, _)| {
                let raw = state.kv_get(&crate::router::free_quota_key(n)).ok().flatten()?;
                let snap: Value = serde_json::from_str(&raw).ok()?;
                Some((n.clone(), free_quota_summary(&snap, now), snap))
            })
            .collect();
        if targets.is_empty() && snapshots.is_empty() && !json_out {
            let _ = writeln!(out, "\nno remote quota source on any provider");
            return Ok(());
        }
        let fetches = targets.iter().map(|(name, p, url, acct)| {
            let http = &http;
            let secrets = &secrets;
            let state = &state;
            async move {
                let (line, body) =
                    fetch_balance(name, p, url, *acct, secrets, state, http).await;
                (name.clone(), line, body)
            }
        });
        let results = futures_util::future::join_all(fetches).await;
        if json_out {
            for (name, line, body) in results {
                json_remote.insert(
                    name,
                    json!({"summary": line, "data": body.unwrap_or(Value::Null)}),
                );
            }
            for (name, line, snap) in snapshots {
                // A live endpoint, where one exists, outranks a remembered
                // header.
                json_remote.entry(name).or_insert(json!({"summary": line, "data": snap}));
            }
        } else {
            let _ = writeln!(out, "\nremote balances:");
            let reported: Vec<String> = results.iter().map(|(n, _, _)| n.clone()).collect();
            for (name, line, _) in &results {
                let _ = writeln!(out, "  {name:<20} {line}");
            }
            for (name, line, _) in snapshots.iter().filter(|(n, _, _)| !reported.contains(n)) {
                let _ = writeln!(out, "  {name:<20} {line}");
            }
        }
    }

    // The pin and the bench: what the route is doing right now, and who is
    // sitting out. Both are read from the same sqlite the daemon writes, so
    // this works with the daemon down too (a stale in-memory-only rpm window
    // is the only thing the CLI can't see).
    let catalog = Catalog::from_config(cfg);
    let route_pin = state.kv_get(crate::router::ROUTE_PIN_KEY).ok().flatten().filter(|p| !p.is_empty());
    // Whether the pin actually steers routing right now — a pin gone stale
    // (model dropped from the catalog) is ignored by resolve_candidates, and
    // reporting it as simply "pinned" would have the panel lie about the walk.
    let route_pin_active = route_pin.as_deref().is_some_and(|p| {
        let resolved = catalog.resolve(cfg, p);
        !resolved.is_empty() && resolved.iter().all(|c| catalog.is_listed(&c.full_id()))
    });
    // The routable group ids, so the desktop panel can walk each chain without
    // parsing config.toml itself.
    let groups: Vec<Value> = catalog
        .groups()
        .map(|(name, g)| json!({"name": name, "label": g.label, "size": g.chain.len()}))
        .collect();
    let cooldowns: Vec<Value> = state
        .active_cooldowns()
        .into_iter()
        .map(|(key, cd)| {
            let left = cd.until.saturating_duration_since(std::time::Instant::now());
            json!({
                "key": key,
                "reason": cd.reason,
                "retryable": cd.retryable,
                "secondsLeft": left.as_secs(),
            })
        })
        .collect();

    if json_out {
        // Model rows travel whole regardless of --provider: that flag exists
        // to keep remote HTTP cheap, while the reader slices model rows by
        // AGENT — filtering them by provider would drop an agent's usage on
        // every provider outside the filter.
        let model_usage: Vec<Value> = state
            .model_usage_rows()?
            .into_iter()
            .map(|r| {
                json!({
                    "day": r.day, "agent": r.agent, "provider": r.provider, "model": r.model,
                    "requests": r.requests,
                    "inputTokens": r.input_tokens, "outputTokens": r.output_tokens,
                })
            })
            .collect();
        let mut root = serde_json::Map::new();
        root.insert("port".into(), json!(cfg.server.port));
        root.insert("routePin".into(), json!(route_pin));
        root.insert("routePinActive".into(), json!(route_pin_active));
        root.insert("groups".into(), Value::Array(groups));
        root.insert("cooldowns".into(), Value::Array(cooldowns));
        root.insert("providers".into(), Value::Object(json_providers));
        root.insert("modelUsage".into(), Value::Array(model_usage));
        if remote {
            root.insert("remote".into(), Value::Object(json_remote));
        }
        let _ = writeln!(out, "{}", Value::Object(root));
    } else {
        if !groups.is_empty() {
            let names: Vec<String> = groups
                .iter()
                .map(|g| format!("{} ({})", g["name"].as_str().unwrap_or("?"), g["size"]))
                .collect();
            let _ = writeln!(out, "\ngroups: {}", names.join(", "));
        }
        if let Some(pin) = &route_pin {
            if route_pin_active {
                let _ = writeln!(out, "route pinned to: {pin} (the group chain is the fallback)");
            } else {
                let _ = writeln!(out, "route pin '{pin}' is STALE (not in the catalog) — group chain priority in effect");
            }
        }
        if !cooldowns.is_empty() {
            let _ = writeln!(out, "\ncooling down:");
            for cd in &cooldowns {
                let _ = writeln!(
                    out,
                    "  {:<28} {}s left — {}{}",
                    cd["key"].as_str().unwrap_or("?"),
                    cd["secondsLeft"].as_u64().unwrap_or(0),
                    cd["reason"].as_str().unwrap_or(""),
                    if cd["retryable"].as_bool() == Some(false) { " (non-retryable)" } else { "" },
                );
            }
        }
    }
    Ok(())
}

/// A remembered allowance snapshot as one line. The age is not decoration:
/// this number is only as current as the last request pxy routed there.
fn free_quota_summary(snap: &Value, now: Timestamp) -> String {
    let pct = snap["usedPct"].as_f64().unwrap_or(0.0);
    let plan = snap["plan"].as_str().filter(|s| !s.is_empty()).unwrap_or("free");
    // Printed as sent: at a few hundred tokens into a 7-day allowance the
    // interesting digit is often the fractional one.
    let mut line = format!("{plan} tier {pct}% of the rolling allowance used");
    let resets = snap["resetsAt"].as_str().unwrap_or("");
    if !resets.is_empty() {
        // chars().take, not byte-slicing: a non-ASCII upstream stamp would
        // panic the CLI on a mid-character byte index.
        let stamp: String = resets.chars().take(16).collect();
        line.push_str(&format!(" · resets {}", stamp.replace('T', " ")));
    }
    let age = snap["observedAt"]
        .as_str()
        .and_then(|s| s.parse::<Timestamp>().ok())
        .map(|t| (now.as_second() - t.as_second()).max(0))
        .unwrap_or(0);
    let age = match age {
        s if s < 90 => format!("{s}s"),
        s if s < 5400 => format!("{}m", s / 60),
        s if s < 172_800 => format!("{}h", s / 3600),
        s => format!("{}d", s / 86_400),
    };
    line.push_str(&format!(" · seen {age} ago (from response headers)"));
    line
}

/// One provider's balance: a human summary line plus, when the endpoint
/// answered, its parsed body (for `--json` consumers, which normalize the
/// provider-specific shape themselves). Never errors — a broken endpoint
/// reports itself in place of a number.
/// (dollars left, dollars granted) from a new-api console body (aihubmix's
/// `GET /api/user/self`). Quota is an integer count of 1/500000 USD; `quota` is
/// what is LEFT and `used_quota` what has been spent, so the grant is their
/// sum — the same two numbers OpenRouter reports, in a different currency.
fn newapi_balance(body: &Value) -> Option<(f64, f64)> {
    const UNITS_PER_USD: f64 = 500_000.0;
    let left = body["data"]["quota"].as_f64()?;
    let used = body["data"]["used_quota"].as_f64()?;
    Some((left / UNITS_PER_USD, (left + used) / UNITS_PER_USD))
}

/// DeepSeek's `GET /user/balance` as a status line, or `None` when this is not
/// that shape. Two traps live here: the amounts are STRINGS (their precision
/// choice), and the number is not the fact that matters — `is_available` is.
/// An expired grant still counts inside `total_balance`, so an account can
/// report money while every call returns "Insufficient Balance". Lead with the
/// money, never omit the verdict, and treat a missing flag as unusable: we
/// cannot claim it works.
fn deepseek_balance(body: &Value) -> Option<String> {
    let infos = body["balance_infos"].as_array()?;
    let num = |v: &Value| v.as_str().and_then(|s| s.parse::<f64>().ok()).unwrap_or(0.0);
    let mut parts: Vec<String> = infos
        .iter()
        .map(|b| {
            let cur = b["currency"].as_str().unwrap_or("?");
            let granted = num(&b["granted_balance"]);
            let total = format!("{:.2} {cur}", num(&b["total_balance"]));
            if granted > 0.0 {
                format!("{total} ({granted:.2} granted)")
            } else {
                total
            }
        })
        .collect();
    if parts.is_empty() {
        parts.push("no balance reported".into());
    }
    Some(match body["is_available"].as_bool() {
        Some(true) => format!("{} left", parts.join(" + ")),
        _ => format!(
            "{} left · ⚠ NOT usable — calls return Insufficient Balance",
            parts.join(" + ")
        ),
    })
}

async fn fetch_balance(
    name: &str,
    p: &crate::config::ProviderConfig,
    url: &str,
    acct: Option<&crate::config::Account>,
    secrets: &Secrets,
    state: &PxyState,
    http: &reqwest::Client,
) -> (String, Option<Value>) {
    // Copilot's quota endpoint wants the long-lived GitHub token, not the
    // minted Copilot bearer the generic prepare() path would send.
    if p.kind == crate::config::ProviderKind::GithubCopilot {
        return match crate::providers::copilot::fetch_quota(name, p, secrets, http).await {
            Ok(v) => {
                let prem = &v["quota_snapshots"]["premium_interactions"];
                if !prem.is_object() {
                    let line = format!(
                        "no premium quota (plan: {})",
                        v["copilot_plan"].as_str().unwrap_or("?")
                    );
                    return (line, Some(v));
                }
                let line = format!(
                    "premium {:.1}/{} left ({:.0}%) · resets {}{}",
                    prem["quota_remaining"].as_f64().unwrap_or(0.0),
                    prem["entitlement"].as_u64().unwrap_or(0),
                    prem["percent_remaining"].as_f64().unwrap_or(0.0),
                    v["quota_reset_date"].as_str().unwrap_or("?"),
                    if prem["overage_permitted"].as_bool().unwrap_or(false) {
                        " · ⚠ overage billing ON"
                    } else {
                        ""
                    },
                );
                (line, Some(v))
            }
            Err(e) => (format!("{e:#}"), None),
        };
    }
    // A dedicated billing credential bypasses the chat auth entirely: new-api
    // consoles (aihubmix) answer 401 to the inference key no matter how it is
    // framed, and want their Manage Key raw — no `Bearer`.
    let mut req = http.get(url);
    if let Some(sref) = &p.balance_key {
        match secrets.resolve_key(sref) {
            Ok(k) => req = req.header("authorization", k),
            Err(e) => return (format!("balance_key error: {e:#}"), None),
        }
        for (k, v) in &p.headers {
            req = req.header(k, v);
        }
    } else {
        let prepared =
            match crate::providers::prepare(name, p, secrets, state, http, acct).await {
            Ok(pr) => pr,
            Err(e) => return (format!("credential error: {e:#}"), None),
        };
        for (k, v) in &prepared.headers {
            req = req.header(k, v);
        }
        // Providers whose chat route wants x-api-key (gorouter, tabitoken)
        // still gate billing behind a Bearer header; mirror the key into both.
        if !prepared.headers.iter().any(|(k, _)| k == "authorization") {
            if let Some((_, key)) = prepared.headers.iter().find(|(k, _)| k == "x-api-key") {
                req = req.header("authorization", format!("Bearer {key}"));
            }
        }
    }
    let resp = match req.send().await {
        Ok(r) => r,
        Err(e) => return (format!("network: {e}"), None),
    };
    let status = resp.status().as_u16();
    let body: Value = match resp.json().await {
        Ok(v) => v,
        Err(_) => return (format!("HTTP {status}: non-JSON body"), None),
    };
    if status >= 400 {
        return (format!("HTTP {status}"), None);
    }
    // OpenRouter: dollars, with the grant total alongside.
    if let (Some(credits), Some(usage)) =
        (body["data"]["total_credits"].as_f64(), body["data"]["total_usage"].as_f64())
    {
        let line = format!("${:.2} left of ${credits:.2}", credits - usage);
        return (line, Some(body));
    }
    if let Some((left, grant)) = newapi_balance(&body) {
        return (format!("${left:.2} left of ${grant:.2}"), Some(body));
    }
    if let Some(line) = deepseek_balance(&body) {
        return (line, Some(body));
    }
    // opencode Go: percent used per window (GET /zen/go/v1/usage).
    if body["usage"]["monthly"].is_object() {
        let win = |w: &str| {
            let pct = body["usage"][w]["percent"].as_u64().unwrap_or(0);
            let limited = body["usage"][w]["status"] == "rate-limited";
            format!("{pct}%{}", if limited { "!" } else { "" })
        };
        let resets = body["usage"]["monthly"]["resetsAt"].as_str().unwrap_or("?");
        let line = format!(
            "5h {} · wk {} · mo {} (mo resets {})",
            win("rolling"),
            win("weekly"),
            win("monthly"),
            &resets.chars().take(10).collect::<String>(),
        );
        return (line, Some(body));
    }
    // Command Code: dollars spent inside each plan window, plus what is left
    // of the credit pool (GET /alpha/billing/credits — the source its own CLI
    // reads for `/usage`). `cap` is the window's dollar allowance, so the
    // percentage has to be derived; `exceeded` marks a window already spent.
    if body["windowLimits"].is_object() {
        let mut parts: Vec<String> = Vec::new();
        for (key, label) in [("fiveHour", "5h"), ("weekly", "wk")] {
            let w = &body["windowLimits"][key];
            let (Some(used), Some(cap)) = (w["used"].as_f64(), w["cap"].as_f64()) else {
                continue;
            };
            let pct = if cap > 0.0 { used / cap * 100.0 } else { 0.0 };
            let flag = if w["exceeded"].as_bool().unwrap_or(false) { "!" } else { "" };
            parts.push(format!("{label} ${used:.2}/${cap:.0} ({pct:.0}%){flag}"));
        }
        let c = &body["credits"];
        let left = c["monthlyCredits"].as_f64().unwrap_or(0.0)
            + c["purchasedCredits"].as_f64().unwrap_or(0.0)
            + c["freeCredits"].as_f64().unwrap_or(0.0);
        parts.push(format!("${left:.2} credits left"));
        return (parts.join(" · "), Some(body));
    }
    // OpenAI/new-api dashboard billing: total_usage in cents.
    if let Some(cents) = body["total_usage"].as_f64() {
        return (format!("used ${:.2}", cents / 100.0), Some(body));
    }
    let line =
        format!("unrecognized shape: {}", &body.to_string().chars().take(120).collect::<String>());
    (line, Some(body))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Only codex gets the manifest dialect, and it announces itself with
    /// `originator` — `codex_cli_rs` in the TUI, `codex_exec` under
    /// `codex exec`. Every other client keeps the OpenAI list.
    /// aihubmix reports quota in units of 1/500000 USD, and reports what is
    /// LEFT rather than what was granted — reading `quota` as dollars would
    /// show a $10 balance as $10,000,000.
    #[test]
    fn newapi_quota_units_convert_to_dollars() {
        // $10 left, $2 already spent -> a $12 grant.
        let body = json!({"data": {"quota": 5_000_000, "used_quota": 1_000_000}});
        let (left, grant) = newapi_balance(&body).unwrap();
        assert!((left - 10.0).abs() < 1e-9, "left = {left}");
        assert!((grant - 12.0).abs() < 1e-9, "grant = {grant}");
        // A body without both halves is not a new-api balance: falling through
        // lets the other shapes (and the "unrecognized" line) have their turn.
        assert!(newapi_balance(&json!({"data": {"quota": 5_000_000}})).is_none());
        assert!(newapi_balance(&json!({"data": {"total_credits": 10.0}})).is_none());
    }

    /// DeepSeek reports money as strings and usability as a separate flag.
    /// Reading the number alone is the trap: an account whose grant expired
    /// still reports a total while refusing every call.
    #[test]
    fn deepseek_balance_reads_strings_and_leads_with_usability() {
        let ok = deepseek_balance(&json!({
            "is_available": true,
            "balance_infos": [{"currency": "CNY", "total_balance": "110.00",
                               "granted_balance": "10.00", "topped_up_balance": "100.00"}],
        }))
        .unwrap();
        assert_eq!(ok, "110.00 CNY (10.00 granted) left");
        // Money on the books, account still dead — the case that bit us.
        let dead = deepseek_balance(&json!({
            "is_available": false,
            "balance_infos": [{"currency": "USD", "total_balance": "5.00",
                               "granted_balance": "5.00", "topped_up_balance": "0"}],
        }))
        .unwrap();
        assert!(dead.contains("NOT usable"), "{dead}");
        // A missing flag must not read as usable either.
        let unknown = deepseek_balance(&json!({
            "balance_infos": [{"currency": "USD", "total_balance": "5.00"}],
        }))
        .unwrap();
        assert!(unknown.contains("NOT usable"), "{unknown}");
        // Purely topped-up account: no "(granted)" noise.
        let paid = deepseek_balance(&json!({
            "is_available": true,
            "balance_infos": [{"currency": "USD", "total_balance": "20.00",
                               "granted_balance": "0", "topped_up_balance": "20.00"}],
        }))
        .unwrap();
        assert_eq!(paid, "20.00 USD left");
        // Not this shape: fall through so the other sniffers get their turn.
        assert!(deepseek_balance(&json!({"data": {"total_credits": 10.0}})).is_none());
    }

    #[test]
    fn codex_is_recognised_by_originator() {
        assert!(is_codex(&headers(&[("originator", "codex_cli_rs")])));
        assert!(is_codex(&headers(&[("originator", "codex_exec")])));
        assert!(!is_codex(&headers(&[("originator", "opencode")])));
        assert!(!is_codex(&headers(&[])));
    }

    /// The manifest is validated as a whole by codex: one entry missing a
    /// required field rejects every model. `base_instructions` is the field it
    /// named explicitly, and the entry is useless in the picker without
    /// `visibility: "list"` and `supported_in_api`.
    #[test]
    fn codex_entry_carries_the_fields_codex_validates() {
        let e = codex_model_entry("prov/model", "prov/model", 128_000);
        assert_eq!(e["slug"], "prov/model");
        assert_eq!(e["visibility"], "list");
        assert_eq!(e["supported_in_api"], true);
        assert_eq!(e["context_window"], 128_000);
        assert!(e["base_instructions"].as_str().is_some_and(|s| !s.is_empty()));
        assert!(e["supported_reasoning_levels"].as_array().is_some_and(|a| !a.is_empty()));
    }

    /// Claude Code resolves the advertised id to a context window only via a
    /// "[1m]" in the id (its discovery schema strips context_length), so the
    /// subscription's 1M models — the ids the mirror filter skips — must get
    /// an explicit "[1m]" variant entry, while sub-1M claude ids get none.
    #[tokio::test]
    async fn claude_1m_ids_get_marker_variants_in_the_listing() {
        let cfg: Config = toml::from_str(
            r#"
            [server]
            [providers.claude]
            kind = "claude-oauth"
            format = "anthropic"
            models = [
              { id = "claude-opus-5", context_length = 1000000 },
              { id = "claude-haiku-4-5-20251001", context_length = 200000 },
            ]
            [providers.zai]
            base_url = "https://z.example/chat"
            models = [{ id = "glm-5.3-flash", context_length = 1048576 }]
            "#,
        )
        .unwrap();
        let dir = std::env::temp_dir().join(format!("pxy-server-it-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let app: SharedApp = Arc::new(App {
            catalog: Catalog::from_config(&cfg),
            secrets: Secrets::new(),
            state: PxyState::open(&dir.join("s.sqlite")).unwrap(),
            http: reqwest::Client::new(),
            cfg,
        });

        let out = models(State(app), HeaderMap::new()).await;
        let data = out.0["data"].as_array().unwrap();
        let find = |id: &str| data.iter().find(|m| m["id"] == json!(id));
        assert!(find("claude/claude-opus-5").is_some(), "plain id stays listed");
        let v = find("claude/claude-opus-5[1m]").expect("1M claude id needs a [1m] variant");
        assert_eq!(v["display_name"], "claude/claude-opus-5 (1M)");
        assert!(find("claude/claude-haiku-4-5-20251001").is_some());
        assert!(
            find("claude/claude-haiku-4-5-20251001[1m]").is_none(),
            "sub-1M claude id must not get a marker variant"
        );
        // Non-claude ids keep the claude/-prefixed mirror with the marker.
        assert!(find("claude/zai/glm-5.3-flash[1m]").is_some());
    }

    fn headers(pairs: &[(&str, &str)]) -> HeaderMap {
        let mut h = HeaderMap::new();
        for (k, v) in pairs {
            h.insert(
                axum::http::HeaderName::from_bytes(k.as_bytes()).unwrap(),
                v.parse().unwrap(),
            );
        }
        h
    }

    #[test]
    fn agent_from_key_suffix_in_either_auth_header() {
        let h = headers(&[("authorization", "Bearer pxy-local:claude")]);
        assert_eq!(client_agent(&h).as_deref(), Some("claude"));
        let h = headers(&[("x-api-key", "pxy-local:opencode")]);
        assert_eq!(client_agent(&h).as_deref(), Some("opencode"));
    }

    #[test]
    fn free_quota_snapshot_line_carries_age_and_reset() {
        let now: Timestamp = "2026-08-26T12:00:00Z".parse().unwrap();
        let line = free_quota_summary(
            &json!({
                "usedPct": 12.0, "plan": "free",
                "resetsAt": "2026-09-02T10:08:33.419881+00:00",
                "observedAt": "2026-08-26T09:30:00Z",
            }),
            now,
        );
        assert!(line.contains("free tier 12% of the rolling allowance used"), "{line}");
        assert!(line.contains("resets 2026-09-02 10:08"), "{line}");
        assert!(line.contains("seen 2h ago"), "{line}");
    }

    #[test]
    fn explicit_header_beats_key_suffix() {
        let h = headers(&[
            ("x-pxy-agent", "pi"),
            ("authorization", "Bearer pxy-local:claude"),
        ]);
        assert_eq!(client_agent(&h).as_deref(), Some("pi"));
    }

    #[test]
    fn plain_key_or_garbage_suffix_yields_no_agent() {
        let h = headers(&[("authorization", "Bearer pxy-local")]);
        assert_eq!(client_agent(&h), None);
        // A real token with colons (JSON-ish, spaces) must not fake an agent.
        let h = headers(&[("authorization", "Bearer sk-x:not an agent!")]);
        assert_eq!(client_agent(&h), None);
        assert_eq!(client_agent(&HeaderMap::new()), None);
    }
}
