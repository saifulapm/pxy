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
    let model = payload["model"].as_str().unwrap_or("auto").to_string();
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
                        _ => {
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
    let model = hdr("ai-language-model-id").unwrap_or("auto").to_string();
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
    let auto_chain = app.catalog.resolve(&app.cfg, "auto");
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
    if !auto_chain.is_empty() {
        let ctx = auto_chain.iter().map(|c| c.model.context_length).min().unwrap_or(0);
        let max_out = auto_chain.iter().map(|c| c.model.max_output_tokens).min().unwrap_or(0);
        push("auto".into(), ctx, max_out, true);
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
    let body: Value = resp.json().await.unwrap_or_else(|_| json!({}));

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

/// Must answer fast: Claude Code's gateway discovery times out at 3s.
async fn models(State(app): State<SharedApp>) -> Json<Value> {
    let created = Timestamp::now().as_second();
    let mut data: Vec<Value> = Vec::new();
    let auto_chain = app.catalog.resolve(&app.cfg, "auto");
    if !auto_chain.is_empty() {
        // min() over chain members: any member may serve a request, so the
        // advertised window must be one they all satisfy. A missing/zero
        // context window disables opencode's auto-compaction entirely and
        // the session grows until history purge destroys the context.
        let ctx = auto_chain.iter().map(|c| c.model.context_length).min().unwrap_or(0);
        let max_out = auto_chain.iter().map(|c| c.model.max_output_tokens).min().unwrap_or(0);
        data.push(json!({
            "id": "auto",
            "object": "model",
            "created": created,
            "owned_by": "pxy",
            "display_name": "auto",
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
    // "claude/" prefix (catalog.resolve strips it) so /model works across every
    // provider. Ids that already carry the substring are left alone — mirroring
    // them too would list each one twice.
    // display_name carries the real id so the picker stays readable.
    let mirrors: Vec<Value> = data
        .iter()
        .filter_map(|m| {
            let id = m["id"].as_str()?;
            let lower = id.to_ascii_lowercase();
            if lower.contains("claude") || lower.contains("anthropic") {
                return None;
            }
            let mut v = m.clone();
            v["id"] = json!(format!("claude/{id}"));
            v["display_name"] = json!(id);
            Some(v)
        })
        .collect();
    data.extend(mirrors);
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
    let wanted = |name: &str| only.is_empty() || only.iter().any(|o| o == name);
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
        let day = state.usage(name, "day", w.day_start).unwrap_or_default();
        let month = state.usage(name, "month", w.month_start).unwrap_or_default();
        let total = state.usage_total(name).unwrap_or_default();
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
        let targets: Vec<(&String, &crate::config::ProviderConfig, &String)> = cfg
            .providers
            .iter()
            .filter(|(n, p)| p.enabled && wanted(n))
            .filter_map(|(n, p)| p.balance_url.as_ref().map(|u| (n, p, u)))
            .collect();
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
        let fetches = targets.iter().map(|(name, p, url)| {
            let http = &http;
            let secrets = &secrets;
            let state = &state;
            async move {
                let (line, body) = fetch_balance(name, p, url, secrets, state, http).await;
                (name.to_string(), line, body)
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

    // The pin and the bench: what the auto route is doing right now, and who
    // is sitting out. Both are read from the same sqlite the daemon writes,
    // so this works with the daemon down too (a stale in-memory-only rpm
    // window is the only thing the CLI can't see).
    let route_pin = state.kv_get(crate::router::ROUTE_PIN_KEY).ok().flatten().filter(|p| !p.is_empty());
    // Whether the pin actually steers routing right now — a pin gone stale
    // (model dropped from the catalog) is ignored by resolve_candidates, and
    // reporting it as simply "pinned" would have the panel lie about the walk.
    let route_pin_active = route_pin.as_deref().is_some_and(|p| {
        let catalog = Catalog::from_config(cfg);
        let resolved = catalog.resolve(cfg, p);
        !resolved.is_empty() && resolved.iter().all(|c| catalog.is_listed(&c.full_id()))
    });
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
        root.insert("cooldowns".into(), Value::Array(cooldowns));
        root.insert("providers".into(), Value::Object(json_providers));
        root.insert("modelUsage".into(), Value::Array(model_usage));
        if remote {
            root.insert("remote".into(), Value::Object(json_remote));
        }
        let _ = writeln!(out, "{}", Value::Object(root));
    } else {
        if let Some(pin) = &route_pin {
            if route_pin_active {
                let _ = writeln!(out, "\nauto route pinned to: {pin} (chain is the fallback)");
            } else {
                let _ = writeln!(out, "\nauto route pin '{pin}' is STALE (not in the catalog) — chain priority in effect");
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
        let stamp = &resets[..resets.len().min(16)];
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
async fn fetch_balance(
    name: &str,
    p: &crate::config::ProviderConfig,
    url: &str,
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
    let prepared = match crate::providers::prepare(name, p, secrets, state, http).await {
        Ok(pr) => pr,
        Err(e) => return (format!("credential error: {e:#}"), None),
    };
    let mut req = http.get(url);
    for (k, v) in &prepared.headers {
        req = req.header(k, v);
    }
    // Providers whose chat route wants x-api-key (gorouter, tabitoken) still
    // gate billing behind a Bearer header; mirror the key into both.
    if !prepared.headers.iter().any(|(k, _)| k == "authorization") {
        if let Some((_, key)) = prepared.headers.iter().find(|(k, _)| k == "x-api-key") {
            req = req.header("authorization", format!("Bearer {key}"));
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
            &resets[..resets.len().min(10)],
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
