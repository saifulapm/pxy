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
    }
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
    // Claude Code's gateway model picker only lists ids beginning
    // "claude"/"anthropic": mirror everything else under a "claude/" prefix
    // (catalog.resolve strips it) so /model works across every provider.
    // display_name carries the real id so the picker stays readable.
    let mirrors: Vec<Value> = data
        .iter()
        .filter_map(|m| {
            let id = m["id"].as_str()?;
            if id.starts_with("claude") || id.starts_with("anthropic") {
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
pub async fn print_status(cfg: &Config, remote: bool) -> Result<()> {
    use std::io::Write;
    let state = PxyState::open(&crate::config::data_dir().join("state.sqlite"))?;
    let now = Timestamp::now();
    let stdout = std::io::stdout();
    let mut out = stdout.lock();
    let _ = writeln!(out, "{:<20} {:>10} {:>14} {:>12} {:>14} {:>16}", "provider", "day reqs", "day tokens", "month reqs", "month tokens", "total tokens");
    for (name, p) in &cfg.providers {
        if !p.enabled {
            continue;
        }
        let default_limits = crate::config::Limits::default();
        let limits = p.limits.as_ref().unwrap_or(&default_limits);
        let Ok(w) = current_windows(limits, now) else { continue };
        let day = state.usage(name, "day", w.day_start).unwrap_or_default();
        let month = state.usage(name, "month", w.month_start).unwrap_or_default();
        let fmt_limit = |used: u64, limit: Option<u64>| match limit {
            Some(l) => format!("{used}/{l}"),
            None => format!("{used}"),
        };
        let total = state.usage_total(name).unwrap_or_default();
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
    // search#name, fetch#name) with default UTC windows.
    let mut service_rows: Vec<(String, Option<u64>, Option<u64>)> = Vec::new();
    for (name, p) in &cfg.providers {
        if let Some(m) = p.media.as_ref().filter(|_| p.enabled) {
            service_rows.push((crate::media::media_key(name), m.daily_requests, None));
        }
    }
    for (scope, svc) in [("search", &cfg.search), ("fetch", &cfg.fetch)] {
        for p in svc.providers.iter().filter(|p| p.enabled) {
            service_rows.push((
                format!("{scope}#{}", p.name),
                p.daily_requests,
                p.monthly_requests,
            ));
        }
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

    if remote {
        let http = reqwest::Client::builder()
            .connect_timeout(std::time::Duration::from_secs(5))
            .timeout(std::time::Duration::from_secs(10))
            .build()?;
        let secrets = Secrets::new();
        let targets: Vec<(&String, &crate::config::ProviderConfig, &String)> = cfg
            .providers
            .iter()
            .filter(|(_, p)| p.enabled)
            .filter_map(|(n, p)| p.balance_url.as_ref().map(|u| (n, p, u)))
            .collect();
        if targets.is_empty() {
            let _ = writeln!(out, "\nno balance_url configured on any provider");
            return Ok(());
        }
        let fetches = targets.iter().map(|(name, p, url)| {
            let http = &http;
            let secrets = &secrets;
            let state = &state;
            async move {
                let line = fetch_balance(name, p, url, secrets, state, http).await;
                (name.to_string(), line)
            }
        });
        let results = futures_util::future::join_all(fetches).await;
        let _ = writeln!(out, "\nremote balances:");
        for (name, line) in results {
            let _ = writeln!(out, "  {name:<20} {line}");
        }
    }
    Ok(())
}

/// One provider's balance line. Never errors — a broken endpoint reports
/// itself in place of a number.
async fn fetch_balance(
    name: &str,
    p: &crate::config::ProviderConfig,
    url: &str,
    secrets: &Secrets,
    state: &PxyState,
    http: &reqwest::Client,
) -> String {
    // Copilot's quota endpoint wants the long-lived GitHub token, not the
    // minted Copilot bearer the generic prepare() path would send.
    if p.kind == crate::config::ProviderKind::GithubCopilot {
        return match crate::providers::copilot::fetch_quota(name, p, secrets, http).await {
            Ok(v) => {
                let prem = &v["quota_snapshots"]["premium_interactions"];
                if !prem.is_object() {
                    return format!(
                        "no premium quota (plan: {})",
                        v["copilot_plan"].as_str().unwrap_or("?")
                    );
                }
                format!(
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
                )
            }
            Err(e) => format!("{e:#}"),
        };
    }
    let prepared = match crate::providers::prepare(name, p, secrets, state, http).await {
        Ok(pr) => pr,
        Err(e) => return format!("credential error: {e:#}"),
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
        Err(e) => return format!("network: {e}"),
    };
    let status = resp.status().as_u16();
    let body: Value = match resp.json().await {
        Ok(v) => v,
        Err(_) => return format!("HTTP {status}: non-JSON body"),
    };
    if status >= 400 {
        return format!("HTTP {status}");
    }
    // OpenRouter: dollars, with the grant total alongside.
    if let (Some(credits), Some(usage)) =
        (body["data"]["total_credits"].as_f64(), body["data"]["total_usage"].as_f64())
    {
        return format!("${:.2} left of ${credits:.2}", credits - usage);
    }
    // opencode Go: percent used per window (GET /zen/go/v1/usage).
    if body["usage"]["monthly"].is_object() {
        let win = |w: &str| {
            let pct = body["usage"][w]["percent"].as_u64().unwrap_or(0);
            let limited = body["usage"][w]["status"] == "rate-limited";
            format!("{pct}%{}", if limited { "!" } else { "" })
        };
        let resets = body["usage"]["monthly"]["resetsAt"].as_str().unwrap_or("?");
        return format!(
            "5h {} · wk {} · mo {} (mo resets {})",
            win("rolling"),
            win("weekly"),
            win("monthly"),
            &resets[..resets.len().min(10)],
        );
    }
    // OpenAI/new-api dashboard billing: total_usage in cents.
    if let Some(cents) = body["total_usage"].as_f64() {
        return format!("used ${:.2}", cents / 100.0);
    }
    format!("unrecognized shape: {}", &body.to_string().chars().take(120).collect::<String>())
}
