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
        .route("/v1/messages", post(messages))
        .route("/v1/messages/count_tokens", post(count_tokens))
        .route("/v1/models", get(models))
        .route("/healthz", get(healthz))
        .fallback(not_found)
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

    let prepared = match crate::providers::prepare(
        &prov_name,
        provider_cfg,
        &app.secrets,
        &app.state,
        &app.http,
    )
    .await
    {
        Ok(p) => p,
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
    for (k, v) in &prepared.headers {
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
    if !app.catalog.resolve(&app.cfg, "auto").is_empty() {
        data.push(json!({
            "id": "auto",
            "object": "model",
            "created": created,
            "owned_by": "pxy",
            "display_name": "auto",
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
pub async fn print_status(cfg: &Config) -> Result<()> {
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
    Ok(())
}
