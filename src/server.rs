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
    let _ = writeln!(out, "{:<20} {:>10} {:>14} {:>12} {:>14}", "provider", "day reqs", "day tokens", "month reqs", "month tokens");
    for (name, p) in &cfg.providers {
        if !p.enabled {
            continue;
        }
        let default_limits = crate::config::Limits {
            rpm: None,
            daily_requests: None,
            daily_tokens: None,
            monthly_requests: None,
            monthly_tokens: None,
            reset: "00:00".into(),
            reset_tz: "UTC".into(),
        };
        let limits = p.limits.as_ref().unwrap_or(&default_limits);
        let Ok(w) = current_windows(limits, now) else { continue };
        let day = state.usage(name, "day", w.day_start).unwrap_or_default();
        let month = state.usage(name, "month", w.month_start).unwrap_or_default();
        let fmt_limit = |used: u64, limit: Option<u64>| match limit {
            Some(l) => format!("{used}/{l}"),
            None => format!("{used}"),
        };
        if writeln!(
            out,
            "{:<20} {:>10} {:>14} {:>12} {:>14}",
            name,
            fmt_limit(day.requests, limits.daily_requests),
            fmt_limit(day.tokens, limits.daily_tokens),
            fmt_limit(month.requests, limits.monthly_requests),
            fmt_limit(month.tokens, limits.monthly_tokens),
        )
        .is_err()
        {
            break;
        }
    }
    Ok(())
}
