//! `/v1/images/generations` — OpenAI Images API in and out.
//! Dialects: openai (agnes included, plus per-provider request defaults) pass
//! through; cloudflare Workers AI returns `result.image` base64 or raw bytes.

use axum::Json;
use axum::extract::State;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use base64::Engine as _;
use jiff::Timestamp;
use serde_json::{Value, json};

use super::{Capability, error_response};
use crate::config::MediaKind;
use crate::router::{App, SharedApp};

pub async fn generations(State(app): State<SharedApp>, Json(payload): Json<Value>) -> Response {
    match run_generate(&app, payload["model"].as_str(), &payload).await {
        Ok((body, wire)) => {
            let mut resp = (StatusCode::OK, Json(body)).into_response();
            if let Ok(v) = wire.parse() {
                resp.headers_mut().insert("x-pxy-provider", v);
            }
            resp
        }
        Err(e) => error_response(StatusCode::BAD_GATEWAY, e),
    }
}

/// The provider walk behind `/v1/images/generations`, without the HTTP layer:
/// the `image_generation` server tool runs through it too, so both share one
/// walk, one failover and one `provider#media` quota. `model` is the
/// requested image model (`None` or empty walks the `[media] image` default
/// chain); returns the normalized OpenAI image body and the `provider/model`
/// that answered.
pub(crate) async fn run_generate(
    app: &App,
    model: Option<&str>,
    payload: &Value,
) -> Result<(Value, String), String> {
    let requested = model.unwrap_or("");
    let resp = super::run_chain(app, Capability::Image, requested, |r| {
        Box::pin(attempt(app, r, payload))
    })
    .await;
    let wire = resp
        .headers()
        .get("x-pxy-provider")
        .and_then(|v| v.to_str().ok())
        .unwrap_or_default()
        .to_string();
    let status = resp.status();
    let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX)
        .await
        .map_err(|e| format!("reading image response: {e}"))?;
    if !status.is_success() {
        let msg = serde_json::from_slice::<Value>(&bytes)
            .ok()
            .and_then(|v| v["error"]["message"].as_str().map(String::from))
            .unwrap_or_else(|| String::from_utf8_lossy(&bytes).into_owned());
        return Err(msg);
    }
    let body: Value =
        serde_json::from_slice(&bytes).map_err(|e| format!("bad image response: {e}"))?;
    Ok((body, wire))
}

async fn attempt(
    app: &crate::router::App,
    r: &super::Resolved<'_>,
    payload: &Value,
) -> super::Attempt {
    use super::Attempt;

    let body = match r.media.kind {
        MediaKind::Cloudflare => cloudflare_request(payload),
        MediaKind::Dashscope => super::dashscope::image_request(payload, &r.model),
        _ => openai_request(payload, r),
    };

    let headers = match super::auth_headers(app, r.cfg) {
        Ok(h) => h,
        Err(e) => {
            return Attempt::Retryable(error_response(StatusCode::BAD_GATEWAY, format!("{e:#}")));
        }
    };
    let mut req = app
        .http
        .post(&r.url)
        .timeout(std::time::Duration::from_secs(r.cfg.timeout_secs))
        .header("content-type", "application/json");
    for (k, v) in &headers {
        req = req.header(k, v);
    }
    let resp = match req.json(&body).send().await {
        Ok(resp) => resp,
        Err(e) => return super::network_failure(app, r, e),
    };

    let status = resp.status().as_u16();
    if status >= 400 {
        return super::failed_attempt(app, r, resp).await;
    }

    let content_type = resp
        .headers()
        .get("content-type")
        .and_then(|v| v.to_str().ok())
        .unwrap_or("")
        .to_string();
    let normalized = if content_type.starts_with("image/") {
        // Some Workers AI models (SDXL) answer with raw image bytes.
        match resp.bytes().await {
            Ok(bytes) => {
                let b64 = base64::engine::general_purpose::STANDARD.encode(&bytes);
                json!({"created": Timestamp::now().as_second(), "data": [{"b64_json": b64}]})
            }
            Err(e) => {
                return Attempt::Retryable(error_response(
                    StatusCode::BAD_GATEWAY,
                    format!("reading image: {e}"),
                ));
            }
        }
    } else {
        let body: Value = match resp.json().await {
            Ok(v) => v,
            Err(e) => {
                return Attempt::Retryable(error_response(
                    StatusCode::BAD_GATEWAY,
                    format!("bad upstream json: {e}"),
                ));
            }
        };
        match r.media.kind {
            MediaKind::Dashscope => match super::dashscope::image_response(&body) {
                Some(v) => v,
                None => {
                    return Attempt::Retryable(error_response(
                        StatusCode::BAD_GATEWAY,
                        format!("no image in upstream response: {body}"),
                    ));
                }
            },
            _ => normalize_json(&body),
        }
    };

    app.state.clear_cooldown(&super::media_key(&r.provider), &r.model);
    Attempt::Ok(tag(normalized, &r.provider, &r.model))
}

fn tag(body: Value, provider: &str, model: &str) -> Response {
    let mut resp = (StatusCode::OK, Json(body)).into_response();
    if let Ok(v) = format!("{provider}/{model}").parse() {
        resp.headers_mut().insert("x-pxy-provider", v);
    }
    resp
}

/// OpenAI-shaped upstreams: forward the client body with the resolved model
/// id, then fill per-provider required defaults (agnes: `size`).
fn openai_request(payload: &Value, r: &super::Resolved<'_>) -> Value {
    let mut body = payload.clone();
    body["model"] = json!(r.model);
    if let Some(obj) = body.as_object_mut() {
        for (k, v) in &r.media.image_defaults {
            obj.entry(k.clone()).or_insert_with(|| v.clone());
        }
    }
    body
}

fn cloudflare_request(payload: &Value) -> Value {
    let mut body = json!({"prompt": payload["prompt"]});
    // flux takes `steps`; pass a few useful knobs through when present.
    for k in ["steps", "seed", "width", "height", "negative_prompt", "guidance"] {
        if !payload[k].is_null() {
            body[k] = payload[k].clone();
        }
    }
    body
}

/// Absorb the response dialects into OpenAI `{created, data: [...]}`:
/// already-OpenAI bodies pass through; cloudflare's `result.image` base64 is
/// wrapped.
pub fn normalize_json(body: &Value) -> Value {
    if body["data"].is_array() {
        return body.clone();
    }
    let created = Timestamp::now().as_second();
    if let Some(img) = body["result"]["image"].as_str() {
        return json!({"created": created, "data": [{"b64_json": img}]});
    }
    if let Some(url) = body["result"]["url"].as_str().or(body["url"].as_str()) {
        return json!({"created": created, "data": [{"url": url}]});
    }
    body.clone()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalizes_cloudflare_and_passes_openai_through() {
        let cf = json!({"result": {"image": "aGk="}, "success": true});
        let n = normalize_json(&cf);
        assert_eq!(n["data"][0]["b64_json"], "aGk=");

        let openai = json!({"created": 1, "data": [{"url": "https://x/y.png"}]});
        assert_eq!(normalize_json(&openai), openai);
    }

    #[test]
    fn cloudflare_request_keeps_prompt_and_knobs_only() {
        let body = cloudflare_request(&json!({
            "model": "m", "prompt": "p", "steps": 4, "response_format": "b64_json"
        }));
        assert_eq!(body["prompt"], "p");
        assert_eq!(body["steps"], 4);
        assert!(body.get("response_format").is_none());
        assert!(body.get("model").is_none());
    }

    /// The walk behind `/v1/images/generations` is the one the
    /// `image_generation` server tool runs: given a provider, it returns the
    /// normalized image body and the `provider/model` that answered.
    #[tokio::test]
    async fn run_generate_walks_a_provider_and_returns_body_and_wire_id() {
        use axum::routing::post;
        let router = axum::Router::new().route(
            "/img",
            post(|| async {
                axum::Json(json!({"created": 1, "data": [{"url": "https://x/y.png"}]}))
            }),
        );
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move { axum::serve(listener, router).await.unwrap() });

        let cfg: crate::config::Config = toml::from_str(&format!(
            r#"
            [server]
            [providers.mock]
            [providers.mock.media]
            images_url = "http://{addr}/img"
            image_models = ["m"]
            "#
        ))
        .unwrap();
        let catalog = crate::catalog::Catalog::from_config(&cfg);
        let dir = std::env::temp_dir().join(format!("pxy-images-walk-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let app = crate::router::App {
            catalog,
            secrets: crate::secrets::Secrets::new(),
            state: crate::state::State::open(&dir.join("s.sqlite")).unwrap(),
            http: reqwest::Client::new(),
            cfg,
        };

        let (body, wire) = run_generate(&app, Some("mock/m"), &json!({"prompt": "p"}))
            .await
            .expect("a configured provider serves the call");
        assert_eq!(body["data"][0]["url"], "https://x/y.png");
        assert_eq!(wire, "mock/m");
    }
}
