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
use crate::router::SharedApp;

pub async fn generations(State(app): State<SharedApp>, Json(payload): Json<Value>) -> Response {
    let requested = payload["model"].as_str().unwrap_or("").to_string();
    let Some(r) = super::resolve(&app.cfg, Capability::Image, &requested) else {
        return error_response(
            StatusCode::NOT_FOUND,
            format!("image model '{requested}' not found"),
        );
    };
    if let Some(resp) = preflight(&app, &r) {
        return resp;
    }

    let body = match r.media.kind {
        MediaKind::Cloudflare => cloudflare_request(&payload),
        _ => openai_request(&payload, &r),
    };

    let headers = match super::auth_headers(&app, r.cfg) {
        Ok(h) => h,
        Err(e) => return error_response(StatusCode::BAD_GATEWAY, format!("{e:#}")),
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
        Err(e) => return error_response(StatusCode::BAD_GATEWAY, format!("network: {e}")),
    };

    let status = resp.status().as_u16();
    if status >= 400 {
        super::cool_on_failure(&app, &r.provider, &r.model, status);
        return super::passthrough_error(resp).await;
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
            Err(e) => return error_response(StatusCode::BAD_GATEWAY, format!("reading image: {e}")),
        }
    } else {
        let body: Value = match resp.json().await {
            Ok(v) => v,
            Err(e) => return error_response(StatusCode::BAD_GATEWAY, format!("bad upstream json: {e}")),
        };
        normalize_json(&body)
    };

    app.state.clear_cooldown(&super::media_key(&r.provider), &r.model);
    tag(normalized, &r.provider, &r.model)
}

/// Cooldown + media-cap gate shared by all media handlers. When it passes,
/// the request is COUNTED immediately (see `record`): the upstream call is
/// about to happen, and a billed-but-lost response must not escape the cap.
pub fn preflight(app: &crate::router::App, r: &super::Resolved<'_>) -> Option<Response> {
    let key = super::media_key(&r.provider);
    if let Some(cd) = app.state.cooldown(&key, &r.model) {
        return Some(error_response(
            StatusCode::TOO_MANY_REQUESTS,
            format!("{key} cooling down ({})", cd.reason),
        ));
    }
    if super::cap_exhausted(app, &r.provider, r.media) {
        return Some(error_response(
            StatusCode::TOO_MANY_REQUESTS,
            format!("{key} daily media cap reached"),
        ));
    }
    super::record(app, &key);
    None
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
}
