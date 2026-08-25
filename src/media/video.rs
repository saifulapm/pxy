//! `/v1/videos/generations` — submit-then-poll job flow (agnes), blocking
//! until done like OmniRoute did: no job API for a single-user proxy.
//! `POST video_url {model, prompt, ...}` -> `{video_id}`; poll
//! `video_status_url` (`{id}` template) until status "completed" ->
//! `metadata.url`. Poll timing is server-controlled — OmniRoute let the
//! client pick and a request could pin the server for arbitrary time.

use axum::Json;
use axum::extract::State;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use jiff::Timestamp;
use serde_json::{Value, json};

use super::{Capability, error_response, images::preflight};
use crate::router::SharedApp;

const POLL_INTERVAL: std::time::Duration = std::time::Duration::from_secs(3);
const MAX_POLLS: u32 = 100; // ~5 minutes

pub async fn generations(State(app): State<SharedApp>, Json(payload): Json<Value>) -> Response {
    let requested = payload["model"].as_str().unwrap_or("").to_string();
    let Some(r) = super::resolve(&app.cfg, Capability::Video, &requested) else {
        return error_response(
            StatusCode::NOT_FOUND,
            format!("video model '{requested}' not found"),
        );
    };
    let Some(status_template) = r.media.video_status_url.clone() else {
        return error_response(
            StatusCode::NOT_IMPLEMENTED,
            format!("provider '{}' has no video_status_url", r.provider),
        );
    };
    if let Some(resp) = preflight(&app, &r) {
        return resp;
    }
    let headers = match super::auth_headers(&app, r.cfg) {
        Ok(h) => h,
        Err(e) => return error_response(StatusCode::BAD_GATEWAY, format!("{e:#}")),
    };

    // Submit. The client body passes through (image / num_frames / ratio…),
    // with the resolved model id.
    let mut body = payload.clone();
    body["model"] = json!(r.model);
    let mut req = app
        .http
        .post(&r.url)
        .timeout(std::time::Duration::from_secs(60))
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
    let submitted: Value = match resp.json().await {
        Ok(v) => v,
        Err(e) => return error_response(StatusCode::BAD_GATEWAY, format!("bad upstream json: {e}")),
    };
    let Some(job_id) = submitted["video_id"].as_str().or(submitted["id"].as_str()) else {
        return error_response(
            StatusCode::BAD_GATEWAY,
            format!("no video_id in submit response: {submitted}"),
        );
    };

    // Poll.
    let poll_url = status_template.replace("{id}", job_id);
    for _ in 0..MAX_POLLS {
        tokio::time::sleep(POLL_INTERVAL).await;
        let mut req = app.http.get(&poll_url).timeout(std::time::Duration::from_secs(30));
        for (k, v) in &headers {
            req = req.header(k, v);
        }
        let Ok(resp) = req.send().await else { continue };
        let Ok(body) = resp.json::<Value>().await else { continue };
        match body["status"].as_str().unwrap_or("") {
            "completed" => {
                app.state.clear_cooldown(&super::media_key(&r.provider), &r.model);
                let url = body["metadata"]["url"].as_str().or(body["url"].as_str());
                let Some(url) = url else {
                    return error_response(
                        StatusCode::BAD_GATEWAY,
                        format!("job completed but no url: {body}"),
                    );
                };
                let out = json!({
                    "created": Timestamp::now().as_second(),
                    "data": [{"url": url, "format": "mp4"}],
                });
                let mut resp = (StatusCode::OK, Json(out)).into_response();
                if let Ok(v) = format!("{}/{}", r.provider, r.model).parse() {
                    resp.headers_mut().insert("x-pxy-provider", v);
                }
                return resp;
            }
            "failed" => {
                super::cool_on_failure(&app, &r.provider, &r.model, 502);
                return error_response(
                    StatusCode::BAD_GATEWAY,
                    format!("video job failed: {body}"),
                );
            }
            _ => {}
        }
    }
    error_response(
        StatusCode::GATEWAY_TIMEOUT,
        format!("video job {job_id} still running after {} polls", MAX_POLLS),
    )
}
