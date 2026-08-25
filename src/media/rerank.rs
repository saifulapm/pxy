//! `/v1/rerank` — Cohere shape in and out (`{model, query, documents,
//! top_n?, return_documents?}` -> `{results: [{index, relevance_score}]}`).
//! Jina is Cohere-compatible (passthrough); voyage takes `top_k` and answers
//! with `data[]`, so both directions are mapped.

use axum::Json;
use axum::extract::State;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use serde_json::{Value, json};

use super::{Capability, error_response, images::preflight};
use crate::config::MediaKind;
use crate::router::SharedApp;

pub async fn rerank(State(app): State<SharedApp>, Json(payload): Json<Value>) -> Response {
    let requested = payload["model"].as_str().unwrap_or("").to_string();
    let Some(r) = super::resolve(&app.cfg, Capability::Rerank, &requested) else {
        return error_response(
            StatusCode::NOT_FOUND,
            format!("rerank model '{requested}' not found"),
        );
    };
    if let Some(resp) = preflight(&app, &r) {
        return resp;
    }

    let body = match r.media.kind {
        MediaKind::Voyage => voyage_request(&payload, &r.model),
        _ => {
            let mut b = payload.clone();
            b["model"] = json!(r.model);
            b
        }
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
    let upstream: Value = match resp.json().await {
        Ok(v) => v,
        Err(e) => return error_response(StatusCode::BAD_GATEWAY, format!("bad upstream json: {e}")),
    };

    let out = match r.media.kind {
        MediaKind::Voyage => voyage_response(&upstream, &payload),
        _ => upstream,
    };
    let tokens = out["usage"]["total_tokens"].as_u64().unwrap_or(0);
    super::add_units(&app, &super::media_key(&r.provider), tokens);
    app.state.clear_cooldown(&super::media_key(&r.provider), &r.model);

    let mut resp = (StatusCode::OK, Json(out)).into_response();
    if let Ok(v) = format!("{}/{}", r.provider, r.model).parse() {
        resp.headers_mut().insert("x-pxy-provider", v);
    }
    resp
}

fn voyage_request(payload: &Value, model: &str) -> Value {
    let mut body = json!({
        "model": model,
        "query": payload["query"],
        "documents": payload["documents"],
    });
    if let Some(n) = payload["top_n"].as_u64().or(payload["top_k"].as_u64()) {
        body["top_k"] = json!(n);
    }
    body
}

/// voyage `data[] {index, relevance_score}` -> cohere `results[]`. Voyage
/// never echoes documents; when the caller asked for them, synthesize from
/// the request's own list (indices are caller positions).
fn voyage_response(upstream: &Value, request: &Value) -> Value {
    let want_docs = request["return_documents"].as_bool().unwrap_or(false);
    let docs = request["documents"].as_array();
    let results: Vec<Value> = upstream["data"]
        .as_array()
        .map(|items| {
            items
                .iter()
                .map(|item| {
                    let mut out = json!({
                        "index": item["index"],
                        "relevance_score": item["relevance_score"],
                    });
                    if want_docs {
                        if let Some(doc) = item["index"]
                            .as_u64()
                            .and_then(|i| docs.and_then(|d| d.get(i as usize)))
                        {
                            let text = doc.as_str().map(Value::from).unwrap_or(doc["text"].clone());
                            out["document"] = json!({"text": text});
                        }
                    }
                    out
                })
                .collect()
        })
        .unwrap_or_default();
    json!({
        "model": upstream["model"],
        "results": results,
        "usage": upstream["usage"],
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn voyage_round_trip() {
        let request = json!({
            "model": "x", "query": "q",
            "documents": ["rust doc", "paris doc"],
            "top_n": 2, "return_documents": true
        });
        let body = voyage_request(&request, "rerank-2.5-lite");
        assert_eq!(body["top_k"], 2);
        assert!(body.get("top_n").is_none());
        assert!(body.get("return_documents").is_none());

        let upstream = json!({
            "object": "list", "model": "rerank-2.5-lite",
            "data": [
                {"relevance_score": 0.9, "index": 1},
                {"relevance_score": 0.2, "index": 0}
            ],
            "usage": {"total_tokens": 12}
        });
        let out = voyage_response(&upstream, &request);
        assert_eq!(out["results"][0]["index"], 1);
        assert_eq!(out["results"][0]["relevance_score"], 0.9);
        assert_eq!(out["results"][0]["document"]["text"], "paris doc");
        assert_eq!(out["usage"]["total_tokens"], 12);
    }
}
