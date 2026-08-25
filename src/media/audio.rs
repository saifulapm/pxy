//! `/v1/audio/transcriptions` (multipart in, JSON out) and
//! `/v1/audio/speech` (JSON in, binary audio streamed out).
//! Dialects: openai passes through; cloudflare wraps base64 JSON; elevenlabs
//! uses its native field names and voice-id paths.

use axum::Json;
use axum::extract::{Multipart, State};
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use base64::Engine as _;
use serde_json::{Value, json};

use super::{Capability, error_response, images::preflight};
use crate::config::MediaKind;
use crate::router::SharedApp;

struct Upload {
    bytes: Vec<u8>,
    filename: String,
    content_type: String,
    /// Non-file fields, forwarded to OpenAI-shaped upstreams.
    fields: Vec<(String, String)>,
}

async fn read_multipart(mut multipart: Multipart) -> anyhow::Result<Upload> {
    let mut upload = Upload {
        bytes: Vec::new(),
        filename: "audio".into(),
        content_type: "application/octet-stream".into(),
        fields: Vec::new(),
    };
    while let Some(field) = multipart.next_field().await? {
        let name = field.name().unwrap_or("").to_string();
        if name == "file" {
            if let Some(f) = field.file_name() {
                upload.filename = f.to_string();
            }
            if let Some(ct) = field.content_type() {
                upload.content_type = ct.to_string();
            }
            upload.bytes = field.bytes().await?.to_vec();
        } else {
            upload.fields.push((name, field.text().await?));
        }
    }
    anyhow::ensure!(!upload.bytes.is_empty(), "multipart field 'file' is required");
    Ok(upload)
}

pub async fn transcriptions(State(app): State<SharedApp>, multipart: Multipart) -> Response {
    let upload = match read_multipart(multipart).await {
        Ok(u) => u,
        Err(e) => return error_response(StatusCode::BAD_REQUEST, format!("{e:#}")),
    };
    let requested = upload
        .fields
        .iter()
        .find(|(k, _)| k == "model")
        .map(|(_, v)| v.clone())
        .unwrap_or_default();
    let Some(r) = super::resolve(&app.cfg, Capability::Transcription, &requested) else {
        return error_response(
            StatusCode::NOT_FOUND,
            format!("transcription model '{requested}' not found"),
        );
    };
    if let Some(resp) = preflight(&app, &r) {
        return resp;
    }
    let headers = match super::auth_headers(&app, r.cfg) {
        Ok(h) => h,
        Err(e) => return error_response(StatusCode::BAD_GATEWAY, format!("{e:#}")),
    };

    let mut req = app
        .http
        .post(&r.url)
        .timeout(std::time::Duration::from_secs(r.cfg.timeout_secs));
    for (k, v) in &headers {
        req = req.header(k, v);
    }

    let resp = match r.media.kind {
        MediaKind::Cloudflare => {
            // Workers AI whisper: JSON {audio: base64} -> {result: {text}}.
            let b64 = base64::engine::general_purpose::STANDARD.encode(&upload.bytes);
            req.header("content-type", "application/json")
                .json(&json!({"audio": b64}))
                .send()
                .await
        }
        MediaKind::Dashscope => {
            let b64 = base64::engine::general_purpose::STANDARD.encode(&upload.bytes);
            let mime = super::dashscope::audio_mime(&upload.filename, &upload.content_type);
            req.header("content-type", "application/json")
                .json(&super::dashscope::transcription_request(&r.model, &mime, &b64))
                .send()
                .await
        }
        _ => {
            // OpenAI multipart (groq, mistral) — elevenlabs differs only in
            // the model field name.
            let model_field = match r.media.kind {
                MediaKind::Elevenlabs => "model_id",
                _ => "model",
            };
            let part = reqwest::multipart::Part::bytes(upload.bytes.clone())
                .file_name(upload.filename.clone())
                .mime_str(&upload.content_type)
                .unwrap_or_else(|_| {
                    reqwest::multipart::Part::bytes(upload.bytes.clone())
                        .file_name(upload.filename.clone())
                });
            let mut form = reqwest::multipart::Form::new()
                .part("file", part)
                .text(model_field.to_string(), r.model.clone());
            for (k, v) in &upload.fields {
                if k != "model" {
                    form = form.text(k.clone(), v.clone());
                }
            }
            req.multipart(form).send().await
        }
    };

    let resp = match resp {
        Ok(resp) => resp,
        Err(e) => return error_response(StatusCode::BAD_GATEWAY, format!("network: {e}")),
    };
    let status = resp.status().as_u16();
    if status >= 400 {
        super::cool_on_failure(&app, &r.provider, &r.model, status);
    }
    let content_type = resp
        .headers()
        .get("content-type")
        .and_then(|v| v.to_str().ok())
        .unwrap_or("application/json")
        .to_string();
    let bytes = match resp.bytes().await {
        Ok(b) => b,
        Err(e) => return error_response(StatusCode::BAD_GATEWAY, format!("reading body: {e}")),
    };

    // Native dialects wrap the transcript; unwrap to the OpenAI `{text}`
    // shape. A 200 without a transcript string is an upstream defect, not a
    // success.
    let wrapped_text = |v: &Value| match r.media.kind {
        MediaKind::Cloudflare => v["result"]["text"].as_str().map(String::from),
        MediaKind::Dashscope => super::dashscope::transcription_text(v),
        _ => None,
    };
    let unwraps = matches!(r.media.kind, MediaKind::Cloudflare | MediaKind::Dashscope);
    let (body, content_type) = if status < 400 && unwraps {
        let text = serde_json::from_slice::<Value>(&bytes).ok().as_ref().and_then(wrapped_text);
        let Some(text) = text else {
            return error_response(
                StatusCode::BAD_GATEWAY,
                format!("no transcript in upstream response: {}", String::from_utf8_lossy(&bytes)),
            );
        };
        (
            serde_json::to_vec(&json!({"text": text})).unwrap_or_default(),
            "application/json".to_string(),
        )
    } else {
        (bytes.to_vec(), content_type)
    };

    if status < 400 {
        // Audio seconds aren't reported by most providers; count any token
        // usage the body happens to carry (mistral does).
        let tokens = serde_json::from_slice::<Value>(&body)
            .ok()
            .and_then(|v| v["usage"]["total_tokens"].as_u64())
            .unwrap_or(0);
        super::add_units(&app, &super::media_key(&r.provider), tokens);
        app.state.clear_cooldown(&super::media_key(&r.provider), &r.model);
    }
    let mut resp = (
        StatusCode::from_u16(status).unwrap_or(StatusCode::BAD_GATEWAY),
        [("content-type", content_type)],
        body,
    )
        .into_response();
    if let Ok(v) = format!("{}/{}", r.provider, r.model).parse() {
        resp.headers_mut().insert("x-pxy-provider", v);
    }
    resp
}

pub async fn speech(State(app): State<SharedApp>, Json(payload): Json<Value>) -> Response {
    let requested = payload["model"].as_str().unwrap_or("").to_string();
    let Some(r) = super::resolve(&app.cfg, Capability::Speech, &requested) else {
        return error_response(
            StatusCode::NOT_FOUND,
            format!("speech model '{requested}' not found"),
        );
    };
    if let Some(resp) = preflight(&app, &r) {
        return resp;
    }
    let input = payload["input"].as_str().unwrap_or("").to_string();
    if input.is_empty() {
        return error_response(StatusCode::BAD_REQUEST, "'input' is required");
    }

    let (url, body) = match r.media.kind {
        MediaKind::Cloudflare => (r.url.clone(), json!({"text": input})),
        MediaKind::Elevenlabs => {
            let voice = resolve_voice(&payload, r.media);
            let url = format!("{}?output_format=mp3_44100_128", r.url.replace("{voice}", &voice));
            (url, json!({"text": input, "model_id": r.model}))
        }
        MediaKind::Dashscope => {
            let voice = resolve_voice(&payload, r.media);
            (r.url.clone(), super::dashscope::speech_request(&r.model, &input, &voice))
        }
        _ => {
            let mut body = payload.clone();
            body["model"] = json!(r.model);
            (r.url.clone(), body)
        }
    };

    let headers = match super::auth_headers(&app, r.cfg) {
        Ok(h) => h,
        Err(e) => return error_response(StatusCode::BAD_GATEWAY, format!("{e:#}")),
    };
    let mut req = app
        .http
        .post(&url)
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

    // TTS quotas are per character (elevenlabs' 10k/month); count them.
    super::add_units(&app, &super::media_key(&r.provider), input.chars().count() as u64);
    app.state.clear_cooldown(&super::media_key(&r.provider), &r.model);

    let content_type = resp
        .headers()
        .get("content-type")
        .and_then(|v| v.to_str().ok())
        .unwrap_or("audio/mpeg")
        .to_string();
    if r.media.kind == MediaKind::Dashscope {
        // DashScope answers with a signed OSS URL; fetch and relay the bytes.
        let v: Value = resp.json().await.unwrap_or_default();
        let Some(url) = super::dashscope::speech_audio_url(&v) else {
            return error_response(
                StatusCode::BAD_GATEWAY,
                format!("no audio url in upstream response: {v}"),
            );
        };
        return match app.http.get(&url).timeout(std::time::Duration::from_secs(60)).send().await {
            Ok(audio) if audio.status().is_success() => {
                let ct = audio
                    .headers()
                    .get("content-type")
                    .and_then(|h| h.to_str().ok())
                    .filter(|c| c.starts_with("audio/"))
                    .unwrap_or("audio/wav")
                    .to_string();
                Response::builder()
                    .status(StatusCode::OK)
                    .header("content-type", ct)
                    .body(axum::body::Body::from_stream(audio.bytes_stream()))
                    .unwrap_or_else(|_| StatusCode::INTERNAL_SERVER_ERROR.into_response())
            }
            Ok(audio) => {
                error_response(StatusCode::BAD_GATEWAY, format!("fetching audio: http {}", audio.status()))
            }
            Err(e) => error_response(StatusCode::BAD_GATEWAY, format!("fetching audio: {e}")),
        };
    }
    if content_type.starts_with("application/json") {
        // JSON-wrapped audio (melotts): {result: {audio: base64}} -> bytes.
        let v: Value = resp.json().await.unwrap_or_default();
        let Some(b64) = v["result"]["audio"].as_str().or(v["audio"].as_str()) else {
            return error_response(StatusCode::BAD_GATEWAY, "no audio in upstream response");
        };
        match base64::engine::general_purpose::STANDARD.decode(b64) {
            Ok(bytes) => {
                ([("content-type", "audio/mpeg")], bytes).into_response()
            }
            Err(e) => error_response(StatusCode::BAD_GATEWAY, format!("bad audio base64: {e}")),
        }
    } else {
        let mut out = Response::builder()
            .status(StatusCode::OK)
            .header("content-type", content_type);
        if let Ok(v) = format!("{}/{}", r.provider, r.model).parse::<axum::http::HeaderValue>() {
            out = out.header("x-pxy-provider", v);
        }
        out.body(axum::body::Body::from_stream(resp.bytes_stream()))
            .unwrap_or_else(|_| StatusCode::INTERNAL_SERVER_ERROR.into_response())
    }
}

/// Voice for elevenlabs: request `voice` mapped through `media.voices`
/// (OpenAI names like "nova" land on premade voices — library voices 402 on
/// the free plan); an unmapped value passes through as a raw voice id;
/// otherwise the configured "default".
fn resolve_voice(payload: &Value, media: &crate::config::MediaConfig) -> String {
    match payload["voice"].as_str() {
        Some(v) => media.voices.get(v).cloned().unwrap_or_else(|| v.to_string()),
        None => media.voices.get("default").cloned().unwrap_or_default(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn voice_mapping() {
        let media: crate::config::MediaConfig = toml::from_str(
            r#"
            kind = "elevenlabs"
            [voices]
            default = "DEF"
            nova = "NOVA_ID"
            "#,
        )
        .unwrap();
        assert_eq!(resolve_voice(&json!({"voice": "nova"}), &media), "NOVA_ID");
        assert_eq!(resolve_voice(&json!({"voice": "RAW123"}), &media), "RAW123");
        assert_eq!(resolve_voice(&json!({}), &media), "DEF");
    }
}
