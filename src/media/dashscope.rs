//! DashScope (Alibaba Model Studio) native dialect — one endpoint
//! (`.../services/aigc/multimodal-generation/generation`) serves image gen,
//! TTS and ASR; only the payload differs. All shapes verified live
//! 2026-08-25 (qwen-image-3.0, z-image-turbo, qwen3-tts-flash,
//! qwen3-asr-flash incl. base64 data-URI audio).

use jiff::Timestamp;
use serde_json::{Value, json};

pub fn image_request(payload: &Value, model: &str) -> Value {
    let mut body = json!({
        "model": model,
        "input": {"messages": [
            {"role": "user", "content": [{"text": payload["prompt"]}]}
        ]},
    });
    let mut params = serde_json::Map::new();
    if let Some(n) = payload["n"].as_u64() {
        params.insert("n".into(), json!(n));
    }
    if let Some(size) = payload["size"].as_str() {
        // OpenAI "1024x1024" -> dashscope "1024*1024"
        params.insert("size".into(), json!(size.replace('x', "*")));
    }
    if !params.is_empty() {
        body["parameters"] = Value::Object(params);
    }
    body
}

/// `output.choices[].message.content[].image` URLs -> OpenAI images shape.
pub fn image_response(body: &Value) -> Option<Value> {
    let urls: Vec<Value> = body["output"]["choices"]
        .as_array()?
        .iter()
        .flat_map(|c| c["message"]["content"].as_array().into_iter().flatten())
        .filter_map(|part| part["image"].as_str())
        .map(|url| json!({"url": url}))
        .collect();
    if urls.is_empty() {
        return None;
    }
    Some(json!({"created": Timestamp::now().as_second(), "data": urls}))
}

pub fn speech_request(model: &str, input: &str, voice: &str) -> Value {
    json!({"model": model, "input": {"text": input, "voice": voice}})
}

/// TTS answers with a signed OSS URL (`output.audio.url`), not bytes.
pub fn speech_audio_url(body: &Value) -> Option<String> {
    body["output"]["audio"]["url"].as_str().map(String::from)
}

pub fn transcription_request(model: &str, mime: &str, b64: &str) -> Value {
    json!({
        "model": model,
        "input": {"messages": [
            {"role": "user", "content": [{"audio": format!("data:{mime};base64,{b64}")}]}
        ]},
    })
}

pub fn transcription_text(body: &Value) -> Option<String> {
    body["output"]["choices"][0]["message"]["content"]
        .as_array()?
        .iter()
        .find_map(|part| part["text"].as_str())
        .map(String::from)
}

/// Mime for the ASR data URI. The upload's content type is often the useless
/// octet-stream default (the CLI sends none), so the extension wins.
pub fn audio_mime(filename: &str, fallback: &str) -> String {
    let ext = filename.rsplit('.').next().unwrap_or("").to_ascii_lowercase();
    match ext.as_str() {
        "mp3" | "mpga" => "audio/mpeg".into(),
        "wav" => "audio/wav".into(),
        "ogg" | "opus" => "audio/ogg".into(),
        "m4a" | "mp4" => "audio/mp4".into(),
        "flac" => "audio/flac".into(),
        "webm" => "audio/webm".into(),
        "aac" => "audio/aac".into(),
        _ => fallback.into(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn image_round_trip() {
        let req = image_request(&json!({"prompt": "a cube", "n": 1, "size": "1024x1024"}), "qwen-image-3.0");
        assert_eq!(req["input"]["messages"][0]["content"][0]["text"], "a cube");
        assert_eq!(req["parameters"]["size"], "1024*1024");

        let resp = json!({"output": {"choices": [{"finish_reason": "stop", "message": {
            "content": [{"image": "https://oss/x.png", "type": "image"}], "role": "assistant"
        }}]}, "usage": {}});
        let out = image_response(&resp).unwrap();
        assert_eq!(out["data"][0]["url"], "https://oss/x.png");
        assert!(image_response(&json!({"output": {}})).is_none());
    }

    #[test]
    fn transcription_shapes() {
        let req = transcription_request("qwen3-asr-flash", "audio/mpeg", "aGk=");
        assert_eq!(
            req["input"]["messages"][0]["content"][0]["audio"],
            "data:audio/mpeg;base64,aGk="
        );
        let resp = json!({"output": {"choices": [{"message": {
            "annotations": [{"type": "audio_info"}],
            "content": [{"text": "Hello world."}], "role": "assistant"
        }}]}});
        assert_eq!(transcription_text(&resp).unwrap(), "Hello world.");
    }

    #[test]
    fn mime_from_extension_beats_fallback() {
        assert_eq!(audio_mime("a.mp3", "application/octet-stream"), "audio/mpeg");
        assert_eq!(audio_mime("weird.bin", "audio/x-custom"), "audio/x-custom");
    }
}
