//! Collect a complete SSE stream into the equivalent non-streaming JSON
//! response body. Used for `force_stream` models: the upstream only behaves
//! with `stream: true`, but the client asked for ordinary JSON.

use std::collections::BTreeMap;

use serde_json::{json, Map, Value};

use super::sse::SseEvent;

/// OpenAI chat-completion chunks -> a `chat.completion` response body.
pub fn openai(events: &[SseEvent]) -> Value {
    let mut id = Value::Null;
    let mut model = Value::Null;
    let mut created = Value::Null;
    let mut usage = Value::Null;

    #[derive(Default)]
    struct Choice {
        role: Option<String>,
        content: Option<String>,
        reasoning: Option<String>,
        // index -> (id, type, name, arguments)
        tools: BTreeMap<u64, (Option<String>, Option<String>, Option<String>, String)>,
        finish: Option<String>,
    }
    let mut choices: BTreeMap<u64, Choice> = BTreeMap::new();

    for ev in events {
        if ev.data.trim() == "[DONE]" {
            continue;
        }
        let Ok(v) = serde_json::from_str::<Value>(&ev.data) else { continue };
        if id.is_null() && v["id"].is_string() {
            id = v["id"].clone();
        }
        if model.is_null() && v["model"].is_string() {
            model = v["model"].clone();
        }
        if created.is_null() && v["created"].is_number() {
            created = v["created"].clone();
        }
        if v["usage"].is_object() {
            usage = v["usage"].clone();
        }
        let Some(chunk_choices) = v["choices"].as_array() else { continue };
        for c in chunk_choices {
            let idx = c["index"].as_u64().unwrap_or(0);
            let acc = choices.entry(idx).or_default();
            if let Some(f) = c["finish_reason"].as_str() {
                acc.finish = Some(f.to_string());
            }
            let delta = &c["delta"];
            if let Some(r) = delta["role"].as_str() {
                acc.role = Some(r.to_string());
            }
            if let Some(t) = delta["content"].as_str() {
                acc.content.get_or_insert_with(String::new).push_str(t);
            }
            // `reasoning` is the z-ai/GLM spelling of `reasoning_content`;
            // normalised here so the response translators see one field.
            for key in ["reasoning_content", "reasoning"] {
                if let Some(t) = delta[key].as_str() {
                    acc.reasoning.get_or_insert_with(String::new).push_str(t);
                }
            }
            for tc in delta["tool_calls"].as_array().into_iter().flatten() {
                let tidx = tc["index"].as_u64().unwrap_or(0);
                let slot = acc.tools.entry(tidx).or_default();
                if let Some(s) = tc["id"].as_str() {
                    slot.0 = Some(s.to_string());
                }
                if let Some(s) = tc["type"].as_str() {
                    slot.1 = Some(s.to_string());
                }
                if let Some(s) = tc["function"]["name"].as_str() {
                    slot.2 = Some(s.to_string());
                }
                if let Some(s) = tc["function"]["arguments"].as_str() {
                    slot.3.push_str(s);
                }
            }
        }
    }

    let choices: Vec<Value> = choices
        .into_iter()
        .map(|(idx, acc)| {
            let mut message = Map::new();
            message.insert("role".into(), json!(acc.role.as_deref().unwrap_or("assistant")));
            message.insert("content".into(), acc.content.map(Value::from).unwrap_or(Value::Null));
            if let Some(r) = acc.reasoning {
                message.insert("reasoning_content".into(), json!(r));
            }
            if !acc.tools.is_empty() {
                let tools: Vec<Value> = acc
                    .tools
                    .into_values()
                    .map(|(tid, ttype, name, args)| {
                        json!({
                            "id": tid,
                            "type": ttype.as_deref().unwrap_or("function"),
                            "function": {"name": name, "arguments": args},
                        })
                    })
                    .collect();
                message.insert("tool_calls".into(), json!(tools));
            }
            json!({
                "index": idx,
                "message": message,
                "finish_reason": acc.finish,
            })
        })
        .collect();

    json!({
        "id": id,
        "object": "chat.completion",
        "created": created,
        "model": model,
        "choices": choices,
        "usage": usage,
    })
}

/// Anthropic message events -> a `message` response body.
pub fn anthropic(events: &[SseEvent]) -> Value {
    let mut message = json!({
        "type": "message",
        "role": "assistant",
        "content": [],
        "stop_reason": null,
        "stop_sequence": null,
        "usage": {"input_tokens": 0, "output_tokens": 0},
    });
    // index -> accumulated partial_json for tool_use inputs
    let mut partials: BTreeMap<usize, String> = BTreeMap::new();

    for ev in events {
        let Ok(v) = serde_json::from_str::<Value>(&ev.data) else { continue };
        match v["type"].as_str() {
            Some("message_start") => {
                if v["message"].is_object() {
                    message = v["message"].clone();
                    if !message["content"].is_array() {
                        message["content"] = json!([]);
                    }
                }
            }
            Some("content_block_start") => {
                let idx = v["index"].as_u64().unwrap_or(0) as usize;
                if let Some(content) = message["content"].as_array_mut() {
                    while content.len() <= idx {
                        content.push(Value::Null);
                    }
                    content[idx] = v["content_block"].clone();
                }
            }
            Some("content_block_delta") => {
                let idx = v["index"].as_u64().unwrap_or(0) as usize;
                let delta = &v["delta"];
                // A scalar block (pathological content_block_start) would
                // panic the mutable field indexes below — skip it whole.
                let Some(block) = message["content"].get_mut(idx).filter(|b| b.is_object()) else {
                    continue;
                };
                match delta["type"].as_str() {
                    Some("text_delta") => append_str(block, "text", delta["text"].as_str()),
                    Some("thinking_delta") => {
                        append_str(block, "thinking", delta["thinking"].as_str())
                    }
                    Some("signature_delta") => {
                        if let Some(s) = delta["signature"].as_str() {
                            block["signature"] = json!(s);
                        }
                    }
                    Some("input_json_delta") => {
                        if let Some(s) = delta["partial_json"].as_str() {
                            partials.entry(idx).or_default().push_str(s);
                        }
                    }
                    _ => {}
                }
            }
            Some("content_block_stop") => {
                let idx = v["index"].as_u64().unwrap_or(0) as usize;
                if let Some(partial) = partials.remove(&idx) {
                    if let Some(block) = message["content"].get_mut(idx).filter(|b| b.is_object()) {
                        block["input"] = serde_json::from_str(&partial).unwrap_or(json!({}));
                    }
                }
            }
            Some("message_delta") => {
                if let Some(delta) = v["delta"].as_object() {
                    for (k, val) in delta {
                        // Anthropic's usage rides message_delta's TOP level
                        // (handled below), never inside delta — copying a
                        // scalar delta `usage` here would poison the object.
                        if k == "usage" {
                            continue;
                        }
                        message[k] = val.clone();
                    }
                }
                if let Some(o) = v["usage"]["output_tokens"].as_u64() {
                    // `usage` must be an object (or absent) for the mut index;
                    // a scalar would have arrived via the delta loop above.
                    if message.get("usage").map_or(true, |u| u.is_object() || u.is_null()) {
                        message["usage"]["output_tokens"] = json!(o);
                    }
                }
            }
            _ => {}
        }
    }
    if let Some(content) = message["content"].as_array_mut() {
        // Null placeholders and scalar garbage (pathological content_block
        // values) must not reach the client: every real block is an object.
        content.retain(|b| b.is_object());
    }
    message
}

fn append_str(block: &mut Value, field: &str, add: Option<&str>) {
    let Some(add) = add else { return };
    let prior = block[field].as_str().unwrap_or("");
    block[field] = json!(format!("{prior}{add}"));
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::translate::sse::SseParser;

    fn events(raw: &str) -> Vec<SseEvent> {
        SseParser::new().feed(raw.as_bytes())
    }

    #[test]
    fn openai_text_and_usage() {
        let evs = events(concat!(
            "data: {\"id\":\"c1\",\"model\":\"m\",\"created\":5,\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"content\":\"Hel\"}}]}\n\n",
            "data: {\"choices\":[{\"index\":0,\"delta\":{\"content\":\"lo\"},\"finish_reason\":\"stop\"}]}\n\n",
            "data: {\"choices\":[],\"usage\":{\"prompt_tokens\":3,\"completion_tokens\":2}}\n\n",
            "data: [DONE]\n\n",
        ));
        let v = openai(&evs);
        assert_eq!(v["id"], "c1");
        assert_eq!(v["object"], "chat.completion");
        assert_eq!(v["choices"][0]["message"]["content"], "Hello");
        assert_eq!(v["choices"][0]["finish_reason"], "stop");
        assert_eq!(v["usage"]["prompt_tokens"], 3);
    }

    #[test]
    fn openai_tool_call_fragments() {
        let evs = events(concat!(
            "data: {\"id\":\"c1\",\"choices\":[{\"index\":0,\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"t1\",\"type\":\"function\",\"function\":{\"name\":\"get\",\"arguments\":\"{\\\"a\\\":\"}}]}}]}\n\n",
            "data: {\"choices\":[{\"index\":0,\"delta\":{\"tool_calls\":[{\"index\":0,\"function\":{\"arguments\":\"1}\"}}]},\"finish_reason\":\"tool_calls\"}]}\n\n",
            "data: [DONE]\n\n",
        ));
        let v = openai(&evs);
        let tc = &v["choices"][0]["message"]["tool_calls"][0];
        assert_eq!(tc["id"], "t1");
        assert_eq!(tc["function"]["name"], "get");
        assert_eq!(tc["function"]["arguments"], "{\"a\":1}");
        assert_eq!(v["choices"][0]["message"]["content"], Value::Null);
        assert_eq!(v["choices"][0]["finish_reason"], "tool_calls");
    }

    #[test]
    fn anthropic_text_tool_and_usage() {
        let evs = events(concat!(
            "event: message_start\ndata: {\"type\":\"message_start\",\"message\":{\"id\":\"m1\",\"type\":\"message\",\"role\":\"assistant\",\"model\":\"x\",\"content\":[],\"usage\":{\"input_tokens\":7,\"output_tokens\":1}}}\n\n",
            "event: content_block_start\ndata: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"text\",\"text\":\"\"}}\n\n",
            "event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"Hi\"}}\n\n",
            "event: content_block_stop\ndata: {\"type\":\"content_block_stop\",\"index\":0}\n\n",
            "event: content_block_start\ndata: {\"type\":\"content_block_start\",\"index\":1,\"content_block\":{\"type\":\"tool_use\",\"id\":\"t1\",\"name\":\"get\",\"input\":{}}}\n\n",
            "event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":1,\"delta\":{\"type\":\"input_json_delta\",\"partial_json\":\"{\\\"a\\\":1\"}}\n\n",
            "event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":1,\"delta\":{\"type\":\"input_json_delta\",\"partial_json\":\"}\"}}\n\n",
            "event: content_block_stop\ndata: {\"type\":\"content_block_stop\",\"index\":1}\n\n",
            "event: message_delta\ndata: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"tool_use\"},\"usage\":{\"output_tokens\":9}}\n\n",
            "event: message_stop\ndata: {\"type\":\"message_stop\"}\n\n",
        ));
        let v = anthropic(&evs);
        assert_eq!(v["id"], "m1");
        assert_eq!(v["content"][0]["text"], "Hi");
        assert_eq!(v["content"][1]["type"], "tool_use");
        assert_eq!(v["content"][1]["input"]["a"], 1);
        assert_eq!(v["stop_reason"], "tool_use");
        assert_eq!(v["usage"]["input_tokens"], 7);
        assert_eq!(v["usage"]["output_tokens"], 9);
    }

    /// Scalar blocks and a scalar `usage` (pathological upstream events) must
    /// degrade to skips, not panic the mutable field indexes.
    #[test]
    fn pathological_blocks_never_panic() {
        let evs = events(concat!(
            "data: {\"type\":\"message_start\",\"message\":{\"role\":\"assistant\",\"content\":[]}}\n\n",
            "data: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":5}\n\n",
            "data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"signature_delta\",\"signature\":\"sig\"}}\n\n",
            "data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"x\"}}\n\n",
            "data: {\"type\":\"content_block_stop\",\"index\":0}\n\n",
            "data: {\"type\":\"message_delta\",\"delta\":{\"usage\":5},\"usage\":{\"output_tokens\":9}}\n\n",
        ));
        let v = anthropic(&evs);
        // The scalar block was dropped by the object retain, and the scalar
        // delta `usage` must not have blocked the real usage update.
        assert!(v["content"].as_array().unwrap().is_empty());
        assert_eq!(v["usage"]["output_tokens"], 9);
    }
}
