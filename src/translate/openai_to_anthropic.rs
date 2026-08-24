//! OpenAI-client path: OpenAI chat completions request -> Anthropic Messages,
//! and Anthropic response -> OpenAI response (stream + non-stream).

use serde_json::{json, Map, Value};

use super::sse::format_data;
use super::TokenUsage;

// ---------------------------------------------------------------------------
// Request: openai -> anthropic
// ---------------------------------------------------------------------------

pub fn request(openai: &Value, default_max_tokens: u64) -> Value {
    let mut system_parts: Vec<String> = Vec::new();
    let mut messages: Vec<Value> = Vec::new();
    let mut pending_tool_results: Vec<Value> = Vec::new();

    let flush_tools = |pending: &mut Vec<Value>, messages: &mut Vec<Value>| {
        if !pending.is_empty() {
            messages.push(json!({"role": "user", "content": std::mem::take(pending)}));
        }
    };

    for msg in openai["messages"].as_array().unwrap_or(&Vec::new()) {
        let role = msg["role"].as_str().unwrap_or("user");
        match role {
            "system" | "developer" => {
                system_parts.push(flatten_content(&msg["content"]));
            }
            "tool" => {
                pending_tool_results.push(json!({
                    "type": "tool_result",
                    "tool_use_id": msg["tool_call_id"],
                    "content": flatten_content(&msg["content"]),
                }));
            }
            "assistant" => {
                flush_tools(&mut pending_tool_results, &mut messages);
                let mut blocks: Vec<Value> = Vec::new();
                let text = flatten_content(&msg["content"]);
                if !text.is_empty() {
                    blocks.push(json!({"type": "text", "text": text}));
                }
                if let Some(calls) = msg["tool_calls"].as_array() {
                    for call in calls {
                        let args = call["function"]["arguments"].as_str().unwrap_or("{}");
                        let input: Value =
                            serde_json::from_str(args).unwrap_or_else(|_| json!({}));
                        blocks.push(json!({
                            "type": "tool_use",
                            "id": call["id"],
                            "name": call["function"]["name"],
                            "input": input,
                        }));
                    }
                }
                if !blocks.is_empty() {
                    messages.push(json!({"role": "assistant", "content": blocks}));
                }
            }
            _ => {
                // user
                flush_tools(&mut pending_tool_results, &mut messages);
                let content = user_content(&msg["content"]);
                if !content.is_null() {
                    messages.push(json!({"role": "user", "content": content}));
                }
            }
        }
    }
    flush_tools(&mut pending_tool_results, &mut messages);

    let mut out = Map::new();
    out.insert("messages".into(), Value::Array(messages));
    if !system_parts.is_empty() {
        out.insert("system".into(), json!(system_parts.join("\n")));
    }

    let mut max_tokens = openai["max_tokens"]
        .as_u64()
        .or_else(|| openai["max_completion_tokens"].as_u64())
        .unwrap_or(default_max_tokens);

    for key in ["temperature", "top_p", "stream"] {
        if !openai[key].is_null() {
            out.insert(key.into(), openai[key].clone());
        }
    }
    match &openai["stop"] {
        Value::String(s) => {
            out.insert("stop_sequences".into(), json!([s]));
        }
        Value::Array(a) if !a.is_empty() => {
            out.insert("stop_sequences".into(), json!(a));
        }
        _ => {}
    }

    let mut drop_tools = false;
    match &openai["tool_choice"] {
        Value::String(s) => match s.as_str() {
            "required" => {
                out.insert("tool_choice".into(), json!({"type": "any"}));
            }
            "none" => drop_tools = true,
            _ => {
                out.insert("tool_choice".into(), json!({"type": "auto"}));
            }
        },
        Value::Object(o) => {
            if let Some(name) = o["function"]["name"].as_str() {
                out.insert("tool_choice".into(), json!({"type": "tool", "name": name}));
            }
        }
        _ => {}
    }
    if !drop_tools {
        if let Some(tools) = openai["tools"].as_array() {
            let mapped: Vec<Value> = tools
                .iter()
                .filter_map(|t| {
                    let f = &t["function"];
                    f["name"].as_str().map(|name| {
                        json!({
                            "name": name,
                            "description": f["description"].as_str().unwrap_or(""),
                            "input_schema": if f["parameters"].is_object() {
                                f["parameters"].clone()
                            } else {
                                json!({"type": "object", "properties": {}})
                            },
                        })
                    })
                })
                .collect();
            if !mapped.is_empty() {
                out.insert("tools".into(), Value::Array(mapped));
            }
        }
    } else {
        out.remove("tool_choice");
    }

    // reasoning_effort -> thinking budget (reverse of the OmniRoute buckets)
    if let Some(effort) = openai["reasoning_effort"].as_str() {
        let budget: u64 = match effort {
            "low" | "minimal" => 1024,
            "medium" => 8192,
            _ => 16384,
        };
        // Anthropic requires max_tokens > budget_tokens
        if max_tokens <= budget {
            max_tokens = budget + 4096;
        }
        out.insert(
            "thinking".into(),
            json!({"type": "enabled", "budget_tokens": budget}),
        );
    }
    out.insert("max_tokens".into(), json!(max_tokens));

    Value::Object(out)
}

fn flatten_content(content: &Value) -> String {
    match content {
        Value::String(s) => s.clone(),
        Value::Array(parts) => parts
            .iter()
            .filter_map(|p| p["text"].as_str())
            .collect::<Vec<_>>()
            .join("\n"),
        _ => String::new(),
    }
}

fn user_content(content: &Value) -> Value {
    match content {
        Value::String(s) => {
            if s.is_empty() {
                Value::Null
            } else {
                json!(s)
            }
        }
        Value::Array(parts) => {
            let blocks: Vec<Value> = parts
                .iter()
                .filter_map(|p| match p["type"].as_str() {
                    Some("text") => Some(json!({"type": "text", "text": p["text"]})),
                    Some("image_url") => {
                        let url = p["image_url"]["url"].as_str().unwrap_or("");
                        if let Some(rest) = url.strip_prefix("data:") {
                            let (media, data) = rest.split_once(";base64,")?;
                            Some(json!({"type": "image", "source": {
                                "type": "base64", "media_type": media, "data": data}}))
                        } else {
                            Some(json!({"type": "image", "source": {"type": "url", "url": url}}))
                        }
                    }
                    _ => None,
                })
                .collect();
            if blocks.is_empty() {
                Value::Null
            } else {
                Value::Array(blocks)
            }
        }
        _ => Value::Null,
    }
}

// ---------------------------------------------------------------------------
// Non-streaming response: anthropic -> openai
// ---------------------------------------------------------------------------

pub fn response(anthropic: &Value, model: &str) -> Value {
    let mut text = String::new();
    let mut reasoning = String::new();
    let mut tool_calls: Vec<Value> = Vec::new();
    for block in anthropic["content"].as_array().unwrap_or(&Vec::new()) {
        match block["type"].as_str() {
            Some("text") => text.push_str(block["text"].as_str().unwrap_or("")),
            Some("thinking") => reasoning.push_str(block["thinking"].as_str().unwrap_or("")),
            Some("tool_use") => tool_calls.push(json!({
                "id": block["id"],
                "type": "function",
                "function": {
                    "name": block["name"],
                    "arguments": serde_json::to_string(&block["input"]).unwrap_or_else(|_| "{}".into()),
                }
            })),
            _ => {}
        }
    }

    let mut message = Map::new();
    message.insert("role".into(), json!("assistant"));
    message.insert(
        "content".into(),
        if text.is_empty() { Value::Null } else { json!(text) },
    );
    if !reasoning.is_empty() {
        message.insert("reasoning_content".into(), json!(reasoning));
    }
    if !tool_calls.is_empty() {
        message.insert("tool_calls".into(), Value::Array(tool_calls));
    }

    let usage = TokenUsage::from_anthropic(&anthropic["usage"]);
    json!({
        "id": anthropic["id"].as_str().unwrap_or("chatcmpl_pxy"),
        "object": "chat.completion",
        "created": now_secs(),
        "model": model,
        "choices": [{
            "index": 0,
            "message": Value::Object(message),
            "finish_reason": map_stop_reason(anthropic["stop_reason"].as_str()),
        }],
        "usage": {
            "prompt_tokens": usage.input,
            "completion_tokens": usage.output,
            "total_tokens": usage.input + usage.output,
        },
    })
}

fn map_stop_reason(reason: Option<&str>) -> &'static str {
    match reason {
        Some("max_tokens") => "length",
        Some("tool_use") => "tool_calls",
        _ => "stop",
    }
}

fn now_secs() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

// ---------------------------------------------------------------------------
// Streaming response: anthropic SSE events -> openai chunks
// ---------------------------------------------------------------------------

#[derive(Default)]
pub struct StreamState {
    model: String,
    message_id: String,
    /// anthropic block index -> openai tool index (for tool_use blocks)
    tool_indices: Vec<(u64, u64)>,
    next_tool_index: u64,
    finished: bool,
    pub usage: TokenUsage,
}

impl StreamState {
    pub fn new(model: &str) -> Self {
        Self {
            model: model.to_string(),
            message_id: format!("chatcmpl_pxy_{}", std::process::id()),
            ..Default::default()
        }
    }

    /// Translate one parsed Anthropic SSE event into OpenAI chunk text.
    pub fn on_event(&mut self, event_type: Option<&str>, data: &str) -> String {
        let payload: Value = match serde_json::from_str(data) {
            Ok(v) => v,
            Err(_) => return String::new(),
        };
        let etype = event_type
            .map(String::from)
            .or_else(|| payload["type"].as_str().map(String::from))
            .unwrap_or_default();

        match etype.as_str() {
            "message_start" => {
                if let Some(id) = payload["message"]["id"].as_str() {
                    self.message_id = id.to_string();
                }
                self.usage.input = payload["message"]["usage"]["input_tokens"]
                    .as_u64()
                    .unwrap_or(0);
                self.chunk(json!({"role": "assistant", "content": ""}), None)
            }
            "content_block_start" => {
                let block = &payload["content_block"];
                if block["type"].as_str() == Some("tool_use") {
                    let a_idx = payload["index"].as_u64().unwrap_or(0);
                    let t_idx = self.next_tool_index;
                    self.next_tool_index += 1;
                    self.tool_indices.push((a_idx, t_idx));
                    self.chunk(
                        json!({"tool_calls": [{
                            "index": t_idx,
                            "id": block["id"],
                            "type": "function",
                            "function": {"name": block["name"], "arguments": ""},
                        }]}),
                        None,
                    )
                } else {
                    String::new()
                }
            }
            "content_block_delta" => {
                let delta = &payload["delta"];
                match delta["type"].as_str() {
                    Some("text_delta") => {
                        self.chunk(json!({"content": delta["text"]}), None)
                    }
                    Some("thinking_delta") => {
                        self.chunk(json!({"reasoning_content": delta["thinking"]}), None)
                    }
                    Some("input_json_delta") => {
                        let a_idx = payload["index"].as_u64().unwrap_or(0);
                        let t_idx = self
                            .tool_indices
                            .iter()
                            .find(|(a, _)| *a == a_idx)
                            .map(|(_, t)| *t)
                            .unwrap_or(0);
                        self.chunk(
                            json!({"tool_calls": [{
                                "index": t_idx,
                                "function": {"arguments": delta["partial_json"]},
                            }]}),
                            None,
                        )
                    }
                    _ => String::new(),
                }
            }
            "message_delta" => {
                if let Some(o) = payload["usage"]["output_tokens"].as_u64() {
                    self.usage.output = o;
                }
                let finish = map_stop_reason(payload["delta"]["stop_reason"].as_str());
                self.finished = true;
                let mut out = self.chunk_with_usage(json!({}), Some(finish));
                out.push_str("data: [DONE]\n\n");
                out
            }
            "message_stop" => {
                if self.finished {
                    String::new()
                } else {
                    self.finished = true;
                    let mut out = self.chunk_with_usage(json!({}), Some("stop"));
                    out.push_str("data: [DONE]\n\n");
                    out
                }
            }
            "error" => {
                // Pass upstream error through as a data event so clients see it
                format_data(&payload)
            }
            _ => String::new(), // ping, content_block_stop
        }
    }

    fn chunk(&self, delta: Value, finish: Option<&str>) -> String {
        format_data(&json!({
            "id": self.message_id,
            "object": "chat.completion.chunk",
            "created": now_secs(),
            "model": self.model,
            "choices": [{
                "index": 0,
                "delta": delta,
                "finish_reason": finish,
            }],
        }))
    }

    fn chunk_with_usage(&self, delta: Value, finish: Option<&str>) -> String {
        format_data(&json!({
            "id": self.message_id,
            "object": "chat.completion.chunk",
            "created": now_secs(),
            "model": self.model,
            "choices": [{
                "index": 0,
                "delta": delta,
                "finish_reason": finish,
            }],
            "usage": {
                "prompt_tokens": self.usage.input,
                "completion_tokens": self.usage.output,
                "total_tokens": self.usage.input + self.usage.output,
            },
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn request_maps_tools_and_history() {
        let req = json!({
            "messages": [
                {"role": "system", "content": "sys"},
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": null, "tool_calls": [
                    {"id": "c1", "type": "function",
                     "function": {"name": "Bash", "arguments": "{\"x\":1}"}}
                ]},
                {"role": "tool", "tool_call_id": "c1", "content": "done"}
            ],
            "tools": [{"type": "function", "function":
                {"name": "Bash", "description": "d", "parameters": {"type": "object"}}}],
            "max_tokens": 500
        });
        let out = request(&req, 8192);
        assert_eq!(out["system"], "sys");
        let msgs = out["messages"].as_array().unwrap();
        assert_eq!(msgs[0]["role"], "user");
        assert_eq!(msgs[1]["content"][0]["type"], "tool_use");
        assert_eq!(msgs[1]["content"][0]["input"]["x"], 1);
        assert_eq!(msgs[2]["content"][0]["type"], "tool_result");
        assert_eq!(out["max_tokens"], 500);
        assert_eq!(out["tools"][0]["input_schema"]["type"], "object");
    }

    #[test]
    fn request_requires_max_tokens_default() {
        let out = request(&json!({"messages": []}), 4096);
        assert_eq!(out["max_tokens"], 4096);
    }

    #[test]
    fn tool_choice_none_drops_tools() {
        let req = json!({
            "messages": [],
            "tools": [{"type": "function", "function": {"name": "X", "parameters": {}}}],
            "tool_choice": "none"
        });
        let out = request(&req, 100);
        assert!(out.get("tools").is_none());
        assert!(out.get("tool_choice").is_none());
    }

    #[test]
    fn stream_translates_text_and_tools() {
        let mut st = StreamState::new("m");
        let start = json!({"type": "message_start", "message":
            {"id": "msg1", "usage": {"input_tokens": 9}}});
        st.on_event(Some("message_start"), &start.to_string());
        let tb = json!({"type": "content_block_start", "index": 1, "content_block":
            {"type": "tool_use", "id": "t1", "name": "Bash", "input": {}}});
        let out = st.on_event(Some("content_block_start"), &tb.to_string());
        assert!(out.contains("\"name\":\"Bash\""));
        let d = json!({"type": "content_block_delta", "index": 1,
            "delta": {"type": "input_json_delta", "partial_json": "{\"a\":1}"}});
        let out = st.on_event(Some("content_block_delta"), &d.to_string());
        assert!(out.contains("arguments"));
        let md = json!({"type": "message_delta",
            "delta": {"stop_reason": "tool_use"}, "usage": {"output_tokens": 4}});
        let out = st.on_event(Some("message_delta"), &md.to_string());
        assert!(out.contains("\"finish_reason\":\"tool_calls\""));
        assert!(out.contains("[DONE]"));
        assert_eq!(st.usage.input, 9);
        assert_eq!(st.usage.output, 4);
    }
}
