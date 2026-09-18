//! An upstream that speaks the OpenAI Responses API (`format = "responses"`).
//!
//! opencode.ai's Zen Go serves gpt-5.6-luna and muse-spark-* only on
//! `/responses` (verified live 2026-09-18: both 500 on chat/completions and on
//! messages). pxy's whole pipeline reasons in Chat Completions, so this module
//! sits at the wire: the outgoing chat body becomes a Responses request at the
//! send site, and the incoming bytes become chat chunks before anything else
//! reads them. Nothing downstream knows the difference.
//!
//! Shapes here were taken from real Zen Go traffic, not inferred: the stream
//! event names, `function_call` items carrying `call_id` + `name` +
//! `arguments`, and `usage.input_tokens_details.cached_tokens`.
//!
//! Reasoning items are not replayed. A client's transcript carries reasoning
//! as text (`reasoning_content`), never the encrypted item, and the API
//! accepts a `function_call` without its reasoning item as long as the item
//! carries no `id` (verified against gpt-5.6-luna on a tool turn).

use std::collections::HashMap;

use bytes::Bytes;
use futures_util::stream::BoxStream;
use futures_util::StreamExt;
use serde_json::{json, Map, Value};

use super::sse::{format_data, SseParser};

/// Chat Completions request -> Responses request.
pub fn request(chat: &Value) -> Value {
    let mut out = Map::new();
    for key in ["model", "stream", "temperature", "top_p", "parallel_tool_calls"] {
        if !chat[key].is_null() {
            out.insert(key.into(), chat[key].clone());
        }
    }
    // pxy keeps no server-side state, and a stored response would leak the
    // conversation into the provider's dashboard.
    out.insert("store".into(), json!(false));
    if let Some(m) = chat["max_tokens"].as_u64().or_else(|| chat["max_completion_tokens"].as_u64()) {
        out.insert("max_output_tokens".into(), json!(m));
    }
    if let Some(effort) = chat["reasoning_effort"].as_str() {
        // `summary: auto` is what makes the model's reasoning stream back as
        // `reasoning_summary_text.delta`, which becomes `reasoning_content`.
        out.insert("reasoning".into(), json!({"effort": effort, "summary": "auto"}));
    }
    match chat["response_format"]["type"].as_str() {
        Some("json_object") => {
            out.insert("text".into(), json!({"format": {"type": "json_object"}}));
        }
        Some("json_schema") => {
            let js = &chat["response_format"]["json_schema"];
            out.insert(
                "text".into(),
                json!({"format": {
                    "type": "json_schema",
                    "name": js["name"].as_str().unwrap_or("response"),
                    "schema": js["schema"],
                }}),
            );
        }
        _ => {}
    }

    let mut instructions: Vec<String> = Vec::new();
    let mut input: Vec<Value> = Vec::new();
    let empty = Vec::new();
    for msg in chat["messages"].as_array().unwrap_or(&empty) {
        match msg["role"].as_str().unwrap_or("user") {
            "system" | "developer" => {
                let text = flatten_text(&msg["content"]);
                if !text.is_empty() {
                    instructions.push(text);
                }
            }
            "assistant" => {
                let text = flatten_text(&msg["content"]);
                if !text.is_empty() {
                    input.push(json!({
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": text}],
                    }));
                }
                for call in msg["tool_calls"].as_array().unwrap_or(&empty) {
                    let name = call["function"]["name"].as_str().unwrap_or("").trim();
                    let call_id = call["id"].as_str().unwrap_or("").trim();
                    if name.is_empty() || call_id.is_empty() {
                        continue;
                    }
                    let arguments = match &call["function"]["arguments"] {
                        Value::String(s) if !s.trim().is_empty() => s.clone(),
                        Value::String(_) | Value::Null => "{}".to_string(),
                        other => other.to_string(),
                    };
                    // No `id`: with one, the API demands the reasoning item
                    // that preceded the call, which the client never had.
                    input.push(json!({
                        "type": "function_call",
                        "call_id": call_id,
                        "name": name,
                        "arguments": arguments,
                    }));
                }
            }
            "tool" => {
                let call_id = msg["tool_call_id"].as_str().unwrap_or("").trim();
                if call_id.is_empty() {
                    continue;
                }
                input.push(json!({
                    "type": "function_call_output",
                    "call_id": call_id,
                    "output": flatten_text(&msg["content"]),
                }));
            }
            _ => {
                let content = user_content(&msg["content"]);
                input.push(json!({"role": "user", "content": content}));
            }
        }
    }
    if !instructions.is_empty() {
        out.insert("instructions".into(), json!(instructions.join("\n\n")));
    }
    out.insert("input".into(), Value::Array(input));

    let tools: Vec<Value> = chat["tools"]
        .as_array()
        .unwrap_or(&empty)
        .iter()
        .filter_map(|t| {
            let f = &t["function"];
            let name = f["name"].as_str()?.trim();
            if name.is_empty() {
                return None;
            }
            let mut tool = Map::new();
            tool.insert("type".into(), json!("function"));
            tool.insert("name".into(), json!(name));
            if let Some(d) = f["description"].as_str() {
                tool.insert("description".into(), json!(d));
            }
            tool.insert(
                "parameters".into(),
                if f["parameters"].is_null() {
                    json!({"type": "object", "properties": {}})
                } else {
                    f["parameters"].clone()
                },
            );
            if !f["strict"].is_null() {
                tool.insert("strict".into(), f["strict"].clone());
            }
            Some(Value::Object(tool))
        })
        .collect();
    if !tools.is_empty() {
        out.insert("tools".into(), Value::Array(tools));
        match &chat["tool_choice"] {
            Value::String(s) => {
                out.insert("tool_choice".into(), json!(s));
            }
            v if v["function"]["name"].is_string() => {
                out.insert(
                    "tool_choice".into(),
                    json!({"type": "function", "name": v["function"]["name"]}),
                );
            }
            _ => {}
        }
    }
    Value::Object(out)
}

/// Text of a chat message's content: a string, or the text parts joined.
fn flatten_text(content: &Value) -> String {
    match content {
        Value::String(s) => s.clone(),
        Value::Array(parts) => parts
            .iter()
            .filter_map(|p| p["text"].as_str())
            .collect::<Vec<_>>()
            .join(""),
        _ => String::new(),
    }
}

/// A user message's content as Responses input parts (text stays a string).
fn user_content(content: &Value) -> Value {
    let Value::Array(parts) = content else {
        return json!(flatten_text(content));
    };
    let converted: Vec<Value> = parts
        .iter()
        .filter_map(|p| match p["type"].as_str() {
            Some("text") => Some(json!({"type": "input_text", "text": p["text"]})),
            Some("image_url") => {
                let url = p["image_url"]["url"].as_str().or_else(|| p["image_url"].as_str())?;
                let mut part = json!({"type": "input_image", "image_url": url});
                if let Some(d) = p["image_url"]["detail"].as_str() {
                    part["detail"] = json!(d);
                }
                Some(part)
            }
            _ => None,
        })
        .collect();
    Value::Array(converted)
}

/// A completed Responses body -> a chat completion body.
pub fn response(resp: &Value) -> Value {
    let mut content = String::new();
    let mut reasoning = String::new();
    let mut tool_calls: Vec<Value> = Vec::new();
    let empty = Vec::new();
    for item in resp["output"].as_array().unwrap_or(&empty) {
        match item["type"].as_str() {
            Some("message") => {
                for part in item["content"].as_array().unwrap_or(&empty) {
                    if part["type"].as_str() == Some("output_text") {
                        content.push_str(part["text"].as_str().unwrap_or(""));
                    }
                }
            }
            Some("reasoning") => {
                for part in item["summary"].as_array().unwrap_or(&empty) {
                    reasoning.push_str(part["text"].as_str().unwrap_or(""));
                }
            }
            Some("function_call") => {
                tool_calls.push(json!({
                    "id": item["call_id"],
                    "type": "function",
                    "function": {"name": item["name"], "arguments": item["arguments"]},
                }));
            }
            _ => {}
        }
    }
    let mut message = json!({"role": "assistant", "content": content});
    if !reasoning.is_empty() {
        message["reasoning_content"] = json!(reasoning);
    }
    if !tool_calls.is_empty() {
        message["tool_calls"] = Value::Array(tool_calls.clone());
    }
    let finish = finish_reason(resp, !tool_calls.is_empty());
    let mut out = json!({
        "id": resp["id"],
        "object": "chat.completion",
        "created": resp["created_at"],
        "model": resp["model"],
        "choices": [{"index": 0, "message": message, "finish_reason": finish}],
    });
    if resp["usage"].is_object() {
        out["usage"] = usage(&resp["usage"]);
    }
    out
}

fn finish_reason(resp: &Value, has_tool_calls: bool) -> &'static str {
    if resp["incomplete_details"]["reason"].as_str() == Some("max_output_tokens") {
        "length"
    } else if has_tool_calls {
        "tool_calls"
    } else {
        "stop"
    }
}

fn usage(u: &Value) -> Value {
    let input = u["input_tokens"].as_u64().unwrap_or(0);
    let output = u["output_tokens"].as_u64().unwrap_or(0);
    json!({
        "prompt_tokens": input,
        "completion_tokens": output,
        "total_tokens": u["total_tokens"].as_u64().unwrap_or(input + output),
        "prompt_tokens_details": {
            "cached_tokens": u["input_tokens_details"]["cached_tokens"].as_u64().unwrap_or(0),
        },
        "completion_tokens_details": {
            "reasoning_tokens": u["output_tokens_details"]["reasoning_tokens"].as_u64().unwrap_or(0),
        },
    })
}

/// Responses stream events -> chat completion chunks, one call at a time.
#[derive(Default)]
pub struct StreamState {
    id: String,
    model: String,
    created: u64,
    /// function_call item id -> chat tool_call index.
    tools: HashMap<String, u64>,
    /// output_index -> chat tool_call index, for gateways that omit item_id.
    tools_by_output: HashMap<u64, u64>,
    next_tool: u64,
    /// Reasoning already streamed as deltas, so a reasoning item's `done`
    /// summary must not be emitted a second time.
    reasoning_streamed: bool,
    done: bool,
}

impl StreamState {
    pub fn new() -> Self {
        Self::default()
    }

    fn chunk(&self, delta: Value, finish: Option<&str>) -> String {
        let mut choice = json!({"index": 0, "delta": delta});
        choice["finish_reason"] = match finish {
            Some(f) => json!(f),
            None => Value::Null,
        };
        format_data(&json!({
            "id": self.id,
            "object": "chat.completion.chunk",
            "created": self.created,
            "model": self.model,
            "choices": [choice],
        }))
    }

    fn note_response(&mut self, resp: &Value) {
        if let Some(id) = resp["id"].as_str() {
            self.id = id.to_string();
        }
        if let Some(m) = resp["model"].as_str() {
            self.model = m.to_string();
        }
        if let Some(c) = resp["created_at"].as_u64() {
            self.created = c;
        }
    }

    /// The chat SSE text one upstream event produces (often nothing).
    pub fn on_event(&mut self, data: &str) -> String {
        if self.done {
            return String::new();
        }
        let Ok(ev) = serde_json::from_str::<Value>(data.trim()) else {
            return String::new();
        };
        // A bare error object (Zen wraps upstream errors this way) is relayed
        // as the chat-dialect error chunk the router already classifies.
        if ev["type"].as_str() == Some("error") || (ev["type"].is_null() && ev["error"].is_object()) {
            self.done = true;
            let err = if ev["error"].is_object() { ev["error"].clone() } else { ev.clone() };
            return format_data(&json!({"error": err}));
        }
        match ev["type"].as_str().unwrap_or("") {
            "response.created" => {
                self.note_response(&ev["response"]);
                self.chunk(json!({"role": "assistant", "content": ""}), None)
            }
            "response.output_text.delta" => {
                self.chunk(json!({"content": ev["delta"]}), None)
            }
            "response.reasoning_summary_text.delta" | "response.reasoning_text.delta" => {
                self.reasoning_streamed = true;
                self.chunk(json!({"reasoning_content": ev["delta"]}), None)
            }
            "response.output_item.added" if ev["item"]["type"].as_str() == Some("function_call") => {
                let item = &ev["item"];
                let index = self.next_tool;
                self.next_tool += 1;
                if let Some(id) = item["id"].as_str() {
                    self.tools.insert(id.to_string(), index);
                }
                if let Some(o) = ev["output_index"].as_u64() {
                    self.tools_by_output.insert(o, index);
                }
                self.chunk(
                    json!({"tool_calls": [{
                        "index": index,
                        "id": item["call_id"],
                        "type": "function",
                        "function": {"name": item["name"], "arguments": ""},
                    }]}),
                    None,
                )
            }
            "response.function_call_arguments.delta" => {
                let index = ev["item_id"]
                    .as_str()
                    .and_then(|id| self.tools.get(id).copied())
                    .or_else(|| ev["output_index"].as_u64().and_then(|o| self.tools_by_output.get(&o).copied()));
                match index {
                    Some(index) => self.chunk(
                        json!({"tool_calls": [{"index": index, "function": {"arguments": ev["delta"]}}]}),
                        None,
                    ),
                    None => String::new(),
                }
            }
            // A reasoning item whose summary never streamed (some gateways
            // deliver it whole) still reaches the client once.
            "response.output_item.done"
                if ev["item"]["type"].as_str() == Some("reasoning") && !self.reasoning_streamed =>
            {
                let empty = Vec::new();
                let text: String = ev["item"]["summary"]
                    .as_array()
                    .unwrap_or(&empty)
                    .iter()
                    .filter_map(|p| p["text"].as_str())
                    .collect();
                if text.is_empty() {
                    String::new()
                } else {
                    self.chunk(json!({"reasoning_content": text}), None)
                }
            }
            "response.completed" | "response.incomplete" => {
                self.done = true;
                let resp = &ev["response"];
                self.note_response(resp);
                let finish = finish_reason(resp, self.next_tool > 0);
                let mut out = self.chunk(json!({}), Some(finish));
                if resp["usage"].is_object() {
                    out.push_str(&format_data(&json!({
                        "id": self.id,
                        "object": "chat.completion.chunk",
                        "created": self.created,
                        "model": self.model,
                        "choices": [],
                        "usage": usage(&resp["usage"]),
                    })));
                }
                out.push_str("data: [DONE]\n\n");
                out
            }
            "response.failed" => {
                self.done = true;
                let err = if ev["response"]["error"].is_object() {
                    ev["response"]["error"].clone()
                } else {
                    json!({"message": "response failed", "type": "server_error"})
                };
                format_data(&json!({"error": err}))
            }
            _ => String::new(),
        }
    }
}

/// Wrap an upstream Responses byte stream so it reads as a chat completions
/// SSE stream. Empty translations are skipped, so the first yielded bytes are
/// the first real chat event (the router's pre-commit hold relies on that).
pub fn chat_stream(
    upstream: BoxStream<'static, reqwest::Result<Bytes>>,
) -> BoxStream<'static, reqwest::Result<Bytes>> {
    futures_util::stream::unfold(
        (upstream, SseParser::new(), StreamState::new()),
        |(mut upstream, mut parser, mut state)| async move {
            loop {
                match upstream.next().await {
                    Some(Ok(bytes)) => {
                        let out: String =
                            parser.feed(&bytes).iter().map(|e| state.on_event(&e.data)).collect();
                        if !out.is_empty() {
                            return Some((Ok(Bytes::from(out)), (upstream, parser, state)));
                        }
                    }
                    Some(Err(e)) => return Some((Err(e), (upstream, parser, state))),
                    None => return None,
                }
            }
        },
    )
    .boxed()
}

/// Non-streaming re-assembly for `force_stream`: Responses events -> the
/// chat chunk events `aggregate::openai` reads.
pub fn chat_events(events: &[super::sse::SseEvent]) -> Vec<super::sse::SseEvent> {
    let mut state = StreamState::new();
    let text: String = events.iter().map(|e| state.on_event(&e.data)).collect();
    let mut parser = SseParser::new();
    parser.feed(text.as_bytes())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn request_maps_messages_tools_and_knobs() {
        let chat = json!({
            "model": "gpt-5.6-luna",
            "stream": true,
            "max_tokens": 400,
            "reasoning_effort": "low",
            "temperature": 0.2,
            "stream_options": {"include_usage": true},
            "messages": [
                {"role": "system", "content": "Be brief."},
                {"role": "developer", "content": [{"type": "text", "text": "Use tools."}]},
                {"role": "user", "content": [
                    {"type": "text", "text": "Weather in Dhaka?"},
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA", "detail": "low"}}
                ]},
                {"role": "assistant", "content": null, "reasoning_content": "thinking",
                 "tool_calls": [{"id": "call_1", "type": "function",
                                 "function": {"name": "get_weather", "arguments": "{\"city\":\"Dhaka\"}"}}]},
                {"role": "tool", "tool_call_id": "call_1", "content": "31C"},
                {"role": "assistant", "content": "It is 31C."},
                {"role": "user", "content": "thanks"}
            ],
            "tools": [{"type": "function", "function": {
                "name": "get_weather", "description": "Get weather",
                "parameters": {"type": "object", "properties": {"city": {"type": "string"}}}}}],
            "tool_choice": {"type": "function", "function": {"name": "get_weather"}},
        });
        let out = request(&chat);
        assert_eq!(out["model"], "gpt-5.6-luna");
        assert_eq!(out["stream"], true);
        assert_eq!(out["store"], false);
        assert_eq!(out["max_output_tokens"], 400);
        assert_eq!(out["reasoning"], json!({"effort": "low", "summary": "auto"}));
        assert_eq!(out["temperature"], 0.2);
        assert!(out.get("stream_options").is_none());
        assert!(out.get("max_tokens").is_none());
        assert_eq!(out["instructions"], "Be brief.\n\nUse tools.");
        let input = out["input"].as_array().unwrap();
        assert_eq!(input[0]["role"], "user");
        assert_eq!(input[0]["content"][0], json!({"type": "input_text", "text": "Weather in Dhaka?"}));
        assert_eq!(
            input[0]["content"][1],
            json!({"type": "input_image", "image_url": "data:image/png;base64,AAAA", "detail": "low"})
        );
        // The tool call rides as a function_call item with NO `id`, and the
        // reasoning text is not replayed.
        assert_eq!(
            input[1],
            json!({"type": "function_call", "call_id": "call_1", "name": "get_weather",
                   "arguments": "{\"city\":\"Dhaka\"}"})
        );
        assert_eq!(input[2], json!({"type": "function_call_output", "call_id": "call_1", "output": "31C"}));
        assert_eq!(input[3]["role"], "assistant");
        assert_eq!(input[3]["content"][0], json!({"type": "output_text", "text": "It is 31C."}));
        assert_eq!(input[4], json!({"role": "user", "content": "thanks"}));
        assert_eq!(input.len(), 5);
        assert_eq!(out["tools"][0]["name"], "get_weather");
        assert_eq!(out["tools"][0]["type"], "function");
        assert!(out["tools"][0].get("function").is_none(), "Responses tools are flat");
        assert_eq!(out["tool_choice"], json!({"type": "function", "name": "get_weather"}));
    }

    #[test]
    fn request_without_tools_or_effort_stays_minimal() {
        let chat = json!({"model": "m", "messages": [{"role": "user", "content": "hi"}],
            "tools": [], "tool_choice": "auto", "max_completion_tokens": 10});
        let out = request(&chat);
        assert!(out.get("tools").is_none());
        assert!(out.get("tool_choice").is_none());
        assert!(out.get("reasoning").is_none());
        assert!(out.get("instructions").is_none());
        assert_eq!(out["max_output_tokens"], 10);
        assert_eq!(out["input"], json!([{"role": "user", "content": "hi"}]));
    }

    /// The completed body Zen Go returned for a tool call (trimmed).
    fn completed_tool_call() -> Value {
        json!({
            "id": "resp_1", "object": "response", "created_at": 1789736470, "status": "completed",
            "model": "gpt-5.6-luna", "error": null, "incomplete_details": null,
            "output": [
                {"id": "rs_1", "type": "reasoning", "summary": [{"type": "summary_text", "text": "Need the tool."}]},
                {"id": "fc_1", "type": "function_call", "status": "completed", "name": "get_weather",
                 "call_id": "call_zKA", "arguments": "{\"city\":\"Dhaka\"}"}
            ],
            "usage": {"input_tokens": 60, "output_tokens": 19, "total_tokens": 79,
                      "input_tokens_details": {"cached_tokens": 8},
                      "output_tokens_details": {"reasoning_tokens": 5}}
        })
    }

    #[test]
    fn response_maps_output_items_finish_and_usage() {
        let out = response(&completed_tool_call());
        assert_eq!(out["id"], "resp_1");
        assert_eq!(out["object"], "chat.completion");
        assert_eq!(out["created"], 1789736470);
        assert_eq!(out["model"], "gpt-5.6-luna");
        let msg = &out["choices"][0]["message"];
        assert_eq!(msg["role"], "assistant");
        assert_eq!(msg["content"], "");
        assert_eq!(msg["reasoning_content"], "Need the tool.");
        assert_eq!(
            msg["tool_calls"][0],
            json!({"id": "call_zKA", "type": "function",
                   "function": {"name": "get_weather", "arguments": "{\"city\":\"Dhaka\"}"}})
        );
        assert_eq!(out["choices"][0]["finish_reason"], "tool_calls");
        assert_eq!(out["usage"]["prompt_tokens"], 60);
        assert_eq!(out["usage"]["completion_tokens"], 19);
        assert_eq!(out["usage"]["total_tokens"], 79);
        assert_eq!(out["usage"]["prompt_tokens_details"]["cached_tokens"], 8);
        assert_eq!(out["usage"]["completion_tokens_details"]["reasoning_tokens"], 5);

        let text = json!({"id": "r", "status": "completed", "model": "muse",
            "output": [{"type": "message", "content": [{"type": "output_text", "text": "Hi"},
                                                        {"type": "output_text", "text": "!"}]}]});
        let out = response(&text);
        assert_eq!(out["choices"][0]["message"]["content"], "Hi!");
        assert_eq!(out["choices"][0]["finish_reason"], "stop");
        assert!(out["choices"][0]["message"].get("reasoning_content").is_none());
        assert!(out.get("usage").is_none());

        let cut = json!({"id": "r", "status": "incomplete", "model": "muse",
            "incomplete_details": {"reason": "max_output_tokens"}, "output": []});
        assert_eq!(response(&cut)["choices"][0]["finish_reason"], "length");
    }

    fn events(state: &mut StreamState, evs: &[Value]) -> Vec<Value> {
        let text: String = evs.iter().map(|e| state.on_event(&e.to_string())).collect();
        text.split("\n\n")
            .filter(|l| !l.is_empty())
            .map(|l| {
                let d = l.strip_prefix("data: ").unwrap();
                if d == "[DONE]" { json!("[DONE]") } else { serde_json::from_str(d).unwrap() }
            })
            .collect()
    }

    #[test]
    fn stream_translates_a_tool_call_turn() {
        let mut st = StreamState::new();
        let out = events(&mut st, &[
            json!({"type": "ping"}),
            json!({"type": "response.created", "response": {"id": "resp_1", "model": "gpt-5.6-luna", "created_at": 7}}),
            json!({"type": "response.in_progress", "response": {}}),
            json!({"type": "response.output_item.added", "output_index": 0,
                   "item": {"id": "fc_1", "type": "function_call", "name": "get_weather", "call_id": "call_zKA", "arguments": ""}}),
            json!({"type": "response.function_call_arguments.delta", "output_index": 0, "item_id": "fc_1", "delta": "{\"city\":", "obfuscation": "x"}),
            json!({"type": "response.function_call_arguments.delta", "output_index": 0, "item_id": "fc_1", "delta": "\"Dhaka\"}"}),
            json!({"type": "response.function_call_arguments.done", "item_id": "fc_1", "arguments": "{\"city\":\"Dhaka\"}"}),
            json!({"type": "response.output_item.done", "output_index": 0, "item": {"type": "function_call"}}),
            json!({"type": "response.completed", "response": completed_tool_call()}),
        ]);
        assert_eq!(out[0]["choices"][0]["delta"], json!({"role": "assistant", "content": ""}));
        assert_eq!(out[0]["id"], "resp_1");
        assert_eq!(out[0]["model"], "gpt-5.6-luna");
        assert_eq!(out[0]["object"], "chat.completion.chunk");
        assert_eq!(
            out[1]["choices"][0]["delta"]["tool_calls"][0],
            json!({"index": 0, "id": "call_zKA", "type": "function", "function": {"name": "get_weather", "arguments": ""}})
        );
        assert_eq!(out[2]["choices"][0]["delta"]["tool_calls"][0], json!({"index": 0, "function": {"arguments": "{\"city\":"}}));
        assert_eq!(out[3]["choices"][0]["delta"]["tool_calls"][0]["function"]["arguments"], "\"Dhaka\"}");
        assert_eq!(out[4]["choices"][0]["finish_reason"], "tool_calls");
        assert_eq!(out[4]["choices"][0]["delta"], json!({}));
        assert_eq!(out[5]["usage"]["prompt_tokens"], 60);
        assert_eq!(out[5]["choices"], json!([]));
        assert_eq!(out[6], "[DONE]");
        assert_eq!(out.len(), 7);
        // Nothing after completion.
        assert!(st.on_event(r#"{"type":"response.output_text.delta","delta":"late"}"#).is_empty());
    }

    #[test]
    fn stream_translates_reasoning_and_text() {
        let mut st = StreamState::new();
        let out = events(&mut st, &[
            json!({"type": "response.created", "response": {"id": "r", "model": "muse", "created_at": 1}}),
            json!({"type": "response.output_item.added", "output_index": 0, "item": {"id": "rs_1", "type": "reasoning"}}),
            json!({"type": "response.reasoning_summary_text.delta", "item_id": "rs_1", "delta": "Simple."}),
            json!({"type": "response.output_item.done", "output_index": 0,
                   "item": {"id": "rs_1", "type": "reasoning", "summary": [{"type": "summary_text", "text": "Simple."}]}}),
            json!({"type": "response.output_item.added", "output_index": 1, "item": {"id": "msg_1", "type": "message"}}),
            json!({"type": "response.content_part.added", "item_id": "msg_1", "part": {"type": "output_text", "text": ""}}),
            json!({"type": "response.output_text.delta", "item_id": "msg_1", "delta": "Hi"}),
            json!({"type": "response.incomplete", "response": {"id": "r", "model": "muse",
                   "incomplete_details": {"reason": "max_output_tokens"}, "output": []}}),
        ]);
        assert_eq!(out[1]["choices"][0]["delta"], json!({"reasoning_content": "Simple."}));
        // The done summary is not repeated after it streamed.
        assert_eq!(out[2]["choices"][0]["delta"], json!({"content": "Hi"}));
        assert_eq!(out[3]["choices"][0]["finish_reason"], "length");
        assert_eq!(out[4], "[DONE]");
        assert_eq!(out.len(), 5);

        // A whole-summary gateway (no deltas) still delivers the reasoning once.
        let mut st = StreamState::new();
        let out = events(&mut st, &[
            json!({"type": "response.created", "response": {"id": "r"}}),
            json!({"type": "response.output_item.done", "output_index": 0,
                   "item": {"type": "reasoning", "summary": [{"type": "summary_text", "text": "Whole."}]}}),
        ]);
        assert_eq!(out[1]["choices"][0]["delta"], json!({"reasoning_content": "Whole."}));
    }

    #[test]
    fn stream_relays_errors_as_chat_error_chunks() {
        let mut st = StreamState::new();
        let out = events(&mut st, &[
            json!({"type": "error", "error": {"type": "MissingSessionID", "message": "no session"}}),
        ]);
        assert_eq!(out[0]["error"]["message"], "no session");
        let mut st = StreamState::new();
        let out = events(&mut st, &[
            json!({"type": "response.created", "response": {"id": "r"}}),
            json!({"type": "response.failed", "response": {"error": {"code": "server_error", "message": "boom"}}}),
        ]);
        assert_eq!(out[1]["error"]["message"], "boom");
        assert_eq!(out.len(), 2);
    }

    #[test]
    fn chat_events_reassemble_for_force_stream() {
        let raw = [
            json!({"type": "response.created", "response": {"id": "r", "model": "m", "created_at": 1}}),
            json!({"type": "response.output_text.delta", "delta": "Hi"}),
            json!({"type": "response.completed", "response": {"id": "r", "model": "m", "output": [],
                   "usage": {"input_tokens": 3, "output_tokens": 1}}}),
        ];
        let events: Vec<super::super::sse::SseEvent> = raw
            .iter()
            .map(|d| super::super::sse::SseEvent { event: None, data: d.to_string() })
            .collect();
        let chat = chat_events(&events);
        let body = super::super::aggregate::openai(&chat);
        assert_eq!(body["choices"][0]["message"]["content"], "Hi");
        assert_eq!(body["usage"]["prompt_tokens"], 3);
    }
}
