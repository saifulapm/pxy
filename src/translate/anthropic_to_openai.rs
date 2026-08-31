//! Claude-client path: Anthropic Messages request -> OpenAI chat completions,
//! and OpenAI response -> Anthropic Messages response (stream + non-stream).

use serde_json::{json, Map, Value};

use super::sse::format_event;
use super::web_search;
use super::TokenUsage;

// ---------------------------------------------------------------------------
// Request: anthropic -> openai
// ---------------------------------------------------------------------------

pub fn request(anthropic: &Value) -> Value {
    let mut messages: Vec<Value> = Vec::new();

    // system: string or array of text blocks -> one system message
    match &anthropic["system"] {
        Value::String(s) if !s.is_empty() => {
            messages.push(json!({"role": "system", "content": s}));
        }
        Value::Array(blocks) => {
            let text = blocks
                .iter()
                .filter_map(|b| b["text"].as_str())
                .collect::<Vec<_>>()
                .join("\n");
            if !text.is_empty() {
                messages.push(json!({"role": "system", "content": text}));
            }
        }
        _ => {}
    }

    for msg in anthropic["messages"].as_array().unwrap_or(&Vec::new()) {
        let role = msg["role"].as_str().unwrap_or("user");
        match &msg["content"] {
            Value::String(s) => {
                messages.push(json!({"role": role, "content": s}));
            }
            Value::Array(blocks) => {
                if role == "assistant" {
                    push_assistant_turn(&mut messages, blocks);
                } else {
                    push_user_turn(&mut messages, blocks);
                }
            }
            _ => {}
        }
    }

    repair_tool_pairs(&mut messages);

    let mut out = Map::new();
    out.insert("messages".into(), Value::Array(messages));

    if let Some(mt) = anthropic["max_tokens"].as_u64() {
        out.insert("max_tokens".into(), json!(mt));
    }
    for key in ["temperature", "top_p", "stream"] {
        if !anthropic[key].is_null() {
            out.insert(key.into(), anthropic[key].clone());
        }
    }
    if let Some(stops) = anthropic["stop_sequences"].as_array() {
        if !stops.is_empty() {
            out.insert("stop".into(), Value::Array(stops.clone()));
        }
    }

    if let Some(tools) = anthropic["tools"].as_array() {
        let mut mapped: Vec<Value> = tools
            .iter()
            // Server tools (web_search, code_execution, …) carry a `name` but
            // no schema because the API, not the model, runs them. Mapping one
            // to a function yields a zero-argument call the client can't
            // execute, so they never reach the upstream as functions:
            // web_search is substituted below, the rest are dropped.
            .filter(|t| t["name"].is_string() && t["input_schema"].is_object())
            .map(|t| {
                let mut params = t["input_schema"].clone();
                // OpenAI strict mode chokes on schemas without properties
                if params.get("properties").is_none() {
                    params["properties"] = json!({});
                }
                json!({
                    "type": "function",
                    "function": {
                        "name": t["name"],
                        "description": t["description"].as_str().unwrap_or(""),
                        "parameters": params,
                    }
                })
            })
            .collect();
        // web_search is the one server tool pxy can run itself: it comes back
        // as a real function, and the router intercepts the calls — for a
        // non-streaming client too, by running the request through the stream
        // machinery and re-assembling its JSON. Whether pxy can actually
        // serve it (search provider configured, upstream format) is the
        // router's call: attempt() strips the function whenever nothing would
        // intercept it, so injecting on the dialect evidence alone is safe.
        if web_search::plan(anthropic).is_some()
            && !mapped.iter().any(|t| t["function"]["name"] == web_search::TOOL_NAME)
        {
            mapped.insert(0, web_search::tool_def());
        }
        if !mapped.is_empty() {
            out.insert("tools".into(), Value::Array(mapped));
        }
    }

    match anthropic["tool_choice"]["type"].as_str() {
        Some("auto") => {
            out.insert("tool_choice".into(), json!("auto"));
        }
        Some("any") => {
            out.insert("tool_choice".into(), json!("required"));
        }
        Some("tool") => {
            out.insert(
                "tool_choice".into(),
                json!({"type": "function", "function": {"name": anthropic["tool_choice"]["name"]}}),
            );
        }
        Some("none") => {
            // Safest portable interpretation: no tools at all.
            out.remove("tools");
        }
        _ => {}
    }

    // thinking.budget_tokens -> reasoning_effort buckets (OmniRoute mapping).
    // "adaptive" and other beta shapes are dropped for OpenAI upstreams.
    if anthropic["thinking"]["type"].as_str() == Some("enabled") {
        let budget = anthropic["thinking"]["budget_tokens"].as_u64().unwrap_or(0);
        let effort = if budget <= 1024 {
            "low"
        } else if budget <= 10240 {
            "medium"
        } else {
            "high"
        };
        out.insert("reasoning_effort".into(), json!(effort));
    }

    Value::Object(out)
}

fn flatten_text(content: &Value) -> String {
    match content {
        Value::String(s) => s.clone(),
        Value::Array(blocks) => blocks
            .iter()
            .filter_map(|b| {
                // Tool messages on OpenAI upstreams are text-only, but a
                // client tool may return screenshots: the image must degrade
                // to a visible marker, or the model never knows it got one.
                if b["type"].as_str() == Some("image") {
                    return Some("[image omitted]".to_string());
                }
                b["text"].as_str().map(String::from)
            })
            .collect::<Vec<_>>()
            .join("\n"),
        _ => String::new(),
    }
}

fn push_user_turn(messages: &mut Vec<Value>, blocks: &[Value]) {
    // tool_result blocks become role:"tool" messages, emitted BEFORE the
    // remaining user content (they answer the previous assistant turn).
    let mut parts: Vec<Value> = Vec::new();
    for block in blocks {
        match block["type"].as_str() {
            Some("tool_result") => {
                messages.push(json!({
                    "role": "tool",
                    "tool_call_id": block["tool_use_id"],
                    "content": flatten_text(&block["content"]),
                }));
            }
            Some("text") => {
                parts.push(json!({"type": "text", "text": block["text"]}));
            }
            Some("image") => {
                let src = &block["source"];
                let url = if src["type"].as_str() == Some("url") {
                    src["url"].as_str().unwrap_or("").to_string()
                } else {
                    format!(
                        "data:{};base64,{}",
                        src["media_type"].as_str().unwrap_or("image/png"),
                        src["data"].as_str().unwrap_or("")
                    )
                };
                parts.push(json!({"type": "image_url", "image_url": {"url": url}}));
            }
            _ => {} // document, cache markers etc: dropped for openai upstreams
        }
    }
    if !parts.is_empty() {
        // Plain-text-only content collapses to a string (widest compatibility)
        let all_text = parts.iter().all(|p| p["type"] == "text");
        let content = if all_text {
            Value::String(
                parts
                    .iter()
                    .filter_map(|p| p["text"].as_str())
                    .collect::<Vec<_>>()
                    .join("\n"),
            )
        } else {
            Value::Array(parts)
        };
        messages.push(json!({"role": "user", "content": content}));
    }
}

fn push_assistant_turn(messages: &mut Vec<Value>, blocks: &[Value]) {
    let mut text = String::new();
    let mut tool_calls: Vec<Value> = Vec::new();
    for block in blocks {
        match block["type"].as_str() {
            Some("text") => {
                if !text.is_empty() {
                    text.push('\n');
                }
                text.push_str(block["text"].as_str().unwrap_or(""));
            }
            Some("tool_use") => {
                tool_calls.push(json!({
                    "id": block["id"],
                    "type": "function",
                    "function": {
                        "name": block["name"],
                        "arguments": serde_json::to_string(&block["input"]).unwrap_or_else(|_| "{}".into()),
                    }
                }));
            }
            // The search blocks pxy emitted last turn come back here. The
            // upstream has no notion of a server tool, so they replay as prose
            // — dropping them would lose what the search found.
            Some("server_tool_use") | Some("web_search_tool_result") => {
                if let Some(line) = web_search::flatten_history_block(block) {
                    if !text.is_empty() {
                        text.push('\n');
                    }
                    text.push_str(&line);
                }
            }
            // thinking / redacted_thinking: dropped for openai upstreams
            _ => {}
        }
    }
    let mut msg = Map::new();
    msg.insert("role".into(), json!("assistant"));
    msg.insert(
        "content".into(),
        if text.is_empty() { Value::Null } else { Value::String(text) },
    );
    if !tool_calls.is_empty() {
        msg.insert("tool_calls".into(), Value::Array(tool_calls));
    }
    messages.push(Value::Object(msg));
}

/// OpenAI upstreams 400 on malformed tool conversations. Two repairs
/// (OmniRoute's regroupToolMessages + fixMissingToolResponses):
/// every assistant tool_call must be followed by a matching tool message;
/// tool messages without a preceding matching tool_call are dropped.
fn repair_tool_pairs(messages: &mut Vec<Value>) {
    let mut repaired: Vec<Value> = Vec::with_capacity(messages.len());
    let mut pending: Vec<String> = Vec::new(); // unanswered tool_call ids

    for msg in messages.drain(..) {
        let role = msg["role"].as_str().unwrap_or("");
        if role == "tool" {
            let id = msg["tool_call_id"].as_str().unwrap_or("").to_string();
            if let Some(pos) = pending.iter().position(|p| *p == id) {
                pending.remove(pos);
                repaired.push(msg);
            }
            // orphan tool message: dropped
            continue;
        }
        // A non-tool message ends the answer window: close out unanswered calls
        for id in pending.drain(..) {
            repaired.push(json!({
                "role": "tool",
                "tool_call_id": id,
                "content": "[No response received]",
            }));
        }
        if role == "assistant" {
            if let Some(calls) = msg["tool_calls"].as_array() {
                pending = calls
                    .iter()
                    .filter_map(|c| c["id"].as_str().map(String::from))
                    .collect();
            }
        }
        repaired.push(msg);
    }
    for id in pending.drain(..) {
        repaired.push(json!({
            "role": "tool",
            "tool_call_id": id,
            "content": "[No response received]",
        }));
    }
    *messages = repaired;
}

// ---------------------------------------------------------------------------
// Non-streaming response: openai -> anthropic
// ---------------------------------------------------------------------------

/// Chain-of-thought text from a message or delta. OpenAI-compatible upstreams
/// disagree on the field: most say `reasoning_content`, the z-ai/GLM family
/// says `reasoning`. Reading only the first name drops the entire thinking
/// phase — the client gets no bytes at all while the model reasons, and Claude
/// Code calls 20s of silence a stalled stream.
fn reasoning_text(v: &Value) -> Option<&str> {
    ["reasoning_content", "reasoning"]
        .into_iter()
        .find_map(|k| v[k].as_str())
        .filter(|s| !s.is_empty())
}

pub fn response(openai: &Value, model: &str) -> Value {
    let choice = &openai["choices"][0];
    let message = &choice["message"];
    let mut content: Vec<Value> = Vec::new();

    if let Some(reason) = reasoning_text(message) {
        // No signature field: a fabricated `signature: ""` gets stored in
        // client history and poisons every later replay to a real
        // Anthropic upstream (400 invalid signature, forever).
        content.push(json!({"type": "thinking", "thinking": reason}));
    }
    if let Some(text) = message["content"].as_str() {
        if !text.is_empty() {
            content.push(json!({"type": "text", "text": text}));
        }
    }
    if let Some(calls) = message["tool_calls"].as_array() {
        for call in calls {
            let args = call["function"]["arguments"].as_str().unwrap_or("{}");
            let input: Value = serde_json::from_str(args).unwrap_or_else(|_| json!({}));
            content.push(json!({
                "type": "tool_use",
                "id": call["id"],
                "name": call["function"]["name"],
                "input": input,
            }));
        }
    }

    let usage = TokenUsage::from_openai(&openai["usage"]);
    json!({
        "id": openai["id"].as_str().unwrap_or("msg_pxy"),
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": content,
        "stop_reason": map_finish_reason(choice["finish_reason"].as_str()),
        "stop_sequence": null,
        "usage": {"input_tokens": usage.input, "output_tokens": usage.output},
    })
}

fn map_finish_reason(reason: Option<&str>) -> &'static str {
    match reason {
        Some("length") => "max_tokens",
        Some("tool_calls") | Some("function_call") => "tool_use",
        _ => "end_turn",
    }
}

// ---------------------------------------------------------------------------
// Streaming response: openai chunks -> anthropic SSE events
// ---------------------------------------------------------------------------

#[derive(Default)]
pub struct StreamState {
    started: bool,
    model: String,
    message_id: String,
    next_block: usize,
    /// currently open non-tool block: (index, kind) where kind: 0=text 1=thinking
    open_text: Option<(usize, u8)>,
    /// openai tool index -> ToolBlock
    tools: Vec<ToolSlot>,
    finish_reason: Option<String>,
    pub usage: TokenUsage,
    input_estimate: u64,
}

struct ToolSlot {
    openai_index: u64,
    block_index: Option<usize>, // None until content_block_start emitted
    id: String,
    name: String,
    args: String,
}

impl StreamState {
    pub fn new(model: &str, input_estimate: u64) -> Self {
        Self {
            model: model.to_string(),
            message_id: format!("msg_pxy_{}", std::process::id()),
            input_estimate,
            ..Default::default()
        }
    }

    /// Translate one parsed OpenAI SSE data payload into Anthropic event text.
    pub fn on_data(&mut self, data: &str) -> String {
        if data.trim() == "[DONE]" {
            return self.finish();
        }
        let chunk: Value = match serde_json::from_str(data) {
            Ok(v) => v,
            Err(_) => return String::new(),
        };
        let mut out = String::new();

        if !self.started {
            self.started = true;
            if let Some(id) = chunk["id"].as_str() {
                self.message_id = id.to_string();
            }
            out.push_str(&format_event(
                "message_start",
                &json!({"type": "message_start", "message": {
                    "id": self.message_id, "type": "message", "role": "assistant",
                    "model": self.model, "content": [],
                    "stop_reason": null, "stop_sequence": null,
                    "usage": {"input_tokens": self.input_estimate, "output_tokens": 0},
                }}),
            ));
        }

        if let Some(usage) = chunk.get("usage").filter(|u| u.is_object()) {
            let u = TokenUsage::from_openai(usage);
            if u.input > 0 {
                self.usage.input = u.input;
            }
            if u.output > 0 {
                self.usage.output = u.output;
            }
        }

        let choice = &chunk["choices"][0];
        let delta = &choice["delta"];

        if let Some(reason) = reasoning_text(delta) {
            out.push_str(&self.text_delta(reason, 1));
        }
        if let Some(text) = delta["content"].as_str() {
            if !text.is_empty() {
                out.push_str(&self.text_delta(text, 0));
            }
        }
        if let Some(calls) = delta["tool_calls"].as_array() {
            for call in calls {
                out.push_str(&self.tool_delta(call));
            }
        }
        if let Some(reason) = choice["finish_reason"].as_str() {
            self.finish_reason = Some(reason.to_string());
        }
        out
    }

    /// Text or thinking delta; opens/switches blocks as needed.
    fn text_delta(&mut self, text: &str, kind: u8) -> String {
        let mut out = String::new();
        match self.open_text {
            Some((_, k)) if k == kind => {}
            other => {
                if let Some((idx, _)) = other {
                    out.push_str(&stop_block(idx));
                    self.open_text = None;
                }
                let idx = self.next_block;
                self.next_block += 1;
                let block = if kind == 1 {
                    // No fabricated signature — see the non-streaming path.
                    json!({"type": "thinking", "thinking": ""})
                } else {
                    json!({"type": "text", "text": ""})
                };
                out.push_str(&format_event(
                    "content_block_start",
                    &json!({"type": "content_block_start", "index": idx, "content_block": block}),
                ));
                self.open_text = Some((idx, kind));
            }
        }
        let (idx, _) = self.open_text.unwrap();
        let delta = if kind == 1 {
            json!({"type": "thinking_delta", "thinking": text})
        } else {
            json!({"type": "text_delta", "text": text})
        };
        out.push_str(&format_event(
            "content_block_delta",
            &json!({"type": "content_block_delta", "index": idx, "delta": delta}),
        ));
        out
    }

    fn tool_delta(&mut self, call: &Value) -> String {
        let mut out = String::new();
        let oi = call["index"].as_u64().unwrap_or(0);
        let slot_pos = match self.tools.iter().position(|t| t.openai_index == oi) {
            Some(p) => p,
            None => {
                self.tools.push(ToolSlot {
                    openai_index: oi,
                    block_index: None,
                    id: String::new(),
                    name: String::new(),
                    args: String::new(),
                });
                self.tools.len() - 1
            }
        };
        if let Some(id) = call["id"].as_str() {
            if !id.is_empty() {
                self.tools[slot_pos].id = id.to_string();
            }
        }
        if let Some(name) = call["function"]["name"].as_str() {
            if !name.is_empty() && self.tools[slot_pos].name.is_empty() {
                self.tools[slot_pos].name = name.to_string();
            }
        }

        // Defer content_block_start until the tool NAME is known — Anthropic
        // gives no way to patch a started block (some providers split id and
        // name across chunks).
        if self.tools[slot_pos].block_index.is_none() && !self.tools[slot_pos].name.is_empty() {
            // close any open text/thinking block first
            if let Some((idx, _)) = self.open_text.take() {
                out.push_str(&stop_block(idx));
            }
            let idx = self.next_block;
            self.next_block += 1;
            self.tools[slot_pos].block_index = Some(idx);
            let id = if self.tools[slot_pos].id.is_empty() {
                format!("toolu_pxy_{idx}")
            } else {
                self.tools[slot_pos].id.clone()
            };
            let name = self.tools[slot_pos].name.clone();
            out.push_str(&format_event(
                "content_block_start",
                &json!({"type": "content_block_start", "index": idx, "content_block": {
                    "type": "tool_use", "id": id, "name": name, "input": {},
                }}),
            ));
            // Flush args that arrived before the name did
            if !self.tools[slot_pos].args.is_empty() {
                out.push_str(&format_event(
                    "content_block_delta",
                    &json!({"type": "content_block_delta", "index": idx, "delta": {
                        "type": "input_json_delta",
                        "partial_json": self.tools[slot_pos].args,
                    }}),
                ));
            }
        }

        if let Some(args) = call["function"]["arguments"].as_str() {
            if !args.is_empty() {
                let slot = &mut self.tools[slot_pos];
                // Snapshot-style upstreams resend the full accumulated args
                // every chunk; emit only the new suffix.
                let delta_str = if !slot.args.is_empty() && args.starts_with(&slot.args) {
                    args[slot.args.len()..].to_string()
                } else {
                    args.to_string()
                };
                slot.args.push_str(&delta_str);
                if let (Some(idx), false) = (slot.block_index, delta_str.is_empty()) {
                    out.push_str(&format_event(
                        "content_block_delta",
                        &json!({"type": "content_block_delta", "index": idx, "delta": {
                            "type": "input_json_delta", "partial_json": delta_str,
                        }}),
                    ));
                }
            }
        }
        out
    }

    /// Close all open blocks and emit message_delta + message_stop.
    /// Splice a whole content block into the stream — the `server_tool_use` /
    /// `web_search_tool_result` pair pxy fills in itself. Any open text or
    /// thinking block is closed first so block indices stay ordered.
    pub fn emit_block(&mut self, mut block: Value) -> String {
        let mut out = String::new();
        if let Some((idx, _)) = self.open_text.take() {
            out.push_str(&stop_block(idx));
        }
        // server_tool_use streams its arguments the way tool_use does: the
        // client reads the query off input_json_delta, not off the start event.
        let input_json = (block["type"] == "server_tool_use").then(|| {
            let input = block["input"].take();
            block["input"] = json!({});
            serde_json::to_string(&input).unwrap_or_else(|_| "{}".into())
        });

        let idx = self.next_block;
        self.next_block += 1;
        out.push_str(&format_event(
            "content_block_start",
            &json!({"type": "content_block_start", "index": idx, "content_block": block}),
        ));
        if let Some(js) = input_json {
            out.push_str(&format_event(
                "content_block_delta",
                &json!({"type": "content_block_delta", "index": idx,
                        "delta": {"type": "input_json_delta", "partial_json": js}}),
            ));
        }
        out.push_str(&stop_block(idx));
        out
    }

    /// Hand over the usage counted so far and zero it. A server-tool
    /// continuation is a second upstream call: its final chunk overwrites
    /// these numbers, so they have to be banked before it starts or the first
    /// call's tokens never reach the quota windows.
    pub fn take_usage(&mut self) -> TokenUsage {
        std::mem::take(&mut self.usage)
    }

    /// Forget a finish_reason recorded before a server-tool continuation: the
    /// turn isn't over, and a stale `tool_calls` would reach the client as
    /// `stop_reason: tool_use` with no tool_use block left to answer.
    pub fn clear_finish_reason(&mut self) {
        self.finish_reason = None;
    }

    pub fn finish(&mut self) -> String {
        let mut out = String::new();
        if !self.started {
            return out;
        }
        if let Some((idx, _)) = self.open_text.take() {
            out.push_str(&stop_block(idx));
        }
        for t in &mut self.tools {
            if let Some(idx) = t.block_index.take() {
                out.push_str(&stop_block(idx));
            }
        }
        let stop_reason = map_finish_reason(self.finish_reason.as_deref());
        out.push_str(&format_event(
            "message_delta",
            &json!({"type": "message_delta",
                "delta": {"stop_reason": stop_reason, "stop_sequence": null},
                "usage": {"output_tokens": self.usage.output},
            }),
        ));
        out.push_str(&format_event("message_stop", &json!({"type": "message_stop"})));
        self.started = false; // idempotent finish
        out
    }
}

fn stop_block(index: usize) -> String {
    format_event(
        "content_block_stop",
        &json!({"type": "content_block_stop", "index": index}),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The web_search server tool becomes a real function pxy can intercept,
    /// and client tools are untouched. Every other server tool is dropped:
    /// mapped as a function it would reach the client as a call for a tool it
    /// never registered.
    #[test]
    fn server_tools_are_substituted_or_dropped() {
        let req = json!({
            "stream": true,
            "messages": [{"role": "user", "content": "hi"}],
            "tools": [
                {"type": "web_search_20250305", "name": "web_search", "max_uses": 3},
                {"type": "code_execution_20250522", "name": "code_execution"},
                {"name": "Bash", "input_schema": {"type": "object", "properties": {}}},
            ],
        });
        let out = request(&req);
        let names: Vec<&str> = out["tools"]
            .as_array()
            .unwrap()
            .iter()
            .map(|t| t["function"]["name"].as_str().unwrap())
            .collect();
        assert_eq!(names, vec![web_search::TOOL_NAME, "Bash"]);
        assert_eq!(
            out["tools"][0]["function"]["parameters"]["required"],
            json!(["query"])
        );
    }

    /// Non-streaming requests get the function too: the router runs them
    /// through the stream machinery and re-assembles the JSON, so search
    /// works either way. (It also strips the tool when it can't serve it.)
    #[test]
    fn web_search_is_offered_without_streaming() {
        let req = json!({
            "messages": [{"role": "user", "content": "hi"}],
            "tools": [{"type": "web_search_20250305", "name": "web_search"}],
        });
        assert_eq!(
            request(&req)["tools"][0]["function"]["name"],
            web_search::TOOL_NAME
        );
    }

    /// A replayed turn carries pxy's own search blocks back. The upstream has
    /// no server tools, so they land in the assistant message as prose.
    #[test]
    fn replayed_search_blocks_become_history_text() {
        let req = json!({
            "messages": [{"role": "assistant", "content": [
                {"type": "server_tool_use", "id": "s1", "name": "web_search",
                 "input": {"query": "rust"}},
                {"type": "web_search_tool_result", "tool_use_id": "s1",
                 "content": [{"type": "web_search_result", "title": "T", "url": "u"}]},
                {"type": "text", "text": "Rust is a language."},
            ]}],
        });
        let out = request(&req);
        let content = out["messages"][0]["content"].as_str().unwrap();
        assert_eq!(
            content,
            "[web search: rust]\n[web search results]\n- T (u)\nRust is a language."
        );
    }

    #[test]
    fn request_maps_system_tools_and_tool_results() {
        let req = json!({
            "model": "m", "max_tokens": 100,
            "system": [{"type": "text", "text": "sys"}],
            "messages": [
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": [
                    {"type": "text", "text": "calling"},
                    {"type": "tool_use", "id": "t1", "name": "Bash", "input": {"cmd": "ls"}}
                ]},
                {"role": "user", "content": [
                    {"type": "tool_result", "tool_use_id": "t1", "content": "ok"}
                ]}
            ],
            "tools": [{"name": "Bash", "description": "run", "input_schema": {"type": "object"}}]
        });
        let out = request(&req);
        let msgs = out["messages"].as_array().unwrap();
        assert_eq!(msgs[0]["role"], "system");
        assert_eq!(msgs[1]["role"], "user");
        assert_eq!(msgs[2]["role"], "assistant");
        assert_eq!(msgs[2]["tool_calls"][0]["function"]["name"], "Bash");
        assert_eq!(msgs[3]["role"], "tool");
        assert_eq!(msgs[3]["tool_call_id"], "t1");
        assert_eq!(out["tools"][0]["function"]["parameters"]["properties"], json!({}));
    }

    /// An image inside a tool_result (a screenshot a client tool returned)
    /// must degrade to a visible marker for text-only tool messages, not
    /// vanish — the model would otherwise answer without its screenshot.
    #[test]
    fn tool_result_images_degrade_to_a_marker() {
        let req = json!({
            "model": "m", "max_tokens": 100,
            "messages": [
                {"role": "assistant", "content": [
                    {"type": "tool_use", "id": "t1", "name": "Shot", "input": {}}
                ]},
                {"role": "user", "content": [
                    {"type": "tool_result", "tool_use_id": "t1", "content": [
                        {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "AAAA"}},
                        {"type": "text", "text": "the chart"}
                    ]}
                ]}
            ]
        });
        let out = request(&req);
        let msgs = out["messages"].as_array().unwrap();
        let tool = msgs.iter().find(|m| m["role"] == "tool").unwrap();
        let content = tool["content"].as_str().unwrap();
        assert!(content.contains("[image omitted]"), "{content}");
        assert!(content.contains("the chart"), "{content}");
    }

    #[test]
    fn repair_injects_missing_tool_response() {
        let req = json!({
            "messages": [
                {"role": "assistant", "content": [
                    {"type": "tool_use", "id": "t1", "name": "X", "input": {}}
                ]},
                {"role": "user", "content": "continue"}
            ]
        });
        let out = request(&req);
        let msgs = out["messages"].as_array().unwrap();
        assert_eq!(msgs[1]["role"], "tool");
        assert_eq!(msgs[1]["content"], "[No response received]");
        assert_eq!(msgs[2]["role"], "user");
    }

    #[test]
    fn stream_defers_tool_start_until_name() {
        let mut st = StreamState::new("m", 10);
        // first chunk: id only, no name
        let c1 = json!({"id": "x", "choices": [{"delta": {"tool_calls": [
            {"index": 0, "id": "call_1", "function": {"arguments": ""}}
        ]}}]});
        let out1 = st.on_data(&c1.to_string());
        assert!(!out1.contains("content_block_start\ndata: {\"content_block\":{\"type\":\"tool_use\""));
        // second chunk: name arrives
        let c2 = json!({"choices": [{"delta": {"tool_calls": [
            {"index": 0, "function": {"name": "Bash", "arguments": "{\"c"}}
        ]}}]});
        let out2 = st.on_data(&c2.to_string());
        assert!(out2.contains("tool_use"));
        assert!(out2.contains("\"name\":\"Bash\""));
        assert!(out2.contains("input_json_delta"));
    }

    #[test]
    fn stream_snapshot_args_emit_suffix_only() {
        let mut st = StreamState::new("m", 0);
        let c1 = json!({"choices": [{"delta": {"tool_calls": [
            {"index": 0, "id": "c1", "function": {"name": "T", "arguments": "{\"a\":"}}
        ]}}]});
        st.on_data(&c1.to_string());
        let c2 = json!({"choices": [{"delta": {"tool_calls": [
            {"index": 0, "function": {"arguments": "{\"a\":1}"}}
        ]}}]});
        let out = st.on_data(&c2.to_string());
        assert!(out.contains("\"partial_json\":\"1}\""));
    }

    #[test]
    fn stream_text_then_finish() {
        let mut st = StreamState::new("m", 5);
        let c = json!({"id": "r1", "choices": [{"delta": {"content": "hel"}}]});
        let out = st.on_data(&c.to_string());
        assert!(out.contains("message_start"));
        assert!(out.contains("text_delta"));
        let f = json!({"choices": [{"delta": {}, "finish_reason": "stop"}],
                       "usage": {"prompt_tokens": 7, "completion_tokens": 3}});
        st.on_data(&f.to_string());
        let done = st.on_data("[DONE]");
        assert!(done.contains("content_block_stop"));
        assert!(done.contains("\"stop_reason\":\"end_turn\""));
        assert!(done.contains("\"output_tokens\":3"));
        assert!(done.contains("message_stop"));
    }

    #[test]
    fn length_maps_to_max_tokens() {
        assert_eq!(map_finish_reason(Some("length")), "max_tokens");
    }
}
