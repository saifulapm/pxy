//! OpenAI Responses API (codex-cli) <-> OpenAI chat completions.
//!
//! Request: Responses `{instructions, input: [...]}` -> chat `{messages}`.
//! Response: chat SSE chunks -> Responses SSE events (`response.created`,
//! `response.output_item.added`, `response.output_text.delta`, ...), plus a
//! non-streaming body builder. Shapes verified against OmniRoute's
//! responsesTransformer/openai-responses translator (references/) — the
//! battle-tested field names live there, the subset codex actually sends
//! lives here.

use std::collections::BTreeMap;

use serde_json::{json, Map, Value};

use super::TokenUsage;

// ---------------------------------------------------------------------------
// Request: Responses -> chat completions
// ---------------------------------------------------------------------------

pub fn request(payload: &Value) -> Value {
    let mut out = Map::new();
    for key in ["model", "stream", "temperature", "top_p", "parallel_tool_calls"] {
        if !payload[key].is_null() {
            out.insert(key.into(), payload[key].clone());
        }
    }
    if let Some(m) = payload["max_output_tokens"].as_u64() {
        out.insert("max_tokens".into(), json!(m));
    }
    if let Some(effort) = payload["reasoning"]["effort"].as_str() {
        out.insert("reasoning_effort".into(), json!(effort));
    }
    match payload["text"]["format"]["type"].as_str() {
        Some("json_object") => {
            out.insert("response_format".into(), json!({"type": "json_object"}));
        }
        Some("json_schema") if !payload["text"]["format"]["schema"].is_null() => {
            let f = &payload["text"]["format"];
            out.insert(
                "response_format".into(),
                json!({"type": "json_schema", "json_schema": {
                    "name": f["name"].as_str().unwrap_or("response"),
                    "schema": f["schema"],
                }}),
            );
        }
        _ => {}
    }

    let mut messages: Vec<Value> = Vec::new();
    if let Some(instructions) = payload["instructions"].as_str() {
        if !instructions.is_empty() {
            messages.push(json!({"role": "system", "content": instructions}));
        }
    }

    // Consecutive function_call items merge into one assistant message.
    let mut pending_assistant: Option<Value> = None;
    let empty = Vec::new();
    for item in payload["input"].as_array().unwrap_or(&empty) {
        // A bare string is a user message.
        if let Some(text) = item.as_str() {
            flush(&mut messages, &mut pending_assistant);
            messages.push(json!({"role": "user", "content": text}));
            continue;
        }
        let itype = item["type"]
            .as_str()
            .unwrap_or(if item["role"].is_string() { "message" } else { "" });
        match itype {
            "message" => {
                let role = item["role"].as_str().unwrap_or("user");
                let content = convert_message_content(&item["content"]);
                if role == "assistant" {
                    // Merge with a pending tool-call assistant message.
                    match &mut pending_assistant {
                        Some(msg) if msg["content"].is_null() => msg["content"] = content,
                        Some(_) => {
                            flush(&mut messages, &mut pending_assistant);
                            pending_assistant =
                                Some(json!({"role": "assistant", "content": content}));
                        }
                        None => {
                            pending_assistant =
                                Some(json!({"role": "assistant", "content": content}))
                        }
                    }
                } else {
                    flush(&mut messages, &mut pending_assistant);
                    messages.push(json!({"role": role, "content": content}));
                }
            }
            "function_call" | "custom_tool_call" => {
                let name = item["name"].as_str().unwrap_or("").trim();
                let call_id = item["call_id"].as_str().unwrap_or("").trim();
                // Nameless/id-less calls can never match an output; drop the pair.
                if name.is_empty() || call_id.is_empty() {
                    continue;
                }
                let arguments = if itype == "custom_tool_call" {
                    // Custom tools carry a raw string input, not JSON arguments.
                    json!({"input": item["input"]}).to_string()
                } else {
                    match &item["arguments"] {
                        Value::String(s) => s.clone(),
                        Value::Null => "{}".to_string(),
                        other => other.to_string(),
                    }
                };
                let call = json!({
                    "id": call_id,
                    "type": "function",
                    "function": {"name": name, "arguments": arguments},
                });
                match &mut pending_assistant {
                    Some(msg) => {
                        if !msg["tool_calls"].is_array() {
                            msg["tool_calls"] = json!([]);
                        }
                        msg["tool_calls"].as_array_mut().unwrap().push(call);
                    }
                    None => {
                        pending_assistant = Some(
                            json!({"role": "assistant", "content": null, "tool_calls": [call]}),
                        );
                    }
                }
            }
            "function_call_output" | "custom_tool_call_output" => {
                flush(&mut messages, &mut pending_assistant);
                let mut content = tool_output_to_string(&item["output"]);
                // Unwrap codex's {"output": "...", "metadata": ...} envelope.
                if let Ok(v) = serde_json::from_str::<Value>(&content) {
                    if let Some(inner) = v["output"].as_str() {
                        content = inner.to_string();
                    }
                }
                messages.push(json!({
                    "role": "tool",
                    "tool_call_id": item["call_id"].as_str().unwrap_or(""),
                    "content": content,
                }));
            }
            // reasoning / tool_search_call / other Responses-only metadata:
            // nothing chat can represent — skip rather than fail the turn.
            _ => {}
        }
    }
    flush(&mut messages, &mut pending_assistant);

    // Drop tool results whose call never made it into an assistant message —
    // upstreams 400 on orphaned role:"tool" messages.
    let call_ids: std::collections::BTreeSet<String> = messages
        .iter()
        .flat_map(|m| m["tool_calls"].as_array().into_iter().flatten())
        .filter_map(|tc| tc["id"].as_str().map(String::from))
        .collect();
    messages.retain(|m| {
        m["role"] != "tool" || call_ids.contains(m["tool_call_id"].as_str().unwrap_or(""))
    });

    // Upstreams reject an empty messages list.
    if messages.is_empty() {
        messages.push(json!({"role": "user", "content": "..."}));
    }
    out.insert("messages".into(), json!(messages));

    if let Some(tools) = payload["tools"].as_array() {
        let converted: Vec<Value> = tools.iter().filter_map(convert_tool).collect();
        if !converted.is_empty() {
            out.insert("tools".into(), json!(converted));
        }
    }
    match &payload["tool_choice"] {
        Value::String(s) => {
            out.insert("tool_choice".into(), json!(s));
        }
        v if v.is_object() => {
            if v["type"] == "function" && v["name"].is_string() {
                out.insert(
                    "tool_choice".into(),
                    json!({"type": "function", "function": {"name": v["name"]}}),
                );
            } else if v["type"] == "local_shell" {
                out.insert(
                    "tool_choice".into(),
                    json!({"type": "function", "function": {"name": "shell"}}),
                );
            }
            // Other object forms (allowed_tools, hosted tools) are dropped:
            // "auto" behaviour is the safe degradation.
        }
        _ => {}
    }
    Value::Object(out)
}

fn flush(messages: &mut Vec<Value>, pending: &mut Option<Value>) {
    if let Some(msg) = pending.take() {
        messages.push(msg);
    }
}

/// Responses message content parts -> chat content parts.
fn convert_message_content(content: &Value) -> Value {
    let Some(parts) = content.as_array() else {
        return content.clone();
    };
    let converted: Vec<Value> = parts
        .iter()
        .filter_map(|p| {
            if let Some(s) = p.as_str() {
                return Some(json!({"type": "text", "text": s}));
            }
            match p["type"].as_str() {
                Some("input_text") | Some("output_text") | Some("text") => {
                    Some(json!({"type": "text", "text": p["text"].as_str().unwrap_or("")}))
                }
                Some("refusal") => {
                    Some(json!({"type": "text", "text": p["refusal"].as_str().unwrap_or("")}))
                }
                Some("input_image") => Some(json!({
                    "type": "image_url",
                    "image_url": {"url": p["image_url"].as_str().unwrap_or("")},
                })),
                _ => None,
            }
        })
        .collect();
    json!(converted)
}

/// Tool output may be a string or an array of content parts.
fn tool_output_to_string(output: &Value) -> String {
    if let Some(s) = output.as_str() {
        return s.to_string();
    }
    let Some(parts) = output.as_array() else {
        return output.to_string();
    };
    parts
        .iter()
        .map(|p| match p["type"].as_str() {
            Some("input_text") | Some("output_text") => p["text"].as_str().unwrap_or("").to_string(),
            Some("input_image") => "[Image omitted]".to_string(),
            _ => p.to_string(),
        })
        .collect::<Vec<_>>()
        .join("\n")
}

/// Responses tool declaration -> chat tool. Codex's flat `{type:"function",
/// name, parameters}` shape plus the built-ins it injects; hosted tools with
/// no chat equivalent return None.
fn convert_tool(tool: &Value) -> Option<Value> {
    // Already chat-shaped.
    if tool["function"].is_object() {
        return Some(tool.clone());
    }
    match tool["type"].as_str() {
        Some("function") => {
            let name = tool["name"].as_str()?.trim();
            if name.is_empty() {
                return None;
            }
            let mut f = Map::new();
            f.insert("name".into(), json!(name));
            if let Some(d) = tool["description"].as_str() {
                f.insert("description".into(), json!(d));
            }
            f.insert(
                "parameters".into(),
                if tool["parameters"].is_null() {
                    json!({"type": "object", "properties": {}})
                } else {
                    tool["parameters"].clone()
                },
            );
            if !tool["strict"].is_null() {
                f.insert("strict".into(), tool["strict"].clone());
            }
            Some(json!({"type": "function", "function": f}))
        }
        // Custom/freeform tools (codex apply_patch): model must produce
        // { input: string }.
        Some("custom") => {
            let name = tool["name"].as_str()?.trim();
            if name.is_empty() {
                return None;
            }
            Some(json!({"type": "function", "function": {
                "name": name,
                "description": tool["description"].as_str().unwrap_or(""),
                "parameters": {
                    "type": "object",
                    "properties": {"input": {"type": "string"}},
                    "required": ["input"],
                    "additionalProperties": false,
                },
            }}))
        }
        // Codex injects local_shell for shell execution; map to a plain
        // function the response side emits back as function_call "shell".
        Some("local_shell") => Some(json!({"type": "function", "function": {
            "name": "shell",
            "description": "Run a shell command and return its output.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "array", "items": {"type": "string"},
                                "description": "Command and arguments to execute."},
                    "workdir": {"type": "string", "description": "Working directory."},
                    "timeout_ms": {"type": "number", "description": "Timeout in milliseconds."},
                },
                "required": ["command"],
            },
        }})),
        // web_search / image_generation / other hosted tools: no equivalent.
        _ => None,
    }
}

// ---------------------------------------------------------------------------
// Response: chat -> Responses (non-streaming)
// ---------------------------------------------------------------------------

pub fn response(chat: &Value, model_id: &str) -> Value {
    let message = &chat["choices"][0]["message"];
    let id = format!("resp_{}", chat["id"].as_str().unwrap_or("pxy"));
    let mut output: Vec<Value> = Vec::new();
    if let Some(r) = message["reasoning_content"].as_str() {
        if !r.is_empty() {
            output.push(json!({
                "id": format!("rs_{id}"),
                "type": "reasoning",
                "summary": [{"type": "summary_text", "text": r}],
            }));
        }
    }
    if let Some(text) = message["content"].as_str() {
        if !text.is_empty() {
            output.push(json!({
                "id": format!("msg_{id}"),
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "annotations": [], "logprobs": [], "text": text}],
            }));
        }
    }
    for tc in message["tool_calls"].as_array().into_iter().flatten() {
        output.push(json!({
            "id": format!("fc_{}", tc["id"].as_str().unwrap_or("call")),
            "type": "function_call",
            "call_id": tc["id"],
            "name": tc["function"]["name"],
            "arguments": tc["function"]["arguments"],
        }));
    }
    json!({
        "id": id,
        "object": "response",
        "created_at": chat["created"].as_i64().unwrap_or(0),
        "status": "completed",
        "background": false,
        "error": null,
        "model": model_id,
        "output": output,
        "usage": usage_from_chat(&chat["usage"]),
    })
}

fn usage_from_chat(usage: &Value) -> Value {
    let input = usage["prompt_tokens"].as_u64().unwrap_or(0);
    let output = usage["completion_tokens"].as_u64().unwrap_or(0);
    json!({
        "input_tokens": input,
        "input_tokens_details": {
            "cached_tokens": usage["prompt_tokens_details"]["cached_tokens"].as_u64().unwrap_or(0),
        },
        "output_tokens": output,
        "output_tokens_details": {
            "reasoning_tokens":
                usage["completion_tokens_details"]["reasoning_tokens"].as_u64().unwrap_or(0),
        },
        "total_tokens": usage["total_tokens"].as_u64().unwrap_or(input + output),
    })
}

// ---------------------------------------------------------------------------
// Response: chat SSE chunks -> Responses SSE events (streaming)
// ---------------------------------------------------------------------------

#[derive(Default)]
struct ToolSlot {
    output_index: u64,
    call_id: String,
    name: String,
    args: String,
    added: bool,
    done: bool,
}

pub struct StreamState {
    seq: u64,
    response_id: String,
    created: i64,
    started: bool,
    next_index: u64,
    // (output_index, buffered text, closed)
    reasoning: Option<(u64, String, bool)>,
    message: Option<(u64, String, bool)>,
    tools: BTreeMap<u64, ToolSlot>,
    completed_items: Vec<Value>,
    chat_usage: Value,
    pub usage: TokenUsage,
    awaiting_usage: bool,
    completed_sent: bool,
}

impl StreamState {
    pub fn new(created: i64) -> Self {
        Self {
            seq: 0,
            response_id: "resp_pxy".into(),
            created,
            started: false,
            next_index: 0,
            reasoning: None,
            message: None,
            tools: BTreeMap::new(),
            completed_items: Vec::new(),
            chat_usage: Value::Null,
            usage: TokenUsage::default(),
            awaiting_usage: false,
            completed_sent: false,
        }
    }

    fn emit(&mut self, event: &str, mut data: Value) -> String {
        self.seq += 1;
        data["type"] = json!(event);
        data["sequence_number"] = json!(self.seq);
        format!("event: {event}\ndata: {data}\n\n")
    }

    fn response_skeleton(&self, status: &str) -> Value {
        json!({
            "id": self.response_id,
            "object": "response",
            "created_at": self.created,
            "status": status,
            "background": false,
            "error": null,
            "output": [],
        })
    }

    /// One chat SSE `data:` payload in, Responses SSE text out.
    pub fn on_data(&mut self, data: &str) -> String {
        if data.trim() == "[DONE]" {
            return String::new(); // finish() emits our own terminator
        }
        let Ok(chunk) = serde_json::from_str::<Value>(data) else {
            return String::new();
        };
        let mut out = String::new();

        if chunk["usage"].is_object() {
            self.chat_usage = chunk["usage"].clone();
            self.usage = TokenUsage::from_openai(&chunk["usage"]);
        }

        let choices = chunk["choices"].as_array().cloned().unwrap_or_default();
        if choices.is_empty() {
            // Trailing usage-only chunk after finish_reason deferred completion.
            if self.awaiting_usage && !self.completed_sent {
                out.push_str(&self.send_completed());
            }
            return out;
        }

        if !self.started {
            self.started = true;
            if let Some(id) = chunk["id"].as_str() {
                if !id.is_empty() {
                    self.response_id = format!("resp_{id}");
                }
            }
            let created = self.response_skeleton("in_progress");
            out.push_str(&self.emit("response.created", json!({"response": created})));
            let in_progress = self.response_skeleton("in_progress");
            out.push_str(&self.emit("response.in_progress", json!({"response": in_progress})));
        }

        let choice = &choices[0];
        let delta = &choice["delta"];

        if let Some(r) = delta["reasoning_content"].as_str() {
            if !r.is_empty() {
                out.push_str(&self.reasoning_delta(r));
            }
        }

        if let Some(text) = delta["content"].as_str() {
            let mut text = text.to_string();
            if self.message.as_ref().map(|(_, buf, _)| buf.is_empty()).unwrap_or(true) {
                text = text.trim_start().to_string();
            }
            if !text.is_empty() {
                out.push_str(&self.close_reasoning());
                out.push_str(&self.text_delta(&text));
            }
        }

        if let Some(tcs) = delta["tool_calls"].as_array().cloned() {
            out.push_str(&self.close_reasoning());
            out.push_str(&self.close_message());
            for tc in &tcs {
                out.push_str(&self.tool_call_delta(tc));
            }
        }

        if choice["finish_reason"].is_string() {
            out.push_str(&self.close_all());
            if self.chat_usage.is_object() {
                out.push_str(&self.send_completed());
            } else {
                // Wait for the trailing usage-only chunk (include_usage).
                self.awaiting_usage = true;
            }
        }
        out
    }

    /// End of upstream stream: close whatever is open and terminate.
    pub fn finish(&mut self) -> String {
        let mut out = self.close_all();
        if !self.completed_sent {
            out.push_str(&self.send_completed());
        }
        out.push_str("data: [DONE]\n\n");
        out
    }

    fn reasoning_delta(&mut self, text: &str) -> String {
        let mut out = String::new();
        if self.reasoning.is_none() {
            let idx = self.next_index;
            self.next_index += 1;
            self.reasoning = Some((idx, String::new(), false));
            let id = self.reasoning_id();
            out.push_str(&self.emit(
                "response.output_item.added",
                json!({"output_index": idx, "item": {"id": id, "type": "reasoning", "summary": []}}),
            ));
            out.push_str(&self.emit(
                "response.reasoning_summary_part.added",
                json!({"item_id": id, "output_index": idx, "summary_index": 0,
                       "part": {"type": "summary_text", "text": ""}}),
            ));
        }
        let (idx, buf, _) = self.reasoning.as_mut().unwrap();
        let idx = *idx;
        buf.push_str(text);
        let id = self.reasoning_id();
        out.push_str(&self.emit(
            "response.reasoning_summary_text.delta",
            json!({"item_id": id, "output_index": idx, "summary_index": 0, "delta": text}),
        ));
        out
    }

    fn reasoning_id(&self) -> String {
        format!("rs_{}", self.response_id)
    }
    fn message_id(&self) -> String {
        format!("msg_{}", self.response_id)
    }

    fn close_reasoning(&mut self) -> String {
        let Some((idx, buf, done)) = self.reasoning.clone() else { return String::new() };
        if done {
            return String::new();
        }
        self.reasoning = Some((idx, buf.clone(), true));
        let id = self.reasoning_id();
        let mut out = String::new();
        out.push_str(&self.emit(
            "response.reasoning_summary_text.done",
            json!({"item_id": id, "output_index": idx, "summary_index": 0, "text": buf}),
        ));
        out.push_str(&self.emit(
            "response.reasoning_summary_part.done",
            json!({"item_id": id, "output_index": idx, "summary_index": 0,
                   "part": {"type": "summary_text", "text": buf}}),
        ));
        let item = json!({"id": id, "type": "reasoning",
                          "summary": [{"type": "summary_text", "text": buf}]});
        out.push_str(&self.emit(
            "response.output_item.done",
            json!({"output_index": idx, "item": item}),
        ));
        self.completed_items.push(item);
        out
    }

    fn text_delta(&mut self, text: &str) -> String {
        let mut out = String::new();
        if self.message.is_none() {
            let idx = self.next_index;
            self.next_index += 1;
            self.message = Some((idx, String::new(), false));
            let id = self.message_id();
            out.push_str(&self.emit(
                "response.output_item.added",
                json!({"output_index": idx,
                       "item": {"id": id, "type": "message", "content": [], "role": "assistant"}}),
            ));
            out.push_str(&self.emit(
                "response.content_part.added",
                json!({"item_id": id, "output_index": idx, "content_index": 0,
                       "part": {"type": "output_text", "annotations": [], "logprobs": [], "text": ""}}),
            ));
        }
        let (idx, buf, _) = self.message.as_mut().unwrap();
        let idx = *idx;
        buf.push_str(text);
        let id = self.message_id();
        out.push_str(&self.emit(
            "response.output_text.delta",
            json!({"item_id": id, "output_index": idx, "content_index": 0,
                   "delta": text, "logprobs": []}),
        ));
        out
    }

    fn close_message(&mut self) -> String {
        let Some((idx, buf, done)) = self.message.clone() else { return String::new() };
        if done {
            return String::new();
        }
        self.message = Some((idx, buf.clone(), true));
        let id = self.message_id();
        let mut out = String::new();
        out.push_str(&self.emit(
            "response.output_text.done",
            json!({"item_id": id, "output_index": idx, "content_index": 0,
                   "text": buf, "logprobs": []}),
        ));
        out.push_str(&self.emit(
            "response.content_part.done",
            json!({"item_id": id, "output_index": idx, "content_index": 0,
                   "part": {"type": "output_text", "annotations": [], "logprobs": [], "text": buf}}),
        ));
        let item = json!({"id": id, "type": "message", "role": "assistant",
                          "content": [{"type": "output_text", "annotations": [],
                                       "logprobs": [], "text": buf}]});
        out.push_str(&self.emit(
            "response.output_item.done",
            json!({"output_index": idx, "item": item}),
        ));
        self.completed_items.push(item);
        out
    }

    fn tool_call_delta(&mut self, tc: &Value) -> String {
        let mut out = String::new();
        let key = tc["index"].as_u64().unwrap_or(0);
        if !self.tools.contains_key(&key) {
            let idx = self.next_index;
            self.next_index += 1;
            self.tools.insert(key, ToolSlot { output_index: idx, ..Default::default() });
        }
        // Two-phase borrow: collect what to do, then emit (emit needs &mut self).
        let (added_now, delta_to_emit, item_id, output_index, added_item);
        {
            let slot = self.tools.get_mut(&key).unwrap();
            if let Some(id) = tc["id"].as_str() {
                if slot.call_id.is_empty() {
                    slot.call_id = id.to_string();
                }
            }
            if let Some(name) = tc["function"]["name"].as_str() {
                if slot.name.is_empty() {
                    slot.name = name.to_string();
                }
            }
            let arg_delta = tc["function"]["arguments"].as_str().unwrap_or("");
            // The item is announced once both id and name are known; args that
            // arrived earlier ride along as the first delta.
            if !slot.added && !slot.call_id.is_empty() && !slot.name.is_empty() {
                slot.added = true;
                added_now = true;
                slot.args.push_str(arg_delta);
                delta_to_emit = slot.args.clone();
            } else {
                added_now = false;
                slot.args.push_str(arg_delta);
                delta_to_emit = if slot.added { arg_delta.to_string() } else { String::new() };
            }
            item_id = format!("fc_{}", slot.call_id);
            output_index = slot.output_index;
            added_item = json!({"id": item_id, "type": "function_call", "arguments": "",
                                "call_id": slot.call_id, "name": slot.name});
        }
        if added_now {
            out.push_str(&self.emit(
                "response.output_item.added",
                json!({"output_index": output_index, "item": added_item}),
            ));
        }
        if !delta_to_emit.is_empty() {
            out.push_str(&self.emit(
                "response.function_call_arguments.delta",
                json!({"item_id": item_id, "output_index": output_index, "delta": delta_to_emit}),
            ));
        }
        out
    }

    fn close_tools(&mut self) -> String {
        let mut out = String::new();
        let keys: Vec<u64> = self.tools.keys().copied().collect();
        for key in keys {
            let slot = self.tools.get(&key).unwrap();
            if slot.done || slot.call_id.is_empty() {
                continue;
            }
            let (idx, call_id, name) =
                (slot.output_index, slot.call_id.clone(), slot.name.clone());
            let args = if slot.args.is_empty() { "{}".to_string() } else { slot.args.clone() };
            let item_id = format!("fc_{call_id}");
            out.push_str(&self.emit(
                "response.function_call_arguments.done",
                json!({"item_id": item_id, "output_index": idx, "arguments": args}),
            ));
            let item = json!({"id": item_id, "type": "function_call", "arguments": args,
                              "call_id": call_id, "name": name});
            out.push_str(&self.emit(
                "response.output_item.done",
                json!({"output_index": idx, "item": item}),
            ));
            self.completed_items.push(item);
            self.tools.get_mut(&key).unwrap().done = true;
        }
        out
    }

    fn close_all(&mut self) -> String {
        let mut out = String::new();
        out.push_str(&self.close_message());
        out.push_str(&self.close_reasoning());
        out.push_str(&self.close_tools());
        out
    }

    fn send_completed(&mut self) -> String {
        if self.completed_sent {
            return String::new();
        }
        self.completed_sent = true;
        let mut response = self.response_skeleton("completed");
        response["output"] = json!(self.completed_items);
        if self.chat_usage.is_object() {
            response["usage"] = usage_from_chat(&self.chat_usage);
        }
        self.emit("response.completed", json!({"response": response}))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn request_converts_instructions_input_and_tools() {
        let payload = json!({
            "model": "auto",
            "stream": true,
            "instructions": "Be brief.",
            "max_output_tokens": 42,
            "reasoning": {"effort": "low"},
            "input": [
                {"type": "message", "role": "user",
                 "content": [{"type": "input_text", "text": "hi"}]},
                {"type": "reasoning", "summary": []},
                {"type": "function_call", "call_id": "c1", "name": "get",
                 "arguments": "{\"a\":1}"},
                {"type": "function_call_output", "call_id": "c1",
                 "output": "{\"output\":\"result\",\"metadata\":{}}"},
                {"type": "message", "role": "assistant",
                 "content": [{"type": "output_text", "text": "done"}]},
            ],
            "tools": [
                {"type": "function", "name": "get", "description": "d",
                 "parameters": {"type": "object", "properties": {}}},
                {"type": "local_shell"},
                {"type": "web_search"},
            ],
        });
        let chat = request(&payload);
        assert_eq!(chat["model"], "auto");
        assert_eq!(chat["max_tokens"], 42);
        assert_eq!(chat["reasoning_effort"], "low");
        let msgs = chat["messages"].as_array().unwrap();
        assert_eq!(msgs[0]["role"], "system");
        assert_eq!(msgs[0]["content"], "Be brief.");
        assert_eq!(msgs[1]["content"][0]["text"], "hi");
        assert_eq!(msgs[2]["tool_calls"][0]["id"], "c1");
        assert_eq!(msgs[3]["role"], "tool");
        assert_eq!(msgs[3]["content"], "result"); // envelope unwrapped
        assert_eq!(msgs[4]["role"], "assistant");
        let tools = chat["tools"].as_array().unwrap();
        assert_eq!(tools.len(), 2); // web_search dropped
        assert_eq!(tools[0]["function"]["name"], "get");
        assert_eq!(tools[1]["function"]["name"], "shell");
    }

    #[test]
    fn request_drops_orphan_tool_results() {
        let payload = json!({
            "input": [
                {"type": "function_call_output", "call_id": "ghost", "output": "x"},
            ],
        });
        let chat = request(&payload);
        let msgs = chat["messages"].as_array().unwrap();
        assert_eq!(msgs.len(), 1);
        assert_eq!(msgs[0]["role"], "user"); // placeholder, orphan dropped
    }

    #[test]
    fn nonstreaming_response_shape() {
        let chat = json!({
            "id": "abc", "created": 7,
            "choices": [{"message": {
                "role": "assistant", "content": "hello",
                "tool_calls": [{"id": "c9", "type": "function",
                                "function": {"name": "get", "arguments": "{}"}}],
            }}],
            "usage": {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8},
        });
        let v = response(&chat, "prov/model");
        assert_eq!(v["object"], "response");
        assert_eq!(v["status"], "completed");
        assert_eq!(v["output"][0]["type"], "message");
        assert_eq!(v["output"][0]["content"][0]["text"], "hello");
        assert_eq!(v["output"][1]["type"], "function_call");
        assert_eq!(v["output"][1]["call_id"], "c9");
        assert_eq!(v["usage"]["input_tokens"], 5);
        assert_eq!(v["usage"]["total_tokens"], 8);
    }

    #[test]
    fn stream_text_lifecycle() {
        let mut st = StreamState::new(1);
        let a = st.on_data(r#"{"id":"x1","choices":[{"index":0,"delta":{"role":"assistant","content":"Hel"}}]}"#);
        assert!(a.contains("response.created"));
        assert!(a.contains("response.output_item.added"));
        assert!(a.contains("response.content_part.added"));
        assert!(a.contains(r#""delta":"Hel""#));
        let b = st.on_data(r#"{"choices":[{"index":0,"delta":{"content":"lo"},"finish_reason":"stop"}]}"#);
        assert!(b.contains("response.output_text.done"));
        assert!(b.contains(r#""text":"Hello""#));
        // finish deferred: waiting for trailing usage chunk
        assert!(!b.contains("response.completed"));
        let c = st.on_data(r#"{"choices":[],"usage":{"prompt_tokens":2,"completion_tokens":1}}"#);
        assert!(c.contains("response.completed"));
        assert!(c.contains(r#""input_tokens":2"#));
        let d = st.finish();
        assert!(d.contains("[DONE]"));
        assert!(!d.contains("response.completed")); // not duplicated
        assert_eq!(st.usage.input, 2);
    }

    #[test]
    fn stream_tool_call_lifecycle() {
        let mut st = StreamState::new(1);
        let a = st.on_data(r#"{"id":"x","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"c1","type":"function","function":{"name":"get","arguments":"{\"a\":"}}]}}]}"#);
        assert!(a.contains("response.output_item.added"));
        assert!(a.contains(r#""call_id":"c1""#));
        assert!(a.contains("response.function_call_arguments.delta"));
        let b = st.on_data(r#"{"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"1}"}}]},"finish_reason":"tool_calls"}]}"#);
        assert!(b.contains("response.function_call_arguments.done"));
        assert!(b.contains(r#""arguments":"{\"a\":1}""#));
        let c = st.finish();
        assert!(c.contains("response.completed"));
        assert!(c.contains("[DONE]"));
    }

    #[test]
    fn stream_reasoning_then_text() {
        let mut st = StreamState::new(1);
        let a = st.on_data(r#"{"id":"x","choices":[{"index":0,"delta":{"reasoning_content":"think"}}]}"#);
        assert!(a.contains("response.reasoning_summary_text.delta"));
        let b = st.on_data(r#"{"choices":[{"index":0,"delta":{"content":"answer"},"finish_reason":null}]}"#);
        // reasoning closes before the message opens
        assert!(b.contains("response.reasoning_summary_text.done"));
        assert!(b.contains(r#""delta":"answer""#));
        let c = st.finish();
        // completed output holds reasoning item then message item
        let completed_line = c.lines().find(|l| l.contains("response.completed")).unwrap_or("");
        assert!(completed_line.is_empty() || c.contains("reasoning"));
    }
}
