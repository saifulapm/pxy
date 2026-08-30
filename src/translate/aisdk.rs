//! Vercel AI SDK **LanguageModel specification v4** — the dialect the `fx`
//! agent speaks to its gateway (`POST /v3/ai/language-model`). Neither
//! OpenAI nor Anthropic: `prompt[]` instead of `messages[]`, the model id in
//! a header rather than the body, streaming selected by header, and a
//! line-oriented SSE vocabulary of typed parts.
//!
//! Same shape as `translate/responses.rs` (the codex path): translate to
//! chat-completions, run the normal router, translate the OpenAI stream back.
//! Contract extracted from the fx source (references/fx, Zig) — every rule
//! below is enforced by fx's own parser:
//!
//! - SSE lines need the space in `data: ` (`data:{…}` is silently dropped).
//! - `finishReason` must be an OBJECT with a `unified` string from a closed
//!   set; an unknown value aborts fx's whole stream.
//! - usage is nested (`inputTokens.total`), not flat.
//! - `tool-call.input` may be raw JSON; no `tool-input-*` preamble needed.

use serde_json::{json, Map, Value};

use super::sse::format_data;

// ---------------------------------------------------------------------------
// Request: AI SDK v4 -> OpenAI chat completions
// ---------------------------------------------------------------------------

/// `model` and `stream` come from headers, so the caller passes them in.
pub fn request(body: &Value, model: &str, stream: bool) -> Value {
    let mut messages: Vec<Value> = Vec::new();

    for msg in body["prompt"].as_array().unwrap_or(&Vec::new()) {
        let role = msg["role"].as_str().unwrap_or("user");
        match role {
            // System content is a bare string in this dialect.
            "system" => messages.push(json!({
                "role": "system",
                "content": msg["content"].as_str().unwrap_or_default(),
            })),
            "assistant" => {
                let mut text = String::new();
                let mut tool_calls: Vec<Value> = Vec::new();
                for part in msg["content"].as_array().unwrap_or(&Vec::new()) {
                    match part["type"].as_str() {
                        Some("text") => text.push_str(part["text"].as_str().unwrap_or_default()),
                        Some("tool-call") => tool_calls.push(json!({
                            "id": part["toolCallId"],
                            "type": "function",
                            "function": {
                                "name": part["toolName"],
                                // fx sends `input` as raw JSON; OpenAI wants a string.
                                "arguments": match &part["input"] {
                                    Value::String(s) => s.clone(),
                                    other => other.to_string(),
                                },
                            },
                        })),
                        _ => {}
                    }
                }
                let mut m = Map::new();
                m.insert("role".into(), json!("assistant"));
                m.insert(
                    "content".into(),
                    if text.is_empty() { Value::Null } else { json!(text) },
                );
                if !tool_calls.is_empty() {
                    m.insert("tool_calls".into(), Value::Array(tool_calls));
                }
                messages.push(Value::Object(m));
            }
            // Tool results arrive in their own turn; each becomes an OpenAI
            // `role: "tool"` message.
            "tool" => {
                for part in msg["content"].as_array().unwrap_or(&Vec::new()) {
                    if part["type"] != "tool-result" {
                        continue;
                    }
                    let out = &part["output"];
                    let content = match out["type"].as_str() {
                        Some("text") | Some("error-text") => {
                            out["value"].as_str().unwrap_or_default().to_string()
                        }
                        _ => out["value"].to_string(),
                    };
                    messages.push(json!({
                        "role": "tool",
                        "tool_call_id": part["toolCallId"],
                        "content": content,
                    }));
                }
            }
            _ => {
                // user
                let mut parts: Vec<Value> = Vec::new();
                let mut plain = String::new();
                for part in msg["content"].as_array().unwrap_or(&Vec::new()) {
                    match part["type"].as_str() {
                        Some("text") => {
                            let t = part["text"].as_str().unwrap_or_default();
                            plain.push_str(t);
                            parts.push(json!({"type": "text", "text": t}));
                        }
                        Some("file") => {
                            let media = part["mediaType"].as_str().unwrap_or("image/png");
                            let data = part["data"].as_str().unwrap_or_default();
                            parts.push(json!({
                                "type": "image_url",
                                "image_url": {"url": format!("data:{media};base64,{data}")},
                            }));
                        }
                        _ => {}
                    }
                }
                let has_media = parts.iter().any(|p| p["type"] != "text");
                messages.push(json!({
                    "role": "user",
                    // Keep the simple string shape when there's no media: more
                    // free providers accept it than accept part arrays.
                    "content": if has_media { Value::Array(parts) } else { json!(plain) },
                }));
            }
        }
    }

    let mut out = Map::new();
    out.insert("model".into(), json!(model));
    out.insert("messages".into(), Value::Array(messages));
    out.insert("stream".into(), json!(stream));

    if let Some(max) = body["maxOutputTokens"].as_u64() {
        out.insert("max_tokens".into(), json!(max));
    }
    // `inputSchema` (flat) -> `function.parameters` (nested).
    if let Some(tools) = body["tools"].as_array() {
        let mapped: Vec<Value> = tools
            .iter()
            .filter(|t| t["type"] != "provider") // gateway-executed; we can't run them
            .filter_map(|t| {
                let name = t["name"].as_str()?;
                Some(json!({
                    "type": "function",
                    "function": {
                        "name": name,
                        "description": t["description"].as_str().unwrap_or(""),
                        "parameters": if t["inputSchema"].is_object() {
                            t["inputSchema"].clone()
                        } else {
                            json!({"type": "object", "properties": {}})
                        },
                    },
                }))
            })
            .collect();
        if !mapped.is_empty() {
            out.insert("tools".into(), Value::Array(mapped));
        }
    }
    match body["toolChoice"]["type"].as_str() {
        Some("none") => {
            out.insert("tool_choice".into(), json!("none"));
        }
        Some("required") => {
            out.insert("tool_choice".into(), json!("required"));
        }
        Some("tool") => {
            if let Some(name) = body["toolChoice"]["toolName"].as_str() {
                out.insert(
                    "tool_choice".into(),
                    json!({"type": "function", "function": {"name": name}}),
                );
            }
        }
        _ => {}
    }
    if body["responseFormat"]["type"].as_str() == Some("json") {
        let schema = &body["responseFormat"]["schema"];
        if schema.is_object() {
            out.insert(
                "response_format".into(),
                json!({"type": "json_schema", "json_schema": {
                    "name": body["responseFormat"]["name"].as_str().unwrap_or("response"),
                    "schema": schema,
                }}),
            );
        } else {
            out.insert("response_format".into(), json!({"type": "json_object"}));
        }
    }
    Value::Object(out)
}

/// OpenAI finish_reason -> the CLOSED set fx accepts. An unrecognized value
/// aborts fx's stream entirely, so anything unknown maps to "other".
///
/// `has_tool_calls` is load-bearing, not cosmetic: fx's
/// classifyProviderCompletion treats stop/other TOGETHER WITH tool calls as
/// an invalid completion and kills the agentic turn. Providers that emit
/// native tool_calls with finish_reason "stop" (common on free pools) would
/// wedge every tool-using turn without this.
fn unified_finish(reason: Option<&str>, has_tool_calls: bool) -> &'static str {
    let mapped = match reason {
        Some("stop") => "stop",
        Some("length") => "length",
        Some("content_filter") => "content-filter",
        Some("tool_calls") | Some("function_call") => "tool-calls",
        Some("error") => "error",
        None => "stop",
        _ => "other",
    };
    // length/content-filter are truthful about truncation; everything else
    // must agree with the tool calls actually emitted.
    if has_tool_calls && matches!(mapped, "stop" | "other" | "error") {
        return "tool-calls";
    }
    mapped
}

// ---------------------------------------------------------------------------
// Non-streaming response: OpenAI -> AI SDK
// ---------------------------------------------------------------------------

/// fx's non-streaming path parses a plain OpenAI body, with two hard
/// requirements: `arguments` must be a JSON *string*, and `finish_reason`
/// must be in fx's legacy closed set — anything else and fx rejects the
/// whole response as malformed.
///
/// Indexing is fallible throughout: a 200 body with no `choices` is a real
/// shape in the free-provider pool, and serde_json's mut-index PANICS on it.
pub fn response(openai: &Value) -> Value {
    let mut out = openai.clone();
    let Some(choice) = out.get_mut("choices").and_then(|c| c.get_mut(0)) else {
        return out;
    };
    let mut has_calls = false;
    if let Some(calls) = choice
        .get_mut("message")
        .and_then(|m| m.get_mut("tool_calls"))
        .and_then(Value::as_array_mut)
    {
        has_calls = !calls.is_empty();
        for call in calls {
            if let Some(args) = call.get_mut("function").and_then(|f| f.get_mut("arguments"))
                && !args.is_string()
            {
                *args = json!(args.to_string());
            }
        }
    }
    // Same closed-set normalization the streaming path does, and the same
    // tool-call coherence rule (fx errors on stop+tool_calls).
    let reason = choice["finish_reason"].as_str().map(String::from);
    choice["finish_reason"] = json!(legacy_finish(reason.as_deref(), has_calls));
    out
}

/// fx's non-streaming parser accepts only {stop, length, content_filter,
/// content-filter, tool_calls, tool-calls, error, other}.
fn legacy_finish(reason: Option<&str>, has_tool_calls: bool) -> &'static str {
    match unified_finish(reason, has_tool_calls) {
        "content-filter" => "content_filter",
        "tool-calls" => "tool_calls",
        other => other,
    }
}

// ---------------------------------------------------------------------------
// Streaming: OpenAI SSE -> AI SDK SSE
// ---------------------------------------------------------------------------

#[derive(Default)]
pub struct StreamState {
    model: String,
    /// openai tool index -> (id, name, accumulated argument text)
    tools: Vec<(u64, String, String, String)>,
    finish_reason: Option<String>,
    input_tokens: u64,
    output_tokens: u64,
    started: bool,
    finished: bool,
    /// An upstream error event or a broken transport. Must surface as
    /// `unified: "error"`, not a clean "stop" — fx retries on the former and
    /// silently renders an empty successful turn on the latter.
    failed: bool,
}

impl StreamState {
    pub fn new(model: &str) -> Self {
        Self { model: model.to_string(), ..Default::default() }
    }

    /// Translate one OpenAI SSE data payload into AI SDK events.
    pub fn on_data(&mut self, data: &str) -> String {
        let trimmed = data.trim();
        if trimmed == "[DONE]" {
            return self.finish();
        }
        let Ok(v) = serde_json::from_str::<Value>(trimmed) else {
            return String::new();
        };
        let mut out = String::new();

        if !self.started {
            self.started = true;
            out.push_str(&format_data(&json!({
                "type": "response-metadata",
                "modelId": self.model,
                // Strict UTC ISO-8601 with Z; fx REJECTS "+00:00".
                "timestamp": jiff::Timestamp::now().to_string(),
            })));
        }

        // Upstream error surfaced inside a 200 stream.
        if !v["error"].is_null() {
            self.failed = true;
            out.push_str(&format_data(&json!({
                "type": "error",
                "error": v["error"].clone(),
            })));
            return out;
        }

        if let Some(u) = v["usage"].as_object() {
            if let Some(n) = u.get("prompt_tokens").and_then(Value::as_u64) {
                self.input_tokens = n;
            }
            if let Some(n) = u.get("completion_tokens").and_then(Value::as_u64) {
                self.output_tokens = n;
            }
        }

        let choice = &v["choices"][0];
        let delta = &choice["delta"];
        if let Some(text) = delta["content"].as_str()
            && !text.is_empty()
        {
            out.push_str(&format_data(&json!({
                "type": "text-delta", "id": "text", "delta": text,
            })));
        }
        for key in ["reasoning_content", "reasoning"] {
            if let Some(text) = delta[key].as_str()
                && !text.is_empty()
            {
                out.push_str(&format_data(&json!({
                    "type": "reasoning-delta", "id": "reasoning", "delta": text,
                })));
            }
        }
        if let Some(calls) = delta["tool_calls"].as_array() {
            for call in calls {
                let idx = call["index"].as_u64().unwrap_or(0);
                let slot = match self.tools.iter().position(|(i, ..)| *i == idx) {
                    Some(p) => p,
                    None => {
                        self.tools.push((idx, String::new(), String::new(), String::new()));
                        self.tools.len() - 1
                    }
                };
                if let Some(id) = call["id"].as_str()
                    && !id.is_empty()
                {
                    self.tools[slot].1 = id.to_string();
                }
                if let Some(name) = call["function"]["name"].as_str()
                    && !name.is_empty()
                {
                    self.tools[slot].2 = name.to_string();
                }
                if let Some(args) = call["function"]["arguments"].as_str() {
                    self.tools[slot].3.push_str(args);
                }
            }
        }
        if let Some(reason) = choice["finish_reason"].as_str() {
            self.finish_reason = Some(reason.to_string());
        }
        out
    }

    /// Mark the stream as failed (transport died mid-stream): the finish
    /// event must say so rather than claim a clean stop.
    pub fn fail(&mut self) {
        self.failed = true;
    }

    /// Emit buffered tool calls, the finish event and the terminator. fx also
    /// accepts a bare EOF, but an explicit finish carries usage.
    pub fn finish(&mut self) -> String {
        if self.finished {
            return String::new();
        }
        self.finished = true;
        let mut out = String::new();
        // A transport failure after partial tool calls must NOT report
        // success: the buffered arguments are truncated (often unparseable,
        // replaced with {}), and executing them would run tools on garbage.
        // fx also rejects an error finish combined with tool calls, so the
        // calls are dropped, not emitted — the turn ends as the error it was.
        let emitted_calls = if self.failed {
            std::mem::take(&mut self.tools);
            false
        } else {
            let mut emitted = false;
            for (idx, id, name, args) in std::mem::take(&mut self.tools) {
                if name.is_empty() {
                    continue;
                }
                emitted = true;
                // fx accepts `input` as raw JSON or as a JSON string; send parsed
                // when possible so a malformed fragment can't break the turn.
                let input: Value = serde_json::from_str(&args).unwrap_or_else(|_| json!({}));
                out.push_str(&format_data(&json!({
                    "type": "tool-call",
                    // The index keeps parallel calls to the SAME tool distinct:
                    // duplicate ids make fx reject the next turn's history.
                    "toolCallId": if id.is_empty() { format!("call_{idx}_{name}") } else { id },
                    "toolName": name,
                    "input": input,
                })));
            }
            emitted
        };
        let unified = if self.failed {
            "error"
        } else {
            unified_finish(self.finish_reason.as_deref(), emitted_calls)
        };
        out.push_str(&format_data(&json!({
            "type": "finish",
            "finishReason": {"unified": unified},
            "usage": {
                "inputTokens": {"total": self.input_tokens},
                "outputTokens": {"total": self.output_tokens},
            },
        })));
        out.push_str("data: [DONE]\n\n");
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A transport failure must win over buffered tool calls: the turn ends
    /// as `error` with NO tool-call events (fx rejects error+tool-calls, and
    /// truncated arguments would execute tools on garbage).
    #[test]
    fn failed_stream_drops_partial_tool_calls_and_ends_as_error() {
        let mut st = StreamState::new("m");
        st.on_data(r#"{"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"c1","function":{"name":"Bash","arguments":"{\"cm"}}]}}]}"#);
        st.fail();
        let out = st.finish();
        assert!(out.contains(r#""unified":"error""#), "{out}");
        assert!(!out.contains("tool-call"), "{out}");
        // finish() stays idempotent.
        assert_eq!(st.finish(), "");
    }

    #[test]
    fn request_maps_prompt_tools_and_choice() {
        let body = json!({
            "prompt": [
                {"role": "system", "content": "sys"},
                {"role": "user", "content": [{"type": "text", "text": "hi"}]},
                {"role": "assistant", "content": [
                    {"type": "tool-call", "toolCallId": "c1", "toolName": "Bash",
                     "input": {"cmd": "ls"}}
                ]},
                {"role": "tool", "content": [
                    {"type": "tool-result", "toolCallId": "c1", "toolName": "Bash",
                     "output": {"type": "text", "value": "done"}}
                ]}
            ],
            "tools": [{"type": "function", "name": "Bash", "description": "d",
                       "inputSchema": {"type": "object"}}],
            "toolChoice": {"type": "required"},
            "maxOutputTokens": 4096
        });
        let out = request(&body, "zai/glm-4.7-flash", true);
        assert_eq!(out["model"], "zai/glm-4.7-flash");
        assert_eq!(out["stream"], true);
        assert_eq!(out["max_tokens"], 4096);
        let m = out["messages"].as_array().unwrap();
        assert_eq!(m[0]["content"], "sys");
        assert_eq!(m[1]["content"], "hi", "text-only user stays a plain string");
        // raw JSON input becomes an arguments STRING
        assert_eq!(m[2]["tool_calls"][0]["function"]["arguments"], "{\"cmd\":\"ls\"}");
        assert_eq!(m[3]["role"], "tool");
        assert_eq!(m[3]["content"], "done");
        // inputSchema -> function.parameters
        assert_eq!(out["tools"][0]["function"]["parameters"]["type"], "object");
        assert_eq!(out["tool_choice"], "required");
    }

    #[test]
    fn request_maps_image_parts_and_skips_provider_tools() {
        let body = json!({
            "prompt": [{"role": "user", "content": [
                {"type": "text", "text": "look"},
                {"type": "file", "mediaType": "image/png", "data": "aGk="}
            ]}],
            "tools": [
                {"type": "provider", "id": "gateway.perplexity_search", "name": "search"},
                {"type": "function", "name": "Read", "inputSchema": {"type": "object"}}
            ]
        });
        let out = request(&body, "m", false);
        let content = out["messages"][0]["content"].as_array().unwrap();
        assert_eq!(content[1]["image_url"]["url"], "data:image/png;base64,aGk=");
        let tools = out["tools"].as_array().unwrap();
        assert_eq!(tools.len(), 1, "gateway-executed tools are not ours to run");
        assert_eq!(tools[0]["function"]["name"], "Read");
    }

    #[test]
    fn stream_emits_text_toolcall_and_finish() {
        let mut st = StreamState::new("m");
        let first = st.on_data(r#"{"choices":[{"index":0,"delta":{"content":"ok"}}]}"#);
        assert!(first.contains("\"type\":\"response-metadata\""), "metadata first: {first}");
        assert!(first.contains("\"type\":\"text-delta\""));
        assert!(first.contains("\"delta\":\"ok\""));
        // split tool call across chunks
        st.on_data(
            r#"{"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"c1","function":{"name":"Bash","arguments":"{\"cmd\":"}}]}}]}"#,
        );
        st.on_data(
            r#"{"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"\"ls\"}"}}]}}]}"#,
        );
        st.on_data(
            r#"{"choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}],"usage":{"prompt_tokens":7,"completion_tokens":9}}"#,
        );
        let tail = st.finish();
        assert!(tail.contains("\"type\":\"tool-call\""), "{tail}");
        assert!(tail.contains("\"toolName\":\"Bash\""));
        assert!(tail.contains("\"cmd\":\"ls\""), "assembled args: {tail}");
        assert!(tail.contains("\"unified\":\"tool-calls\""));
        assert!(tail.contains("\"inputTokens\":{\"total\":7}"));
        assert!(tail.ends_with("data: [DONE]\n\n"));
        // every event line carries the space fx requires
        for line in tail.lines().filter(|l| l.starts_with("data")) {
            assert!(line.starts_with("data: "), "missing space: {line}");
        }
    }

    #[test]
    fn finish_reasons_stay_in_the_closed_set() {
        assert_eq!(unified_finish(Some("stop"), false), "stop");
        assert_eq!(unified_finish(Some("length"), false), "length");
        assert_eq!(unified_finish(Some("content_filter"), false), "content-filter");
        assert_eq!(unified_finish(Some("tool_calls"), false), "tool-calls");
        assert_eq!(unified_finish(None, false), "stop");
        // An unknown upstream value must NOT be passed through: fx aborts the
        // whole stream on an unrecognized unified value.
        assert_eq!(unified_finish(Some("eos_token"), false), "other");
    }

    #[test]
    fn finish_reason_agrees_with_emitted_tool_calls() {
        // fx's classifyProviderCompletion rejects stop/other/error WITH tool
        // calls as an invalid completion and kills the agentic turn. Free
        // providers really do send tool_calls + finish_reason "stop".
        assert_eq!(unified_finish(Some("stop"), true), "tool-calls");
        assert_eq!(unified_finish(Some("eos_token"), true), "tool-calls");
        assert_eq!(unified_finish(None, true), "tool-calls");
        // Truncation stays truthful — the turn really was cut short.
        assert_eq!(unified_finish(Some("length"), true), "length");
        assert_eq!(unified_finish(Some("content_filter"), true), "content-filter");
    }

    #[test]
    fn native_tool_calls_with_stop_reason_finish_as_tool_calls() {
        let mut st = StreamState::new("m");
        st.on_data(
            r#"{"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"c1","function":{"name":"Bash","arguments":"{}"}}]}}]}"#,
        );
        st.on_data(r#"{"choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}"#);
        let tail = st.finish();
        assert!(tail.contains("\"unified\":\"tool-calls\""), "{tail}");
    }

    #[test]
    fn transport_failure_finishes_as_error_not_stop() {
        let mut st = StreamState::new("m");
        st.on_data(r#"{"choices":[{"index":0,"delta":{"content":"partial"}}]}"#);
        st.fail(); // mid-stream transport death
        let tail = st.finish();
        assert!(tail.contains("\"unified\":\"error\""), "truncation must not read as success: {tail}");

        // An in-stream error event marks the same way.
        let mut st2 = StreamState::new("m");
        let out = st2.on_data(r#"{"error":{"message":"upstream exploded"}}"#);
        assert!(out.contains("\"type\":\"error\""));
        assert!(st2.finish().contains("\"unified\":\"error\""));
    }

    #[test]
    fn idless_parallel_calls_to_one_tool_get_distinct_ids() {
        let mut st = StreamState::new("m");
        st.on_data(
            r#"{"choices":[{"index":0,"delta":{"tool_calls":[
                {"index":0,"function":{"name":"Read","arguments":"{\"p\":\"a\"}"}},
                {"index":1,"function":{"name":"Read","arguments":"{\"p\":\"b\"}"}}]}}]}"#,
        );
        let tail = st.finish();
        // Duplicate toolCallIds make fx reject the NEXT turn's history.
        assert!(tail.contains("\"toolCallId\":\"call_0_Read\""), "{tail}");
        assert!(tail.contains("\"toolCallId\":\"call_1_Read\""), "{tail}");
    }

    #[test]
    fn response_survives_bodies_without_choices() {
        // 200-with-no-choices is a real free-provider shape; serde_json's
        // mut-index would PANIC on it.
        assert_eq!(response(&json!({"choices": []})), json!({"choices": []}));
        assert_eq!(response(&json!({})), json!({}));
        // finish_reason is normalized into fx's legacy closed set.
        let out = response(&json!({"choices": [{"finish_reason": "eos",
            "message": {"role": "assistant", "content": "hi"}}]}));
        assert_eq!(out["choices"][0]["finish_reason"], "other");
        let out = response(&json!({"choices": [{"finish_reason": "stop", "message":
            {"role": "assistant", "tool_calls": [{"id": "c1", "type": "function",
             "function": {"name": "B", "arguments": "{}"}}]}}]}));
        assert_eq!(out["choices"][0]["finish_reason"], "tool_calls", "must agree with the calls");
    }

    #[test]
    fn finish_is_idempotent() {
        let mut st = StreamState::new("m");
        let a = st.finish();
        assert!(a.contains("[DONE]"));
        assert!(st.finish().is_empty(), "double-finish must not duplicate");
    }

    #[test]
    fn nonstreaming_response_stringifies_tool_arguments() {
        let body = json!({"choices": [{"finish_reason": "tool_calls", "message":
            {"role": "assistant", "tool_calls": [{"id": "c1", "type": "function",
             "function": {"name": "Bash", "arguments": {"cmd": "ls"}}}]}}]});
        let out = response(&body);
        assert_eq!(
            out["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"],
            "{\"cmd\":\"ls\"}"
        );
    }
}
