//! Anthropic messages <-> CodeWhisperer (`kiro`).
//!
//! Kiro speaks neither OpenAI nor Anthropic: requests are a `conversationState`
//! object and responses are `vnd.amazon.eventstream` binary frames. To keep the
//! rest of pxy unchanged, this module converts an Anthropic-shaped request into
//! conversationState, and converts decoded frames back into **OpenAI SSE text**
//! so the existing SSE pipeline (and the OpenAI->Anthropic stream translator)
//! handle the client side as usual.
//!
//! Shape rules are not guesses — CodeWhisperer 400s on violations, and each one
//! below was verified against OmniRoute's translator (openai-to-kiro.ts):
//!   * no `system` role: system text is wrapped in a system-reminder tag and
//!     merged into the first user turn;
//!   * history must alternate user/assistant and START with user — synthetic
//!     "(empty)" turns are inserted where it doesn't;
//!   * empty content is never sent: "(empty)" for turns, "(no output)" for
//!     tool results;
//!   * tool schemas must not contain `additionalProperties`, `$schema`,
//!     `anyOf`/`oneOf`/`allOf`, or an empty `required` array;
//!   * `tools` may only appear on the CURRENT message, never in history;
//!   * unknown top-level fields are rejected, so only conversationState /
//!     profileArn / inferenceConfig are sent.

use serde_json::{Map, Value, json};

use super::eventstream::Frame;

/// uuidv5 namespace OmniRoute uses for conversation ids; keeping the same
/// namespace means a resumed conversation keeps AWS's prompt-cache affinity.
const NAMESPACE_KIRO: &str = "34f7193f-561d-4050-bc84-9547d953d6bf";
const ORIGIN: &str = "AI_EDITOR";
const EMPTY: &str = "(empty)";

/// Build the CodeWhisperer payload from an Anthropic-format request.
pub fn request(body: &Value, model: &str, profile_arn: &str, now_iso: &str) -> Value {
    let mut history: Vec<Value> = Vec::new();
    let mut system_text = collect_system(body);

    for msg in body["messages"].as_array().unwrap_or(&vec![]).iter() {
        let role = msg["role"].as_str().unwrap_or("user");
        let (text, tool_uses, tool_results) = split_content(&msg["content"]);
        if role == "assistant" {
            let mut m = json!({ "content": non_empty(&text) });
            if !tool_uses.is_empty() {
                m["toolUses"] = json!(tool_uses);
            }
            history.push(json!({ "assistantResponseMessage": m }));
        } else {
            // system text rides on the first user turn (there is no system role)
            let mut content = text;
            if !system_text.is_empty() {
                content = format!(
                    "<system-reminder>\n{system_text}\n</system-reminder>\n\n{content}"
                );
                system_text.clear();
            }
            let mut uim = json!({
                "content": if content.trim().is_empty() && !tool_results.is_empty() {
                    String::new()
                } else {
                    non_empty(&content)
                },
                "modelId": model,
                "origin": ORIGIN,
            });
            if !tool_results.is_empty() {
                uim["userInputMessageContext"] = json!({ "toolResults": tool_results });
            }
            history.push(json!({ "userInputMessage": uim }));
        }
    }

    normalize_turns(&mut history, model);

    // The trailing user turn becomes currentMessage; tools attach only there.
    let mut current = match history.last() {
        Some(v) if v.get("userInputMessage").is_some() => history.pop().unwrap(),
        // A conversation ending on an assistant turn needs a filler prompt.
        // OmniRoute settled on "..." — "Continue" changed model behaviour.
        _ => json!({
            "userInputMessage": { "content": "...", "modelId": model, "origin": ORIGIN }
        }),
    };

    // conversationId seeds AWS's prompt cache: same opening turn => same id.
    let seed = history
        .iter()
        .find_map(|h| h["userInputMessage"]["content"].as_str())
        .unwrap_or_else(|| current["userInputMessage"]["content"].as_str().unwrap_or(""));
    let conversation_id = uuid_v5(NAMESPACE_KIRO, &seed.chars().take(4000).collect::<String>());

    if let Some(specs) = tool_specs(&body["tools"]) {
        let ctx = current["userInputMessage"]
            .as_object_mut()
            .unwrap()
            .entry("userInputMessageContext")
            .or_insert_with(|| json!({}));
        ctx["tools"] = specs;
    }

    // Timestamp prefix mirrors the Kiro IDE client; models rely on it for
    // "what time is it" style questions since there is no system prompt.
    if let Some(c) = current["userInputMessage"]["content"].as_str() {
        current["userInputMessage"]["content"] =
            json!(format!("[Context: Current time is {now_iso}]\n\n{c}"));
    }

    let mut payload = json!({
        "conversationState": {
            "chatTriggerType": "MANUAL",
            "conversationId": conversation_id,
            "currentMessage": current,
            "history": history,
        },
        "profileArn": profile_arn,
    });

    let mut inference = Map::new();
    if let Some(m) = body["max_tokens"].as_u64() {
        inference.insert("maxTokens".into(), json!(m));
    }
    if let Some(t) = body["temperature"].as_f64() {
        inference.insert("temperature".into(), json!(t));
    }
    if let Some(p) = body["top_p"].as_f64() {
        inference.insert("topP".into(), json!(p));
    }
    if !inference.is_empty() {
        payload["inferenceConfig"] = Value::Object(inference);
    }
    payload
}

fn collect_system(body: &Value) -> String {
    match &body["system"] {
        Value::String(s) => s.clone(),
        Value::Array(a) => a
            .iter()
            .filter_map(|b| b["text"].as_str())
            .collect::<Vec<_>>()
            .join("\n\n"),
        _ => String::new(),
    }
}

/// Split an Anthropic content field into (text, toolUses, toolResults).
fn split_content(content: &Value) -> (String, Vec<Value>, Vec<Value>) {
    let mut text = String::new();
    let (mut uses, mut results) = (Vec::new(), Vec::new());
    match content {
        Value::String(s) => text.push_str(s),
        Value::Array(blocks) => {
            for b in blocks {
                match b["type"].as_str().unwrap_or("") {
                    "text" => {
                        if !text.is_empty() {
                            text.push('\n');
                        }
                        text.push_str(b["text"].as_str().unwrap_or(""));
                    }
                    "tool_use" => uses.push(json!({
                        "toolUseId": b["id"].as_str().unwrap_or(""),
                        "name": b["name"].as_str().unwrap_or(""),
                        "input": b["input"].clone(),
                    })),
                    "tool_result" => {
                        let body = tool_result_text(&b["content"]);
                        results.push(json!({
                            "toolUseId": b["tool_use_id"].as_str().unwrap_or(""),
                            "status": if b["is_error"].as_bool().unwrap_or(false) { "error" } else { "success" },
                            // never [] or [{text:""}] — CodeWhisperer 400s
                            "content": [{ "text": body }],
                        }));
                    }
                    _ => {}
                }
            }
        }
        _ => {}
    }
    (text, uses, results)
}

fn tool_result_text(content: &Value) -> String {
    let s = match content {
        Value::String(s) => s.clone(),
        Value::Array(a) => a
            .iter()
            .filter_map(|b| {
                b["text"]
                    .as_str()
                    .map(String::from)
                    .or_else(|| b.as_str().map(String::from))
            })
            .collect::<Vec<_>>()
            .join("\n"),
        Value::Null => String::new(),
        other => other.to_string(),
    };
    if s.trim().is_empty() {
        "(no output)".into()
    } else {
        s
    }
}

fn non_empty(s: &str) -> String {
    if s.trim().is_empty() {
        EMPTY.into()
    } else {
        s.to_string()
    }
}

/// Force the shape CodeWhisperer requires: starts with user, strictly
/// alternating. Synthetic turns are cheap; a 400 costs a whole request.
fn normalize_turns(history: &mut Vec<Value>, model: &str) {
    if history.is_empty() {
        return;
    }
    if history[0].get("userInputMessage").is_none() {
        history.insert(
            0,
            json!({"userInputMessage": {"content": EMPTY, "modelId": model, "origin": ORIGIN}}),
        );
    }
    let mut i = 0;
    while i + 1 < history.len() {
        let a_user = history[i].get("userInputMessage").is_some();
        let b_user = history[i + 1].get("userInputMessage").is_some();
        if a_user == b_user {
            let filler = if a_user {
                json!({"assistantResponseMessage": {"content": EMPTY}})
            } else {
                json!({"userInputMessage": {"content": EMPTY, "modelId": model, "origin": ORIGIN}})
            };
            history.insert(i + 1, filler);
        }
        i += 1;
    }
}

/// Anthropic tools -> toolSpecification wrappers, with schemas sanitized.
fn tool_specs(tools: &Value) -> Option<Value> {
    let arr = tools.as_array()?;
    if arr.is_empty() {
        return None;
    }
    let specs: Vec<Value> = arr
        .iter()
        .map(|t| {
            let name = t["name"].as_str().unwrap_or("tool");
            let desc = t["description"].as_str().unwrap_or("");
            json!({
                "toolSpecification": {
                    "name": name,
                    // an empty description is rejected
                    "description": if desc.trim().is_empty() { format!("Tool: {name}") } else { desc.to_string() },
                    "inputSchema": { "json": sanitize_schema(&t["input_schema"]) },
                }
            })
        })
        .collect();
    Some(json!(specs))
}

/// Strip JSON-Schema keywords CodeWhisperer rejects ("Improperly formed
/// request"), recursively. Also drops `required: []`.
fn sanitize_schema(schema: &Value) -> Value {
    const STRIP: &[&str] = &[
        "additionalProperties",
        "anyOf",
        "oneOf",
        "allOf",
        "not",
        "$schema",
        "$id",
        "$ref",
        "$defs",
        "definitions",
        "if",
        "then",
        "else",
        "unevaluatedProperties",
        "unevaluatedItems",
        "contentEncoding",
        "contentMediaType",
    ];
    match schema {
        Value::Object(map) => {
            let mut out = Map::new();
            for (k, v) in map {
                if STRIP.contains(&k.as_str()) {
                    continue;
                }
                if k == "required" && v.as_array().is_some_and(|a| a.is_empty()) {
                    continue;
                }
                out.insert(k.clone(), sanitize_schema(v));
            }
            if out.is_empty() {
                json!({"type": "object", "properties": {}})
            } else {
                Value::Object(out)
            }
        }
        Value::Array(a) => Value::Array(a.iter().map(sanitize_schema).collect()),
        other => other.clone(),
    }
}

// ---------------------------------------------------------------------------
// Response: eventstream frames -> OpenAI SSE text
// ---------------------------------------------------------------------------

/// Accumulates streaming state across frames. Tool inputs are the subtle part:
/// CodeWhisperer resends the WHOLE input object as it grows, so emitting each
/// one as an argument delta would concatenate overlapping prefixes into
/// unparseable JSON. Object-form inputs are therefore buffered and emitted
/// once at stop; string-form inputs are already deltas and pass straight
/// through.
pub struct StreamState {
    id: String,
    model: String,
    role_sent: bool,
    tool_index: usize,
    seen_tools: Vec<(String, usize)>,
    buffered_args: Vec<(usize, String)>,
    pub saw_tool_use: bool,
    pub context_pct: f64,
    pub credits: f64,
    pub text_len: usize,
}

impl StreamState {
    pub fn new(model: &str) -> Self {
        Self {
            id: format!("chatcmpl-kiro-{}", model.replace('/', "-")),
            model: model.to_string(),
            role_sent: false,
            tool_index: 0,
            seen_tools: Vec::new(),
            buffered_args: Vec::new(),
            saw_tool_use: false,
            context_pct: 0.0,
            credits: 0.0,
            text_len: 0,
        }
    }

    /// Convert decoded frames into OpenAI SSE text (may be empty).
    pub fn frames_to_sse(&mut self, frames: Vec<Frame>) -> String {
        let mut out = String::new();
        for f in frames {
            let payload: Value = serde_json::from_slice(&f.payload).unwrap_or(Value::Null);
            match f.event_type.as_str() {
                "assistantResponseEvent" | "codeEvent" => {
                    if let Some(text) = payload["content"].as_str() {
                        if !text.is_empty() {
                            self.text_len += text.len();
                            out.push_str(&self.chunk(json!({"content": text})));
                        }
                    }
                }
                "reasoningContentEvent" => {
                    let text = payload["reasoningText"]["text"]
                        .as_str()
                        .or_else(|| payload["reasoningText"].as_str())
                        .or_else(|| payload["text"].as_str())
                        .unwrap_or("");
                    if !text.is_empty() {
                        out.push_str(&self.chunk(json!({"reasoning_content": text})));
                    }
                }
                "toolUseEvent" => {
                    let items = if payload.is_array() {
                        payload.as_array().cloned().unwrap_or_default()
                    } else {
                        vec![payload.clone()]
                    };
                    for item in items {
                        out.push_str(&self.tool_use(&item));
                    }
                }
                "contextUsageEvent" => {
                    if let Some(p) = payload["contextUsagePercentage"].as_f64() {
                        if p > 0.0 {
                            self.context_pct = p;
                        }
                    }
                }
                "meteringEvent" => {
                    if let Some(u) = payload["usage"].as_f64() {
                        self.credits += u;
                    }
                }
                _ => {}
            }
        }
        out
    }

    fn tool_use(&mut self, item: &Value) -> String {
        let name = item["name"].as_str().unwrap_or("");
        if name.is_empty() {
            return String::new();
        }
        let id = item["toolUseId"].as_str().unwrap_or("").to_string();
        self.saw_tool_use = true;

        let idx = match self.seen_tools.iter().find(|(k, _)| *k == id) {
            Some((_, i)) => *i,
            None => {
                let i = self.tool_index;
                self.tool_index += 1;
                self.seen_tools.push((id.clone(), i));
                let start = self.chunk(json!({"tool_calls": [{
                    "index": i,
                    "id": if id.is_empty() { format!("call_{i}") } else { id.clone() },
                    "type": "function",
                    "function": {"name": name, "arguments": ""},
                }]}));
                match &item["input"] {
                    Value::String(s) => return start + &self.arg_delta(i, s),
                    Value::Null => return start,
                    v => {
                        self.buffer_args(i, v.to_string());
                        return start;
                    }
                }
            }
        };
        match &item["input"] {
            Value::String(s) => self.arg_delta(idx, s),
            Value::Null => String::new(),
            v => {
                self.buffer_args(idx, v.to_string());
                String::new()
            }
        }
    }

    fn buffer_args(&mut self, idx: usize, canonical: String) {
        match self.buffered_args.iter_mut().find(|(i, _)| *i == idx) {
            Some(slot) => slot.1 = canonical,
            None => self.buffered_args.push((idx, canonical)),
        }
    }

    fn arg_delta(&mut self, idx: usize, s: &str) -> String {
        self.chunk(json!({"tool_calls": [{
            "index": idx, "function": {"arguments": s}
        }]}))
    }

    /// Flush buffered tool arguments and close the stream.
    pub fn finish(&mut self) -> String {
        let mut out = String::new();
        let buffered = std::mem::take(&mut self.buffered_args);
        for (idx, canonical) in buffered {
            out.push_str(&self.chunk(json!({"tool_calls": [{
                "index": idx, "function": {"arguments": canonical}
            }]})));
        }
        let reason = if self.saw_tool_use { "tool_calls" } else { "stop" };
        out.push_str(&format!(
            "data: {}\n\n",
            json!({
                "id": self.id,
                "object": "chat.completion.chunk",
                "model": self.model,
                "choices": [{"index": 0, "delta": {}, "finish_reason": reason}],
            })
        ));
        out.push_str("data: [DONE]\n\n");
        out
    }

    fn chunk(&mut self, mut delta: Value) -> String {
        if !self.role_sent {
            self.role_sent = true;
            delta["role"] = json!("assistant");
        }
        format!(
            "data: {}\n\n",
            json!({
                "id": self.id,
                "object": "chat.completion.chunk",
                "model": self.model,
                "choices": [{"index": 0, "delta": delta, "finish_reason": null}],
            })
        )
    }
}

/// Kiro has no non-streaming mode: even a plain request returns eventstream.
/// Collect the whole thing into one OpenAI-shaped response.
pub fn collect_response(bytes: &[u8], model: &str, id: &str, context_length: u64) -> (Value, super::TokenUsage) {
    let mut decoder = super::eventstream::EventStreamDecoder::new();
    decoder.push(bytes);
    let frames = decoder.drain();

    let mut text = String::new();
    let mut tool_calls: Vec<Value> = Vec::new();
    let mut state = StreamState::new(model);

    for f in &frames {
        let p: Value = serde_json::from_slice(&f.payload).unwrap_or(Value::Null);
        match f.event_type.as_str() {
            "assistantResponseEvent" | "codeEvent" => {
                if let Some(t) = p["content"].as_str() {
                    text.push_str(t);
                }
            }
            "toolUseEvent" => {
                let items = if p.is_array() {
                    p.as_array().cloned().unwrap_or_default()
                } else {
                    vec![p.clone()]
                };
                for item in items {
                    let Some(name) = item["name"].as_str() else { continue };
                    let tid = item["toolUseId"].as_str().unwrap_or("").to_string();
                    let slot = match tool_calls.iter().position(|c| c["id"] == json!(tid)) {
                        Some(i) => i,
                        None => {
                            tool_calls.push(json!({
                                "id": if tid.is_empty() { format!("call_{}", tool_calls.len()) } else { tid },
                                "type": "function",
                                "function": {"name": name, "arguments": ""},
                            }));
                            tool_calls.len() - 1
                        }
                    };
                    // String fragments CONCATENATE into the JSON arguments
                    // ("{\"city\":" + " \"" + "Dhak" + "a\"}"); whole-object
                    // inputs are resends where the latest wins. The trailing
                    // {"stop":true} frame carries no input at all.
                    match &item["input"] {
                        Value::String(s) => {
                            let cur = tool_calls[slot]["function"]["arguments"]
                                .as_str()
                                .unwrap_or("")
                                .to_string();
                            tool_calls[slot]["function"]["arguments"] = json!(cur + s);
                        }
                        Value::Null => {}
                        v => tool_calls[slot]["function"]["arguments"] = json!(v.to_string()),
                    }
                }
            }
            "contextUsageEvent" => {
                if let Some(v) = p["contextUsagePercentage"].as_f64() {
                    if v > 0.0 {
                        state.context_pct = v;
                    }
                }
            }
            "meteringEvent" => {
                if let Some(u) = p["usage"].as_f64() {
                    state.credits += u;
                }
            }
            _ => {}
        }
    }

    state.text_len = text.len();
    let usage = state.usage(context_length);
    let mut message = json!({"role": "assistant", "content": text});
    if !tool_calls.is_empty() {
        message["tool_calls"] = json!(tool_calls);
    }
    let body = json!({
        "id": id,
        "object": "chat.completion",
        "model": model,
        "choices": [{
            "index": 0,
            "message": message,
            "finish_reason": if tool_calls.is_empty() { "stop" } else { "tool_calls" },
        }],
        "usage": {
            "prompt_tokens": usage.input,
            "completion_tokens": usage.output,
            "total_tokens": usage.input + usage.output,
        },
    });
    (body, usage)
}

impl StreamState {
    /// Kiro reports no token counts — only a context-usage percentage and a
    /// credit figure. Derive an estimate so accounting isn't blank: output
    /// from emitted characters, input from percent-of-context minus output.
    pub fn usage(&self, context_length: u64) -> super::TokenUsage {
        let output = (self.text_len / 4) as u64;
        let total = if self.context_pct > 0.0 {
            ((self.context_pct / 100.0) * context_length as f64) as u64
        } else {
            0
        };
        super::TokenUsage {
            input: total.saturating_sub(output),
            output,
        }
    }
}

// ---------------------------------------------------------------------------
// uuid v5 (SHA-1 based). Implemented here rather than pulling in `uuid` +
// `sha1` crates for one call site.
// ---------------------------------------------------------------------------

fn uuid_v5(namespace: &str, name: &str) -> String {
    let mut input = parse_uuid(namespace).to_vec();
    input.extend_from_slice(name.as_bytes());
    let h = sha1(&input);
    let mut b = [0u8; 16];
    b.copy_from_slice(&h[..16]);
    b[6] = (b[6] & 0x0f) | 0x50; // version 5
    b[8] = (b[8] & 0x3f) | 0x80; // RFC 4122 variant
    format!(
        "{}-{}-{}-{}-{}",
        hex(&b[0..4]),
        hex(&b[4..6]),
        hex(&b[6..8]),
        hex(&b[8..10]),
        hex(&b[10..16])
    )
}

fn parse_uuid(s: &str) -> [u8; 16] {
    let hexchars: Vec<u8> = s.bytes().filter(|c| *c != b'-').collect();
    let mut out = [0u8; 16];
    for (i, pair) in hexchars.chunks(2).take(16).enumerate() {
        out[i] = u8::from_str_radix(std::str::from_utf8(pair).unwrap_or("00"), 16).unwrap_or(0);
    }
    out
}

fn hex(b: &[u8]) -> String {
    b.iter().map(|x| format!("{x:02x}")).collect()
}

fn sha1(data: &[u8]) -> [u8; 20] {
    let mut h: [u32; 5] = [0x67452301, 0xEFCDAB89, 0x98BADCFE, 0x10325476, 0xC3D2E1F0];
    let ml = (data.len() as u64) * 8;
    let mut msg = data.to_vec();
    msg.push(0x80);
    while msg.len() % 64 != 56 {
        msg.push(0);
    }
    msg.extend_from_slice(&ml.to_be_bytes());

    for block in msg.chunks(64) {
        let mut w = [0u32; 80];
        for i in 0..16 {
            w[i] = u32::from_be_bytes([
                block[i * 4],
                block[i * 4 + 1],
                block[i * 4 + 2],
                block[i * 4 + 3],
            ]);
        }
        for i in 16..80 {
            w[i] = (w[i - 3] ^ w[i - 8] ^ w[i - 14] ^ w[i - 16]).rotate_left(1);
        }
        let (mut a, mut b, mut c, mut d, mut e) = (h[0], h[1], h[2], h[3], h[4]);
        for (i, wi) in w.iter().enumerate() {
            let (f, k) = match i {
                0..=19 => ((b & c) | (!b & d), 0x5A827999),
                20..=39 => (b ^ c ^ d, 0x6ED9EBA1),
                40..=59 => ((b & c) | (b & d) | (c & d), 0x8F1BBCDC),
                _ => (b ^ c ^ d, 0xCA62C1D6),
            };
            let tmp = a
                .rotate_left(5)
                .wrapping_add(f)
                .wrapping_add(e)
                .wrapping_add(k)
                .wrapping_add(*wi);
            e = d;
            d = c;
            c = b.rotate_left(30);
            b = a;
            a = tmp;
        }
        h[0] = h[0].wrapping_add(a);
        h[1] = h[1].wrapping_add(b);
        h[2] = h[2].wrapping_add(c);
        h[3] = h[3].wrapping_add(d);
        h[4] = h[4].wrapping_add(e);
    }
    let mut out = [0u8; 20];
    for (i, v) in h.iter().enumerate() {
        out[i * 4..i * 4 + 4].copy_from_slice(&v.to_be_bytes());
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sha1_matches_rfc3174_vectors() {
        assert_eq!(hex(&sha1(b"abc")), "a9993e364706816aba3e25717850c26c9cd0d89d");
        assert_eq!(hex(&sha1(b"")), "da39a3ee5e6b4b0d3255bfef95601890afd80709");
        assert_eq!(
            hex(&sha1(b"abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq")),
            "84983e441c3bd26ebaae4aa1f95129e5e54670f1"
        );
    }

    #[test]
    fn uuid_v5_matches_rfc_example() {
        // RFC 4122 / python uuid5(NAMESPACE_DNS, "python.org")
        assert_eq!(
            uuid_v5("6ba7b810-9dad-11d1-80b4-00c04fd430c8", "python.org"),
            "886313e1-3b8a-5372-9b90-0c9aee199e5d"
        );
    }

    #[test]
    fn conversation_id_is_stable_for_the_same_opening_turn() {
        let body = json!({"messages": [{"role": "user", "content": "hello there"}]});
        let a = request(&body, "claude-haiku-4.5", "arn:x", "t");
        let b = request(&body, "claude-haiku-4.5", "arn:x", "t2");
        assert_eq!(
            a["conversationState"]["conversationId"],
            b["conversationState"]["conversationId"]
        );
    }

    #[test]
    fn system_prompt_merges_into_the_first_user_turn() {
        let body = json!({
            "system": "be terse",
            "messages": [{"role": "user", "content": "hi"}],
        });
        let p = request(&body, "m", "arn", "T");
        let content = p["conversationState"]["currentMessage"]["userInputMessage"]["content"]
            .as_str()
            .unwrap();
        assert!(content.contains("<system-reminder>\nbe terse\n</system-reminder>"));
        assert!(content.contains("hi"));
        assert!(content.starts_with("[Context: Current time is T]"));
        assert!(p["conversationState"]["currentMessage"]["userInputMessage"]["role"].is_null());
    }

    #[test]
    fn history_alternates_and_starts_with_user() {
        let body = json!({"messages": [
            {"role": "assistant", "content": "first?"},
            {"role": "assistant", "content": "twice"},
            {"role": "user", "content": "now me"},
        ]});
        let p = request(&body, "m", "arn", "T");
        let hist = p["conversationState"]["history"].as_array().unwrap();
        assert!(hist[0].get("userInputMessage").is_some(), "must start with user");
        for pair in hist.windows(2) {
            let a = pair[0].get("userInputMessage").is_some();
            let b = pair[1].get("userInputMessage").is_some();
            assert_ne!(a, b, "turns must alternate");
        }
    }

    #[test]
    fn tool_schema_is_sanitized_and_only_on_current_message() {
        let body = json!({
            "messages": [{"role": "user", "content": "go"}],
            "tools": [{
                "name": "run",
                "description": "",
                "input_schema": {
                    "type": "object",
                    "additionalProperties": false,
                    "$schema": "http://json-schema.org/draft-07/schema#",
                    "required": [],
                    "properties": {"cmd": {"type": "string", "anyOf": [{"type": "string"}]}},
                },
            }],
        });
        let p = request(&body, "m", "arn", "T");
        let spec = &p["conversationState"]["currentMessage"]["userInputMessage"]
            ["userInputMessageContext"]["tools"][0]["toolSpecification"];
        assert_eq!(spec["name"], "run");
        assert_eq!(spec["description"], "Tool: run", "empty description rejected upstream");
        let schema = &spec["inputSchema"]["json"];
        assert!(schema["additionalProperties"].is_null());
        assert!(schema["$schema"].is_null());
        assert!(schema["required"].is_null(), "empty required must be dropped");
        assert!(schema["properties"]["cmd"]["anyOf"].is_null());
        assert_eq!(schema["properties"]["cmd"]["type"], "string");
    }

    #[test]
    fn tool_results_never_send_empty_content() {
        let body = json!({"messages": [
            {"role": "user", "content": "go"},
            {"role": "assistant", "content": [
                {"type": "tool_use", "id": "t1", "name": "run", "input": {"cmd": "ls"}}]},
            {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "t1", "content": ""}]},
        ]});
        let p = request(&body, "m", "arn", "T");
        let cur = &p["conversationState"]["currentMessage"]["userInputMessage"];
        let results = &cur["userInputMessageContext"]["toolResults"];
        assert_eq!(results[0]["toolUseId"], "t1");
        assert_eq!(results[0]["status"], "success");
        assert_eq!(results[0]["content"][0]["text"], "(no output)");
    }

    #[test]
    fn only_allowlisted_top_level_fields_are_sent() {
        let body = json!({
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 100, "temperature": 0.5, "stream": true, "metadata": {"x": 1},
        });
        let p = request(&body, "m", "arn", "T");
        let keys: Vec<&str> = p.as_object().unwrap().keys().map(|s| s.as_str()).collect();
        assert_eq!(keys, ["conversationState", "inferenceConfig", "profileArn"]);
        assert_eq!(p["inferenceConfig"]["maxTokens"], 100);
    }

    fn frame(t: &str, payload: &str) -> Frame {
        Frame {
            event_type: t.into(),
            message_type: "event".into(),
            payload: payload.as_bytes().to_vec(),
        }
    }

    #[test]
    fn text_frames_become_openai_chunks_with_role_once() {
        let mut s = StreamState::new("claude-haiku-4.5");
        let sse = s.frames_to_sse(vec![
            frame("assistantResponseEvent", r#"{"content":"He"}"#),
            frame("assistantResponseEvent", r#"{"content":"llo"}"#),
        ]);
        assert_eq!(sse.matches("\"role\":\"assistant\"").count(), 1);
        assert!(sse.contains(r#""content":"He""#));
        assert!(sse.contains(r#""content":"llo""#));
        let fin = s.finish();
        assert!(fin.contains(r#""finish_reason":"stop""#));
        assert!(fin.ends_with("data: [DONE]\n\n"));
    }

    #[test]
    fn growing_tool_input_objects_emit_once_not_as_overlapping_deltas() {
        let mut s = StreamState::new("m");
        let sse = s.frames_to_sse(vec![
            frame("toolUseEvent", r#"{"toolUseId":"t1","name":"run","input":{"cmd":"cat /home"}}"#),
            frame("toolUseEvent", r#"{"toolUseId":"t1","name":"run","input":{"cmd":"cat /home/x"}}"#),
        ]);
        // start frame only; partial inputs must NOT be emitted as deltas
        assert_eq!(sse.matches("\"arguments\"").count(), 1, "only the empty start");
        assert!(sse.contains(r#""arguments":"""#));
        let fin = s.finish();
        assert!(fin.contains(r#"cat /home/x"#), "final canonical args emitted at finish");
        assert_eq!(fin.matches("cat /home\\\"").count(), 0, "no stale prefix");
        assert!(fin.contains(r#""finish_reason":"tool_calls""#));
    }

    /// Real capture: Kiro sends tool arguments as concatenating STRING
    /// fragments plus a trailing {"stop":true} frame with no input.
    #[test]
    fn collect_response_concatenates_string_tool_fragments() {
        let mut raw = Vec::new();
        for p in [
            r#"{"name":"get_weather","toolUseId":"t1"}"#,
            r#"{"input":"","name":"get_weather","toolUseId":"t1"}"#,
            r#"{"input":"{\"city\":","name":"get_weather","toolUseId":"t1"}"#,
            r#"{"input":" \"","name":"get_weather","toolUseId":"t1"}"#,
            r#"{"input":"Dhak","name":"get_weather","toolUseId":"t1"}"#,
            r#"{"input":"a\"}","name":"get_weather","toolUseId":"t1"}"#,
            r#"{"name":"get_weather","stop":true,"toolUseId":"t1"}"#,
        ] {
            raw.extend(build_frame("toolUseEvent", p.as_bytes()));
        }
        let (body, _) = collect_response(&raw, "claude-haiku-4.5", "id", 200_000);
        let call = &body["choices"][0]["message"]["tool_calls"][0];
        assert_eq!(call["function"]["name"], "get_weather");
        let args = call["function"]["arguments"].as_str().unwrap();
        assert_eq!(args, r#"{"city": "Dhaka"}"#);
        let parsed: Value = serde_json::from_str(args).expect("arguments must be valid JSON");
        assert_eq!(parsed["city"], "Dhaka");
        assert_eq!(body["choices"][0]["finish_reason"], "tool_calls");
    }

    /// Same fragments through the streaming path: each string is a delta, so
    /// concatenating them client-side must yield the same JSON.
    #[test]
    fn streaming_tool_fragments_reassemble_client_side() {
        let mut s = StreamState::new("m");
        let sse = s.frames_to_sse(vec![
            frame("toolUseEvent", r#"{"name":"get_weather","toolUseId":"t1"}"#),
            frame("toolUseEvent", r#"{"input":"{\"city\":","name":"get_weather","toolUseId":"t1"}"#),
            frame("toolUseEvent", r#"{"input":" \"Dhaka\"}","name":"get_weather","toolUseId":"t1"}"#),
            frame("toolUseEvent", r#"{"name":"get_weather","stop":true,"toolUseId":"t1"}"#),
        ]);
        let mut assembled = String::new();
        for line in sse.lines().filter(|l| l.starts_with("data: ")) {
            let v: Value = serde_json::from_str(&line[6..]).unwrap();
            if let Some(a) = v["choices"][0]["delta"]["tool_calls"][0]["function"]["arguments"].as_str() {
                assembled.push_str(a);
            }
        }
        assert_eq!(assembled, r#"{"city": "Dhaka"}"#);
        let parsed: Value = serde_json::from_str(&assembled).unwrap();
        assert_eq!(parsed["city"], "Dhaka");
    }

    /// Minimal AWS frame encoder for building fixtures.
    fn build_frame(event_type: &str, payload: &[u8]) -> Vec<u8> {
        let mut headers = Vec::new();
        for (n, v) in [(":event-type", event_type), (":message-type", "event")] {
            headers.push(n.len() as u8);
            headers.extend_from_slice(n.as_bytes());
            headers.push(7);
            headers.extend_from_slice(&(v.len() as u16).to_be_bytes());
            headers.extend_from_slice(v.as_bytes());
        }
        let total = 12 + headers.len() + payload.len() + 4;
        let mut out = Vec::new();
        out.extend_from_slice(&(total as u32).to_be_bytes());
        out.extend_from_slice(&(headers.len() as u32).to_be_bytes());
        out.extend_from_slice(&0u32.to_be_bytes());
        out.extend_from_slice(&headers);
        out.extend_from_slice(payload);
        out.extend_from_slice(&0u32.to_be_bytes());
        out
    }

    #[test]
    fn metering_and_context_usage_are_captured() {
        let mut s = StreamState::new("m");
        s.frames_to_sse(vec![
            frame("contextUsageEvent", r#"{"contextUsagePercentage":2.05}"#),
            frame("meteringEvent", r#"{"unit":"credit","usage":0.0052}"#),
        ]);
        assert!((s.context_pct - 2.05).abs() < 1e-9);
        assert!((s.credits - 0.0052).abs() < 1e-9);
    }
}
