pub mod aggregate;
pub mod anthropic_sanitize;
pub mod cache_control;
pub mod aisdk;
pub mod anthropic_to_openai;
pub mod heal;
pub mod openai_to_anthropic;
pub mod responses;
pub mod responses_upstream;
pub mod server_tools;
pub mod sse;
pub mod think;
pub mod tool_search;
pub mod tool_text;
pub mod web_search;

/// Token usage extracted from a response, in provider-neutral terms.
#[derive(Debug, Clone, Copy, Default)]
pub struct TokenUsage {
    pub input: u64,
    pub output: u64,
}

impl TokenUsage {
    pub fn from_openai(usage: &serde_json::Value) -> Self {
        Self {
            input: usage["prompt_tokens"].as_u64().unwrap_or(0),
            output: usage["completion_tokens"].as_u64().unwrap_or(0),
        }
    }
    pub fn from_anthropic(usage: &serde_json::Value) -> Self {
        // Anthropic's `input_tokens` EXCLUDES cache traffic; the cache fields
        // are where most real input lands on cached agent sessions. Leaving
        // them out silently under-counts every quota budget.
        Self {
            input: usage["input_tokens"].as_u64().unwrap_or(0)
                + usage["cache_creation_input_tokens"].as_u64().unwrap_or(0)
                + usage["cache_read_input_tokens"].as_u64().unwrap_or(0),
            output: usage["output_tokens"].as_u64().unwrap_or(0),
        }
    }
}

/// Rough token estimate over EVERY content block type. (Counting only text
/// blocks broke Claude Code auto-compaction in OmniRoute.) ASCII runs count
/// chars/4; every non-ASCII codepoint counts ~1 token — CJK is the big
/// chars/4 under-count (400 CJK chars ≈ 400 tokens, not 300), and
/// over-estimating the tail scripts slightly is the safe direction: the
/// reactive SkipContextWindow absorbs under-counts with a burned call, but
/// an over-count only skips a model a re-measured request could still fit.
pub fn estimate_tokens(value: &serde_json::Value) -> u64 {
    let mut ascii = 0usize;
    let mut wide = 0usize;
    count_chars(value, &mut ascii, &mut wide);
    (ascii / 4 + wide) as u64
}

/// What one image costs in the estimate, whatever its byte size. Providers
/// bill an image by its pixels, not its base64 (Anthropic: width x height /
/// 750, at most ~1,600 for a 1568px edge; OpenAI's high detail tops out
/// near 1,100), while base64 is 4 chars per 3 bytes: a 25 KB page scan
/// read as text came to ~35k tokens and was skipped as too large for a
/// 32k-window OCR model (2026-09-19). The top of the real range is used, in
/// the over-count direction the doc above prefers.
const IMAGE_TOKENS: usize = 1_600;

/// An image or file part in any dialect pxy takes: OpenAI `image_url`,
/// Anthropic `image` (and `document`, a PDF the file-parser has not yet
/// replaced), Responses `input_image` / `input_file`, and the chat `file`
/// part.
fn is_binary_part(o: &serde_json::Map<String, serde_json::Value>) -> bool {
    matches!(
        o.get("type").and_then(|t| t.as_str()),
        Some("image_url" | "image" | "input_image" | "input_file" | "file" | "document")
    )
}

fn is_data_url(s: &str) -> bool {
    s.starts_with("data:") && s.contains(";base64,")
}

fn count_chars(v: &serde_json::Value, ascii: &mut usize, wide: &mut usize) {
    match v {
        serde_json::Value::String(s) => {
            if is_data_url(s) {
                *ascii += IMAGE_TOKENS * 4;
                return;
            }
            for c in s.chars() {
                if c.is_ascii() {
                    *ascii += 1;
                } else {
                    *wide += 1;
                }
            }
        }
        serde_json::Value::Array(a) => a.iter().for_each(|x| count_chars(x, ascii, wide)),
        serde_json::Value::Object(o) if is_binary_part(o) => *ascii += IMAGE_TOKENS * 4,
        serde_json::Value::Object(o) => o.values().for_each(|x| count_chars(x, ascii, wide)),
        _ => *ascii += 4,
    }
}

/// OpenAI's `developer` role is its `system` role under a newer name, and the
/// OpenAI-compatible providers pxy fronts overwhelmingly reject the literal
/// variant (`deepseek` 400s with "unknown variant `developer`"). pxy forwards
/// an OpenAI-dialect body to an OpenAI upstream verbatim, so the role is
/// rewritten here for every such provider except the ones that opted into the
/// native dialect (`openai_native`). A no-op when there is no
/// `messages` array or no `developer` entry, and content is left untouched.
pub fn developer_role_to_system(body: &mut serde_json::Value) {
    let Some(messages) = body.get_mut("messages").and_then(|m| m.as_array_mut()) else {
        return;
    };
    for msg in messages {
        if msg["role"] == "developer" {
            msg["role"] = serde_json::json!("system");
        }
    }
}

/// OpenAI renamed `max_tokens` to `max_completion_tokens`, and its reasoning
/// models now reject the old field. Third-party OpenAI-compatible providers
/// are the opposite: they honor `max_tokens` and ignore or reject the new one
/// (opencode-go accepted `max_completion_tokens: 1` and answered with 38
/// completion tokens — the cap silently did nothing). pxy's passthrough
/// forwards whichever field the client sent, so the body is normalized to the
/// upstream's spelling here, in either direction. Exactly one of the two
/// fields survives, so no later reader has to guess which cap is real.
/// `use_max_completion_tokens` is the OpenAI/Azure-native case.
pub fn normalize_max_tokens_field(body: &mut serde_json::Value, use_max_completion_tokens: bool) {
    let Some(obj) = body.as_object_mut() else {
        return;
    };
    let (keep, drop) = if use_max_completion_tokens {
        ("max_completion_tokens", "max_tokens")
    } else {
        ("max_tokens", "max_completion_tokens")
    };
    // Move the client's value under the upstream's field name — unless the
    // upstream's field is already there, in which case the explicit one wins
    // and the alias is simply dropped.
    if let Some(v) = obj.remove(drop)
        && !obj.contains_key(keep)
    {
        obj.insert(keep.into(), v);
    }
}

/// Chain-of-thought text from a message or delta. OpenAI-compatible upstreams
/// disagree on the field name — `reasoning_content` (most), `reasoning`
/// (z-ai/GLM), `reasoning_text`, `thinking`, `thought` — and a client that does
/// not recognize the spelling it receives replays nothing. Reading one name
/// silently drops the whole thinking phase. `reasoning_details` is OpenAI's
/// structured form; its text/summary parts are joined.
pub fn reasoning_text(v: &serde_json::Value) -> Option<String> {
    for key in [
        "reasoning_content",
        "reasoning",
        "reasoning_text",
        "thinking",
        "thought",
    ] {
        if let Some(s) = v[key].as_str().filter(|s| !s.is_empty()) {
            return Some(s.to_string());
        }
    }
    let details = v["reasoning_details"].as_array()?;
    let joined: String = details
        .iter()
        .filter_map(|d| {
            d["text"]
                .as_str()
                .or_else(|| d["summary"].as_str())
                .filter(|s| !s.is_empty())
        })
        .collect::<Vec<_>>()
        .join("\n");
    (!joined.is_empty()).then_some(joined)
}

/// Restore a tool name's declared capitalization. Some providers (Gemini,
/// several gateways) lowercase tool names, and a client that matches a tool
/// call by name would see an unexecutable `bash` for its declared `Bash`.
pub fn restore_tool_name(name: &str, declared: Option<&std::collections::HashSet<String>>) -> String {
    if let Some(set) = declared
        && let Some(d) = set.iter().find(|d| d.eq_ignore_ascii_case(name))
    {
        return d.clone();
    }
    name.to_string()
}

/// JSON-Schema annotation keywords some gateways reject. They carry no
/// constraint, so removing them is lossless; `$defs`/`$ref` are structural
/// and stay.
pub fn strip_schema_annotations(schema: &mut serde_json::Value) {
    match schema {
        serde_json::Value::Object(map) => {
            map.remove("$schema");
            map.remove("$id");
            map.remove("$comment");
            for v in map.values_mut() {
                strip_schema_annotations(v);
            }
        }
        serde_json::Value::Array(items) => items.iter_mut().for_each(strip_schema_annotations),
        _ => {}
    }
}

/// Apply `strip_schema_annotations` to a request body's tool definitions, in
/// whichever dialect the body speaks.
pub fn sanitize_tool_schemas(body: &mut serde_json::Value, anthropic: bool) {
    let Some(tools) = body.get_mut("tools").and_then(|t| t.as_array_mut()) else {
        return;
    };
    for t in tools {
        let slot = if anthropic {
            &mut t["input_schema"]
        } else {
            &mut t["function"]["parameters"]
        };
        strip_schema_annotations(slot);
    }
}

/// Provider-specific fields that leak through a passthrough response and break
/// strict OpenAI clients (`x_groq`, for one, is not in the schema and Pydantic
/// rejects unknown root fields). Only clearly foreign `x_`/`x-` keys go; the
/// OpenAI response schema itself is untouched.
pub fn strip_foreign_response_fields(v: &mut serde_json::Value) {
    let is_foreign = |k: &str| {
        let k = k.to_ascii_lowercase();
        k.starts_with("x_") || k.starts_with("x-")
    };
    if let Some(o) = v.as_object_mut() {
        o.retain(|k, _| !is_foreign(k));
    }
    if let Some(choices) = v.get_mut("choices").and_then(|c| c.as_array_mut()) {
        for c in choices {
            if let Some(o) = c.as_object_mut() {
                o.retain(|k, _| !is_foreign(k));
            }
            for slot in ["message", "delta"] {
                if let Some(o) = c.get_mut(slot).and_then(|s| s.as_object_mut()) {
                    o.retain(|k, _| !is_foreign(k));
                }
            }
        }
    }
}

/// DeepSeek-style thinking-mode providers 400 a tool turn whose assistant
/// message omits the reasoning that produced the call ("The
/// `reasoning_content` in the thinking mode must be passed back to the API").
/// Clients behind a proxy usually drop it, so for a provider that requires
/// replay pxy injects the minimum the API accepts — litellm's exact remedy (a
/// single space). Only tool-carrying turns need it; a plain assistant turn is
/// accepted without.
pub fn backfill_openai_reasoning(body: &mut serde_json::Value) {
    let Some(messages) = body.get_mut("messages").and_then(|m| m.as_array_mut()) else {
        return;
    };
    for msg in messages {
        if msg["role"] != "assistant" || !msg["tool_calls"].as_array().is_some_and(|c| !c.is_empty())
        {
            continue;
        }
        if reasoning_text(msg).is_none() {
            msg["reasoning_content"] = serde_json::json!(" ");
        }
    }
}

/// The Anthropic half of `backfill_openai_reasoning`: DeepSeek's Anthropic
/// endpoint rejects a `tool_use` whose turn lacks a `thinking` block ("The
/// `content[].thinking` in the thinking mode must be passed back to the
/// API"). The injected block carries a non-empty signature so the sanitizer
/// (which strips unsigned thinking) keeps it; DeepSeek accepts any signature.
pub fn backfill_anthropic_thinking(body: &mut serde_json::Value) {
    let Some(messages) = body.get_mut("messages").and_then(|m| m.as_array_mut()) else {
        return;
    };
    for msg in messages {
        if msg["role"] != "assistant" {
            continue;
        }
        let Some(blocks) = msg.get_mut("content").and_then(|c| c.as_array_mut()) else {
            continue;
        };
        if !blocks.iter().any(|b| b["type"] == "tool_use") {
            continue;
        }
        if blocks
            .iter()
            .any(|b| matches!(b["type"].as_str(), Some("thinking") | Some("redacted_thinking")))
        {
            continue;
        }
        blocks.insert(
            0,
            serde_json::json!({"type": "thinking", "thinking": " ", "signature": "pxy-replay"}),
        );
    }
}

#[cfg(test)]
mod usage_tests {
    use super::{estimate_tokens, TokenUsage};
    use serde_json::json;

    #[test]
    fn anthropic_usage_includes_cache_tokens() {
        let u = TokenUsage::from_anthropic(&json!({
            "input_tokens": 10,
            "cache_creation_input_tokens": 2000,
            "cache_read_input_tokens": 30000,
            "output_tokens": 50,
        }));
        assert_eq!(u.input, 32010, "cached input is still consumed input");
        assert_eq!(u.output, 50);
        // Absent cache fields (non-caching upstreams) change nothing.
        let plain = TokenUsage::from_anthropic(&json!({"input_tokens": 7, "output_tokens": 3}));
        assert_eq!(plain.input, 7);
    }

    #[test]
    fn developer_role_is_rewritten_to_system() {
        let mut body = json!({"messages": [
            {"role": "developer", "content": "rules"},
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "ok"},
        ]});
        super::developer_role_to_system(&mut body);
        assert_eq!(body["messages"][0]["role"], "system");
        assert_eq!(body["messages"][0]["content"], "rules");
        assert_eq!(body["messages"][1]["role"], "user");
        assert_eq!(body["messages"][2]["role"], "assistant");
    }

    #[test]
    fn developer_role_rewrite_tolerates_other_shapes() {
        // No messages array: nothing to do, and nothing may panic.
        let mut body = json!({"input": "x"});
        super::developer_role_to_system(&mut body);
        assert_eq!(body["input"], "x");
    }

    #[test]
    fn max_completion_tokens_becomes_max_tokens_for_compatible_upstreams() {
        let mut body = json!({"max_completion_tokens": 42, "messages": []});
        super::normalize_max_tokens_field(&mut body, false);
        assert_eq!(body["max_tokens"], 42);
        assert!(body.get("max_completion_tokens").is_none());
    }

    #[test]
    fn max_tokens_becomes_max_completion_tokens_for_openai_native() {
        let mut body = json!({"max_tokens": 7, "messages": []});
        super::normalize_max_tokens_field(&mut body, true);
        assert_eq!(body["max_completion_tokens"], 7);
        assert!(body.get("max_tokens").is_none());
    }

    #[test]
    fn max_tokens_field_keeps_the_upstream_spelling_and_drops_the_alias() {
        // Already the upstream's field: left alone.
        let mut body = json!({"max_tokens": 7});
        super::normalize_max_tokens_field(&mut body, false);
        assert_eq!(body["max_tokens"], 7);
        assert!(body.get("max_completion_tokens").is_none());

        // Both present: the upstream's field wins, the alias is dropped.
        let mut body = json!({"max_tokens": 1, "max_completion_tokens": 2});
        super::normalize_max_tokens_field(&mut body, false);
        assert_eq!(body["max_tokens"], 1, "explicit upstream field wins");
        assert!(body.get("max_completion_tokens").is_none());

        let mut body = json!({"max_tokens": 1, "max_completion_tokens": 2});
        super::normalize_max_tokens_field(&mut body, true);
        assert_eq!(body["max_completion_tokens"], 2);
        assert!(body.get("max_tokens").is_none());

        // Neither present: nothing invented.
        let mut body = json!({"messages": []});
        super::normalize_max_tokens_field(&mut body, false);
        assert!(body.get("max_tokens").is_none());
    }

    #[test]
    fn reasoning_text_reads_every_alias() {
        for key in [
            "reasoning_content",
            "reasoning",
            "reasoning_text",
            "thinking",
            "thought",
        ] {
            let mut v = json!({});
            v[key] = json!("r");
            assert_eq!(super::reasoning_text(&v).as_deref(), Some("r"), "{key}");
        }
        // Structured form: text and summary parts join.
        let v = json!({"reasoning_details": [
            {"type": "reasoning.text", "text": "a"},
            {"type": "reasoning.summary", "summary": "b"},
        ]});
        assert_eq!(super::reasoning_text(&v).as_deref(), Some("a\nb"));
        assert_eq!(super::reasoning_text(&json!({"reasoning_content": ""})), None);
    }

    #[test]
    fn backfill_openai_reasoning_only_fills_tool_turns() {
        let mut body = json!({"messages": [
            {"role": "assistant", "content": "plain"},
            {"role": "assistant", "content": null, "tool_calls": [{"id": "t1"}]},
            {"role": "assistant", "content": null, "tool_calls": [{"id": "t2"}],
             "reasoning_content": "real"},
        ]});
        super::backfill_openai_reasoning(&mut body);
        assert!(body["messages"][0].get("reasoning_content").is_none());
        assert_eq!(body["messages"][1]["reasoning_content"], " ");
        assert_eq!(body["messages"][2]["reasoning_content"], "real");
    }

    #[test]
    fn backfill_anthropic_thinking_only_fills_tool_turns() {
        let mut body = json!({"messages": [
            {"role": "assistant", "content": [{"type": "text", "text": "plain"}]},
            {"role": "assistant", "content": [
                {"type": "tool_use", "id": "t1", "name": "get", "input": {}}]},
            {"role": "assistant", "content": [
                {"type": "thinking", "thinking": "r", "signature": "s"},
                {"type": "tool_use", "id": "t2", "name": "get", "input": {}}]},
        ]});
        super::backfill_anthropic_thinking(&mut body);
        assert_eq!(body["messages"][0]["content"].as_array().unwrap().len(), 1);
        let b = body["messages"][1]["content"].as_array().unwrap();
        assert_eq!(b[0]["type"], "thinking");
        assert_eq!(b[0]["signature"], "pxy-replay");
        assert_eq!(b[1]["type"], "tool_use");
        assert_eq!(body["messages"][2]["content"].as_array().unwrap().len(), 2);
    }

    #[test]
    fn schema_annotations_stripped_but_structure_kept() {
        let mut body = json!({"tools": [{"function": {"name": "f", "parameters": {
            "$schema": "http://json-schema.org/draft-07/schema#",
            "type": "object",
            "properties": {"x": {"$comment": "c", "type": "string"}},
            "$defs": {"D": {"type": "number"}},
            "$ref": "#/$defs/D",
        }}}]});
        super::sanitize_tool_schemas(&mut body, false);
        let p = &body["tools"][0]["function"]["parameters"];
        assert!(p.get("$schema").is_none());
        assert!(p["properties"]["x"].get("$comment").is_none());
        assert_eq!(p["properties"]["x"]["type"], "string");
        assert!(p["$defs"].is_object(), "$defs is structural");
        assert_eq!(p["$ref"], "#/$defs/D");
        // Anthropic dialect slot.
        let mut body = json!({"tools": [{"name": "f", "input_schema": {"$id": "x", "type": "object"}}]});
        super::sanitize_tool_schemas(&mut body, true);
        assert!(body["tools"][0]["input_schema"].get("$id").is_none());
        // serde_json's IndexMut materializes a missing key as Null; a body
        // with no tools must come out with no tools, not `"tools": null`.
        let mut body = json!({"messages": []});
        super::sanitize_tool_schemas(&mut body, false);
        assert!(body.get("tools").is_none(), "must not inject tools: {body}");
    }

    #[test]
    fn foreign_response_fields_are_stripped() {
        let mut v = json!({
            "id": "x", "x_groq": {"id": "y"},
            "choices": [{"index": 0, "x-request-id": "z",
                "message": {"role": "assistant", "content": "hi", "x_groq": {"usage": 1}},
                "delta": {"content": "hi", "x_extra": true}}],
        });
        super::strip_foreign_response_fields(&mut v);
        assert!(v.get("x_groq").is_none());
        assert_eq!(v["id"], "x");
        assert!(v["choices"][0].get("x-request-id").is_none());
        assert!(v["choices"][0]["message"].get("x_groq").is_none());
        assert!(v["choices"][0]["delta"].get("x_extra").is_none());
        assert_eq!(v["choices"][0]["message"]["content"], "hi");
        // A response without choices must not gain `"choices": null`.
        let mut v = json!({"id": "y"});
        super::strip_foreign_response_fields(&mut v);
        assert!(v.get("choices").is_none());
    }

    /// CJK was the big chars/4 under-count: 400 CJK chars are ~400 tokens,
    /// not 1200 bytes/4 = 300. Pure ASCII keeps the chars/4 behavior.
    #[test]
    fn cjk_estimates_higher_than_ascii_quarters() {
        let cjk: String = std::iter::repeat('你').take(400).collect();
        let est = estimate_tokens(&json!({"content": cjk}));
        assert!(est >= 400, "400 CJK chars must estimate >= 400, got {est}");
        let ascii: String = std::iter::repeat('a').take(400).collect();
        assert_eq!(estimate_tokens(&json!({"content": ascii})), 100);
        // Mixed content combines both halves.
        let mixed = estimate_tokens(&json!({"content": format!("{}{}", "a".repeat(400), cjk)}));
        assert!(mixed >= 500, "mixed must be >= 100 + 400, got {mixed}");
    }

    #[test]
    fn estimate_counts_an_image_flat_not_by_its_base64() {
        let b64 = "A".repeat(40_000);
        let openai = serde_json::json!([{"role":"user","content":[
            {"type":"image_url","image_url":{"url":format!("data:image/jpeg;base64,{b64}")}},
            {"type":"text","text":"what is this"}]}]);
        let est = estimate_tokens(&openai);
        assert!(est < 2_000, "one image must cost ~1,600, got {est}");
        let anthropic = serde_json::json!([{"role":"user","content":[
            {"type":"image","source":{"type":"base64","media_type":"image/jpeg","data":b64}}]}]);
        assert!(estimate_tokens(&anthropic) < 2_000);
        // A bare data URL outside a typed part (a chat `file` part's
        // file_data) is an image too.
        let file = serde_json::json!({"file_data": format!("data:application/pdf;base64,{b64}")});
        assert!(estimate_tokens(&file) < 2_000);
        // Text is still text.
        assert_eq!(estimate_tokens(&serde_json::json!("abcdefgh")), 2);
    }

}
