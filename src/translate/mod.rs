pub mod aggregate;
pub mod anthropic_sanitize;
pub mod cache_control;
pub mod aisdk;
pub mod anthropic_to_openai;
pub mod openai_to_anthropic;
pub mod responses;
pub mod sse;
pub mod think;
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

fn count_chars(v: &serde_json::Value, ascii: &mut usize, wide: &mut usize) {
    match v {
        serde_json::Value::String(s) => {
            for c in s.chars() {
                if c.is_ascii() {
                    *ascii += 1;
                } else {
                    *wide += 1;
                }
            }
        }
        serde_json::Value::Array(a) => a.iter().for_each(|x| count_chars(x, ascii, wide)),
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
    let Some(messages) = body["messages"].as_array_mut() else {
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
}
