pub mod aggregate;
pub mod anthropic_sanitize;
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
