pub mod aggregate;
pub mod anthropic_sanitize;
pub mod anthropic_to_openai;
pub mod eventstream;
pub mod kiro;
pub mod openai_to_anthropic;
pub mod responses;
pub mod sse;
pub mod think;
pub mod tool_text;

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

/// Rough token estimate: chars/4 over EVERY content block type.
/// (Counting only text blocks broke Claude Code auto-compaction in OmniRoute.)
pub fn estimate_tokens(value: &serde_json::Value) -> u64 {
    let mut chars = 0usize;
    count_chars(value, &mut chars);
    (chars / 4) as u64
}

fn count_chars(v: &serde_json::Value, chars: &mut usize) {
    match v {
        serde_json::Value::String(s) => *chars += s.len(),
        serde_json::Value::Array(a) => a.iter().for_each(|x| count_chars(x, chars)),
        serde_json::Value::Object(o) => o.values().for_each(|x| count_chars(x, chars)),
        _ => *chars += 4,
    }
}

#[cfg(test)]
mod usage_tests {
    use super::TokenUsage;
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
}
