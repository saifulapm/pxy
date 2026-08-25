pub mod anthropic_to_openai;
pub mod eventstream;
pub mod openai_to_anthropic;
pub mod sse;
pub mod think;

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
        Self {
            input: usage["input_tokens"].as_u64().unwrap_or(0),
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
