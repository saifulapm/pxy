//! `<think>...</think>` extraction (OmniRoute thinkTagParser pattern).
//!
//! Models like MiniMax-M3, DeepSeek and Qwen embed chain-of-thought as literal
//! `<think>` tags in `content` instead of using `reasoning_content`. Left
//! alone, that text pollutes agent replies AND conversation history (burning
//! context every turn). This filter splits streamed text into (reasoning,
//! content), buffering partial tags that straddle chunk boundaries.

const OPEN: &str = "<think>";
const CLOSE: &str = "</think>";

#[derive(Default)]
pub struct ThinkFilter {
    inside: bool,
    /// Undecided tail that may be a partial tag split across chunks.
    buffer: String,
}

impl ThinkFilter {
    pub fn new() -> Self {
        Self::default()
    }

    /// Feed a text delta; returns (reasoning_out, content_out).
    pub fn push(&mut self, text: &str) -> (String, String) {
        self.buffer.push_str(text);
        let mut reasoning = String::new();
        let mut content = String::new();

        loop {
            let tag = if self.inside { CLOSE } else { OPEN };
            if let Some(idx) = self.buffer.find(tag) {
                let before: String = self.buffer[..idx].to_string();
                self.buffer.drain(..idx + tag.len());
                if self.inside {
                    reasoning.push_str(&before);
                } else {
                    content.push_str(&before);
                }
                self.inside = !self.inside;
                continue;
            }
            // No full tag: keep only a tail that could still become one.
            let keep = partial_suffix_len(&self.buffer, tag);
            let emit_len = self.buffer.len() - keep;
            let emitted: String = self.buffer[..emit_len].to_string();
            self.buffer.drain(..emit_len);
            if self.inside {
                reasoning.push_str(&emitted);
            } else {
                content.push_str(&emitted);
            }
            break;
        }
        (reasoning, content)
    }

    /// End of stream: flush any held-back partial tag. An unclosed
    /// `<think>` means the remainder was reasoning (OmniRoute rule).
    pub fn flush(&mut self) -> (String, String) {
        let rest = std::mem::take(&mut self.buffer);
        if self.inside {
            (rest, String::new())
        } else {
            (String::new(), rest)
        }
    }
}

/// Length of the longest suffix of `s` that is a proper prefix of `tag`.
fn partial_suffix_len(s: &str, tag: &str) -> usize {
    let max = (tag.len() - 1).min(s.len());
    for keep in (1..=max).rev() {
        if !s.is_char_boundary(s.len() - keep) {
            continue;
        }
        if tag.starts_with(&s[s.len() - keep..]) {
            return keep;
        }
    }
    0
}

/// One-shot extraction for non-streaming responses.
/// Returns (reasoning, content); reasoning is None when no tags were found.
pub fn extract(text: &str) -> (Option<String>, String) {
    if !text.contains(OPEN) {
        return (None, text.to_string());
    }
    let mut f = ThinkFilter::new();
    let (mut reasoning, mut content) = f.push(text);
    let (r2, c2) = f.flush();
    reasoning.push_str(&r2);
    content.push_str(&c2);
    let reasoning = reasoning.trim().to_string();
    (
        if reasoning.is_empty() { None } else { Some(reasoning) },
        content.trim_start().to_string(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extracts_leading_block() {
        let (r, c) = extract("<think>step 1</think>answer");
        assert_eq!(r.as_deref(), Some("step 1"));
        assert_eq!(c, "answer");
    }

    #[test]
    fn unclosed_tag_is_all_reasoning() {
        let (r, c) = extract("<think>never closed");
        assert_eq!(r.as_deref(), Some("never closed"));
        assert_eq!(c, "");
    }

    #[test]
    fn no_tags_untouched() {
        let (r, c) = extract("plain answer with < brackets >");
        assert!(r.is_none());
        assert_eq!(c, "plain answer with < brackets >");
    }

    #[test]
    fn streaming_split_across_chunks() {
        let mut f = ThinkFilter::new();
        // tag split mid-token: "<thi" + "nk>reason</th" + "ink>done"
        let (r1, c1) = f.push("<thi");
        assert_eq!((r1.as_str(), c1.as_str()), ("", ""));
        let (r2, c2) = f.push("nk>reason</th");
        assert_eq!(r2, "reason");
        assert_eq!(c2, "");
        let (r3, c3) = f.push("ink>done");
        assert_eq!(r3, "");
        assert_eq!(c3, "done");
        let (rf, cf) = f.flush();
        assert_eq!((rf.as_str(), cf.as_str()), ("", ""));
    }

    #[test]
    fn partial_lookalike_is_released() {
        let mut f = ThinkFilter::new();
        // "<t" could start a tag -> held; "able>" disproves it -> released
        let (_, c1) = f.push("use <t");
        assert_eq!(c1, "use ");
        let (_, c2) = f.push("able> tags");
        assert_eq!(c2, "<table> tags");
    }

    #[test]
    fn multiple_blocks() {
        let (r, c) = extract("<think>a</think>mid<think>b</think>end");
        assert_eq!(r.as_deref(), Some("ab"));
        assert_eq!(c, "midend");
    }
}
