//! Text-embedded tool-call extraction. Free Qwen/GLM/DeepSeek-class models
//! routinely emit tool calls as TEXT instead of the native tool_calls field:
//!
//!   `<tool_call>{"name": "Bash", "arguments": {"cmd": "ls"}}</tool_call>`
//!   `<invoke name="Bash"><parameter name="cmd">ls</parameter></invoke>`
//!
//! Passed through as prose, the client never executes anything and the
//! agent stalls. This filter turns those spans into real tool calls.
//!
//! False-positive containment (an agent's context is FULL of code that may
//! legitimately contain these literals):
//! - extraction only runs when the request declared tools (caller gates);
//! - a parsed call is accepted ONLY when its name matches a declared tool —
//!   anything else is re-emitted as the original text, byte-for-byte;
//! - an opener with no closer within `HOLD_CAP` is flushed back as text.
//!
//! Streaming: emitted text holds back the longest suffix that is a proper
//! prefix of an opener (the ThinkFilter pattern), so markup split across
//! chunk boundaries still assembles.

use std::collections::HashSet;

use serde_json::{json, Value};

/// Longest span we'll buffer waiting for a closing tag before deciding the
/// opener was ordinary text.
const HOLD_CAP: usize = 16 * 1024;

const OPENERS: [&str; 2] = ["<tool_call>", "<invoke "];

pub enum Op {
    Text(String),
    Call { name: String, arguments: String },
}

pub struct ToolTextFilter {
    names: HashSet<String>,
    buf: String,
    in_call: bool,
    /// Extracted call count (drives the finish_reason override).
    pub calls: u64,
}

impl ToolTextFilter {
    pub fn new(names: HashSet<String>) -> Self {
        Self { names, buf: String::new(), in_call: false, calls: 0 }
    }

    pub fn push(&mut self, text: &str) -> Vec<Op> {
        self.buf.push_str(text);
        let mut out = Vec::new();
        loop {
            if self.in_call {
                match self.try_complete() {
                    Some(op) => {
                        self.in_call = false;
                        out.push(op);
                    }
                    None if self.buf.len() > HOLD_CAP => {
                        // No closer in sight: it was just text after all.
                        out.push(Op::Text(std::mem::take(&mut self.buf)));
                        self.in_call = false;
                        break;
                    }
                    None => break, // wait for more input
                }
            } else {
                let opener = OPENERS
                    .iter()
                    .filter_map(|o| self.buf.find(o).map(|p| (p, *o)))
                    .min_by_key(|(p, _)| *p);
                match opener {
                    Some((pos, _)) => {
                        if pos > 0 {
                            out.push(Op::Text(self.buf[..pos].to_string()));
                            self.buf.drain(..pos);
                        }
                        self.in_call = true;
                    }
                    None => {
                        // Emit everything except a trailing proper prefix of
                        // an opener (it may complete in the next chunk).
                        let hold = longest_opener_prefix(&self.buf);
                        let emit_len = self.buf.len() - hold;
                        if emit_len > 0 {
                            out.push(Op::Text(self.buf[..emit_len].to_string()));
                            self.buf.drain(..emit_len);
                        }
                        break;
                    }
                }
            }
        }
        out.retain(|op| !matches!(op, Op::Text(t) if t.is_empty()));
        out
    }

    /// End of stream: whatever is still buffered was ordinary text.
    pub fn flush(&mut self) -> Option<String> {
        self.in_call = false;
        let rest = std::mem::take(&mut self.buf);
        (!rest.is_empty()).then_some(rest)
    }

    /// Buffer starts with an opener; try to parse a complete call. On any
    /// parse/validation failure the raw span is returned as Text so nothing
    /// the model said is ever lost.
    fn try_complete(&mut self) -> Option<Op> {
        if self.buf.starts_with("<tool_call>") {
            let end = self.buf.find("</tool_call>")?;
            let inner = self.buf["<tool_call>".len()..end].trim().to_string();
            let raw: String = self.buf.drain(..end + "</tool_call>".len()).collect();
            return Some(self.validate(parse_json_call(&inner), raw));
        }
        if self.buf.starts_with("<invoke ") {
            let end = self.buf.find("</invoke>")?;
            let span = self.buf[..end + "</invoke>".len()].to_string();
            let raw: String = self.buf.drain(..end + "</invoke>".len()).collect();
            return Some(self.validate(parse_invoke_call(&span), raw));
        }
        // Shouldn't happen (in_call implies an opener at position 0).
        Some(Op::Text(std::mem::take(&mut self.buf)))
    }

    fn validate(&mut self, parsed: Option<(String, String)>, raw: String) -> Op {
        match parsed {
            Some((name, arguments)) if self.names.contains(&name) => {
                self.calls += 1;
                Op::Call { name, arguments }
            }
            _ => Op::Text(raw),
        }
    }
}

/// Longest suffix of `s` that is a proper prefix of any opener. Openers are
/// pure ASCII, so a suffix starting mid-way through a multi-byte char can
/// never match — skip non-boundaries instead of slicing into them (the
/// think.rs lesson: an unguarded slice panics on any CJK/emoji text).
fn longest_opener_prefix(s: &str) -> usize {
    let max_check = OPENERS.iter().map(|o| o.len() - 1).max().unwrap_or(0);
    for take in (1..=max_check.min(s.len())).rev() {
        let start = s.len() - take;
        if !s.is_char_boundary(start) {
            continue;
        }
        if OPENERS.iter().any(|o| o.starts_with(&s[start..])) {
            return take;
        }
    }
    0
}

/// `{"name": "X", "arguments": {...}}` — arguments may also be named
/// args/parameters, be a pre-parsed object, or a JSON-encoded string.
fn parse_json_call(inner: &str) -> Option<(String, String)> {
    let v: Value = serde_json::from_str(inner).ok()?;
    let name = v["name"]
        .as_str()
        .or_else(|| v["tool_name"].as_str())?
        .to_string();
    let args = ["arguments", "args", "parameters"]
        .iter()
        .map(|k| &v[*k])
        .find(|a| !a.is_null());
    let arguments = match args {
        Some(Value::String(s)) => s.clone(),
        Some(obj) => obj.to_string(),
        None => "{}".to_string(),
    };
    Some((name, arguments))
}

/// `<invoke name="X"><parameter name="k">v</parameter>…</invoke>` — all
/// parameter values are strings (the dialect carries no types).
fn parse_invoke_call(span: &str) -> Option<(String, String)> {
    let name_start = span.find("name=\"")? + "name=\"".len();
    let name_end = span[name_start..].find('"')? + name_start;
    let name = span[name_start..name_end].to_string();

    let mut params = serde_json::Map::new();
    let mut rest = &span[name_end..];
    while let Some(p) = rest.find("<parameter name=\"") {
        let key_start = p + "<parameter name=\"".len();
        let key_end = rest[key_start..].find('"')? + key_start;
        let key = rest[key_start..key_end].to_string();
        let val_start = rest[key_end..].find('>')? + key_end + 1;
        let val_end = rest[val_start..].find("</parameter>")? + val_start;
        params.insert(key, json!(rest[val_start..val_end]));
        rest = &rest[val_end..];
    }
    Some((name, Value::Object(params).to_string()))
}

/// Non-streaming: rewrite a complete OpenAI response body in place. Returns
/// true when at least one call was extracted (finish_reason updated too).
pub fn extract_from_response(body: &mut Value, names: &HashSet<String>) -> bool {
    let Some(choices) = body["choices"].as_array_mut() else { return false };
    let mut any = false;
    for choice in choices {
        let Some(text) = choice["message"]["content"].as_str().map(String::from) else { continue };
        // Native tool calls present: the model speaks the protocol; leave it.
        if choice["message"]["tool_calls"].as_array().is_some_and(|c| !c.is_empty()) {
            continue;
        }
        let mut filter = ToolTextFilter::new(names.clone());
        let mut ops = filter.push(&text);
        if let Some(rest) = filter.flush() {
            ops.push(Op::Text(rest));
        }
        if filter.calls == 0 {
            continue;
        }
        let mut kept_text = String::new();
        let mut calls: Vec<Value> = Vec::new();
        for op in ops {
            match op {
                Op::Text(t) => kept_text.push_str(&t),
                Op::Call { name, arguments } => calls.push(json!({
                    "id": format!("textcall_{}", calls.len()),
                    "type": "function",
                    "function": {"name": name, "arguments": arguments},
                })),
            }
        }
        let kept_text = kept_text.trim();
        choice["message"]["content"] =
            if kept_text.is_empty() { Value::Null } else { json!(kept_text) };
        choice["message"]["tool_calls"] = Value::Array(calls);
        choice["finish_reason"] = json!("tool_calls");
        any = true;
    }
    any
}

#[cfg(test)]
mod tests {
    use super::*;

    fn names(list: &[&str]) -> HashSet<String> {
        list.iter().map(|s| s.to_string()).collect()
    }

    fn run(filter: &mut ToolTextFilter, chunks: &[&str]) -> (String, Vec<(String, String)>) {
        let mut text = String::new();
        let mut calls = Vec::new();
        for chunk in chunks {
            for op in filter.push(chunk) {
                match op {
                    Op::Text(t) => text.push_str(&t),
                    Op::Call { name, arguments } => calls.push((name, arguments)),
                }
            }
        }
        if let Some(rest) = filter.flush() {
            text.push_str(&rest);
        }
        (text, calls)
    }

    #[test]
    fn json_dialect_split_across_chunks() {
        let mut f = ToolTextFilter::new(names(&["Bash"]));
        let (text, calls) = run(
            &mut f,
            &["I'll run it now.\n<tool_", "call>{\"name\": \"Bash\", \"argu",
              "ments\": {\"cmd\": \"ls\"}}</tool_call>"],
        );
        assert_eq!(text, "I'll run it now.\n");
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].0, "Bash");
        assert_eq!(
            serde_json::from_str::<Value>(&calls[0].1).unwrap()["cmd"],
            "ls"
        );
        assert_eq!(f.calls, 1);
    }

    #[test]
    fn invoke_dialect_with_parameters() {
        let mut f = ToolTextFilter::new(names(&["Grep"]));
        let (text, calls) = run(
            &mut f,
            &[r#"<invoke name="Grep"><parameter name="pattern">fn main</parameter><parameter name="path">src</parameter></invoke>done"#],
        );
        assert_eq!(text, "done");
        let args: Value = serde_json::from_str(&calls[0].1).unwrap();
        assert_eq!(args["pattern"], "fn main");
        assert_eq!(args["path"], "src");
    }

    #[test]
    fn undeclared_tool_name_is_reemitted_as_text() {
        let mut f = ToolTextFilter::new(names(&["Bash"]));
        let raw = r#"<tool_call>{"name": "rm_rf", "arguments": {}}</tool_call>"#;
        let (text, calls) = run(&mut f, &[raw]);
        assert!(calls.is_empty());
        assert_eq!(text, raw, "unknown tools come back byte-for-byte");
    }

    #[test]
    fn unterminated_opener_flushes_as_text() {
        let mut f = ToolTextFilter::new(names(&["Bash"]));
        let (text, calls) = run(&mut f, &["look at <invoke name=\"Bash\"> in the docs"]);
        assert!(calls.is_empty());
        assert_eq!(text, "look at <invoke name=\"Bash\"> in the docs");
    }

    #[test]
    fn arguments_variants_and_string_form() {
        assert_eq!(
            parse_json_call(r#"{"name":"X","args":{"a":1}}"#).unwrap().1,
            r#"{"a":1}"#
        );
        assert_eq!(
            parse_json_call(r#"{"name":"X","arguments":"{\"a\":1}"}"#).unwrap().1,
            r#"{"a":1}"#
        );
        assert_eq!(parse_json_call(r#"{"name":"X"}"#).unwrap().1, "{}");
    }

    #[test]
    fn plain_text_streams_through_with_partial_hold() {
        let mut f = ToolTextFilter::new(names(&["Bash"]));
        // "<" could start an opener: held until disambiguated.
        let ops = f.push("a < b and more");
        let text: String = ops
            .iter()
            .filter_map(|o| match o {
                Op::Text(t) => Some(t.as_str()),
                _ => None,
            })
            .collect();
        assert!(text.starts_with("a "), "prefix must flow immediately");
        let (rest, calls) = run(&mut f, &[" end"]);
        assert!(calls.is_empty());
        assert_eq!(format!("{text}{rest}"), "a < b and more end");
    }

    #[test]
    fn multibyte_text_never_panics() {
        // The think.rs lesson, relearned: unguarded suffix slicing panics on
        // any CJK/emoji within opener-length distance of the chunk end.
        assert_eq!(longest_opener_prefix("好的，我来"), 0);
        assert_eq!(longest_opener_prefix("看 <tool_"), "<tool_".len());
        let mut f = ToolTextFilter::new(names(&["Bash"]));
        let (text, calls) = run(&mut f, &["好的，我来运行 🚀", "，请稍等"]);
        assert!(calls.is_empty());
        assert_eq!(text, "好的，我来运行 🚀，请稍等");
    }

    #[test]
    fn nonstreaming_response_extraction() {
        let mut body = serde_json::json!({
            "choices": [{"index": 0, "finish_reason": "stop",
                "message": {"role": "assistant",
                    "content": "Running.\n<tool_call>{\"name\":\"Bash\",\"arguments\":{\"cmd\":\"ls\"}}</tool_call>"}}]
        });
        assert!(extract_from_response(&mut body, &names(&["Bash"])));
        let msg = &body["choices"][0]["message"];
        assert_eq!(msg["content"], "Running.");
        assert_eq!(msg["tool_calls"][0]["function"]["name"], "Bash");
        assert_eq!(body["choices"][0]["finish_reason"], "tool_calls");

        // Native tool_calls present: untouched.
        let mut native = serde_json::json!({
            "choices": [{"index": 0, "finish_reason": "tool_calls",
                "message": {"role": "assistant",
                    "content": "<tool_call>{\"name\":\"Bash\"}</tool_call>",
                    "tool_calls": [{"id": "n1", "type": "function",
                        "function": {"name": "Bash", "arguments": "{}"}}]}}]
        });
        let before = native.clone();
        assert!(!extract_from_response(&mut native, &names(&["Bash"])));
        assert_eq!(native, before);
    }
}
