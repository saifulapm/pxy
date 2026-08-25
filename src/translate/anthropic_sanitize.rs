//! Sanitize an Anthropic-format request body before it reaches an
//! Anthropic-format upstream (or the kiro conversationState builder, which
//! consumes the same shape). Every fix here corresponds to a hard 400 from
//! Anthropic's validator, and every one has been produced by a real client:
//!
//! - Thinking blocks with an empty/missing signature: Anthropic verifies
//!   signatures on replay, and only its own blocks carry valid ones. Blocks
//!   that came from non-Anthropic models (or from pxy's own translation)
//!   can never validate — one poisoned block in history means 400 on every
//!   later turn, so they are stripped, not forwarded.
//! - Empty text blocks ("text content blocks must be non-empty"): agent
//!   frameworks emit them routinely.
//! - A `tool_use` with no `tool_result` in the next user turn (Ctrl-C'd
//!   agent turns leave these orphans behind).
//! - A `tool_result` answering no `tool_use` from the previous assistant
//!   turn (converted to visible text rather than dropped).
//! - First message not from `user`, or an empty `messages` array.
//! - Trailing whitespace on the final assistant message.

use serde_json::{json, Value};

pub fn sanitize(body: &mut Value) {
    let Some(messages) = body["messages"].as_array_mut() else { return };
    strip_invalid_blocks(messages);
    repair_tool_pairs(messages);
    ensure_first_user(messages);
    trim_trailing_assistant(messages);
    if messages.is_empty() {
        // All-system requests (title generation) otherwise 400.
        messages.push(json!({"role": "user", "content": "."}));
    }
}

/// Drop unverifiable thinking blocks and empty text blocks; drop messages
/// that end up with no content at all.
fn strip_invalid_blocks(messages: &mut Vec<Value>) {
    for msg in messages.iter_mut() {
        let Some(blocks) = msg["content"].as_array_mut() else { continue };
        blocks.retain(|b| match b["type"].as_str() {
            Some("thinking") => {
                b["signature"].as_str().is_some_and(|s| !s.is_empty())
            }
            Some("redacted_thinking") => {
                b["data"].as_str().is_some_and(|s| !s.is_empty())
            }
            Some("text") => !b["text"].as_str().unwrap_or_default().is_empty(),
            _ => true,
        });
    }
    messages.retain(|m| match &m["content"] {
        Value::String(s) => !s.is_empty(),
        Value::Array(a) => !a.is_empty(),
        _ => false,
    });
}

/// Every `tool_use` must be answered by a `tool_result` in the NEXT user
/// message; every `tool_result` must answer the PREVIOUS assistant turn.
fn repair_tool_pairs(messages: &mut Vec<Value>) {
    let mut i = 0;
    // tool_use ids from the most recent assistant message, not yet consumed
    let mut open_calls: Vec<String> = Vec::new();
    while i < messages.len() {
        match messages[i]["role"].as_str().unwrap_or("") {
            "assistant" => {
                open_calls = tool_use_ids(&messages[i]);
                if open_calls.is_empty() {
                    i += 1;
                    continue;
                }
                // Which of them does the next user message answer?
                let answered: Vec<String> = messages
                    .get(i + 1)
                    .filter(|m| m["role"] == "user")
                    .map(tool_result_ids)
                    .unwrap_or_default();
                let missing: Vec<&String> =
                    open_calls.iter().filter(|id| !answered.contains(id)).collect();
                if missing.is_empty() {
                    i += 1;
                    continue;
                }
                let repairs: Vec<Value> = missing
                    .iter()
                    .map(|id| {
                        json!({
                            "type": "tool_result",
                            "tool_use_id": id,
                            "content": "[No response received]",
                        })
                    })
                    .collect();
                match messages.get_mut(i + 1) {
                    Some(next) if next["role"] == "user" => {
                        // tool_result blocks belong at the front of the turn;
                        // a plain-string turn is converted to block form so
                        // no extra (non-alternating) user message appears.
                        if let Some(s) = next["content"].as_str().map(String::from) {
                            next["content"] = json!([{"type": "text", "text": s}]);
                        }
                        let arr = next["content"].as_array_mut().unwrap();
                        for (pos, r) in repairs.into_iter().enumerate() {
                            arr.insert(pos, r);
                        }
                    }
                    _ => {
                        messages.insert(i + 1, json!({"role": "user", "content": repairs}));
                    }
                }
            }
            "user" => {
                // tool_results answering nothing become visible text so the
                // model still sees the information (dropping loses data).
                if let Some(blocks) = messages[i]["content"].as_array_mut() {
                    for b in blocks.iter_mut() {
                        if b["type"] == "tool_result"
                            && !open_calls
                                .iter()
                                .any(|id| b["tool_use_id"].as_str() == Some(id))
                        {
                            let text = match b["content"].as_str() {
                                Some(s) => s.to_string(),
                                None => b["content"].to_string(),
                            };
                            *b = json!({
                                "type": "text",
                                "text": format!("[Tool result]: {text}"),
                            });
                        }
                    }
                }
                open_calls.clear();
            }
            _ => {}
        }
        i += 1;
    }
}

fn tool_use_ids(msg: &Value) -> Vec<String> {
    msg["content"]
        .as_array()
        .map(|blocks| {
            blocks
                .iter()
                .filter(|b| b["type"] == "tool_use")
                .filter_map(|b| b["id"].as_str().map(String::from))
                .collect()
        })
        .unwrap_or_default()
}

fn tool_result_ids(msg: &Value) -> Vec<String> {
    msg["content"]
        .as_array()
        .map(|blocks| {
            blocks
                .iter()
                .filter(|b| b["type"] == "tool_result")
                .filter_map(|b| b["tool_use_id"].as_str().map(String::from))
                .collect()
        })
        .unwrap_or_default()
}

fn ensure_first_user(messages: &mut Vec<Value>) {
    if messages.first().is_some_and(|m| m["role"] != "user") {
        messages.insert(0, json!({"role": "user", "content": "."}));
    }
}

fn trim_trailing_assistant(messages: &mut [Value]) {
    let Some(last) = messages.last_mut() else { return };
    if last["role"] != "assistant" {
        return;
    }
    match &mut last["content"] {
        Value::String(s) => {
            let trimmed = s.trim_end();
            if trimmed.len() != s.len() {
                *s = trimmed.to_string();
            }
        }
        Value::Array(blocks) => {
            // Only the FINAL text block's trailing whitespace matters.
            if let Some(b) = blocks.iter_mut().rev().find(|b| b["type"] == "text") {
                if let Some(s) = b["text"].as_str() {
                    let trimmed = s.trim_end();
                    if trimmed.len() != s.len() {
                        b["text"] = json!(trimmed);
                    }
                }
            }
        }
        _ => {}
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn msgs(body: &Value) -> &Vec<Value> {
        body["messages"].as_array().unwrap()
    }

    #[test]
    fn strips_unverifiable_thinking_blocks() {
        let mut body = json!({"messages": [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": [
                {"type": "thinking", "thinking": "hmm", "signature": ""},
                {"type": "thinking", "thinking": "hmm2"},
                {"type": "thinking", "thinking": "real", "signature": "sig123"},
                {"type": "text", "text": "answer"},
            ]},
        ]});
        sanitize(&mut body);
        let content = msgs(&body)[1]["content"].as_array().unwrap();
        assert_eq!(content.len(), 2, "empty/missing signatures stripped: {content:?}");
        assert_eq!(content[0]["signature"], "sig123", "signed block survives");
    }

    #[test]
    fn message_emptied_by_stripping_is_dropped() {
        let mut body = json!({"messages": [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": [
                {"type": "thinking", "thinking": "only-poison", "signature": ""},
            ]},
            {"role": "user", "content": "next"},
        ]});
        sanitize(&mut body);
        assert_eq!(msgs(&body).len(), 2);
    }

    #[test]
    fn orphan_tool_use_gets_synthetic_result() {
        // A Ctrl-C'd turn: assistant called a tool, no result ever recorded.
        let mut body = json!({"messages": [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": [
                {"type": "tool_use", "id": "t1", "name": "Bash", "input": {}},
            ]},
            {"role": "user", "content": "continue please"},
        ]});
        sanitize(&mut body);
        let m = msgs(&body);
        // The plain-string user turn is converted to blocks with the repair
        // prepended — no extra non-alternating message.
        assert_eq!(m.len(), 3);
        assert_eq!(m[2]["content"][0]["type"], "tool_result");
        assert_eq!(m[2]["content"][0]["tool_use_id"], "t1");
        assert_eq!(m[2]["content"][1]["text"], "continue please");
    }

    #[test]
    fn partial_answers_are_completed_in_place() {
        let mut body = json!({"messages": [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": [
                {"type": "tool_use", "id": "t1", "name": "A", "input": {}},
                {"type": "tool_use", "id": "t2", "name": "B", "input": {}},
            ]},
            {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "t2", "content": "ok"},
            ]},
        ]});
        sanitize(&mut body);
        let content = msgs(&body)[2]["content"].as_array().unwrap();
        assert_eq!(content.len(), 2);
        assert_eq!(content[0]["tool_use_id"], "t1", "repair prepended");
        assert_eq!(content[1]["tool_use_id"], "t2");
    }

    #[test]
    fn orphan_tool_result_becomes_text() {
        let mut body = json!({"messages": [
            {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "ghost", "content": "data"},
                {"type": "text", "text": "hi"},
            ]},
        ]});
        sanitize(&mut body);
        let content = msgs(&body)[0]["content"].as_array().unwrap();
        assert_eq!(content[0]["type"], "text");
        assert!(content[0]["text"].as_str().unwrap().contains("data"));
    }

    #[test]
    fn first_message_forced_to_user_and_empty_texts_stripped() {
        let mut body = json!({"messages": [
            {"role": "assistant", "content": [
                {"type": "text", "text": ""},
                {"type": "text", "text": "leftover"},
            ]},
        ]});
        sanitize(&mut body);
        let m = msgs(&body);
        assert_eq!(m[0]["role"], "user");
        assert_eq!(m[1]["content"].as_array().unwrap().len(), 1);
    }

    #[test]
    fn trailing_assistant_whitespace_trimmed() {
        let mut body = json!({"messages": [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "answer...  \n\n"},
        ]});
        sanitize(&mut body);
        assert_eq!(msgs(&body)[1]["content"], "answer...");
    }

    #[test]
    fn empty_messages_synthesized() {
        let mut body = json!({"messages": []});
        sanitize(&mut body);
        assert_eq!(msgs(&body)[0]["content"], ".");
    }

    #[test]
    fn healthy_history_untouched() {
        let original = json!({"messages": [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": [
                {"type": "tool_use", "id": "t1", "name": "Bash", "input": {"c": "ls"}},
            ]},
            {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "t1", "content": "ok"},
            ]},
            {"role": "assistant", "content": "done"},
        ], "system": "s", "max_tokens": 5});
        let mut body = original.clone();
        sanitize(&mut body);
        assert_eq!(body, original);
    }
}
