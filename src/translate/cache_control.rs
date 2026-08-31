//! Inject prompt-cache breakpoints into an Anthropic-bound request that has
//! none (docs/11 §4.1).
//!
//! On the paid Anthropic-format reserves the dominant input cost is the
//! transcript replayed every turn; a cache hit prices that prefix at ~10% and
//! is lossless. Anthropic-dialect clients (Claude Code) set their own markers
//! and pass through untouched — the gap is OpenAI-dialect clients (codex,
//! opencode, fx), whose protocol has no way to express one, so pxy sets the
//! standard ones for them. Shape follows CLIProxyAPI's ensureCacheControl and
//! litellm's anthropic_cache_control_hook:
//!
//! - **Yield-to-client** (mandatory): any client-set marker anywhere stops
//!   injection cold — fighting a client that knows what it's doing corrupts
//!   its TTLs and can blow the cap.
//! - Last system block (covers tools + system, the stable prefix); the last
//!   tool only when there is no system prompt.
//! - The last two cacheable messages: the last one writes this turn's entry,
//!   the second-to-last aligns with the previous turn's marker so that entry
//!   is actually read (lookup happens at the current request's breakpoints).
//! - Anthropic caps breakpoints at 4; injection stays at ≤3.
//! - A message whose final block is thinking-like cannot host a marker and is
//!   skipped whole (the API rejects markers there).
//!
//! Gated per provider (`inject_cache_control`, default off): several
//! OpenAI-compatible-but-Anthropic-format gateways 400 on the field, so it is
//! enabled per aggregator only after `cache_read_input_tokens` on turn 2+
//! proves the marker is relayed.

use serde_json::{json, Value};

/// pxy injects at most this many of Anthropic's 4 allowed breakpoints.
const MAX_INJECTED: usize = 3;

fn marker() -> Value {
    json!({"type": "ephemeral"})
}

pub fn inject(body: &mut Value) {
    if client_set_any(body) {
        return;
    }
    let mut left = MAX_INJECTED;
    if mark_system(body) || mark_last_tool(body) {
        left -= 1;
    }
    let Some(messages) = body["messages"].as_array_mut() else { return };
    let mut marked = 0;
    for msg in messages.iter_mut().rev() {
        if left == 0 || marked == 2 {
            break;
        }
        if mark_message(msg) {
            left -= 1;
            marked += 1;
        }
    }
}

/// Any cache_control the client set itself: system blocks, message content
/// blocks, tool entries.
fn client_set_any(body: &Value) -> bool {
    let has = |v: &Value| v["cache_control"].is_object();
    let block_arrays = [&body["system"], &body["tools"]];
    if block_arrays
        .iter()
        .filter_map(|v| v.as_array())
        .flatten()
        .any(has)
    {
        return true;
    }
    body["messages"]
        .as_array()
        .into_iter()
        .flatten()
        .filter_map(|m| m["content"].as_array())
        .flatten()
        .any(has)
}

/// Mark the LAST system block; a string system converts to block form first
/// (both shapes are valid Anthropic).
fn mark_system(body: &mut Value) -> bool {
    if let Some(s) = body["system"].as_str() {
        if s.is_empty() {
            return false;
        }
        body["system"] = json!([{"type": "text", "text": s, "cache_control": marker()}]);
        return true;
    }
    let Some(blocks) = body["system"].as_array_mut() else { return false };
    let Some(last) = blocks.last_mut().filter(|b| b.is_object()) else { return false };
    last["cache_control"] = marker();
    true
}

/// Mark the last tool definition — only reached when there is no system
/// prompt (tools precede system in the cached prefix, so a system marker
/// already covers them).
fn mark_last_tool(body: &mut Value) -> bool {
    let Some(tools) = body["tools"].as_array_mut() else { return false };
    let Some(last) = tools.last_mut().filter(|t| t.is_object()) else { return false };
    last["cache_control"] = marker();
    true
}

/// Mark a message's final content block. False (skip to an earlier message)
/// when the block cannot host a marker: thinking-like blocks are rejected by
/// the API. String content converts to block form.
fn mark_message(msg: &mut Value) -> bool {
    if let Some(s) = msg["content"].as_str() {
        if s.is_empty() {
            return false;
        }
        msg["content"] = json!([{"type": "text", "text": s, "cache_control": marker()}]);
        return true;
    }
    let Some(blocks) = msg["content"].as_array_mut() else { return false };
    let Some(last) = blocks.last_mut().filter(|b| b.is_object()) else { return false };
    if matches!(last["type"].as_str(), Some("thinking") | Some("redacted_thinking")) {
        return false;
    }
    last["cache_control"] = marker();
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    fn count(body: &Value) -> usize {
        let mut n = 0;
        for v in [&body["system"], &body["tools"]] {
            n += v.as_array().into_iter().flatten().filter(|b| b["cache_control"].is_object()).count();
        }
        n += body["messages"]
            .as_array()
            .into_iter()
            .flatten()
            .filter_map(|m| m["content"].as_array())
            .flatten()
            .filter(|b| b["cache_control"].is_object())
            .count();
        n
    }

    #[test]
    fn standard_breakpoints_system_plus_last_two_messages() {
        let mut body = json!({
            "system": "you are helpful",
            "messages": [
                {"role": "user", "content": "turn 1"},
                {"role": "assistant", "content": [{"type": "text", "text": "a1"}]},
                {"role": "user", "content": [{"type": "text", "text": "turn 2"}]},
            ],
            "tools": [{"name": "Bash", "input_schema": {"type": "object"}}],
        });
        inject(&mut body);
        // String system converted to block form and marked.
        assert_eq!(body["system"][0]["cache_control"]["type"], "ephemeral", "{body}");
        assert_eq!(body["system"][0]["text"], "you are helpful");
        // Last two messages marked (string content converted), the first not.
        assert!(body["messages"][2]["content"][0]["cache_control"].is_object());
        assert!(body["messages"][1]["content"][0]["cache_control"].is_object());
        assert!(body["messages"][0]["content"].is_string(), "untouched: {body}");
        // Tools NOT marked (system covers them), total within the cap.
        assert!(!body["tools"][0]["cache_control"].is_object());
        assert_eq!(count(&body), 3);
    }

    /// litellm's mandatory rule: a client that set any marker anywhere is in
    /// charge — injection must not add to (or fight) its breakpoints.
    #[test]
    fn client_markers_stop_injection() {
        let orig = json!({
            "system": [{"type": "text", "text": "s", "cache_control": {"type": "ephemeral", "ttl": "1h"}}],
            "messages": [{"role": "user", "content": [{"type": "text", "text": "hi"}]}],
        });
        let mut body = orig.clone();
        inject(&mut body);
        assert_eq!(body, orig, "client-set markers must be preserved verbatim");

        let orig = json!({
            "messages": [{"role": "user", "content": [
                {"type": "text", "text": "hi", "cache_control": {"type": "ephemeral"}}]}],
        });
        let mut body = orig.clone();
        inject(&mut body);
        assert_eq!(body, orig);
    }

    /// A final assistant turn ending in a thinking block cannot host a
    /// marker; the message is skipped whole and an earlier one marked.
    #[test]
    fn thinking_final_blocks_are_skipped() {
        let mut body = json!({
            "messages": [
                {"role": "user", "content": "q1"},
                {"role": "user", "content": "q2"},
                {"role": "assistant", "content": [
                    {"type": "text", "text": "a"},
                    {"type": "thinking", "thinking": "…", "signature": "sig"},
                ]},
            ],
        });
        inject(&mut body);
        let blocks = body["messages"][2]["content"].as_array().unwrap();
        assert!(blocks.iter().all(|b| !b["cache_control"].is_object()), "{body}");
        assert!(body["messages"][1]["content"][0]["cache_control"].is_object());
        assert!(body["messages"][0]["content"][0]["cache_control"].is_object());
    }

    /// Without a system prompt the last tool hosts the prefix marker.
    #[test]
    fn tools_marked_only_without_system() {
        let mut body = json!({
            "messages": [{"role": "user", "content": "hi"}],
            "tools": [
                {"name": "A", "input_schema": {"type": "object"}},
                {"name": "B", "input_schema": {"type": "object"}},
            ],
        });
        inject(&mut body);
        assert!(!body["tools"][0]["cache_control"].is_object());
        assert!(body["tools"][1]["cache_control"].is_object(), "{body}");
        assert_eq!(count(&body), 2);
    }

    /// tool_use / tool_result final blocks host markers fine — an agent
    /// transcript rarely ends in plain text.
    #[test]
    fn tool_blocks_host_markers() {
        let mut body = json!({
            "system": [{"type": "text", "text": "s"}],
            "messages": [
                {"role": "assistant", "content": [
                    {"type": "tool_use", "id": "t1", "name": "Bash", "input": {}}]},
                {"role": "user", "content": [
                    {"type": "tool_result", "tool_use_id": "t1", "content": "ok"}]},
            ],
        });
        inject(&mut body);
        assert!(body["messages"][0]["content"][0]["cache_control"].is_object());
        assert!(body["messages"][1]["content"][0]["cache_control"].is_object());
        assert_eq!(count(&body), 3);
    }
}
