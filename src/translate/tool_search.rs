//! `tool_search`, the served tool that lets a client declare far more function
//! tools than it sends.
//!
//! A client function entry marked `defer_loading: true` is held out of the
//! upstream body; the model calls the reserved function with a regular
//! expression, and the matching definitions are appended to the continuation
//! body so it can call them on the next round. The matcher lives here with the
//! Anthropic blocks the client's transcript records; the deferral itself and
//! the reveal are the router's.

use std::collections::BTreeSet;

use regex::RegexBuilder;
use serde_json::{Value, json};

use crate::router::ClientFormat;

/// The per-call result cap an absent `max_results` falls back to
/// (OpenRouter's default, read 2026-09-18).
pub const DEFAULT_MAX_RESULTS: u64 = 5;

/// The longest pattern the matcher will compile. A model that sends more is
/// pasting, not searching, and a long pattern is where the regex engine's
/// pathological cases live.
pub const MAX_PATTERN_CHARS: usize = 200;

/// The Anthropic native spelling of the tool, as its blocks name it.
pub const ANTHROPIC_NAME: &str = "tool_search_tool_regex";

/// Names of the deferred tools whose definition matches `pattern`, at most
/// `limit` of them, in declaration order.
///
/// `Err` is the message the model is shown: an unusable pattern is an error
/// result, never a request failure.
pub fn search_deferred(
    deferred: &[Value],
    pattern: &str,
    limit: usize,
) -> Result<Vec<String>, String> {
    if pattern.chars().count() > MAX_PATTERN_CHARS {
        return Err(format!(
            "invalid regular expression: longer than {MAX_PATTERN_CHARS} characters"
        ));
    }
    let re = RegexBuilder::new(pattern)
        .case_insensitive(true)
        .size_limit(1 << 20)
        .build()
        .map_err(|e| format!("invalid regular expression: {e}"))?;

    Ok(deferred
        .iter()
        .map(|entry| &entry["function"])
        .filter(|f| searchable_text(f).iter().any(|text| re.is_match(text)))
        .filter_map(|f| f["name"].as_str().map(str::to_string))
        .take(limit)
        .collect())
}

/// Everything of a function definition the pattern is tested against: its
/// name, its description, and every argument's key and description. The
/// argument types and the rest of the JSON Schema are noise a model would
/// match by accident.
fn searchable_text(function: &Value) -> Vec<&str> {
    let mut text: Vec<&str> = Vec::new();
    text.extend(function["name"].as_str());
    text.extend(function["description"].as_str());
    if let Some(properties) = function["parameters"]["properties"].as_object() {
        for (key, property) in properties {
            text.push(key);
            text.extend(property["description"].as_str());
        }
    }
    text
}

/// The deferred tools this request already shows the model using, read off the
/// client's own payload before translation.
///
/// Such a tool is sent up front rather than held back: the model is mid-call,
/// and a definition it has already used must not vanish under it. A tool
/// revealed stays revealed for the turn, so the history is what carries it
/// into the next one.
pub fn revealed_in_history(payload: &Value, client: ClientFormat) -> BTreeSet<String> {
    let mut names = BTreeSet::new();
    let messages = payload["messages"].as_array().into_iter().flatten();
    match client {
        ClientFormat::Anthropic => {
            for block in messages.flat_map(|m| m["content"].as_array().into_iter().flatten()) {
                match block["type"].as_str() {
                    Some("tool_use") => names.extend(block["name"].as_str().map(str::to_string)),
                    // pxy's own result block from an earlier turn: the names it
                    // revealed then are the names the model still expects.
                    Some("tool_search_tool_result") => {
                        let references =
                            block["content"]["tool_references"].as_array().into_iter().flatten();
                        names.extend(
                            references.filter_map(|r| r["tool_name"].as_str().map(str::to_string)),
                        );
                    }
                    _ => {}
                }
            }
        }
        // Chat Completions and Responses share this format but not their
        // history's shape: an assistant turn's `tool_calls` for one, a
        // `function_call` item in `input` for the other.
        ClientFormat::Openai => {
            for call in messages.flat_map(|m| m["tool_calls"].as_array().into_iter().flatten()) {
                names.extend(call["function"]["name"].as_str().map(str::to_string));
            }
            for item in payload["input"].as_array().into_iter().flatten() {
                if item["type"] == "function_call" {
                    names.extend(item["name"].as_str().map(str::to_string));
                }
            }
        }
    }
    names
}

// ---------------------------------------------------------------------------
// Client-facing blocks
// ---------------------------------------------------------------------------

/// `server_tool_use` — the search pxy ran, as the client's transcript records
/// it.
pub fn server_tool_use_block(id: &str, pattern: &str, limit: u64) -> Value {
    json!({
        "type": "server_tool_use",
        "id": id,
        "name": ANTHROPIC_NAME,
        "input": {"pattern": pattern, "limit": limit},
    })
}

/// `tool_search_tool_result` naming the tools the search revealed. The client
/// reads the names off `tool_references`; the definitions themselves it
/// already has, since they are its own.
pub fn result_block(id: &str, found: &[String]) -> Value {
    let references: Vec<Value> = found
        .iter()
        .map(|name| json!({"type": "tool_reference", "tool_name": name}))
        .collect();
    json!({
        "type": "tool_search_tool_result",
        "tool_use_id": id,
        "content": {"type": "tool_search_tool_search_result", "tool_references": references},
    })
}

/// The error shape the API uses when the pattern itself was unusable.
pub fn error_block(id: &str, message: &str) -> Value {
    json!({
        "type": "tool_search_tool_result",
        "tool_use_id": id,
        "content": {
            "type": "tool_search_tool_result_error",
            "error_code": "invalid_tool_input",
            "error_message": message,
        },
    })
}

/// Flatten the blocks pxy emitted back into prose when the client replays them
/// in a later turn. An OpenAI upstream has no notion of a server tool, and
/// dropping the pair would leave the model's own search unaccounted for.
pub fn flatten_history_block(block: &Value) -> Option<String> {
    match block["type"].as_str()? {
        "server_tool_use" if block["name"] == ANTHROPIC_NAME => {
            let pattern = block["input"]["pattern"].as_str().unwrap_or("");
            Some(format!("[tool search: {pattern}]"))
        }
        "tool_search_tool_result" => {
            let content = &block["content"];
            if let Some(message) = content["error_message"].as_str() {
                return Some(format!("[tool search failed] {message}"));
            }
            let names: Vec<&str> = content["tool_references"]
                .as_array()
                .into_iter()
                .flatten()
                .filter_map(|r| r["tool_name"].as_str())
                .collect();
            if names.is_empty() {
                return Some("[tool search found nothing]".into());
            }
            Some(format!(
                "[tools revealed]\n{}",
                names.iter().map(|n| format!("- {n}")).collect::<Vec<_>>().join("\n")
            ))
        }
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Three deferred tools, each field probed by a token that appears in it
    /// and nowhere else.
    fn deferred() -> Vec<Value> {
        vec![
            json!({
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Current conditions for a city.",
                    "parameters": {
                        "type": "object",
                        "properties": {"city": {"type": "string", "description": "IANA city name."}},
                    },
                    "defer_loading": true,
                },
            }),
            json!({
                "type": "function",
                "function": {
                    "name": "send_email",
                    "description": "Deliver a note.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "recipient": {"type": "string", "description": "Where to deliver it."},
                            "body": {"type": "string"},
                        },
                    },
                    "defer_loading": true,
                },
            }),
            json!({
                "type": "function",
                "function": {
                    "name": "list_files",
                    "description": "Enumerate a directory.",
                    "parameters": {
                        "type": "object",
                        "properties": {"path": {"type": "string", "description": "Absolute path."}},
                    },
                    "defer_loading": true,
                },
            }),
        ]
    }

    fn found(pattern: &str) -> Vec<String> {
        search_deferred(&deferred(), pattern, 10).unwrap()
    }

    #[test]
    fn matches_the_name_the_description_and_both_argument_fields() {
        // The name, case-insensitively.
        assert_eq!(found("WEATHER"), vec!["get_weather"]);
        // The function's own description.
        assert_eq!(found("enumerate"), vec!["list_files"]);
        // An argument's key.
        assert_eq!(found("recipient"), vec!["send_email"]);
        // An argument's description.
        assert_eq!(found("absolute"), vec!["list_files"]);
        // A pattern nothing carries.
        assert!(found("kubernetes").is_empty());
    }

    #[test]
    fn limit_truncates_the_matches_in_declaration_order() {
        let all = search_deferred(&deferred(), "e", 10).unwrap();
        assert_eq!(all, vec!["get_weather", "send_email", "list_files"]);
        assert_eq!(search_deferred(&deferred(), "e", 2).unwrap(), vec!["get_weather", "send_email"]);
    }

    #[test]
    fn a_malformed_or_overlong_pattern_is_an_error_the_model_reads() {
        let bad = search_deferred(&deferred(), "get_(weather", 10).unwrap_err();
        assert!(bad.starts_with("invalid regular expression: "), "{bad}");

        let long = search_deferred(&deferred(), &"a".repeat(MAX_PATTERN_CHARS + 1), 10).unwrap_err();
        assert!(long.starts_with("invalid regular expression: "), "{long}");
        // The limit itself is usable.
        assert!(search_deferred(&deferred(), &"a".repeat(MAX_PATTERN_CHARS), 10).is_ok());
    }

    fn names(payload: Value, client: ClientFormat) -> Vec<String> {
        revealed_in_history(&payload, client).into_iter().collect()
    }

    /// A tool the model is already using must go up whole, whatever dialect
    /// the request arrived in.
    #[test]
    fn revealed_in_history_reads_calls_tool_use_blocks_and_references() {
        // Chat Completions: an assistant turn's tool_calls.
        let chat = json!({"messages": [
            {"role": "user", "content": "weather?"},
            {"role": "assistant", "tool_calls": [
                {"id": "c1", "type": "function", "function": {"name": "get_weather", "arguments": "{}"}}
            ]},
            {"role": "tool", "tool_call_id": "c1", "content": "sunny"},
        ]});
        assert_eq!(names(chat, ClientFormat::Openai), vec!["get_weather"]);

        // Responses: a function_call item in the input.
        let responses = json!({"input": [
            {"type": "message", "role": "user", "content": "list them"},
            {"type": "function_call", "call_id": "c1", "name": "list_files", "arguments": "{}"},
        ]});
        assert_eq!(names(responses, ClientFormat::Openai), vec!["list_files"]);

        // Messages: a tool_use block, and the names pxy's own result block
        // revealed last turn.
        let anthropic = json!({"messages": [
            {"role": "assistant", "content": [
                {"type": "tool_use", "id": "c1", "name": "get_weather", "input": {}},
                result_block("srvtoolu_1", &["send_email".to_string()]),
            ]},
        ]});
        assert_eq!(names(anthropic, ClientFormat::Anthropic), vec!["get_weather", "send_email"]);

        // A request with no history reveals nothing.
        assert!(names(json!({"messages": []}), ClientFormat::Openai).is_empty());
    }

    #[test]
    fn history_blocks_flatten_to_prose() {
        let use_block = server_tool_use_block("srvtoolu_1", "weather|forecast", 5);
        assert_eq!(
            flatten_history_block(&use_block).unwrap(),
            "[tool search: weather|forecast]"
        );

        let found = result_block("srvtoolu_1", &["get_weather".to_string()]);
        assert_eq!(flatten_history_block(&found).unwrap(), "[tools revealed]\n- get_weather");

        let empty = result_block("srvtoolu_1", &[]);
        assert_eq!(flatten_history_block(&empty).unwrap(), "[tool search found nothing]");

        let failed = error_block("srvtoolu_1", "invalid regular expression: bad");
        assert_eq!(
            flatten_history_block(&failed).unwrap(),
            "[tool search failed] invalid regular expression: bad"
        );

        // Another tool's blocks are not this tool's to flatten.
        let advisor = json!({"type": "server_tool_use", "name": "advisor", "input": {"prompt": "?"}});
        assert!(flatten_history_block(&advisor).is_none());
        assert!(flatten_history_block(&json!({"type": "text", "text": "hi"})).is_none());
    }

    #[test]
    fn the_blocks_carry_the_pattern_and_the_names() {
        let use_block = server_tool_use_block("srvtoolu_1", "weather", 5);
        assert_eq!(use_block["type"], "server_tool_use");
        assert_eq!(use_block["name"], ANTHROPIC_NAME);
        assert_eq!(use_block["input"], json!({"pattern": "weather", "limit": 5}));

        let result = result_block("srvtoolu_1", &["get_weather".to_string()]);
        assert_eq!(result["type"], "tool_search_tool_result");
        assert_eq!(result["tool_use_id"], "srvtoolu_1");
        assert_eq!(
            result["content"],
            json!({
                "type": "tool_search_tool_search_result",
                "tool_references": [{"type": "tool_reference", "tool_name": "get_weather"}],
            })
        );

        let error = error_block("srvtoolu_1", "invalid regular expression: bad");
        assert_eq!(error["type"], "tool_search_tool_result");
        assert_eq!(
            error["content"],
            json!({
                "type": "tool_search_tool_result_error",
                "error_code": "invalid_tool_input",
                "error_message": "invalid regular expression: bad",
            })
        );
    }
}
