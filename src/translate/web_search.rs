//! Anthropic's `web_search` **server** tool, run by pxy for upstreams that
//! can't run it themselves.
//!
//! Claude Code declares `{"type":"web_search_20250305","name":"web_search"}` —
//! a tool with no `input_schema`, because the API is supposed to execute it and
//! hand back `server_tool_use` + `web_search_tool_result` blocks. An
//! OpenAI-compatible upstream knows nothing about that: forwarded as-is by the
//! generic tool mapping it became a zero-argument function, the model called it
//! with `{}`, and the client got a `tool_use` for a tool it never registered —
//! "Did 0 searches".
//!
//! So on the way out the server tool is swapped for [`TOOL_NAME`], a real
//! function with a `query` parameter; the router intercepts calls to it, runs
//! them through `/v1/search`'s provider walk, and feeds the results back to the
//! model. What reaches the client is the protocol shape it expects.

use serde_json::{Value, json};

/// The function pxy substitutes for the server tool. Prefixed so it can never
/// collide with a client tool named `web_search` (Claude Code's own client
/// tools are what the rest of the `tools` array carries).
pub const TOOL_NAME: &str = "pxy_web_search";

/// Anthropic ships dated variants (`web_search_20250305`, `_20260209`, …), so
/// match on the prefix. A `function` field means it's an ordinary function tool
/// that merely happens to be named that, and is left alone.
fn is_server_tool(tool: &Value) -> bool {
    tool["type"].as_str().is_some_and(|t| t.starts_with("web_search"))
        && tool.get("function").is_none()
}

/// How many searches this request allows, when it asked for search at all.
/// `max_uses` is optional in the tool definition; [`DEFAULT_MAX_USES`] stands
/// in for an absent one so a model that loops can't spend the search quota.
pub struct Plan {
    pub max_uses: u64,
}

pub const DEFAULT_MAX_USES: u64 = 5;

pub fn plan(payload: &Value) -> Option<Plan> {
    let tools = payload["tools"].as_array()?;
    let tool = tools.iter().find(|t| is_server_tool(t))?;
    Some(Plan {
        max_uses: tool["max_uses"].as_u64().unwrap_or(DEFAULT_MAX_USES).clamp(1, 20),
    })
}

/// The OpenAI function tool that replaces the server tool. Only `query` is
/// modelled: pxy's search providers take a query and a count, and every extra
/// knob is one more thing a small model can get wrong.
pub fn tool_def() -> Value {
    json!({
        "type": "function",
        "function": {
            "name": TOOL_NAME,
            "description": "Search the public web and return cited results \
                (title, url, snippet). Use it whenever the answer depends on \
                current events, prices, releases, or anything else that may \
                have changed since training.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query.",
                    },
                },
                "required": ["query"],
            },
        }
    })
}

// ---------------------------------------------------------------------------
// Client-facing blocks
// ---------------------------------------------------------------------------

/// `server_tool_use` — the query pxy ran, as the client's transcript records
/// it. The id prefix mirrors the real API's `srvtoolu_`.
pub fn server_tool_use_block(id: &str, query: &str) -> Value {
    json!({
        "type": "server_tool_use",
        "id": id,
        "name": "web_search",
        "input": {"query": query},
    })
}

/// `web_search_tool_result` carrying the hits. Claude Code reads `title` and
/// `url` off each entry; `encrypted_content` is the real API's replay token and
/// has no meaning here, so the snippet rides along as the page content instead.
pub fn result_block(id: &str, results: &[Value]) -> Value {
    let content: Vec<Value> = results
        .iter()
        .map(|r| {
            json!({
                "type": "web_search_result",
                "url": r["url"].as_str().unwrap_or(""),
                "title": r["title"].as_str().unwrap_or(""),
                "page_age": Value::Null,
            })
        })
        .collect();
    json!({"type": "web_search_tool_result", "tool_use_id": id, "content": content})
}

/// The error shape the API uses when a search fails: `content` becomes a single
/// error object instead of a list. `unavailable` is the catch-all code.
pub fn error_block(id: &str) -> Value {
    json!({
        "type": "web_search_tool_result",
        "tool_use_id": id,
        "content": {"type": "web_search_tool_result_error", "error_code": "unavailable"},
    })
}

/// What the model is shown as the tool's return value.
pub fn results_for_model(results: &[Value]) -> String {
    if results.is_empty() {
        return "No results found.".into();
    }
    results
        .iter()
        .enumerate()
        .map(|(i, r)| {
            format!(
                "[{}] {}\n{}\n{}",
                i + 1,
                r["title"].as_str().unwrap_or(""),
                r["url"].as_str().unwrap_or(""),
                r["snippet"].as_str().unwrap_or("")
            )
        })
        .collect::<Vec<_>>()
        .join("\n\n")
}

/// Flatten the blocks pxy emitted back into text when the client replays them
/// in a later turn. They are pxy's own invention as far as the upstream is
/// concerned — an OpenAI model has no `server_tool_use` — but dropping them
/// would lose what the search found, so the history keeps them as prose.
pub fn flatten_history_block(block: &Value) -> Option<String> {
    match block["type"].as_str()? {
        "server_tool_use" => {
            let query = block["input"]["query"].as_str().unwrap_or("");
            Some(format!("[web search: {query}]"))
        }
        "web_search_tool_result" => {
            let Some(items) = block["content"].as_array() else {
                return Some("[web search failed]".into());
            };
            let lines: Vec<String> = items
                .iter()
                .map(|r| {
                    format!(
                        "- {} ({})",
                        r["title"].as_str().unwrap_or(""),
                        r["url"].as_str().unwrap_or("")
                    )
                })
                .collect();
            Some(format!("[web search results]\n{}", lines.join("\n")))
        }
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detects_dated_server_tool_and_ignores_functions() {
        let payload = json!({"tools": [
            {"type": "web_search_20250305", "name": "web_search", "max_uses": 3},
            {"name": "Bash", "input_schema": {"type": "object"}},
        ]});
        assert_eq!(plan(&payload).unwrap().max_uses, 3);

        // A custom function that merely shares the name is not the server tool.
        let payload = json!({"tools": [
            {"type": "function", "function": {"name": "web_search_helper"}},
        ]});
        assert!(plan(&payload).is_none());
    }

    #[test]
    fn max_uses_defaults_and_clamps() {
        let p = plan(&json!({"tools": [{"type": "web_search_20250305"}]})).unwrap();
        assert_eq!(p.max_uses, DEFAULT_MAX_USES);
        let p = plan(&json!({"tools": [{"type": "web_search_20250305", "max_uses": 900}]})).unwrap();
        assert_eq!(p.max_uses, 20);
    }

    #[test]
    fn history_blocks_flatten_to_prose() {
        let q = server_tool_use_block("srvtoolu_1", "rust async");
        assert_eq!(flatten_history_block(&q).unwrap(), "[web search: rust async]");

        let r = result_block("srvtoolu_1", &[json!({"title": "T", "url": "u"})]);
        assert_eq!(
            flatten_history_block(&r).unwrap(),
            "[web search results]\n- T (u)"
        );
        assert_eq!(
            flatten_history_block(&error_block("srvtoolu_1")).unwrap(),
            "[web search failed]"
        );
        assert!(flatten_history_block(&json!({"type": "text", "text": "hi"})).is_none());
    }
}
