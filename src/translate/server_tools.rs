//! The registry of **server tools**: tools pxy executes itself, OpenRouter
//! style. A client declares one by its `type`; pxy recognises it, injects a
//! reserved OpenAI function the upstream can call, runs the call and feeds the
//! result back inside the same turn.
//!
//! This module owns the vocabulary and nothing else. The loop that intercepts
//! the calls and re-issues the request lives in the router; the client-facing
//! blocks live with each tool (`translate/web_search.rs`). Later tools extend
//! [`Tool`] and [`from_type`]; `web_search` is the first and only entry today.

use serde_json::Value;

use super::web_search;

/// A tool pxy can serve.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Tool {
    WebSearch,
}

/// Map a declared tool's `type` to the tool pxy serves for it.
///
/// `openrouter:*` is accepted so an OpenRouter-shaped client can point at pxy
/// unchanged; `pxy:*` is the native spelling. The bare `web_search*` names are
/// the Anthropic and Responses spellings (`web_search`, `web_search_preview`,
/// and dated variants like `web_search_20250305`). Anything else is a server
/// tool pxy does not know, and `None` lets the caller treat it as unservable.
pub fn from_type(ty: &str) -> Option<Tool> {
    match ty {
        "openrouter:web_search" | "pxy:web_search" => Some(Tool::WebSearch),
        t if t.starts_with("web_search") => Some(Tool::WebSearch),
        _ => None,
    }
}

/// The reserved function name the upstream sees for a served tool. The `pxy_`
/// prefix keeps it from colliding with a client tool that merely shares the
/// tool's own name.
pub fn function_name(tool: Tool) -> String {
    match tool {
        Tool::WebSearch => web_search::TOOL_NAME.to_string(),
    }
}

/// The OpenAI function tool injected into an OpenAI-format body for a served
/// tool. `params` is the declaration's own `parameters`, when it carries one;
/// `web_search` models only its query, so it takes none.
pub fn tool_def(tool: Tool, _params: &Value) -> Value {
    match tool {
        Tool::WebSearch => web_search::tool_def(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn from_type_maps_every_web_search_spelling() {
        for ty in [
            "openrouter:web_search",
            "pxy:web_search",
            "web_search_20250305",
            "web_search_preview",
            // The bare Responses name and future dated variants match too.
            "web_search",
            "web_search_20260209",
        ] {
            assert_eq!(from_type(ty), Some(Tool::WebSearch), "{ty}");
        }
    }

    #[test]
    fn from_type_rejects_unknown_types() {
        assert_eq!(from_type("openrouter:not_a_tool"), None);
        assert_eq!(from_type("code_execution_20250522"), None);
        assert_eq!(from_type("function"), None);
    }

    #[test]
    fn function_name_and_def_carry_the_reserved_prefix() {
        assert_eq!(function_name(Tool::WebSearch), web_search::TOOL_NAME);
        let def = tool_def(Tool::WebSearch, &Value::Null);
        assert_eq!(def["function"]["name"], "pxy_web_search");
        assert_eq!(def["function"]["parameters"]["required"], json!(["query"]));
    }
}
