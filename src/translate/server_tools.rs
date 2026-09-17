//! The registry of **server tools**: tools pxy executes itself, OpenRouter
//! style. A client declares one by its `type`; pxy recognises it, injects a
//! reserved OpenAI function the upstream can call, runs the call and feeds the
//! result back inside the same turn.
//!
//! This module owns the vocabulary and the executors. The loop that intercepts
//! the calls and re-issues the request lives in the router; the client-facing
//! blocks live with each tool (`translate/web_search.rs`). Later tools add a
//! [`Tool`] variant, a [`from_type`] arm and an [`Tool::execute`] arm.

use serde_json::{Value, json};
use tracing::{info, warn};

use super::web_search;
use crate::router::App;

/// A tool pxy can serve.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Tool {
    WebSearch,
}

impl Tool {
    /// Every tool this build can serve. Drives the default `[server_tools]
    /// enabled` list, so a tool added here is enabled unless config says
    /// otherwise.
    pub fn implemented() -> &'static [Tool] {
        &[Tool::WebSearch]
    }

    /// The tool's canonical name: the spelling `[server_tools] enabled` lists,
    /// without the reserved prefix.
    pub fn name(self) -> &'static str {
        match self {
            Tool::WebSearch => "web_search",
        }
    }

    /// Run one call for this tool. `Err` means the call could not be served at
    /// all — a missing or unusable argument — so the loop leaves it out of the
    /// replay. A tool that ran and failed reports the failure through
    /// `model_output`, the way the real API does.
    pub async fn execute(self, ctx: &ToolCtx<'_>, args: &Value) -> Result<Ran, String> {
        match self {
            Tool::WebSearch => run_web_search(ctx, args).await,
        }
    }

    /// Map a reserved `pxy_*` function name back to its tool, so the loop can
    /// dispatch a call the upstream returned.
    pub fn from_function_name(name: &str) -> Option<Tool> {
        Tool::implemented().iter().copied().find(|t| function_name(*t) == name)
    }
}

/// What running one served call produced: the text the model is shown as the
/// tool's return, and the material the client's transcript records.
pub struct Ran {
    pub model_output: String,
    pub client: ClientRender,
}

/// The client-facing half of a served call. The loop emits it in the client's
/// dialect without knowing which tool produced it: an Anthropic stream gets
/// the blocks, an OpenAI / Responses client gets the marker payload under the
/// tool's reserved function name.
pub struct ClientRender {
    /// Blocks spliced into an Anthropic stream, in order.
    pub blocks: Vec<Value>,
    /// Payload of the empty-choices marker chunk a Responses client reads.
    pub marker: Value,
}

/// Everything an executor needs besides the call's own arguments: the app, for
/// providers, state and secrets, and the call id the client transcript keys on.
pub struct ToolCtx<'a> {
    pub app: &'a App,
    pub call_id: &'a str,
}

impl<'a> ToolCtx<'a> {
    pub fn new(app: &'a App, call_id: &'a str) -> Self {
        Self { app, call_id }
    }
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
    format!("pxy_{}", tool.name())
}

/// The OpenAI function tool injected into an OpenAI-format body for a served
/// tool. `params` is the declaration's own `parameters`, when it carries one;
/// `web_search` models only its query, so it takes none.
pub fn tool_def(tool: Tool, _params: &Value) -> Value {
    match tool {
        Tool::WebSearch => web_search::tool_def(),
    }
}

/// Run one web_search call through the provider walk. A missing or empty query
/// argument makes the call unserved; a provider failure is reported to the
/// model, the way the real API does.
async fn run_web_search(ctx: &ToolCtx<'_>, args: &Value) -> Result<Ran, String> {
    let query =
        args["query"].as_str().filter(|q| !q.is_empty()).ok_or("missing query")?.to_string();
    let found = crate::media::search::run_search(ctx.app, &query, 5, None).await;
    Ok(match &found {
        Ok((provider, results)) => {
            info!(%query, %provider, hits = results.len(), "web_search served");
            Ran {
                model_output: web_search::results_for_model(results),
                client: ClientRender {
                    blocks: vec![
                        web_search::server_tool_use_block(ctx.call_id, &query),
                        web_search::result_block(ctx.call_id, results),
                    ],
                    marker: json!({"id": ctx.call_id, "query": &query}),
                },
            }
        }
        Err(e) => {
            warn!(%query, error = %e, "web_search failed");
            Ran {
                model_output: format!("Search failed: {e}"),
                client: ClientRender {
                    blocks: vec![
                        web_search::server_tool_use_block(ctx.call_id, &query),
                        web_search::error_block(ctx.call_id),
                    ],
                    marker: json!({"id": ctx.call_id, "query": &query}),
                },
            }
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;

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
        assert_eq!(function_name(Tool::WebSearch), "pxy_web_search");
        let def = tool_def(Tool::WebSearch, &Value::Null);
        assert_eq!(def["function"]["name"], "pxy_web_search");
        assert_eq!(def["function"]["parameters"]["required"], json!(["query"]));
    }

    /// The loop dispatches upstream calls by reserved function name, so every
    /// implemented tool must round-trip through `from_function_name`, and a
    /// reserved name nobody implements must not.
    #[test]
    fn reserved_function_names_round_trip_and_unknowns_do_not() {
        for tool in Tool::implemented() {
            assert_eq!(Tool::from_function_name(&function_name(*tool)), Some(*tool));
        }
        assert_eq!(Tool::from_function_name("pxy_not_a_tool"), None);
    }

    /// `name()` is what `[server_tools] enabled` lists and what the reserved
    /// prefix is built from; a tool whose spelling drifts breaks both.
    #[test]
    fn canonical_names_match_the_registry_spelling() {
        for tool in Tool::implemented() {
            assert_eq!(from_type(&format!("pxy:{}", tool.name())), Some(*tool));
        }
        assert_eq!(Tool::WebSearch.name(), "web_search");
    }
}
