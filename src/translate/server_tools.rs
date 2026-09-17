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
    WebFetch,
}

impl Tool {
    /// Every tool this build can serve. Drives the default `[server_tools]
    /// enabled` list, so a tool added here is enabled unless config says
    /// otherwise.
    pub fn implemented() -> &'static [Tool] {
        &[Tool::WebSearch, Tool::WebFetch]
    }

    /// The tool's canonical name: the spelling `[server_tools] enabled` lists,
    /// without the reserved prefix.
    pub fn name(self) -> &'static str {
        match self {
            Tool::WebSearch => "web_search",
            Tool::WebFetch => "web_fetch",
        }
    }

    /// Run one call for this tool. `Err` means the call could not be served at
    /// all — a missing or unusable argument — so the loop leaves it out of the
    /// replay. A tool that ran and failed reports the failure through
    /// `model_output`, the way the real API does.
    pub async fn execute(self, ctx: &ToolCtx<'_>, args: &Value) -> Result<Ran, String> {
        match self {
            Tool::WebSearch => run_web_search(ctx, args).await,
            Tool::WebFetch => run_web_fetch(ctx, args).await,
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
        "openrouter:web_fetch" | "pxy:web_fetch" => Some(Tool::WebFetch),
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
/// neither tool's model-facing call takes anything from it yet — `web_search`
/// models only its query, `web_fetch` only its URL.
pub fn tool_def(tool: Tool, _params: &Value) -> Value {
    match tool {
        Tool::WebSearch => web_search::tool_def(),
        Tool::WebFetch => json!({
            "type": "function",
            "function": {
                "name": function_name(Tool::WebFetch),
                "description": "Fetch a URL and return its extracted text.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "url": {"type": "string", "description": "The http(s) URL to fetch."}
                    },
                    "required": ["url"],
                },
            },
        }),
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

/// How much of a fetched page the model is shown. One fetch must not be able
/// to swallow the context window; the tail is dropped, not summarised.
const FETCH_CONTENT_CHARS: usize = 100_000;

/// The model-facing text for a fetched page: its URL, then the extracted
/// content, capped.
fn fetched_for_model(url: &str, content: &str) -> String {
    let body = match content.char_indices().nth(FETCH_CONTENT_CHARS) {
        Some((cut, _)) => format!("{}\n\n[truncated]", &content[..cut]),
        None => content.to_string(),
    };
    format!("Content of {url}:\n\n{body}")
}

/// Run one web_fetch call through the fetch provider walk (the same walk
/// `/v1/fetch` runs). A missing URL makes the call unserved; a provider
/// failure is reported to the model. web_fetch has no documented Anthropic
/// result block, so it renders nothing for a Messages client.
async fn run_web_fetch(ctx: &ToolCtx<'_>, args: &Value) -> Result<Ran, String> {
    let url = args["url"].as_str().filter(|u| !u.is_empty()).ok_or("missing url")?.to_string();
    let found = crate::media::search::run_fetch(ctx.app, &url, None).await;
    Ok(match &found {
        Ok((provider, content)) => {
            info!(%url, %provider, bytes = content.len(), "web_fetch served");
            Ran {
                model_output: fetched_for_model(&url, content),
                client: ClientRender {
                    blocks: Vec::new(),
                    marker: json!({"id": ctx.call_id, "url": &url}),
                },
            }
        }
        Err(e) => {
            warn!(%url, error = %e, "web_fetch failed");
            Ran {
                model_output: format!("Fetch failed: {e}"),
                client: ClientRender {
                    blocks: Vec::new(),
                    marker: json!({"id": ctx.call_id, "url": &url}),
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
        assert_eq!(Tool::WebFetch.name(), "web_fetch");
    }

    /// The web_fetch executor runs the same provider walk `/v1/fetch` runs,
    /// against a provider pointed at a local mock through `base_url`. The
    /// model must be shown the page text, not only the URL.
    #[tokio::test]
    async fn web_fetch_returns_the_page_text_from_a_mock_provider() {
        use axum::routing::post;
        let router = axum::Router::new().route(
            "/scrape",
            post(|| async { axum::Json(json!({"data": {"markdown": "hello from the page"}})) }),
        );
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move { axum::serve(listener, router).await.unwrap() });
        let app = mock_app(
            &format!(
                r#"
                [server]
                [[fetch.providers]]
                name = "mock"
                kind = "firecrawl-scrape"
                api_key = "k"
                base_url = "http://{addr}/scrape"
                "#
            ),
            "web_fetch_mock",
        );
        let ctx = ToolCtx::new(&app, "call_1");
        let ran = Tool::WebFetch
            .execute(&ctx, &json!({"url": "https://example.com/article"}))
            .await
            .expect("a configured provider serves the call");
        assert!(ran.model_output.contains("hello from the page"), "{}", ran.model_output);
        assert_eq!(ran.client.marker["url"], "https://example.com/article");
        assert!(ran.client.blocks.is_empty(), "web_fetch has no Anthropic block");
    }

    /// A URL-less call is malformed, not a failure to report back: the loop
    /// leaves it out of the replay instead of inventing a tool result.
    #[tokio::test]
    async fn web_fetch_without_a_url_is_unserved() {
        let app = mock_app("[server]", "web_fetch_missing");
        let ctx = ToolCtx::new(&app, "c");
        assert!(Tool::WebFetch.execute(&ctx, &json!({})).await.is_err());
    }

    /// One page must not be able to swallow the context window, so the model
    /// is shown a capped prefix and told it was cut.
    #[test]
    fn fetched_content_is_capped() {
        let long = "x".repeat(FETCH_CONTENT_CHARS + 10);
        let out = fetched_for_model("https://e/x", &long);
        assert!(out.ends_with("[truncated]"), "the model must be told");
        assert!(out.contains(&"x".repeat(FETCH_CONTENT_CHARS)));
        assert!(!out.contains(&"x".repeat(FETCH_CONTENT_CHARS + 1)));
    }

    /// A minimal app for the executor tests, mirroring `router`'s test app.
    fn mock_app(cfg_toml: &str, name: &str) -> std::sync::Arc<App> {
        let cfg: crate::config::Config = toml::from_str(cfg_toml).unwrap();
        let catalog = crate::catalog::Catalog::from_config(&cfg);
        let dir =
            std::env::temp_dir().join(format!("pxy-server-tools-{}-{name}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::sync::Arc::new(App {
            catalog,
            secrets: crate::secrets::Secrets::new(),
            state: crate::state::State::open(&dir.join("s.sqlite")).unwrap(),
            http: reqwest::Client::new(),
            cfg,
        })
    }
}
