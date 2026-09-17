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
use crate::catalog::Catalog;
use crate::router::{App, ClientFormat};

/// A tool pxy can serve.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Tool {
    WebSearch,
    WebFetch,
    Datetime,
    SearchModels,
}

impl Tool {
    /// Every tool this build can serve. Drives the default `[server_tools]
    /// enabled` list, so a tool added here is enabled unless config says
    /// otherwise.
    pub fn implemented() -> &'static [Tool] {
        &[Tool::WebSearch, Tool::WebFetch, Tool::Datetime, Tool::SearchModels]
    }

    /// The tool's canonical name: the spelling `[server_tools] enabled` lists,
    /// without the reserved prefix.
    pub fn name(self) -> &'static str {
        match self {
            Tool::WebSearch => "web_search",
            Tool::WebFetch => "web_fetch",
            Tool::Datetime => "datetime",
            Tool::SearchModels => "search_models",
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
            Tool::Datetime => run_datetime(ctx, args),
            Tool::SearchModels => run_search_models(ctx, args),
        }
    }

    /// Map a reserved `pxy_*` function name back to its tool, so the loop can
    /// dispatch a call the upstream returned.
    pub fn from_function_name(name: &str) -> Option<Tool> {
        Tool::implemented().iter().copied().find(|t| function_name(*t) == name)
    }

    /// Whether pxy can run this tool for this app right now: the tool is
    /// enabled and has whatever executor it needs — a search pool, a fetch
    /// pool, or, for datetime, nothing.
    pub fn servable(self, app: &App) -> bool {
        if !app.cfg.server_tools.enabled.iter().any(|n| self.name() == n) {
            return false;
        }
        match self {
            Tool::WebSearch => !app.cfg.search.providers.is_empty(),
            Tool::WebFetch => !app.cfg.fetch.providers.is_empty(),
            Tool::Datetime => true,
            Tool::SearchModels => true,
        }
    }

    /// Whether this client dialect is offered the tool. pxy never invents a
    /// client result block, so a tool with none on Anthropic Messages is not
    /// offered there; Chat Completions and Responses share
    /// [`ClientFormat::Openai`].
    pub fn served_on(self, client: ClientFormat) -> bool {
        match client {
            ClientFormat::Openai => true,
            ClientFormat::Anthropic => self == Tool::WebSearch,
        }
    }

    /// This tool's own call cap for the request: the declaration's `max_uses`
    /// (Anthropic spells it top-level, OpenRouter nestles it under
    /// `parameters`) or `parameters.max_tool_calls` (the wiki's spelling),
    /// default 5, clamped to 1..=20.
    pub fn max_uses(self, payload: &Value) -> u64 {
        declaration(payload, self)
            .and_then(|t| {
                t["max_uses"]
                    .as_u64()
                    .or_else(|| t["parameters"]["max_uses"].as_u64())
                    .or_else(|| t["parameters"]["max_tool_calls"].as_u64())
            })
            .unwrap_or(web_search::DEFAULT_MAX_USES)
            .clamp(1, 20)
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
    /// The wall clock the tool sees. `new` reads the real clock; a test builds
    /// the struct directly to fix it.
    pub now: jiff::Timestamp,
}

impl<'a> ToolCtx<'a> {
    pub fn new(app: &'a App, call_id: &'a str) -> Self {
        Self { app, call_id, now: jiff::Timestamp::now() }
    }
}

/// The request's declaration of this tool, when it carries one. A `function`
/// field means an ordinary function tool that merely shares the name, and is
/// left alone.
fn declaration<'a>(payload: &'a Value, tool: Tool) -> Option<&'a Value> {
    payload["tools"]
        .as_array()?
        .iter()
        .find(|t| t.get("function").is_none() && t["type"].as_str().and_then(from_type) == Some(tool))
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
        "openrouter:datetime" | "pxy:datetime" => Some(Tool::Datetime),
        "openrouter:experimental__search_models" | "pxy:search_models" => {
            Some(Tool::SearchModels)
        }
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
        Tool::Datetime => json!({
            "type": "function",
            "function": {
                "name": function_name(Tool::Datetime),
                "description": "The current date and time, optionally in an IANA timezone.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "timezone": {
                            "type": "string",
                            "description": "IANA timezone name, e.g. America/New_York. Default UTC."
                        }
                    },
                },
            },
        }),
        Tool::SearchModels => json!({
            "type": "function",
            "function": {
                "name": function_name(Tool::SearchModels),
                "description": "Search pxy's own model catalog for an id to call. \
                    Every filter is optional; use it to find a model with the \
                    context window, provider or capabilities the task needs.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Free-text match against provider/model ids."
                        },
                        "min_context_length": {
                            "type": "integer",
                            "description": "Only models with at least this many tokens of context."
                        },
                        "provider": {
                            "type": "string",
                            "description": "Only models served by this provider."
                        },
                        "tool_call": {
                            "type": "boolean",
                            "description": "Only models known to support tool calling."
                        },
                        "reasoning": {
                            "type": "boolean",
                            "description": "Only models known to support extended thinking."
                        },
                        "free": {
                            "type": "boolean",
                            "description": "true for free models only, false to exclude them."
                        },
                    },
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
    // The same check /v1/fetch makes: a reader endpoint is not a general
    // fetcher, and a self-hosted base_url must not widen what a model can ask
    // it to read.
    if !url.starts_with("http://") && !url.starts_with("https://") {
        return Ok(Ran {
            model_output: "Fetch failed: url must be http(s)".into(),
            client: ClientRender {
                blocks: Vec::new(),
                marker: json!({"id": ctx.call_id, "url": &url}),
            },
        });
    }
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

/// The current time as ISO-8601 with its offset, then the zone it is in. No
/// network, and a zone jiff does not know is reported to the model rather than
/// failing the call. An empty or absent `timezone` means UTC.
fn run_datetime(ctx: &ToolCtx<'_>, args: &Value) -> Result<Ran, String> {
    let requested = args["timezone"].as_str().filter(|z| !z.is_empty());
    let (zoned, label) = match requested {
        Some(name) => match ctx.now.in_tz(name) {
            Ok(z) => (z, name.to_string()),
            Err(_) => {
                return Ok(Ran {
                    model_output: format!("Unknown timezone: {name}"),
                    client: ClientRender {
                        blocks: Vec::new(),
                        marker: json!({"id": ctx.call_id, "timezone": name}),
                    },
                });
            }
        },
        None => (ctx.now.to_zoned(jiff::tz::TimeZone::UTC), "UTC".to_string()),
    };
    Ok(Ran {
        model_output: format!("{} ({label})", zoned.strftime("%Y-%m-%dT%H:%M:%S%:z")),
        client: ClientRender {
            blocks: Vec::new(),
            marker: json!({"id": ctx.call_id, "timezone": label}),
        },
    })
}

/// Run one search_models call against pxy's own catalog. Every argument is
/// optional and a missing one filters nothing. The model is told each
/// matching `provider/model`, the facts it routes on, and the group aliases
/// that reach it.
fn run_search_models(ctx: &ToolCtx<'_>, args: &Value) -> Result<Ran, String> {
    let matches = matching_models(&ctx.app.catalog, args);
    let total = ctx.app.catalog.models().len();
    let output = models_for_model(&matches, total);
    Ok(Ran {
        model_output: output,
        client: ClientRender {
            blocks: Vec::new(),
            marker: json!({"id": ctx.call_id, "matches": matches, "total": total}),
        },
    })
}

/// The catalog entries a search_models call matches, in catalog order, each as
/// the JSON object the model is told about. A filter that is absent filters
/// nothing; one that names a capability only matches a model asserted to have
/// it, so an unknown capability is not a match.
fn matching_models(catalog: &Catalog, args: &Value) -> Vec<Value> {
    let query = args["query"].as_str().unwrap_or("").to_ascii_lowercase();
    let min_context = args["min_context_length"].as_u64();
    let provider = args["provider"].as_str();
    let tool_call = args["tool_call"].as_bool();
    let reasoning = args["reasoning"].as_bool();
    let free = args["free"].as_bool();

    catalog
        .models()
        .iter()
        .filter(|c| query.is_empty() || c.full_id().to_ascii_lowercase().contains(&query))
        .filter(|c| min_context.is_none_or(|n| c.model.context_length >= n))
        .filter(|c| provider.is_none_or(|p| c.provider == p))
        .filter(|c| tool_call.is_none_or(|b| c.model.tool_call == Some(b)))
        .filter(|c| reasoning.is_none_or(|b| c.model.reasoning == Some(b)))
        .filter(|c| free.is_none_or(|b| c.model.free == Some(b)))
        .map(|c| {
            let id = c.full_id();
            let groups: Vec<String> = catalog
                .groups()
                .filter(|(_, g)| g.chain.iter().any(|m| m.full_id() == id))
                .map(|(name, _)| name.clone())
                .collect();
            json!({
                "id": id,
                "context_length": c.model.context_length,
                "tool_call": c.model.tool_call,
                "reasoning": c.model.reasoning,
                "free": c.model.free,
                "groups": groups,
            })
        })
        .collect()
}

/// The model-facing report for a search_models call: one line per match, then
/// how many of the catalog it is. No matches is an answer, not an error.
fn models_for_model(matches: &[Value], total: usize) -> String {
    if matches.is_empty() {
        return format!("0 of {total} models matched.");
    }
    let mut out = format!("{} of {total} models:\n", matches.len());
    for m in matches {
        let mut facts =
            vec![format!("context {}", m["context_length"].as_u64().unwrap_or(0))];
        if m["tool_call"] == true {
            facts.push("tools".into());
        }
        if m["reasoning"] == true {
            facts.push("reasoning".into());
        }
        if m["free"] == true {
            facts.push("free".into());
        }
        let groups: Vec<&str> = m["groups"]
            .as_array()
            .map(|a| a.iter().filter_map(|g| g.as_str()).collect())
            .unwrap_or_default();
        let suffix = if groups.is_empty() {
            String::new()
        } else {
            format!(" [groups: {}]", groups.join(", "))
        };
        out.push_str(&format!(
            "- {} ({}){suffix}\n",
            m["id"].as_str().unwrap_or(""),
            facts.join(", ")
        ));
    }
    out.trim_end().to_string()
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
        assert_eq!(Tool::Datetime.name(), "datetime");
        assert_eq!(Tool::SearchModels.name(), "search_models");
    }

    /// datetime is clock- and zone-only: the same fixed instant read as UTC
    /// and as an IANA zone must differ by that zone's offset.
    #[tokio::test]
    async fn datetime_defaults_to_utc_and_accepts_an_iana_zone() {
        let app = mock_app("[server]", "datetime_clock");
        let now: jiff::Timestamp = "2025-07-15T18:30:00Z".parse().unwrap();
        let ctx = ToolCtx { app: &app, call_id: "call_1", now };

        let utc = Tool::Datetime.execute(&ctx, &json!({})).await.unwrap();
        assert_eq!(utc.model_output, "2025-07-15T18:30:00+00:00 (UTC)");

        let ny = Tool::Datetime
            .execute(&ctx, &json!({"timezone": "America/New_York"}))
            .await
            .unwrap();
        assert_eq!(ny.model_output, "2025-07-15T14:30:00-04:00 (America/New_York)");
    }

    /// A zone jiff cannot resolve is the model's mistake to see, not a reason
    /// to drop the call from the replay.
    #[tokio::test]
    async fn datetime_reports_an_unknown_zone() {
        let app = mock_app("[server]", "datetime_bad_zone");
        let now: jiff::Timestamp = "2025-07-15T18:30:00Z".parse().unwrap();
        let ctx = ToolCtx { app: &app, call_id: "call_1", now };
        let ran = Tool::Datetime
            .execute(&ctx, &json!({"timezone": "Mars/Olympus"}))
            .await
            .unwrap();
        assert!(ran.model_output.contains("Mars/Olympus"), "{}", ran.model_output);
    }

    /// The injected datetime function models one optional argument; the tool
    /// must advertise it or the model can never set a zone.
    #[test]
    fn datetime_def_advertises_an_optional_timezone() {
        let def = tool_def(Tool::Datetime, &Value::Null);
        assert_eq!(def["function"]["name"], "pxy_datetime");
        assert_eq!(def["function"]["parameters"]["properties"]["timezone"]["type"], "string");
        assert!(def["function"]["parameters"].get("required").is_none());
    }

    /// Each tool's own `max_uses` caps its calls: Anthropic spells it at the
    /// top level, OpenRouter under `parameters`, and a function that merely
    /// shares the name is not the server tool.
    #[test]
    fn max_uses_reads_both_dialects_and_clamps() {
        let anthropic = json!({"tools": [{"type": "web_search_20250305", "max_uses": 3}]});
        assert_eq!(Tool::WebSearch.max_uses(&anthropic), 3);
        let openrouter =
            json!({"tools": [{"type": "openrouter:web_fetch", "parameters": {"max_uses": 7}}]});
        assert_eq!(Tool::WebFetch.max_uses(&openrouter), 7);
        let bare = json!({"tools": [{"type": "pxy:datetime"}]});
        assert_eq!(Tool::Datetime.max_uses(&bare), web_search::DEFAULT_MAX_USES);
        let big = json!({"tools": [{"type": "pxy:web_fetch", "parameters": {"max_uses": 900}}]});
        assert_eq!(Tool::WebFetch.max_uses(&big), 20);
        // The wiki spells a tool's own cap `parameters.max_tool_calls`; accept
        // it beside OpenRouter's `max_uses` so neither spelling is silently
        // ignored.
        let wiki = json!({"tools": [{"type": "pxy:web_fetch", "parameters": {"max_tool_calls": 2}}]});
        assert_eq!(Tool::WebFetch.max_uses(&wiki), 2);
        let f = json!({"tools": [{"type": "function", "function": {"name": "web_search"}}]});
        assert_eq!(Tool::WebSearch.max_uses(&f), web_search::DEFAULT_MAX_USES);
    }

    /// A tool is servable only with the executor it needs, and only when it is
    /// enabled; datetime needs nothing but its own entry.
    #[test]
    fn servable_needs_the_executor_its_tool_uses() {
        let none = mock_app("[server]", "servable_none");
        assert!(!Tool::WebSearch.servable(&none), "no search pool");
        assert!(!Tool::WebFetch.servable(&none), "no fetch pool");
        assert!(Tool::Datetime.servable(&none), "datetime needs nothing");

        let search = mock_app(
            r#"
            [server]
            [[search.providers]]
            name = "s"
            kind = "brave"
            api_key = "k"
            "#,
            "servable_search",
        );
        assert!(Tool::WebSearch.servable(&search));
        assert!(!Tool::WebFetch.servable(&search), "a search pool is not a fetch pool");

        let disabled = mock_app(
            r#"
            [server]
            [server_tools]
            enabled = ["datetime"]
            [[fetch.providers]]
            name = "f"
            kind = "jina-reader"
            api_key = "k"
            "#,
            "servable_disabled",
        );
        assert!(!Tool::WebFetch.servable(&disabled), "not enabled");
        assert!(Tool::Datetime.servable(&disabled));
    }

    /// No tool gets a client result block of its own on Anthropic Messages, so
    /// web_fetch and datetime are not offered there.
    #[test]
    fn anthropic_messages_is_offered_only_web_search() {
        for tool in Tool::implemented() {
            assert!(tool.served_on(ClientFormat::Openai), "{tool:?}");
        }
        assert!(Tool::WebSearch.served_on(ClientFormat::Anthropic));
        assert!(!Tool::WebFetch.served_on(ClientFormat::Anthropic));
        assert!(!Tool::Datetime.served_on(ClientFormat::Anthropic));
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

    /// A URL that is not http(s) is refused before any provider sees it: the
    /// fetch walk's caller owns the shape check, exactly as /v1/fetch does.
    #[tokio::test]
    async fn web_fetch_rejects_non_http_urls() {
        let app = mock_app("[server]", "web_fetch_scheme");
        let ctx = ToolCtx::new(&app, "c");
        let ran = Tool::WebFetch
            .execute(&ctx, &json!({"url": "file:///etc/passwd"}))
            .await
            .unwrap();
        assert!(ran.model_output.contains("http(s)"), "{}", ran.model_output);
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

    /// search_models answers from pxy's own catalog: every filter is optional,
    /// the report names only the entries that match, and it says how many of
    /// the catalog they are. The group a model is reachable through rides
    /// along, because that alias is usually the id to call.
    #[tokio::test]
    async fn search_models_filters_the_catalog_and_reports_a_total() {
        let app = mock_app(
            r#"
            [server]
            [providers.alpha]
            base_url = "https://alpha.example/chat"
            models = [
                { id = "big", context_length = 200000, tool_call = true, reasoning = true },
                { id = "small", context_length = 8000 },
            ]
            [providers.beta]
            base_url = "https://beta.example/chat"
            models = [{ id = "tiny", context_length = 1000, free = true }]
            [groups.aaa]
            models = ["alpha/big", "beta/tiny"]
            "#,
            "search_models",
        );
        let ctx = ToolCtx::new(&app, "c");

        let ran = Tool::SearchModels
            .execute(&ctx, &json!({"query": "big", "min_context_length": 100000}))
            .await
            .unwrap();
        assert!(ran.model_output.contains("alpha/big"), "{}", ran.model_output);
        assert!(ran.model_output.contains("1 of 3"), "{}", ran.model_output);
        assert!(!ran.model_output.contains("alpha/small"), "{}", ran.model_output);
        assert!(!ran.model_output.contains("beta/tiny"), "{}", ran.model_output);
        assert!(ran.model_output.contains("aaa"), "the group alias: {}", ran.model_output);

        // A capability filter reads the catalog's asserted metadata.
        let ran = Tool::SearchModels.execute(&ctx, &json!({"tool_call": true})).await.unwrap();
        assert!(ran.model_output.contains("alpha/big"), "{}", ran.model_output);
        assert!(!ran.model_output.contains("alpha/small"), "{}", ran.model_output);
        assert!(!ran.model_output.contains("beta/tiny"), "{}", ran.model_output);

        // No match is a report, not an unserved call.
        let ran = Tool::SearchModels.execute(&ctx, &json!({"query": "nope"})).await.unwrap();
        assert!(ran.model_output.contains("0 of 3"), "{}", ran.model_output);
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
