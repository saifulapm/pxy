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
use crate::router::{App, ClientContext, ClientFormat, Outcome, SharedApp, handle_chat};

/// A tool pxy can serve.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Tool {
    WebSearch,
    WebFetch,
    Datetime,
    SearchModels,
    ImageGeneration,
    Advisor,
    Subagent,
}

impl Tool {
    /// Every tool this build can serve. Drives the default `[server_tools]
    /// enabled` list, so a tool added here is enabled unless config says
    /// otherwise.
    pub fn implemented() -> &'static [Tool] {
        &[
            Tool::WebSearch,
            Tool::WebFetch,
            Tool::Datetime,
            Tool::SearchModels,
            Tool::ImageGeneration,
            Tool::Advisor,
            Tool::Subagent,
        ]
    }

    /// The tool's canonical name: the spelling `[server_tools] enabled` lists,
    /// without the reserved prefix.
    pub fn name(self) -> &'static str {
        match self {
            Tool::WebSearch => "web_search",
            Tool::WebFetch => "web_fetch",
            Tool::Datetime => "datetime",
            Tool::SearchModels => "search_models",
            Tool::ImageGeneration => "image_generation",
            Tool::Advisor => "advisor",
            Tool::Subagent => "subagent",
        }
    }

    /// Whether this tool's executor issues a sub-request through pxy's own
    /// router. A meta-tool is stripped from a sub-request, so recursion stops
    /// at one level.
    pub fn is_meta(self) -> bool {
        matches!(self, Tool::Advisor | Tool::Subagent)
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
            Tool::ImageGeneration => run_image_generation(ctx, args).await,
            Tool::Advisor => run_advisor(ctx, args).await,
            Tool::Subagent => run_subagent(ctx, args).await,
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
            Tool::ImageGeneration => crate::media::image_chain_can_return_urls(&app.cfg),
            // A meta-tool needs only the router it is already in.
            Tool::Advisor | Tool::Subagent => true,
        }
    }

    /// Whether this client dialect is offered the tool. pxy never invents a
    /// client result block, so a tool with none on Anthropic Messages is not
    /// offered there; Chat Completions and Responses share
    /// [`ClientFormat::Openai`].
    pub fn served_on(self, client: ClientFormat) -> bool {
        match client {
            ClientFormat::Openai => true,
            // The advisor has an Anthropic result block; the subagent does not.
            ClientFormat::Anthropic => matches!(self, Tool::WebSearch | Tool::Advisor),
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
    pub app: &'a SharedApp,
    pub call_id: &'a str,
    /// The declaration's own `parameters` object, when the request carried
    /// one. The meta-tools read their configured model and instructions here.
    pub params: Value,
    /// The wall clock the tool sees. `new` reads the real clock; a test builds
    /// the struct directly to fix it.
    pub now: jiff::Timestamp,
}

impl<'a> ToolCtx<'a> {
    pub fn new(app: &'a SharedApp, call_id: &'a str) -> Self {
        Self { app, call_id, params: Value::Null, now: jiff::Timestamp::now() }
    }
}

/// Issue one sub-request through pxy's own router and return the assistant's
/// text. The meta-tools consult a model this way: the leg resolves providers,
/// passes limits and cooldowns, translates, records usage and fails over
/// exactly like a client turn, because it is one. `params` is merged into the
/// payload beside `model`, `messages` and `stream`, which pxy owns here.
pub async fn run_internal_chat(
    app: &SharedApp,
    model: &str,
    messages: Vec<Value>,
    params: Value,
) -> Result<String, String> {
    let mut payload = json!({
        "model": model,
        "messages": messages,
        "stream": false,
    });
    if let (Some(dst), Some(src)) = (payload.as_object_mut(), params.as_object()) {
        for (k, v) in src {
            if k != "model" && k != "messages" && k != "stream" {
                dst.insert(k.clone(), v.clone());
            }
        }
    }
    // handle_chat is already boxed to break the meta-tool recursion cycle.
    let ctx = ClientContext { tool_depth: 1, ..ClientContext::default() };
    let outcome = handle_chat(app.clone(), ClientFormat::Openai, payload, ctx).await;
    match outcome {
        Outcome::Json { status, body, .. } if status < 400 => body["choices"][0]["message"]
            ["content"]
            .as_str()
            .map(str::to_string)
            .ok_or_else(|| "sub-request returned no assistant text".to_string()),
        Outcome::Json { status, body, .. } => {
            let message = body["error"]["message"].as_str().unwrap_or("sub-request failed");
            Err(format!("sub-request failed ({status}): {message}"))
        }
        Outcome::Stream { .. } => Err("sub-request unexpectedly streamed".to_string()),
    }
}

/// Run the same prompt across a panel of models at once and keep each leg's
/// answer under the model that gave it. Every leg is an independent
/// sub-request through pxy's own router, so provider limits, cooldowns and
/// usage accounting apply to each. A leg's failure is its own: it is returned
/// beside the answers that succeeded, because a partial panel is still worth
/// reading.
pub async fn run_panel(
    app: &SharedApp,
    models: &[String],
    prompt: &str,
    params: &Value,
) -> Vec<(String, Result<String, String>)> {
    let legs = models.iter().map(|model| {
        let messages = vec![json!({"role": "user", "content": prompt})];
        let params = params.clone();
        async move {
            let answer = run_internal_chat(app, model, messages, params).await;
            (model.clone(), answer)
        }
    });
    futures_util::future::join_all(legs).await
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
        "openrouter:image_generation" | "pxy:image_generation" => Some(Tool::ImageGeneration),
        "openrouter:advisor" | "pxy:advisor" => Some(Tool::Advisor),
        "openrouter:subagent" | "pxy:subagent" => Some(Tool::Subagent),
        // The Anthropic native advisor is dated, like its other server tools.
        t if t.starts_with("advisor_") => Some(Tool::Advisor),
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
        Tool::ImageGeneration => json!({
            "type": "function",
            "function": {
                "name": function_name(Tool::ImageGeneration),
                "description": "Generate an image from a text prompt and return its URL.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "prompt": {
                            "type": "string",
                            "description": "What the image should show."
                        },
                        "model": {
                            "type": "string",
                            "description": "Image model to use; defaults to the configured [media] image chain."
                        },
                    },
                    "required": ["prompt"],
                },
            },
        }),
        Tool::Advisor => json!({
            "type": "function",
            "function": {
                "name": function_name(Tool::Advisor),
                "description": "Consult a stronger model for guidance before committing to \
                    an approach, when stuck, or before declaring a task done.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "prompt": {
                            "type": "string",
                            "description": "What to get advice on."
                        },
                        "model": {
                            "type": "string",
                            "description": "Advisor model to use; only honored when the \
                                declaration did not pin one."
                        },
                    },
                    "required": ["prompt"],
                },
            },
        }),
        Tool::Subagent => json!({
            "type": "function",
            "function": {
                "name": function_name(Tool::Subagent),
                "description": "Delegate a self-contained task to a cheaper worker \
                    model and get its result back. The worker sees only the task \
                    description, so include all relevant context and the expected \
                    output format.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "task_name": {
                            "type": "string",
                            "description": "A short identifier for the delegated task."
                        },
                        "task_description": {
                            "type": "string",
                            "description": "Everything the worker needs: context, inputs, \
                                constraints, and the expected output format."
                        },
                    },
                    "required": ["task_name", "task_description"],
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
        .filter(|c| tool_call.is_none_or(|b| matches_flag(c.model.tool_call, b)))
        .filter(|c| reasoning.is_none_or(|b| matches_flag(c.model.reasoning, b)))
        .filter(|c| free.is_none_or(|b| matches_flag(c.model.free, b)))
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

/// A capability filter: `true` demands an asserted yes; `false` demands
/// anything but an asserted yes, because an unasserted capability is not a no.
fn matches_flag(asserted: Option<bool>, want: bool) -> bool {
    if want { asserted == Some(true) } else { asserted != Some(true) }
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

/// The declaration's own `parameters` object for a tool, when the request
/// carries one. The loop hands it to the executor through [`ToolCtx::params`],
/// where the meta-tools read their configured model and instructions.
pub fn declared_parameters(payload: &Value, tool: Tool) -> Value {
    let Some(t) = declaration(payload, tool) else { return Value::Null };
    let mut params = t.get("parameters").cloned().filter(Value::is_object).unwrap_or_else(|| json!({}));
    // The Anthropic native advisor shape carries its model on the entry itself,
    // not nested under `parameters`.
    if params["model"].is_null() && !t["model"].is_null() {
        params["model"] = t["model"].clone();
    }
    params
}

/// Run one advisor call: the model's `prompt` goes to the advisor as a user
/// turn, the declaration's `instructions` as the system turn, and the
/// advisor's answer comes back as the tool result. The advisor sees only that
/// — pxy keeps no conversation, so cross-request memory is out of scope. A
/// failure is reported to the outer model, which continues without the advice.
async fn run_advisor(ctx: &ToolCtx<'_>, args: &Value) -> Result<Ran, String> {
    let prompt = args["prompt"].as_str().filter(|p| !p.is_empty()).ok_or("missing prompt")?;
    // The declaration's model wins; the call's own is honoured only when the
    // declaration pins none.
    let model = ctx.params["model"]
        .as_str()
        .or_else(|| args["model"].as_str())
        .filter(|m| !m.is_empty())
        .ok_or("no advisor model")?;
    let mut messages: Vec<Value> = Vec::new();
    if let Some(instructions) = ctx.params["instructions"].as_str().filter(|s| !s.is_empty()) {
        messages.push(json!({"role": "system", "content": instructions}));
    }
    messages.push(json!({"role": "user", "content": prompt}));
    let failed = |e: String| Ran {
        model_output: json!({"status": "error", "error": format!("Advisor call failed: {e}")})
            .to_string(),
        client: ClientRender {
            blocks: Vec::new(),
            marker: json!({"id": ctx.call_id, "model": model, "prompt": prompt}),
        },
    };
    match run_internal_chat(ctx.app, model, messages, json!({})).await {
        Ok(advice) => {
            info!(%model, "advisor served");
            Ok(Ran {
                model_output: json!({"status": "ok", "model": model, "advice": advice})
                    .to_string(),
                client: ClientRender {
                    // The documented Anthropic pair: the consultation, then the
                    // advice. Other dialects read the marker instead.
                    blocks: vec![
                        advisor_server_tool_use_block(ctx.call_id, prompt),
                        advisor_result_block(ctx.call_id, &advice),
                    ],
                    marker: json!({"id": ctx.call_id, "model": model, "advice": advice}),
                },
            })
        }
        Err(e) => {
            warn!(%model, error = %e, "advisor failed");
            Ok(failed(e))
        }
    }
}

/// `server_tool_use` naming the advisor, as the Anthropic transcript records a
/// consultation. The id prefix mirrors the real API's `srvtoolu_`.
fn advisor_server_tool_use_block(id: &str, prompt: &str) -> Value {
    json!({
        "type": "server_tool_use",
        "id": id,
        "name": "advisor",
        "input": {"prompt": prompt},
    })
}

/// `advisor_tool_result` carrying the advice, the documented Anthropic shape.
fn advisor_result_block(id: &str, advice: &str) -> Value {
    json!({
        "type": "advisor_tool_result",
        "tool_use_id": id,
        "content": {"type": "advisor_result", "text": advice},
    })
}

/// Flatten the advisor blocks back into prose when a client replays them in a
/// later turn. An OpenAI upstream has no notion of a server tool, but dropping
/// them would lose what the advisor said.
pub fn flatten_advisor_history_block(block: &Value) -> Option<String> {
    match block["type"].as_str()? {
        "server_tool_use" if block["name"] == "advisor" => {
            let prompt = block["input"]["prompt"].as_str().unwrap_or("");
            Some(format!("[advisor consulted: {prompt}]"))
        }
        "advisor_tool_result" => {
            let text = block["content"]["text"].as_str().unwrap_or("");
            Some(format!("[advisor advice] {text}"))
        }
        _ => None,
    }
}

/// Run one subagent call: a self-contained task goes to the worker as a user
/// turn, the declaration's `instructions` as the system turn, and the worker's
/// final text comes back as the outcome. The worker never sees the parent
/// conversation. A served tool the declaration lists runs inside the worker's
/// own sub-request, through the same loop the outer turn uses.
async fn run_subagent(ctx: &ToolCtx<'_>, args: &Value) -> Result<Ran, String> {
    let task_name =
        args["task_name"].as_str().filter(|s| !s.is_empty()).ok_or("missing task_name")?;
    let task = args["task_description"]
        .as_str()
        .filter(|s| !s.is_empty())
        .ok_or("missing task_description")?;
    // The worker is fixed by the declaration; the model does not choose it.
    let model = ctx.params["model"].as_str().filter(|m| !m.is_empty()).ok_or("no subagent model")?;
    let mut messages: Vec<Value> = Vec::new();
    if let Some(instructions) = ctx.params["instructions"].as_str().filter(|s| !s.is_empty()) {
        messages.push(json!({"role": "system", "content": instructions}));
    }
    messages.push(json!({"role": "user", "content": task}));
    let mut params = json!({});
    // The worker's own served tools ride the sub-request: the loop injects and
    // serves them there, exactly as it does for a client turn. `tool_depth` 1
    // strips any meta-tool, so a worker can never re-enter one.
    if let Some(tools) = ctx.params["tools"].as_array().filter(|t| !t.is_empty()) {
        params["tools"] = Value::Array(tools.clone());
    }
    for key in ["max_tool_calls", "max_completion_tokens", "temperature", "reasoning"] {
        if !ctx.params[key].is_null() {
            params[key] = ctx.params[key].clone();
        }
    }
    match run_internal_chat(ctx.app, model, messages, params).await {
        Ok(outcome) => {
            info!(%model, %task_name, "subagent served");
            Ok(Ran {
                model_output: json!({
                    "status": "ok", "model": model, "task_name": task_name, "outcome": outcome,
                })
                .to_string(),
                client: ClientRender {
                    blocks: Vec::new(),
                    marker: json!({
                        "id": ctx.call_id, "model": model, "task_name": task_name,
                        "outcome": outcome,
                    }),
                },
            })
        }
        Err(e) => {
            warn!(%model, %task_name, error = %e, "subagent failed");
            Ok(Ran {
                model_output: json!({
                    "status": "error", "task_name": task_name,
                    "error": format!("Subagent call failed: {e}"),
                })
                .to_string(),
                client: ClientRender {
                    blocks: Vec::new(),
                    marker: json!({"id": ctx.call_id, "model": model, "task_name": task_name}),
                },
            })
        }
    }
}

/// Run one image_generation call through the image media walk — the same walk
/// `/v1/images/generations` runs, so the media quota and failover are shared.
/// A missing prompt makes the call unserved; a provider failure is reported
/// to the model. The model is handed the image URL, not the bytes.
async fn run_image_generation(ctx: &ToolCtx<'_>, args: &Value) -> Result<Ran, String> {
    let prompt = args["prompt"].as_str().filter(|p| !p.is_empty()).ok_or("missing prompt")?;
    let model = args["model"].as_str().filter(|m| !m.is_empty());
    match crate::media::images::run_generate(ctx.app, model, &json!({"prompt": prompt}), true).await {
        // The walk is told the tool needs a URL, so a base64-only candidate
        // fails over and a success here carries one.
        Ok((body, provider)) => {
            let url = body["data"][0]["url"].as_str().unwrap_or("");
            info!(%provider, "image_generation served");
            Ok(Ran {
                model_output: format!("Generated image: {url}"),
                client: ClientRender {
                    blocks: Vec::new(),
                    marker: json!({"id": ctx.call_id, "provider": provider, "url": url}),
                },
            })
        }
        Err(resp) => {
            let status = resp.status();
            let bytes =
                axum::body::to_bytes(resp.into_body(), 1 << 20).await.unwrap_or_default();
            let detail = String::from_utf8_lossy(&bytes);
            warn!(%status, error = %detail, "image_generation failed");
            Ok(Ran {
                model_output: format!("Image generation failed ({status}): {detail}"),
                client: ClientRender {
                    blocks: Vec::new(),
                    marker: json!({"id": ctx.call_id, "model": model}),
                },
            })
        }
    }
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
        assert_eq!(Tool::ImageGeneration.name(), "image_generation");
    }

    /// datetime is clock- and zone-only: the same fixed instant read as UTC
    /// and as an IANA zone must differ by that zone's offset.
    #[tokio::test]
    async fn datetime_defaults_to_utc_and_accepts_an_iana_zone() {
        let app = mock_app("[server]", "datetime_clock");
        let now: jiff::Timestamp = "2025-07-15T18:30:00Z".parse().unwrap();
        let ctx = ToolCtx { app: &app, call_id: "call_1", params: Value::Null, now };

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
        let ctx = ToolCtx { app: &app, call_id: "call_1", params: Value::Null, now };
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
        assert!(!Tool::ImageGeneration.servable(&none), "no image chain");

        let cloudflare = mock_app(
            r#"
            [server]
            [providers.cf]
            [providers.cf.media]
            kind = "cloudflare"
            images_url = "https://cf.example/img"
            image_models = ["m"]
            [media]
            image = ["cf/m"]
            "#,
            "servable_cloudflare",
        );
        assert!(
            !Tool::ImageGeneration.servable(&cloudflare),
            "a base64-only chain cannot feed a text tool result"
        );

        let image_url = mock_app(
            r#"
            [server]
            [providers.o]
            [providers.o.media]
            images_url = "https://o.example/img"
            image_models = ["m"]
            [media]
            image = ["o/m"]
            "#,
            "servable_image_url",
        );
        assert!(Tool::ImageGeneration.servable(&image_url), "an OpenAI image chain serves a URL");

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

    /// On Anthropic Messages only the tools with a documented result block are
    /// offered: web_search and the advisor. web_fetch, datetime, search_models,
    /// image_generation and the subagent have none.
    #[test]
    fn anthropic_messages_is_offered_web_search_and_advisor() {
        for tool in Tool::implemented() {
            assert!(tool.served_on(ClientFormat::Openai), "{tool:?}");
        }
        assert!(Tool::WebSearch.served_on(ClientFormat::Anthropic));
        assert!(Tool::Advisor.served_on(ClientFormat::Anthropic));
        assert!(!Tool::WebFetch.served_on(ClientFormat::Anthropic));
        assert!(!Tool::Datetime.served_on(ClientFormat::Anthropic));
        assert!(!Tool::Subagent.served_on(ClientFormat::Anthropic));
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

        // `free: false` excludes only asserted-free models; an unasserted
        // capability is not a no, so the two unknowns match.
        let ran = Tool::SearchModels.execute(&ctx, &json!({"free": false})).await.unwrap();
        assert!(ran.model_output.contains("2 of 3"), "{}", ran.model_output);
        assert!(ran.model_output.contains("alpha/big"), "{}", ran.model_output);
        assert!(ran.model_output.contains("alpha/small"), "{}", ran.model_output);
        assert!(!ran.model_output.contains("beta/tiny"), "{}", ran.model_output);

        let ran = Tool::SearchModels.execute(&ctx, &json!({"free": true})).await.unwrap();
        assert!(ran.model_output.contains("1 of 3"), "{}", ran.model_output);
        assert!(ran.model_output.contains("beta/tiny"), "{}", ran.model_output);

        // No match is a report, not an unserved call.
        let ran = Tool::SearchModels.execute(&ctx, &json!({"query": "nope"})).await.unwrap();
        assert!(ran.model_output.contains("0 of 3"), "{}", ran.model_output);
    }

    /// image_generation runs the image media walk and hands the model the URL
    /// the provider answered with; the generation is a media cost, not a chat
    /// one.
    #[tokio::test]
    async fn image_generation_returns_the_image_url_from_a_mock_provider() {
        use axum::routing::post;
        let router = axum::Router::new().route(
            "/img",
            post(|| async {
                axum::Json(json!({"created": 1, "data": [{"url": "https://x/y.png"}]}))
            }),
        );
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move { axum::serve(listener, router).await.unwrap() });
        let app = mock_app(
            &format!(
                r#"
                [server]
                [providers.mock]
                [providers.mock.media]
                images_url = "http://{addr}/img"
                image_models = ["m"]
                [media]
                image = ["mock/m"]
                "#
            ),
            "image_generation",
        );
        let ctx = ToolCtx::new(&app, "call_1");
        let ran = Tool::ImageGeneration
            .execute(&ctx, &json!({"prompt": "a cat"}))
            .await
            .expect("a configured image chain serves the call");
        assert!(ran.model_output.contains("https://x/y.png"), "{}", ran.model_output);
        assert!(ran.client.blocks.is_empty(), "no Anthropic block for image_generation");
    }

    /// A base64-only chain has no URL to hand the model. The walk fails over,
    /// and with no URL-capable candidate left the tool reports the failure
    /// rather than a hollow success.
    #[tokio::test]
    async fn image_generation_reports_a_base64_only_chain_as_failed() {
        use axum::routing::post;
        let router = axum::Router::new().route(
            "/img",
            post(|| async { axum::Json(json!({"result": {"image": "aGk="}, "success": true})) }),
        );
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move { axum::serve(listener, router).await.unwrap() });
        let app = mock_app(
            &format!(
                r#"
                [server]
                [providers.cf]
                [providers.cf.media]
                kind = "cloudflare"
                images_url = "http://{addr}/img"
                image_models = ["m"]
                [media]
                image = ["cf/m"]
                "#
            ),
            "image_generation_b64",
        );
        let ctx = ToolCtx::new(&app, "call_1");
        let ran = Tool::ImageGeneration
            .execute(&ctx, &json!({"prompt": "a cat"}))
            .await
            .unwrap();
        assert!(ran.model_output.contains("base64"), "{}", ran.model_output);
        assert!(ran.model_output.contains("failed"), "{}", ran.model_output);
    }

    /// A prompt-less call is malformed, not a failure to report back.
    #[tokio::test]
    async fn image_generation_without_a_prompt_is_unserved() {
        let app = mock_app("[server]", "image_generation_no_prompt");
        let ctx = ToolCtx::new(&app, "c");
        assert!(Tool::ImageGeneration.execute(&ctx, &json!({})).await.is_err());
    }

    /// A replayed advisor consultation flattens to prose for an OpenAI
    /// upstream, and the flattener must leave a web_search block alone so its
    /// own flattener handles it.
    #[test]
    fn advisor_history_blocks_flatten_to_prose() {
        let q = json!({"type": "server_tool_use", "id": "s1", "name": "advisor",
            "input": {"prompt": "how?"}});
        assert_eq!(flatten_advisor_history_block(&q).unwrap(), "[advisor consulted: how?]");
        let r = json!({"type": "advisor_tool_result", "tool_use_id": "s1",
            "content": {"type": "advisor_result", "text": "like this"}});
        assert_eq!(flatten_advisor_history_block(&r).unwrap(), "[advisor advice] like this");
        let ws = json!({"type": "server_tool_use", "id": "s2", "name": "web_search",
            "input": {"query": "rust"}});
        assert!(flatten_advisor_history_block(&ws).is_none());
    }

    /// The meta-tools issue their sub-request through pxy's own router, so a
    /// consultation is a real turn: it resolves the provider, reaches the
    /// upstream, and records usage like any client request.
    #[tokio::test]
    async fn internal_chat_returns_the_answer_and_records_usage() {
        use axum::routing::post;
        let router = axum::Router::new().route(
            "/c",
            post(|| async {
                axum::Json(json!({"id": "x", "choices": [{"index": 0,
                    "message": {"role": "assistant", "content": "advisor says hi"},
                    "finish_reason": "stop"}],
                    "usage": {"prompt_tokens": 3, "completion_tokens": 2}}))
            }),
        );
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move { axum::serve(listener, router).await.unwrap() });
        let app = mock_app(
            &format!(
                r#"
                [server]
                [providers.p]
                base_url = "http://{addr}/c"
                models = ["m"]
                "#
            ),
            "internal_chat",
        );
        let text = run_internal_chat(
            &app,
            "p/m",
            vec![json!({"role": "user", "content": "advise me"})],
            json!({}),
        )
        .await
        .unwrap();
        assert_eq!(text, "advisor says hi");
        assert_eq!(app.state.usage_total("p").unwrap().requests, 1);
    }

    /// The worker sees only the task description, never the parent
    /// conversation, and a served tool the declaration listed is offered to
    /// it. The nested tool is injected as the reserved function so the loop
    /// would serve it inside the worker's own sub-request.
    #[tokio::test]
    async fn subagent_sees_only_the_task_and_its_declared_tools() {
        use std::sync::Mutex;
        use axum::response::IntoResponse;
        use axum::routing::post;
        let worker: std::sync::Arc<Mutex<Vec<Value>>> = std::sync::Arc::new(Mutex::new(Vec::new()));
        let sink = worker.clone();
        let router = axum::Router::new().route(
            "/worker",
            post(move |axum::Json(body): axum::Json<Value>| {
                let sink = sink.clone();
                async move {
                    sink.lock().unwrap().push(body);
                    let sse = concat!(
                        "data: {\"id\":\"c1\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"worker outcome\"}}]}\n\n",
                        "data: {\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n",
                        "data: [DONE]\n\n",
                    );
                    axum::http::Response::builder()
                        .header("content-type", "text/event-stream")
                        .body(axum::body::Body::from(sse))
                        .unwrap()
                        .into_response()
                }
            }),
        );
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move { axum::serve(listener, router).await.unwrap() });
        let app = mock_app(
            &format!(
                r#"
                [server]
                [providers.worker]
                base_url = "http://{addr}/worker"
                models = ["w"]
                [[search.providers]]
                name = "s"
                kind = "brave"
                api_key = "k"
                base_url = "http://{addr}/search"
                "#
            ),
            "subagent_tools",
        );
        let ctx = ToolCtx {
            app: &app,
            call_id: "call_1",
            params: json!({
                "model": "worker/w",
                "tools": [{"type": "openrouter:web_search"}],
            }),
            now: jiff::Timestamp::now(),
        };
        let ran = Tool::Subagent
            .execute(
                &ctx,
                &json!({"task_name": "t", "task_description": "summarize rust"}),
            )
            .await
            .unwrap();
        assert!(ran.model_output.contains("worker outcome"), "{}", ran.model_output);
        assert!(ran.model_output.contains("t"), "the task name rides the result: {}", ran.model_output);

        let bodies = worker.lock().unwrap();
        let first = &bodies[0];
        let messages = first["messages"].as_array().expect("the worker got messages");
        assert_eq!(messages.len(), 1, "only the task description: {first}");
        assert_eq!(messages[0]["role"], "user", "{first}");
        assert_eq!(messages[0]["content"], "summarize rust", "{first}");
        assert!(
            first["tools"].as_array().is_some_and(|ts| {
                ts.iter().any(|t| t["function"]["name"] == "pxy_web_search")
            }),
            "the declared served tool must be offered: {first}"
        );
    }

    /// The panel runs every model at once and keeps each answer under the
    /// model that gave it. Three legs that all wait on one barrier can only
    /// pass if the panel issued them concurrently; a sequential panel would
    /// hang on the first wait.
    #[tokio::test]
    async fn panel_runs_its_models_concurrently_and_labels_each_answer() {
        use std::sync::Arc;
        use axum::routing::post;
        use tokio::sync::Barrier;
        let barrier = Arc::new(Barrier::new(3));
        let router = axum::Router::new().route(
            "/c",
            post(move |axum::Json(body): axum::Json<Value>| {
                let barrier = barrier.clone();
                async move {
                    // Only a concurrently-issued panel reaches all three.
                    barrier.wait().await;
                    let model = body["model"].as_str().unwrap_or("?");
                    axum::Json(json!({"id": "x", "choices": [{"index": 0,
                        "message": {"role": "assistant",
                            "content": format!("answer from {model}")},
                        "finish_reason": "stop"}]}))
                }
            }),
        );
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move { axum::serve(listener, router).await.unwrap() });
        let app = mock_app(
            &format!(
                r#"
                [server]
                [providers.p]
                base_url = "http://{addr}/c"
                models = ["a", "b", "c"]
                "#
            ),
            "panel_parallel",
        );
        let models =
            vec!["p/a".to_string(), "p/b".to_string(), "p/c".to_string()];
        let results = tokio::time::timeout(
            std::time::Duration::from_secs(10),
            run_panel(&app, &models, "compare answers", &json!({})),
        )
        .await
        .expect("the panel must not serialise: the barrier never released");

        assert_eq!(results.len(), 3);
        assert_eq!(results[0].0, "p/a");
        assert_eq!(results[0].1.as_ref().unwrap(), "answer from a");
        assert_eq!(results[1].0, "p/b");
        assert_eq!(results[1].1.as_ref().unwrap(), "answer from b");
        assert_eq!(results[2].0, "p/c");
        assert_eq!(results[2].1.as_ref().unwrap(), "answer from c");
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
