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

use super::{tool_search, web_search};
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
    Fusion,
    ToolSearch,
    DescribeImage,
    Memory,
    FindDocs,
    Verify,
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
            Tool::Fusion,
            Tool::ToolSearch,
            Tool::DescribeImage,
            Tool::Memory,
            Tool::FindDocs,
            Tool::Verify,
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
            Tool::Fusion => "fusion",
            Tool::ToolSearch => "tool_search",
            Tool::DescribeImage => "describe_image",
            Tool::Memory => "memory",
            Tool::FindDocs => "find_docs",
            Tool::Verify => "verify",
        }
    }

    /// Whether this tool's executor issues a sub-request through pxy's own
    /// router. A meta-tool is stripped from a sub-request, so recursion stops
    /// at one level.
    pub fn is_meta(self) -> bool {
        matches!(self, Tool::Advisor | Tool::Subagent | Tool::Fusion | Tool::DescribeImage)
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
            Tool::Fusion => run_fusion(ctx, args).await,
            Tool::ToolSearch => run_tool_search(ctx, args),
            Tool::DescribeImage => run_describe_image(ctx, args).await,
            Tool::Memory => run_memory(ctx, args),
            Tool::FindDocs => run_find_docs(ctx, args).await,
            Tool::Verify => run_verify(ctx, args).await,
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
            // A meta-tool needs only the router it is already in; fusion
            // additionally needs a panel to convene.
            Tool::Advisor | Tool::Subagent => true,
            Tool::Fusion => !app.cfg.server_tools.fusion_panel.is_empty(),
            // The search runs over the request's own deferred tools, so it
            // needs nothing of the app at all.
            Tool::ToolSearch => true,
            // Its vision model is named on the declaration, and a declaration
            // that names none is an error the calling model reads.
            Tool::DescribeImage => true,
            // A directory under the data dir, made on first use.
            Tool::Memory => true,
            // Context7 answers without an account, so the key is optional and
            // there is nothing to configure before the tool can be served.
            Tool::FindDocs => true,
            // Jev rides whichever gateways the config lists; no chain, no tool.
            Tool::Verify => {
                !crate::media::resolve_chain(&app.cfg, crate::media::Capability::SystemOne, "")
                    .is_empty()
            }
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
    /// The client behind this turn: a meta-tool's leg bills to its agent and
    /// keeps its opencode session affinity, exactly as the turn itself does.
    pub agent: Option<String>,
    pub session: Option<String>,
    /// The model id the client requested. A meta-tool whose declaration pins
    /// no model consults this one, as OpenRouter's does.
    pub outer_model: String,
    /// Search results already returned to the model this turn, for
    /// web_search's `max_total_results`.
    pub results_used: u64,
    /// The turn's conversation as the upstream sees it, for an advisor
    /// declared with `forward_transcript`.
    pub transcript: Vec<Value>,
    /// The client function definitions held out of the upstream body this
    /// turn, for tool_search to match against.
    pub deferred: Vec<Value>,
    /// The image URLs swapped out of the transcript for a `vision = false`
    /// candidate, in the order the placeholders number them, so describe_image
    /// resolves `[image 2]` to the second entry.
    pub images: Vec<String>,
}

impl<'a> ToolCtx<'a> {
    pub fn new(app: &'a SharedApp, call_id: &'a str) -> Self {
        Self {
            app,
            call_id,
            params: Value::Null,
            now: jiff::Timestamp::now(),
            agent: None,
            session: None,
            outer_model: String::new(),
            results_used: 0,
            transcript: Vec::new(),
            deferred: Vec::new(),
            images: Vec::new(),
        }
    }

    /// The declaration's model, else the client's own: OpenRouter's fallback
    /// for every meta-tool. `None` when neither names one.
    fn declared_or_outer_model(&self) -> Option<&str> {
        self.params["model"]
            .as_str()
            .filter(|m| !m.is_empty())
            .or_else(|| Some(self.outer_model.as_str()).filter(|m| !m.is_empty()))
    }

    /// The context a sub-request through pxy's router runs under: the
    /// client's own agent and session, one level deeper.
    pub fn sub_context(&self) -> ClientContext {
        ClientContext {
            agent: self.agent.clone(),
            session: self.session.clone(),
            tool_depth: 1,
            ..ClientContext::default()
        }
    }
}

/// Issue one sub-request through pxy's own router and return the assistant's
/// text. The meta-tools consult a model this way: the leg resolves providers,
/// passes limits and cooldowns, translates, records usage and fails over
/// exactly like a client turn, because it is one. `params` is merged into the
/// payload beside `model`, `messages` and `stream`, which pxy owns here.
/// `caller` is the client's context (agent, session); the leg always runs one
/// level deeper, so a meta-tool can never re-enter one.
pub async fn run_internal_chat(
    app: &SharedApp,
    model: &str,
    messages: Vec<Value>,
    params: Value,
    caller: &ClientContext,
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
    let ctx = ClientContext { tool_depth: 1, ..caller.clone() };
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
    caller: &ClientContext,
) -> Vec<(String, Result<String, String>)> {
    let legs = models.iter().map(|model| {
        let messages = vec![json!({"role": "user", "content": prompt})];
        let params = params.clone();
        async move {
            let answer = run_internal_chat(app, model, messages, params, caller).await;
            (model.clone(), answer)
        }
    });
    futures_util::future::join_all(legs).await
}

/// The analyst's standing instruction: compare the panel, answer in JSON. It
/// is the system turn of every analyst sub-request.
const ANALYST_INSTRUCTIONS: &str = "You are the analyst of a panel of models. \
Compare the panel's answers to the question and reply with a single JSON \
object and nothing else, with exactly these keys: \
\"consensus\" (points all or most answers agree on, as strings), \
\"contradictions\" (objects with \"topic\" and \"stances\", each stance an \
object with \"model\" and \"stance\"), \
\"partial_coverage\" (objects with \"models\" and \"point\", for points only \
some answers covered), \
\"unique_insights\" (objects with \"model\" and \"insight\", for what only one \
answer raised) and \
\"blind_spots\" (topics no answer addressed, as strings). \
Treat what most agree on as higher-confidence, keep every contradiction, and \
never merge the answers into one: a writer produces the final answer from \
your analysis.";

/// Have one model analyse the panel's answers. The analyst sees the original
/// question and every leg's answer, labelled by the model that gave it; a leg
/// that failed is shown as a failure, not hidden. It always runs at
/// temperature 0. Its reply must be a JSON object — a reply that will not
/// parse is an error, and the caller keeps the raw panel.
pub async fn run_analyst(
    app: &SharedApp,
    model: &str,
    prompt: &str,
    answers: &[(String, Result<String, String>)],
    params: &Value,
    caller: &ClientContext,
) -> Result<Value, String> {
    let mut panel = String::new();
    for (name, answer) in answers {
        match answer {
            Ok(text) => panel.push_str(&format!("### {name}\n{text}\n\n")),
            Err(e) => panel.push_str(&format!("### {name}\n[failed: {e}]\n\n")),
        }
    }
    let messages = vec![
        json!({"role": "system", "content": ANALYST_INSTRUCTIONS}),
        json!({
            "role": "user",
            "content": format!("Question:\n{prompt}\n\nPanel answers:\n{panel}"),
        }),
    ];
    let mut params = params.clone();
    params["temperature"] = json!(0);
    let text = match run_internal_chat(app, model, messages.clone(), params.clone(), caller).await {
        Ok(text) => text,
        // A reasoning model that refuses the parameter outright (gpt-5.6:
        // "'temperature' is not supported with this model") still makes a
        // fine analyst at its own default.
        Err(e) if e.contains("temperature") => {
            params.as_object_mut().map(|p| p.remove("temperature"));
            run_internal_chat(app, model, messages, params, caller).await?
        }
        Err(e) => return Err(e),
    };
    let analysis: Value =
        serde_json::from_str(text.trim()).map_err(|e| format!("analyst returned non-JSON: {e}"))?;
    if !analysis.is_object() {
        return Err("analyst returned JSON that is not an object".to_string());
    }
    Ok(analysis)
}

/// The declaration's options that ride a meta-tool's sub-request: the served
/// tools it lists (or `default_tools` when it lists none at all), so the
/// sub-request can use them, and the token, sampling and step caps.
/// `run_internal_chat` owns `model`, `messages` and `stream`.
fn leg_params(declaration: &Value, default_tools: &[Tool]) -> Value {
    let mut params = json!({});
    let tools: Vec<Value> = match declaration["tools"].as_array() {
        Some(listed) => listed.clone(),
        None => default_tools.iter().map(|t| json!({"type": format!("pxy:{}", t.name())})).collect(),
    };
    if !tools.is_empty() {
        params["tools"] = Value::Array(tools);
    }
    for key in ["max_tool_calls", "max_completion_tokens", "temperature", "reasoning"] {
        if !declaration[key].is_null() {
            params[key] = declaration[key].clone();
        }
    }
    params
}

/// The output budget a fusion leg gets when its declaration sets none:
/// OpenRouter's default, sized so a reasoning-heavy panelist still produces
/// visible text.
const FUSION_LEG_MAX_COMPLETION_TOKENS: u64 = 16_000;

/// How many models a declaration may put on the panel.
const FUSION_PANEL_MAX: usize = 8;

/// OpenRouter's typed reason for a fusion run that produced nothing useful,
/// read off the legs' failures.
fn fusion_failure_reason(errors: &[&String]) -> &'static str {
    let any = |needle: &[&str]| errors.iter().any(|e| needle.iter().any(|n| e.contains(n)));
    if any(&["(402)", "insufficient credit", "insufficient_credits", "Insufficient credit"]) {
        "insufficient_credits"
    } else if any(&["(429)", "rate limit", "rate_limit", "Rate limit"]) {
        "rate_limited"
    } else {
        "all_panels_failed"
    }
}

/// Run one fusion call: every panel member answers the same prompt at once,
/// one analyst compares the answers, and the outer model writes the final
/// answer from the analysis. The panel is the declaration's
/// `analysis_models` (at most eight) over `[server_tools] fusion_panel`; the
/// analyst is the declaration's `model` over `fusion_analyst`, over the
/// client's own model, over the first panel member. The model calling the
/// tool never chooses either. Every leg has web_search and web_fetch unless
/// the declaration lists `tools` itself. Degradation is deliberate — a failed
/// analyst leaves the raw panel intact, a failed member is listed beside the
/// answers, and only an all-failed panel is an error, with a typed reason.
async fn run_fusion(ctx: &ToolCtx<'_>, args: &Value) -> Result<Ran, String> {
    let prompt = args["prompt"].as_str().filter(|p| !p.is_empty()).ok_or("missing prompt")?;
    let declared: Vec<String> = ctx.params["analysis_models"]
        .as_array()
        .into_iter()
        .flatten()
        .filter_map(|m| m.as_str())
        .filter(|m| !m.is_empty())
        .map(str::to_string)
        .collect();
    let mut panel = if declared.is_empty() {
        ctx.app.cfg.server_tools.fusion_panel.clone()
    } else {
        declared
    };
    panel.truncate(FUSION_PANEL_MAX);
    if panel.is_empty() {
        return Err("no fusion panel configured".to_string());
    }
    let analyst = ctx.params["model"]
        .as_str()
        .filter(|m| !m.is_empty())
        .map(str::to_string)
        .or_else(|| ctx.app.cfg.server_tools.fusion_analyst.clone())
        .or_else(|| Some(ctx.outer_model.clone()).filter(|m| !m.is_empty()))
        .unwrap_or_else(|| panel[0].clone());
    let mut params = leg_params(&ctx.params, &[Tool::WebSearch, Tool::WebFetch]);
    if params["max_completion_tokens"].is_null() {
        params["max_completion_tokens"] = json!(FUSION_LEG_MAX_COMPLETION_TOKENS);
    }

    let caller = ctx.sub_context();
    let answers = run_panel(ctx.app, &panel, prompt, &params, &caller).await;
    let responses: Vec<Value> = answers
        .iter()
        .filter_map(|(model, answer)| {
            answer.as_ref().ok().map(|content| json!({"model": model, "content": content}))
        })
        .collect();
    let failed_models: Vec<Value> = answers
        .iter()
        .filter_map(|(model, answer)| {
            answer.as_ref().err().map(|error| json!({"model": model, "error": error}))
        })
        .collect();
    if responses.is_empty() {
        let errors: Vec<&String> = answers.iter().filter_map(|(_, a)| a.as_ref().err()).collect();
        warn!(panel = panel.len(), "fusion panel failed");
        return Ok(Ran {
            model_output: json!({
                "status": "error",
                "error": "all panel models failed",
                "failure_reason": fusion_failure_reason(&errors),
                "failed_models": failed_models,
            })
            .to_string(),
            client: fusion_render(ctx.call_id, "error"),
        });
    }

    let mut result = json!({"status": "ok", "responses": responses});
    if !failed_models.is_empty() {
        result["failed_models"] = Value::Array(failed_models);
    }
    match run_analyst(ctx.app, &analyst, prompt, &answers, &params, &caller).await {
        Ok(analysis) => result["analysis"] = analysis,
        Err(e) => warn!(%analyst, error = %e, "fusion analyst failed; returning the panel"),
    }
    info!(panel = panel.len(), "fusion served");
    Ok(Ran { model_output: result.to_string(), client: fusion_render(ctx.call_id, "ok") })
}

/// Fusion's client material. No dialect documents a fusion result block, so
/// the marker carries the status for the dialect layer to read.
fn fusion_render(call_id: &str, status: &str) -> ClientRender {
    ClientRender { blocks: Vec::new(), marker: json!({"id": call_id, "status": status}) }
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
        "openrouter:fusion" | "pxy:fusion" => Some(Tool::Fusion),
        "openrouter:tool_search" | "pxy:tool_search" | "tool_search" => Some(Tool::ToolSearch),
        // pxy's own; OpenRouter has no equivalent, so there is one spelling.
        "pxy:describe_image" => Some(Tool::DescribeImage),
        // Anthropic's native memory entry is dated; pxy serves the one command
        // set that date names.
        "pxy:memory" | "memory_20250818" => Some(Tool::Memory),
        // pxy's own, like describe_image: OpenRouter has no docs tool.
        "pxy:find_docs" => Some(Tool::FindDocs),
        "pxy:verify" => Some(Tool::Verify),
        // Anthropic names the matcher in the type. Only the regex one is pxy's;
        // `tool_search_tool_bm25*` falls through to `None` and is unservable,
        // because pxy would otherwise answer a BM25 search with regex results.
        t if t.starts_with("tool_search_tool_regex") => Some(Tool::ToolSearch),
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
pub fn tool_def(tool: Tool, params: &Value) -> Value {
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
        Tool::Datetime => {
            // The model reads the default zone off the description: told
            // "default UTC" it asked for UTC, and the configured zone was
            // never used.
            let zone = params["timezone"].as_str().filter(|z| !z.is_empty()).unwrap_or("UTC");
            json!({
                "type": "function",
                "function": {
                    "name": function_name(Tool::Datetime),
                    "description": format!(
                        "The current date and time. Called with no arguments it answers in \
                         the default zone, {zone}; pass timezone only when a different zone \
                         is wanted."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "timezone": {
                                "type": "string",
                                "description": format!(
                                    "IANA timezone name, e.g. America/New_York. Omit for {zone}."
                                )
                            }
                        },
                    },
                },
            })
        }
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
                        "offset": {
                            "type": "integer",
                            "description": "Skip this many matches, to page through a long result."
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
        Tool::Fusion => json!({
            "type": "function",
            "function": {
                "name": function_name(Tool::Fusion),
                "description": "Convene a panel of models on one question and get a \
                    structured comparison of their answers. Use it when a question \
                    deserves several independent attempts.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "prompt": {
                            "type": "string",
                            "description": "The question for the panel."
                        },
                    },
                    "required": ["prompt"],
                },
            },
        }),
        Tool::ToolSearch => json!({
            "type": "function",
            "function": {
                "name": function_name(Tool::ToolSearch),
                "description": "Find tools that are available but not yet loaded. Some of \
                    this conversation's tools are held back to save context; search for \
                    them by capability before concluding a task cannot be done, and the \
                    matches become callable on the next step.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "pattern": {
                            "type": "string",
                            "description": "A regular expression, matched case-insensitively \
                                against each held-back tool's name, description and argument \
                                names. Alternate with | to cover synonyms."
                        },
                        "limit": {
                            "type": "integer",
                            "description": "How many tools to reveal at most."
                        },
                    },
                    "required": ["pattern"],
                },
            },
        }),
        Tool::DescribeImage => json!({
            "type": "function",
            "function": {
                "name": function_name(Tool::DescribeImage),
                "description": "Look at an image this conversation carries. An image you \
                    cannot see yourself appears in the transcript as [image N]; pass that \
                    number to have a vision model describe it or transcribe its text.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "image": {
                            "type": "string",
                            "description": "The number of an [image N] placeholder, or an \
                                https:// or data: image URL."
                        },
                        "prompt": {
                            "type": "string",
                            "description": "What you want to know about the image. Omit for a \
                                full description."
                        },
                        "mode": {
                            "type": "string",
                            "enum": ["describe", "ocr"],
                            "description": "describe (default) for a description, ocr to \
                                transcribe a document into Markdown."
                        },
                    },
                    "required": ["image"],
                },
            },
        }),
        Tool::Memory => json!({
            "type": "function",
            "function": {
                "name": function_name(Tool::Memory),
                "description": "Your memory directory, which survives between \
                    conversations. Every path starts with /memories.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "command": {
                            "type": "string",
                            "enum": ["view", "create", "str_replace", "insert", "delete", "rename"],
                            "description": "view a directory or a file, create a file \
                                (overwriting it), str_replace or insert in one, delete a \
                                path, or rename one."
                        },
                        "path": {
                            "type": "string",
                            "description": "The path the command acts on, under /memories."
                        },
                        "view_range": {
                            "type": "array",
                            "items": {"type": "integer"},
                            "description": "view: the first and last line to show, 1-based; \
                                -1 as the last means the end of the file."
                        },
                        "file_text": {"type": "string", "description": "create: the whole file."},
                        "old_str": {
                            "type": "string",
                            "description": "str_replace: the text to replace, which must \
                                appear exactly once."
                        },
                        "new_str": {
                            "type": "string",
                            "description": "str_replace: what replaces it. Omit to delete \
                                old_str."
                        },
                        "insert_line": {
                            "type": "integer",
                            "description": "insert: the line to insert after; 0 inserts \
                                before the first line."
                        },
                        "insert_text": {"type": "string", "description": "insert: the text."},
                        "old_path": {"type": "string", "description": "rename: the path now."},
                        "new_path": {"type": "string", "description": "rename: the path after."},
                    },
                    "required": ["command"],
                },
            },
        }),
        Tool::FindDocs => json!({
            "type": "function",
            "function": {
                "name": function_name(Tool::FindDocs),
                "description": "Read a library's current documentation. Use it before \
                    writing against an API you are unsure of, instead of guessing from \
                    memory: the snippets come from the library's own docs, as they are now.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "library": {
                            "type": "string",
                            "description": "The library's name, as it is published \
                                (\"axum\", \"next.js\", \"stripe\"), or its Context7 id \
                                (\"/launchbadge/sqlx\") when the name is shared across \
                                languages and the wrong one came back."
                        },
                        "query": {
                            "type": "string",
                            "description": "The topic you need, in a few words: \
                                \"nested routing\", \"webhook signature verification\"."
                        },
                        "version": {
                            "type": "string",
                            "description": "The version you are on (\"0.8\"), when the API \
                                differs between them. Omit for the current one."
                        },
                        "max_tokens": {
                            "type": "integer",
                            "description": "How much documentation to read back, in tokens. \
                                Default 4000."
                        },
                    },
                    "required": ["library", "query"],
                },
            },
        }),
        Tool::Verify => json!({
            "type": "function",
            "function": {
                "name": function_name(Tool::Verify),
                "description": "Check one claim against one passage of evidence. Answers \
                    supported, contradicted, unsupported or uncertain, with calibrated \
                    probabilities. Use it on a fact you are about to assert from a page \
                    you fetched or a snippet you searched. The verdict is evidence to \
                    weigh, not a ruling: it is right between 68% and 91% of the time.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "claim": {
                            "type": "string",
                            "description": "The single statement to check, in one sentence."
                        },
                        "evidence": {
                            "type": "string",
                            "description": "The passage to check it against — a fetched page, \
                                a search snippet, a quoted paragraph. Not your own reasoning."
                        },
                    },
                    "required": ["claim", "evidence"],
                },
            },
        }),
    }
}

/// Anthropic injects this into the system prompt whenever the memory tool is
/// declared, and a model trained on the tool expects to read it; pxy appends
/// it verbatim (wiki:memory-tool).
pub const MEMORY_PROTOCOL: &str = "IMPORTANT: ALWAYS VIEW YOUR MEMORY DIRECTORY BEFORE DOING ANYTHING ELSE.\n\
MEMORY PROTOCOL:\n\
1. Use the `view` command of your `memory` tool to check for earlier progress.\n\
2. ... (work on the task) ...\n   \
- As you make progress, record status / progress / thoughts etc in your memory.\n\
ASSUME INTERRUPTION: Your context window might be reset at any moment, so you risk losing any progress that is not recorded in your memory directory.";

/// The directory one memory call works in: the declaration's `store`, else the
/// client's agent, else `default`. `Err` is an unserved call — pxy will not
/// guess at a name it cannot put on disk.
fn memory_root(ctx: &ToolCtx<'_>) -> Result<std::path::PathBuf, String> {
    let named = ctx.params["store"]
        .as_str()
        .filter(|s| !s.is_empty())
        .or_else(|| ctx.agent.as_deref().filter(|s| !s.is_empty()))
        .unwrap_or("default");
    let store = named.to_ascii_lowercase();
    let usable = !store.is_empty()
        && store.len() <= 64
        && store.chars().all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '-');
    if !usable {
        return Err(format!("unusable memory store name '{named}'"));
    }
    Ok(crate::config::data_dir().join("memory").join(store))
}

/// Run one memory command against the turn's store. The store answers the
/// model on both arms — a refusal is its own `Error: ` sentence, not the
/// loop's JSON — so only a store pxy cannot name leaves the call unserved.
fn run_memory(ctx: &ToolCtx<'_>, args: &Value) -> Result<Ran, String> {
    let root = memory_root(ctx)?;
    let read_only = ctx.params["read_only"].as_bool().unwrap_or(false);
    let store = crate::memory::MemoryStore::open(root);
    let model_output = match store.run(args, read_only) {
        Ok(out) => out,
        Err(out) => out,
    };
    // No documented Anthropic result block, so the round is served silently on
    // every dialect; the marker is all the client transcript records.
    let path = args["path"].as_str().or_else(|| args["old_path"].as_str()).unwrap_or("");
    Ok(Ran {
        model_output,
        client: ClientRender {
            blocks: Vec::new(),
            marker: json!({
                "id": ctx.call_id,
                "command": args["command"].as_str().unwrap_or(""),
                "path": path,
            }),
        },
    })
}

/// Read a library's documentation out of Context7 (wiki:find-docs). Both
/// legs — the name to an id, the id to snippets — are served from `kv` for a
/// week before Context7 is asked again. A missing library or query leaves the
/// call unserved; everything Context7 does wrong is an error result the model
/// reads and can correct.
async fn run_find_docs(ctx: &ToolCtx<'_>, args: &Value) -> Result<Ran, String> {
    let arg = |key: &str| args[key].as_str().map(str::trim).filter(|s| !s.is_empty());
    let library = arg("library").ok_or("find_docs call without a library")?;
    let query = arg("query").ok_or("find_docs call without a query")?;
    let version = arg("version");
    // The declaration caps the call: a model that asks for the whole library
    // gets what the request was willing to spend.
    let cap = ctx.params["max_tokens"].as_u64().unwrap_or(4000).clamp(500, 20_000);
    let tokens = args["max_tokens"].as_u64().unwrap_or(cap).clamp(500, cap);

    let state = &ctx.app.state;
    let render = |library: &str| ClientRender {
        blocks: Vec::new(),
        marker: json!({"id": ctx.call_id, "library": library, "query": query}),
    };
    let failed = |library: &str, e: String| {
        warn!(%library, %query, error = %e, "find_docs failed");
        Ran {
            model_output: json!({"status": "error", "error": e}).to_string(),
            client: render(library),
        }
    };

    // A Context7 id is the model naming the entry itself: no search, and the
    // version already sits in the id (`/tokio-rs/axum/axum_v0_8_4`).
    let id = if library.starts_with('/') {
        if version.is_some() {
            return Ok(failed(library, "a Context7 id carries its version in the id itself".into()));
        }
        library.to_string()
    } else {
        match search_and_pick(ctx, library, version).await {
            Ok(id) => id,
            Err(e) => return Ok(failed(library, e)),
        }
    };

    let key = crate::docs::docs_key(&id, query, tokens);
    let text = match crate::docs::cache_get(state, &key, crate::docs::CACHE_TTL_SECS) {
        Some(cached) => cached.as_str().unwrap_or_default().to_string(),
        None => match crate::docs::fetch_docs(ctx.app, &id, query, tokens).await {
            Ok(text) => {
                if !text.trim().is_empty() {
                    crate::docs::cache_put(state, &key, &Value::String(text.clone()));
                }
                text
            }
            Err(e) => return Ok(failed(&id, e)),
        },
    };

    info!(%id, %query, tokens, "find_docs served");
    Ok(Ran {
        model_output: format!(
            "Documentation for {id} (context7), topic \"{query}\":\n{text}\nSource: context7:{id}"
        ),
        client: render(&id),
    })
}

/// Jev's verdict on one claim against one passage (wiki:jev). One choice
/// question, three options; the confidence it reports is the whole reason the
/// tool exists, so a confidence under the declaration's `abstain_below` is
/// answered `uncertain` rather than passed off as a verdict.
/// Context7's search for a library name, cached, then the pick.
async fn search_and_pick(ctx: &ToolCtx<'_>, library: &str, version: Option<&str>) -> Result<String, String> {
    let state = &ctx.app.state;
    let key = crate::docs::search_key(library);
    let results = match crate::docs::cache_get(state, &key, crate::docs::CACHE_TTL_SECS) {
        Some(cached) => cached.as_array().cloned().unwrap_or_default(),
        None => {
            let results = crate::docs::search_library(ctx.app, library).await?;
            // A miss is the one answer likely to change: a library Context7
            // has not indexed yet is listed next week, and remembering the
            // miss would hide it until then.
            if !results.is_empty() {
                crate::docs::cache_put(state, &key, &Value::Array(results.clone()));
            }
            results
        }
    };
    crate::docs::pick_library(&results, library, version)
}

async fn run_verify(ctx: &ToolCtx<'_>, args: &Value) -> Result<Ran, String> {
    let arg = |key: &str| args[key].as_str().map(str::trim).filter(|s| !s.is_empty());
    let claim = arg("claim").ok_or("verify call without a claim")?;
    let evidence = arg("evidence").ok_or("verify call without evidence")?;
    let abstain_below = ctx.params["abstain_below"].as_f64().unwrap_or(0.5);

    let questions = json!({"support": {
        "type": "choice",
        "instructions": "Does the evidence support the claim?",
        "criteria": {
            "supported": "The evidence states or directly implies the claim",
            "contradicted": "The evidence states the opposite of the claim",
            "unsupported": "The evidence does not address the claim",
        },
    }});
    let state = json!({"claim": claim, "evidence": evidence});

    let answered = crate::media::systemone::ask(ctx.app, "auto", state, questions).await;
    let body = match answered {
        Ok(body) => body,
        Err(e) => {
            warn!(%claim, error = %e, "verify failed");
            return Ok(Ran {
                model_output: json!({"status": "error", "error": e}).to_string(),
                client: ClientRender {
                    blocks: Vec::new(),
                    marker: json!({"id": ctx.call_id, "verdict": "error"}),
                },
            });
        }
    };

    let answer = &body["answers"]["support"];
    let confidence = answer["confidence"].as_f64().unwrap_or(0.0);
    let verdict = match answer["choice"].as_str() {
        Some(choice) if confidence >= abstain_below => choice,
        _ => "uncertain",
    };
    info!(%claim, %verdict, confidence, "verify served");
    Ok(Ran {
        model_output: json!({
            "status": "ok",
            "verdict": verdict,
            "probabilities": answer["probabilities"],
            "confidence": confidence,
            "model": body["model"],
        })
        .to_string(),
        client: ClientRender {
            blocks: Vec::new(),
            marker: json!({"id": ctx.call_id, "verdict": verdict}),
        },
    })
}

/// Score every hit against the query in one Jev call, drop what scores under
/// `min`, sort the rest by score and cut to `n` (wiki:jev). Search providers
/// rank for their own purposes, and the model pays for every snippet it is
/// shown, so a relevance score is worth one 70–500 ms call.
///
/// Fails open, in place: a Jev that is down, slow or answering nonsense leaves
/// the provider's list and its order exactly as they came, because a worse
/// order is better than no results.
async fn rerank_hits(ctx: &ToolCtx<'_>, query: &str, hits: &mut Vec<Value>, n: usize, min: f64) {
    if hits.is_empty() {
        return;
    }
    let numbered: Vec<Value> = hits
        .iter()
        .enumerate()
        .map(|(i, h)| json!({"result": i, "title": h["title"], "url": h["url"], "snippet": h["snippet"]}))
        .collect();
    let questions: serde_json::Map<String, Value> = (0..hits.len())
        .map(|i| {
            (
                format!("r{i}"),
                json!({
                    "type": "noul",
                    "instructions": format!("Is result {i} relevant to the query?"),
                }),
            )
        })
        .collect();

    let state = json!({"query": query, "results": numbered});
    let body = match crate::media::systemone::ask(
        ctx.app,
        "auto",
        state,
        Value::Object(questions),
    )
    .await
    {
        Ok(body) => body,
        Err(e) => {
            warn!(%query, error = %e, "web_search re-ranking failed; keeping the provider's order");
            hits.truncate(n);
            return;
        }
    };

    let answers = &body["answers"];
    let scores: Vec<Option<f64>> =
        (0..hits.len()).map(|i| answers[format!("r{i}")]["noul"].as_f64()).collect();
    // Every hit unscored is an answer pxy cannot read, not a verdict that
    // nothing is relevant: fail open rather than hand the model an empty list.
    if scores.iter().all(Option::is_none) {
        warn!(%query, "web_search re-ranking scored nothing; keeping the provider's order");
        hits.truncate(n);
        return;
    }
    let mut scored: Vec<(f64, Value)> = hits
        .drain(..)
        .zip(scores)
        .filter_map(|(hit, score)| score.filter(|s| *s >= min).map(|s| (s, hit)))
        .collect();
    scored.sort_by(|a, b| b.0.total_cmp(&a.0));
    scored.truncate(n);
    let kept = scored.len();
    *hits = scored.into_iter().map(|(_, hit)| hit).collect();
    info!(%query, kept, "web_search re-ranked");
}

/// A declaration's `engine`: `auto` (or absent) walks the configured pool;
/// a configured provider's name pins the walk to it; anything else is an
/// error the model is told about, never a request failure.
fn engine_choice<'a>(
    params: &'a Value,
    configured: impl Iterator<Item = &'a str>,
) -> Result<Option<&'a str>, String> {
    let Some(engine) = params["engine"].as_str().filter(|e| !e.is_empty() && *e != "auto") else {
        return Ok(None);
    };
    let names: Vec<&str> = configured.collect();
    if names.contains(&engine) {
        Ok(Some(engine))
    } else {
        Err(format!("unknown engine '{engine}'; configured: {}", names.join(", ")))
    }
}

/// The strings of a declaration's list parameter (`allowed_domains`,
/// `excluded_domains`, `blocked_domains`, …).
fn param_list(params: &Value, key: &str) -> Vec<String> {
    params[key]
        .as_array()
        .into_iter()
        .flatten()
        .filter_map(|v| v.as_str())
        .map(|d| d.trim().trim_start_matches("*.").to_ascii_lowercase())
        .filter(|d| !d.is_empty())
        .collect()
}

/// The host of a URL, lowercased, without port or credentials.
fn url_host(url: &str) -> String {
    let rest = url.split_once("://").map(|(_, r)| r).unwrap_or(url);
    let authority = rest.split(['/', '?', '#']).next().unwrap_or("");
    let host = authority.rsplit('@').next().unwrap_or(authority);
    host.split(':').next().unwrap_or("").to_ascii_lowercase()
}

/// Whether a URL's host passes the declaration's domain lists: a listed
/// domain matches itself and every subdomain (`example.com` covers
/// `docs.example.com`). An empty `allowed` allows everything; `blocked`
/// always wins.
pub fn domain_allowed(url: &str, allowed: &[String], blocked: &[String]) -> bool {
    let host = url_host(url);
    let matches = |d: &String| host == *d || host.ends_with(&format!(".{d}"));
    if blocked.iter().any(matches) {
        return false;
    }
    allowed.is_empty() || allowed.iter().any(matches)
}

/// A string cut to at most `max` characters, on a character boundary.
fn cut_chars(text: &str, max: usize) -> &str {
    match text.char_indices().nth(max) {
        Some((cut, _)) => &text[..cut],
        None => text,
    }
}

/// Run one web_search call through the provider walk. A missing or empty
/// query argument makes the call unserved; a provider failure is reported to
/// the model, the way the real API does. The declaration's `parameters`
/// shape the search: `max_results` per call, `max_total_results` per turn,
/// `allowed_domains` / `excluded_domains` on the hits, `max_characters` on
/// each snippet, `engine` on the provider walked.
async fn run_web_search(ctx: &ToolCtx<'_>, args: &Value) -> Result<Ran, String> {
    let query =
        args["query"].as_str().filter(|q| !q.is_empty()).ok_or("missing query")?.to_string();
    let params = &ctx.params;
    let unrun = |message: String| Ran {
        model_output: format!("Search failed: {message}"),
        client: ClientRender { blocks: Vec::new(), marker: Value::Null },
    };
    let only = match engine_choice(params, ctx.app.cfg.search.providers.iter().map(|p| p.name.as_str()))
    {
        Ok(only) => only,
        Err(e) => return Ok(unrun(e)),
    };
    let max_results = params["max_results"].as_u64().unwrap_or(5).clamp(1, 25);
    let n = match params["max_total_results"].as_u64() {
        Some(total) if ctx.results_used >= total => {
            return Ok(unrun(format!(
                "max_total_results ({total}) already returned this turn; answer from what you have"
            )));
        }
        Some(total) => max_results.min(total - ctx.results_used),
        None => max_results,
    };
    let allowed = param_list(params, "allowed_domains");
    let excluded = param_list(params, "excluded_domains");
    let max_chars = params["max_characters"].as_u64().map(|c| c.clamp(1, 100_000) as usize);

    // Re-ranking needs more than `n` to choose between, so the provider is
    // asked for a wider list and Jev cuts it back.
    let jev = &ctx.app.cfg.server_tools.jev;
    let asked = if jev.rerank_web_search { (n * 3).max(10).min(25) } else { n };

    let found = crate::media::search::run_search(ctx.app, &query, asked, only).await;
    Ok(match found {
        Ok((provider, mut results)) => {
            if jev.rerank_web_search {
                rerank_hits(ctx, &query, &mut results, n as usize, jev.rerank_min).await;
            }
            results.retain(|r| domain_allowed(r["url"].as_str().unwrap_or(""), &allowed, &excluded));
            if let Some(max) = max_chars {
                for r in &mut results {
                    if let Some(snippet) = r["snippet"].as_str().map(|t| cut_chars(t, max).to_string()) {
                        r["snippet"] = json!(snippet);
                    }
                }
            }
            info!(%query, %provider, hits = results.len(), "web_search served");
            Ran {
                model_output: web_search::results_for_model(&results),
                client: ClientRender {
                    blocks: vec![
                        web_search::server_tool_use_block(ctx.call_id, &query),
                        web_search::result_block(ctx.call_id, &results),
                    ],
                    marker: json!({"id": ctx.call_id, "query": &query, "hits": results.len()}),
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
                    marker: json!({"id": ctx.call_id, "query": &query, "hits": 0}),
                },
            }
        }
    })
}

/// How much of a fetched page the model is shown when the declaration sets
/// no `max_content_tokens`. One fetch must not be able to swallow the context
/// window; the tail is dropped, not summarised.
const FETCH_CONTENT_CHARS: usize = 100_000;

/// The model-facing result for a fetched page, in OpenRouter's documented
/// shape: the URL, the extracted content (capped, and marked when cut), the
/// status and when it was read.
fn fetched_for_model(url: &str, content: &str, max_chars: usize, retrieved_at: &str) -> String {
    let body = match content.char_indices().nth(max_chars) {
        Some((cut, _)) => format!("{}\n\n[truncated]", &content[..cut]),
        None => content.to_string(),
    };
    json!({"url": url, "content": body, "status": "completed", "retrieved_at": retrieved_at})
        .to_string()
}

/// The model-facing result for a fetch that did not happen or failed.
fn fetch_failed(url: &str, error: &str) -> String {
    json!({"url": url, "status": "failed", "error": error}).to_string()
}

/// Run one web_fetch call through the fetch provider walk (the same walk
/// `/v1/fetch` runs). A missing URL makes the call unserved; a provider
/// failure is reported to the model. The declaration's `allowed_domains` /
/// `blocked_domains` are checked before anything is fetched,
/// `max_content_tokens` (four characters each) caps what the model is shown,
/// and `engine` pins the walk to one configured provider. web_fetch has no
/// documented Anthropic result block, so it renders nothing for a Messages
/// client.
async fn run_web_fetch(ctx: &ToolCtx<'_>, args: &Value) -> Result<Ran, String> {
    let url = args["url"].as_str().filter(|u| !u.is_empty()).ok_or("missing url")?.to_string();
    let params = &ctx.params;
    let render = || ClientRender { blocks: Vec::new(), marker: json!({"id": ctx.call_id, "url": &url}) };
    let failed = |error: String| Ran { model_output: fetch_failed(&url, &error), client: render() };
    // The same check /v1/fetch makes: a reader endpoint is not a general
    // fetcher, and a self-hosted base_url must not widen what a model can ask
    // it to read.
    if !url.starts_with("http://") && !url.starts_with("https://") {
        return Ok(failed("url must be http(s)".into()));
    }
    let allowed = param_list(params, "allowed_domains");
    let blocked = param_list(params, "blocked_domains");
    if !domain_allowed(&url, &allowed, &blocked) {
        return Ok(failed("domain not allowed by the tool's domain lists".into()));
    }
    let only = match engine_choice(params, ctx.app.cfg.fetch.providers.iter().map(|p| p.name.as_str()))
    {
        Ok(only) => only,
        Err(e) => return Ok(failed(e)),
    };
    let max_chars = params["max_content_tokens"]
        .as_u64()
        .map(|t| t.saturating_mul(4).max(1) as usize)
        .unwrap_or(FETCH_CONTENT_CHARS);
    let found = crate::media::search::run_fetch(ctx.app, &url, only).await;
    Ok(match &found {
        Ok((provider, content)) => {
            info!(%url, %provider, bytes = content.len(), "web_fetch served");
            let retrieved_at = ctx.now.to_string();
            Ran {
                model_output: fetched_for_model(&url, content, max_chars, &retrieved_at),
                client: render(),
            }
        }
        Err(e) => {
            warn!(%url, error = %e, "web_fetch failed");
            failed(e.clone())
        }
    })
}

/// The current time, in OpenRouter's documented shape: `datetime` as
/// ISO-8601 with milliseconds and offset, and the zone it is in. No network.
/// The call's own `timezone` wins over the declaration's, which wins over
/// UTC; a zone jiff does not know is reported to the model rather than
/// failing the call.
fn run_datetime(ctx: &ToolCtx<'_>, args: &Value) -> Result<Ran, String> {
    let requested = args["timezone"]
        .as_str()
        .or_else(|| ctx.params["timezone"].as_str())
        .filter(|z| !z.is_empty());
    let (zoned, label) = match requested {
        Some(name) => match ctx.now.in_tz(name) {
            Ok(z) => (z, name.to_string()),
            Err(_) => {
                return Ok(Ran {
                    model_output: json!({"error": format!("Unknown timezone: {name}")}).to_string(),
                    client: ClientRender {
                        blocks: Vec::new(),
                        marker: json!({"id": ctx.call_id, "timezone": name}),
                    },
                });
            }
        },
        None => (ctx.now.to_zoned(jiff::tz::TimeZone::UTC), "UTC".to_string()),
    };
    let datetime = format!(
        "{}.{:03}{}",
        zoned.strftime("%Y-%m-%dT%H:%M:%S"),
        zoned.millisecond(),
        zoned.strftime("%:z")
    );
    Ok(Ran {
        model_output: json!({"datetime": datetime, "timezone": label}).to_string(),
        client: ClientRender {
            blocks: Vec::new(),
            marker: json!({"id": ctx.call_id, "timezone": label}),
        },
    })
}

/// Run one search_models call against pxy's own catalog. Every argument is
/// optional and a missing one filters nothing. The model is told each
/// matching `provider/model`, the facts it routes on, and the group aliases
/// that reach it, in OpenRouter's documented shape: `models`, how many
/// matched (`total_results`) and how many were returned (`showing`), capped
/// by the declaration's `max_results` (default 10, at most 50).
fn run_search_models(ctx: &ToolCtx<'_>, args: &Value) -> Result<Ran, String> {
    let mut matches = matching_models(&ctx.app.catalog, args);
    let total_results = matches.len();
    let max_results = ctx.params["max_results"].as_u64().unwrap_or(10).clamp(1, 50) as usize;
    let offset = (args["offset"].as_u64().unwrap_or(0) as usize).min(matches.len());
    matches.drain(..offset);
    matches.truncate(max_results);
    let output = json!({
        "models": matches,
        "total_results": total_results,
        "showing": matches.len(),
    });
    Ok(Ran {
        model_output: output.to_string(),
        client: ClientRender {
            blocks: Vec::new(),
            marker: json!({"id": ctx.call_id, "matches": matches, "total_results": total_results}),
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

/// Run one tool_search call over the definitions the router held back this
/// turn. Nothing outside the request is consulted, so the only unserved call
/// is one with no pattern at all; an unusable pattern is reported to the
/// model, which can then write a better one.
///
/// The declaration's `max_results` caps the call's own `limit` and stands in
/// for it when the model sends none.
fn run_tool_search(ctx: &ToolCtx<'_>, args: &Value) -> Result<Ran, String> {
    let pattern =
        args["pattern"].as_str().filter(|p| !p.is_empty()).ok_or("missing pattern")?.to_string();
    let max_results = ctx.params["max_results"]
        .as_u64()
        .unwrap_or(tool_search::DEFAULT_MAX_RESULTS)
        .clamp(1, 50);
    let limit = args["limit"].as_u64().unwrap_or(max_results).clamp(1, max_results);

    let (model_output, found, result) =
        match tool_search::search_deferred(&ctx.deferred, &pattern, limit as usize) {
            Ok(found) => {
                info!(%pattern, revealed = found.len(), "tool_search served");
                let mut output = json!({
                    "status": "ok",
                    "found": found,
                    "total_matches": found.len(),
                });
                if !found.is_empty() {
                    output["note"] = json!("these tools are now available to call");
                }
                let block = tool_search::result_block(ctx.call_id, &found);
                (output, found, block)
            }
            Err(e) => {
                warn!(%pattern, error = %e, "tool_search pattern rejected");
                let block = tool_search::error_block(ctx.call_id, &e);
                (json!({"status": "error", "error": e}), Vec::new(), block)
            }
        };

    Ok(Ran {
        model_output: model_output.to_string(),
        client: ClientRender {
            blocks: vec![
                tool_search::server_tool_use_block(ctx.call_id, &pattern, limit),
                result,
            ],
            // The loop reads `found` off the marker to move those definitions
            // into the continuation body, so it rides every outcome.
            marker: json!({"id": ctx.call_id, "pattern": &pattern, "found": found}),
        },
    })
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

/// Replace every `image_url` part of a chat body with the text part
/// `[image N]`, N counting from 1 across the whole transcript in order, and
/// return the URLs in that same order. The swap is what a `vision = false`
/// candidate gets instead of a 400: the model is told an image was there and
/// can ask describe_image about it by its number (`wiki:vision`).
pub fn swap_image_parts(body: &mut Value) -> Vec<String> {
    let mut urls: Vec<String> = Vec::new();
    let Some(messages) = body["messages"].as_array_mut() else { return urls };
    for message in messages {
        let Some(parts) = message["content"].as_array_mut() else { continue };
        for part in parts {
            if part["type"] != "image_url" {
                continue;
            }
            // `{"url": ".."}` is the documented shape; a bare string is what
            // some clients send, and responses_upstream already reads both.
            let url = part["image_url"]["url"]
                .as_str()
                .or_else(|| part["image_url"].as_str())
                .unwrap_or_default()
                .to_string();
            urls.push(url);
            *part = json!({"type": "text", "text": format!("[image {}]", urls.len())});
        }
    }
    urls
}

/// The turn's conversation as prose an advisor can read without the tools
/// that produced it: system turns stay, text turns stay, an assistant's tool
/// calls become bracketed lines in its text, and a tool result becomes a
/// user turn. No tool needs declaring on the advisor's request, so any
/// upstream accepts it.
fn transcript_as_prose(messages: &[Value]) -> Vec<Value> {
    let text_of = |content: &Value| -> String {
        match content {
            Value::String(s) => s.clone(),
            Value::Array(parts) => parts
                .iter()
                .filter_map(|p| p["text"].as_str())
                .collect::<Vec<_>>()
                .join("\n"),
            _ => String::new(),
        }
    };
    let mut out: Vec<Value> = Vec::new();
    for m in messages {
        let role = m["role"].as_str().unwrap_or("user");
        let mut text = text_of(&m["content"]);
        match role {
            "system" | "developer" => {
                out.push(json!({"role": "system", "content": text}));
                continue;
            }
            "assistant" => {
                for call in m["tool_calls"].as_array().into_iter().flatten() {
                    let name = call["function"]["name"].as_str().unwrap_or("?");
                    let args = call["function"]["arguments"].as_str().unwrap_or("");
                    if !text.is_empty() {
                        text.push('\n');
                    }
                    text.push_str(&format!("[tool call {name} {args}]"));
                }
            }
            "tool" => {
                let id = m["tool_call_id"].as_str().unwrap_or("");
                text = format!("[tool result {id}] {text}");
            }
            _ => {}
        }
        if text.is_empty() {
            continue;
        }
        let role = if role == "assistant" { "assistant" } else { "user" };
        out.push(json!({"role": role, "content": text}));
    }
    out
}

/// Run one advisor call: the model's `prompt` goes to the advisor as a user
/// turn, the declaration's `instructions` as the system turn, and the
/// advisor's answer comes back as the tool result. The advisor model is the
/// declaration's, else the call's own, else the client's model. With
/// `forward_transcript` the advisor also sees the turn's conversation as
/// prose, and the prompt becomes optional. A failure is reported to the outer
/// model, which continues without the advice.
async fn run_advisor(ctx: &ToolCtx<'_>, args: &Value) -> Result<Ran, String> {
    let forward = ctx.params["forward_transcript"].as_bool().unwrap_or(false);
    let prompt = args["prompt"].as_str().filter(|p| !p.is_empty());
    if prompt.is_none() && !forward {
        return Err("missing prompt".to_string());
    }
    let prompt = prompt.unwrap_or("");
    // The declaration's model wins; the call's own is honoured only when the
    // declaration pins none; the client's own model is the last resort.
    let model = ctx.params["model"]
        .as_str()
        .or_else(|| args["model"].as_str())
        .filter(|m| !m.is_empty())
        .or_else(|| Some(ctx.outer_model.as_str()).filter(|m| !m.is_empty()))
        .ok_or("no advisor model")?;
    let mut messages: Vec<Value> = Vec::new();
    if let Some(instructions) = ctx.params["instructions"].as_str().filter(|s| !s.is_empty()) {
        messages.push(json!({"role": "system", "content": instructions}));
    }
    if forward {
        messages.extend(transcript_as_prose(&ctx.transcript));
    }
    if !prompt.is_empty() {
        messages.push(json!({"role": "user", "content": prompt}));
    }
    let knobs = leg_params(&ctx.params, &[]);
    let failed = |e: String| Ran {
        model_output: json!({"status": "error", "error": format!("Advisor call failed: {e}")})
            .to_string(),
        client: ClientRender {
            blocks: Vec::new(),
            marker: json!({"id": ctx.call_id, "model": model, "prompt": prompt}),
        },
    };
    match run_internal_chat(ctx.app, model, messages, knobs, &ctx.sub_context()).await {
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

/// What a `describe` call asks the vision model when the caller says nothing.
const DESCRIBE_PROMPT: &str = "Describe this image in detail, transcribing all visible text.";

/// The same for `ocr`, whose job is the document rather than the picture. The
/// file-parser plugin reads a scanned PDF page with the same words.
pub(crate) const OCR_PROMPT: &str = "Transcribe this document into clean Markdown in natural reading order.";

/// Run one describe_image call: a model that cannot see an image asks one that
/// can. `image` is the number of an `[image N]` placeholder — the URLs swapped
/// out of this turn's transcript, in that order — or an image URL of its own.
/// The leg is a single user message, the prompt then the image: DeepSeek 400s
/// on an image part in a system or assistant message. `Err` is a call pxy will
/// not send at all; a leg that ran and failed reports to the calling model.
async fn run_describe_image(ctx: &ToolCtx<'_>, args: &Value) -> Result<Ran, String> {
    let image = args["image"].as_str().filter(|i| !i.is_empty()).ok_or("missing image")?;
    let mode = match args["mode"].as_str().filter(|m| !m.is_empty()).unwrap_or("describe") {
        m @ ("describe" | "ocr") => m,
        other => return Err(format!("unknown mode '{other}': use describe or ocr")),
    };
    let url = match image.parse::<usize>() {
        // An index the transcript does not have would otherwise be sent as a
        // URL the vision model cannot fetch.
        Ok(n) => ctx
            .images
            .get(n.checked_sub(1).ok_or("image 0: placeholders count from 1")?)
            .ok_or_else(|| format!("no [image {n}] in this conversation"))?
            .as_str(),
        Err(_) if image.starts_with("https://") || image.starts_with("http://") => image,
        Err(_) if image.starts_with("data:") => image,
        Err(_) => return Err(format!("image must be a placeholder number or a URL, got '{image}'")),
    };
    // The declaration's model only: falling back to the model that called the
    // tool would send the image to the one that cannot see it.
    let model = match mode {
        "ocr" => ctx.params["ocr_model"].as_str().or_else(|| ctx.params["model"].as_str()),
        _ => ctx.params["model"].as_str(),
    }
    .filter(|m| !m.is_empty())
    .ok_or("no describe_image model configured")?;
    let prompt = args["prompt"]
        .as_str()
        .filter(|p| !p.is_empty())
        .unwrap_or(if mode == "ocr" { OCR_PROMPT } else { DESCRIBE_PROMPT });
    let messages = vec![json!({"role": "user", "content": [
        {"type": "text", "text": prompt},
        {"type": "image_url", "image_url": {"url": url}},
    ]})];
    let knobs = leg_params(&ctx.params, &[]);
    // No documented Anthropic block for this tool, so every dialect reads the
    // marker and Messages is served silently.
    let render = || ClientRender {
        blocks: Vec::new(),
        marker: json!({"id": ctx.call_id, "model": model, "mode": mode}),
    };
    match run_internal_chat(ctx.app, model, messages, knobs, &ctx.sub_context()).await {
        Ok(description) => {
            info!(%model, mode, "describe_image served");
            Ok(Ran {
                model_output: json!({
                    "status": "ok", "model": model, "mode": mode, "description": description,
                })
                .to_string(),
                client: render(),
            })
        }
        Err(e) => {
            warn!(%model, mode, error = %e, "describe_image failed");
            Ok(Ran {
                model_output: json!({
                    "status": "error",
                    "error": format!("describe_image call failed: {e}"),
                })
                .to_string(),
                client: render(),
            })
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
    // The worker is fixed by the declaration, else it is the client's own
    // model; the delegating model does not choose it.
    let model = ctx.declared_or_outer_model().ok_or("no subagent model")?;
    let mut messages: Vec<Value> = Vec::new();
    if let Some(instructions) = ctx.params["instructions"].as_str().filter(|s| !s.is_empty()) {
        messages.push(json!({"role": "system", "content": instructions}));
    }
    messages.push(json!({"role": "user", "content": task}));
    // The worker's own served tools ride the sub-request: the loop injects and
    // serves them there, exactly as it does for a client turn. `tool_depth` 1
    // strips any meta-tool, so a worker can never re-enter one.
    let params = leg_params(&ctx.params, &[]);
    match run_internal_chat(ctx.app, model, messages, params, &ctx.sub_context()).await {
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
/// to the model. The declaration's `model` is the default when the call names
/// none, and every other parameter (`quality`, `size`, `aspect_ratio`,
/// `background`, `output_format`, …) rides the generation body as given. The
/// model is handed the image URL in OpenRouter's documented shape, not the
/// bytes.
async fn run_image_generation(ctx: &ToolCtx<'_>, args: &Value) -> Result<Ran, String> {
    let prompt = args["prompt"].as_str().filter(|p| !p.is_empty()).ok_or("missing prompt")?;
    let model = args["model"]
        .as_str()
        .or_else(|| ctx.params["model"].as_str())
        .filter(|m| !m.is_empty());
    let mut payload = json!({"prompt": prompt});
    if let (Some(dst), Some(src)) = (payload.as_object_mut(), ctx.params.as_object()) {
        for (k, v) in src {
            if !matches!(k.as_str(), "model" | "max_uses" | "max_tool_calls" | "prompt") {
                dst.insert(k.clone(), v.clone());
            }
        }
    }
    match crate::media::images::run_generate(ctx.app, model, &payload, true).await {
        // The walk is told the tool needs a URL, so a base64-only candidate
        // fails over and a success here carries one.
        Ok((body, provider)) => {
            let url = body["data"][0]["url"].as_str().unwrap_or("");
            info!(%provider, "image_generation served");
            Ok(Ran {
                model_output: json!({"status": "ok", "imageUrl": url}).to_string(),
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
                model_output: json!({
                    "status": "error",
                    "error": format!("Image generation failed ({status}): {detail}"),
                })
                .to_string(),
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

    /// Anthropic names the memory tool by date, and a client that declares it
    /// natively must reach the same executor as `pxy:memory`.
    #[test]
    fn from_type_maps_both_memory_spellings() {
        assert_eq!(from_type("pxy:memory"), Some(Tool::Memory));
        assert_eq!(from_type("memory_20250818"), Some(Tool::Memory));
    }

    /// A store name from the declaration beats the agent, which beats
    /// `default`; a name pxy cannot put on disk leaves the call unserved
    /// rather than landing somewhere it was not asked to write.
    #[test]
    fn memory_store_resolves_from_the_declaration_then_the_agent_then_default() {
        let app = mock_app("[server]", "memory_store_name");
        let under = |name: &str| crate::config::data_dir().join("memory").join(name);

        let ctx = ToolCtx::new(&app, "call_1");
        assert_eq!(memory_root(&ctx), Ok(under("default")));

        let agent = ToolCtx { agent: Some("Claude".into()), ..ToolCtx::new(&app, "call_1") };
        assert_eq!(memory_root(&agent), Ok(under("claude")), "lowercased");

        let declared = ToolCtx {
            params: json!({"store": "shared"}),
            agent: Some("claude".into()),
            ..ToolCtx::new(&app, "call_1")
        };
        assert_eq!(memory_root(&declared), Ok(under("shared")));

        for bad in ["../escape", "a b", &"x".repeat(65)] {
            let ctx = ToolCtx { params: json!({"store": bad}), ..ToolCtx::new(&app, "call_1") };
            assert!(memory_root(&ctx).is_err(), "{bad}");
        }
    }

    /// The round is served silently: the model reads the store's own string,
    /// the client transcript gets a marker and no block at all.
    #[tokio::test]
    async fn memory_is_served_silently_and_keeps_what_it_wrote() {
        let app = mock_app("[server]", "memory_served");
        let store = format!("pxy-test-served-{}", std::process::id());
        let root = crate::config::data_dir().join("memory").join(&store);
        let _ = std::fs::remove_dir_all(&root);
        let ctx = ToolCtx { params: json!({"store": store}), ..ToolCtx::new(&app, "call_1") };

        let created = Tool::Memory
            .execute(&ctx, &json!({"command": "create", "path": "/memories/n.md", "file_text": "kept\n"}))
            .await
            .unwrap();
        assert_eq!(created.model_output, "File created successfully at: /memories/n.md");
        assert!(created.client.blocks.is_empty(), "memory has no Anthropic block");
        assert_eq!(
            created.client.marker,
            json!({"id": "call_1", "command": "create", "path": "/memories/n.md"})
        );

        let viewed = Tool::Memory
            .execute(&ctx, &json!({"command": "view", "path": "/memories/n.md"}))
            .await
            .unwrap();
        assert!(viewed.model_output.ends_with("\n     1\tkept"), "{}", viewed.model_output);

        let read_only =
            ToolCtx { params: json!({"store": store, "read_only": true}), ..ToolCtx::new(&app, "call_2") };
        let refused = Tool::Memory
            .execute(&read_only, &json!({"command": "delete", "path": "/memories/n.md"}))
            .await
            .unwrap();
        assert_eq!(refused.model_output, "Error: memory is read-only");

        let _ = std::fs::remove_dir_all(&root);
    }

    /// A call pxy cannot even shape is unserved, so the loop leaves it out of
    /// the replay instead of spending a Context7 request on half a question.
    #[tokio::test]
    async fn find_docs_without_a_library_or_query_is_unserved() {
        let app = mock_app("[server]", "find_docs_unserved");
        let ctx = ToolCtx::new(&app, "call_1");
        for args in [
            json!({"query": "nested routing"}),
            json!({"library": "axum"}),
            json!({"library": " ", "query": "nested routing"}),
        ] {
            assert!(Tool::FindDocs.execute(&ctx, &args).await.is_err(), "{args}");
        }
    }

    /// Both legs come out of kv while the rows are fresh: the mock app has no
    /// Context7 to fall back on, so an answer at all proves the cache served
    /// it. The model reads text, never the raw JSON.
    #[tokio::test]
    async fn find_docs_reads_both_halves_from_the_cache() {
        let app = mock_app("[server]", "find_docs_cached");
        let results = json!([{"id": "/tokio-rs/axum", "versions": ["axum_v0_8_4"]}]);
        // Written under the name the model used, lowercased.
        crate::docs::cache_put(&app.state, &crate::docs::search_key("Axum"), &results);
        crate::docs::cache_put(
            &app.state,
            &crate::docs::docs_key("/tokio-rs/axum", "nested routing", 4000),
            &json!("### Nest\nRouter::new().nest(\"/api\", api)"),
        );

        let ctx = ToolCtx::new(&app, "call_1");
        let ran = Tool::FindDocs
            .execute(&ctx, &json!({"library": "axum", "query": "nested routing"}))
            .await
            .unwrap();
        assert_eq!(
            ran.model_output,
            "Documentation for /tokio-rs/axum (context7), topic \"nested routing\":\n\
             ### Nest\nRouter::new().nest(\"/api\", api)\n\
             Source: context7:/tokio-rs/axum"
        );
        assert!(ran.client.blocks.is_empty(), "find_docs is served silently");
        assert_eq!(
            ran.client.marker,
            json!({"id": "call_1", "library": "/tokio-rs/axum", "query": "nested routing"})
        );
    }

    /// A library given as a Context7 id is the entry itself: no search runs,
    /// and a version beside it is an error, since the id already carries one.
    #[tokio::test]
    async fn find_docs_takes_a_context7_id_without_searching() {
        let app = mock_app("[server]", "find_docs_pinned");
        crate::docs::cache_put(
            &app.state,
            &crate::docs::docs_key("/launchbadge/sqlx", "sqlite pool", 4000),
            &json!("### SqlitePool"),
        );
        let ctx = ToolCtx::new(&app, "call_1");
        let ran = Tool::FindDocs
            .execute(&ctx, &json!({"library": "/launchbadge/sqlx", "query": "sqlite pool"}))
            .await
            .unwrap();
        assert!(ran.model_output.starts_with("Documentation for /launchbadge/sqlx"), "{}", ran.model_output);
        assert!(
            crate::docs::cache_get(&app.state, &crate::docs::search_key("/launchbadge/sqlx"), u64::MAX).is_none(),
            "a pinned id is never searched for"
        );

        let ran = Tool::FindDocs
            .execute(&ctx, &json!({"library": "/launchbadge/sqlx", "query": "x", "version": "0.8"}))
            .await
            .unwrap();
        assert!(ran.model_output.contains("carries its version"), "{}", ran.model_output);
    }

    /// A library Context7 does not have and a version it does not publish are
    /// error results the model can correct, not failed requests.
    #[tokio::test]
    async fn find_docs_reports_what_context7_could_not_answer() {
        let app = mock_app("[server]", "find_docs_errors");
        let ctx = ToolCtx::new(&app, "call_1");
        let error = |ran: &Ran| -> String {
            let out: Value = serde_json::from_str(&ran.model_output).unwrap();
            assert_eq!(out["status"], "error", "{out}");
            out["error"].as_str().unwrap_or_default().to_string()
        };

        crate::docs::cache_put(&app.state, &crate::docs::search_key("nosuchlib"), &json!([]));
        let ran = Tool::FindDocs
            .execute(&ctx, &json!({"library": "nosuchlib", "query": "x"}))
            .await
            .unwrap();
        assert!(error(&ran).contains("no library matched"), "{}", ran.model_output);

        crate::docs::cache_put(
            &app.state,
            &crate::docs::search_key("axum"),
            &json!([{"id": "/tokio-rs/axum", "versions": ["axum_v0_8_4"]}]),
        );
        let ran = Tool::FindDocs
            .execute(&ctx, &json!({"library": "axum", "query": "x", "version": "9.9"}))
            .await
            .unwrap();
        assert!(error(&ran).contains("axum_v0_8_4"), "the error says what there is");
    }

    /// The declaration caps what one call may ask for; the cache key carries
    /// the number that was actually spent.
    #[tokio::test]
    async fn find_docs_clamps_max_tokens_under_the_declaration() {
        let app = mock_app("[server]", "find_docs_tokens");
        crate::docs::cache_put(
            &app.state,
            &crate::docs::search_key("axum"),
            &json!([{"id": "/tokio-rs/axum"}]),
        );
        crate::docs::cache_put(
            &app.state,
            &crate::docs::docs_key("/tokio-rs/axum", "extractors", 1000),
            &json!("snippets"),
        );

        let ctx = ToolCtx { params: json!({"max_tokens": 1000}), ..ToolCtx::new(&app, "call_1") };
        let ran = Tool::FindDocs
            .execute(&ctx, &json!({"library": "axum", "query": "extractors", "max_tokens": 9000}))
            .await
            .unwrap();
        assert!(ran.model_output.contains("snippets"), "9000 capped to 1000: {}", ran.model_output);
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
        let ctx = ToolCtx { now, ..ToolCtx::new(&app, "call_1") };

        let utc = Tool::Datetime.execute(&ctx, &json!({})).await.unwrap();
        assert_eq!(
            utc.model_output,
            json!({"datetime": "2025-07-15T18:30:00.000+00:00", "timezone": "UTC"}).to_string()
        );

        let ny = Tool::Datetime
            .execute(&ctx, &json!({"timezone": "America/New_York"}))
            .await
            .unwrap();
        assert_eq!(
            ny.model_output,
            json!({"datetime": "2025-07-15T14:30:00.000-04:00", "timezone": "America/New_York"})
                .to_string()
        );

        // The declaration's `timezone` is the default; the call's own wins.
        let declared = ToolCtx { params: json!({"timezone": "Asia/Dhaka"}), ..ToolCtx { now, ..ToolCtx::new(&app, "call_1") } };
        let dhaka = Tool::Datetime.execute(&declared, &json!({})).await.unwrap();
        assert!(dhaka.model_output.contains("2025-07-16T00:30:00.000+06:00"), "{}", dhaka.model_output);
        let call_wins = Tool::Datetime.execute(&declared, &json!({"timezone": "UTC"})).await.unwrap();
        assert!(call_wins.model_output.contains("+00:00"), "{}", call_wins.model_output);
    }

    /// A zone jiff cannot resolve is the model's mistake to see, not a reason
    /// to drop the call from the replay.
    #[tokio::test]
    async fn datetime_reports_an_unknown_zone() {
        let app = mock_app("[server]", "datetime_bad_zone");
        let now: jiff::Timestamp = "2025-07-15T18:30:00Z".parse().unwrap();
        let ctx = ToolCtx { now, ..ToolCtx::new(&app, "call_1") };
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
        assert!(def["function"]["description"].as_str().unwrap().contains("UTC"));
        // The declared default zone is what the model is told, so it does
        // not ask for UTC out of habit.
        let dhaka = tool_def(Tool::Datetime, &json!({"timezone": "Asia/Dhaka"}));
        assert!(dhaka["function"]["description"].as_str().unwrap().contains("Asia/Dhaka"), "{dhaka}");
        assert!(!dhaka["function"]["description"].as_str().unwrap().contains("UTC"), "{dhaka}");
    }

    /// Every spelling `wiki:tool-search` lists maps to the tool; the BM25
    /// variants map to nothing, because pxy matches by regex only and an
    /// unknown server tool is unservable rather than quietly mis-served.
    #[test]
    fn from_type_maps_every_tool_search_spelling_but_not_bm25() {
        for ty in [
            "openrouter:tool_search",
            "pxy:tool_search",
            "tool_search",
            "tool_search_tool_regex",
            "tool_search_tool_regex_20251119",
        ] {
            assert_eq!(from_type(ty), Some(Tool::ToolSearch), "{ty}");
        }
        for ty in ["tool_search_tool_bm25", "tool_search_tool_bm25_20251119"] {
            assert_eq!(from_type(ty), None, "{ty}");
        }
    }

    /// The search reads the turn's deferred definitions off the context, tells
    /// the model what it revealed, and leaves the names on the marker for the
    /// loop and on the blocks for a Messages client.
    #[tokio::test]
    async fn tool_search_reveals_matches_and_renders_the_blocks() {
        let app = mock_app("[server]", "tool_search_reveal");
        let deferred = vec![
            json!({"type": "function", "function": {"name": "get_weather", "description": "Conditions for a city."}}),
            json!({"type": "function", "function": {"name": "send_email", "description": "Deliver a note."}}),
        ];
        let ctx = ToolCtx { deferred: deferred.clone(), ..ToolCtx::new(&app, "call_1") };

        let hit = Tool::ToolSearch.execute(&ctx, &json!({"pattern": "weather"})).await.unwrap();
        let output: Value = serde_json::from_str(&hit.model_output).unwrap();
        assert_eq!(output["status"], "ok");
        assert_eq!(output["found"], json!(["get_weather"]));
        assert_eq!(output["total_matches"], 1);
        assert_eq!(hit.client.marker["found"], json!(["get_weather"]));
        assert_eq!(hit.client.blocks[0]["name"], tool_search::ANTHROPIC_NAME);
        assert_eq!(hit.client.blocks[1]["content"]["tool_references"][0]["tool_name"], "get_weather");

        // No match is an answer, not a failure: the model must stop looking.
        let miss = Tool::ToolSearch.execute(&ctx, &json!({"pattern": "kubernetes"})).await.unwrap();
        let output: Value = serde_json::from_str(&miss.model_output).unwrap();
        assert_eq!(output["found"], json!([]));
        assert_eq!(output["total_matches"], 0);

        // A pattern the engine rejects reaches the model as an error result.
        let bad = Tool::ToolSearch.execute(&ctx, &json!({"pattern": "get_("})).await.unwrap();
        let output: Value = serde_json::from_str(&bad.model_output).unwrap();
        assert_eq!(output["status"], "error");
        assert!(
            output["error"].as_str().unwrap().starts_with("invalid regular expression: "),
            "{}",
            bad.model_output
        );
        assert_eq!(bad.client.blocks[1]["content"]["error_code"], "invalid_tool_input");
        assert_eq!(bad.client.marker["found"], json!([]));

        // `limit` on the call is clamped to the declaration's `max_results`.
        let capped = ToolCtx {
            deferred,
            params: json!({"max_results": 1}),
            ..ToolCtx::new(&app, "call_2")
        };
        let both = Tool::ToolSearch.execute(&capped, &json!({"pattern": "e", "limit": 9})).await.unwrap();
        let output: Value = serde_json::from_str(&both.model_output).unwrap();
        assert_eq!(output["found"], json!(["get_weather"]));
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

    /// A model asserted `vision = false` 400s on an image part, so each one
    /// becomes the text `[image N]` — numbered across the whole transcript in
    /// order, so the number the model reads is the index describe_image takes.
    #[test]
    fn swap_image_parts_numbers_every_image_across_the_transcript() {
        let mut body = json!({"messages": [
            {"role": "user", "content": [
                {"type": "text", "text": "compare these"},
                {"type": "image_url", "image_url": {"url": "https://e.test/a.png"}},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,aGk="}},
            ]},
            {"role": "assistant", "content": "the first is darker"},
            {"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": "https://e.test/c.png"}},
            ]},
        ]});
        let urls = swap_image_parts(&mut body);
        assert_eq!(
            urls,
            vec!["https://e.test/a.png", "data:image/png;base64,aGk=", "https://e.test/c.png"]
        );
        assert_eq!(
            body["messages"][0]["content"],
            json!([
                {"type": "text", "text": "compare these"},
                {"type": "text", "text": "[image 1]"},
                {"type": "text", "text": "[image 2]"},
            ])
        );
        assert_eq!(body["messages"][1]["content"], "the first is darker");
        assert_eq!(
            body["messages"][2]["content"],
            json!([{"type": "text", "text": "[image 3]"}])
        );

        // Nothing to swap: a body of plain text is returned untouched.
        let mut text = json!({"messages": [{"role": "user", "content": "hello"}]});
        assert!(swap_image_parts(&mut text).is_empty());
        assert_eq!(text["messages"][0]["content"], "hello");
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
        let out: Value = serde_json::from_str(&ran.model_output).unwrap();
        assert_eq!(out["content"], "hello from the page", "{out}");
        assert_eq!(out["status"], "completed", "{out}");
        assert_eq!(out["url"], "https://example.com/article", "{out}");
        assert!(out["retrieved_at"].as_str().is_some_and(|t| t.contains('T')), "{out}");
        assert_eq!(ran.client.marker["url"], "https://example.com/article");
        assert!(ran.client.blocks.is_empty(), "web_fetch has no Anthropic block");

        // The declaration's parameters: a content cap in tokens (four
        // characters each), domain lists checked before any fetch, and an
        // engine that must name a configured provider.
        let fetch = |params: Value| {
            let app = app.clone();
            async move {
            let ctx = ToolCtx { params, ..ToolCtx::new(&app, "call_2") };
            let ran = Tool::WebFetch
                .execute(&ctx, &json!({"url": "https://docs.example.com/page"}))
                .await
                .unwrap();
            serde_json::from_str::<Value>(&ran.model_output).unwrap()
            }
        };
        let capped = fetch(json!({"max_content_tokens": 1})).await;
        assert_eq!(capped["content"], "hell\n\n[truncated]", "{capped}");
        let blocked = fetch(json!({"blocked_domains": ["example.com"]})).await;
        assert_eq!(blocked["status"], "failed", "{blocked}");
        assert!(blocked["error"].as_str().unwrap().contains("domain"), "{blocked}");
        let not_allowed = fetch(json!({"allowed_domains": ["other.org"]})).await;
        assert_eq!(not_allowed["status"], "failed", "{not_allowed}");
        let allowed = fetch(json!({"allowed_domains": ["example.com"]})).await;
        assert_eq!(allowed["status"], "completed", "a subdomain of an allowed domain: {allowed}");
        let engine = fetch(json!({"engine": "nope"})).await;
        assert!(engine["error"].as_str().unwrap().contains("unknown engine 'nope'; configured: mock"), "{engine}");
        let named = fetch(json!({"engine": "mock"})).await;
        assert_eq!(named["status"], "completed", "{named}");
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
        let out: Value =
            serde_json::from_str(&fetched_for_model("https://e/x", &long, FETCH_CONTENT_CHARS, "t")).unwrap();
        let content = out["content"].as_str().unwrap();
        assert!(content.ends_with("[truncated]"), "the model must be told");
        assert!(content.contains(&"x".repeat(FETCH_CONTENT_CHARS)));
        assert!(!content.contains(&"x".repeat(FETCH_CONTENT_CHARS + 1)));
    }

    /// A listed domain covers itself and its subdomains; `blocked` wins over
    /// `allowed`; an empty `allowed` allows everything.
    #[test]
    fn domain_lists_match_hosts_and_subdomains() {
        let allowed = vec!["example.com".to_string()];
        let blocked = vec!["private.example.com".to_string()];
        assert!(domain_allowed("https://example.com/x", &allowed, &blocked));
        assert!(domain_allowed("https://docs.example.com:8443/x?y#z", &allowed, &blocked));
        assert!(!domain_allowed("https://notexample.com/x", &allowed, &blocked));
        assert!(!domain_allowed("https://private.example.com/x", &allowed, &blocked));
        assert!(!domain_allowed("https://deep.private.example.com/x", &allowed, &blocked));
        assert!(domain_allowed("https://anything.org/", &[], &blocked));
        assert!(domain_allowed("https://user:pw@Example.COM/", &allowed, &[]));
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

        let search = |args: Value, params: Value| {
            let app = app.clone();
            async move {
            let ctx = ToolCtx { params, ..ToolCtx::new(&app, "c") };
            let ran = Tool::SearchModels.execute(&ctx, &args).await.unwrap();
            let out: Value = serde_json::from_str(&ran.model_output).unwrap();
            let ids: Vec<String> = out["models"]
                .as_array()
                .unwrap()
                .iter()
                .map(|m| m["id"].as_str().unwrap().to_string())
                .collect();
            (out, ids)
            }
        };
        let (out, ids) =
            search(json!({"query": "big", "min_context_length": 100000}), json!({})).await;
        assert_eq!(ids, vec!["alpha/big"], "{out}");
        assert_eq!(out["total_results"], 1, "{out}");
        assert_eq!(out["showing"], 1, "{out}");
        assert_eq!(out["models"][0]["groups"], json!(["aaa"]), "the group alias: {out}");

        // A capability filter reads the catalog's asserted metadata.
        let (out, ids) = search(json!({"tool_call": true}), json!({})).await;
        assert_eq!(ids, vec!["alpha/big"], "{out}");

        // `free: false` excludes only asserted-free models; an unasserted
        // capability is not a no, so the two unknowns match.
        let (out, ids) = search(json!({"free": false}), json!({})).await;
        assert_eq!(ids, vec!["alpha/big", "alpha/small"], "{out}");
        assert_eq!(out["total_results"], 2, "{out}");

        let (_, ids) = search(json!({"free": true}), json!({})).await;
        assert_eq!(ids, vec!["beta/tiny"]);

        // No match is a report, not an unserved call.
        let (out, ids) = search(json!({"query": "nope"}), json!({})).await;
        assert!(ids.is_empty(), "{out}");
        assert_eq!(out["total_results"], 0, "{out}");

        // The declaration's `max_results` caps what is shown, not what matched,
        // and the call's `offset` pages through the rest.
        let (out, ids) = search(json!({}), json!({"max_results": 1})).await;
        assert_eq!(ids.len(), 1, "{out}");
        assert_eq!(out["total_results"], 3, "{out}");
        assert_eq!(out["showing"], 1, "{out}");
        let (_, first) = search(json!({}), json!({"max_results": 2})).await;
        let (out, paged) = search(json!({"offset": 2}), json!({"max_results": 2})).await;
        assert_eq!(paged.len(), 1, "{out}");
        assert!(!first.contains(&paged[0]), "{first:?} vs {paged:?}");
        let (out, none) = search(json!({"offset": 99}), json!({})).await;
        assert!(none.is_empty(), "{out}");
    }

    /// image_generation runs the image media walk and hands the model the URL
    /// the provider answered with; the generation is a media cost, not a chat
    /// one.
    #[tokio::test]
    async fn image_generation_returns_the_image_url_from_a_mock_provider() {
        use axum::routing::post;
        let seen: std::sync::Arc<std::sync::Mutex<Vec<Value>>> = Default::default();
        let sink = seen.clone();
        let router = axum::Router::new().route(
            "/img",
            post(move |axum::Json(body): axum::Json<Value>| {
                let sink = sink.clone();
                async move {
                    sink.lock().unwrap().push(body);
                    axum::Json(json!({"created": 1, "data": [{"url": "https://x/y.png"}]}))
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
        let ctx = ToolCtx {
            params: json!({"model": "mock/m", "quality": "high", "size": "1024x1024", "max_uses": 3}),
            ..ToolCtx::new(&app, "call_1")
        };
        let ran = Tool::ImageGeneration
            .execute(&ctx, &json!({"prompt": "a cat"}))
            .await
            .expect("a configured image chain serves the call");
        let out: Value = serde_json::from_str(&ran.model_output).unwrap();
        assert_eq!(out["status"], "ok", "{out}");
        assert_eq!(out["imageUrl"], "https://x/y.png", "{out}");
        assert!(ran.client.blocks.is_empty(), "no Anthropic block for image_generation");
        // Every parameter but the tool's own rides the generation body.
        let body = seen.lock().unwrap()[0].clone();
        assert_eq!(body["prompt"], "a cat", "{body}");
        assert_eq!(body["quality"], "high", "{body}");
        assert_eq!(body["size"], "1024x1024", "{body}");
        assert!(body.get("max_uses").is_none(), "{body}");
        assert_eq!(body["model"], "m", "the declaration's model, as the chain spells it: {body}");
    }

    /// web_search's parameters: `max_results` is what the provider is asked
    /// for, `excluded_domains` / `allowed_domains` filter the hits,
    /// `max_characters` cuts each snippet, `max_total_results` ends searching
    /// for the turn, and `engine` must name a configured provider.
    #[tokio::test]
    async fn web_search_honours_its_parameters() {
        use axum::routing::post;
        let seen: std::sync::Arc<std::sync::Mutex<Vec<Value>>> = Default::default();
        let sink = seen.clone();
        let router = axum::Router::new().route(
            "/s",
            post(move |axum::Json(body): axum::Json<Value>| {
                let sink = sink.clone();
                async move {
                    sink.lock().unwrap().push(body);
                    axum::Json(json!({"data": [
                        {"title": "A", "url": "https://a.example.com/1", "description": "alpha beta gamma"},
                        {"title": "B", "url": "https://b.other.org/2", "description": "bravo"},
                    ]}))
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
                [[search.providers]]
                name = "mock"
                kind = "jina"
                api_key = "k"
                base_url = "http://{addr}/s"
                "#
            ),
            "web_search_params",
        );
        let ctx = ToolCtx {
            params: json!({"max_results": 2, "excluded_domains": ["other.org"], "max_characters": 5}),
            ..ToolCtx::new(&app, "call_1")
        };
        let ran = Tool::WebSearch.execute(&ctx, &json!({"query": "q"})).await.unwrap();
        assert_eq!(seen.lock().unwrap()[0]["num"], 2, "max_results is what the provider is asked for");
        assert!(ran.model_output.contains("https://a.example.com/1"), "{}", ran.model_output);
        assert!(!ran.model_output.contains("other.org"), "excluded: {}", ran.model_output);
        assert!(ran.model_output.contains("alpha"), "{}", ran.model_output);
        assert!(!ran.model_output.contains("alpha beta"), "cut at 5 characters: {}", ran.model_output);
        assert_eq!(ran.client.marker["hits"], 1);

        let spent = ToolCtx {
            params: json!({"max_total_results": 3}),
            results_used: 3,
            ..ToolCtx::new(&app, "call_2")
        };
        let ran = Tool::WebSearch.execute(&spent, &json!({"query": "q"})).await.unwrap();
        assert!(ran.model_output.contains("max_total_results (3)"), "{}", ran.model_output);
        assert!(ran.client.marker.is_null(), "nothing ran");
        assert_eq!(seen.lock().unwrap().len(), 1, "no provider call past the total");

        let engine = ToolCtx { params: json!({"engine": "nope"}), ..ToolCtx::new(&app, "call_3") };
        let ran = Tool::WebSearch.execute(&engine, &json!({"query": "q"})).await.unwrap();
        assert!(ran.model_output.contains("unknown engine 'nope'; configured: mock"), "{}", ran.model_output);
        let named = ToolCtx { params: json!({"engine": "mock", "max_total_results": 3, "max_results": 5}), results_used: 2, ..ToolCtx::new(&app, "call_4") };
        let ran = Tool::WebSearch.execute(&named, &json!({"query": "q"})).await.unwrap();
        assert_eq!(seen.lock().unwrap()[1]["num"], 1, "only the remaining total is asked for");
        assert_eq!(ran.client.marker["hits"], 2);
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
            &ClientContext { agent: Some("pi".into()), ..ClientContext::default() },
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
            params: json!({
                "model": "worker/w",
                "tools": [{"type": "openrouter:web_search"}],
            }),
            ..ToolCtx::new(&app, "call_1")
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
            run_panel(&app, &models, "compare answers", &json!({}), &ClientContext::default()),
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

    /// The analyst is shown the question and every panel answer, labelled by
    /// the model that gave it, and its JSON reply comes back parsed.
    #[tokio::test]
    async fn analyst_receives_every_answer_and_parses_its_json() {
        use std::sync::{Arc, Mutex};
        use axum::routing::post;
        let seen: Arc<Mutex<Vec<Value>>> = Arc::new(Mutex::new(Vec::new()));
        let sink = seen.clone();
        let router = axum::Router::new().route(
            "/c",
            post(move |axum::Json(body): axum::Json<Value>| {
                let sink = sink.clone();
                async move {
                    sink.lock().unwrap().push(body.clone());
                    let content = if body["model"] == "an" {
                        r#"{"summary":"they agree","differences":[]}"#.to_string()
                    } else {
                        format!("answer from {}", body["model"].as_str().unwrap_or("?"))
                    };
                    axum::Json(json!({"id": "x", "choices": [{"index": 0,
                        "message": {"role": "assistant", "content": content},
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
                models = ["a", "b", "c", "an"]
                "#
            ),
            "analyst",
        );
        let models = vec!["p/a".to_string(), "p/b".to_string(), "p/c".to_string()];
        let answers =
            run_panel(&app, &models, "which is best?", &json!({}), &ClientContext::default()).await;
        let analysis =
            run_analyst(&app, "p/an", "which is best?", &answers, &json!({}), &ClientContext::default())
                .await
                .unwrap();
        assert_eq!(analysis["summary"], "they agree");

        let bodies = seen.lock().unwrap();
        let ask = bodies.iter().find(|b| b["model"] == "an").expect("the analyst ran");
        let text = ask["messages"]
            .as_array()
            .unwrap()
            .iter()
            .map(|m| m["content"].as_str().unwrap_or(""))
            .collect::<Vec<_>>()
            .join("\n");
        for answer in ["answer from a", "answer from b", "answer from c", "p/a", "p/b", "p/c"] {
            assert!(text.contains(answer), "the analyst must see {answer}: {text}");
        }
        assert!(text.contains("which is best?"), "the question rides along: {text}");
    }

    /// A model that rejects `temperature` outright still serves as the
    /// analyst: the leg is retried once without it.
    #[tokio::test]
    async fn analyst_retries_without_temperature_when_the_model_refuses_it() {
        use axum::routing::post;
        let router = axum::Router::new().route(
            "/c",
            post(|axum::Json(body): axum::Json<Value>| async move {
                if !body["temperature"].is_null() {
                    return (
                        axum::http::StatusCode::BAD_REQUEST,
                        axum::Json(json!({"error": {"message": "Unsupported parameter: 'temperature' is not supported with this model."}})),
                    );
                }
                (axum::http::StatusCode::OK, axum::Json(json!({"id": "x", "choices": [{"index": 0,
                    "message": {"role": "assistant", "content": "{\"consensus\":[]}"},
                    "finish_reason": "stop"}]})))
            }),
        );
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move { axum::serve(listener, router).await.unwrap() });
        let app = mock_app(
            &format!("[server]\n[providers.p]\nbase_url = \"http://{addr}/c\"\nmodels = [\"an\"]\n"),
            "analyst_no_temperature",
        );
        let answers = vec![("p/a".to_string(), Ok("hi".to_string()))];
        let analysis = run_analyst(&app, "p/an", "q", &answers, &json!({}), &ClientContext::default())
            .await
            .unwrap();
        assert_eq!(analysis["consensus"], json!([]));
    }

    /// An analyst that answers prose instead of JSON is a failure, so the
    /// caller can fall back to the raw panel answers.
    #[tokio::test]
    async fn analyst_non_json_is_a_failure() {
        use axum::routing::post;
        let router = axum::Router::new().route(
            "/c",
            post(|| async {
                axum::Json(json!({"id": "x", "choices": [{"index": 0,
                    "message": {"role": "assistant", "content": "not json at all"},
                    "finish_reason": "stop"}]}))
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
                models = ["an"]
                "#
            ),
            "analyst_non_json",
        );
        let answers = vec![("p/a".to_string(), Ok("hi".to_string()))];
        let err = run_analyst(&app, "p/an", "q", &answers, &json!({}), &ClientContext::default())
            .await
            .unwrap_err();
        assert!(err.contains("non-JSON"), "{err}");
    }

    /// Fusion runs the configured panel and analyst, and an analyst that
    /// fails does not fail the call: the raw panel answers come back with no
    /// analysis field, so the outer model can still answer.
    #[tokio::test]
    async fn fusion_returns_the_panel_when_the_analyst_fails() {
        use axum::routing::post;
        let router = axum::Router::new().route(
            "/c",
            post(|axum::Json(body): axum::Json<Value>| async move {
                let content = if body["model"] == "an" {
                    "prose, not json".to_string()
                } else {
                    format!("answer from {}", body["model"].as_str().unwrap_or("?"))
                };
                axum::Json(json!({"id": "x", "choices": [{"index": 0,
                    "message": {"role": "assistant", "content": content},
                    "finish_reason": "stop"}]}))
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
                models = ["a", "b", "an"]
                [server_tools]
                fusion_panel = ["p/a", "p/b"]
                fusion_analyst = "p/an"
                "#
            ),
            "fusion_analyst_failure",
        );
        let ctx = ToolCtx::new(&app, "call_1");
        let ran = Tool::Fusion
            .execute(&ctx, &json!({"prompt": "which is best?"}))
            .await
            .unwrap();
        let out: Value = serde_json::from_str(&ran.model_output).unwrap();
        assert_eq!(out["status"], "ok", "{out}");
        let panel = out["responses"].as_array().expect("the panel answers ride along");
        assert_eq!(panel.len(), 2, "{out}");
        assert!(panel.iter().any(|a| a["content"] == "answer from a"), "{out}");
        assert!(panel.iter().any(|a| a["content"] == "answer from b"), "{out}");
        assert!(out.get("analysis").is_none(), "a failed analyst adds no analysis: {out}");
        assert!(out.get("failed_models").is_none(), "nobody failed: {out}");
    }

    /// The declaration's `analysis_models` and `model` win over the
    /// configured panel and analyst; every leg is offered web_search and
    /// web_fetch; the analyst runs at temperature 0 and its JSON object is
    /// the `analysis`; a member that fails is listed beside the answers.
    #[tokio::test]
    async fn fusion_honours_the_declared_panel_and_analyst() {
        use axum::routing::post;
        let seen: std::sync::Arc<std::sync::Mutex<Vec<Value>>> = Default::default();
        let sink = seen.clone();
        let router = axum::Router::new().route(
            "/c",
            post(move |axum::Json(body): axum::Json<Value>| {
                let sink = sink.clone();
                async move {
                    sink.lock().unwrap().push(body.clone());
                    let content = if body["model"] == "an" {
                        json!({"consensus": ["x"], "contradictions": [], "partial_coverage": [],
                               "unique_insights": [], "blind_spots": []}).to_string()
                    } else {
                        format!("answer from {}", body["model"].as_str().unwrap_or("?"))
                    };
                    axum::Json(json!({"id": "x", "choices": [{"index": 0,
                        "message": {"role": "assistant", "content": content},
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
                models = ["a", "x", "an"]
                [server_tools]
                fusion_panel = ["p/a"]
                fusion_analyst = "p/a"
                "#
            ),
            "fusion_declared",
        );
        // Every leg is offered web_search and web_fetch unless the
        // declaration lists tools itself (none are servable in this app, so
        // the router strips them before the mock sees the body).
        let legs = leg_params(&json!({}), &[Tool::WebSearch, Tool::WebFetch]);
        assert_eq!(legs["tools"], json!([{"type": "pxy:web_search"}, {"type": "pxy:web_fetch"}]));
        let listed = leg_params(&json!({"tools": []}), &[Tool::WebSearch, Tool::WebFetch]);
        assert!(listed.get("tools").is_none(), "an explicit empty list means none: {listed}");
        let ctx = ToolCtx {
            params: json!({
                "analysis_models": ["p/x", "missing/y"],
                "model": "p/an",
                "temperature": 0.7,
            }),
            ..ToolCtx::new(&app, "call_1")
        };
        let ran = Tool::Fusion.execute(&ctx, &json!({"prompt": "why?"})).await.unwrap();
        let out: Value = serde_json::from_str(&ran.model_output).unwrap();
        assert_eq!(out["status"], "ok", "{out}");
        assert_eq!(out["responses"], json!([{"model": "p/x", "content": "answer from x"}]), "{out}");
        assert_eq!(out["failed_models"][0]["model"], "missing/y", "{out}");
        assert_eq!(out["analysis"]["consensus"], json!(["x"]), "{out}");

        let bodies = seen.lock().unwrap();
        let leg = bodies.iter().find(|b| b["model"] == "x").expect("the declared panel member ran");
        assert_eq!(leg["temperature"], 0.7, "{leg}");
        assert_eq!(leg["max_tokens"].as_u64().or(leg["max_completion_tokens"].as_u64()), Some(16_000), "{leg}");
        let analyst = bodies.iter().find(|b| b["model"] == "an").expect("the declared analyst ran");
        assert_eq!(analyst["temperature"], 0, "{analyst}");
        assert!(bodies.iter().all(|b| b["model"] != "a"), "the configured panel is overridden");
    }

    /// An all-failed panel is a tool error the outer model reads, not an `ok`
    /// with an empty analysis: every member's failure rides back labelled and
    /// the status tells the dialect layer to render an error.
    #[tokio::test]
    async fn fusion_reports_error_when_every_panel_member_fails() {
        let app = mock_app(
            r#"
            [server]
            [providers.p]
            base_url = "http://127.0.0.1:1/c"
            models = ["m"]
            [server_tools]
            fusion_panel = ["missing/one", "missing/two"]
            "#,
            "fusion_all_fail",
        );
        let ctx = ToolCtx::new(&app, "call_1");
        let ran = Tool::Fusion
            .execute(&ctx, &json!({"prompt": "which is best?"}))
            .await
            .unwrap();
        let out: Value = serde_json::from_str(&ran.model_output).unwrap();
        assert_eq!(out["status"], "error", "{out}");
        let panel = out["failed_models"].as_array().expect("every failure rides along");
        assert_eq!(panel.len(), 2, "{out}");
        assert!(panel.iter().all(|a| a["error"].is_string()), "{out}");
        assert_eq!(out["error"], "all panel models failed", "{out}");
        assert_eq!(out["failure_reason"], "all_panels_failed", "{out}");
        assert_eq!(ran.client.marker["status"], "error", "the dialect layer reads the marker");
    }

    /// A panel whose every failure was a rate limit or a credit exhaustion
    /// says so with OpenRouter's typed reason.
    #[test]
    fn fusion_failure_reason_is_typed() {
        let rl = "sub-request failed (429): slow down".to_string();
        let credit = "sub-request failed (402): insufficient credits".to_string();
        let other = "sub-request failed (500): boom".to_string();
        assert_eq!(fusion_failure_reason(&[&other]), "all_panels_failed");
        assert_eq!(fusion_failure_reason(&[&other, &rl]), "rate_limited");
        assert_eq!(fusion_failure_reason(&[&rl, &credit]), "insufficient_credits");
    }

    /// The advisor consults the client's own model when neither the
    /// declaration nor the call names one, and forwards the declaration's
    /// sampling knobs to that leg.
    #[tokio::test]
    async fn advisor_falls_back_to_the_outer_model_and_forwards_knobs() {
        use axum::routing::post;
        let seen: std::sync::Arc<std::sync::Mutex<Vec<Value>>> = Default::default();
        let sink = seen.clone();
        let router = axum::Router::new().route(
            "/c",
            post(move |axum::Json(body): axum::Json<Value>| {
                let sink = sink.clone();
                async move {
                    sink.lock().unwrap().push(body);
                    axum::Json(json!({"id": "x", "choices": [{"index": 0,
                        "message": {"role": "assistant", "content": "advice"},
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
                models = ["m"]
                "#
            ),
            "advisor_outer",
        );
        let ctx = ToolCtx {
            outer_model: "p/m".into(),
            params: json!({"temperature": 0.3, "max_completion_tokens": 99, "reasoning": {"effort": "low"}}),
            ..ToolCtx::new(&app, "call_1")
        };
        let ran = Tool::Advisor.execute(&ctx, &json!({"prompt": "how?"})).await.unwrap();
        let out: Value = serde_json::from_str(&ran.model_output).unwrap();
        assert_eq!(out["status"], "ok", "{out}");
        assert_eq!(out["model"], "p/m", "{out}");
        let body = seen.lock().unwrap()[0].clone();
        assert_eq!(body["model"], "m", "{body}");
        assert_eq!(body["temperature"], 0.3, "{body}");
        assert_eq!(body["max_tokens"].as_u64().or(body["max_completion_tokens"].as_u64()), Some(99), "{body}");
        assert_eq!(body["reasoning"]["effort"], "low", "{body}");

        // No model anywhere is a call pxy cannot serve.
        let bare = ToolCtx::new(&app, "call_2");
        assert!(Tool::Advisor.execute(&bare, &json!({"prompt": "how?"})).await.is_err());

        // The subagent falls back the same way.
        let worker = ToolCtx { outer_model: "p/m".into(), ..ToolCtx::new(&app, "call_3") };
        let ran = Tool::Subagent
            .execute(&worker, &json!({"task_name": "t", "task_description": "do it"}))
            .await
            .unwrap();
        let out: Value = serde_json::from_str(&ran.model_output).unwrap();
        assert_eq!(out["model"], "p/m", "{out}");
    }

    /// describe_image is the eye of a model that has none: the `[image N]` the
    /// transcript carries resolves to the URL swapped out of it, and the leg is
    /// one user message, the prompt then the image, because DeepSeek 400s on an
    /// image anywhere else. `ocr` mode swaps both the model and the prompt.
    #[tokio::test]
    async fn describe_image_resolves_an_index_and_sends_one_user_message() {
        use axum::routing::post;
        let seen: std::sync::Arc<std::sync::Mutex<Vec<Value>>> = Default::default();
        let sink = seen.clone();
        let router = axum::Router::new().route(
            "/c",
            post(move |axum::Json(body): axum::Json<Value>| {
                let sink = sink.clone();
                async move {
                    sink.lock().unwrap().push(body);
                    axum::Json(json!({"id": "x", "choices": [{"index": 0,
                        "message": {"role": "assistant", "content": "a red barn"},
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
                models = ["sees", "reads"]
                "#
            ),
            "describe_image",
        );
        let ctx = ToolCtx {
            params: json!({"model": "p/sees", "ocr_model": "p/reads"}),
            images: vec!["https://e.test/a.png".into(), "data:image/png;base64,aGk=".into()],
            ..ToolCtx::new(&app, "call_1")
        };

        let ran = Tool::DescribeImage.execute(&ctx, &json!({"image": "2"})).await.unwrap();
        let out: Value = serde_json::from_str(&ran.model_output).unwrap();
        assert_eq!(out["status"], "ok", "{out}");
        assert_eq!(out["mode"], "describe", "{out}");
        assert_eq!(out["model"], "p/sees", "{out}");
        assert_eq!(out["description"], "a red barn", "{out}");
        // No Anthropic block for this tool: the round is served silently.
        assert!(ran.client.blocks.is_empty());
        assert_eq!(ran.client.marker["mode"], "describe");
        let body = seen.lock().unwrap()[0].clone();
        assert_eq!(body["model"], "sees", "{body}");
        assert_eq!(
            body["messages"],
            json!([{"role": "user", "content": [
                {"type": "text", "text": "Describe this image in detail, transcribing all visible text."},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,aGk="}},
            ]}]),
            "{body}"
        );

        // `ocr` reads the document instead: its own model, its own prompt, and
        // a URL passed straight through rather than an index.
        let ran = Tool::DescribeImage
            .execute(&ctx, &json!({"image": "https://e.test/scan.png", "mode": "ocr"}))
            .await
            .unwrap();
        let out: Value = serde_json::from_str(&ran.model_output).unwrap();
        assert_eq!(out["model"], "p/reads", "{out}");
        assert_eq!(out["mode"], "ocr", "{out}");
        let body = seen.lock().unwrap()[1].clone();
        assert_eq!(body["model"], "reads", "{body}");
        assert_eq!(
            body["messages"][0]["content"][0]["text"],
            "Transcribe this document into clean Markdown in natural reading order.",
            "{body}"
        );
        assert_eq!(
            body["messages"][0]["content"][1]["image_url"]["url"],
            "https://e.test/scan.png",
            "{body}"
        );

        // The caller's own question replaces the default prompt.
        let _ = Tool::DescribeImage
            .execute(&ctx, &json!({"image": "1", "prompt": "what colour is the door?"}))
            .await
            .unwrap();
        let body = seen.lock().unwrap()[2].clone();
        assert_eq!(body["messages"][0]["content"][0]["text"], "what colour is the door?");
        assert_eq!(
            body["messages"][0]["content"][1]["image_url"]["url"],
            "https://e.test/a.png",
            "{body}"
        );
    }

    /// Calls describe_image cannot serve at all: the loop answers each with the
    /// error result the model reads, and nothing reaches a vision model.
    #[tokio::test]
    async fn describe_image_refuses_an_unusable_call() {
        let app = mock_app(
            r#"
            [server]
            [providers.p]
            base_url = "http://127.0.0.1:1/c"
            models = ["sees"]
            "#,
            "describe_image_unusable",
        );
        let ctx = ToolCtx {
            params: json!({"model": "p/sees"}),
            images: vec!["https://e.test/a.png".into()],
            ..ToolCtx::new(&app, "call_1")
        };
        for args in [
            json!({}),                                  // no image
            json!({"image": "2"}),                      // no such placeholder
            json!({"image": "ftp://e.test/a.png"}),     // not a URL pxy will send
            json!({"image": "1", "mode": "haiku"}),     // no such mode
        ] {
            assert!(
                Tool::DescribeImage.execute(&ctx, &args).await.is_err(),
                "{args} must not be served"
            );
        }
        // A declaration that pins no model cannot be served either: the tool
        // never falls back to the model that called it, which is the one that
        // could not see the image in the first place.
        let modelless = ToolCtx {
            outer_model: "p/sees".into(),
            images: vec!["https://e.test/a.png".into()],
            ..ToolCtx::new(&app, "call_2")
        };
        assert!(Tool::DescribeImage.execute(&modelless, &json!({"image": "1"})).await.is_err());
    }

    /// `forward_transcript` hands the advisor the turn's conversation as
    /// prose — tool calls and results included, as bracketed text — and makes
    /// the prompt optional.
    #[tokio::test]
    async fn advisor_forwards_the_transcript_as_prose() {
        use axum::routing::post;
        let seen: std::sync::Arc<std::sync::Mutex<Vec<Value>>> = Default::default();
        let sink = seen.clone();
        let router = axum::Router::new().route(
            "/c",
            post(move |axum::Json(body): axum::Json<Value>| {
                let sink = sink.clone();
                async move {
                    sink.lock().unwrap().push(body);
                    axum::Json(json!({"id": "x", "choices": [{"index": 0,
                        "message": {"role": "assistant", "content": "advice"},
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
                models = ["m"]
                "#
            ),
            "advisor_transcript",
        );
        let ctx = ToolCtx {
            params: json!({"model": "p/m", "instructions": "be terse", "forward_transcript": true}),
            transcript: vec![
                json!({"role": "system", "content": "you are helpful"}),
                json!({"role": "user", "content": [{"type": "text", "text": "build a pool"}]}),
                json!({"role": "assistant", "content": null, "tool_calls": [
                    {"id": "c1", "type": "function", "function": {"name": "pxy_datetime", "arguments": "{}"}}]}),
                json!({"role": "tool", "tool_call_id": "c1", "content": "noon"}),
            ],
            ..ToolCtx::new(&app, "call_1")
        };
        let ran = Tool::Advisor.execute(&ctx, &json!({})).await.expect("no prompt needed");
        assert!(ran.model_output.contains("advice"));
        let body = seen.lock().unwrap()[0].clone();
        let msgs = body["messages"].as_array().unwrap();
        assert_eq!(msgs[0]["content"], "be terse", "{body}");
        assert_eq!(msgs[1]["content"], "you are helpful", "{body}");
        assert_eq!(msgs[2]["content"], "build a pool", "{body}");
        assert_eq!(msgs[3]["role"], "assistant", "{body}");
        assert!(msgs[3]["content"].as_str().unwrap().contains("[tool call pxy_datetime {}]"), "{body}");
        assert_eq!(msgs[4]["role"], "user", "{body}");
        assert!(msgs[4]["content"].as_str().unwrap().contains("[tool result c1] noon"), "{body}");
        assert!(body.get("tools").is_none(), "prose needs no tools: {body}");
    }

    /// A minimal app for the executor tests, mirroring `router`'s test app.
    /// With re-ranking on the provider is over-fetched, Jev scores every hit,
    /// the weak ones go and the rest come back best first, cut to the
    /// declaration's `max_results`.
    #[tokio::test]
    async fn rerank_over_fetches_then_scores_sorts_and_cuts() {
        let searched: Sink = Default::default();
        let asked: Sink = Default::default();
        // r1 and r3 are the good ones; r0 and r2 fall under rerank_min.
        const SCORES: [f64; 10] = [0.1, 0.95, 0.2, 0.9, 0.5, 0.4, 0.4, 0.4, 0.4, 0.4];
        let addr = search_and_jev(searched.clone(), asked.clone(), 10, &SCORES).await;
        let app = mock_app(&rerank_cfg(&addr, "rerank_web_search = true"), "rerank_cut");
        let ctx = ToolCtx { params: json!({"max_results": 3}), ..ToolCtx::new(&app, "call_1") };

        let ran = Tool::WebSearch.execute(&ctx, &json!({"query": "q"})).await.unwrap();

        assert_eq!(
            searched.lock().unwrap()[0]["num"], 10,
            "max(3 x 3, 10) hits to choose between"
        );
        let jev = asked.lock().unwrap()[0].clone();
        assert_eq!(jev["state"]["query"], "q");
        assert_eq!(jev["state"]["results"][2]["result"], 2, "the hits are numbered for the questions");
        assert_eq!(jev["state"]["results"][2]["url"], "https://h2.example/p");
        assert_eq!(jev["questions"].as_object().unwrap().len(), 10, "one noul per hit");
        assert_eq!(jev["questions"]["r2"]["type"], "noul");
        assert_eq!(jev["questions"]["r2"]["instructions"], "Is result 2 relevant to the query?");

        // The model reads "[1] <title>", then the url, then the snippet.
        let titles: Vec<&str> = ran.model_output.lines().filter(|l| l.starts_with('[')).collect();
        assert_eq!(ran.client.marker["hits"], 3, "cut to max_results: {}", ran.model_output);
        assert!(!ran.model_output.contains("h0"), "under rerank_min: {}", ran.model_output);
        assert!(!ran.model_output.contains("h2"), "under rerank_min: {}", ran.model_output);
        let order: Vec<usize> = ["h1", "h3", "h4"]
            .iter()
            .map(|t| titles.iter().position(|l| l.contains(t)).expect(t))
            .collect();
        assert_eq!(order, [0, 1, 2], "best first: {}", ran.model_output);
    }

    /// The over-fetch stops at 25: no declaration can make one search cost a
    /// hundred snippets.
    #[tokio::test]
    async fn rerank_caps_the_over_fetch_at_twenty_five() {
        let searched: Sink = Default::default();
        let asked: Sink = Default::default();
        let addr = search_and_jev(searched.clone(), asked.clone(), 1, &[0.9]).await;
        let app = mock_app(&rerank_cfg(&addr, "rerank_web_search = true"), "rerank_cap");
        let ctx = ToolCtx { params: json!({"max_results": 25}), ..ToolCtx::new(&app, "call_1") };

        Tool::WebSearch.execute(&ctx, &json!({"query": "q"})).await.unwrap();
        assert_eq!(searched.lock().unwrap()[0]["num"], 25, "3 x 25 is capped");
    }

    /// Jev down is not search down: the provider's own list and order stand,
    /// still cut to what the declaration asked for.
    #[tokio::test]
    async fn rerank_keeps_the_providers_order_when_jev_fails() {
        let searched: Sink = Default::default();
        let asked: Sink = Default::default();
        let addr = search_and_jev(searched.clone(), asked.clone(), 10, &[]).await;
        // A chain pointing nowhere: `ask` fails on every candidate.
        let cfg = format!(
            r#"
            [server]
            [[search.providers]]
            name = "mock"
            kind = "jina"
            api_key = "k"
            base_url = "http://{addr}/s"
            [server_tools.jev]
            rerank_web_search = true
            "#
        );
        let app = mock_app(&cfg, "rerank_jev_down");
        let ctx = ToolCtx { params: json!({"max_results": 2}), ..ToolCtx::new(&app, "call_1") };

        let ran = Tool::WebSearch.execute(&ctx, &json!({"query": "q"})).await.unwrap();

        assert_eq!(searched.lock().unwrap()[0]["num"], 10, "the over-fetch still happened");
        assert_eq!(ran.client.marker["hits"], 2, "truncated to max_results");
        assert!(ran.model_output.contains("h0"), "the provider's own first: {}", ran.model_output);
        assert!(ran.model_output.contains("h1"), "{}", ran.model_output);
        assert!(!ran.model_output.contains("h2"), "{}", ran.model_output);
    }

    /// The flag off is the search pxy ran before Jev existed: `max_results`
    /// hits asked for, and nothing else consulted.
    #[tokio::test]
    async fn rerank_off_asks_the_provider_for_exactly_max_results() {
        let searched: Sink = Default::default();
        let asked: Sink = Default::default();
        let addr = search_and_jev(searched.clone(), asked.clone(), 3, &[0.9, 0.9, 0.9]).await;
        let app = mock_app(&rerank_cfg(&addr, ""), "rerank_off");
        let ctx = ToolCtx { params: json!({"max_results": 3}), ..ToolCtx::new(&app, "call_1") };

        let ran = Tool::WebSearch.execute(&ctx, &json!({"query": "q"})).await.unwrap();

        assert_eq!(searched.lock().unwrap()[0]["num"], 3, "no over-fetch");
        assert!(asked.lock().unwrap().is_empty(), "Jev is never called");
        assert_eq!(ran.client.marker["hits"], 3);
        assert!(ran.model_output.contains("h0"), "{}", ran.model_output);
    }

    type Sink = std::sync::Arc<std::sync::Mutex<Vec<Value>>>;

    fn rerank_cfg(addr: &str, jev_table: &str) -> String {
        format!(
            r#"
            [server]
            [[search.providers]]
            name = "mock"
            kind = "jina"
            api_key = "k"
            base_url = "http://{addr}/s"
            [providers.openrouter]
            [providers.openrouter.media]
            systemone_url = "http://{addr}/decisions"
            systemone_models = ["typesafe/jev-1.13"]
            [media]
            systemone = "openrouter/typesafe/jev-1.13"
            [server_tools.jev]
            {jev_table}
            "#
        )
    }

    /// One server holding both legs of a re-ranked search: a jina-shaped
    /// provider answering `hits` numbered results, and a Jev gateway scoring
    /// them in order.
    async fn search_and_jev(
        searched: Sink,
        asked: Sink,
        hits: usize,
        scores: &'static [f64],
    ) -> String {
        use axum::routing::post;
        let data: Vec<Value> = (0..hits)
            .map(|i| json!({"title": format!("h{i}"), "url": format!("https://h{i}.example/p"),
                            "description": format!("body {i}")}))
            .collect();
        let answers: serde_json::Map<String, Value> = scores
            .iter()
            .enumerate()
            .map(|(i, s)| (format!("r{i}"), json!({"type": "noul", "noul": s})))
            .collect();
        let router = axum::Router::new()
            .route(
                "/s",
                post(move |axum::Json(body): axum::Json<Value>| {
                    let (sink, data) = (searched.clone(), data.clone());
                    async move {
                        sink.lock().unwrap().push(body);
                        axum::Json(json!({"data": data}))
                    }
                }),
            )
            .route(
                "/decisions",
                post(move |axum::Json(body): axum::Json<Value>| {
                    let (sink, answers) = (asked.clone(), answers.clone());
                    async move {
                        sink.lock().unwrap().push(body);
                        axum::Json(json!({"model": "typesafe/jev-1.13", "answers": answers,
                                          "usage": {"input_tokens": 400, "output_tokens": 0}}))
                    }
                }),
            );
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move { axum::serve(listener, router).await.unwrap() });
        addr.to_string()
    }

    /// A call pxy cannot shape is unserved: Jev is asked nothing at all.
    #[tokio::test]
    async fn verify_without_a_claim_or_evidence_is_unserved() {
        let app = mock_app("[server]", "verify_unserved");
        let ctx = ToolCtx::new(&app, "call_1");
        for args in [
            json!({"evidence": "the page says so"}),
            json!({"claim": "axum 0.8 renamed the path syntax"}),
            json!({"claim": "  ", "evidence": "the page says so"}),
        ] {
            assert!(Tool::Verify.execute(&ctx, &args).await.is_err(), "{args}");
        }
    }

    /// One choice question over `{claim, evidence}`, and the verdict the model
    /// reads is Jev's own choice with its probabilities kept.
    #[tokio::test]
    async fn verify_asks_one_choice_question_and_reports_the_verdict() {
        let seen: std::sync::Arc<std::sync::Mutex<Vec<Value>>> = Default::default();
        let addr = jev_upstream(seen.clone(), "supported", 0.82).await;
        let app = mock_app(&jev_cfg(&addr), "verify_ok");
        let ctx = ToolCtx::new(&app, "call_1");

        let ran = Tool::Verify
            .execute(
                &ctx,
                &json!({"claim": "axum 0.8 uses {id} in paths",
                        "evidence": "In 0.8 the path syntax changed from :id to {id}."}),
            )
            .await
            .unwrap();

        let sent = seen.lock().unwrap()[0].clone();
        assert_eq!(sent["state"]["claim"], "axum 0.8 uses {id} in paths");
        assert_eq!(
            sent["state"]["evidence"],
            "In 0.8 the path syntax changed from :id to {id}."
        );
        let q = &sent["questions"]["support"];
        assert_eq!(sent["questions"].as_object().unwrap().len(), 1, "one question: {sent}");
        assert_eq!(q["type"], "choice");
        assert_eq!(q["instructions"], "Does the evidence support the claim?");
        assert_eq!(q["criteria"]["supported"], "The evidence states or directly implies the claim");
        assert_eq!(q["criteria"]["contradicted"], "The evidence states the opposite of the claim");
        assert_eq!(q["criteria"]["unsupported"], "The evidence does not address the claim");

        let out: Value = serde_json::from_str(&ran.model_output).unwrap();
        assert_eq!(out["status"], "ok");
        assert_eq!(out["verdict"], "supported");
        assert_eq!(out["confidence"], 0.82);
        assert_eq!(out["probabilities"]["supported"], 0.9);
        assert_eq!(out["model"], "typesafe/jev-1.13", "the model comes off the answer");
        assert!(ran.client.blocks.is_empty(), "verify is served silently");
        assert_eq!(ran.client.marker, json!({"id": "call_1", "verdict": "supported"}));
    }

    /// A confidence under `abstain_below` is not a verdict: Jev is right
    /// between 68% and 91% of the time, so an unsure answer says so and keeps
    /// the probabilities for the model to weigh.
    #[tokio::test]
    async fn verify_abstains_under_the_declarations_threshold() {
        let seen: std::sync::Arc<std::sync::Mutex<Vec<Value>>> = Default::default();
        let addr = jev_upstream(seen.clone(), "contradicted", 0.41).await;
        let app = mock_app(&jev_cfg(&addr), "verify_abstain");
        let args = json!({"claim": "c", "evidence": "e"});

        // Default threshold 0.5: 0.41 does not clear it.
        let ctx = ToolCtx::new(&app, "call_1");
        let ran = Tool::Verify.execute(&ctx, &args).await.unwrap();
        let out: Value = serde_json::from_str(&ran.model_output).unwrap();
        assert_eq!(out["verdict"], "uncertain");
        assert_eq!(out["probabilities"]["contradicted"], 0.9, "the probabilities are kept");
        assert_eq!(ran.client.marker["verdict"], "uncertain");

        // The declaration can lower the bar; the same answer is a verdict then.
        let ctx = ToolCtx { params: json!({"abstain_below": 0.4}), ..ToolCtx::new(&app, "call_2") };
        let ran = Tool::Verify.execute(&ctx, &args).await.unwrap();
        let out: Value = serde_json::from_str(&ran.model_output).unwrap();
        assert_eq!(out["verdict"], "contradicted");
    }

    /// A chain that is down is reported to the model, not to the client as a
    /// failed turn.
    #[tokio::test]
    async fn verify_reports_a_dead_chain_to_the_model() {
        let app = mock_app("[server]", "verify_nochain");
        let ctx = ToolCtx::new(&app, "call_1");
        let ran = Tool::Verify
            .execute(&ctx, &json!({"claim": "c", "evidence": "e"}))
            .await
            .unwrap();
        let out: Value = serde_json::from_str(&ran.model_output).unwrap();
        assert_eq!(out["status"], "error");
        assert_eq!(ran.client.marker["verdict"], "error");
    }

    /// No `[media] systemone` chain, no tool: it is never offered.
    #[test]
    fn verify_is_servable_only_behind_a_systemone_chain() {
        let bare = mock_app("[server]", "verify_servable_no");
        assert!(!Tool::Verify.servable(&bare));
        let wired = mock_app(&jev_cfg("127.0.0.1:1"), "verify_servable_yes");
        assert!(Tool::Verify.servable(&wired));
    }

    fn jev_cfg(addr: &str) -> String {
        format!(
            r#"
            [server]
            [providers.openrouter]
            [providers.openrouter.media]
            systemone_url = "http://{addr}/decisions"
            systemone_models = ["typesafe/jev-1.13"]
            [media]
            systemone = "openrouter/typesafe/jev-1.13"
            "#
        )
    }

    /// A Jev gateway answering one choice question, recording what it was asked.
    async fn jev_upstream(
        sink: std::sync::Arc<std::sync::Mutex<Vec<Value>>>,
        choice: &'static str,
        confidence: f64,
    ) -> String {
        use axum::routing::post;
        let router = axum::Router::new().route(
            "/decisions",
            post(move |axum::Json(body): axum::Json<Value>| {
                let sink = sink.clone();
                async move {
                    sink.lock().unwrap().push(body);
                    axum::Json(json!({
                        "model": "typesafe/jev-1.13",
                        "answers": {"support": {
                            "type": "choice",
                            "choice": choice,
                            "probabilities": {choice: 0.9},
                            "confidence": confidence,
                        }},
                        "usage": {"input_tokens": 120, "output_tokens": 0},
                    }))
                }
            }),
        );
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move { axum::serve(listener, router).await.unwrap() });
        addr.to_string()
    }

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
