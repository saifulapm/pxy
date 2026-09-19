//! The routing engine: candidate filtering, fallback walk, error
//! classification, usage recording. Synthesis of the OmniRoute + litellm
//! research.

use std::sync::Arc;
use std::time::Duration;

use bytes::Bytes;
use futures_util::StreamExt;
use jiff::Timestamp;
use serde_json::{json, Value};
use tracing::{debug, info, warn};

use crate::catalog::{Candidate, Catalog};
use crate::config::{Config, ErrorAction, ServerToolsConfig, WireFormat};
use crate::media::pdf;
use crate::secrets::Secrets;
use crate::state::State;
use crate::translate::server_tools;
use crate::translate::sse::{SseEvent, SseParser};
use crate::translate::tool_search;
use crate::translate::think::ThinkFilter;
use crate::translate::tool_text::ToolTextFilter;
use crate::translate::{anthropic_to_openai, estimate_tokens, openai_to_anthropic, responses_upstream, TokenUsage};
use crate::usage::current_windows;

pub struct App {
    pub cfg: Config,
    pub catalog: Catalog,
    pub secrets: Secrets,
    pub state: State,
    pub http: reqwest::Client,
}

pub type SharedApp = Arc<App>;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ClientFormat {
    Openai,
    Anthropic,
}

/// Extra request context forwarded from the client connection.
#[derive(Debug, Default, Clone)]
pub struct ClientContext {
    /// anthropic-beta headers, forwarded verbatim to anthropic upstreams
    pub anthropic_beta: Option<String>,
    /// Which coding agent sent this request ("claude", "codex", …), parsed
    /// from the x-pxy-agent header or the api-key suffix `pxy launch` sets.
    /// Only feeds the per-model usage stats — never routing.
    pub agent: Option<String>,
    /// The client's own conversation id, if it sent one: `x-opencode-session`,
    /// or the `x-session-affinity` / `x-session-id` spelling opencode falls
    /// back to when the provider it is configured with isn't named `opencode*`
    /// (which is pxy's case — `pxy launch opencode` registers it as `pxy`).
    /// Forwarded ONLY to opencode-family upstreams; see `opencode_session`.
    pub session: Option<String>,
    /// Set by `/v1/responses`: both it and Chat Completions speak
    /// [`ClientFormat::Openai`], but only the Responses dialect understands
    /// pxy's served-tool marker chunk, so only it may receive one.
    pub responses: bool,
    /// How many sub-requests deep this turn is: 0 for a client request, 1 for a
    /// meta-tool's sub-request. At 1 or more the meta-tools are stripped, so
    /// recursion stops at one level.
    pub tool_depth: u8,
}

/// Outcome handed to the HTTP layer.
pub enum Outcome {
    Json {
        status: u16,
        body: Value,
        provider: Option<String>,
        /// Upstream response headers to relay (see `forwardable_headers`).
        headers: Headers,
    },
    Stream {
        provider: String,
        body: axum::body::Body,
        headers: Headers,
    },
}

/// Relayed upstream response headers, in wire order.
pub type Headers = Vec<(String, String)>;

/// Everything classification and passthrough need about a failed upstream
/// response, taken from that one response so they cannot drift apart.
struct UpstreamError {
    status: u16,
    retry_after: Option<Duration>,
    body: String,
    headers: Headers,
}

/// The newest real upstream error seen during a walk, kept so that an
/// exhausted single-candidate walk can answer with the upstream's own
/// response instead of a synthetic one.
struct RawError {
    status: u16,
    body: String,
    candidate: String,
    headers: Headers,
}

/// The upstream response headers that are safe and useful to hand the client.
///
/// Dropped: hop-by-hop and entity headers, which describe pxy's connection to
/// the UPSTREAM and would misdescribe the one pxy is writing (litellm's
/// exclusion set); and AI-gateway telemetry prefixes, which Claude Code's own
/// telemetry inspects to detect that it is talking through a gateway
/// (CLIProxyAPI strips the same list, for the same reason).
///
/// Everything else is forwarded verbatim, which is the entire point:
/// `retry-after` and the `x-ratelimit-*`/`anthropic-ratelimit-*` families are
/// what a harness's own backoff and limit UI read. pxy dropping them is why
/// agents fight pxy's retry timing instead of the upstream's.
fn forwardable_headers(h: &reqwest::header::HeaderMap) -> Headers {
    const DROP: &[&str] = &[
        "transfer-encoding",
        "content-encoding",
        "content-length",
        "connection",
        "keep-alive",
        "server",
        "date",
        // pxy writes its own on the response it builds.
        "content-type",
        // pxy owns this one; an upstream value would name the wrong hop.
        "x-pxy-provider",
        // These describe pxy's connection to the UPSTREAM host, not the
        // loopback one pxy is answering on, so relaying them is at best noise
        // and at worst misleading: a session cookie for the upstream handed to
        // a local agent that is not a browser and will never use it, an
        // alternative-service advert for a host the client is not talking to,
        // and a transport-security policy for a different origin.
        "set-cookie",
        "set-cookie2",
        "alt-svc",
        "strict-transport-security",
    ];
    const DROP_PREFIX: &[&str] =
        &["x-litellm-", "helicone-", "x-portkey-", "cf-aig-", "x-kong-", "x-bt-"];
    h.iter()
        .filter(|(k, _)| {
            let n = k.as_str();
            !DROP.contains(&n) && !DROP_PREFIX.iter().any(|p| n.starts_with(p))
        })
        // Non-UTF8 header values cannot be relayed safely; there are none in
        // practice and dropping one is better than mangling it.
        .filter_map(|(k, v)| Some((k.as_str().to_string(), v.to_str().ok()?.to_string())))
        .collect()
}

/// kv key holding one group's route pin, set from `pxy route` / the desktop
/// panel. Read per request so a pin takes effect without a daemon restart.
/// Pins are per group: pinning deepseek-flash into `deepseek` must not steer
/// a `glm` session.
pub fn route_pin_key(group: &str) -> String {
    format!("route_pin:{group}")
}

/// The group's active pin: the stored model resolved to listed candidates, or
/// None when nothing is pinned or the pin went stale. `is_listed`, not just
/// resolves: resolve() fabricates a candidate for any id under an enabled
/// provider, and a pin gone stale (config edit, refresh dropping the model)
/// must degrade to the chain, not put a phantom at the head of the walk.
pub fn active_route_pin(
    catalog: &Catalog,
    cfg: &Config,
    state: &State,
    group: &str,
) -> Option<Vec<Candidate>> {
    let pin = state.kv_get(&route_pin_key(group)).ok().flatten().filter(|p| !p.is_empty())?;
    let resolved = catalog.resolve(cfg, &pin);
    if !resolved.is_empty() && resolved.iter().all(|c| catalog.is_listed(&c.full_id())) {
        Some(resolved)
    } else {
        warn!(group, pin, "route pin is not in the catalog; using the group chain");
        None
    }
}

/// Candidates for a request, honoring the group's route pin: on a GROUP
/// request the pinned model is walked FIRST, with the group's chain behind it
/// as fallback — pinning must never cost the failover safety a group exists
/// for. An agent is launched with a fixed group id, so the pin is the only way
/// to steer a running session, and it steers only that group. Explicit model
/// requests are untouched, and a pin that no longer resolves (config edit,
/// provider disabled) degrades to the plain chain.
pub fn resolve_candidates(
    catalog: &Catalog,
    cfg: &Config,
    state: &State,
    requested: &str,
    session: Option<&str>,
) -> Vec<Candidate> {
    // Explicit single-model requests skip the pin/affinity logic but STILL
    // get the multi-account expansion — an account walk is exactly what an
    // explicit request wants when account #1 is cooling.
    if !catalog.is_group(requested) {
        return catalog
            .resolve(cfg, requested)
            .into_iter()
            .flat_map(|c| expand_accounts(cfg, c))
            .collect();
    }
    let mut chain = catalog.resolve(cfg, requested);
    // Headroom ranking (opt-in): more remaining LOCAL allowance first, ties in
    // config order (stable sort). Applied to the chain only — a manual pin and
    // session affinity still precede it.
    if catalog
        .group_config(cfg, requested)
        .and_then(|g| g.headroom)
        .unwrap_or(false)
    {
        chain.sort_by(|a, b| {
            remaining_headroom(cfg, state, b)
                .partial_cmp(&remaining_headroom(cfg, state, a))
                .unwrap_or(std::cmp::Ordering::Equal)
        });
    }
    // Manual pin: walked FIRST, ahead of session affinity — `pxy route` is an
    // explicit human decision.
    let pinned = catalog
        .group_name(requested)
        .and_then(|g| active_route_pin(catalog, cfg, state, g));
    // Session affinity: the candidate this conversation last won on walks
    // first, so a post-failover conversation keeps its prompt-cache locality
    // instead of bouncing back to the chain head. A stale or unlisted
    // binding is ignored — the walk's winner rebinds it (self-healing). So
    // is one outside this group's chain: the key is the conversation's
    // opener, which two sessions on different groups can share, and a `gpt`
    // walk must never start on the model a `muse` session won on.
    let mut affinity = None;
    if pinned.is_none() {
        if let Some(key) = session {
            if let Some(full_id) = state.session_get(key) {
                let bound = catalog.resolve(cfg, &full_id);
                let in_chain = |c: &Candidate| chain.iter().any(|m| m.full_id() == c.full_id());
                if !bound.is_empty() && bound.iter().all(in_chain) {
                    affinity = Some(bound);
                }
            }
        }
    }
    let mut out = Vec::new();
    if let Some(p) = &pinned {
        let ids: Vec<String> = p.iter().map(|c| c.full_id()).collect();
        chain.retain(|c| !ids.contains(&c.full_id()));
        out.extend(p.clone());
    }
    if let Some(a) = &affinity {
        let ids: Vec<String> = a.iter().map(|c| c.full_id()).collect();
        chain.retain(|c| !ids.contains(&c.full_id()));
        out.extend(a.clone());
    }
    out.extend(chain);
    // Multi-account expansion, LAST: every bare candidate becomes one
    // candidate per configured account (config order = fill-first), so the
    // ordinary walk/cooldown machinery below works per account unchanged.
    out.into_iter().flat_map(|c| expand_accounts(cfg, c)).collect()
}

/// Expand one bare candidate into its configured accounts. Providers without
/// `accounts` yield themselves unchanged (implicit single default).
fn expand_accounts(cfg: &Config, c: Candidate) -> Vec<Candidate> {
    let Some(pc) = cfg.providers.get(&c.provider) else { return vec![c] };
    if pc.accounts.is_empty() {
        return vec![c];
    }
    pc.accounts
        .iter()
        .map(|a| Candidate {
            account: Some(a.name.clone()),
            provider: c.provider.clone(),
            model: c.model.clone(),
        })
        .collect()
}

/// Stable conversation fingerprint for session affinity: Claude Code always
/// sends `metadata.user_id`; opencode sends `user`; otherwise hash the first
/// message (stable within a conversation). FNV-1a, not DefaultHasher — the
/// std hasher is keyed per process, which would silently invalidate every
/// stored binding on daemon restart.
fn session_key(payload: &Value) -> Option<String> {
    if let Some(id) = payload["metadata"]["user_id"].as_str().filter(|s| !s.is_empty()) {
        return Some(format!("uid:{id}"));
    }
    if let Some(id) = payload["user"].as_str().filter(|s| !s.is_empty()) {
        return Some(format!("user:{id}"));
    }
    let first = payload["messages"].as_array()?.first()?;
    let text = match &first["content"] {
        Value::String(s) => s.clone(),
        Value::Array(parts) => parts
            .iter()
            .filter_map(|p| p["text"].as_str())
            .collect::<Vec<_>>()
            .join("\n"),
        _ => return None,
    };
    if text.is_empty() {
        return None;
    }
    Some(format!("hash:{:016x}", fnv1a(text.as_bytes())))
}

fn fnv1a(bytes: &[u8]) -> u64 {
    let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
    for b in bytes {
        hash ^= u64::from(*b);
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    hash
}

/// A reasoning override parsed off the model id: `no-think/<id>` and a
/// trailing `-low`/`-medium`/`-high`/`-xhigh`/`-max`.
#[derive(Debug, Clone, PartialEq)]
enum ModelVariant {
    NoThink,
    Effort(String),
}

/// Split a requested id into its base plus an optional reasoning variant. A
/// suffix is stripped only when the base actually resolves, so a real model id
/// ending in `-max` (there are several) is never mangled.
fn split_model_variant(
    catalog: &Catalog,
    cfg: &Config,
    requested: &str,
) -> (String, Option<ModelVariant>) {
    if let Some(base) = requested.strip_prefix("no-think/")
        && resolves_id(catalog, cfg, base)
    {
        return (base.to_string(), Some(ModelVariant::NoThink));
    }
    // Longest suffixes first, so `…-xhigh` is not read as `…-high`.
    for level in ["minimal", "xhigh", "medium", "high", "low", "max"] {
        if let Some(base) = requested.strip_suffix(&format!("-{level}"))
            && resolves_id(catalog, cfg, base)
        {
            return (base.to_string(), Some(ModelVariant::Effort(level.to_string())));
        }
    }
    (requested.to_string(), None)
}

fn resolves_id(catalog: &Catalog, cfg: &Config, id: &str) -> bool {
    if catalog.is_group(id) {
        return true;
    }
    // For a provider-qualified id, `resolve` fabricates a spec for ANY model
    // under a known provider — so a real `p/qwen3.8-max` would look like
    // `p/qwen3.8` resolves and get mangled. The catalog list is the strict
    // test. A bare id goes through `resolve`, which is already strict there.
    if id.contains('/') {
        return catalog.is_listed(id);
    }
    !catalog.resolve(cfg, id).is_empty()
}

/// The listed effort nearest to `requested` in canonical order, when the
/// model lists efforts and `requested` is not among them. Ties go to the
/// higher level: the caller asked to think, so the clamp errs toward more
/// of it, not less. An unlisted spelling and an empty list both mean
/// nothing to do — the former is not pxy's to reinterpret, the latter is
/// nobody knowing.
fn clamp_effort(requested: &str, allowed: &[String]) -> Option<String> {
    let order = &crate::refresh::EFFORTS;
    let want = order.iter().position(|e| *e == requested)?;
    if allowed.is_empty() || allowed.iter().any(|a| a == requested) {
        return None;
    }
    order
        .iter()
        .enumerate()
        .filter(|(_, e)| allowed.iter().any(|a| a == *e))
        .min_by_key(|(i, _)| (i.abs_diff(want), std::cmp::Reverse(*i)))
        .map(|(_, e)| e.to_string())
}

/// Write a model-variant override into the request body. An OpenAI client
/// carries it as `reasoning_effort`; an Anthropic client as `thinking` (which
/// `anthropic_to_openai` maps back to effort for an OpenAI upstream). The
/// field is only written in the client's OWN dialect — an `effort` key on an
/// Anthropic payload would otherwise leak straight to an Anthropic upstream
/// on the passthrough path.
fn apply_model_variant(payload: &mut Value, client_format: ClientFormat, variant: &ModelVariant) {
    let (effort, budget) = match variant {
        ModelVariant::NoThink => ("none", None),
        ModelVariant::Effort(level) => {
            let budget = match level.as_str() {
                "low" | "minimal" => 1024,
                "medium" => 8192,
                _ => 16384,
            };
            (level.as_str(), Some(budget))
        }
    };
    if client_format == ClientFormat::Openai {
        payload["reasoning_effort"] = json!(effort);
        return;
    }
    match budget {
        // No thinking: the field's absence is the portable "off".
        None => {
            if let Some(o) = payload.as_object_mut() {
                o.remove("thinking");
            }
        }
        Some(b) => {
            payload["thinking"] = json!({"type": "enabled", "budget_tokens": b});
            // Anthropic requires max_tokens > budget_tokens.
            if payload["max_tokens"].as_u64().unwrap_or(0) <= b {
                payload["max_tokens"] = json!(b + 4096);
            }
        }
    }
}

/// Route one chat request. Boxed because a meta-tool's executor issues its
/// sub-request through this same function from inside the router's stream
/// future: an unboxed call would make the two futures' types cyclic and the
/// compiler could not prove either `Send`. Recursion itself is stopped by
/// `ClientContext::tool_depth`.
pub fn handle_chat(
    app: SharedApp,
    client_format: ClientFormat,
    payload: Value,
    ctx: ClientContext,
) -> futures_util::future::BoxFuture<'static, Outcome> {
    Box::pin(async move {
        // Read the gate before the payload is moved; apply the repair after
        // the whole walk (and any served-tool loop) has settled on an answer,
        // which is also before the Responses route rewrites it.
        let heal = should_heal(&app.cfg, client_format, &payload);
        let mut outcome = handle_chat_inner(app, client_format, payload, ctx).await;
        if heal {
            heal_json_outcome(&mut outcome);
        }
        outcome
    })
}

/// Plugin ids pxy implements. OpenRouter's vocabulary is larger; an id that is
/// not here is ignored with a log, because a client that names one still wants
/// its answer, not a 400.
const KNOWN_PLUGINS: [&str; 2] = [RESPONSE_HEALING, FILE_PARSER];
const RESPONSE_HEALING: &str = "response-healing";
const FILE_PARSER: &str = "file-parser";

/// Far past any real document, and small enough that a data URL cannot eat
/// the process.
const MAX_FILE_BYTES: usize = 50 * 1024 * 1024;

/// The plugin ids a request turned on, from OpenRouter's `plugins` key. An
/// entry pxy cannot read an id off is skipped.
pub fn request_plugins(payload: &Value) -> Vec<String> {
    let Some(entries) = payload["plugins"].as_array() else {
        return Vec::new();
    };
    let mut ids = Vec::new();
    for entry in entries {
        let Some(id) = entry["id"].as_str() else { continue };
        if !KNOWN_PLUGINS.contains(&id) {
            debug!(id, "plugin not implemented, ignored");
        }
        ids.push(id.to_string());
    }
    ids
}

/// Replace every file part with the text pxy parsed out of it (wiki:plugins).
///
/// Runs before the candidate walk, so every candidate sees the same body and
/// the parse happens once whichever model wins, and only at depth 0: a
/// meta-tool's leg carries text its caller already parsed. A file pxy cannot
/// read becomes one line saying why, never nothing: a model told a document
/// failed can ask for it another way, where a model told nothing answers as
/// if the document did not exist.
pub async fn parse_file_parts(payload: &mut Value, app: &SharedApp, ctx: &ClientContext) {
    let cfg = &app.cfg.plugins.file_parser;
    if !cfg.enabled || ctx.tool_depth != 0 {
        return;
    }
    let engine = requested_engine(payload);
    let Some(messages) = payload["messages"].as_array_mut() else { return };
    for message in messages {
        let Some(parts) = message["content"].as_array_mut() else { continue };
        for part in parts {
            if part["type"] != "file" {
                continue;
            }
            let name =
                part["file"]["filename"].as_str().unwrap_or("document.pdf").to_string();
            let data = part["file"]["file_data"].as_str().unwrap_or("").to_string();
            let text = match parse_one_file(app, &data, engine, cfg, ctx).await {
                Ok(markdown) => format!("[file {name}]\n{markdown}"),
                Err(why) => {
                    warn!(file = %name, error = %why, "file-parser could not read a file");
                    format!("[file {name}: {why}]")
                }
            };
            *part = json!({"type": "text", "text": text});
        }
    }
}

/// One file's Markdown, from `kv` when these bytes have been parsed before.
async fn parse_one_file(
    app: &SharedApp,
    file_data: &str,
    engine: pdf::Engine,
    cfg: &crate::config::FileParserConfig,
    ctx: &ClientContext,
) -> Result<String, String> {
    let bytes = file_bytes(app, file_data).await?;
    // The magic bytes, not the declared media type: poppler is the only
    // parser here, and a client's label is not evidence.
    if !bytes.starts_with(b"%PDF") {
        return Err("not a PDF".to_string());
    }
    let key = format!("file_parse:{}", pdf::sha256_hex(&bytes));
    if let Ok(Some(cached)) = app.state.kv_get(&key) {
        return Ok(cached);
    }
    let markdown =
        pdf::pdf_to_markdown(app, &bytes, engine, cfg.ocr_model.as_deref(), cfg.max_pages, ctx)
            .await?;
    // No TTL: the same bytes parse to the same text forever, and a client
    // resends its whole history every turn.
    if let Err(e) = app.state.kv_set(&key, &markdown) {
        warn!(error = %e, "file-parser could not cache a parse");
    }
    Ok(markdown)
}

/// `file_data` is a base64 data URL or a URL to fetch.
async fn file_bytes(app: &SharedApp, file_data: &str) -> Result<Vec<u8>, String> {
    use base64::Engine as _;
    if let Some((_, b64)) =
        file_data.strip_prefix("data:").and_then(|rest| rest.split_once(";base64,"))
    {
        // Checked before decoding: the encoding is 4 characters per 3 bytes,
        // so an oversize file is refused without ever being materialised.
        if b64.len() / 4 * 3 > MAX_FILE_BYTES {
            return Err("file is too large".to_string());
        }
        return base64::engine::general_purpose::STANDARD
            .decode(b64)
            .map_err(|e| format!("unreadable base64: {e}"));
    }
    if file_data.starts_with("https://") || file_data.starts_with("http://") {
        let resp = app
            .http
            .get(file_data)
            .timeout(Duration::from_secs(60))
            .send()
            .await
            .map_err(|e| format!("cannot fetch the file: {e}"))?;
        if !resp.status().is_success() {
            return Err(format!("fetch failed: {}", resp.status().as_u16()));
        }
        if resp.content_length().is_some_and(|n| n > MAX_FILE_BYTES as u64) {
            return Err("file is too large".to_string());
        }
        let bytes = resp.bytes().await.map_err(|e| format!("cannot read the file: {e}"))?;
        if bytes.len() > MAX_FILE_BYTES {
            return Err("file is too large".to_string());
        }
        return Ok(bytes.to_vec());
    }
    Err("file_data must be a data URL or an https URL".to_string())
}

/// The engine the client asked for on its `file-parser` entry. Config has no
/// say here: the engine is about one document, not the daemon.
fn requested_engine(payload: &Value) -> pdf::Engine {
    if !request_plugins(payload).iter().any(|id| id == FILE_PARSER) {
        return pdf::Engine::Auto;
    }
    let asked = payload["plugins"]
        .as_array()
        .and_then(|entries| entries.iter().find(|e| e["id"] == FILE_PARSER))
        .and_then(|entry| entry["pdf"]["engine"].as_str())
        .unwrap_or_default();
    pdf::Engine::from_name(asked)
}

/// Whether this turn's answer gets `response-healing`.
///
/// Narrow on purpose: healing rewrites what the model said, so it needs the
/// client to have declared JSON mode (which is the client promising itself it
/// will parse the content) and an answer that exists as one value. A stream is
/// handed to the client chunk by chunk with nothing to rewrite, and an
/// Anthropic Messages client has no `response_format` to declare in the first
/// place.
fn should_heal(cfg: &Config, client_format: ClientFormat, payload: &Value) -> bool {
    let asked = request_plugins(payload).iter().any(|id| id == RESPONSE_HEALING);
    if client_format != ClientFormat::Openai || payload["stream"] == json!(true) {
        return false;
    }
    let json_mode = matches!(
        payload["response_format"]["type"].as_str(),
        Some("json_object" | "json_schema")
    );
    json_mode && (asked || cfg.plugins.response_healing)
}

/// Replace the assistant's content with its repair, when there is one. A body
/// no closer set rescues is left exactly as the upstream sent it.
fn heal_json_outcome(outcome: &mut Outcome) {
    let Outcome::Json { status: 200, body, .. } = outcome else {
        return;
    };
    // Read through an immutable index: `IndexMut` would materialize a
    // `choices` array on a response that has none.
    let Some(healed) = body["choices"][0]["message"]["content"]
        .as_str()
        .and_then(crate::translate::heal::heal_json)
    else {
        return;
    };
    debug!("response-healing repaired the JSON answer");
    body["choices"][0]["message"]["content"] = json!(healed);
}

async fn handle_chat_inner(
    app: SharedApp,
    client_format: ClientFormat,
    mut payload: Value,
    ctx: ClientContext,
) -> Outcome {
    // axum's Json<Value> accepts any valid JSON, so `[]`, `"x"` and `5` all
    // reach here. Everything downstream indexes the body by key, and
    // serde_json's IndexMut PANICS on a non-object — so a malformed request
    // would kill the handler task instead of answering. Reject it at the door.
    if !payload.is_object() {
        return error_outcome(
            client_format,
            400,
            "invalid_request_error",
            "request body must be a JSON object",
        );
    }
    // Before anything reads the transcript: a PDF a client attached becomes
    // the text every candidate, every translator and the tool loop then see.
    parse_file_parts(&mut payload, &app, &ctx).await;
    let requested = payload["model"]
        .as_str()
        .filter(|m| !m.is_empty())
        .map(str::to_string)
        .unwrap_or_else(|| app.cfg.default_route());
    // Model-variant grammar (`no-think/…`, `…-high`): resolved BEFORE candidate
    // lookup, because the base id is what routes. The override is written back
    // into the body so it survives translation in either direction.
    let (requested, variant) = split_model_variant(&app.catalog, &app.cfg, &requested);
    if let Some(v) = &variant {
        apply_model_variant(&mut payload, client_format, v);
    }
    let stream = payload["stream"].as_bool().unwrap_or(false);

    // In-band magic prompt: a last user message of exactly "@@usage" is
    // answered locally with the quota report — zero tokens, no upstream.
    if is_usage_magic(&payload) {
        return usage_outcome(usage_report(&app), client_format, stream);
    }

    // Session affinity key (group walks only): keeps a conversation on its
    // last winning candidate for prompt-cache locality.
    let session_key = if app.catalog.is_group(&requested) { session_key(&payload) } else { None };
    // A win under a route pin is the pin's doing, not the conversation's:
    // binding it would keep a session on the pinned model for an hour after
    // `pxy route --clear`, when the chain is what was asked for.
    let pinned_ids: Vec<String> = app
        .catalog
        .group_name(&requested)
        .and_then(|g| active_route_pin(&app.catalog, &app.cfg, &app.state, g))
        .map(|p| p.iter().map(|c| c.full_id()).collect())
        .unwrap_or_default();
    let candidates =
        resolve_candidates(&app.catalog, &app.cfg, &app.state, &requested, session_key.as_deref());
    if candidates.is_empty() {
        return error_outcome(
            client_format,
            404,
            "not_found_error",
            &format!("model '{requested}' not found"),
        );
    }

    let input_estimate = estimate_tokens(&payload["messages"])
        + estimate_tokens(&payload["system"])
        + estimate_tokens(&payload["tools"]);
    let wants_tools = payload["tools"].as_array().is_some_and(|a| !a.is_empty());
    let server_tools = openai_unservable_server_tools(&payload, &app.cfg.server_tools);
    // Config-declared server tools ride every client turn. After the two
    // reads above on purpose: a default never makes a candidate unservable
    // and never turns a toolless request into a tools request for routing.
    if ctx.tool_depth == 0 {
        inject_default_server_tools(&mut payload, &app);
    }

    let mut skipped: Vec<String> = Vec::new();
    let multi = candidates.len() > 1;

    // A single-candidate request bound for an OpenAI upstream with server
    // tools pxy can't translate gets an honest 400 up front: the translator
    // would drop the tools silently, and the upstream can't answer for what
    // it never saw. Multi-candidate walks skip such candidates instead
    // (check_candidate) — an Anthropic-format peer can serve them natively.
    if !multi && !server_tools.is_empty() {
        let cand = &candidates[0];
        let openai_bound = app
            .cfg
            .providers
            .get(&cand.provider)
            .is_some_and(|p| cand.format(p) != WireFormat::Anthropic);
        if openai_bound {
            return error_outcome(
                client_format,
                400,
                "invalid_request_error",
                &format!(
                    "server tool(s) {} cannot be served: '{}' speaks the OpenAI protocol and \
                     pxy cannot run them",
                    server_tools.join(", "),
                    cand.full_id()
                ),
            );
        }
    }
    // Set when an upstream 400s with a context-window error: our chars/4
    // estimate under-counted, so every candidate at that context size or
    // below would fail identically — skip them instead of burning calls.
    let mut ctx_too_small: Option<u64> = None;
    // Sticky across walks: ANY non-deterministic obstacle (cooldown, rpm,
    // limits, a 429/5xx attempt…) means the terminal error must stay
    // retryable — only when context size or unservable server tools were the
    // sole problem is a 400 the honest answer.
    let mut other_failures = false;
    let mut server_tool_skips = false;
    // Newest real upstream error of the walk (status, body, candidate).
    let mut last_raw: Option<RawError> = None;

    // Free-first chains: fall onto a paid step only when every prior failure
    // was quota exhaustion (opt-in per group). "Paid" = the model is not
    // marked `free = true`.
    let fallback_only_quota = app
        .catalog
        .group_config(&app.cfg, &requested)
        .and_then(|g| g.fallback_only_on_quota_exhaustion)
        .unwrap_or(false);
    let mut quota_only = true;

    for attempt in 0..=MAX_RETRIES {
        skipped.clear();
        let mut saw_rpm_limit = false;
        for cand in &candidates {
            if fallback_only_quota && cand.model.free != Some(true) && !quota_only {
                skipped.push(format!(
                    "{}: paid reserve held (prior failure was not quota exhaustion)",
                    cand.full_id()
                ));
                continue;
            }
            if ctx_too_small.is_some_and(|c| cand.model.context_length <= c) {
                skipped.push(format!("{}: context window too small", cand.full_id()));
                continue;
            }
            if let Err(reason) = check_candidate(
                &app,
                cand,
                input_estimate,
                wants_tools,
                !server_tools.is_empty(),
                multi,
            ) {
                saw_rpm_limit |= reason == "rpm limit" || reason == "tpm limit";
                // Filter reasons for context start with "context", the
                // server-tool skip with "server tools" — both deterministic.
                // Everything else (cooldown/rpm/limits/disabled) is an
                // obstacle a later retry might clear.
                server_tool_skips |= reason.starts_with("server tools");
                other_failures |=
                    !(reason.starts_with("context") || reason.starts_with("server tools"));
                if !is_quota_failure(None, &reason) {
                    quota_only = false;
                }
                skipped.push(format!("{}: {reason}", cand.full_id()));
                continue;
            }

            match try_candidate(&app, cand, client_format, &payload, stream, input_estimate, &ctx, multi)
                .await
            {
                AttemptResult::Done(outcome) => {
                    // A real success repairs the model's failure-rate record.
                    app.state.model_result(&cand.state_provider(), &cand.model.id, true);
                    // ...and rebinds the conversation's session affinity,
                    // unless the pin chose this candidate.
                    if let Some(key) = &session_key {
                        if !pinned_ids.contains(&cand.full_id()) {
                            app.state.session_set(key, &cand.full_id());
                        }
                    }
                    return outcome;
                }
                AttemptResult::Skip(reason) => {
                    warn!(candidate = %cand.full_id(), %reason, "failover");
                    other_failures = true;
                    if !is_quota_failure(None, &reason) {
                        quota_only = false;
                    }
                    // A real attempt failed: feed the failure-rate rule.
                    app.state.model_result(&cand.state_provider(), &cand.model.id, false);
                    skipped.push(format!("{}: {reason}", cand.full_id()));
                }
                AttemptResult::SkipRaw { reason, status, body, headers } => {
                    warn!(candidate = %cand.full_id(), %reason, "failover");
                    other_failures = true;
                    if !is_quota_failure(Some(status), &body) {
                        quota_only = false;
                    }
                    app.state.model_result(&cand.state_provider(), &cand.model.id, false);
                    // Keep the newest upstream error: if the walk ends with
                    // nothing better, this is what the client should see.
                    last_raw =
                        Some(RawError { status, body, candidate: cand.full_id(), headers });
                    skipped.push(format!("{}: {reason}", cand.full_id()));
                }
                AttemptResult::SkipContextWindow(reason) => {
                    // The real tokenizer overruled our estimate. No cooldown
                    // (a smaller request to this model would work fine).
                    warn!(candidate = %cand.full_id(), %reason, "failover (context window)");
                    quota_only = false;
                    let c = ctx_too_small.get_or_insert(0);
                    *c = (*c).max(cand.model.context_length);
                    skipped.push(format!("{}: {reason}", cand.full_id()));
                }
                AttemptResult::Fatal(outcome) => return outcome,
            }
        }

        // The whole chain came up empty. Switching costs nothing so it never
        // waits (litellm rule); only back off now that we're out of options,
        // and only when something can actually recover within the wait.
        if attempt == MAX_RETRIES {
            break;
        }
        let Some(wait) = retry_wait(soonest_recovery(&app, &candidates), saw_rpm_limit) else {
            break;
        };
        info!(
            attempt = attempt + 1,
            wait_ms = wait.as_millis() as u64,
            "no candidate available; retrying after backoff"
        );
        tokio::time::sleep(wait).await;
    }

    // Honest terminal status: if the only real failures were context-window
    // 400s, telling the client "rate limited" makes it back off pointlessly —
    // the request itself is too large and retrying can't fix that.
    if ctx_too_small.is_some() && !other_failures {
        return error_outcome(
            client_format,
            400,
            "invalid_request_error",
            &format!(
                "input exceeds the context window of every available candidate for '{requested}' \
                 (tried/skipped: {})",
                skipped.join("; ")
            ),
        );
    }
    // Same honesty rule for server tools: when every candidate was skipped
    // because none can serve the declared server tools, retrying can't fix
    // the request — a 429 would just make the client back off pointlessly.
    if server_tool_skips && !other_failures {
        return error_outcome(
            client_format,
            400,
            "invalid_request_error",
            &format!(
                "request declares server tool(s) {} that no available candidate for '{requested}' \
                 can serve (tried/skipped: {})",
                server_tools.join(", "),
                skipped.join("; ")
            ),
        );
    }
    // The retries are spent and every candidate failed. On a single-candidate
    // request the upstream's own error IS the story — the client asked for
    // exactly this model, so replacing a real 429 "usage limit reached, resets
    // at 5pm" with a synthetic overloaded_error throws away the only thing its
    // limit UI and status-specific retry logic can act on. A multi-candidate
    // walk keeps the aggregate below: N different failures don't reduce to one.
    if !multi && let Some(e) = last_raw {
        return passthrough_json(client_format, e.status, &e.body, &e.candidate, e.headers);
    }
    // Structured, not just prose : when the earliest recovery
    // is known — a cooldown that will expire — the client gets `Retry-After`
    // plus machine-readable reset fields, so its own backoff can wait exactly
    // that long instead of guessing. A harness can act on a number; it cannot
    // act on a sentence.
    let mut body = error_body(
        client_format,
        "overloaded_error",
        &format!(
            "no provider available for '{requested}' (tried/skipped: {})",
            skipped.join("; ")
        ),
    );
    let mut headers: Headers = Vec::new();
    if let Some(wait) = soonest_cooldown_end(&app, &candidates) {
        let secs = wait.as_secs().max(1);
        headers.push(("retry-after".into(), secs.to_string()));
        body["error"]["code"] = json!("model_cooldown");
        body["error"]["reset_seconds"] = json!(secs);
        if let Ok(at) = Timestamp::from_second(Timestamp::now().as_second() + secs as i64) {
            body["error"]["reset_time"] = json!(at.to_string());
        }
    }
    Outcome::Json { status: 429, body, provider: None, headers }
}

// ---------------------------------------------------------------------------
// @@usage — in-band quota report (answered locally, zero tokens)
// ---------------------------------------------------------------------------

/// True when the LAST user message is exactly the magic token. Works from
/// inside any agent: type "@@usage" (or "@@pxy-usage"), get the report.
fn is_usage_magic(payload: &Value) -> bool {
    // The magic message must be the FINAL message: an assistant-final
    // continuation whose previous user turn was "@@usage" is a real
    // request, not a report query.
    let Some(last) = payload["messages"]
        .as_array()
        .and_then(|m| m.last())
        .filter(|m| m["role"] == "user")
    else {
        return false;
    };
    let text = match &last["content"] {
        Value::String(s) => s.trim(),
        Value::Array(parts) if parts.len() == 1 => {
            parts[0]["text"].as_str().unwrap_or("").trim()
        }
        _ => return false,
    };
    text == "@@usage" || text == "@@pxy-usage"
}

fn human_tokens(n: u64) -> String {
    if n >= 1_000_000 {
        format!("{:.1}M", n as f64 / 1e6)
    } else if n >= 1_000 {
        format!("{:.0}k", n as f64 / 1e3)
    } else {
        n.to_string()
    }
}

fn usage_report(app: &App) -> String {
    let now = Timestamp::now();
    let default_limits = crate::config::Limits::default();
    let mut lines = vec!["pxy usage (today / this month)".to_string()];

    for (name, p) in &app.cfg.providers {
        if !p.enabled {
            continue;
        }
        let limits = p.limits.as_ref().unwrap_or(&default_limits);
        let Ok(w) = crate::usage::current_windows(limits, now) else { continue };
        for key in [name.clone(), crate::media::media_key(name)] {
            let day = app.state.usage(&key, "day", w.day_start).unwrap_or_default();
            let month = app.state.usage(&key, "month", w.month_start).unwrap_or_default();
            if day.requests == 0 && month.requests == 0 {
                continue;
            }
            let cap = match limits.daily_requests {
                Some(c) if !key.contains('#') => format!("/{c}"),
                _ => String::new(),
            };
            lines.push(format!(
                "  {key}: {}{} req, {} tok | month: {} req, {} tok",
                day.requests,
                cap,
                human_tokens(day.tokens),
                month.requests,
                human_tokens(month.tokens),
            ));
        }
    }

    let cooldowns = app.state.active_cooldowns();
    if !cooldowns.is_empty() {
        lines.push("cooldowns:".to_string());
        for (key, cd) in cooldowns {
            let left = cd.until.saturating_duration_since(std::time::Instant::now()).as_secs();
            let left = if left >= 120 {
                format!("{}m", left / 60)
            } else {
                format!("{left}s")
            };
            lines.push(format!("  {key}: {} ({left} left)", cd.reason));
        }
    }
    if lines.len() == 1 {
        lines.push("  (no usage recorded yet today)".to_string());
    }
    lines.join("\n")
}

/// Shape the report as a protocol-correct response in the client's dialect,
/// streaming included — no upstream is contacted.
fn usage_outcome(report: String, client_format: ClientFormat, stream: bool) -> Outcome {
    use crate::translate::sse::{format_data, format_event};
    if !stream {
        let body = match client_format {
            ClientFormat::Anthropic => json!({
                "id": "msg_pxy_usage", "type": "message", "role": "assistant",
                "model": "pxy",
                "content": [{"type": "text", "text": report}],
                "stop_reason": "end_turn", "stop_sequence": null,
                "usage": {"input_tokens": 0, "output_tokens": 0},
            }),
            ClientFormat::Openai => json!({
                "id": "chatcmpl_pxy_usage", "object": "chat.completion",
                "created": Timestamp::now().as_second(), "model": "pxy",
                "choices": [{"index": 0,
                    "message": {"role": "assistant", "content": report},
                    "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            }),
        };
        return Outcome::Json {
            status: 200,
            body,
            provider: Some("pxy".into()),
            headers: Vec::new(),
        };
    }
    let sse = match client_format {
        ClientFormat::Anthropic => {
            let mut s = String::new();
            s.push_str(&format_event("message_start", &json!({
                "type": "message_start",
                "message": {"id": "msg_pxy_usage", "type": "message", "role": "assistant",
                            "model": "pxy", "content": [],
                            "usage": {"input_tokens": 0, "output_tokens": 0}},
            })));
            s.push_str(&format_event("content_block_start", &json!({
                "type": "content_block_start", "index": 0,
                "content_block": {"type": "text", "text": ""},
            })));
            s.push_str(&format_event("content_block_delta", &json!({
                "type": "content_block_delta", "index": 0,
                "delta": {"type": "text_delta", "text": report},
            })));
            s.push_str(&format_event("content_block_stop",
                &json!({"type": "content_block_stop", "index": 0})));
            s.push_str(&format_event("message_delta", &json!({
                "type": "message_delta",
                "delta": {"stop_reason": "end_turn", "stop_sequence": null},
                "usage": {"output_tokens": 0},
            })));
            s.push_str(&format_event("message_stop", &json!({"type": "message_stop"})));
            s
        }
        ClientFormat::Openai => {
            let chunk = |delta: Value, finish: Value| {
                format_data(&json!({
                    "id": "chatcmpl_pxy_usage", "object": "chat.completion.chunk",
                    "created": Timestamp::now().as_second(), "model": "pxy",
                    "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
                }))
            };
            format!(
                "{}{}data: [DONE]\n\n",
                chunk(json!({"role": "assistant", "content": report}), Value::Null),
                chunk(json!({}), json!("stop")),
            )
        }
    };
    Outcome::Stream {
        provider: "pxy".into(),
        body: axum::body::Body::from(sse),
        headers: Vec::new(),
    }
}

/// Extra full-chain walks after the first one fails (3 walks total).
const MAX_RETRIES: u32 = 2;
/// Longest we'll hold a request waiting for a cooldown to expire. Past this,
/// fail fast — agents have their own retry logic and a better error message.
const MAX_RETRY_WAIT: Duration = Duration::from_secs(10);

/// Soonest a cooled-down candidate becomes eligible again, as a wait from now.
/// None when no candidate can recover by waiting: hard limits don't expire in
/// seconds, and non-retryable cooldowns (auth/credits) don't expire at all in
/// any sense worth re-firing a dead key over.
fn soonest_recovery(app: &App, candidates: &[Candidate]) -> Option<Duration> {
    candidates
        .iter()
        // state_provider, not provider: multi-account cooldowns live under
        // `provider#account` keys, and reading the bare name missed them.
        .filter_map(|c| app.state.recovery_wait(&c.state_provider(), &c.model.id))
        .min()
}

/// The Retry-After hint for the terminal 429: soonest moment ANY candidate's
/// cooldowns (retryable or not) will have expired. None when nothing is
/// cooling — then the walk failed for reasons no wait fixes.
fn soonest_cooldown_end(app: &App, candidates: &[Candidate]) -> Option<Duration> {
    candidates
        .iter()
        .filter_map(|c| app.state.cooldown_remaining(&c.state_provider(), &c.model.id))
        .min()
}

/// How long to sleep before re-walking the chain, or None to give up now.
fn retry_wait(soonest: Option<Duration>, saw_rpm_limit: bool) -> Option<Duration> {
    // An rpm window slides continuously; a couple of seconds frees capacity.
    let rpm_hint = saw_rpm_limit.then_some(Duration::from_secs(2));
    let wait = match (soonest, rpm_hint) {
        (Some(a), Some(b)) => a.min(b),
        (a, b) => a.or(b)?,
    };
    if wait > MAX_RETRY_WAIT {
        return None;
    }
    // Epsilon so the cooldown has actually expired when the re-walk checks.
    Some(wait + Duration::from_millis(250))
}

/// Filter stage: cooldown, rpm, daily/monthly limits, context window,
/// tool-calling capability.
/// Remaining fraction of every local allowance this candidate has (1.0 =
/// untouched), the MINIMUM across configured windows: the window closest to
/// exhaustion is the one that will actually stop the next request. No
/// configured limits means unconstrained -> 1.0.
fn remaining_headroom(cfg: &Config, state: &State, cand: &Candidate) -> f64 {
    let Some(limits) = cfg.providers.get(&cand.provider).and_then(|p| p.limits.as_ref()) else {
        return 1.0;
    };
    let mut used: Vec<f64> = Vec::new();
    if let Some(rpm) = limits.rpm.filter(|r| *r > 0) {
        used.push(state.rpm_effective(&cand.state_provider()) / rpm as f64);
    }
    if let Some(tpm) = limits.tpm.filter(|t| *t > 0) {
        used.push(state.tpm_effective(&cand.state_provider()) / tpm as f64);
    }
    if let Ok(w) = current_windows(limits, Timestamp::now()) {
        let day = state
            .usage(&cand.state_provider(), "day", w.day_start)
            .unwrap_or_default();
        let month = state
            .usage(&cand.state_provider(), "month", w.month_start)
            .unwrap_or_default();
        let mut frac = |v: u64, l: Option<u64>| {
            if let Some(l) = l.filter(|l| *l > 0) {
                used.push(v as f64 / l as f64);
            }
        };
        frac(day.requests, limits.daily_requests);
        frac(day.tokens, limits.daily_tokens);
        frac(month.requests, limits.monthly_requests);
        frac(month.tokens, limits.monthly_tokens);
    }
    if limits.total_requests.is_some() || limits.total_tokens.is_some() {
        let total = state.usage_total(&cand.state_provider()).unwrap_or_default();
        if let Some(l) = limits.total_requests.filter(|l| *l > 0) {
            used.push(total.requests as f64 / l as f64);
        }
        if let Some(l) = limits.total_tokens.filter(|l| *l > 0) {
            used.push(total.tokens as f64 / l as f64);
        }
    }
    (1.0 - used.into_iter().fold(0.0_f64, f64::max)).clamp(0.0, 1.0)
}

/// Is this failure genuine quota exhaustion, as opposed to a transient error?
/// `fallback_only_on_quota_exhaustion` uses it to decide whether a paid reserve
/// may be touched. A 402, or a 429 whose body names a quota window or credits
/// condition, is quota; an rpm throttle, a 5xx, a timeout or an auth error is
/// not — those clear on their own and must not spend paid credit.
fn is_quota_failure(status: Option<u16>, reason: &str) -> bool {
    match status {
        Some(402) => return true,
        Some(s) if s != 429 => return false,
        _ => {}
    }
    quota_window_cooldown(None, reason).is_some()
}

fn check_candidate(
    app: &App,
    cand: &Candidate,
    input_estimate: u64,
    wants_tools: bool,
    unservable_server_tools: bool,
    multi_candidate: bool,
) -> Result<(), String> {
    let provider = match app.cfg.providers.get(&cand.provider) {
        Some(p) if p.enabled => p,
        _ => return Err("provider disabled".into()),
    };

    // A curated/probed `tool_call = false` is a fact: routing a tools
    // request there burns the call and returns prose. Unknown stays
    // eligible (fail open), and an explicitly-addressed single model is
    // exempt like the cooldown filter — let the upstream answer for itself
    // rather than synthesizing a retryable 429 for a deterministic no.
    if multi_candidate && wants_tools && cand.model.tool_call == Some(false) {
        return Err("model cannot tool-call".into());
    }

    // Declared server tools pxy cannot fulfil on an OpenAI upstream are a
    // deterministic no there too (wiki:routing): the translator would drop
    // them SILENTLY, which is worse than skipping the candidate — the model
    // would answer without capabilities the harness is built around. Unlike
    // the tool_call rule, single-candidate is NOT exempt (handle_chat 400s it
    // before the walk): the drop happens in pxy's own translation, so the
    // upstream never gets to answer for itself.
    if multi_candidate && unservable_server_tools && cand.format(provider) != WireFormat::Anthropic {
        return Err("server tools unsupported on this upstream".into());
    }

    // Single-candidate requests skip the cooldown filter (litellm's
    // single-deployment exemption): blocking your only option converts a
    // partial outage into a total one.
    if multi_candidate {
        if let Some(cd) = app.state.cooldown(&cand.state_provider(), &cand.model.id) {
            return Err(format!("cooldown ({})", cd.reason));
        }
        // Failure-rate rule: a model that fails half its recent attempts sits
        // out even after each individual error cooldown expires (flapping
        // 200/500 upstreams never trip the per-error ladder).
        if app.state.model_unhealthy(&cand.state_provider(), &cand.model.id) {
            return Err("recent failure rate".into());
        }
    }

    if input_estimate > cand.model.context_length {
        return Err(format!(
            "context too large (~{input_estimate} > {})",
            cand.model.context_length
        ));
    }

    if let Some(limits) = &provider.limits {
        if let Some(rpm) = limits.rpm {
            if app.state.rpm_effective(&cand.state_provider()) >= rpm as f64 {
                return Err("rpm limit".into());
            }
        }
        if let Some(tpm) = limits.tpm {
            if app.state.tpm_effective(&cand.state_provider()) >= tpm as f64 {
                return Err("tpm limit".into());
            }
        }
        // Limit checks fail open on infrastructure errors (litellm rule):
        // a broken tzdb/db must never block routing.
        if let Ok(w) = current_windows(limits, Timestamp::now()) {
            let day = app.state.usage(&cand.state_provider(), "day", w.day_start).unwrap_or_default();
            let month = app
                .state
                .usage(&cand.state_provider(), "month", w.month_start)
                .unwrap_or_default();
            if let Some(l) = limits.daily_requests {
                if day.requests >= l {
                    return Err("daily request limit".into());
                }
            }
            if let Some(l) = limits.daily_tokens {
                if day.tokens >= l {
                    return Err("daily token limit".into());
                }
            }
            if let Some(l) = limits.monthly_requests {
                if month.requests >= l {
                    return Err("monthly request limit".into());
                }
            }
            if let Some(l) = limits.monthly_tokens {
                if month.tokens >= l {
                    return Err("monthly token limit".into());
                }
            }
        }
        if limits.total_requests.is_some() || limits.total_tokens.is_some() {
            let total = app.state.usage_total(&cand.state_provider()).unwrap_or_default();
            if let Some(l) = limits.total_requests {
                if total.requests >= l {
                    return Err("total request budget exhausted".into());
                }
            }
            if let Some(l) = limits.total_tokens {
                if total.tokens >= l {
                    return Err("total token budget exhausted".into());
                }
            }
        }
    }
    Ok(())
}

enum AttemptResult {
    Done(Outcome),
    /// Retryable/skippable failure: try the next candidate.
    Skip(String),
    /// Skip, but the upstream answered with a real error. The status and body
    /// are carried so a walk that ends with nothing better can hand the client
    /// the upstream's OWN answer instead of a synthetic one: Claude Code reads
    /// "usage limit reached, resets at …" out of that body, and its
    /// status-specific retry logic keys off the real type.
    SkipRaw { reason: String, status: u16, body: String, headers: Headers },
    /// Upstream 400'd because the input exceeds THIS model's real context
    /// window (our estimate under-counted): skip it and every candidate
    /// with the same or smaller window, no cooldown.
    SkipContextWindow(String),
    /// Fatal for the whole request: return this to the client.
    Fatal(Outcome),
}

/// Wrapper owning this attempt's capture transaction: artifacts share one id
/// and, under `on_error`, are written only when the attempt fails.
async fn try_candidate(
    app: &SharedApp,
    cand: &Candidate,
    client_format: ClientFormat,
    payload: &Value,
    stream: bool,
    input_estimate: u64,
    ctx: &ClientContext,
    multi: bool,
) -> AttemptResult {
    let mut cap = crate::capture::Txn::new(&app.cfg);
    let result = try_candidate_inner(
        app, &mut cap, cand, client_format, payload, stream, input_estimate, ctx, multi,
    )
    .await;
    cap.finish(matches!(result, AttemptResult::Done(_)));
    result
}

// The 9th argument is this attempt's capture transaction; threading a context
// struct through this ~600-line function would be churn for a lint.
#[allow(clippy::too_many_arguments)]
async fn try_candidate_inner(
    app: &SharedApp,
    cap: &mut crate::capture::Txn,
    cand: &Candidate,
    client_format: ClientFormat,
    payload: &Value,
    stream: bool,
    input_estimate: u64,
    ctx: &ClientContext,
    multi: bool,
) -> AttemptResult {
    let provider_cfg = app.cfg.providers.get(&cand.provider).unwrap();
    // A Responses upstream is an OpenAI upstream to everything in here; only
    // the wire differs, and that is handled at the send and the read.
    let wire = cand.format(provider_cfg);
    let responses_upstream = wire == WireFormat::Responses;
    let upstream_format = if responses_upstream { WireFormat::Openai } else { wire };

    // Build the upstream body.
    let mut body = match (client_format, upstream_format) {
        (ClientFormat::Openai, WireFormat::Openai) => payload.clone(),
        (ClientFormat::Anthropic, WireFormat::Anthropic) => payload.clone(),
        (ClientFormat::Anthropic, WireFormat::Openai) => {
            anthropic_to_openai::request(payload, provider_cfg.requires_reasoning_replay)
        }
        (ClientFormat::Openai, WireFormat::Anthropic) => openai_to_anthropic::request(
            payload,
            cand.model.max_output_tokens,
            provider_cfg.requires_reasoning_replay,
        ),
        (_, WireFormat::Responses) => unreachable!("responses wire is folded into openai"),
    };
    // Anthropic validates history strictly (thinking signatures, tool
    // pairing, empty blocks) and the passthrough path replays whatever the
    // client accumulated — repair it at the one choke point.
    if upstream_format == WireFormat::Anthropic {
        // The served-tool loop needs an OpenAI upstream, and no upstream
        // knows pxy's own `pxy:*` spelling: it would 400 here. The
        // `openrouter:*` spelling passes through, as it always has — the
        // upstream may be OpenRouter itself, which serves it natively.
        drop_hosted_server_tools(&mut body);
        crate::translate::anthropic_sanitize::sanitize(&mut body);
        // Prompt-cache breakpoints for clients whose dialect can't set them
        // after the sanitizer so markers land on blocks
        // that will actually be sent. Opt-in per provider: some gateways
        // 400 on the field, and relay must be proven before it is enabled.
        if provider_cfg.inject_cache_control {
            crate::translate::cache_control::inject(&mut body);
        }
    }
    body["model"] = json!(cand.model.id);
    // pxy consumes `max_tool_calls` itself (the loop's turn budget) and
    // `plugins` (wiki:plugins) the same way; neither is a Chat Completions
    // field, so neither may reach an upstream that would reject the unknown
    // key.
    if let Some(o) = body.as_object_mut() {
        o.remove("max_tool_calls");
        o.remove("plugins");
    }

    // OpenAI's newest dialect differs from the compatible-provider majority in
    // two request-body spellings, and the clients pxy fronts speak the newest
    // one: pi sends `developer` and `max_completion_tokens` for any reasoning
    // model because it cannot see the real upstream behind the proxy. The
    // upstreams mostly want `system` and `max_tokens` (DeepSeek 400s on the
    // `developer` variant; opencode-go ignored `max_completion_tokens`, so the
    // client's output cap silently did nothing). Only OpenAI/Azure itself opts
    // out with `openai_native`.
    if upstream_format == WireFormat::Openai {
        if !provider_cfg.openai_native {
            crate::translate::developer_role_to_system(&mut body);
        }
        crate::translate::normalize_max_tokens_field(&mut body, provider_cfg.openai_native);
        if provider_cfg.requires_reasoning_replay {
            crate::translate::backfill_openai_reasoning(&mut body);
        }
        crate::translate::sanitize_tool_schemas(&mut body, false);
    } else if provider_cfg.requires_reasoning_replay {
        // After the sanitizer, so the placeholder is not stripped as unsigned.
        crate::translate::backfill_anthropic_thinking(&mut body);
    }

    // A Chat Completions client has no translator pair (OpenAI -> OpenAI is
    // payload.clone()), so its body still carries a declared server tool as a
    // hosted entry (`{"type":"openrouter:web_search"}`). Swap each for the
    // reserved function pxy intercepts, as the Anthropic and Responses
    // translators already did for their bodies.
    if upstream_format == WireFormat::Openai {
        swap_declared_served_tools(&mut body);
    }

    // A candidate that cannot take an image part is sent `[image N]` in its
    // place: being told an image is there beats a 400, and describe_image
    // resolves the number back to a URL. Before the loop copies the body
    // below, so a continuation replays the placeholders, and only at depth 0 —
    // the describe_image leg itself must still carry its image.
    let images = if cand.model.vision == Some(false) && ctx.tool_depth == 0 {
        server_tools::swap_image_parts(&mut body)
    } else {
        Vec::new()
    };

    // Hosted server tools pxy runs itself. An OpenAI upstream has no hosted
    // tools of its own, so whoever built the body — anthropic_to_openai for
    // Claude Code's server tool, responses for `codex --search` — injected a
    // reserved pxy_* function for each declared tool.
    //
    // Those translators inject on their own dialect's evidence; only here is
    // it known whether pxy can actually SERVE the call: the interception
    // lives in StreamCtx, so it needs an OpenAI upstream, the tool offered on
    // this dialect, and whatever executor the tool needs. A non-streaming
    // client is served by streaming upstream anyway (force_stream below) and
    // re-assembling its JSON from the translated stream: a non-streaming turn
    // must not silently lose a tool it asked for.
    let injected: Vec<server_tools::Tool> = server_tools::Tool::implemented()
        .iter()
        .copied()
        .filter(|t| reserved_function_in_body(&body, *t))
        .collect();
    let servable: Vec<server_tools::Tool> = if upstream_format == WireFormat::Openai {
        injected
            .iter()
            .copied()
            .filter(|t| t.servable(app))
            // A sub-request may not re-enter a meta-tool: at depth 1 the
            // advisor and subagent are dropped like a tool pxy cannot serve.
            .filter(|t| !(ctx.tool_depth >= 1 && t.is_meta()))
            // A model asserted unable to tool-call is offered nothing: the
            // walk only skips such a model when the CLIENT declared tools,
            // and a config default must not fail a toolless turn there.
            .filter(|_| cand.model.tool_call != Some(false))
            .collect()
    } else {
        Vec::new()
    };
    let search_served = !servable.is_empty();

    // The client's deferred tools are held back while pxy is offering the
    // search that can reveal them again, and pxy's own `defer_loading` key
    // comes off either way. Before `settle_tool_choice` below, which would
    // delete a choice naming a tool that had just left the body.
    let deferred = defer_loading_tools(
        &mut body,
        payload,
        client_format,
        upstream_format,
        servable.contains(&server_tools::Tool::ToolSearch),
    );

    if !injected.is_empty() {
        // Offering a function nobody will intercept hands the client a
        // tool_use for a tool it never declared, which wedges the turn. Drop
        // every reserved function pxy will not answer, and let the model work
        // without it.
        let keep: Vec<String> =
            servable.iter().map(|t| server_tools::function_name(*t)).collect();
        if let Some(tools) = body.get_mut("tools").and_then(|t| t.as_array_mut()) {
            tools.retain(|t| match t["function"]["name"].as_str() {
                Some(n) if server_tools::Tool::from_function_name(n).is_some() => {
                    keep.iter().any(|k| k == n)
                }
                _ => true,
            });
            if tools.is_empty() {
                body.as_object_mut().map(|o| o.remove("tools"));
            }
        }
        settle_tool_choice(&mut body);
        for tool in injected.iter().filter(|t| !servable.contains(t)) {
            debug!(
                candidate = %cand.full_id(),
                tool = tool.name(),
                "server tool dropped: pxy cannot serve it here"
            );
        }
    }

    // Anthropic's API injects a memory protocol into the system prompt
    // whenever the memory tool is declared, and a model trained on the tool
    // waits to be told to read its directory first. Only here is it known that
    // the reserved function actually survived on the body, and it goes in
    // before the loop copies the body, so a continuation carries it too.
    if servable.contains(&server_tools::Tool::Memory) {
        append_system_text(&mut body, server_tools::MEMORY_PROTOCOL);
    }

    // force_stream: the upstream misbehaves without `stream: true` on this
    // model — or a served web_search needs the stream machinery — so stream
    // upstream regardless and re-assemble JSON for a non-streaming client.
    let force_stream = !stream && (cand.model.force_stream || search_served);
    if force_stream {
        body["stream"] = json!(true);
    }
    if (stream || force_stream)
        && upstream_format == WireFormat::Openai
        && body.get("stream_options").is_none()
    {
        // Ask OpenAI upstreams to report usage in the final chunk.
        body["stream_options"] = json!({"include_usage": true});
    }

    // An effort the model does not take is a 400 for a request that was
    // otherwise fine (agentrouter's glm-5.3 refuses "medium"), so it is moved
    // to the nearest level the model lists. Only the OpenAI wire carries a
    // named level; an Anthropic upstream takes a token budget instead.
    if let Some(level) = body["reasoning_effort"].as_str()
        && let Some(clamped) = clamp_effort(level, &cand.model.effort)
    {
        info!(candidate = %cand.full_id(), from = level, to = %clamped, "reasoning effort clamped");
        body["reasoning_effort"] = json!(clamped);
    }

    // Keys this upstream 400s on, dropped LAST so they win over anything the
    // translation or the stream_options default put in the body. `model` and
    // `stream` are pxy's own routing keys — dropping them wouldn't disable a
    // param, it would corrupt the request pxy itself built, so they're
    // ignored here.
    if !provider_cfg.drop_params.is_empty() || !cand.model.drop_params.is_empty() {
        for key in provider_cfg.drop_params.iter().chain(&cand.model.drop_params) {
            if key == "model" || key == "stream" {
                warn!(candidate = %cand.full_id(), key, "drop_params ignores pxy's own key");
                continue;
            }
            remove_param_path(&mut body, key);
        }
    }

    // This candidate's configured account (multi-account providers expand
    // into one candidate per account at resolve time).
    let account = cand.account.as_ref().and_then(|name| {
        provider_cfg
            .accounts
            .iter()
            .find(|a| &a.name == name)
    });
    let mut prepared =
        match crate::providers::prepare(&cand.provider, provider_cfg, &app.secrets, account) {
            Ok(p) => p,
            Err(e) => return AttemptResult::Skip(format!("prepare failed: {e:#}")),
        };
    // One provider, two endpoints: a model on the Responses wire lives next to
    // the chat one (opencode-go's /zen/go/v1/{chat/completions,responses}).
    if responses_upstream {
        if let Some(base) = prepared.url.strip_suffix("/chat/completions") {
            prepared.url = format!("{base}/responses");
        }
    }

    // Tool names as the client declared them: some upstreams (Gemini, several
    // gateways) lowercase them, and a client matches a call by name.
    let declared_names = declared_tool_names(payload);
    // Textual tool-call extraction: OpenAI upstreams only, and only when the
    // request declared tools (otherwise the markup is content, not protocol).
    let tool_names = (upstream_format == WireFormat::Openai)
        .then(|| declared_names.clone())
        .flatten();

    // The total timeout must NOT apply to client-streaming requests: it
    // spans "connect until the body finishes" (reqwest semantics), so a long
    // agentic turn dies mid-stream at timeout_secs with a clean-looking
    // end_turn. Streams are bounded instead by the headers deadline below
    // and a per-read stall deadline in the unfold. Non-streaming keeps the
    // total timeout (it also bounds force_stream's re-assembly).
    // opencode.ai routes on `x-opencode-session` and, from 2026-09-06, errors
    // without it. Client's own id wins; otherwise fingerprint the conversation
    // so the value is stable across the turns of one session. Scoped to
    // opencode upstreams — this identifies a conversation and is nobody
    // else's business. It lives in the header list, not on this one request,
    // so a served tool's continuation carries it too.
    let mut upstream_headers = prepared.headers.clone();
    if is_opencode_provider(&cand.provider)
        && !upstream_headers.iter().any(|(k, _)| k.eq_ignore_ascii_case("x-opencode-session"))
    {
        let session =
            ctx.session.clone().unwrap_or_else(|| conversation_fingerprint(payload));
        upstream_headers.push(("x-opencode-session".to_string(), session));
    }
    let mut req = app.http.post(&prepared.url).header("content-type", "application/json");
    if !stream {
        req = req.timeout(Duration::from_secs(provider_cfg.timeout_secs));
    }
    for (k, v) in &upstream_headers {
        req = req.header(k, v);
    }
    if upstream_format == WireFormat::Anthropic {
        if let Some(beta) = &ctx.anthropic_beta {
            req = req.header("anthropic-beta", beta.clone());
        }
        if !prepared.headers.iter().any(|(k, _)| k == "anthropic-version") {
            req = req.header("anthropic-version", "2023-06-01");
        }
    }

    // The server-tool loop's replay request: built before the send so the
    // follow-up call replays this exact request with the results appended.
    // The tool reaches the wire only if pxy will intercept it — that is the
    // invariant this site enforces (the strip below is its other half).
    let search = search_served.then(|| ServerToolLoop {
        filter: ServerCallFilter::default(),
        uses_left: turn_budget(payload, &app.cfg.server_tools),
        per_tool: servable
            .iter()
            .map(|t| (server_tools::function_name(*t), t.max_uses(payload)))
            .collect(),
        params: servable
            .iter()
            .map(|t| {
                (server_tools::function_name(*t), server_tools::declared_parameters(payload, *t))
            })
            .collect(),
        url: prepared.url.clone(),
        headers: upstream_headers.clone(),
        body: body.clone(),
        timeout: Duration::from_secs(provider_cfg.timeout_secs),
        responses_upstream,
        agent: ctx.agent.clone().unwrap_or_default(),
        session: ctx.session.clone(),
        outer_model: payload["model"]
            .as_str()
            .filter(|m| !m.is_empty())
            .map(str::to_string)
            .unwrap_or_else(|| app.cfg.default_route()),
        served: std::collections::BTreeMap::new(),
        results_served: 0,
        deferred,
        images,
    });

    app.state.rpm_increment(&cand.state_provider());
    // A streaming upstream returns headers as soon as it accepts the request,
    // so silence here means it is not answering at all. With a chain to fall
    // back on, waiting out timeout_secs (600s by default) for that is the
    // wrong trade — one dead provider at the head of a group would strand
    // every request. An explicitly named model still gets the full timeout:
    // there is nothing to fail over to. Non-streaming is exempt because its
    // headers legitimately arrive only once the whole answer is generated.
    // Opt-in artifact capture: the client request and the exact upstream
    // request body, before the wire.
    cap.add("client-request", payload);
    let wire_body;
    let send_body: &Value = if responses_upstream {
        wire_body = responses_upstream::request(&body);
        &wire_body
    } else {
        &body
    };
    cap.add(
        "upstream-request",
        &json!({"candidate": cand.full_id(), "url": &prepared.url, "body": send_body}),
    );
    let send = req.json(send_body).send();
    let resp = if multi && (stream || force_stream) {
        match tokio::time::timeout(HEADERS_DEADLINE, send).await {
            Ok(r) => r,
            Err(_) => {
                // Provider-scoped, like a network error: an endpoint that
                // accepts the connection and then says nothing is not
                // answering for ANY of its models, and the escalating default
                // starts at 3s — long gone by the next candidate from the same
                // provider, so the walk would pay the deadline again and again.
                app.state.set_cooldown(
                    &cand.state_provider(),
                    None,
                    Some(HEADERS_COOLDOWN),
                    true,
                    "no response headers",
                );
                return AttemptResult::Skip(format!(
                    "no response after {}s",
                    HEADERS_DEADLINE.as_secs()
                ));
            }
        }
    } else if stream {
        // Single-candidate stream: nothing to fail over to, so keep today's
        // full timeout_secs wait for headers (the per-request total timeout
        // that used to bound this phase is gone). Once headers arrive, the
        // body streams without a total bound.
        match tokio::time::timeout(Duration::from_secs(provider_cfg.timeout_secs), send).await {
            Ok(r) => r,
            Err(_) => {
                app.state.set_cooldown(&cand.state_provider(), None, None, true, "network error");
                return AttemptResult::Skip(format!(
                    "no response after {}s",
                    provider_cfg.timeout_secs
                ));
            }
        }
    } else {
        send.await
    };
    let resp = match resp {
        Ok(r) => r,
        Err(e) => {
            // Network failures are our-side/transport, not model-specific.
            app.state.set_cooldown(&cand.state_provider(), None, None, true, "network error");
            return AttemptResult::Skip(format!("network: {e}"));
        }
    };

    let status = resp.status().as_u16();
    // Captured before the body is consumed; relayed on success, on a raw error
    // passthrough, and on the terminal error of an exhausted single-candidate
    // walk — `retry-after` and the rate-limit families are what the client's
    // own backoff reads.
    let fwd_headers = forwardable_headers(resp.headers());
    if status >= 400 {
        let retry_after = parse_retry_after(resp.headers());
        let err_body = resp.text().await.unwrap_or_default();
        return classify_error(
            app,
            cand,
            client_format,
            UpstreamError { status, retry_after, body: err_body, headers: fwd_headers },
            multi,
        );
    }

    // Success: count the request now; tokens follow when usage is known.
    let agent = ctx.agent.as_deref().unwrap_or("");
    record_request(app, agent, &cand.state_provider(), &cand.provider, &cand.model.id);
    app.state.clear_cooldown(&cand.state_provider(), &cand.model.id);
    // After the clear: a success response can still carry "you just used the
    // last of your quota" headers, and that cooldown must survive it.
    check_quota_exhaustion(&app.state, &cand.state_provider(), resp.headers());
    record_allowances(&app.state, &cand.state_provider(), resp.headers());
    if !stream && search.is_none() {
        info!(candidate = %cand.full_id(), stream, "routed");
    }

    if stream || search.is_some() {
        // A 200 status is not a commitment yet: hold the response until the
        // upstream produces a real first event, so a stream that dies before
        // saying anything fails over instead of reaching the client truncated.
        match stream_outcome(
            app.clone(),
            agent,
            cand,
            client_format,
            upstream_format,
            resp,
            input_estimate,
            tool_names,
            declared_names.clone(),
            search,
            // Cloned: the pre-first-event error path below still needs them to
            // relay `retry-after` on a stream that died before committing.
            fwd_headers.clone(),
            ctx.responses,
            responses_upstream,
        )
        .await
        {
            Ok(outcome) if stream => AttemptResult::Done(outcome),
            // Non-streaming client whose web_search pxy is serving: the loop
            // ran inside the stream machinery; re-assemble the client
            // dialect's JSON from the translated stream.
            Ok(outcome) => AttemptResult::Done(collect_stream_json(outcome, client_format).await),
            Err(StreamFailure::ErrorEvent(data)) => {
                // The stream's first event was the real error: classify it
                // exactly like an HTTP error status would have been, so a
                // fatal 4xx still passes through unmodified instead of
                // becoming a retry storm plus a synthetic 429.
                match error_event_status(&data) {
                    Some(status) => {
                        classify_error(
                            app,
                            cand,
                            client_format,
                            UpstreamError {
                                status,
                                retry_after: None,
                                body: data,
                                headers: fwd_headers,
                            },
                            multi,
                        )
                    }
                    None => {
                        app.state.set_cooldown(
                            &cand.state_provider(),
                            Some(&cand.model.id),
                            None,
                            true,
                            "stream error event",
                        );
                        AttemptResult::Skip(format!("stream error event: {}", truncate(&data, 200)))
                    }
                }
            }
            Err(StreamFailure::Dead(reason)) => {
                // Same scope as a 5xx: the model misbehaved, not the account.
                app.state.set_cooldown(
                    &cand.state_provider(),
                    Some(&cand.model.id),
                    None,
                    true,
                    "stream died before first event",
                );
                AttemptResult::Skip(reason)
            }
        }
    } else {
        let mut upstream_body: Value = if force_stream {
            // Collect the whole upstream stream, then re-assemble the JSON
            // response the client actually asked for.
            let bytes = match resp.bytes().await {
                Ok(b) => b,
                Err(e) => {
                    // Cool the model instead of re-probing a
                    // 200-then-truncate upstream for free.
                    app.state.set_cooldown(
                        &cand.state_provider(),
                        Some(&cand.model.id),
                        None,
                        true,
                        "stream read",
                    );
                    return AttemptResult::Skip(format!("stream read: {e}"));
                }
            };
            let mut parser = SseParser::new();
            let mut events = parser.feed(&bytes);
            // Flush a final event the upstream didn't terminate with \n\n.
            events.extend(parser.feed(b"\n\n"));
            if responses_upstream {
                events = responses_upstream::chat_events(&events);
            }
            match upstream_format {
                WireFormat::Openai | WireFormat::Responses => {
                    crate::translate::aggregate::openai(&events)
                }
                WireFormat::Anthropic => crate::translate::aggregate::anthropic(&events),
            }
        } else {
            match resp.json().await {
                Ok(v) if responses_upstream => responses_upstream::response(&v),
                Ok(v) => v,
                Err(e) => {
                    // A 200 with an unparseable body counts as a request and
                    // already cleared cooldowns above — without this cooldown
                    // a garbage-200 upstream would be re-attempted first on
                    // every walk, burning counted requests with no backoff.
                    app.state.set_cooldown(
                        &cand.state_provider(),
                        Some(&cand.model.id),
                        None,
                        true,
                        "unparseable 200",
                    );
                    return AttemptResult::Skip(format!("bad upstream json: {e}"));
                }
            }
        };
        if provider_cfg.parse_think_tags && upstream_format == WireFormat::Openai {
            extract_think_from_response(&mut upstream_body);
        }
        if let Some(names) = &tool_names {
            crate::translate::tool_text::extract_from_response(&mut upstream_body, names);
        }
        cap.add("upstream-response", &upstream_body);
        let usage = match upstream_format {
            WireFormat::Openai | WireFormat::Responses => {
                TokenUsage::from_openai(&upstream_body["usage"])
            }
            WireFormat::Anthropic => TokenUsage::from_anthropic(&upstream_body["usage"]),
        };
        record_tokens(app, agent, &cand.state_provider(), &cand.provider, &cand.model.id, usage);
        let client_body = match (client_format, upstream_format) {
            (ClientFormat::Openai, WireFormat::Openai) => {
                let mut b = upstream_body;
                crate::translate::strip_foreign_response_fields(&mut b);
                b
            }
            (ClientFormat::Anthropic, WireFormat::Anthropic) => upstream_body,
            (ClientFormat::Anthropic, WireFormat::Openai) => {
                anthropic_to_openai::response(&upstream_body, &cand.full_id(), declared_names.as_ref())
            }
            (ClientFormat::Openai, WireFormat::Anthropic) => {
                openai_to_anthropic::response(&upstream_body, &cand.full_id(), declared_names.as_ref())
            }
            (_, WireFormat::Responses) => unreachable!("responses wire is folded into openai"),
        };
        AttemptResult::Done(Outcome::Json {
            status: 200,
            body: client_body,
            provider: Some(cand.full_id()),
            headers: fwd_headers,
        })
    }
}

/// litellm's cascade, collapsed to our three-way split:
/// retryable (408/409/429/5xx) and auth (401/403) -> skip candidate;
/// everything else 4xx -> fatal, pass upstream error through unmodified
/// (Claude Code's auto-retry needs the raw body).
///
/// Exception: on a multi-candidate walk (a group), a 404 also skips. Free
/// model lists churn, and one delisted id must not kill the whole chain
/// (zenmux delisting glm-5.3-free took the chain down, 2026-08-25). On a
/// single-candidate request the 404 still passes through raw.
/// Remove a (possibly dotted) key path from a JSON object, pruning parents
/// the removal left empty — `thinking.budget_tokens` must not leave a bare
/// `{"thinking":{}}` behind, which is itself a 400 on some upstreams.
fn remove_param_path(body: &mut Value, path: &str) {
    match path.split_once('.') {
        None => {
            if let Some(obj) = body.as_object_mut() {
                obj.remove(path);
            }
        }
        Some((head, rest)) => {
            let Some(obj) = body.as_object_mut() else { return };
            let Some(child) = obj.get_mut(head) else { return };
            remove_param_path(child, rest);
            if child.as_object().is_some_and(|c| c.is_empty()) {
                obj.remove(head);
            }
        }
    }
}

/// A 429 body that names a QUOTA WINDOW gets a cooldown sized to that
/// window instead of the 3s→120s exponential backoff — re-probing a drained
/// daily tier every two minutes until midnight is hundreds of wasted calls
/// (and on some providers failed calls count against quota too).
///
/// Deliberately conservative (each exclusion is an OmniRoute post-mortem):
/// a bare "quota"/"exhausted" is NOT enough — Gemini's transient free-tier
/// 429 says "You exceeded your current quota" and "Resource has been
/// exhausted" for what is an rpm throttle. A window word (or an unambiguous
/// credits phrase) must be present.
fn quota_window_cooldown(
    limits: Option<&crate::config::Limits>,
    body: &str,
) -> Option<(Duration, &'static str)> {
    let b = body.to_lowercase();
    let has = |words: &[&str]| words.iter().any(|w| b.contains(w));
    // A quota-ish signal must accompany the window word ("daily" alone in
    // prose must not trip this).
    if !has(&["quota", "limit", "exceed", "exhaust", "allocation", "credit", "balance", "insufficient"]) {
        return None;
    }
    if has(&["per month", "monthly", "this month", "billing cycle"]) {
        // Never lock a whole month on a text match; recheck within hours.
        return Some((Duration::from_secs(6 * 3600), "monthly quota reported exhausted"));
    }
    if has(&["per week", "weekly", "this week"]) {
        return Some((Duration::from_secs(4 * 3600), "weekly quota reported exhausted"));
    }
    if has(&["per hour", "hourly", "-hour", "rolling"]) {
        // A rolling window (opencode Go's 5h) drains as it goes; recheck
        // well inside it. The Retry-After header, when sent, wins anyway.
        return Some((Duration::from_secs(30 * 60), "hourly quota reported exhausted"));
    }
    if has(&["per day", "daily", "today", "free allocation"]) {
        // Until the provider's next daily reset (+2 min margin). Fail open
        // to a 6h recheck if the window computation errors.
        let wait = limits
            .and_then(|l| {
                let now = Timestamp::now();
                let w = crate::usage::current_windows(l, now).ok()?;
                let secs = w.day_start.as_second() + 86_400 - now.as_second() + 120;
                u64::try_from(secs).ok()
            })
            .map(|s| s.clamp(900, 26 * 3600))
            .unwrap_or(6 * 3600);
        return Some((Duration::from_secs(wait), "daily quota reported exhausted"));
    }
    if has(&[
        "insufficient credits",
        "insufficient balance",
        "out of credits",
        "credits exhausted",
        "insufficient promotional resources",
    ]) {
        return Some((Duration::from_secs(3600), "credits reported exhausted"));
    }
    // A usage limit with no window named (opencode's FreeUsageLimitError
    // body, "usage limit reached"): an allowance, not a rate, so an hour.
    if has(&["usagelimiterror", "usage limit reached"]) {
        return Some((Duration::from_secs(3600), "usage limit reported reached"));
    }
    None
}

/// Upstream told us the input exceeds the model's real context window.
/// Substring set from litellm's ExceptionCheckers (nine phrasings across
/// OpenAI/Anthropic/Gemini dialects) with its two known false positives
/// excluded. Only consulted for 400-class responses.
fn is_context_window_error(body: &str) -> bool {
    let b = body.to_lowercase();
    // OpenAI uses this code for an over-long single STRING field, and the
    // "invalid 'user'" param error contains "maximum length" — neither is
    // a context-window condition.
    if b.contains("string_above_max_length") || b.contains("invalid 'user'") {
        return false;
    }
    [
        "maximum context length",
        "context length exceeded",
        "context_length_exceeded",
        "context window",
        "prompt is too long",
        "input is too long",
        "input tokens exceed",
        "exceeds the maximum number of tokens",
        "too many total text bytes",
    ]
    .iter()
    .any(|needle| b.contains(needle))
}

fn classify_error(
    app: &App,
    cand: &Candidate,
    client_format: ClientFormat,
    err: UpstreamError,
    multi: bool,
) -> AttemptResult {
    let UpstreamError { status, retry_after, body: err_body, headers: fwd_headers } = err;
    // Context-window 400s fail over on multi-candidate walks: the estimate
    // under-counted for THIS model, but larger-window candidates further
    // down the chain can still serve the request. Single-model requests
    // get the raw 400 (nothing to fail over to, body passes through below).
    // 429s are deliberately excluded — token-ish wording there is TPM rate
    // limiting, which the ordinary skip path already handles.
    if multi && matches!(status, 400 | 413 | 422) && is_context_window_error(&err_body) {
        return AttemptResult::SkipContextWindow(format!(
            "{status}: {}",
            truncate(&err_body, 200)
        ));
    }

    // Request-scoped error rules: per-provider body overrides that beat the
    // status ladder. First matching rule wins (case-insensitive substring).
    let rule = app.cfg.providers.get(&cand.provider).and_then(|p| {
        let lower = err_body.to_ascii_lowercase();
        p.errors
            .iter()
            .find(|r| !r.matches.is_empty() && lower.contains(&r.matches.to_ascii_lowercase()))
    });
    if let Some(rule) = rule {
        let reason = format!("error rule: {}", truncate(&err_body, 200));
        let cool = || {
            app.state.set_cooldown(
                &cand.provider,
                Some(&cand.model.id),
                None,
                true,
                "error rule match",
            );
        };
        return match rule.action {
            ErrorAction::Skip => AttemptResult::Skip(reason),
            ErrorAction::SkipCooldown => {
                cool();
                AttemptResult::Skip(reason)
            }
            ErrorAction::Passthrough => {
                passthrough_outcome(client_format, status, &err_body, &cand.full_id(), fwd_headers)
            }
            ErrorAction::PassthroughCooldown => {
                cool();
                passthrough_outcome(client_format, status, &err_body, &cand.full_id(), fwd_headers)
            }
        };
    }

    // 402 included: aggregators (ZenMux, OpenRouter, DeepSeek) use it for
    // exhausted quota/credits — an account problem, not a request problem.
    let skip = matches!(status, 401 | 402 | 403 | 408 | 409 | 429)
        || status >= 500
        || (multi && status == 404);
    if skip {
        let reason = match status {
            401 | 403 => "auth error",
            402 => "quota/credits exhausted",
            404 => "model not found upstream",
            429 => "rate limited",
            _ => "upstream error",
        };
        let limits = app.cfg.providers.get(&cand.provider).and_then(|p| p.limits.as_ref());
        // A 429 whose body names a spent allowance (a daily/weekly/monthly
        // window, credits, a usage limit) is the account's quota, not this
        // model's rate: every model on the account is out until the reset.
        let quota = (status == 429)
            .then(|| quota_window_cooldown(limits, &err_body))
            .flatten();
        // Account-wide problems cool the whole provider (per account on a
        // multi-account provider); plain rate limits and upstream errors are
        // usually per-model on aggregators, so they must not sideline the
        // provider's other models.
        let account_wide = matches!(status, 401 | 402 | 403) || quota.is_some();
        let model_scope = (!account_wide).then_some(cand.model.id.as_str());
        // Auth/credit failures and delisted models don't heal in seconds, so
        // the retry loop must not wait on them (or re-fire a dead key).
        let retryable = !account_wide && status != 404;
        // Header wins for the wait; the body's window names the reason. Else
        // the window's own horizon; else a 402 without any hint waits an
        // hour (credits don't reappear in 120s); else the ordinary
        // exponential backoff.
        let why = match &quota {
            Some((_, why)) => format!("429 {why}"),
            None => format!("{status} {reason}"),
        };
        let (wait, retryable) = match (retry_after, &quota) {
            (Some(d), _) => (Some(d), retryable),
            (None, Some((d, _))) => (Some(*d), false),
            (None, None) if status == 402 => (Some(Duration::from_secs(3600)), false),
            (None, None) => (None, retryable),
        };
        app.state.set_cooldown(&cand.state_provider(), model_scope, wait, retryable, &why);
        return AttemptResult::SkipRaw {
            reason: format!("{status}: {}", truncate(&err_body, 200)),
            status,
            body: err_body,
            headers: fwd_headers,
        };
    }
    passthrough_outcome(client_format, status, &err_body, &cand.full_id(), fwd_headers)
}

/// Return the upstream error to the client unmodified (the upstream's JSON
/// when it parses, pxy's error shape otherwise). Claude Code's auto-retry
/// depends on unmodified error bodies — this is also what `passthrough`
/// error rules resolve to.
/// The upstream's own error, handed to the client unchanged. Its body is
/// reused verbatim when it parses as JSON — Claude Code's retry and limit UI
/// read the real `type`/`message` out of it.
fn passthrough_json(
    client_format: ClientFormat,
    status: u16,
    err_body: &str,
    full_id: &str,
    headers: Headers,
) -> Outcome {
    let body = serde_json::from_str::<Value>(err_body)
        .unwrap_or_else(|_| error_body(client_format, "api_error", &truncate(err_body, 500)));
    Outcome::Json { status, body, provider: Some(full_id.to_string()), headers }
}

fn passthrough_outcome(
    client_format: ClientFormat,
    status: u16,
    err_body: &str,
    full_id: &str,
    headers: Headers,
) -> AttemptResult {
    AttemptResult::Fatal(passthrough_json(client_format, status, err_body, full_id, headers))
}

/// Is this an opencode.ai upstream? Their gateway wants an `x-opencode-session`
/// on every request (from 2026-09-06 requests without one "may error"), and it
/// is the key their side routes on — same session means the same upstream
/// provider, which is what makes prompt caching hit across a conversation.
///
/// Matched on the provider NAME, which is exactly the rule the opencode CLI
/// itself uses to decide whether to emit the header set. The scope matters:
/// the id fingerprints the conversation, so it must not be handed to unrelated
/// third-party upstreams.
fn is_opencode_provider(provider: &str) -> bool {
    provider.starts_with("opencode")
}

/// A conversation-stable session id derived from the request itself, for
/// clients that send none of their own (claude, codex, fx …).
///
/// Deliberately NOT random per request: opencode routes on this value, so a
/// fresh id every turn would scatter one conversation across upstreams and
/// lose every prompt-cache hit — the exact bug OmniRoute #10571 fixed. Only
/// fields that survive a conversation are mixed in: the model, the system
/// prompt, the FIRST user message, and the tool names. Later turns append
/// messages, so anything reading the whole history would change every turn.
///
/// FNV-1a rather than a crypto hash: this is an opaque grouping key, not a
/// security boundary, and 64 bits is plenty to keep distinct conversations
/// apart. Hand-rolled because it must stay stable across restarts and Rust
/// versions, which `DefaultHasher` explicitly does not promise.
fn conversation_fingerprint(payload: &Value) -> String {
    fn fnv1a(bytes: &[u8], mut h: u64) -> u64 {
        for b in bytes {
            h ^= *b as u64;
            h = h.wrapping_mul(0x0000_0100_0000_01b3);
        }
        h
    }
    // Flatten a string-or-content-array field to the text it carries.
    fn text_of(v: &Value) -> String {
        match v {
            Value::String(s) => s.clone(),
            Value::Array(parts) => parts
                .iter()
                .filter_map(|p| p["text"].as_str().or_else(|| p.as_str()))
                .collect::<Vec<_>>()
                .join(" "),
            _ => String::new(),
        }
    }

    let mut h = 0xcbf2_9ce4_8422_2325;
    if let Some(model) = payload["model"].as_str() {
        h = fnv1a(model.as_bytes(), h);
    }
    // Anthropic carries the system prompt beside the messages; OpenAI puts it
    // in the first message. Mix in whichever exists.
    h = fnv1a(text_of(&payload["system"]).as_bytes(), h);
    if let Some(msgs) = payload["messages"].as_array() {
        for role in ["system", "user"] {
            if let Some(m) = msgs.iter().find(|m| m["role"] == role) {
                h = fnv1a(text_of(&m["content"]).as_bytes(), h);
            }
        }
    }
    // Tool names, sorted: the tool set identifies the harness, and its wire
    // order is not stable across turns.
    if let Some(tools) = payload["tools"].as_array() {
        let mut names: Vec<&str> = tools
            .iter()
            .filter_map(|t| t["name"].as_str().or_else(|| t["function"]["name"].as_str()))
            .collect();
        names.sort_unstable();
        h = fnv1a(names.join(",").as_bytes(), h);
    }
    format!("pxy_{h:016x}")
}

/// kv key holding a provider's last-seen rolling-allowance snapshot.
pub fn free_quota_key(provider: &str) -> String {
    format!("free_quota:{provider}")
}

/// kv key for the PAID plan's allowance, metered independently of the free one
/// (tokenharbor's Agent Pass runs alongside the free window, not instead of it).
pub fn plan_quota_key(provider: &str) -> String {
    format!("plan_quota:{provider}")
}

/// TokenHarbor meters its allowances as personal rolling 4-week windows, priced
/// by the list-price value of the work — pxy cannot compute that, and there is
/// no balance endpoint to poll (every /v1/usage-shaped path 404s). The only
/// readout is a set of undocumented headers on a successful completion, so
/// remember the last one: `pxy status --remote` reports the snapshot with its
/// age instead of a number nobody can fetch.
///
/// There are TWO independent windows and a response carries whichever one it
/// drew on: `x-th-free-*` on the free lineup, `x-th-plan-*` on the models a
/// paid pass unlocks (`x-th-plan` names the pass — "free", "agent", …). They
/// are snapshotted under separate keys so both show up side by side.
///
/// At 100% used the provider is cooled until the window actually rolls. That
/// is deliberately provider-wide and deliberately BLUNT: past the included
/// usage TokenHarbor does not stop, it falls through to pay-as-you-go (the
/// Agent Pass advertises "then 5% off pay-as-you-go"), so an exhausted window
/// is the moment calls start costing real money. Cooling the whole provider
/// is the billing guard; it is over-broad in one direction — a spent free
/// window also parks the paid models, which still have their own allowance —
/// and that is the safe direction to be wrong in.
fn record_allowances(state: &State, provider: &str, headers: &reqwest::header::HeaderMap) {
    let windows = [("free", free_quota_key(provider)), ("plan", plan_quota_key(provider))];
    for (family, key) in windows {
        record_allowance(state, provider, headers, family, &key);
    }
}

fn record_allowance(
    state: &State,
    provider: &str,
    headers: &reqwest::header::HeaderMap,
    family: &str,
    key: &str,
) {
    let get = |name: &str| headers.get(name).and_then(|v| v.to_str().ok()).map(str::trim);
    let Some(pct) = get(&format!("x-th-{family}-used-pct"))
        .and_then(|s| s.trim_end_matches('%').trim().parse::<f64>().ok())
    else {
        return;
    };
    let resets = get(&format!("x-th-{family}-resets")).unwrap_or_default();
    let plan = get("x-th-plan").unwrap_or_default();
    let _ = state.kv_set(
        key,
        &json!({
            "usedPct": pct,
            "resetsAt": resets,
            "plan": plan,
            "observedAt": Timestamp::now().to_string(),
        })
        .to_string(),
    );
    if pct >= 100.0 {
        let wait = resets
            .parse::<Timestamp>()
            .ok()
            .map(|t| t.as_second() - Timestamp::now().as_second())
            .filter(|secs| *secs > 0)
            .map(|secs| Duration::from_secs(secs as u64))
            .unwrap_or(Duration::from_secs(3600));
        warn!(
            provider,
            family,
            pct,
            wait_secs = wait.as_secs(),
            "upstream reports the allowance spent; cooling down until it rolls \
             (further calls would bill pay-as-you-go)"
        );
        state.set_cooldown(
            provider,
            None,
            Some(wait),
            false,
            &format!("{family} allowance spent (resets {resets})"),
        );
    }
}

/// Upstream self-reported exhaustion on a SUCCESS response: openadapter's
/// `X-Quota-5h/Week/Month` used-percentages and the standard
/// `x-ratelimit-remaining-*` family (groq, mistral, openrouter). Cooling the
/// provider down now saves the next request from burning into a 429 — which
/// matters where failed requests count against quota too (openadapter).
/// Error responses already cool down via classify_error.
fn check_quota_exhaustion(state: &State, provider: &str, headers: &reqwest::header::HeaderMap) {
    let get = |name: &str| headers.get(name).and_then(|v| v.to_str().ok()).map(str::trim);

    // Percentage-used style. No reset time is reported, so the wait scales
    // with the window: recheck well before the window could have rolled.
    for (h, wait_secs) in [
        ("x-quota-5h", 30 * 60),
        ("x-quota-week", 2 * 3600),
        ("x-quota-month", 6 * 3600),
    ] {
        let Some(pct) = get(h).and_then(|s| s.trim_end_matches('%').trim().parse::<f64>().ok())
        else {
            continue;
        };
        if pct >= 100.0 {
            warn!(provider, header = h, pct, "upstream reports quota exhausted; cooling down");
            state.set_cooldown(
                provider,
                None,
                Some(Duration::from_secs(wait_secs)),
                false,
                &format!("{h} at {pct}%"),
            );
            return;
        }
    }

    // Remaining-count style, with an optional reset hint.
    for (rem_h, reset_h) in [
        ("x-ratelimit-remaining-requests", "x-ratelimit-reset-requests"),
        ("x-ratelimit-remaining-tokens", "x-ratelimit-reset-tokens"),
        ("x-ratelimit-remaining", "x-ratelimit-reset"),
    ] {
        let Some(rem) = get(rem_h).and_then(|s| s.parse::<f64>().ok()) else { continue };
        if rem <= 0.0 {
            let wait = get(reset_h)
                .and_then(parse_reset)
                .unwrap_or(Duration::from_secs(60))
                .min(Duration::from_secs(3600));
            warn!(
                provider,
                header = rem_h,
                wait_secs = wait.as_secs(),
                "upstream reports rate limit exhausted; cooling down"
            );
            state.set_cooldown(provider, None, Some(wait), false, &format!("{rem_h} exhausted"));
            return;
        }
    }
}

/// A reset header value in any of the three dialects upstreams use:
/// plain seconds ("30"), Go-style durations ("2m59.56s" — groq), and epoch
/// timestamps in seconds or milliseconds (openrouter).
fn parse_reset(s: &str) -> Option<Duration> {
    let s = s.trim();
    if let Ok(n) = s.parse::<f64>() {
        if n <= 0.0 {
            return None;
        }
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .ok()?
            .as_secs_f64();
        let secs = if n >= 1e11 {
            n / 1000.0 - now // epoch millis
        } else if n >= 1e9 {
            n - now // epoch seconds (2001+)
        } else {
            n // relative seconds
        };
        return (secs > 0.0).then(|| Duration::from_secs_f64(secs));
    }
    parse_go_duration(s)
}

/// "1h30m", "2m59.56s", "250ms" → Duration. None on anything unrecognized.
fn parse_go_duration(s: &str) -> Option<Duration> {
    let mut total = 0f64;
    let mut num = String::new();
    let mut matched = false;
    let mut chars = s.chars().peekable();
    while let Some(c) = chars.next() {
        if c.is_ascii_digit() || c == '.' {
            num.push(c);
            continue;
        }
        let factor = match c {
            'h' => 3600.0,
            'm' if chars.peek() == Some(&'s') => {
                chars.next();
                0.001
            }
            'm' => 60.0,
            's' => 1.0,
            _ => return None,
        };
        total += num.parse::<f64>().ok()? * factor;
        num.clear();
        matched = true;
    }
    if !num.is_empty() || !matched || total < 0.0 {
        return None;
    }
    Some(Duration::from_secs_f64(total))
}

fn parse_retry_after(headers: &reqwest::header::HeaderMap) -> Option<Duration> {
    let v = headers.get("retry-after")?.to_str().ok()?.trim();
    // Bare seconds (the RFC form), a duration like "5m" / "2m59s" (groq), or
    // the RFC IMF-fixdate form ("Wed, 21 Oct 2026 07:28:00 GMT") — ignored
    // before, which re-probed the provider every ≤2m instead of at the
    // stated time.
    let dur = match v.parse::<u64>() {
        Ok(secs) => Duration::from_secs(secs),
        Err(_) => parse_go_duration(v).or_else(|| parse_http_date(v))?,
    };
    let secs = dur.as_secs();
    // Sanity clamp: obey only plausible waits, else exponential backoff. A
    // week covers every real quota window (5h rolling, daily, weekly); a
    // monthly reset still lands under it on the recheck. litellm's 1h clamp
    // threw away exactly the headers that matter most — a provider stating
    // "back in 2 days" was re-probed every 4h instead.
    if secs > 0 && secs <= 7 * 24 * 3600 { Some(dur) } else { None }
}

/// IMF-fixdate as used in Retry-After / Date headers. Returns the wait from
/// now; a past date yields None (the caller falls back to backoff).
fn parse_http_date(v: &str) -> Option<Duration> {
    let parsed = jiff::fmt::strtime::parse("%a, %d %b %Y %H:%M:%S GMT", v).ok()?;
    let at = parsed.to_datetime().ok()?.to_zoned(jiff::tz::TimeZone::UTC).ok()?;
    let secs = at.timestamp().as_second() - jiff::Timestamp::now().as_second();
    (secs > 0).then_some(Duration::from_secs(secs as u64))
}

// ---------------------------------------------------------------------------
// Streaming
// ---------------------------------------------------------------------------

enum StreamKind {
    /// passthrough, tap usage from openai chunks
    OpenaiPass,
    /// passthrough, tap usage from anthropic events
    AnthropicPass,
    /// openai upstream -> anthropic client
    ToAnthropic(anthropic_to_openai::StreamState),
    /// anthropic upstream -> openai client
    ToOpenai(openai_to_anthropic::StreamState),
}

struct StreamCtx {
    parser: SseParser,
    kind: StreamKind,
    /// Present when the provider has parse_think_tags (openai upstreams only)
    think: Option<ThinkFilter>,
    /// Present when the request declared tools and the upstream is OpenAI:
    /// extracts text-embedded tool calls (free models emit them as prose).
    tooltext: Option<ToolTextFilter>,
    usage: TokenUsage,
    agent: String,
    /// Bare provider name: logs and model_usage.
    provider: String,
    /// The state-key provider scope (`provider#account`): usage windows.
    state_provider: String,
    model: String,
    app: SharedApp,
    upstream: futures_util::stream::BoxStream<'static, reqwest::Result<Bytes>>,
    done: bool,
    /// Max gap between upstream chunks. Streaming requests carry no total
    /// timeout (a long turn must not die at timeout_secs), so this is the
    /// death signal instead: silence for this long ends the stream.
    stall: Duration,
    /// Present when the request carried a served server tool and the upstream
    /// speaks OpenAI: pxy runs the tool and continues the turn.
    search: Option<ServerToolLoop>,
    /// The client speaks the Responses dialect, so it can read pxy's
    /// served-tool marker chunks. A Chat Completions client must not be sent
    /// one: it has no item for a served tool and would see a `pxy_*` name.
    responses_client: bool,
    /// The OpenAI-dialect client already received a terminal `finish_reason`.
    /// A Chat Completions stream that ends without one is a protocol error to
    /// every client (pi: "Stream ended without finish_reason"), and an
    /// upstream that dies mid-body leaves exactly that hole, so `finish()`
    /// closes it.
    client_done: bool,
    /// The client's turn has been closed (`[DONE]` forwarded, or the
    /// Anthropic message stopped). Some upstreams send one more chunk after
    /// their own `[DONE]` (opencode-go: `{"choices":[],"cost":"0"}`); read as
    /// a fresh chunk it would open a second, empty message on an Anthropic
    /// client and corrupt a non-streaming re-assembly. Nothing follows a
    /// closed turn.
    terminated: bool,
    /// Same-delta counter: a model stuck emitting one chunk forever is cut
    /// off before it spends the whole window.
    repeat: RepeatGuard,
}

/// litellm's rule (streaming_handler.raise_on_model_repetition, default 100):
/// this many consecutive identical content deltas of more than two characters
/// is a loop, not an answer. Short deltas ("\n", ".") repeat legitimately.
const REPEATED_CHUNK_LIMIT: u32 = 100;

#[derive(Default)]
struct RepeatGuard {
    last: Option<String>,
    count: u32,
}

impl RepeatGuard {
    /// Feed one batch of upstream events; true when the limit is reached.
    /// Reads the content delta in the upstream's own dialect: an OpenAI
    /// `choices[0].delta.content` or an Anthropic `content_block_delta` text.
    fn feed(&mut self, events: &[SseEvent], anthropic_upstream: bool) -> bool {
        for ev in events {
            let Ok(v) = serde_json::from_str::<Value>(&ev.data) else { continue };
            let content = if anthropic_upstream {
                (v["type"] == "content_block_delta").then(|| v["delta"]["text"].as_str()).flatten()
            } else {
                v["choices"][0]["delta"]["content"].as_str()
            };
            let Some(content) = content else { continue };
            if content.chars().count() <= 2 {
                self.last = None;
                self.count = 0;
                continue;
            }
            if self.last.as_deref() == Some(content) {
                self.count += 1;
            } else {
                self.last = Some(content.to_string());
                self.count = 1;
            }
            if self.count >= REPEATED_CHUNK_LIMIT {
                return true;
            }
        }
        false
    }
}

/// Accumulates the model's calls to a reserved `pxy_*` server function while
/// stripping them from the chunks, so the client never sees a tool_use it
/// can't run.
#[derive(Default)]
struct ServerCallFilter {
    /// openai tool_call index -> (id, arguments so far). The function name
    /// only rides the first chunk of a call, so later chunks are matched on
    /// the index instead.
    ours: std::collections::HashMap<u64, (String, String)>,
    /// openai tool_call index -> the reserved function name, captured from
    /// the first chunk so the continuation can pick the tool to run.
    names: std::collections::HashMap<u64, String>,
    /// A client tool was called in the same turn. Anthropic's API doesn't run
    /// the search then either — it hands the client tools back first and
    /// searches on the next turn — so pxy leaves the turn alone.
    saw_other: bool,
}

/// The served-tool loop: what the model asked for, what's left of its budget,
/// and the request to replay once results are in.
struct ServerToolLoop {
    filter: ServerCallFilter,
    /// Steps left in the turn, across every served tool.
    uses_left: u64,
    /// Calls left per reserved function name, from each tool's own `max_uses`.
    per_tool: std::collections::HashMap<String, u64>,
    /// Each served tool's declaration `parameters`, keyed by reserved function
    /// name. The meta-tools read their model and instructions from here.
    params: std::collections::HashMap<String, Value>,
    url: String,
    headers: Vec<(String, String)>,
    body: Value,
    timeout: Duration,
    /// The upstream speaks the Responses wire: the replayed body is
    /// translated out and the continuation's events back in, as on the first
    /// call.
    responses_upstream: bool,
    /// The client behind the turn, handed to every executor: a meta-tool's
    /// leg bills to this agent and keeps this session's affinity.
    agent: String,
    session: Option<String>,
    /// The model id the client requested, for a meta-tool whose declaration
    /// pins none.
    outer_model: String,
    /// Calls actually run this turn, by canonical tool name — the
    /// `server_tool_use` the final usage reports.
    served: std::collections::BTreeMap<String, u64>,
    /// Search hits returned to the model this turn, for web_search's
    /// `max_total_results`.
    results_served: u64,
    /// The client function definitions held out of the upstream body, for
    /// tool_search to match against and reveal into the replay.
    deferred: Vec<Value>,
    /// The image URLs this turn's placeholders stand for, for describe_image
    /// to resolve an index against.
    images: Vec<String>,
}

impl ServerToolLoop {
    /// Every captured server call this turn, lowest tool-call index first: a
    /// model may issue SEVERAL in one turn (parallel calls), and running only
    /// an arbitrary one silently lost the model's other queries. Capped by
    /// the remaining search budget; empty when another tool call shared the
    /// turn (the continuation can't fake that one) or the budget is spent.
    #[cfg(test)]
    fn pending_calls(&self) -> Vec<ServerCall> {
        self.split_calls().0
    }

    /// The captured calls, lowest index first, split into those the budget
    /// lets run and those it does not. Selection only; [`commit`](Self::commit)
    /// charges what was served. A call whose tool has no budget left is over
    /// budget (it gets an error result, never another tool's turn budget); a
    /// call to a function that was never offered is in neither list, since
    /// replaying it would name a tool the body does not declare.
    fn split_calls(&self) -> (Vec<ServerCall>, Vec<ServerCall>) {
        if self.filter.saw_other {
            return (Vec::new(), Vec::new());
        }
        let mut keys: Vec<u64> = self.filter.ours.keys().copied().collect();
        keys.sort_unstable();
        let mut left = self.uses_left;
        let mut per_tool = self.per_tool.clone();
        let mut fits: Vec<ServerCall> = Vec::new();
        let mut over: Vec<ServerCall> = Vec::new();
        for k in keys {
            let Some((id, args)) = self.filter.ours.get(&k) else { continue };
            let name = self.filter.names.get(&k).cloned().unwrap_or_default();
            let Some(tool_left) = per_tool.get_mut(&name) else { continue };
            let call = ServerCall { name, id: id.clone(), args: args.clone() };
            if left == 0 || *tool_left == 0 {
                over.push(call);
                continue;
            }
            *tool_left -= 1;
            left -= 1;
            fits.push(call);
        }
        (fits, over)
    }

    /// Whether the upstream call that just ended must be followed by a
    /// continuation: the model made at least one reserved call and no client
    /// tool shared the turn. Over-budget calls count — they are answered with
    /// an error result so the model can still finish its answer.
    fn has_replay(&self) -> bool {
        !self.filter.saw_other && !self.filter.ours.is_empty()
    }

    /// The `server_tool_use` object the final usage reports:
    /// `{"<tool>_requests": n}` per tool that ran, or Null when none did.
    fn server_tool_use(&self) -> Value {
        if self.served.is_empty() {
            return Value::Null;
        }
        let counts: serde_json::Map<String, Value> = self
            .served
            .iter()
            .map(|(name, n)| (format!("{name}_requests"), json!(n)))
            .collect();
        Value::Object(counts)
    }

    /// Charge the calls that were actually served against the turn budget and
    /// each tool's own cap.
    fn commit(&mut self, renders: &[(String, server_tools::ClientRender)]) {
        for (name, render) in renders {
            self.uses_left = self.uses_left.saturating_sub(1);
            if let Some(left) = self.per_tool.get_mut(name) {
                *left = left.saturating_sub(1);
            }
            // A Null marker is an error result pxy wrote itself; only a call
            // that reached its executor is a request.
            if !render.marker.is_null() {
                if let Some(tool) = server_tools::Tool::from_function_name(name) {
                    *self.served.entry(tool.name().to_string()).or_default() += 1;
                }
                self.results_served += render.marker["hits"].as_u64().unwrap_or(0);
            }
        }
    }

    /// Move every definition the model's search found out of `deferred` and
    /// into the replay body's `tools`, so the next round can call it. A tool
    /// revealed stays revealed for the rest of the turn; the next request's
    /// history decides again (wiki:tool-search).
    fn reveal_found(&mut self, renders: &[(String, server_tools::ClientRender)]) {
        let searched = server_tools::function_name(server_tools::Tool::ToolSearch);
        let found: Vec<&Value> = renders
            .iter()
            .filter(|(name, _)| *name == searched)
            .flat_map(|(_, render)| render.marker["found"].as_array().into_iter().flatten())
            .collect();
        if found.is_empty() {
            return;
        }
        let mut revealed: Vec<Value> = Vec::new();
        self.deferred.retain(|definition| {
            if !found.iter().any(|name| **name == definition["function"]["name"]) {
                return true;
            }
            revealed.push(definition.clone());
            false
        });
        if revealed.is_empty() {
            return;
        }
        match self.body.get_mut("tools").and_then(|t| t.as_array_mut()) {
            Some(tools) => tools.extend(revealed),
            None => self.body["tools"] = Value::Array(revealed),
        }
    }

    /// The captured calls as (id, arguments), for telling whether one is
    /// queued; the continuation uses [`pending_calls`](Self::pending_calls)
    /// so it also knows which tool each call belongs to.
    #[cfg(test)]
    fn pending(&self) -> Vec<(String, String)> {
        self.pending_calls().into_iter().map(|c| (c.id, c.args)).collect()
    }
}

/// One reserved server call captured from the upstream stream: the function
/// the model called, its id, and the arguments seen so far.
struct ServerCall {
    name: String,
    id: String,
    args: String,
}

/// The replay material one continuation step assembles from the calls it
/// served: the assistant message's tool_calls, the tool results, and each
/// call's client render paired with the reserved name the loop emits it
/// under.
struct Served {
    tool_calls: Vec<Value>,
    tool_results: Vec<Value>,
    renders: Vec<(String, server_tools::ClientRender)>,
}

/// Run every captured call through its tool, lowest first, and answer the
/// rest. A call whose arguments will not parse or whose executor refuses it
/// gets an error result, as does every call the budget left over: the model
/// asked, so the model is answered, and the turn never ends on the empty
/// assistant message a silently dropped call produced. Only a reserved name
/// that maps to no tool is skipped — replaying it would name a function the
/// body never declared. Returns None when nothing at all could be replayed.
async fn serve_calls(
    app: &SharedApp,
    calls: Vec<ServerCall>,
    over_budget: Vec<ServerCall>,
    loop_: &ServerToolLoop,
) -> Option<Served> {
    let mut tool_calls: Vec<Value> = Vec::new();
    let mut tool_results: Vec<Value> = Vec::new();
    let mut renders: Vec<(String, server_tools::ClientRender)> = Vec::new();
    let mut record = |call: &ServerCall, output: String, render: server_tools::ClientRender| {
        tool_calls.push(json!({
            "id": &call.id,
            "type": "function",
            "function": {"name": &call.name, "arguments": &call.args},
        }));
        tool_results.push(json!({
            "role": "tool", "tool_call_id": &call.id, "content": output,
        }));
        renders.push((call.name.clone(), render));
    };
    let error = |message: String| {
        (
            json!({"status": "error", "error": message}).to_string(),
            server_tools::ClientRender { blocks: Vec::new(), marker: Value::Null },
        )
    };
    // Hits this batch has already returned, so parallel searches share one
    // `max_total_results` budget.
    let mut batch_hits: u64 = 0;
    for call in &calls {
        let Some(tool) = server_tools::Tool::from_function_name(&call.name) else {
            continue;
        };
        let args = match serde_json::from_str::<Value>(&call.args) {
            Ok(args) => args,
            Err(e) => {
                let (output, render) = error(format!("arguments are not valid JSON: {e}"));
                record(call, output, render);
                continue;
            }
        };
        let mut ctx = server_tools::ToolCtx::new(app, &call.id);
        ctx.params = loop_.params.get(&call.name).cloned().unwrap_or(Value::Null);
        ctx.agent = (!loop_.agent.is_empty()).then(|| loop_.agent.clone());
        ctx.session = loop_.session.clone();
        ctx.outer_model = loop_.outer_model.clone();
        ctx.results_used = loop_.results_served + batch_hits;
        ctx.deferred = loop_.deferred.clone();
        ctx.images = loop_.images.clone();
        ctx.transcript = loop_.body["messages"].as_array().cloned().unwrap_or_default();
        match tool.execute(&ctx, &args).await {
            Ok(ran) => {
                batch_hits += ran.client.marker["hits"].as_u64().unwrap_or(0);
                record(call, ran.model_output, ran.client)
            }
            Err(e) => {
                let (output, render) = error(e);
                record(call, output, render);
            }
        }
    }
    for call in &over_budget {
        if server_tools::Tool::from_function_name(&call.name).is_none() {
            continue;
        }
        let (output, render) = error(
            "tool call budget exhausted for this turn; answer with what you have".to_string(),
        );
        record(call, output, render);
    }
    (!tool_calls.is_empty()).then_some(Served { tool_calls, tool_results, renders })
}

/// Take every reserved function out of a replay body once the budget is
/// spent, so the model is asked to answer rather than to call again.
fn strip_reserved_functions(body: &mut Value) {
    let Some(tools) = body["tools"].as_array_mut() else { return };
    tools.retain(|t| {
        !t["function"]["name"].as_str().is_some_and(|n| server_tools::Tool::from_function_name(n).is_some())
    });
    let emptied = tools.is_empty();
    let chosen_reserved = body["tool_choice"]["function"]["name"]
        .as_str()
        .is_some_and(|n| server_tools::Tool::from_function_name(n).is_some());
    let body = body.as_object_mut().expect("tools was indexed on an object");
    if emptied {
        body.remove("tools");
    }
    if emptied || chosen_reserved {
        body.remove("tool_choice");
    }
}

/// Insert the turn's `server_tool_use` into an OpenAI usage chunk. Only a
/// chunk carrying a usage object changes; every other one passes as-is.
fn add_server_tool_use(data: &str, loop_: &ServerToolLoop) -> String {
    let counts = loop_.server_tool_use();
    if counts.is_null() || !data.contains("\"usage\"") {
        return data.to_string();
    }
    let Ok(mut v) = serde_json::from_str::<Value>(data) else {
        return data.to_string();
    };
    if !v["usage"].is_object() {
        return data.to_string();
    }
    v["usage"]["server_tool_use"] = counts;
    v.to_string()
}

/// The prefix pxy reserves for the OpenAI functions it injects for served
/// tools. A client tool that merely shares a tool's own name (`web_search`)
/// never carries it, so it is never intercepted.
const SERVER_CALL_PREFIX: &str = "pxy_";

/// Is this upstream function name one pxy reserved for a server tool?
fn is_server_call(name: &str) -> bool {
    name.starts_with(SERVER_CALL_PREFIX)
}

/// Strip calls to a reserved `pxy_*` server function out of an openai chunk,
/// remembering id + arguments. Returns the rewritten chunk.
fn rewrite_chunk_server_calls(data: &str, f: &mut ServerCallFilter) -> String {
    let Ok(mut v) = serde_json::from_str::<Value>(data) else {
        return data.to_string();
    };
    if v["choices"][0].is_null() {
        return data.to_string();
    }
    // `choices[0]` must be an object for the mutable indexes below: serde_json
    // panics on `["key"]` against a scalar (IndexMut only auto-vivifies Null).
    if !v["choices"][0].is_object() {
        return data.to_string();
    }
    let mut changed = false;

    if let Some(calls) = v["choices"][0]["delta"]["tool_calls"].as_array() {
        let mut keep: Vec<Value> = Vec::new();
        for call in calls {
            let idx = call["index"].as_u64().unwrap_or(0);
            let name = call["function"]["name"].as_str().unwrap_or("");
            if !is_server_call(name) && !f.ours.contains_key(&idx) {
                if !name.is_empty() {
                    f.saw_other = true;
                }
                keep.push(call.clone());
                continue;
            }
            changed = true;
            if !name.is_empty() {
                f.names.insert(idx, name.to_string());
            }
            let entry = f.ours.entry(idx).or_default();
            if let Some(id) = call["id"].as_str().filter(|s| !s.is_empty()) {
                entry.0 = id.to_string();
            }
            if let Some(args) = call["function"]["arguments"].as_str() {
                entry.1.push_str(args);
            }
        }
        if changed {
            if keep.is_empty() {
                v["choices"][0]["delta"].as_object_mut().map(|d| d.remove("tool_calls"));
            } else {
                v["choices"][0]["delta"]["tool_calls"] = Value::Array(keep);
            }
        }
    }

    // The close usually arrives as its own chunk — `finish_reason: tool_calls`
    // with an empty delta — so this can't live in the branch above. Left alone
    // it ends the client's turn before the search has run: the Responses
    // translator completes the response on the spot, and Anthropic clients get
    // a `stop_reason: tool_use` with no tool_use block to answer.
    if !f.ours.is_empty()
        && !f.saw_other
        && v["choices"][0]["finish_reason"].as_str() == Some("tool_calls")
    {
        v["choices"][0]["finish_reason"] = Value::Null;
        changed = true;
    }

    if !changed {
        return data.to_string();
    }
    v.to_string()
}

/// Declared server / Anthropic-defined tools pxy cannot serve on an OpenAI
/// upstream: entries with a `type` that isn't `function`/`custom` and no
/// `input_schema` (their schemas live server-side, so the translator's
/// function mapping has nothing to send — code_execution_*, bash_*,
/// text_editor_*, computer_*). Whether pxy can serve a declared server tool is
/// the registry and `[server_tools] enabled`'s call, not a bare name prefix:
/// [`server_tool_served`] asks both, so a tool left out of `enabled` is
/// unservable. Anthropic-format upstreams are unaffected — the tools pass
/// through and the upstream answers for itself.
fn openai_unservable_server_tools(payload: &Value, cfg: &ServerToolsConfig) -> Vec<String> {
    payload["tools"]
        .as_array()
        .into_iter()
        .flatten()
        .filter(|t| !t["input_schema"].is_object())
        .filter(|t| {
            t["type"].as_str().is_some_and(|ty| {
                ty != "function" && ty != "custom" && !server_tool_served(ty, cfg)
            })
        })
        .map(|t| {
            t["name"]
                .as_str()
                .or_else(|| t["type"].as_str())
                .unwrap_or("?")
                .to_string()
        })
        .collect()
}

/// Is this declared tool type one pxy serves right now? The registry maps the
/// spelling and `[server_tools] enabled` gates it. The translator injects one
/// reserved function per served tool, and the router strips it when no
/// upstream can run it.
fn server_tool_served(ty: &str, cfg: &ServerToolsConfig) -> bool {
    let Some(tool) = server_tools::from_type(ty) else { return false };
    cfg.enabled
        .iter()
        .any(|name| server_tools::from_type(&format!("pxy:{name}")) == Some(tool))
}

/// Is this tool's reserved function present in the body? The translators
/// inject one per declared served tool; this is how the router learns which
/// ones the upstream could actually call.
fn reserved_function_in_body(body: &Value, tool: server_tools::Tool) -> bool {
    let name = server_tools::function_name(tool);
    body["tools"]
        .as_array()
        .is_some_and(|ts| ts.iter().any(|t| t["function"]["name"] == name.as_str()))
}

/// Append a paragraph to an OpenAI-format body's system message, making one
/// when the request carried none. Appended unconditionally: a client that
/// already sent the same text gets it twice rather than have pxy guess at
/// which of two near-identical paragraphs the model should read.
fn append_system_text(body: &mut Value, text: &str) {
    let existing = body["messages"].as_array().and_then(|messages| {
        messages
            .iter()
            .position(|m| matches!(m["role"].as_str(), Some("system" | "developer")))
    });
    let Some(messages) = body["messages"].as_array_mut() else { return };
    match existing {
        Some(at) => match &mut messages[at]["content"] {
            Value::String(content) => content.push_str(&format!("\n\n{text}")),
            Value::Array(parts) => parts.push(json!({"type": "text", "text": text})),
            content => *content = json!(text),
        },
        None => messages.insert(0, json!({"role": "system", "content": text})),
    }
}

/// Swap declared served-tool entries in an OpenAI-format body for the reserved
/// functions pxy intercepts. One def per tool, whatever spellings declared it —
/// a second identical def risks an upstream 400 and hands the model two
/// indistinguishable functions. Ordinary function tools and hosted tools pxy
/// does not know pass through untouched.
fn swap_declared_served_tools(body: &mut Value) {
    let Some(tools) = body["tools"].as_array().cloned() else { return };
    let mut seen: Vec<server_tools::Tool> = Vec::new();
    let converted: Vec<Value> = tools
        .iter()
        .filter_map(|t| {
            // A real function tool: `sanitize_tool_schemas` has already run and
            // auto-vivifies a null `function` key on hosted entries, so the
            // name is what tells them apart.
            if t["function"]["name"].is_string() {
                return Some(t.clone());
            }
            let Some(tool) = t["type"].as_str().and_then(server_tools::from_type) else {
                return Some(t.clone());
            };
            if seen.contains(&tool) {
                return None;
            }
            seen.push(tool);
            Some(server_tools::tool_def(tool, &t["parameters"]))
        })
        .collect();
    body["tools"] = Value::Array(converted);
}

/// Hold the client's `defer_loading` function tools out of the upstream body,
/// returning the definitions taken out for the loop to reveal as the model
/// searches for them.
///
/// Nothing is deferred unless pxy is offering `pxy_tool_search` on this body:
/// a client that defers tools without declaring the search would lose them
/// with no way to ask for them back. An Anthropic-format upstream is left
/// alone entirely, since it cannot run the loop and has native deferral of
/// its own (wiki:tool-search).
fn defer_loading_tools(
    body: &mut Value,
    payload: &Value,
    client: ClientFormat,
    upstream: WireFormat,
    search_offered: bool,
) -> Vec<Value> {
    if upstream != WireFormat::Openai {
        return Vec::new();
    }
    let mut deferred = Vec::new();
    if search_offered {
        let mut revealed = tool_search::revealed_in_history(payload, client);
        // Read before `settle_tool_choice` runs: a choice naming a tool that
        // just left the body would be deleted there, so the choice is
        // evidence that its tool has to go up.
        revealed.extend(body["tool_choice"]["function"]["name"].as_str().map(str::to_string));
        if let Some(tools) = body.get_mut("tools").and_then(|t| t.as_array_mut()) {
            tools.retain(|t| {
                if !is_deferred(t) {
                    return true;
                }
                if revealed.contains(t["function"]["name"].as_str().unwrap_or("")) {
                    return true;
                }
                let mut definition = t.clone();
                strip_defer_loading(&mut definition);
                deferred.push(definition);
                false
            });
        }
    }
    // Whether or not anything was deferred: the key is pxy's own, and such
    // upstreams reject an unknown key on a function entry.
    for tool in body.get_mut("tools").and_then(|t| t.as_array_mut()).into_iter().flatten() {
        strip_defer_loading(tool);
    }
    deferred
}

/// Whether the client asked for this tool to be held back. The translators
/// write the flag onto the chat entry's `function`; a Chat Completions client
/// writes it itself, and may put it beside `type` instead, so both spellings
/// count.
fn is_deferred(tool: &Value) -> bool {
    tool["function"]["defer_loading"] == true || tool["defer_loading"] == true
}

/// Take pxy's own `defer_loading` key off one tool entry, both spellings.
fn strip_defer_loading(tool: &mut Value) {
    tool.as_object_mut().map(|o| o.remove("defer_loading"));
    tool.get_mut("function").and_then(|f| f.as_object_mut()).map(|o| o.remove("defer_loading"));
}

/// Make `tool_choice` name a function the body still offers. A choice naming
/// a served tool by its own name (`web_search`, as Claude Code forces it) is
/// rewritten to the reserved function when that is offered; a choice naming
/// a function that is not in the body — a reserved one pxy just dropped, or
/// a bare name nothing maps to — is removed, since an upstream that validates
/// it would 400. A choice with no tools at all is already invalid.
fn settle_tool_choice(body: &mut Value) {
    let Some(chosen) = body["tool_choice"]["function"]["name"].as_str().map(str::to_string) else {
        if body["tools"].is_null() {
            body.as_object_mut().map(|o| o.remove("tool_choice"));
        }
        return;
    };
    let offered = |body: &Value, name: &str| {
        body["tools"]
            .as_array()
            .is_some_and(|ts| ts.iter().any(|t| t["function"]["name"] == name))
    };
    if offered(body, &chosen) {
        return;
    }
    let reserved = server_tools::Tool::implemented()
        .iter()
        .find(|t| t.name() == chosen)
        .map(|t| server_tools::function_name(*t));
    match reserved {
        Some(name) if offered(body, &name) => body["tool_choice"]["function"]["name"] = json!(name),
        _ => {
            body.as_object_mut().map(|o| o.remove("tool_choice"));
        }
    }
}

/// Declare `[server_tools.defaults]` on this turn as if the client had sent
/// them: one `{"type":"pxy:<tool>","parameters":{..}}` entry per default the
/// request can use. A tool the client declared itself is left to the client's
/// declaration, whatever spelling it used. Only a tool this app can serve is
/// added — a default is an offer, and offering a function nobody intercepts
/// hands the client a tool_use it never declared.
fn inject_default_server_tools(payload: &mut Value, app: &App) {
    if app.cfg.server_tools.defaults.is_empty() {
        return;
    }
    let declared: Vec<server_tools::Tool> = payload["tools"]
        .as_array()
        .into_iter()
        .flatten()
        .filter_map(|t| t["type"].as_str().and_then(server_tools::from_type))
        .collect();
    let mut added: Vec<Value> = Vec::new();
    for (name, params) in &app.cfg.server_tools.defaults {
        let Some(tool) = server_tools::from_type(&format!("pxy:{name}")) else { continue };
        if declared.contains(&tool) || !tool.servable(app) {
            continue;
        }
        let mut entry = json!({"type": format!("pxy:{name}")});
        let params = serde_json::to_value(params).unwrap_or(Value::Null);
        if params.as_object().is_some_and(|o| !o.is_empty()) {
            entry["parameters"] = params;
        }
        added.push(entry);
    }
    if added.is_empty() {
        return;
    }
    match payload["tools"].as_array_mut() {
        Some(tools) => tools.extend(added),
        None => payload["tools"] = Value::Array(added),
    }
}

/// Remove every `pxy:*` tool entry from a body bound for an Anthropic-format
/// upstream: that spelling is pxy's own, and the loop that serves it needs an
/// OpenAI upstream. Every other spelling (`openrouter:*`, `web_search_20250305`)
/// passes through for the upstream to answer for itself.
fn drop_hosted_server_tools(body: &mut Value) {
    let Some(tools) = body["tools"].as_array_mut() else { return };
    let hosted = |t: &Value| t["type"].as_str().is_some_and(|ty| ty.starts_with("pxy:"));
    if !tools.iter().any(hosted) {
        return;
    }
    tools.retain(|t| !hosted(t));
    if tools.is_empty() {
        let body = body.as_object_mut().expect("tools was indexed on an object");
        body.remove("tools");
        body.remove("tool_choice");
    }
}

/// The step budget for one turn: a request-level `max_tool_calls` over
/// `[server_tools] max_tool_calls`.
fn turn_budget(payload: &Value, cfg: &ServerToolsConfig) -> u64 {
    payload["max_tool_calls"].as_u64().unwrap_or(cfg.max_tool_calls)
}

/// Tool names the request declared, in either dialect's shape. None when the
/// request has no tools (extraction must not run — `<tool_call>` in a
/// toolless chat is content, not protocol).
fn declared_tool_names(payload: &Value) -> Option<std::collections::HashSet<String>> {
    let tools = payload["tools"].as_array()?;
    let names: std::collections::HashSet<String> = tools
        .iter()
        .filter_map(|t| t["function"]["name"].as_str().or_else(|| t["name"].as_str()))
        .map(String::from)
        .collect();
    (!names.is_empty()).then_some(names)
}

/// Extract text-embedded tool calls from an openai chunk's delta.content.
/// Synthesized calls use indices from 100 up so a (rare) mix with native
/// tool_calls can't collide on index.
fn rewrite_chunk_tools(data: &str, filter: &mut ToolTextFilter) -> String {
    use crate::translate::tool_text::Op;
    let Ok(mut v) = serde_json::from_str::<Value>(data) else {
        return data.to_string();
    };
    // The include_usage final chunk is `{"choices":[],"usage":{...}}` — a
    // mutable index into the empty array PANICS (serde_json semantics), and
    // a scalar `choices[0]`/`delta` would panic the same way.
    if !v["choices"].as_array().is_some_and(|c| !c.is_empty()) {
        return data.to_string();
    }
    let Some(delta) = v["choices"][0].get_mut("delta").filter(|d| d.is_object()) else {
        return data.to_string();
    };
    let delta = delta.as_object_mut().unwrap();
    if let Some(text) = delta.get("content").and_then(|c| c.as_str()).map(String::from) {
        let ops = filter.push(&text);
        let mut kept = String::new();
        let mut calls: Vec<Value> = Vec::new();
        for op in ops {
            match op {
                Op::Text(t) => kept.push_str(&t),
                Op::Call { name, arguments } => {
                    let n = filter.calls - 1;
                    calls.push(json!({
                        "index": 100 + n,
                        "id": format!("textcall_{n}"),
                        "type": "function",
                        "function": {"name": name, "arguments": arguments},
                    }));
                }
            }
        }
        delta.insert("content".into(), json!(kept));
        if !calls.is_empty() {
            let mut existing = delta.get("tool_calls").and_then(|t| t.as_array()).cloned().unwrap_or_default();
            existing.extend(calls);
            delta.insert("tool_calls".into(), Value::Array(existing));
        }
    }
    let finish = v["choices"][0].get_mut("finish_reason");
    if finish.as_deref().and_then(Value::as_str) == Some("stop") && filter.calls > 0 {
        // Clients gate tool execution on this value (audit §2.4).
        if let Some(finish) = finish {
            *finish = json!("tool_calls");
        }
    }
    v.to_string()
}

/// Does this OpenAI-dialect chunk carry the client's terminal signal? A
/// non-null `finish_reason` is the only thing a Chat Completions client treats
/// as "the turn is over"; `null` (and the usage-only final chunk) means still
/// streaming. The cheap `contains` guard keeps this off the hot path.
fn chunk_is_final(data: &str) -> bool {
    data.contains("\"finish_reason\"")
        && serde_json::from_str::<Value>(data)
            .is_ok_and(|v| v["choices"][0]["finish_reason"].is_string())
}

/// Close an OpenAI-dialect client stream the upstream left open. `[DONE]`
/// alone is not a terminator: clients (pi, the official SDKs) require a
/// non-null `finish_reason` and report a protocol error without one.
/// The one line of an upstream error body worth showing a client: its
/// `error.message` when it is JSON, else the body itself, bounded.
fn error_summary(body: &str) -> String {
    let text = serde_json::from_str::<Value>(body)
        .ok()
        .and_then(|v| v["error"]["message"].as_str().map(str::to_string))
        .unwrap_or_else(|| body.trim().to_string());
    match text.char_indices().nth(300) {
        Some((cut, _)) => format!("{}…", &text[..cut]),
        None => text,
    }
}

fn openai_terminator() -> &'static str {
    "data: {\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}]}\n\ndata: [DONE]\n\n"
}

/// Leftover buffered text at stream end (an opener that never closed).
fn tooltext_flush_chunk(filter: &mut ToolTextFilter) -> Option<String> {
    let rest = filter.flush()?;
    Some(json!({"choices": [{"index": 0, "delta": {"content": rest}}]}).to_string())
}

/// Move `<think>` spans in an openai chunk's delta.content into
/// delta.reasoning_content. Returns the rewritten chunk JSON, or the input
/// unchanged when it isn't a parseable chunk.
fn rewrite_chunk_think(data: &str, filter: &mut ThinkFilter) -> String {
    let Ok(mut v) = serde_json::from_str::<Value>(data) else {
        return data.to_string();
    };
    // Same empty-choices guard as rewrite_chunk_tools: the include_usage
    // final chunk has `choices: []` and a mutable index would panic; a
    // scalar `choices[0]`/`delta` would panic the same way.
    if !v["choices"].as_array().is_some_and(|c| !c.is_empty()) {
        return data.to_string();
    }
    let Some(delta) = v["choices"][0].get_mut("delta").filter(|d| d.is_object()) else {
        return data.to_string();
    };
    let delta = delta.as_object_mut().unwrap();
    if let Some(text) = delta.get("content").and_then(|c| c.as_str()).map(String::from) {
        let (reasoning, content) = filter.push(&text);
        if !reasoning.is_empty() {
            let prior = delta.get("reasoning_content").and_then(Value::as_str).unwrap_or("");
            delta.insert("reasoning_content".into(), json!(format!("{prior}{reasoning}")));
        }
        delta.insert("content".into(), json!(content));
        return v.to_string();
    }
    data.to_string()
}

/// Synthetic chunk carrying whatever the filter still buffered at stream end.
fn think_flush_chunk(filter: &mut ThinkFilter) -> Option<String> {
    let (reasoning, content) = filter.flush();
    if reasoning.is_empty() && content.is_empty() {
        return None;
    }
    let mut delta = serde_json::Map::new();
    if !reasoning.is_empty() {
        delta.insert("reasoning_content".into(), json!(reasoning));
    }
    if !content.is_empty() {
        delta.insert("content".into(), json!(content));
    }
    Some(json!({"choices": [{"index": 0, "delta": delta}]}).to_string())
}

/// A client disconnect (Ctrl-C'd agent turn) drops the stream future before
/// `finish()` runs. The upstream still billed those tokens, so whatever real
/// usage the tap saw must reach the counters — otherwise every aborted turn
/// under-counts quota, and the router routes on numbers it knows are low.
impl Drop for StreamCtx {
    fn drop(&mut self) {
        if self.done {
            return; // finish() ran and already recorded
        }
        record_tokens(&self.app, &self.agent, &self.state_provider, &self.provider, &self.model, self.usage);
    }
}

impl StreamCtx {
    /// Process one upstream chunk; returns bytes for the client.
    fn process(&mut self, bytes: &Bytes) -> Bytes {
        if self.terminated {
            return Bytes::new();
        }
        let events = self.parser.feed(bytes);
        let anthropic_upstream =
            matches!(self.kind, StreamKind::AnthropicPass | StreamKind::ToOpenai(_));
        // A looping model is ended like a stalled one: the turn the client
        // has so far is terminated cleanly, and the model sits out a while
        // so the next request does not pay to rediscover it.
        if self.repeat.feed(&events, anthropic_upstream) {
            warn!(provider = %self.provider, model = %self.model, "upstream repeating one chunk; cut");
            self.app.state.set_cooldown(
                &self.state_provider,
                Some(&self.model),
                None,
                true,
                "repeating output",
            );
            self.done = true;
            return self.finish();
        }
        let Self { kind, think, tooltext, usage, search, client_done, terminated, .. } = self;
        match kind {
            StreamKind::OpenaiPass => {
                for ev in &events {
                    if ev.data.contains("\"usage\"") {
                        if let Ok(v) = serde_json::from_str::<Value>(&ev.data) {
                            if v["usage"].is_object() {
                                let u = TokenUsage::from_openai(&v["usage"]);
                                if u.input > 0 {
                                    usage.input = u.input;
                                }
                                if u.output > 0 {
                                    usage.output = u.output;
                                }
                            }
                        }
                    }
                }
                // A `[DONE]` batch must be rewritten even with no filter so a
                // missing terminal reason can be inserted before it; every
                // other unfiltered batch passes through verbatim.
                let has_done = events.iter().any(|ev| ev.data.trim() == "[DONE]");
                if think.is_none() && tooltext.is_none() && search.is_none() && !has_done {
                    if events.iter().any(|ev| chunk_is_final(&ev.data)) {
                        *client_done = true;
                    }
                    return bytes.clone();
                }
                // Any active filter forces chunk rewriting even in passthrough.
                let mut out = String::new();
                for ev in events {
                    if ev.data.trim() == "[DONE]" {
                        if let Some(filter) = think.as_mut()
                            && let Some(tail) = think_flush_chunk(filter)
                        {
                            out.push_str(&format!("data: {tail}\n\n"));
                        }
                        if let Some(tf) = tooltext.as_mut()
                            && let Some(tail) = tooltext_flush_chunk(tf)
                        {
                            out.push_str(&format!("data: {tail}\n\n"));
                        }
                        // Held back while a search is queued: this ends the
                        // upstream call, not the client's turn.
                        if search.as_ref().is_none_or(|s| !s.has_replay()) {
                            if *client_done {
                                out.push_str("data: [DONE]\n\n");
                            } else {
                                out.push_str(openai_terminator());
                                *client_done = true;
                            }
                            *terminated = true;
                            break;
                        }
                    } else {
                        let mut data = ev.data.clone();
                        if let Some(filter) = think.as_mut() {
                            data = rewrite_chunk_think(&data, filter);
                        }
                        if let Some(tf) = tooltext.as_mut() {
                            data = rewrite_chunk_tools(&data, tf);
                        }
                        if let Some(s) = search.as_mut() {
                            data = rewrite_chunk_server_calls(&data, &mut s.filter);
                            data = add_server_tool_use(&data, s);
                        }
                        if chunk_is_final(&data) {
                            *client_done = true;
                        }
                        out.push_str(&format!("data: {data}\n\n"));
                    }
                }
                Bytes::from(out)
            }
            StreamKind::AnthropicPass => {
                for ev in &events {
                    if let Ok(v) = serde_json::from_str::<Value>(&ev.data) {
                        match v["type"].as_str() {
                            Some("message_start") => {
                                usage.input =
                                    TokenUsage::from_anthropic(&v["message"]["usage"]).input;
                            }
                            Some("message_delta") => {
                                if let Some(o) = v["usage"]["output_tokens"].as_u64() {
                                    usage.output = o;
                                }
                            }
                            _ => {}
                        }
                    }
                }
                bytes.clone()
            }
            StreamKind::ToAnthropic(state) => {
                let mut out = String::new();
                for ev in events {
                    if ev.data.trim() == "[DONE]" {
                        if let Some(filter) = think.as_mut()
                            && let Some(tail) = think_flush_chunk(filter)
                        {
                            out.push_str(&state.on_data(&tail));
                        }
                        if let Some(tf) = tooltext.as_mut()
                            && let Some(tail) = tooltext_flush_chunk(tf)
                        {
                            out.push_str(&state.on_data(&tail));
                        }
                        // A search is queued: this [DONE] ends the upstream
                        // call, not the client's turn. Closing the message
                        // here would strand the answer the model still owes.
                        if search.as_ref().is_none_or(|s| !s.has_replay()) {
                            out.push_str(&state.on_data(&ev.data));
                            *terminated = true;
                            break;
                        }
                    } else {
                        let mut data = ev.data.clone();
                        if let Some(filter) = think.as_mut() {
                            // reasoning moved into delta.reasoning_content
                            // becomes a thinking block via on_data
                            data = rewrite_chunk_think(&data, filter);
                        }
                        if let Some(tf) = tooltext.as_mut() {
                            data = rewrite_chunk_tools(&data, tf);
                        }
                        if let Some(s) = search.as_mut() {
                            data = rewrite_chunk_server_calls(&data, &mut s.filter);
                        }
                        out.push_str(&state.on_data(&data));
                    }
                }
                *usage = state.usage;
                Bytes::from(out)
            }
            StreamKind::ToOpenai(state) => {
                let mut out = String::new();
                for ev in &events {
                    out.push_str(&state.on_event(ev.event.as_deref(), &ev.data));
                }
                *usage = state.usage;
                if state.is_finished() {
                    *client_done = true;
                }
                Bytes::from(out)
            }
        }
    }

    /// End-of-stream flush (upstream may end without a terminal marker).
    /// The upstream call ended on a reserved server-tool call. Run each call
    /// through its tool, splice the protocol blocks into the client's stream,
    /// and re-issue the request with the results appended so the model answers
    /// from them — all inside the one message the client is already reading.
    ///
    /// Returns the bytes to emit, or None when there was nothing to serve
    /// (then the turn closes normally). A tool failure is reported to the
    /// model AND to the client rather than aborting: an error block is what
    /// the real API sends too.
    async fn continue_after_server_calls(&mut self) -> Option<Bytes> {
        let (calls, over_budget) = self.search.as_ref()?.split_calls();
        if calls.is_empty() && over_budget.is_empty() {
            return None;
        }

        // Run every call through its tool, accumulating the client-side
        // render and the model-side tool output. A call that could not run
        // is answered with an error result rather than dropped; only a name
        // the registry does not implement is left out of the replay.
        let served = serve_calls(&self.app, calls, over_budget, self.search.as_ref()?).await?;

        let search = self.search.as_mut()?;
        search.commit(&served.renders);
        // What a search found becomes callable on the round that follows it.
        search.reveal_found(&served.renders);
        search.filter = ServerCallFilter::default();

        // The model's own view of the turn: it called the functions, these
        // are what they returned — one assistant message carrying ALL calls,
        // then one tool message per call (the shape parallel calls require).
        if let Some(messages) = search.body.get_mut("messages").and_then(|m| m.as_array_mut()) {
            messages.push(json!({
                "role": "assistant",
                "content": Value::Null,
                "tool_calls": served.tool_calls,
            }));
            messages.extend(served.tool_results);
        }
        // A choice that forced this call must not force it again on every
        // replay, or the model calls the tool until the budget is gone.
        let forced = search.body["tool_choice"] == "required"
            || search.body["tool_choice"]["function"]["name"]
                .as_str()
                .is_some_and(|n| server_tools::Tool::from_function_name(n).is_some());
        if forced {
            search.body["tool_choice"] = json!("auto");
        }
        // Budget spent: the model is asked to answer, not to call again.
        if search.uses_left == 0 {
            strip_reserved_functions(&mut search.body);
        }
        let counts = search.server_tool_use();

        // Same rule as the first call: the total timeout must not span the
        // body, or the answer after a tool call dies at timeout_secs with a
        // clean-looking stop. Bound the wait for headers here; the body is
        // bounded per read by the stall deadline in the unfold.
        let mut req = self
            .app
            .http
            .post(&search.url)
            .header("content-type", "application/json");
        for (k, v) in &search.headers {
            req = req.header(k, v);
        }
        let wire_body;
        let send_body: &Value = if search.responses_upstream {
            wire_body = responses_upstream::request(&search.body);
            &wire_body
        } else {
            &search.body
        };
        let resp = match tokio::time::timeout(search.timeout, req.json(send_body).send()).await {
            Ok(Ok(r)) if r.status().is_success() => r,
            // No second call means no answer. The turn ends here rather than
            // hanging, and the client is told why in the one place it can
            // still read: the stream cannot change its status any more, and
            // an empty answer with a clean `stop` hides the failure entirely.
            Ok(Ok(r)) => {
                let status = r.status().as_u16();
                let body = r.text().await.unwrap_or_default();
                warn!(status, "server tool continuation failed");
                return Some(self.continuation_failed(&format!("{status}: {}", error_summary(&body))));
            }
            Ok(Err(e)) => {
                warn!(error = %e, "server tool continuation failed");
                return Some(self.continuation_failed(&e.to_string()));
            }
            Err(_) => {
                warn!(secs = search.timeout.as_secs(), "server tool continuation: no response");
                return Some(self.continuation_failed("no response from the upstream"));
            }
        };
        self.upstream = if search.responses_upstream {
            responses_upstream::chat_stream(resp.bytes_stream().boxed())
        } else {
            resp.bytes_stream().boxed()
        };
        self.parser = SseParser::new();
        record_request(&self.app, &self.agent, &self.state_provider, &self.provider, &self.model);

        let mut out = String::new();
        match &mut self.kind {
            StreamKind::ToAnthropic(state) => {
                // Bank what the first call spent before the second overwrites it.
                let spent = state.take_usage();
                record_tokens(&self.app, &self.agent, &self.state_provider, &self.provider, &self.model, spent);
                // The call that asked for the search finished with `tool_calls`;
                // the turn hasn't, and the continuation sets its own reason.
                state.clear_finish_reason();
                state.server_tool_use = counts;
                // One server_tool_use + result pair per call the model made.
                for (_, render) in &served.renders {
                    for block in &render.blocks {
                        out.push_str(&state.emit_block(block.clone()));
                    }
                }
            }
            // Passthrough — codex, through /v1/responses. Chat completions has
            // no event for "a search happened", so pxy sends its own marker
            // chunk and translate/responses turns it into the Responses API's
            // `web_search_call` item. Without it the user watches a silent gap
            // for as long as the searches take and assumes it has hung.
            // Responses-only: a Chat Completions client has no item for a
            // served tool, and the marker would put a `pxy_*` name on its
            // wire.
            _ => {
                let spent = std::mem::take(&mut self.usage);
                record_tokens(&self.app, &self.agent, &self.state_provider, &self.provider, &self.model, spent);
                if self.responses_client {
                    // An error result pxy wrote itself has no marker.
                    for (name, render) in served.renders.iter().filter(|(_, r)| !r.marker.is_null()) {
                        let mut chunk = json!({
                            "object": "chat.completion.chunk",
                            "choices": [],
                        });
                        chunk[name.as_str()] = render.marker.clone();
                        out.push_str(&format!("data: {chunk}\n\n"));
                    }
                }
            }
        }
        Some(Bytes::from(out))
    }

    /// Close the served-tool loop after a continuation that produced no
    /// answer, telling the client so in its own dialect as one text delta.
    fn continuation_failed(&mut self, detail: &str) -> Bytes {
        self.search = None;
        let chunk = json!({"choices": [{"index": 0, "delta": {
            "content": format!("[pxy] server tool continuation failed ({detail})"),
        }}]})
        .to_string();
        match &mut self.kind {
            StreamKind::ToAnthropic(state) => Bytes::from(state.on_data(&chunk)),
            _ => Bytes::from(format!("data: {chunk}\n\n")),
        }
    }

    fn finish(&mut self) -> Bytes {
        let out = match &mut self.kind {
            StreamKind::ToAnthropic(state) => {
                let mut s = String::new();
                if let Some(filter) = &mut self.think
                    && let Some(tail) = think_flush_chunk(filter)
                {
                    s.push_str(&state.on_data(&tail));
                }
                if let Some(tf) = &mut self.tooltext
                    && let Some(tail) = tooltext_flush_chunk(tf)
                {
                    s.push_str(&state.on_data(&tail));
                }
                s.push_str(&state.finish());
                self.usage = state.usage;
                Bytes::from(s)
            }
            StreamKind::OpenaiPass | StreamKind::ToOpenai(_) => {
                let mut s = String::new();
                // Abrupt EOF with buffered text: hand it to the client raw.
                if let Some(tail) = self.tooltext.as_mut().and_then(tooltext_flush_chunk) {
                    s.push_str(&format!("data: {tail}\n\n"));
                }
                // A truncated upstream never sent a terminal reason; without
                // one the client sees a broken stream, not a short answer.
                if !self.client_done {
                    s.push_str(openai_terminator());
                    self.client_done = true;
                }
                Bytes::from(s)
            }
            StreamKind::AnthropicPass => Bytes::new(),
        };
        record_tokens(&self.app, &self.agent, &self.state_provider, &self.provider, &self.model, self.usage);
        out
    }
}

/// Why a stream failed before being committed to the client.
enum StreamFailure {
    /// Died without saying anything useful (EOF, transport error, bare
    /// [DONE]): retryable, walk on.
    Dead(String),
    /// The 200 carried an error event as its payload. Carried verbatim so
    /// the caller can classify by the embedded status — a 400-class error
    /// must still pass through unmodified, not turn into a synthetic 429.
    ErrorEvent(String),
}

/// An SSE data payload that means the 200 status lied. Checked on the FIRST
/// event only: aggregators sometimes return 200 and then deliver the real
/// error (or a bare [DONE]) as the only thing in the stream.
fn stream_error_event(data: &str) -> Option<StreamFailure> {
    let trimmed = data.trim();
    if trimmed == "[DONE]" {
        return Some(StreamFailure::Dead("stream closed with no content".into()));
    }
    let v: Value = serde_json::from_str(trimmed).ok()?;
    let is_err = !v["error"].is_null() || v["type"].as_str() == Some("error");
    is_err.then(|| StreamFailure::ErrorEvent(trimmed.to_string()))
}

/// Best-effort HTTP status embedded in a stream error event. Numeric
/// `code`/`status` fields win (aggregators mirror the upstream status there);
/// Anthropic error types map to their documented codes.
fn error_event_status(data: &str) -> Option<u16> {
    let v: Value = serde_json::from_str(data).ok()?;
    let err = if v["error"].is_object() { &v["error"] } else { &v };
    for field in ["code", "status"] {
        let n = err[field]
            .as_u64()
            .or_else(|| err[field].as_str().and_then(|s| s.parse().ok()));
        if let Some(n) = n.filter(|n| (400..600).contains(n)) {
            return Some(n as u16);
        }
    }
    match err["type"].as_str() {
        Some("invalid_request_error") => Some(400),
        Some("authentication_error") => Some(401),
        Some("permission_error") => Some(403),
        Some("not_found_error") => Some(404),
        Some("request_too_large") => Some(413),
        Some("rate_limit_error") => Some(429),
        Some("api_error") => Some(500),
        Some("overloaded_error") => Some(529),
        _ => None,
    }
}

/// Longest we'll hold a fresh stream waiting for its first event. On expiry
/// we COMMIT and stream as-is (the pre-hold behavior): an upstream that is
/// quietly queueing (openrouter emits only `: PROCESSING` keepalive comments
/// while a free model queues) is alive, and failover before the deadline
/// happens only on affirmative evidence of death.
const FIRST_EVENT_DEADLINE: Duration = Duration::from_secs(10);

/// How long a streaming candidate may take to return response headers while
/// other candidates are waiting. Generous enough for a cold model to accept
/// the request, short enough that a hung provider costs seconds, not the
/// 600-second default timeout.
const HEADERS_DEADLINE: Duration = Duration::from_secs(30);

/// How long a provider that failed to answer is left out of the walk. Flat
/// rather than the escalating default: one silent endpoint should cost the
/// chain a single deadline, not one per model it offers.
const HEADERS_COOLDOWN: Duration = Duration::from_secs(60);

/// Returns Err when the stream died before producing a first event — the
/// caller treats that as a failed attempt and walks on. Nothing has been
/// sent to the client at that point, so failover is invisible.
async fn stream_outcome(
    app: SharedApp,
    agent: &str,
    cand: &Candidate,
    client_format: ClientFormat,
    upstream_format: WireFormat,
    resp: reqwest::Response,
    input_estimate: u64,
    tool_names: Option<std::collections::HashSet<String>>,
    declared_names: Option<std::collections::HashSet<String>>,
    search: Option<ServerToolLoop>,
    fwd_headers: Headers,
    responses_client: bool,
    responses_upstream: bool,
) -> Result<Outcome, StreamFailure> {
    let kind = match (client_format, upstream_format) {
        (ClientFormat::Openai, WireFormat::Openai) => StreamKind::OpenaiPass,
        (ClientFormat::Anthropic, WireFormat::Anthropic) => StreamKind::AnthropicPass,
        (ClientFormat::Anthropic, WireFormat::Openai) => StreamKind::ToAnthropic(
            anthropic_to_openai::StreamState::new(&cand.full_id(), input_estimate, declared_names),
        ),
        (ClientFormat::Openai, WireFormat::Anthropic) => StreamKind::ToOpenai(
            openai_to_anthropic::StreamState::new(&cand.full_id(), declared_names),
        ),
        // Folded into Openai by the caller; the wire is translated below.
        (_, WireFormat::Responses) => unreachable!("responses wire is folded into openai"),
    };

    let parse_think = app
        .cfg
        .providers
        .get(&cand.provider)
        .map(|p| p.parse_think_tags)
        .unwrap_or(false)
        && upstream_format == WireFormat::Openai;
    let stall = Duration::from_secs(
        app.cfg
            .providers
            .get(&cand.provider)
            .map(|p| p.timeout_secs)
            .unwrap_or(600),
    );
    let mut ctx = StreamCtx {
        parser: SseParser::new(),
        kind,
        think: parse_think.then(ThinkFilter::new),
        tooltext: tool_names.map(ToolTextFilter::new),
        usage: TokenUsage::default(),
        agent: agent.to_string(),
        provider: cand.provider.clone(),
        state_provider: cand.state_provider(),
        model: cand.model.id.clone(),
        app,
        upstream: if responses_upstream {
            responses_upstream::chat_stream(resp.bytes_stream().boxed())
        } else {
            resp.bytes_stream().boxed()
        },
        done: false,
        stall,
        search,
        responses_client,
        client_done: false,
        terminated: false,
        repeat: RepeatGuard::default(),
    };

    // Pre-commit read: hold processed client bytes until the upstream yields
    // its first complete event. `sniff` re-parses the raw bytes because
    // process() consumes them through translators that don't surface events.
    let mut sniff = SseParser::new();
    let mut held: Vec<Bytes> = Vec::new();
    let deadline = tokio::time::Instant::now() + FIRST_EVENT_DEADLINE;
    loop {
        let next = match tokio::time::timeout_at(deadline, ctx.upstream.next()).await {
            Ok(n) => n,
            // Deadline: no proof of death, so commit and stream as-is.
            Err(_) => break,
        };
        match next {
            Some(Ok(bytes)) => {
                let events = sniff.feed(&bytes);
                let out = ctx.process(&bytes);
                if !out.is_empty() {
                    held.push(out);
                }
                let Some(first) = events.first() else { continue };
                if let Some(failure) = stream_error_event(&first.data) {
                    return Err(failure);
                }
                break;
            }
            Some(Err(e)) => {
                return Err(StreamFailure::Dead(format!("stream failed before first event: {e}")))
            }
            None => return Err(StreamFailure::Dead("stream ended before first event".into())),
        }
    }
    info!(candidate = %cand.full_id(), stream = true, "routed");

    let head = futures_util::stream::iter(held.into_iter().map(Ok::<Bytes, std::io::Error>));
    let rest = futures_util::stream::unfold(ctx, |mut ctx| async move {
        if ctx.done {
            return None;
        }
        // Stall deadline: streaming has no total timeout, so silence for
        // `stall` (the provider's timeout_secs) is the death signal. Treated
        // exactly like a transport error — usage already seen is recorded by
        // finish(), and the client gets the truncated-but-terminated turn.
        match tokio::time::timeout(ctx.stall, ctx.upstream.next()).await {
            Err(_) => {
                warn!(provider = %ctx.provider, stall_secs = ctx.stall.as_secs(), "upstream stream stalled");
                ctx.done = true;
                let tail = ctx.finish();
                Some((Ok(tail), ctx))
            }
            Ok(Some(Ok(bytes))) => {
                let out = ctx.process(&bytes);
                Some((Ok::<Bytes, std::io::Error>(out), ctx))
            }
            Ok(Some(Err(e))) => {
                warn!(provider = %ctx.provider, error = %e, "upstream stream error");
                ctx.done = true;
                let tail = ctx.finish();
                Some((Ok(tail), ctx))
            }
            Ok(None) => {
                // A queued server call swaps in a fresh upstream response and
                // keeps the same client stream going.
                if let Some(blocks) = ctx.continue_after_server_calls().await {
                    return Some((Ok(blocks), ctx));
                }
                ctx.done = true;
                let tail = ctx.finish();
                Some((Ok(tail), ctx))
            }
        }
    });

    Ok(Outcome::Stream {
        provider: cand.full_id(),
        body: axum::body::Body::from_stream(head.chain(rest)),
        headers: fwd_headers,
    })
}

/// Drain a translated client stream into the equivalent non-streaming JSON
/// body. Used when a non-streaming request had to run through the streaming
/// machinery anyway — a served web_search, whose interception and
/// continuation live in StreamCtx. The stream is already in the CLIENT
/// dialect, so the aggregation is too (unlike force_stream's, which
/// re-assembles the raw upstream dialect).
async fn collect_stream_json(outcome: Outcome, client_format: ClientFormat) -> Outcome {
    let Outcome::Stream { provider, body, headers } = outcome else { return outcome };
    let mut parser = SseParser::new();
    let mut events = Vec::new();
    let mut data = body.into_data_stream();
    while let Some(chunk) = data.next().await {
        match chunk {
            Ok(b) => events.extend(parser.feed(&b)),
            // The stream layer already terminated the turn as cleanly as it
            // could (stall deadline, finish()); aggregate what arrived.
            Err(_) => break,
        }
    }
    events.extend(parser.feed(b"\n\n"));
    let body = match client_format {
        ClientFormat::Openai => crate::translate::aggregate::openai(&events),
        ClientFormat::Anthropic => crate::translate::aggregate::anthropic(&events),
    };
    Outcome::Json { status: 200, body, provider: Some(provider), headers }
}

/// Non-streaming: move `<think>` spans in every choice's message.content
/// into message.reasoning_content.
fn extract_think_from_response(body: &mut Value) {
    let Some(choices) = body.get_mut("choices").and_then(|c| c.as_array_mut()) else { return };
    for choice in choices {
        let message = &mut choice["message"];
        let Some(text) = message["content"].as_str() else { continue };
        let (reasoning, content) = crate::translate::think::extract(text);
        if let Some(r) = reasoning {
            let prior = message["reasoning_content"].as_str().unwrap_or("");
            message["reasoning_content"] = json!(format!("{prior}{r}"));
            message["content"] = json!(content);
        }
    }
}

fn truncate(s: &str, n: usize) -> String {
    if s.len() <= n {
        return s.to_string();
    }
    let mut end = n;
    while end > 0 && !s.is_char_boundary(end) {
        end -= 1;
    }
    format!("{}…", &s[..end])
}

/// `state_provider` scopes the usage windows (per account for multi-account
/// providers); `provider` stays the bare wire name for model_usage — the
/// desktop panel and usage-scan consumers group by it.
fn record_request(app: &App, agent: &str, state_provider: &str, provider: &str, model: &str) {
    record_usage_inner(app, agent, state_provider, provider, model, TokenUsage::default(), true);
}

fn record_tokens(
    app: &App,
    agent: &str,
    state_provider: &str,
    provider: &str,
    model: &str,
    usage: TokenUsage,
) {
    if usage.input == 0 && usage.output == 0 {
        return;
    }
    record_usage_inner(app, agent, state_provider, provider, model, usage, false);
}

/// Embeddings count as one request + input tokens against the same windows.
/// No model stats: the usage panel only reads chat traffic.
pub fn record_embedding_usage(app: &App, provider: &str, tokens: u64) {
    record_usage_inner(
        app,
        "",
        provider,
        provider,
        "",
        TokenUsage { input: tokens, output: 0 },
        true,
    );
}

fn record_usage_inner(
    app: &App,
    agent: &str,
    state_provider: &str,
    provider: &str,
    model: &str,
    usage: TokenUsage,
    request: bool,
) {
    let default_limits = crate::config::Limits::default();
    let limits = app
        .cfg
        .providers
        .get(provider)
        .and_then(|p| p.limits.as_ref())
        .unwrap_or(&default_limits);
    app.state.tpm_add(state_provider, usage.input + usage.output);
    if let Ok(w) = current_windows(limits, Timestamp::now()) {
        let res = if request {
            app.state.record_usage(
                state_provider,
                w.day_start,
                w.month_start,
                usage.input,
                usage.output,
            )
        } else {
            app.state.add_tokens(
                state_provider,
                w.day_start,
                w.month_start,
                usage.input,
                usage.output,
            )
        };
        if let Err(e) = res {
            warn!(provider, error = %e, "usage recording failed");
        }
    }
    if !model.is_empty() {
        let agent = if agent.is_empty() { "other" } else { agent };
        if let Err(e) =
            app.state.record_model_usage(agent, provider, model, request, usage.input, usage.output)
        {
            warn!(provider, model, error = %e, "model usage recording failed");
        }
    }
}

// ---------------------------------------------------------------------------
// Error bodies in the client's dialect
// ---------------------------------------------------------------------------

pub fn error_body(format: ClientFormat, etype: &str, message: &str) -> Value {
    match format {
        ClientFormat::Anthropic => json!({
            "type": "error",
            "error": {"type": etype, "message": message},
        }),
        ClientFormat::Openai => json!({
            "error": {"message": message, "type": etype, "code": null},
        }),
    }
}

fn error_outcome(format: ClientFormat, status: u16, etype: &str, message: &str) -> Outcome {
    Outcome::Json {
        status,
        body: error_body(format, etype, message),
        provider: None,
        // pxy's own answer: there is no upstream response to relay from.
        headers: Vec::new(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use reqwest::header::{HeaderMap, HeaderName, HeaderValue};

    /// A call to the injected search function is captured and removed, and the
    /// `tool_calls` finish that came with it is dropped too — leaving it would
    /// close the turn as `stop_reason: tool_use` with nothing to answer.
    #[test]
    fn search_call_is_stripped_from_the_stream() {
        let mut f = ServerCallFilter::default();
        let out = rewrite_chunk_server_calls(
            &json!({"choices": [{"index": 0, "delta": {"tool_calls": [{
                "index": 0, "id": "call_1", "type": "function",
                "function": {"name": server_tools::function_name(server_tools::Tool::WebSearch), "arguments": "{\"query\":"}
            }]}}]})
            .to_string(),
            &mut f,
        );
        assert!(!out.contains("tool_calls"), "{out}");

        // Arguments streamed across chunks: later ones carry no name.
        let out = rewrite_chunk_server_calls(
            &json!({"choices": [{"index": 0, "finish_reason": "tool_calls", "delta": {"tool_calls": [{
                "index": 0, "function": {"arguments": "\"rust\"}"}
            }]}}]})
            .to_string(),
            &mut f,
        );
        let v: Value = serde_json::from_str(&out).unwrap();
        assert!(v["choices"][0]["finish_reason"].is_null(), "{out}");
        assert_eq!(f.ours[&0], ("call_1".into(), "{\"query\":\"rust\"}".into()));
        assert!(!f.saw_other);
    }

    /// The filter is generic: any reserved `pxy_*` function the upstream calls
    /// is captured and stripped, not only `web_search`.
    #[test]
    fn any_reserved_server_call_is_stripped() {
        let mut f = ServerCallFilter::default();
        let out = rewrite_chunk_server_calls(
            &json!({"choices": [{"index": 0, "delta": {"tool_calls": [{
                "index": 0, "id": "call_9", "type": "function",
                "function": {"name": "pxy_web_fetch", "arguments": "{\"url\":\"https://example.com\"}"}
            }]}}]})
            .to_string(),
            &mut f,
        );
        assert!(!out.contains("tool_calls"), "{out}");
        assert_eq!(
            f.ours[&0],
            ("call_9".into(), "{\"url\":\"https://example.com\"}".into())
        );
        assert!(!f.saw_other);
    }

    /// A client tool that merely shares a served tool's own name is not
    /// reserved: only the `pxy_` prefix marks a function pxy injected, so the
    /// call reaches the client untouched and cancels the search.
    #[test]
    fn unprefixed_client_tool_reaches_the_client() {
        let mut f = ServerCallFilter::default();
        let data = json!({"choices": [{"index": 0, "delta": {"tool_calls": [{
            "index": 0, "id": "c1",
            "function": {"name": "web_search", "arguments": "{}"}
        }]}}]})
        .to_string();
        let out = rewrite_chunk_server_calls(&data, &mut f);
        assert_eq!(out, data);
        assert!(f.ours.is_empty());
        assert!(f.saw_other);
    }

    /// A client tool called in the same turn is forwarded untouched, and its
    /// presence cancels the search: Anthropic's API hands the client tools back
    /// first and searches on the following turn.
    #[test]
    fn client_tool_calls_survive_and_cancel_the_search() {
        let mut f = ServerCallFilter::default();
        let out = rewrite_chunk_server_calls(
            &json!({"choices": [{"index": 0, "delta": {"tool_calls": [
                {"index": 0, "id": "c1", "function": {"name": server_tools::function_name(server_tools::Tool::WebSearch), "arguments": "{}"}},
                {"index": 1, "id": "c2", "function": {"name": "Bash", "arguments": "{}"}}
            ]}}]})
            .to_string(),
            &mut f,
        );
        let v: Value = serde_json::from_str(&out).unwrap();
        let calls = v["choices"][0]["delta"]["tool_calls"].as_array().unwrap();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0]["function"]["name"], "Bash");
        assert!(f.saw_other);

        let ws = server_tools::function_name(server_tools::Tool::WebSearch);
        let mut loop_ = ServerToolLoop {
            filter: f,
            uses_left: 5,
            per_tool: std::collections::HashMap::from([(ws, 5)]),
            params: std::collections::HashMap::new(),
            url: String::new(),
            headers: Vec::new(),
            body: Value::Null,
            timeout: Duration::from_secs(1),
            responses_upstream: false,
            agent: String::new(),
            session: None,
            outer_model: String::new(),
            served: std::collections::BTreeMap::new(),
            results_served: 0,
            deferred: Vec::new(),
            images: Vec::new(),
        };
        // The captured search WOULD run...
        loop_.filter.saw_other = false;
        assert_eq!(loop_.pending().len(), 1, "the reserved call is captured");
        // ...but a client tool in the same turn cancels it.
        loop_.filter.saw_other = true;
        assert!(loop_.pending().is_empty());
    }

    /// `commit` charges only what was served, against the turn budget and each
    /// tool's own cap, and neither counter can go below zero on a second
    /// charge.
    #[test]
    fn commit_charges_the_turn_and_the_tool() {
        let ws = server_tools::function_name(server_tools::Tool::WebSearch);
        let mut loop_ = ServerToolLoop {
            filter: ServerCallFilter::default(),
            uses_left: 5,
            per_tool: std::collections::HashMap::from([(ws.clone(), 3)]),
            params: std::collections::HashMap::new(),
            url: String::new(),
            headers: Vec::new(),
            body: Value::Null,
            timeout: Duration::from_secs(1),
            responses_upstream: false,
            agent: String::new(),
            session: None,
            outer_model: String::new(),
            served: std::collections::BTreeMap::new(),
            results_served: 0,
            deferred: Vec::new(),
            images: Vec::new(),
        };
        let render = || server_tools::ClientRender { blocks: Vec::new(), marker: Value::Null };
        loop_.commit(&[(ws.clone(), render()), (ws.clone(), render())]);
        // Non-saturating, so an over-charge is visible: two renders cost two.
        assert_eq!(loop_.uses_left, 3);
        assert_eq!(loop_.per_tool[&ws], 1);
        loop_.commit(&[(ws.clone(), render())]);
        assert_eq!(loop_.uses_left, 2);
        assert_eq!(loop_.per_tool[&ws], 0, "the tool cap must not go negative");
        // A further charge cannot overshoot below zero.
        loop_.commit(&[(ws.clone(), render())]);
        assert_eq!(loop_.uses_left, 1);
        assert_eq!(loop_.per_tool[&ws], 0);
    }

    /// A turn may carry SEVERAL search calls (parallel tool calls): all of
    /// them must be pending, lowest index first, capped by the budget — the
    /// old single-arbitrary-pick silently dropped the model's other queries.
    #[test]
    fn parallel_search_calls_are_all_pending_lowest_index_first() {
        let ws = server_tools::function_name(server_tools::Tool::WebSearch);
        let mut f = ServerCallFilter::default();
        // Insert out of order on purpose.
        for (idx, id, args) in [
            (1u64, "call_2", "{\"query\":\"b\"}"),
            (0, "call_1", "{\"query\":\"a\"}"),
            (2, "call_3", "{\"query\":\"c\"}"),
        ] {
            f.ours.insert(idx, (id.to_string(), args.to_string()));
            f.names.insert(idx, ws.clone());
        }
        let mut loop_ = ServerToolLoop {
            filter: f,
            uses_left: 3,
            per_tool: std::collections::HashMap::from([(ws.clone(), 3)]),
            params: std::collections::HashMap::new(),
            url: String::new(),
            headers: Vec::new(),
            body: Value::Null,
            timeout: Duration::from_secs(1),
            responses_upstream: false,
            agent: String::new(),
            session: None,
            outer_model: String::new(),
            served: std::collections::BTreeMap::new(),
            results_served: 0,
            deferred: Vec::new(),
            images: Vec::new(),
        };
        let pending = loop_.pending();
        assert_eq!(
            pending,
            vec![
                ("call_1".to_string(), "{\"query\":\"a\"}".to_string()),
                ("call_2".to_string(), "{\"query\":\"b\"}".to_string()),
                ("call_3".to_string(), "{\"query\":\"c\"}".to_string()),
            ]
        );
        // The turn budget caps how many run.
        loop_.uses_left = 2;
        assert_eq!(loop_.pending().len(), 2);
        // So does the tool's own cap, even with turn budget to spare.
        loop_.uses_left = 3;
        loop_.per_tool.insert(ws, 1);
        assert_eq!(loop_.pending().len(), 1);
        // A client tool sharing the turn still suppresses every search.
        loop_.filter.saw_other = true;
        assert!(loop_.pending().is_empty());
    }

    /// The continuation dispatches a captured call through the registry by its
    /// reserved function name. A reserved name no tool implements, arguments
    /// that will not parse, and a call the executor refuses are all left
    /// unserved: the replay may only carry the calls pxy actually answered.
    #[tokio::test]
    async fn serve_calls_leaves_unimplemented_and_unusable_calls_unserved() {
        let app = test_app(
            r#"
            [server]
            [providers.p]
            base_url = "https://unused.example/m"
            models = ["m"]
            "#,
            "serve_calls_skip",
        );
        let known = server_tools::function_name(server_tools::Tool::WebSearch);
        let calls = vec![
            ServerCall {
                name: "pxy_not_a_tool".to_string(),
                id: "call_1".to_string(),
                args: "{}".to_string(),
            },
            // Arguments that are not JSON at all.
            ServerCall {
                name: known.clone(),
                id: "call_2".to_string(),
                args: "not json".to_string(),
            },
            // JSON, but nothing the executor can run.
            ServerCall {
                name: known,
                id: "call_3".to_string(),
                args: "{}".to_string(),
            },
        ];
        let bare = ServerToolLoop {
            filter: ServerCallFilter::default(),
            uses_left: 0,
            per_tool: std::collections::HashMap::new(),
            params: std::collections::HashMap::new(),
            url: String::new(),
            headers: Vec::new(),
            body: Value::Null,
            timeout: Duration::from_secs(1),
            responses_upstream: false,
            agent: String::new(),
            session: None,
            outer_model: String::new(),
            served: std::collections::BTreeMap::new(),
            results_served: 0,
            deferred: Vec::new(),
            images: Vec::new(),
        };
        let served = serve_calls(&app, calls, Vec::new(), &bare).await.expect("two answerable calls");
        // The unknown name is left out; the other two are answered with an
        // error result so the model can still finish its answer.
        assert_eq!(served.tool_calls.len(), 2, "{:?}", served.tool_calls);
        for result in &served.tool_results {
            let content: Value = serde_json::from_str(result["content"].as_str().unwrap()).unwrap();
            assert_eq!(content["status"], "error", "{result}");
        }
        assert!(served.renders.iter().all(|(_, r)| r.marker.is_null()));

        let unknown_only = vec![ServerCall {
            name: "pxy_not_a_tool".to_string(),
            id: "call_9".to_string(),
            args: "{}".to_string(),
        }];
        assert!(
            serve_calls(&app, unknown_only, Vec::new(), &bare).await.is_none(),
            "an unknown name alone leaves nothing to replay"
        );
    }

    /// The upstream closes a tool turn with a bare `finish_reason` chunk that
    /// carries no delta. It has to be neutralised too, or the client is told
    /// the turn is over while the search is still running.
    #[test]
    fn bare_finish_reason_chunk_is_neutralised() {
        let mut f = ServerCallFilter::default();
        f.ours.insert(0, ("call_1".into(), "{}".into()));
        let out = rewrite_chunk_server_calls(
            &json!({"choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}]})
                .to_string(),
            &mut f,
        );
        let v: Value = serde_json::from_str(&out).unwrap();
        assert!(v["choices"][0]["finish_reason"].is_null(), "{out}");

        // A client tool in the same turn means no search, so the finish that
        // hands those calls to the client must survive.
        f.saw_other = true;
        let data =
            json!({"choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}]}).to_string();
        assert_eq!(rewrite_chunk_server_calls(&data, &mut f), data);

        // So must an ordinary end-of-turn finish.
        f.saw_other = false;
        let data = json!({"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}).to_string();
        assert_eq!(rewrite_chunk_server_calls(&data, &mut f), data);
    }

    /// The trailing usage-only chunk (`choices: []`) must pass through: it
    /// carries the token counts.
    #[test]
    fn usage_only_chunk_passes_through() {
        let mut f = ServerCallFilter::default();
        f.ours.insert(0, ("call_1".into(), "{}".into()));
        let data = json!({"choices": [], "usage": {"prompt_tokens": 5}}).to_string();
        assert_eq!(rewrite_chunk_server_calls(&data, &mut f), data);
    }

    /// `max_uses` is a hard stop: a model that keeps searching runs out of
    /// budget and the turn closes instead of looping on pxy's search quota.
    #[test]
    fn exhausted_budget_stops_the_loop() {
        let ws = server_tools::function_name(server_tools::Tool::WebSearch);
        let mut filter = ServerCallFilter::default();
        filter.ours.insert(0, ("call_1".into(), "{\"query\":\"x\"}".into()));
        filter.names.insert(0, ws.clone());
        let mut loop_ = ServerToolLoop {
            filter,
            uses_left: 1,
            per_tool: std::collections::HashMap::from([(ws, 1)]),
            params: std::collections::HashMap::new(),
            url: String::new(),
            headers: Vec::new(),
            body: Value::Null,
            timeout: Duration::from_secs(1),
            responses_upstream: false,
            agent: String::new(),
            session: None,
            outer_model: String::new(),
            served: std::collections::BTreeMap::new(),
            results_served: 0,
            deferred: Vec::new(),
            images: Vec::new(),
        };
        assert!(!loop_.pending().is_empty());
        loop_.uses_left = 0;
        assert!(loop_.pending().is_empty());
        // Over budget is still a replay: the call is answered with an error
        // result, not dropped.
        assert!(loop_.has_replay());
        let (fits, over) = loop_.split_calls();
        assert!(fits.is_empty());
        assert_eq!(over.len(), 1);
    }

    /// `[server_tools] enabled` gates servability: a spelling the registry
    /// knows is unservable when its tool is not enabled, whatever dialect
    /// declared it. A tool the registry does not know stays unservable too.
    #[test]
    fn enabled_gates_servability() {
        let payload = json!({"tools": [
            {"type": "web_search_20250305", "name": "web_search"},
            {"type": "openrouter:web_search"},
            {"type": "openrouter:web_fetch"},
        ]});
        // Every declared spelling pxy implements is served by default.
        assert!(
            openai_unservable_server_tools(&payload, &ServerToolsConfig::default()).is_empty()
        );
        // Dropping web_fetch from `enabled` makes its spelling unservable,
        // whatever dialect declared it.
        let without_fetch = ServerToolsConfig {
            enabled: vec!["web_search".to_string()],
            ..ServerToolsConfig::default()
        };
        assert_eq!(
            openai_unservable_server_tools(&payload, &without_fetch),
            vec!["openrouter:web_fetch"]
        );
        let none = ServerToolsConfig { enabled: vec![], ..without_fetch };
        assert_eq!(
            openai_unservable_server_tools(&payload, &none),
            vec![
                "web_search",
                "openrouter:web_search",
                "openrouter:web_fetch"
            ]
        );
    }

    /// A chunk with no tool calls at all comes back byte-identical.
    #[test]
    fn plain_chunks_pass_through_untouched() {
        let mut f = ServerCallFilter::default();
        let data = json!({"choices": [{"index": 0, "delta": {"content": "hi"}}]}).to_string();
        assert_eq!(rewrite_chunk_server_calls(&data, &mut f), data);
        assert_eq!(rewrite_chunk_server_calls("[DONE]", &mut f), "[DONE]");
    }

    fn headers(pairs: &[(&str, &str)]) -> HeaderMap {
        let mut h = HeaderMap::new();
        for (k, v) in pairs {
            h.insert(
                HeaderName::from_bytes(k.as_bytes()).unwrap(),
                HeaderValue::from_str(v).unwrap(),
            );
        }
        h
    }

    fn state(name: &str) -> State {
        let dir =
            std::env::temp_dir().join(format!("pxy-router-test-{}-{name}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        State::open(&dir.join("s.sqlite")).unwrap()
    }

    /// litellm's failure-rate rule: >= half of >= 5 recent attempts failing
    /// marks a model unhealthy even though no single error tripped the
    /// per-error cooldown ladder; one success starts repairing the record.
    #[test]
    fn failure_rate_marks_a_flapping_model_unhealthy() {
        let s = state("failure_rate");
        for _ in 0..3 {
            s.model_result("p", "m", false);
        }
        assert!(!s.model_unhealthy("p", "m"), "3 failures alone must not trip");
        // Two more failures = 5 attempts, 100% fail rate.
        s.model_result("p", "m", false);
        s.model_result("p", "m", false);
        assert!(s.model_unhealthy("p", "m"));
        // Successes repair the record: 5/6, then 5/7 stay above half…
        s.model_result("p", "m", true);
        assert!(s.model_unhealthy("p", "m"));
        s.model_result("p", "m", true);
        assert!(s.model_unhealthy("p", "m"));
        // …6 successes total = 5/11 = 45%, below the threshold.
        for _ in 0..4 {
            s.model_result("p", "m", true);
        }
        assert!(!s.model_unhealthy("p", "m"), "5/11 drops below half");
        // Sibling models are unaffected.
        assert!(!s.model_unhealthy("p", "other"));
        // Threshold is inclusive at exactly half.
        let s2 = state("failure_rate_edge");
        for _ in 0..5 {
            s2.model_result("p", "m", false);
        }
        s2.model_result("p", "m", true);
        s2.model_result("p", "m", true);
        s2.model_result("p", "m", true);
        assert!(s2.model_unhealthy("p", "m"), "3/6 = exactly half trips");
    }

    #[test]
    fn go_durations_parse() {
        assert_eq!(parse_go_duration("2m59.56s"), Some(Duration::from_secs_f64(179.56)));
        assert_eq!(parse_go_duration("1h30m"), Some(Duration::from_secs(5400)));
        assert_eq!(parse_go_duration("250ms"), Some(Duration::from_millis(250)));
        assert_eq!(parse_go_duration("7s"), Some(Duration::from_secs(7)));
        assert_eq!(parse_go_duration("garbage"), None);
        assert_eq!(parse_go_duration("15"), None); // trailing number, no unit
    }

    #[test]
    fn reset_dialects_parse() {
        assert_eq!(parse_reset("30"), Some(Duration::from_secs(30)));
        assert_eq!(parse_reset("6m0s"), Some(Duration::from_secs(360)));
        // epoch seconds ~1 minute ahead of now
        let ahead = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs()
            + 60;
        let d = parse_reset(&ahead.to_string()).unwrap();
        assert!(d.as_secs() > 50 && d.as_secs() <= 60, "epoch parse gave {d:?}");
        // an epoch in the past means "already reset" — no cooldown
        assert_eq!(parse_reset("1000000000"), None);
    }

    #[test]
    fn quota_percent_exhaustion_cools_provider() {
        let s = state("quota_pct");
        check_quota_exhaustion(&s, "oa", &headers(&[("x-quota-5h", "100%")]));
        assert!(s.cooldown("oa", "any").is_some());
        // below 100% must not cool anything
        let s2 = state("quota_pct_ok");
        check_quota_exhaustion(&s2, "oa", &headers(&[("x-quota-5h", "97%")]));
        assert!(s2.cooldown("oa", "any").is_none());
    }

    /// The whole point of the fingerprint: it must NOT move as a conversation
    /// grows, or opencode scatters one session across upstreams and every
    /// prompt-cache hit is lost.
    #[test]
    fn conversation_fingerprint_survives_appended_turns() {
        let turn1 = json!({
            "model": "glm-5.3-flash",
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Refactor the parser"},
            ],
            "tools": [{"function": {"name": "read"}}, {"function": {"name": "edit"}}],
        });
        // Same session, three turns later: history appended, tools reordered.
        let turn4 = json!({
            "model": "glm-5.3-flash",
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Refactor the parser"},
                {"role": "assistant", "content": "Which file?"},
                {"role": "user", "content": "src/lib.rs"},
                {"role": "assistant", "content": "Done."},
            ],
            "tools": [{"function": {"name": "edit"}}, {"function": {"name": "read"}}],
        });
        let id = conversation_fingerprint(&turn1);
        assert_eq!(id, conversation_fingerprint(&turn4));
        assert!(id.starts_with("pxy_"), "{id}");

        // A different conversation must not collide.
        let other = json!({
            "model": "glm-5.3-flash",
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Write the changelog"},
            ],
        });
        assert_ne!(id, conversation_fingerprint(&other));
        // Same prompt on a different model is a different upstream session.
        let mut swapped = turn1.clone();
        swapped["model"] = json!("gpt-5.6-luna");
        assert_ne!(id, conversation_fingerprint(&swapped));

        // Anthropic shape: system beside the messages, content as blocks.
        let anthropic = json!({
            "model": "glm-5.3-flash",
            "system": [{"type": "text", "text": "You are Claude Code."}],
            "messages": [{"role": "user", "content": [{"type": "text", "text": "hi"}]}],
        });
        let a1 = conversation_fingerprint(&anthropic);
        let mut anthropic2 = anthropic.clone();
        anthropic2["messages"] = json!([
            {"role": "user", "content": [{"type": "text", "text": "hi"}]},
            {"role": "assistant", "content": [{"type": "text", "text": "hello"}]},
        ]);
        assert_eq!(a1, conversation_fingerprint(&anthropic2));
        assert_ne!(a1, id);
    }

    #[test]
    fn only_opencode_providers_are_matched() {
        assert!(is_opencode_provider("opencode-go"));
        assert!(is_opencode_provider("opencode-zen"));
        // The conversation id must never leak to unrelated upstreams.
        for p in ["deepseek", "meta", "tokenharbor", "zenmux", "commandcode"] {
            assert!(!is_opencode_provider(p), "{p}");
        }
    }

    #[test]
    fn free_allowance_headers_are_remembered_and_cool_at_100() {
        let s = state("free_quota");
        record_allowances(
            &s,
            "th",
            &headers(&[
                ("x-th-plan", "free"),
                ("x-th-free-used-pct", "12"),
                ("x-th-free-resets", "2099-09-02T10:08:33.419881+00:00"),
            ]),
        );
        let snap: Value =
            serde_json::from_str(&s.kv_get(&free_quota_key("th")).unwrap().unwrap()).unwrap();
        assert_eq!(snap["usedPct"], 12.0);
        assert_eq!(snap["plan"], "free");
        assert_eq!(snap["resetsAt"], "2099-09-02T10:08:33.419881+00:00");
        assert!(snap["observedAt"].as_str().is_some());
        // A partly-spent allowance is a readout, not a verdict.
        assert!(s.cooldown("th", "any").is_none());
        // A free-lineup call says nothing about the paid window.
        assert!(s.kv_get(&plan_quota_key("th")).unwrap().is_none());

        // Spent: cool until the window actually rolls, and don't retry into it.
        record_allowances(
            &s,
            "th",
            &headers(&[
                ("x-th-free-used-pct", "100"),
                ("x-th-free-resets", "2099-09-02T10:08:33.419881+00:00"),
            ]),
        );
        let cd = s.cooldown("th", "any").expect("provider cooled");
        assert!(!cd.retryable);
        // Far-future reset -> a wait measured in days, not the 1h fallback.
        assert!(cd.until.saturating_duration_since(std::time::Instant::now()).as_secs() > 86_400);
        // A provider that reports nothing must not get a phantom row.
        record_allowances(&s, "other", &headers(&[("x-quota-5h", "3%")]));
        assert!(s.kv_get(&free_quota_key("other")).unwrap().is_none());
    }

    /// A paid pass meters a SECOND window: a call on a pass-unlocked model
    /// reports `x-th-plan-*`, which must land under its own key rather than
    /// overwriting (or being mistaken for) the free allowance.
    #[test]
    fn agent_pass_allowance_is_tracked_beside_the_free_one() {
        let s = state("plan_quota");
        record_allowances(
            &s,
            "th",
            &headers(&[
                ("x-th-plan", "agent"),
                ("x-th-plan-used-pct", "7.5"),
                ("x-th-plan-resets", "2099-09-09T10:08:33.419881+00:00"),
            ]),
        );
        let snap: Value =
            serde_json::from_str(&s.kv_get(&plan_quota_key("th")).unwrap().unwrap()).unwrap();
        assert_eq!(snap["usedPct"], 7.5);
        assert_eq!(snap["plan"], "agent");
        assert_eq!(snap["resetsAt"], "2099-09-09T10:08:33.419881+00:00");
        assert!(s.kv_get(&free_quota_key("th")).unwrap().is_none());
        assert!(s.cooldown("th", "any").is_none());

        // Both windows on one response (as the dashboard shows them) are kept
        // apart, each with its own reset.
        record_allowances(
            &s,
            "th2",
            &headers(&[
                ("x-th-plan", "agent"),
                ("x-th-free-used-pct", "40"),
                ("x-th-free-resets", "2099-09-09T10:08:33.419881+00:00"),
                ("x-th-plan-used-pct", "3"),
                ("x-th-plan-resets", "2099-10-01T00:00:00+00:00"),
            ]),
        );
        let free: Value =
            serde_json::from_str(&s.kv_get(&free_quota_key("th2")).unwrap().unwrap()).unwrap();
        let pass: Value =
            serde_json::from_str(&s.kv_get(&plan_quota_key("th2")).unwrap().unwrap()).unwrap();
        assert_eq!(free["usedPct"], 40.0);
        assert_eq!(pass["usedPct"], 3.0);
        assert_eq!(pass["resetsAt"], "2099-10-01T00:00:00+00:00");

        // Pass spent -> cooled, because the next call bills pay-as-you-go.
        record_allowances(
            &s,
            "th3",
            &headers(&[
                ("x-th-plan", "agent"),
                ("x-th-plan-used-pct", "100"),
                ("x-th-plan-resets", "2099-09-09T10:08:33.419881+00:00"),
            ]),
        );
        assert!(!s.cooldown("th3", "any").expect("provider cooled").retryable);
    }

    #[test]
    fn retry_wait_only_when_recovery_is_near() {
        // Nothing cooling down, no rpm pressure: waiting can't help.
        assert_eq!(retry_wait(None, false), None);
        // Cooldown expiring soon: wait it out (plus the epsilon).
        assert_eq!(retry_wait(Some(Duration::from_secs(2)), false), Some(Duration::from_millis(2250)));
        // Recovery too far away: fail fast instead of holding the request.
        assert_eq!(retry_wait(Some(Duration::from_secs(11)), false), None);
        // rpm windows slide continuously — worth a short wait on their own.
        assert_eq!(retry_wait(None, true), Some(Duration::from_millis(2250)));
        // The sooner of the two hints wins.
        assert_eq!(retry_wait(Some(Duration::from_secs(1)), true), Some(Duration::from_millis(1250)));
    }

    /// Retry-After also arrives as IMF-fixdate ("Wed, 21 Oct 2026 07:28:00
    /// GMT") — ignored before, which re-probed the provider every ≤2m.
    #[test]
    fn http_date_retry_after_parses() {
        let mut h = HeaderMap::new();
        h.insert("retry-after", "120".parse().unwrap());
        assert_eq!(parse_retry_after(&h), Some(Duration::from_secs(120)));

        // One minute out: a wait inside the clamp.
        let soon = (jiff::Timestamp::now() + jiff::Span::new().minutes(1))
            .to_zoned(jiff::tz::TimeZone::UTC);
        let s = jiff::fmt::strtime::format("%a, %d %b %Y %H:%M:%S GMT", &soon).unwrap();
        h.insert("retry-after", s.parse().unwrap());
        let got = parse_retry_after(&h).unwrap();
        assert!(got >= Duration::from_secs(30) && got <= Duration::from_secs(60), "{got:?}");

        // A past date is not a wait: fall back to the ordinary ladder.
        let past = (jiff::Timestamp::now() - jiff::Span::new().minutes(1))
            .to_zoned(jiff::tz::TimeZone::UTC);
        let s = jiff::fmt::strtime::format("%a, %d %b %Y %H:%M:%S GMT", &past).unwrap();
        h.insert("retry-after", s.parse().unwrap());
        assert_eq!(parse_retry_after(&h), None);
    }

    #[test]
    fn stream_error_events_detected() {
        assert!(
            matches!(stream_error_event("[DONE]"), Some(StreamFailure::Dead(_))),
            "bare DONE = empty completion"
        );
        assert!(matches!(
            stream_error_event(r#"{"error":{"message":"boom"}}"#),
            Some(StreamFailure::ErrorEvent(_))
        ));
        assert!(matches!(
            stream_error_event(r#"{"type":"error","error":{"type":"overloaded_error"}}"#),
            Some(StreamFailure::ErrorEvent(_))
        ));
        // Normal first chunks of both dialects pass.
        assert!(stream_error_event(
            r#"{"id":"x","choices":[{"index":0,"delta":{"role":"assistant"}}],"error":null}"#
        )
        .is_none());
        assert!(stream_error_event(r#"{"type":"message_start","message":{"usage":{}}}"#).is_none());
        // Unparseable data is not our call to fail.
        assert!(stream_error_event("not json").is_none());
    }

    #[test]
    fn error_event_status_dialects() {
        // Numeric and stringly code/status fields.
        assert_eq!(error_event_status(r#"{"error":{"code":400,"message":"ctx"}}"#), Some(400));
        assert_eq!(error_event_status(r#"{"error":{"status":429}}"#), Some(429));
        assert_eq!(error_event_status(r#"{"error":{"code":"503"}}"#), Some(503));
        // Non-HTTP numeric codes (openai uses vendor codes) are ignored.
        assert_eq!(error_event_status(r#"{"error":{"code":20015}}"#), None);
        // Anthropic error-type mapping, nested and bare.
        assert_eq!(
            error_event_status(r#"{"type":"error","error":{"type":"invalid_request_error"}}"#),
            Some(400)
        );
        assert_eq!(
            error_event_status(r#"{"type":"error","error":{"type":"overloaded_error"}}"#),
            Some(529)
        );
        // No usable status at all.
        assert_eq!(error_event_status(r#"{"error":{"message":"boom"}}"#), None);
    }

    /// serde_json's IndexMut panics on `["key"]` against a scalar (only Null
    /// auto-vivifies). A corrupt or hostile upstream chunk shaped
    /// `{"choices":[5]}` used to kill the client connection with no failover.
    /// All three chunk rewrites must pass malformed shapes through untouched.
    #[test]
    fn malformed_chunks_never_panic_the_rewrites() {
        let samples = [
            r#"{"choices":[5]}"#,
            r#"{"choices":[{"delta":5,"finish_reason":"stop"}]}"#,
            r#"{"choices":[[]]}"#,
            r#"{"choices":[{"delta":{"content":5}}]}"#,
        ];
        for data in samples {
            assert_eq!(rewrite_chunk_server_calls(data, &mut ServerCallFilter::default()), data);
            let names = declared_tool_names(&json!({"tools": [{"function": {"name": "f"}}]})).unwrap();
            assert_eq!(rewrite_chunk_tools(data, &mut ToolTextFilter::new(names)), data);
            assert_eq!(rewrite_chunk_think(data, &mut ThinkFilter::new()), data);
        }
    }

    // ---- integration: failover ladder against a local mock upstream ----

    async fn mock_server(router: axum::Router) -> String {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move { axum::serve(listener, router).await.unwrap() });
        format!("http://{addr}")
    }

    /// An OpenAI-format upstream that answers its n-th call with the n-th
    /// scripted (status, SSE body) and records every body it was sent. A call
    /// past the script repeats the last step.
    async fn scripted_upstream(
        steps: Vec<(u16, &'static str)>,
    ) -> (String, Arc<std::sync::Mutex<Vec<Value>>>) {
        use axum::response::IntoResponse;
        use axum::routing::post;
        let seen = Arc::new(std::sync::Mutex::new(Vec::<Value>::new()));
        let calls = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let sink = seen.clone();
        let router = axum::Router::new().route(
            "/m",
            post(move |axum::Json(body): axum::Json<Value>| {
                let sink = sink.clone();
                let calls = calls.clone();
                let steps = steps.clone();
                async move {
                    sink.lock().unwrap().push(body);
                    let n = calls.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                    let (status, sse) = steps[n.min(steps.len() - 1)];
                    axum::http::Response::builder()
                        .status(status)
                        .header("content-type", "text/event-stream")
                        .body(axum::body::Body::from(sse))
                        .unwrap()
                        .into_response()
                }
            }),
        );
        (mock_server(router).await, seen)
    }

    /// The SSE of a model calling reserved functions, one call per (id,
    /// name, arguments), closed with `tool_calls`.
    fn calls_sse(calls: &[(&str, &str, &str)]) -> String {
        let mut out = String::new();
        for (i, (id, name, args)) in calls.iter().enumerate() {
            let chunk = json!({"id": "c1", "choices": [{"index": 0, "delta": {"tool_calls": [
                {"index": i, "id": id, "type": "function",
                 "function": {"name": name, "arguments": args}}]}}]});
            out.push_str(&format!("data: {chunk}\n\n"));
        }
        out.push_str("data: {\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"tool_calls\"}]}\n\n");
        out.push_str("data: [DONE]\n\n");
        out
    }

    const ANSWER_SSE: &str = concat!(
        "data: {\"id\":\"c2\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"done\"}}]}\n\n",
        "data: {\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n",
        "data: {\"choices\":[],\"usage\":{\"prompt_tokens\":4,\"completion_tokens\":1}}\n\n",
        "data: [DONE]\n\n",
    );

    async fn datetime_turn(base: &str, name: &str, max_tool_calls: u64) -> String {
        let app = test_app(
            &format!(
                r#"
                [server]
                [providers.p]
                base_url = "{base}/m"
                models = ["m"]
                "#
            ),
            name,
        );
        let payload = json!({
            "model": "p/m",
            "stream": true,
            "messages": [{"role": "user", "content": "what time is it?"}],
            "tools": [{"type": "pxy:datetime"}],
            "max_tool_calls": max_tool_calls,
        });
        let out =
            handle_chat(app.clone(), ClientFormat::Openai, payload, ClientContext::default()).await;
        let Outcome::Stream { body, .. } = out else { panic!("expected a stream") };
        let bytes = axum::body::to_bytes(body, 1 << 20).await.unwrap();
        String::from_utf8_lossy(&bytes).into_owned()
    }

    /// One memory turn: the upstream calls `pxy_memory` once with `args`, then
    /// answers. Returns the bodies pxy sent it.
    async fn memory_turn(
        store: &str,
        name: &'static str,
        system: Option<&str>,
        args: &str,
    ) -> Vec<Value> {
        let first = calls_sse(&[("call_1", "pxy_memory", args)]);
        let first: &'static str = Box::leak(first.into_boxed_str());
        let (base, seen) = scripted_upstream(vec![(200, first), (200, ANSWER_SSE)]).await;
        let app = test_app(
            &format!(
                r#"
                [server]
                [providers.p]
                base_url = "{base}/m"
                models = ["m"]
                "#
            ),
            name,
        );
        let mut messages = vec![json!({"role": "user", "content": "remember this"})];
        if let Some(system) = system {
            messages.insert(0, json!({"role": "system", "content": system}));
        }
        let payload = json!({
            "model": "p/m",
            "stream": true,
            "messages": messages,
            "tools": [{"type": "pxy:memory", "parameters": {"store": store}}],
        });
        let out =
            handle_chat(app.clone(), ClientFormat::Openai, payload, ClientContext::default()).await;
        let Outcome::Stream { body, .. } = out else { panic!("expected a stream") };
        let _ = axum::body::to_bytes(body, 1 << 20).await.unwrap();
        let seen = seen.lock().unwrap();
        seen.clone()
    }

    /// A Messages client declares memory the way Anthropic's own API takes it,
    /// by date and with no `parameters`: the same executor answers, the
    /// protocol paragraph joins the system message the client sent as
    /// `system`, and the round leaves no block behind for a client that has no
    /// way to read one.
    #[tokio::test]
    async fn memory_is_served_to_an_anthropic_client_declaring_it_by_date() {
        let store = format!("pxy-test-anthropic-{}", std::process::id());
        let root = crate::config::data_dir().join("memory").join(&store);
        let _ = std::fs::remove_dir_all(&root);
        let first = calls_sse(&[(
            "call_1",
            "pxy_memory",
            r#"{"command":"create","path":"/memories/n.md","file_text":"kept\n"}"#,
        )]);
        let first: &'static str = Box::leak(first.into_boxed_str());
        let (base, seen) = scripted_upstream(vec![(200, first), (200, ANSWER_SSE)]).await;
        let app = test_app(
            &format!(
                r#"
                [server]
                [providers.p]
                base_url = "{base}/m"
                models = ["m"]
                "#
            ),
            "anthropic_memory",
        );
        let payload = json!({
            "model": "p/m",
            "stream": true,
            "max_tokens": 100,
            "system": "You are Claude Code.",
            "messages": [{"role": "user", "content": "x"}],
            "tools": [{"type": "memory_20250818", "name": "memory"}],
        });
        // A native declaration carries no `parameters`, so the store is the
        // agent behind the turn: this is the path a real Claude Code request
        // takes.
        let ctx = ClientContext { agent: Some(store.clone()), ..ClientContext::default() };
        let out = handle_chat(app, ClientFormat::Anthropic, payload, ctx).await;
        let Outcome::Stream { body, .. } = out else { panic!("expected a stream") };
        let bytes = axum::body::to_bytes(body, 1 << 20).await.unwrap();
        let text = String::from_utf8_lossy(&bytes);

        let bodies = seen.lock().unwrap();
        assert_eq!(bodies.len(), 2, "the served call must have been replayed");
        let names: Vec<&str> = bodies[0]["tools"]
            .as_array()
            .expect("the dated declaration must become a reserved function")
            .iter()
            .filter_map(|t| t["function"]["name"].as_str())
            .collect();
        assert_eq!(names, vec!["pxy_memory"], "{names:?}");
        assert_eq!(
            bodies[0]["messages"][0]["content"].as_str().unwrap(),
            format!("You are Claude Code.\n\n{}", server_tools::MEMORY_PROTOCOL)
        );
        assert_eq!(last_message(&bodies[1]), "File created successfully at: /memories/n.md");
        assert!(text.contains("done"), "{text}");
        // The usage still counts the call; what memory has no way to render is
        // a content block the client would have to interpret.
        assert!(text.contains("\"memory_requests\":1"), "{text}");
        assert!(!text.contains("\"type\":\"server_tool_use\""), "{text}");
        assert!(!text.contains("pxy_memory"), "the reserved name must not leak: {text}");

        assert!(root.join("n.md").is_file(), "the file the model created is on disk");
        let _ = std::fs::remove_dir_all(&root);
    }

    /// The memory tool is a directory, not a transcript: a second request
    /// reads what the first wrote. Both turns carry the protocol paragraph,
    /// once per body — the replay copies the body pxy already appended to, so
    /// a continuation must not stack a second copy.
    #[tokio::test]
    async fn memory_keeps_a_file_across_two_requests_and_appends_the_protocol_once() {
        let store = format!("pxy-test-router-{}", std::process::id());
        let root = crate::config::data_dir().join("memory").join(&store);
        let _ = std::fs::remove_dir_all(&root);

        let wrote = memory_turn(
            &store,
            "memory_write",
            None,
            r#"{"command":"create","path":"/memories/n.md","file_text":"kept\n"}"#,
        )
        .await;
        let result = last_message(&wrote[1]);
        assert_eq!(result, "File created successfully at: /memories/n.md", "{}", wrote[1]);
        assert_eq!(wrote[0]["messages"][0]["role"], "system", "{}", wrote[0]);
        assert_eq!(wrote[0]["messages"][0]["content"], server_tools::MEMORY_PROTOCOL);

        let read = memory_turn(
            &store,
            "memory_read",
            Some("be brief"),
            r#"{"command":"view","path":"/memories/n.md"}"#,
        )
        .await;
        assert!(last_message(&read[1]).ends_with("\n     1\tkept"), "{}", read[1]);
        for body in &read {
            let system = body["messages"][0]["content"].as_str().unwrap();
            assert_eq!(
                system,
                format!("be brief\n\n{}", server_tools::MEMORY_PROTOCOL),
                "the client's own system message keeps its text and gains one copy"
            );
        }

        let _ = std::fs::remove_dir_all(&root);
    }

    /// The `role: "tool"` result the replay carries for the last served call.
    fn last_message(body: &Value) -> String {
        let msgs = body["messages"].as_array().unwrap();
        msgs[msgs.len() - 1]["content"].as_str().unwrap().to_string()
    }

    /// A reserved call pxy could not run — arguments that are not JSON here —
    /// is answered with an error result and replayed beside the calls that
    /// ran, so the model still writes its answer and the client sees it.
    #[tokio::test]
    async fn served_turn_answers_a_malformed_call_with_an_error_result() {
        let first = calls_sse(&[
            ("call_1", "pxy_datetime", "{not json"),
            ("call_2", "pxy_datetime", "{}"),
        ]);
        let first: &'static str = Box::leak(first.into_boxed_str());
        let (base, seen) = scripted_upstream(vec![(200, first), (200, ANSWER_SSE)]).await;
        let text = datetime_turn(&base, "served_turn_malformed", 5).await;
        assert!(text.contains("done"), "{text}");
        let bodies = seen.lock().unwrap();
        assert_eq!(bodies.len(), 2, "the turn must be replayed");
        let msgs = bodies[1]["messages"].as_array().unwrap();
        let assistant = &msgs[msgs.len() - 3];
        assert_eq!(assistant["tool_calls"].as_array().unwrap().len(), 2, "{assistant}");
        let bad: Value = serde_json::from_str(msgs[msgs.len() - 2]["content"].as_str().unwrap()).unwrap();
        assert_eq!(bad["status"], "error", "{bad}");
        assert!(bad["error"].as_str().unwrap().contains("not valid JSON"), "{bad}");
        assert!(msgs[msgs.len() - 1]["content"].as_str().unwrap().contains("UTC"), "{}", msgs[msgs.len() - 1]);
        assert!(bodies[1]["tools"].is_array(), "budget left: the tool stays offered");
    }

    /// With the budget spent, the calls it could not cover are answered with
    /// an error result, the replay offers no reserved function, and the
    /// model answers from what it has instead of the turn ending empty.
    #[tokio::test]
    async fn served_turn_over_budget_answers_and_strips_the_tools() {
        let first = calls_sse(&[
            ("call_1", "pxy_datetime", "{}"),
            ("call_2", "pxy_datetime", "{}"),
        ]);
        let first: &'static str = Box::leak(first.into_boxed_str());
        let (base, seen) = scripted_upstream(vec![(200, first), (200, ANSWER_SSE)]).await;
        let text = datetime_turn(&base, "served_turn_budget", 1).await;
        assert!(text.contains("done"), "{text}");
        assert!(text.contains("\"finish_reason\":\"stop\""), "{text}");
        let bodies = seen.lock().unwrap();
        assert_eq!(bodies.len(), 2);
        let msgs = bodies[1]["messages"].as_array().unwrap();
        let over: Value = serde_json::from_str(msgs[msgs.len() - 1]["content"].as_str().unwrap()).unwrap();
        assert_eq!(over["status"], "error", "{over}");
        assert!(over["error"].as_str().unwrap().contains("budget"), "{over}");
        assert!(bodies[1].get("tools").is_none(), "budget spent: no reserved function may be offered: {}", bodies[1]);
    }

    /// A `tool_choice` that forced the served call is reset to `auto` on the
    /// replay: forced once, not on every continuation until the budget is
    /// gone.
    #[tokio::test]
    async fn forced_tool_choice_is_not_replayed() {
        let first = calls_sse(&[("call_1", "pxy_datetime", "{}")]);
        let first: &'static str = Box::leak(first.into_boxed_str());
        let (base, seen) = scripted_upstream(vec![(200, first), (200, ANSWER_SSE)]).await;
        let app = test_app(
            &format!("[server]\n[providers.p]\nbase_url = \"{base}/m\"\nmodels = [\"m\"]\n"),
            "forced_choice",
        );
        let payload = json!({
            "model": "p/m",
            "stream": true,
            "messages": [{"role": "user", "content": "date?"}],
            "tools": [{"type": "pxy:datetime"}],
            "tool_choice": {"type": "function", "function": {"name": "datetime"}},
        });
        let out = handle_chat(app.clone(), ClientFormat::Openai, payload, ClientContext::default()).await;
        let Outcome::Stream { body, .. } = out else { panic!("expected a stream") };
        let text = String::from_utf8_lossy(&axum::body::to_bytes(body, 1 << 20).await.unwrap()).into_owned();
        assert!(text.contains("done"), "{text}");
        let bodies = seen.lock().unwrap();
        assert_eq!(bodies[0]["tool_choice"]["function"]["name"], "pxy_datetime", "{}", bodies[0]);
        assert_eq!(bodies[1]["tool_choice"], "auto", "{}", bodies[1]);
    }

    /// An upstream that sends one more chunk after its `[DONE]` (opencode-go
    /// appends `{"choices":[],"cost":"0"}`) must not open a second message on
    /// an Anthropic client, nor leak past the terminator on an OpenAI one.
    #[tokio::test]
    async fn chunks_after_done_are_ignored() {
        const TRAILING: &str = concat!(
            "data: {\"id\":\"c1\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"hi\"}}]}\n\n",
            "data: {\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n",
            "data: [DONE]\n\n",
            "data: {\"choices\":[],\"cost\":\"0\"}\n\n",
        );
        let (base, _seen) = scripted_upstream(vec![(200, TRAILING)]).await;
        // An advisor default rides the Messages turn, so the non-streaming
        // request runs through the stream machinery, as it does live.
        let app = test_app(
            &format!(
                "[server]\n[providers.p]\nbase_url = \"{base}/m\"\nmodels = [\"m\"]\n\
                 [server_tools.defaults.advisor]\nmodel = \"p/m\"\n"
            ),
            "after_done",
        );
        let out = handle_chat(
            app.clone(),
            ClientFormat::Anthropic,
            json!({"model": "p/m", "max_tokens": 10, "messages": [{"role": "user", "content": "x"}]}),
            ClientContext::default(),
        )
        .await;
        let Outcome::Json { body, .. } = out else { panic!("expected JSON") };
        assert_eq!(body["content"][0]["text"], "hi", "{body}");

        let out = handle_chat(
            app.clone(),
            ClientFormat::Anthropic,
            json!({"model": "p/m", "max_tokens": 10, "stream": true, "messages": [{"role": "user", "content": "x"}]}),
            ClientContext::default(),
        )
        .await;
        let Outcome::Stream { body, .. } = out else { panic!("expected a stream") };
        let text = String::from_utf8_lossy(&axum::body::to_bytes(body, 1 << 20).await.unwrap()).into_owned();
        assert_eq!(text.matches("message_start").count(), 2, "one event line, one data line: {text}");
        assert_eq!(text.matches("message_stop").count(), 2, "{text}");

        let out = handle_chat(
            app.clone(),
            ClientFormat::Openai,
            json!({"model": "p/m", "stream": true, "messages": [{"role": "user", "content": "x"}], "tools": [{"type": "pxy:datetime"}]}),
            ClientContext::default(),
        )
        .await;
        let Outcome::Stream { body, .. } = out else { panic!("expected a stream") };
        let text = String::from_utf8_lossy(&axum::body::to_bytes(body, 1 << 20).await.unwrap()).into_owned();
        assert!(text.trim_end().ends_with("data: [DONE]"), "nothing after the terminator: {text}");
    }

    /// A continuation the upstream refuses cannot change the stream's status
    /// any more; the client is told in the one place it can still read, and
    /// the stream still closes with a terminal reason.
    #[tokio::test]
    async fn served_turn_continuation_failure_is_visible() {
        let first = calls_sse(&[("call_1", "pxy_datetime", "{}")]);
        let first: &'static str = Box::leak(first.into_boxed_str());
        let (base, _seen) = scripted_upstream(vec![
            (200, first),
            (500, "{\"error\":{\"message\":\"backend exploded\"}}"),
        ])
        .await;
        let text = datetime_turn(&base, "served_turn_failure", 5).await;
        assert!(text.contains("server tool continuation failed (500: backend exploded)"), "{text}");
        assert!(text.contains("\"finish_reason\":\"stop\""), "{text}");
        assert!(text.trim_end().ends_with("data: [DONE]"), "{text}");
    }

    /// The final usage of a turn that ran a served tool reports how many
    /// times each tool ran, under OpenRouter's `server_tool_use` spelling.
    #[tokio::test]
    async fn server_tool_use_is_reported_in_the_usage_chunk() {
        let first = calls_sse(&[
            ("call_1", "pxy_datetime", "{}"),
            ("call_2", "pxy_datetime", "{\"timezone\":\"Europe/London\"}"),
        ]);
        let first: &'static str = Box::leak(first.into_boxed_str());
        let (base, _seen) = scripted_upstream(vec![(200, first), (200, ANSWER_SSE)]).await;
        let text = datetime_turn(&base, "server_tool_use", 5).await;
        assert!(
            text.contains("\"server_tool_use\":{\"datetime_requests\":2}"),
            "the usage chunk must carry the counts: {text}"
        );
    }

    fn test_app(cfg_toml: &str, name: &str) -> SharedApp {
        let cfg: Config = toml::from_str(cfg_toml).unwrap();
        let catalog = Catalog::from_config(&cfg);
        let dir = std::env::temp_dir().join(format!("pxy-router-it-{}-{name}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        Arc::new(App {
            catalog,
            secrets: Secrets::new(),
            state: State::open(&dir.join("s.sqlite")).unwrap(),
            http: reqwest::Client::new(),
            cfg,
        })
    }

    #[test]
    fn repeat_guard_trips_on_the_hundredth_identical_delta() {
        let ev = |data: &str| SseEvent { event: None, data: data.to_string() };
        let openai = |s: &str| ev(&json!({"choices":[{"delta":{"content":s}}]}).to_string());
        let mut g = RepeatGuard::default();
        for _ in 0..99 {
            assert!(!g.feed(&[openai("the same")], false));
        }
        assert!(g.feed(&[openai("the same")], false));

        // A different delta resets the count.
        let mut g = RepeatGuard::default();
        for _ in 0..99 {
            g.feed(&[openai("the same")], false);
        }
        assert!(!g.feed(&[openai("other")], false));
        assert!(!g.feed(&[openai("the same")], false));

        // Short deltas repeat legitimately and never count.
        let mut g = RepeatGuard::default();
        for _ in 0..200 {
            assert!(!g.feed(&[openai("\n")], false));
        }

        // Anthropic reads content_block_delta text.
        let anth = |s: &str| {
            ev(&json!({"type":"content_block_delta","delta":{"type":"text_delta","text":s}}).to_string())
        };
        let mut g = RepeatGuard::default();
        for _ in 0..99 {
            assert!(!g.feed(&[anth("again")], true));
        }
        assert!(g.feed(&[anth("again")], true));
    }

    #[test]
    fn clamp_effort_picks_the_nearest_listed_level() {
        let allowed = |xs: &[&str]| xs.iter().map(|s| s.to_string()).collect::<Vec<_>>();
        // Listed: nothing to do.
        assert_eq!(clamp_effort("high", &allowed(&["low", "high", "max"])), None);
        // Nobody knows what the model takes: leave it alone.
        assert_eq!(clamp_effort("medium", &allowed(&[])), None);
        // Equidistant neighbours: the higher one keeps the intent to think.
        assert_eq!(clamp_effort("medium", &allowed(&["low", "high", "max"])), Some("high".into()));
        // Above the top: the top.
        assert_eq!(clamp_effort("xhigh", &allowed(&["low", "medium"])), Some("medium".into()));
        // Below the bottom: the bottom.
        assert_eq!(clamp_effort("minimal", &allowed(&["medium", "high"])), Some("medium".into()));
        // An unknown spelling is not pxy's to reinterpret.
        assert_eq!(clamp_effort("turbo", &allowed(&["low", "high"])), None);
    }

    #[test]
    fn tpm_limit_skips_candidate_after_a_minute_of_tokens() {
        let app = test_app(
            r#"
            [server]
            [providers.p]
            base_url = "http://127.0.0.1:1/c"
            models = ["m"]
            [providers.p.limits]
            tpm = 1000
            [groups.g]
            models = ["p/m"]
            "#,
            "tpm",
        );
        let cand = app.catalog.resolve(&app.cfg, "p/m").remove(0);
        assert_eq!(check_candidate(&app, &cand, 10, false, false, true), Ok(()));
        app.state.tpm_add("p", 999);
        assert_eq!(check_candidate(&app, &cand, 10, false, false, true), Ok(()));
        app.state.tpm_add("p", 1);
        assert_eq!(
            check_candidate(&app, &cand, 10, false, false, true),
            Err("tpm limit".to_string())
        );
        // The window is the same blend as rpm: nearly exhausted means little
        // headroom left.
        assert!(remaining_headroom(&app.cfg, &app.state, &cand) < 0.01);
    }

    #[test]
    fn model_variant_grammar_resolves_base_and_override() {
        let app = test_app(
            r#"
            [server]
            [providers.p]
            base_url = "http://127.0.0.1:1/c"
            models = ["m", "qwen3.8-max"]
            [groups.g]
            models = ["p/m"]
            "#,
            "variant",
        );
        let (cat, cfg) = (&app.catalog, &app.cfg);
        assert_eq!(
            split_model_variant(cat, cfg, "p/m-high"),
            ("p/m".to_string(), Some(ModelVariant::Effort("high".into())))
        );
        assert_eq!(
            split_model_variant(cat, cfg, "g-max"),
            ("g".to_string(), Some(ModelVariant::Effort("max".into())))
        );
        assert_eq!(
            split_model_variant(cat, cfg, "no-think/p/m"),
            ("p/m".to_string(), Some(ModelVariant::NoThink))
        );
        // A listed id ending in -max is NOT mangled: its base is not listed.
        assert_eq!(split_model_variant(cat, cfg, "p/qwen3.8-max"), ("p/qwen3.8-max".into(), None));
        // Neither is an unknown base.
        assert_eq!(split_model_variant(cat, cfg, "p/nope-high"), ("p/nope-high".into(), None));
        assert_eq!(split_model_variant(cat, cfg, "p/m"), ("p/m".into(), None));
    }

    #[test]
    fn apply_model_variant_writes_only_the_clients_dialect() {
        let mut openai = json!({"messages": []});
        apply_model_variant(&mut openai, ClientFormat::Openai, &ModelVariant::Effort("high".into()));
        assert_eq!(openai["reasoning_effort"], "high");
        assert!(openai.get("thinking").is_none(), "no Anthropic field leak");

        let mut anthropic = json!({"messages": [], "max_tokens": 10});
        apply_model_variant(&mut anthropic, ClientFormat::Anthropic, &ModelVariant::Effort("medium".into()));
        assert_eq!(anthropic["thinking"]["budget_tokens"], 8192);
        assert!(anthropic.get("reasoning_effort").is_none(), "no OpenAI field leak");
        assert_eq!(anthropic["max_tokens"], 8192 + 4096, "max_tokens must exceed budget");

        let mut anthropic = json!({"messages": [], "thinking": {"type": "enabled"}});
        apply_model_variant(&mut anthropic, ClientFormat::Anthropic, &ModelVariant::NoThink);
        assert!(anthropic.get("thinking").is_none());
    }

    #[test]
    fn auto_free_is_every_marked_free_model() {
        let app = test_app(
            r#"
            [server]
            [providers.a]
            base_url = "http://127.0.0.1:1/c"
            models = [{ id = "free1", free = true }, "paid1"]
            [providers.b]
            base_url = "http://127.0.0.1:1/c"
            models = [{ id = "free2", free = true }]
            "#,
            "auto_free",
        );
        let ids: Vec<String> = app
            .catalog
            .resolve(&app.cfg, "auto/free")
            .iter()
            .map(|c| c.full_id())
            .collect();
        assert_eq!(ids, vec!["a/free1", "b/free2"]);
    }

    #[test]
    fn quota_failure_classification() {
        assert!(is_quota_failure(Some(402), "insufficient credits"));
        assert!(is_quota_failure(Some(429), "You have exceeded your daily quota"));
        assert!(is_quota_failure(None, "daily request limit"));
        assert!(is_quota_failure(None, "monthly token limit"));
        // A bare rate limit / 5xx / auth error is transient, not exhaustion.
        assert!(!is_quota_failure(Some(429), "rate limit exceeded"));
        assert!(!is_quota_failure(Some(500), "boom"));
        assert!(!is_quota_failure(Some(401), "invalid key"));
        assert!(!is_quota_failure(None, "rpm limit"));
        assert!(!is_quota_failure(None, "network error"));
    }

    #[test]
    fn headroom_ranking_prefers_the_less_used_peer() {
        let app = test_app(
            r#"
            [server]
            [providers.a]
            base_url = "http://127.0.0.1:1/c"
            models = ["m"]
            [providers.a.limits]
            daily_requests = 1
            [providers.b]
            base_url = "http://127.0.0.1:1/c"
            models = ["m"]
            [providers.b.limits]
            daily_requests = 1
            [groups.g]
            headroom = true
            models = ["a/m", "b/m"]
            [groups.h]
            models = ["a/m", "b/m"]
            "#,
            "headroom",
        );
        let limits = app.cfg.providers.get("a").unwrap().limits.as_ref().unwrap();
        let w = crate::usage::current_windows(limits, jiff::Timestamp::now()).unwrap();
        app.state.record_usage("a", w.day_start, w.month_start, 0, 0).unwrap();
        let ids = |group: &str| -> Vec<String> {
            resolve_candidates(&app.catalog, &app.cfg, &app.state, group, None)
                .iter()
                .map(|c| c.full_id())
                .collect()
        };
        assert_eq!(ids("g"), vec!["b/m", "a/m"], "the untouched peer must lead");
        assert_eq!(ids("h"), vec!["a/m", "b/m"], "without the flag, config order wins");
        assert_eq!(
            app.catalog.group_config(&app.cfg, "claude/g").unwrap().headroom,
            Some(true),
            "the claude/ mirror resolves the same group policy"
        );
    }

    #[tokio::test]
    async fn paid_reserve_held_after_a_transient_failure() {
        use std::sync::Mutex;
        use axum::routing::post;
        let paid_seen: Arc<Mutex<bool>> = Arc::new(Mutex::new(false));
        let cap = paid_seen.clone();
        let router = axum::Router::new()
            .route("/free", post(|| async {
                (axum::http::StatusCode::INTERNAL_SERVER_ERROR, r#"{"error":"boom"}"#)
            }))
            .route(
                "/paid",
                post(move || {
                    let cap = cap.clone();
                    async move {
                        *cap.lock().unwrap() = true;
                        axum::Json(json!({"id": "x", "choices": [{"index": 0,
                            "message": {"role": "assistant", "content": "ok"},
                            "finish_reason": "stop"}],
                            "usage": {"prompt_tokens": 1, "completion_tokens": 1}}))
                    }
                }),
            );
        let base = mock_server(router).await;
        let app = test_app(
            &format!(
                r#"
                [server]
                [providers.f]
                base_url = "{base}/free"
                models = [{{ id = "m", free = true }}]
                [providers.p]
                base_url = "{base}/paid"
                models = ["m"]
                [groups.g]
                fallback_only_on_quota_exhaustion = true
                models = ["f/m", "p/m"]
                "#
            ),
            "reserve_transient",
        );
        let payload = json!({"model": "g", "messages": [{"role": "user", "content": "hi"}]});
        let _ = handle_chat(app, ClientFormat::Openai, payload, ClientContext::default()).await;
        assert!(*paid_seen.lock().unwrap() == false, "a 500 must hold the paid reserve");
    }

    #[tokio::test]
    async fn paid_reserve_used_after_quota_exhaustion() {
        use axum::routing::post;
        let router = axum::Router::new()
            .route("/free", post(|| async {
                (
                    axum::http::StatusCode::TOO_MANY_REQUESTS,
                    r#"{"error":{"message":"You have exceeded your daily quota"}}"#,
                )
            }))
            .route("/paid", post(|| async {
                axum::Json(json!({"id": "x", "choices": [{"index": 0,
                    "message": {"role": "assistant", "content": "ok"},
                    "finish_reason": "stop"}],
                    "usage": {"prompt_tokens": 1, "completion_tokens": 1}}))
            }));
        let base = mock_server(router).await;
        let app = test_app(
            &format!(
                r#"
                [server]
                [providers.f]
                base_url = "{base}/free"
                models = [{{ id = "m", free = true }}]
                [providers.p]
                base_url = "{base}/paid"
                models = ["m"]
                [groups.g]
                fallback_only_on_quota_exhaustion = true
                models = ["f/m", "p/m"]
                "#
            ),
            "reserve_quota",
        );
        let payload = json!({"model": "g", "messages": [{"role": "user", "content": "hi"}]});
        match handle_chat(app, ClientFormat::Openai, payload, ClientContext::default()).await {
            Outcome::Json { provider, .. } => {
                assert_eq!(provider.as_deref(), Some("p/m"), "quota exhaustion releases the reserve")
            }
            Outcome::Stream { .. } => panic!("expected json"),
        }
    }

    #[tokio::test]
    async fn absent_tools_is_not_materialized_as_null() {
        use std::sync::Mutex;
        use axum::routing::post;
        let seen: Arc<Mutex<Option<Value>>> = Arc::new(Mutex::new(None));
        let capture = seen.clone();
        let router = axum::Router::new().route(
            "/c",
            post(move |axum::Json(body): axum::Json<Value>| {
                let capture = capture.clone();
                async move {
                    *capture.lock().unwrap() = Some(body);
                    axum::Json(json!({"id": "x", "choices": [{"index": 0,
                        "message": {"role": "assistant", "content": "ok"},
                        "finish_reason": "stop"}],
                        "usage": {"prompt_tokens": 1, "completion_tokens": 1}}))
                }
            }),
        );
        let base = mock_server(router).await;
        let app = test_app(
            &format!(
                r#"
                [server]
                [providers.a]
                base_url = "{base}/c"
                models = ["m"]
                "#
            ),
            "absent_tools",
        );
        let payload = json!({"model": "a/m", "messages": [{"role": "user", "content": "hi"}]});
        let _ = handle_chat(app, ClientFormat::Openai, payload, ClientContext::default()).await;
        let body = seen.lock().unwrap().take().expect("upstream was called");
        assert!(body.get("tools").is_none(), "absent tools must stay absent: {body}");
    }

    #[test]
    fn route_pin_walks_first_with_chain_fallback() {
        let app = test_app(
            r#"
            [server]
            [providers.a]
            base_url = "http://127.0.0.1:1/a"
            models = ["m1"]
            [providers.b]
            base_url = "http://127.0.0.1:1/b"
            models = ["m2"]
            [groups.free]
            models = ["a/m1", "b/m2"]
            [groups.other]
            models = ["a/m1", "b/m2"]
            "#,
            "route_pin",
        );

        // No pin: config order.
        let ids: Vec<String> = resolve_candidates(&app.catalog, &app.cfg, &app.state, "free", None)
            .iter()
            .map(|c| c.full_id())
            .collect();
        assert_eq!(ids, ["a/m1", "b/m2"]);

        // Pinned: the pin leads, the rest of the chain follows, no duplicate.
        app.state.kv_set(&route_pin_key("free"), "b/m2").unwrap();
        let ids: Vec<String> = resolve_candidates(&app.catalog, &app.cfg, &app.state, "free", None)
            .iter()
            .map(|c| c.full_id())
            .collect();
        assert_eq!(ids, ["b/m2", "a/m1"], "pin first, chain as fallback");

        // The pin is the group's own: a sibling group keeps its config order,
        // and the claude/ mirror of the pinned group shares its pin.
        let ids: Vec<String> = resolve_candidates(&app.catalog, &app.cfg, &app.state, "other", None)
            .iter()
            .map(|c| c.full_id())
            .collect();
        assert_eq!(ids, ["a/m1", "b/m2"], "a pin must not leak into another group");
        let ids: Vec<String> =
            resolve_candidates(&app.catalog, &app.cfg, &app.state, "claude/free", None)
                .iter()
                .map(|c| c.full_id())
                .collect();
        assert_eq!(ids, ["b/m2", "a/m1"], "the mirror spelling shares the pin");

        // Explicit model requests ignore the pin entirely.
        let ids: Vec<String> = resolve_candidates(&app.catalog, &app.cfg, &app.state, "a/m1", None)
            .iter()
            .map(|c| c.full_id())
            .collect();
        assert_eq!(ids, ["a/m1"]);

        // A pin that stopped resolving degrades to the plain chain.
        app.state.kv_set(&route_pin_key("free"), "gone/nope").unwrap();
        let ids: Vec<String> = resolve_candidates(&app.catalog, &app.cfg, &app.state, "free", None)
            .iter()
            .map(|c| c.full_id())
            .collect();
        assert_eq!(ids, ["a/m1", "b/m2"]);

        // A pin under a REAL provider but to an unlisted model does too:
        // resolve() fabricates a candidate for it, and without the is_listed
        // gate that phantom would lead every group walk (and a 400 "unknown
        // model" is Fatal — no failover).
        app.state.kv_set(&route_pin_key("free"), "a/ghost").unwrap();
        let ids: Vec<String> = resolve_candidates(&app.catalog, &app.cfg, &app.state, "free", None)
            .iter()
            .map(|c| c.full_id())
            .collect();
        assert_eq!(ids, ["a/m1", "b/m2"], "phantom pin must not enter the walk");
    }

    /// Session affinity: the conversation's last winner walks first, a manual
    /// pin outranks it, and a stale/unlisted binding degrades to the plain
    /// chain (then rebinds to the actual winner).
    #[test]
    fn session_affinity_walks_the_last_winner_first() {
        let app = test_app(
            r#"
            [server]
            [providers.a]
            base_url = "http://127.0.0.1:1/a"
            models = ["m1"]
            [providers.b]
            base_url = "http://127.0.0.1:1/b"
            models = ["m2"]
            [providers.c]
            base_url = "http://127.0.0.1:1/c"
            models = ["m3"]
            [groups.free]
            models = ["a/m1", "b/m2"]
            [groups.other]
            models = ["c/m3"]
            "#,
            "session_affinity",
        );

        // A fresh binding leads the walk over the config-order head.
        app.state.session_set("uid:u1", "b/m2");
        let ids: Vec<String> =
            resolve_candidates(&app.catalog, &app.cfg, &app.state, "free", Some("uid:u1"))
                .iter()
                .map(|c| c.full_id())
                .collect();
        assert_eq!(ids, ["b/m2", "a/m1"], "bound candidate walks first");
        // Other conversations are unaffected.
        let ids: Vec<String> =
            resolve_candidates(&app.catalog, &app.cfg, &app.state, "free", Some("uid:other"))
                .iter()
                .map(|c| c.full_id())
                .collect();
        assert_eq!(ids, ["a/m1", "b/m2"]);

        // A manual pin outranks the affinity binding.
        app.state.kv_set(&route_pin_key("free"), "a/m1").unwrap();
        let ids: Vec<String> =
            resolve_candidates(&app.catalog, &app.cfg, &app.state, "free", Some("uid:u1"))
                .iter()
                .map(|c| c.full_id())
                .collect();
        assert_eq!(ids, ["a/m1", "b/m2"], "pin first, affinity never leads");
        app.state.kv_set(&route_pin_key("free"), "").unwrap();

        // A binding to a model outside this group is ignored: the opener hash
        // is shared across groups, and another group's winner must not lead.
        app.state.session_set("uid:u3", "c/m3");
        let ids: Vec<String> =
            resolve_candidates(&app.catalog, &app.cfg, &app.state, "free", Some("uid:u3"))
                .iter()
                .map(|c| c.full_id())
                .collect();
        assert_eq!(ids, ["a/m1", "b/m2"], "a foreign group's binding never leads");

        // An unlisted binding degrades (is_listed gate, like the pin).
        app.state.session_set("uid:u2", "gone/nope");
        let ids: Vec<String> =
            resolve_candidates(&app.catalog, &app.cfg, &app.state, "free", Some("uid:u2"))
                .iter()
                .map(|c| c.full_id())
                .collect();
        assert_eq!(ids, ["a/m1", "b/m2"]);

        // An EXPIRED binding is ignored: rewrite the row with an old `seen`.
        app.state
            .kv_set("session:uid:u1", r#"{"candidate":"b/m2","seen":1}"#)
            .unwrap();
        let ids: Vec<String> =
            resolve_candidates(&app.catalog, &app.cfg, &app.state, "free", Some("uid:u1"))
                .iter()
                .map(|c| c.full_id())
                .collect();
        assert_eq!(ids, ["a/m1", "b/m2"], "TTL-expired binding must not lead");
    }

    /// The session key extraction ladder: metadata.user_id (Claude Code),
    /// `user` (OpenAI shape), then a stable FNV hash of the first message.
    #[test]
    fn session_key_extraction_ladder() {
        let uid = json!({"metadata": {"user_id": "abc123"}, "messages": [{"role": "user", "content": "hi"}]});
        assert_eq!(session_key(&uid).as_deref(), Some("uid:abc123"));
        let user = json!({"user": "opencode-session-7", "messages": [{"role": "user", "content": "hi"}]});
        assert_eq!(session_key(&user).as_deref(), Some("user:opencode-session-7"));
        // Hash form is stable across identical first messages…
        let h1 = json!({"messages": [{"role": "user", "content": "opener"}]});
        let h2 = json!({"messages": [{"role": "user", "content": "opener"}]});
        assert_eq!(session_key(&h1), session_key(&h2));
        // …and the hash is FNV-1a (fixed constant, stable across restarts).
        let hash = session_key(&h1).unwrap();
        assert!(hash.starts_with("hash:"));
        // Blocks-shaped content and empty content degrade sanely.
        let blocks = json!({"messages": [{"role": "user", "content": [{"type": "text", "text": "opener"}]}]});
        assert_eq!(session_key(&blocks), session_key(&h1));
        assert_eq!(session_key(&json!({"messages": [{"role": "user", "content": ""}]})), None);
        assert_eq!(session_key(&json!({})), None);
    }

    #[tokio::test]
    async fn dead_stream_fails_over_before_first_event() {
        use axum::routing::post;
        // Provider a 200s and immediately ends the body; b streams properly.
        let router = axum::Router::new()
            .route("/a", post(|| async { "" }))
            .route("/b", post(|| async {
                "data: {\"id\":\"x\",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"content\":\"hello\"}}]}\n\ndata: [DONE]\n\n"
            }));
        let base = mock_server(router).await;
        let app = test_app(
            &format!(
                r#"
                [server]
                [providers.a]
                base_url = "{base}/a"
                models = ["m"]
                [providers.b]
                base_url = "{base}/b"
                models = ["m"]
                [groups.free]
                models = ["a/m", "b/m"]
                "#
            ),
            "dead_stream",
        );

        let payload = json!({"model": "free", "stream": true,
            "messages": [{"role": "user", "content": "hi"}]});
        let out = handle_chat(app.clone(), ClientFormat::Openai, payload, ClientContext::default())
            .await;
        match out {
            Outcome::Stream { provider, body, .. } => {
                assert_eq!(provider, "b/m");
                let bytes = axum::body::to_bytes(body, 1 << 20).await.unwrap();
                let text = String::from_utf8_lossy(&bytes);
                assert!(text.contains("hello"), "held first chunk must reach the client: {text}");
            }
            Outcome::Json { status, body, .. } => panic!("expected stream, got {status}: {body}"),
        }
        // The dead model cooled down model-scoped, not provider-wide.
        assert!(app.state.cooldown("a", "m").is_some());
        assert!(app.state.cooldown("a", "other").is_none());
    }

    /// A streamed turn must not die at timeout_secs: the old per-request
    /// total timeout killed any stream longer than 1×timeout_secs mid-body,
    /// truncating the answer with a clean-looking end-of-turn. Chunks here
    /// keep arriving (each gap < stall = timeout_secs) past the total that
    /// used to bound the whole body — every chunk must reach the client.
    #[tokio::test]
    async fn long_stream_survives_past_timeout_secs() {
        use axum::routing::post;
        fn chunk(c: &str) -> String {
            format!("data: {{\"choices\":[{{\"index\":0,\"delta\":{{\"content\":\"{c}\"}}}}]}}\n\n")
        }
        let router = axum::Router::new().route("/slow", post(|| async {
            let stream = futures_util::stream::unfold(0u8, |n| async move {
                let (chunk, next, delay_ms) = match n {
                    0 => (Bytes::from(chunk("one")), 1u8, 0),
                    1 => (Bytes::from(chunk("-two")), 2, 550),
                    2 => (Bytes::from(chunk("-three")), 3, 550),
                    3 => (Bytes::from("data: [DONE]\n\n"), 4, 0),
                    _ => return None,
                };
                if delay_ms > 0 {
                    tokio::time::sleep(Duration::from_millis(delay_ms)).await;
                }
                Some((Ok::<Bytes, std::io::Error>(chunk), next))
            });
            axum::http::Response::builder()
                .body(axum::body::Body::from_stream(stream))
                .unwrap()
        }));
        let base = mock_server(router).await;
        let app = test_app(
            &format!(
                r#"
                [server]
                [providers.slow]
                base_url = "{base}/slow"
                timeout_secs = 1
                models = ["m"]
                "#
            ),
            "long_stream",
        );

        let payload = json!({"model": "slow/m", "stream": true,
            "messages": [{"role": "user", "content": "hi"}]});
        let out = handle_chat(app.clone(), ClientFormat::Openai, payload, ClientContext::default())
            .await;
        match out {
            Outcome::Stream { body, .. } => {
                let bytes = axum::body::to_bytes(body, 1 << 20).await.unwrap();
                let text = String::from_utf8_lossy(&bytes);
                for part in ["one", "-two", "-three", "[DONE]"] {
                    assert!(text.contains(part), "missing {part:?}: {text}");
                }
            }
            Outcome::Json { status, body, .. } => panic!("expected stream, got {status}: {body}"),
        }
    }

    /// opencode.ai errors on a request without `x-opencode-session`, and the
    /// continuation after a served tool call is a second request to the same
    /// upstream: both calls of one turn must carry the same session id.
    #[tokio::test]
    async fn continuation_carries_the_opencode_session_header() {
        use axum::response::IntoResponse;
        use axum::routing::post;
        let calls = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let seen = Arc::new(std::sync::Mutex::new(Vec::<Option<String>>::new()));
        let counter = calls.clone();
        let sink = seen.clone();
        let router = axum::Router::new().route(
            "/m",
            post(move |headers: axum::http::HeaderMap| {
                let counter = counter.clone();
                let sink = sink.clone();
                async move {
                    sink.lock().unwrap().push(
                        headers
                            .get("x-opencode-session")
                            .and_then(|v| v.to_str().ok())
                            .map(str::to_string),
                    );
                    let n = counter.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                    let sse = if n == 0 {
                        concat!(
                            "data: {\"id\":\"c1\",\"choices\":[{\"index\":0,\"delta\":{\"tool_calls\":[",
                            "{\"index\":0,\"id\":\"call_1\",\"type\":\"function\",",
                            "\"function\":{\"name\":\"pxy_datetime\",\"arguments\":\"{}\"}}]}}]}\n\n",
                            "data: {\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"tool_calls\"}]}\n\n",
                            "data: [DONE]\n\n",
                        )
                    } else {
                        concat!(
                            "data: {\"id\":\"c2\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"now\"}}]}\n\n",
                            "data: {\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n",
                            "data: [DONE]\n\n",
                        )
                    };
                    axum::http::Response::builder()
                        .header("content-type", "text/event-stream")
                        .body(axum::body::Body::from(sse))
                        .unwrap()
                        .into_response()
                }
            }),
        );
        let base = mock_server(router).await;
        let app = test_app(
            &format!(
                r#"
                [server]
                [providers.opencode-test]
                base_url = "{base}/m"
                models = ["m"]
                [groups.free]
                models = ["opencode-test/m"]
                "#
            ),
            "continuation_session",
        );
        let payload = json!({
            "model": "free",
            "stream": true,
            "messages": [{"role": "user", "content": "what time is it?"}],
            "tools": [{"type": "pxy:datetime"}],
        });
        let out = handle_chat(app.clone(), ClientFormat::Openai, payload, ClientContext::default())
            .await;
        if let Outcome::Stream { body, .. } = out {
            let _ = axum::body::to_bytes(body, 1 << 20).await.unwrap();
        } else {
            panic!("expected stream");
        }
        let seen = seen.lock().unwrap().clone();
        assert_eq!(seen.len(), 2, "two upstream calls: {seen:?}");
        assert!(seen[0].is_some(), "first call carries the session id");
        assert_eq!(seen[0], seen[1], "the continuation carries the same id: {seen:?}");
    }

    /// A `format = "responses"` model: the body goes out as a Responses
    /// request on the `/responses` sibling of the provider's chat URL, the
    /// events come back as chat chunks, and a served tool's continuation
    /// takes the same wire. Event shapes are Zen Go's, captured live.
    #[tokio::test]
    async fn responses_upstream_is_translated_at_the_wire() {
        use axum::response::IntoResponse;
        use axum::routing::post;
        let calls = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let seen = Arc::new(std::sync::Mutex::new(Vec::<Value>::new()));
        let counter = calls.clone();
        let sink = seen.clone();
        let router = axum::Router::new()
            .route("/v1/chat/completions", post(|| async { "wrong wire" }))
            .route(
                "/v1/responses",
                post(move |axum::Json(body): axum::Json<Value>| {
                    let counter = counter.clone();
                    let sink = sink.clone();
                    async move {
                        sink.lock().unwrap().push(body.clone());
                        let n = counter.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                        if body["stream"] != true {
                            return axum::Json(json!({
                                "id": "resp_j", "object": "response", "created_at": 5, "status": "completed",
                                "model": "luna", "output": [{"type": "message", "content": [
                                    {"type": "output_text", "text": "plain"}]}],
                                "usage": {"input_tokens": 4, "output_tokens": 1, "total_tokens": 5}
                            }))
                            .into_response();
                        }
                        let sse = if n == 0 {
                            concat!(
                                "event: response.created\ndata: {\"type\":\"response.created\",\"response\":{\"id\":\"resp_1\",\"model\":\"luna\",\"created_at\":7}}\n\n",
                                "event: response.output_item.added\ndata: {\"type\":\"response.output_item.added\",\"output_index\":0,\"item\":{\"id\":\"fc_1\",\"type\":\"function_call\",\"name\":\"pxy_datetime\",\"call_id\":\"call_1\",\"arguments\":\"\"}}\n\n",
                                "event: response.function_call_arguments.delta\ndata: {\"type\":\"response.function_call_arguments.delta\",\"output_index\":0,\"item_id\":\"fc_1\",\"delta\":\"{}\"}\n\n",
                                "event: response.completed\ndata: {\"type\":\"response.completed\",\"response\":{\"id\":\"resp_1\",\"model\":\"luna\",\"status\":\"completed\",\"output\":[{\"type\":\"function_call\",\"call_id\":\"call_1\",\"name\":\"pxy_datetime\",\"arguments\":\"{}\"}],\"usage\":{\"input_tokens\":10,\"output_tokens\":2,\"total_tokens\":12}}}\n\n",
                            )
                        } else {
                            concat!(
                                "event: response.created\ndata: {\"type\":\"response.created\",\"response\":{\"id\":\"resp_2\",\"model\":\"luna\",\"created_at\":8}}\n\n",
                                "event: response.reasoning_summary_text.delta\ndata: {\"type\":\"response.reasoning_summary_text.delta\",\"item_id\":\"rs_1\",\"delta\":\"clock\"}\n\n",
                                "event: response.output_text.delta\ndata: {\"type\":\"response.output_text.delta\",\"item_id\":\"msg_1\",\"delta\":\"It is now.\"}\n\n",
                                "event: response.completed\ndata: {\"type\":\"response.completed\",\"response\":{\"id\":\"resp_2\",\"model\":\"luna\",\"status\":\"completed\",\"output\":[],\"usage\":{\"input_tokens\":20,\"output_tokens\":3,\"total_tokens\":23}}}\n\n",
                            )
                        };
                        axum::http::Response::builder()
                            .header("content-type", "text/event-stream")
                            .body(axum::body::Body::from(sse))
                            .unwrap()
                            .into_response()
                    }
                }),
            );
        let base = mock_server(router).await;
        let app = test_app(
            &format!(
                r#"
                [server]
                [providers.p]
                base_url = "{base}/v1/chat/completions"
                models = [{{ id = "luna", format = "responses" }}]
                [groups.free]
                models = ["p/luna"]
                "#
            ),
            "responses_upstream",
        );

        // Streaming, with a served tool: two calls on the Responses wire.
        let payload = json!({
            "model": "free",
            "stream": true,
            "max_tokens": 50,
            "reasoning_effort": "low",
            "messages": [{"role": "system", "content": "Be brief."},
                         {"role": "user", "content": "what time is it?"}],
            "tools": [{"type": "pxy:datetime"}],
        });
        let out = handle_chat(app.clone(), ClientFormat::Openai, payload, ClientContext::default())
            .await;
        let text = match out {
            Outcome::Stream { body, .. } => {
                let bytes = axum::body::to_bytes(body, 1 << 20).await.unwrap();
                String::from_utf8_lossy(&bytes).to_string()
            }
            Outcome::Json { status, body, .. } => panic!("expected stream, got {status}: {body}"),
        };
        assert!(text.contains("\"reasoning_content\":\"clock\""), "{text}");
        assert!(text.contains("\"content\":\"It is now.\""), "{text}");
        assert!(text.contains("\"finish_reason\":\"stop\""), "{text}");
        assert!(text.contains("data: [DONE]"), "{text}");
        assert!(!text.contains("pxy_datetime"), "reserved call leaked: {text}");
        assert!(!text.contains("response.output_text"), "raw Responses event leaked: {text}");
        {
            let seen = seen.lock().unwrap();
            assert_eq!(seen.len(), 2);
            let first = &seen[0];
            assert_eq!(first["stream"], true);
            assert_eq!(first["store"], false);
            assert_eq!(first["max_output_tokens"], 50);
            assert_eq!(first["reasoning"]["effort"], "low");
            assert_eq!(first["instructions"], "Be brief.");
            assert_eq!(first["input"][0]["role"], "user");
            assert_eq!(first["tools"][0]["name"], "pxy_datetime");
            assert!(first.get("messages").is_none(), "chat body reached the Responses wire");
            // The continuation replays the call and its output as items.
            let second = &seen[1];
            let items = second["input"].as_array().unwrap();
            assert_eq!(items[1]["type"], "function_call");
            assert_eq!(items[1]["call_id"], "call_1");
            assert!(items[1].get("id").is_none());
            assert_eq!(items[2]["type"], "function_call_output");
            assert_eq!(items[2]["call_id"], "call_1");
        }
        // Both legs were metered from the translated usage.
        let usage = app.state.model_usage_rows().unwrap();
        let row = usage.iter().find(|r| r.model == "luna").expect("usage row");
        assert_eq!(row.input_tokens, 30);
        assert_eq!(row.output_tokens, 5);

        // Non-streaming: the JSON body is translated to a chat completion.
        let payload = json!({"model": "p/luna", "messages": [{"role": "user", "content": "hi"}]});
        let out = handle_chat(app.clone(), ClientFormat::Openai, payload, ClientContext::default())
            .await;
        match out {
            Outcome::Json { status, body, .. } => {
                assert_eq!(status, 200, "{body}");
                assert_eq!(body["object"], "chat.completion");
                assert_eq!(body["choices"][0]["message"]["content"], "plain");
                assert_eq!(body["choices"][0]["finish_reason"], "stop");
                assert_eq!(body["usage"]["prompt_tokens"], 4);
            }
            Outcome::Stream { .. } => panic!("expected json"),
        }
    }

    /// The continuation after a served tool call is a stream too, and the
    /// same total-timeout rule applies: an answer whose chunks keep arriving
    /// past timeout_secs must reach the client whole, not be cut at the
    /// total with a clean-looking stop.
    #[tokio::test]
    async fn continuation_stream_survives_past_timeout_secs() {
        use axum::response::IntoResponse;
        use axum::routing::post;
        fn chunk(c: &str) -> String {
            format!("data: {{\"choices\":[{{\"index\":0,\"delta\":{{\"content\":\"{c}\"}}}}]}}\n\n")
        }
        let calls = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let counter = calls.clone();
        let router = axum::Router::new().route(
            "/m",
            post(move || {
                let counter = counter.clone();
                async move {
                    let n = counter.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                    if n == 0 {
                        let sse = concat!(
                            "data: {\"id\":\"c1\",\"choices\":[{\"index\":0,\"delta\":{\"tool_calls\":[",
                            "{\"index\":0,\"id\":\"call_1\",\"type\":\"function\",",
                            "\"function\":{\"name\":\"pxy_datetime\",\"arguments\":\"{}\"}}]}}]}\n\n",
                            "data: {\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"tool_calls\"}]}\n\n",
                            "data: [DONE]\n\n",
                        );
                        return axum::http::Response::builder()
                            .header("content-type", "text/event-stream")
                            .body(axum::body::Body::from(sse))
                            .unwrap()
                            .into_response();
                    }
                    // The answer: three chunks spread past timeout_secs = 1.
                    let stream = futures_util::stream::unfold(0u8, |k| async move {
                        let (c, next, delay_ms) = match k {
                            0 => (Bytes::from(chunk("one")), 1u8, 0),
                            1 => (Bytes::from(chunk("-two")), 2, 550),
                            2 => (Bytes::from(chunk("-three")), 3, 550),
                            3 => (
                                Bytes::from("data: {\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}]}\n\ndata: [DONE]\n\n"),
                                4,
                                0,
                            ),
                            _ => return None,
                        };
                        if delay_ms > 0 {
                            tokio::time::sleep(Duration::from_millis(delay_ms)).await;
                        }
                        Some((Ok::<Bytes, std::io::Error>(c), next))
                    });
                    axum::http::Response::builder()
                        .header("content-type", "text/event-stream")
                        .body(axum::body::Body::from_stream(stream))
                        .unwrap()
                        .into_response()
                }
            }),
        );
        let base = mock_server(router).await;
        let app = test_app(
            &format!(
                r#"
                [server]
                [providers.p]
                base_url = "{base}/m"
                timeout_secs = 1
                models = ["m"]
                [groups.free]
                models = ["p/m"]
                "#
            ),
            "continuation_long_stream",
        );
        let payload = json!({
            "model": "free",
            "stream": true,
            "messages": [{"role": "user", "content": "what time is it?"}],
            "tools": [{"type": "pxy:datetime"}],
        });
        let out = handle_chat(app.clone(), ClientFormat::Openai, payload, ClientContext::default())
            .await;
        match out {
            Outcome::Stream { body, .. } => {
                let bytes = axum::body::to_bytes(body, 1 << 20).await.unwrap();
                let text = String::from_utf8_lossy(&bytes);
                for part in ["one", "-two", "-three", "\"finish_reason\":\"stop\"", "[DONE]"] {
                    assert!(text.contains(part), "missing {part:?}: {text}");
                }
                assert!(!text.contains("pxy_datetime"), "reserved call leaked: {text}");
            }
            Outcome::Json { status, body, .. } => panic!("expected stream, got {status}: {body}"),
        }
        assert_eq!(calls.load(std::sync::atomic::Ordering::SeqCst), 2);
    }

    /// An upstream that dies mid-body leaves an OpenAI client stream with no
    /// `finish_reason`. Clients (pi, the SDKs) read that as a protocol error —
    /// "Stream ended without finish_reason" — so pxy closes the turn itself,
    /// after whatever content already reached the wire.
    #[tokio::test]
    async fn mid_stream_death_closes_an_openai_turn() {
        use axum::routing::post;
        let router = axum::Router::new().route(
            "/die",
            post(|| async {
                let stream = futures_util::stream::unfold(0u8, |n| async move {
                    match n {
                        0 => Some((
                            Ok::<Bytes, std::io::Error>(Bytes::from(
                                "data: {\"choices\":[{\"index\":0,\"delta\":{\"content\":\"half\"}}]}\n\n",
                            )),
                            1,
                        )),
                        // Let the first chunk reach reqwest (and pxy commit the
                        // stream) before the connection dies.
                        1 => {
                            tokio::time::sleep(Duration::from_millis(200)).await;
                            Some((Err(std::io::Error::other("connection reset")), 2))
                        }
                        _ => None,
                    }
                });
                axum::http::Response::builder()
                    .body(axum::body::Body::from_stream(stream))
                    .unwrap()
            }),
        );
        let base = mock_server(router).await;
        let app = test_app(
            &format!(
                r#"
                [server]
                [providers.p]
                base_url = "{base}/die"
                models = ["m"]
                "#
            ),
            "mid_stream_death",
        );

        let payload = json!({"model": "p/m", "stream": true,
            "messages": [{"role": "user", "content": "hi"}]});
        let out = handle_chat(app, ClientFormat::Openai, payload, ClientContext::default()).await;
        match out {
            Outcome::Stream { body, .. } => {
                let bytes = axum::body::to_bytes(body, 1 << 20).await.unwrap();
                let text = String::from_utf8_lossy(&bytes);
                assert!(text.contains("half"), "content before the death must survive: {text}");
                assert!(text.contains("\"finish_reason\":\"stop\""), "pxy must close the turn: {text}");
                assert_eq!(text.matches("[DONE]").count(), 1, "exactly one terminator: {text}");
            }
            Outcome::Json { status, body, .. } => panic!("expected stream, got {status}: {body}"),
        }
    }

    /// The same hole with a `[DONE]` and no `finish_reason` before it: some
    /// aggregators close that way. The synthetic reason must land *before*
    /// `[DONE]`, not after — a client that stops at `[DONE]` never sees a
    /// later chunk. This request declares tools, so it takes the rewriting
    /// path a Chat Completions client with tool history actually uses.
    #[tokio::test]
    async fn done_without_finish_reason_gets_one_before_done() {
        use axum::routing::post;
        let router = axum::Router::new().route(
            "/bare",
            post(|| async {
                "data: {\"choices\":[{\"index\":0,\"delta\":{\"content\":\"hi\"}}]}\n\ndata: [DONE]\n\n"
            }),
        );
        let base = mock_server(router).await;
        let app = test_app(
            &format!(
                r#"
                [server]
                [providers.p]
                base_url = "{base}/bare"
                models = ["m"]
                "#
            ),
            "done_without_finish",
        );

        let payload = json!({"model": "p/m", "stream": true,
            "tools": [{"type": "function", "function": {"name": "bash",
                "parameters": {"type": "object", "properties": {}}}}],
            "messages": [{"role": "user", "content": "hi"}]});
        let out = handle_chat(app, ClientFormat::Openai, payload, ClientContext::default()).await;
        match out {
            Outcome::Stream { body, .. } => {
                let bytes = axum::body::to_bytes(body, 1 << 20).await.unwrap();
                let text = String::from_utf8_lossy(&bytes);
                let finish = text.find("\"finish_reason\":\"stop\"").expect("a terminal reason");
                let done = text.find("[DONE]").expect("a terminator");
                assert!(finish < done, "the reason must precede [DONE]: {text}");
                assert_eq!(text.matches("[DONE]").count(), 1, "exactly one terminator: {text}");
            }
            Outcome::Json { status, body, .. } => panic!("expected stream, got {status}: {body}"),
        }
    }

    /// The Anthropic-upstream half of the same guarantee: an Anthropic SSE
    /// stream that stops before `message_stop` reaches an OpenAI client with
    /// no `finish_reason`, so pxy must synthesize one on this path too.
    #[tokio::test]
    async fn truncated_anthropic_upstream_closes_an_openai_turn() {
        use crate::translate::sse::format_event;
        use axum::routing::post;
        let router = axum::Router::new().route(
            "/a",
            post(|| async {
                let mut s = String::new();
                s.push_str(&format_event("message_start", &json!({
                    "type": "message_start",
                    "message": {"id": "msg_1", "type": "message", "role": "assistant",
                                "model": "m", "content": [],
                                "usage": {"input_tokens": 3, "output_tokens": 0}},
                })));
                s.push_str(&format_event("content_block_start", &json!({
                    "type": "content_block_start", "index": 0,
                    "content_block": {"type": "text", "text": ""},
                })));
                s.push_str(&format_event("content_block_delta", &json!({
                    "type": "content_block_delta", "index": 0,
                    "delta": {"type": "text_delta", "text": "half"},
                })));
                // …and the upstream ends here, with no message_stop.
                s
            }),
        );
        let base = mock_server(router).await;
        let app = test_app(
            &format!(
                r#"
                [server]
                [providers.p]
                base_url = "{base}/a"
                format = "anthropic"
                models = ["m"]
                "#
            ),
            "truncated_anthropic",
        );

        let payload = json!({"model": "p/m", "stream": true,
            "messages": [{"role": "user", "content": "hi"}]});
        let out = handle_chat(app, ClientFormat::Openai, payload, ClientContext::default()).await;
        match out {
            Outcome::Stream { body, .. } => {
                let bytes = axum::body::to_bytes(body, 1 << 20).await.unwrap();
                let text = String::from_utf8_lossy(&bytes);
                assert!(text.contains("half"), "content must survive: {text}");
                assert!(text.contains("\"finish_reason\":\"stop\""), "pxy must close the turn: {text}");
                assert_eq!(text.matches("[DONE]").count(), 1, "exactly one terminator: {text}");
            }
            Outcome::Json { status, body, .. } => panic!("expected stream, got {status}: {body}"),
        }
    }

    #[tokio::test]
    async fn retry_after_backoff_recovers_single_candidate() {
        use std::sync::atomic::{AtomicUsize, Ordering};
        use axum::response::IntoResponse;
        let calls = Arc::new(AtomicUsize::new(0));
        let seen = calls.clone();
        let router = axum::Router::new().route(
            "/flaky",
            axum::routing::post(move || {
                let calls = calls.clone();
                async move {
                    if calls.fetch_add(1, Ordering::SeqCst) == 0 {
                        (
                            axum::http::StatusCode::TOO_MANY_REQUESTS,
                            [("retry-after", "1")],
                            r#"{"error":{"message":"slow down"}}"#,
                        )
                            .into_response()
                    } else {
                        axum::Json(json!({
                            "id": "x",
                            "choices": [{"index": 0,
                                "message": {"role": "assistant", "content": "recovered"},
                                "finish_reason": "stop"}],
                            "usage": {"prompt_tokens": 1, "completion_tokens": 1},
                        }))
                        .into_response()
                    }
                }
            }),
        );
        let base = mock_server(router).await;
        let app = test_app(
            &format!(
                r#"
                [server]
                [providers.c]
                base_url = "{base}/flaky"
                models = ["m"]
                "#
            ),
            "retry_after",
        );

        let started = std::time::Instant::now();
        let payload = json!({"model": "c/m", "messages": [{"role": "user", "content": "hi"}]});
        let out = handle_chat(app, ClientFormat::Openai, payload, ClientContext::default()).await;
        match out {
            Outcome::Json { status, body, .. } => {
                assert_eq!(status, 200, "expected recovery, got {body}");
                assert_eq!(body["choices"][0]["message"]["content"], "recovered");
            }
            Outcome::Stream { .. } => panic!("expected json"),
        }
        assert_eq!(seen.load(Ordering::SeqCst), 2, "exactly one retry");
        assert!(started.elapsed() >= Duration::from_secs(1), "must honor retry-after");
    }

    #[tokio::test]
    async fn auth_failure_never_retried() {
        use std::sync::atomic::{AtomicUsize, Ordering};
        use axum::response::IntoResponse;
        let calls = Arc::new(AtomicUsize::new(0));
        let seen = calls.clone();
        let router = axum::Router::new().route(
            "/auth",
            axum::routing::post(move || {
                let calls = calls.clone();
                async move {
                    calls.fetch_add(1, Ordering::SeqCst);
                    (
                        axum::http::StatusCode::UNAUTHORIZED,
                        r#"{"error":{"message":"invalid api key"}}"#,
                    )
                        .into_response()
                }
            }),
        );
        let base = mock_server(router).await;
        let app = test_app(
            &format!(
                r#"
                [server]
                [providers.d]
                base_url = "{base}/auth"
                models = ["m"]
                "#
            ),
            "auth_no_retry",
        );

        let started = std::time::Instant::now();
        let payload = json!({"model": "d/m", "messages": [{"role": "user", "content": "hi"}]});
        let out = handle_chat(app, ClientFormat::Openai, payload, ClientContext::default()).await;
        match out {
            // The client asked for exactly this model, so it gets the
            // upstream's own 401 and body — not a synthetic overloaded_error
            // that hides an invalid key behind "try again later".
            Outcome::Json { status, body, .. } => {
                assert_eq!(status, 401, "upstream error must pass through raw");
                assert_eq!(body["error"]["message"], "invalid api key", "body verbatim: {body}");
            }
            Outcome::Stream { .. } => panic!("expected json"),
        }
        assert_eq!(seen.load(Ordering::SeqCst), 1, "a dead key must not be re-fired");
        assert!(started.elapsed() < Duration::from_millis(500), "must fail fast, not back off");
    }

    #[test]
    fn context_window_errors_detected() {
        assert!(is_context_window_error(
            r#"{"error":{"message":"This model's maximum context length is 128000 tokens"}}"#
        ));
        assert!(is_context_window_error(r#"{"error":{"message":"prompt is too long: 210503 tokens"}}"#));
        assert!(is_context_window_error(r#"{"message":"input tokens exceed the configured limit"}"#));
        // The two known false positives stay out.
        assert!(!is_context_window_error(r#"{"error":{"code":"string_above_max_length"}}"#));
        assert!(!is_context_window_error(r#"{"error":{"message":"invalid 'user': maximum length"}}"#));
        // Ordinary errors don't match.
        assert!(!is_context_window_error(r#"{"error":{"message":"rate limited"}}"#));
    }

    #[tokio::test]
    async fn textual_tool_call_streams_as_real_tool_use() {
        use axum::routing::post;
        // A free-model habit: the tool call arrives as prose, split across
        // chunks, with finish_reason "stop".
        // CJK content and a spec-compliant `choices: []` usage chunk are both
        // present deliberately: each crashed a prior version of the filter.
        let router = axum::Router::new().route("/t", post(|| async {
            concat!(
                "data: {\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"content\":\"好的 On it. <tool_\"}}]}\n\n",
                "data: {\"choices\":[{\"index\":0,\"delta\":{\"content\":\"call>{\\\"name\\\": \\\"Bash\\\", \\\"arguments\\\": {\\\"cmd\\\": \\\"ls\\\"}}</tool_call>\"}}]}\n\n",
                "data: {\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n",
                "data: {\"choices\":[],\"usage\":{\"prompt_tokens\":7,\"completion_tokens\":9}}\n\n",
                "data: [DONE]\n\n",
            )
        }));
        let base = mock_server(router).await;
        let app = test_app(
            &format!(
                r#"
                [server]
                [providers.q]
                base_url = "{base}/t"
                models = ["m"]
                "#
            ),
            "textual_tools",
        );

        let payload = json!({"model": "q/m", "stream": true, "max_tokens": 100,
            "tools": [{"name": "Bash", "input_schema": {"type": "object"}}],
            "messages": [{"role": "user", "content": "list files"}]});
        let out = handle_chat(app, ClientFormat::Anthropic, payload, ClientContext::default())
            .await;
        match out {
            Outcome::Stream { body, .. } => {
                let bytes = axum::body::to_bytes(body, 1 << 20).await.unwrap();
                let text = String::from_utf8_lossy(&bytes);
                assert!(text.contains("\"type\":\"tool_use\""), "real tool_use block: {text}");
                assert!(text.contains("\"name\":\"Bash\""));
                assert!(text.contains("\"stop_reason\":\"tool_use\""),
                    "finish must map to tool_use: {text}");
                assert!(!text.contains("<tool_call>"), "markup must not leak: {text}");
                assert!(text.contains("好的 On it."), "surrounding prose survives: {text}");
            }
            Outcome::Json { status, body, .. } => panic!("expected stream, got {status}: {body}"),
        }
    }

    #[tokio::test]
    async fn usage_magic_answers_locally_in_both_dialects() {
        // No mock upstream at all: the report must never leave the process.
        let app = test_app(
            r#"
            [server]
            [providers.p]
            base_url = "http://127.0.0.1:1/unreachable"
            models = ["m"]
            "#,
            "usage_magic",
        );
        let payload = json!({"model": "free",
            "messages": [{"role": "user", "content": [{"type": "text", "text": " @@usage "}]}]});
        match handle_chat(app.clone(), ClientFormat::Anthropic, payload, ClientContext::default())
            .await
        {
            Outcome::Json { status, body, provider, .. } => {
                assert_eq!(status, 200);
                assert_eq!(provider.as_deref(), Some("pxy"));
                assert!(body["content"][0]["text"].as_str().unwrap().contains("pxy usage"));
            }
            Outcome::Stream { .. } => panic!("expected json"),
        }
        // Streaming OpenAI dialect gets protocol-correct SSE.
        let payload = json!({"model": "free", "stream": true,
            "messages": [{"role": "user", "content": "@@usage"}]});
        match handle_chat(app, ClientFormat::Openai, payload, ClientContext::default()).await {
            Outcome::Stream { provider, body, .. } => {
                assert_eq!(provider, "pxy");
                let bytes = axum::body::to_bytes(body, 1 << 20).await.unwrap();
                let text = String::from_utf8_lossy(&bytes);
                assert!(text.contains("pxy usage") && text.contains("[DONE]"), "{text}");
            }
            Outcome::Json { status, body, .. } => panic!("expected stream, got {status}: {body}"),
        }
        // A normal message containing the token mid-sentence is NOT magic.
        assert!(!is_usage_magic(
            &json!({"messages": [{"role": "user", "content": "what does @@usage do?"}]})
        ));
    }

    #[test]
    fn quota_window_bodies_classified_conservatively() {
        let daily = quota_window_cooldown(None, "You have exceeded your daily free allocation")
            .expect("daily match");
        assert_eq!(daily.0, Duration::from_secs(6 * 3600), "no limits -> 6h fallback");
        let monthly =
            quota_window_cooldown(None, r#"{"error":"monthly quota exceeded"}"#).unwrap();
        assert_eq!(monthly.0, Duration::from_secs(6 * 3600));
        let credits =
            quota_window_cooldown(None, "insufficient promotional resources").unwrap();
        assert_eq!(credits.0, Duration::from_secs(3600));
        // opencode Go's rolling window and its typed limit errors.
        let rolling = quota_window_cooldown(None, "5-hour usage limit reached. Resets in 2 hours").unwrap();
        assert_eq!(rolling.0, Duration::from_secs(30 * 60));
        let free = quota_window_cooldown(None, r#"{"type":"error","error":{"type":"FreeUsageLimitError","message":"..."}}"#).unwrap();
        assert_eq!(free.0, Duration::from_secs(3600));
        // Gemini's TRANSIENT free-tier boilerplate must not classify as a
        // window: no window word, no unambiguous credits phrase.
        assert!(quota_window_cooldown(
            None,
            "You exceeded your current quota, please check your plan and billing details"
        )
        .is_none());
        assert!(quota_window_cooldown(None, "Resource has been exhausted (e.g. check quota)")
            .is_none());
        // A window word with no quota signal at all is prose, not a verdict.
        assert!(quota_window_cooldown(None, "try again later today").is_none());

        // With limits configured, the daily horizon lands before the next
        // reset (+margin), never past ~26h.
        let limits = crate::config::Limits::default();
        let (wait, _) = quota_window_cooldown(Some(&limits), "daily request limit reached").unwrap();
        assert!(wait >= Duration::from_secs(900) && wait <= Duration::from_secs(26 * 3600 + 120),
            "got {wait:?}");
    }

    #[test]
    fn retry_after_duration_forms() {
        assert_eq!(
            parse_retry_after(&headers(&[("retry-after", "30")])),
            Some(Duration::from_secs(30))
        );
        assert_eq!(
            parse_retry_after(&headers(&[("retry-after", "5m")])),
            Some(Duration::from_secs(300))
        );
        assert_eq!(
            parse_retry_after(&headers(&[("retry-after", "2m30s")])),
            Some(Duration::from_secs(150))
        );
        // A stated multi-day reset is obeyed (a weekly allowance).
        assert_eq!(
            parse_retry_after(&headers(&[("retry-after", "172800")])),
            Some(Duration::from_secs(172_800))
        );
        // Over the sanity clamp (a week) or garbage: exponential backoff.
        assert_eq!(parse_retry_after(&headers(&[("retry-after", "700000")])), None);
        assert_eq!(parse_retry_after(&headers(&[("retry-after", "soon")])), None);
    }

    #[test]
    fn remove_param_path_prunes_empty_parents() {
        let mut body = json!({
            "thinking": {"budget_tokens": 5, "type": "enabled"},
            "output_config": {"effort": "high"},
            "top_k": 40,
        });
        remove_param_path(&mut body, "thinking.budget_tokens");
        assert!(body["thinking"]["budget_tokens"].is_null());
        assert_eq!(body["thinking"]["type"], "enabled", "siblings survive");
        remove_param_path(&mut body, "output_config.effort");
        assert!(body.get("output_config").is_none(), "emptied parent pruned");
        remove_param_path(&mut body, "top_k");
        assert!(body.get("top_k").is_none());
        remove_param_path(&mut body, "absent.path");
    }

    #[tokio::test]
    /// A 200 with an unparseable body must cool the model down: the request
    /// was already counted and cooldowns cleared on the OK headers, so without
    /// this a garbage-200 upstream gets re-attempted first on every walk.
    async fn garbage_200_cools_the_model_down() {
        use axum::routing::post;
        let router = axum::Router::new()
            .route("/a", post(|| async { "this is not json" }))
            .route("/b", post(|| async {
                axum::Json(json!({
                    "id": "x",
                    "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"},
                                "finish_reason": "stop"}],
                    "usage": {"prompt_tokens": 1, "completion_tokens": 1},
                }))
            }));
        let base = mock_server(router).await;
        let app = test_app(
            &format!(
                r#"
                [server]
                [providers.a]
                base_url = "{base}/a"
                models = ["m"]
                [providers.b]
                base_url = "{base}/b"
                models = ["m"]
                [groups.free]
                models = ["a/m", "b/m"]
                "#
            ),
            "garbage_200",
        );

        let payload = json!({"model": "free",
            "messages": [{"role": "user", "content": "hi"}]});
        let out = handle_chat(app.clone(), ClientFormat::Openai, payload, ClientContext::default())
            .await;
        match out {
            Outcome::Json { body, provider, .. } => {
                assert_eq!(body["choices"][0]["message"]["content"], "ok");
                assert_eq!(provider.as_deref(), Some("b/m"));
            }
            Outcome::Stream { .. } => panic!("expected json"),
        }
        // The garbage model sits out the next walk instead of being retried
        // first for free.
        assert!(app.state.cooldown("a", "m").is_some());
    }

    /// The walk must consult the failure-rate record BEFORE attempting: a
    /// pre-seeded flapping head-of-chain candidate gets zero upstream hits
    /// while the healthy sibling serves.
    #[tokio::test]
    async fn failure_rate_skips_a_model_before_attempting_it() {
        use axum::routing::post;
        use std::sync::atomic::{AtomicUsize, Ordering};
        let a_hits = Arc::new(AtomicUsize::new(0));
        let seen = a_hits.clone();
        let ok = || async {
            axum::Json(json!({
                "id": "x",
                "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"},
                            "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1},
            }))
        };
        let router = axum::Router::new()
            .route(
                "/a",
                post(move || {
                    let seen = seen.clone();
                    async move {
                        seen.fetch_add(1, Ordering::SeqCst);
                        ok().await
                    }
                }),
            )
            .route("/b", post(ok));
        let base = mock_server(router).await;
        let app = test_app(
            &format!(
                r#"
                [server]
                [providers.a]
                base_url = "{base}/a"
                models = ["m"]
                [providers.b]
                base_url = "{base}/b"
                models = ["m"]
                [groups.free]
                models = ["a/m", "b/m"]
                "#
            ),
            "failure_rate_walk",
        );
        // Pre-seed: a/m has failed 5 recent attempts (in-memory record).
        for _ in 0..5 {
            app.state.model_result("a", "m", false);
        }

        let payload = json!({"model": "free",
            "messages": [{"role": "user", "content": "hi"}]});
        let out = handle_chat(app.clone(), ClientFormat::Openai, payload, ClientContext::default())
            .await;
        match out {
            Outcome::Json { body, provider, .. } => {
                assert_eq!(body["choices"][0]["message"]["content"], "ok");
                assert_eq!(provider.as_deref(), Some("b/m"));
            }
            Outcome::Stream { .. } => panic!("expected json"),
        }
        assert_eq!(
            a_hits.load(Ordering::SeqCst),
            0,
            "unhealthy model must not be attempted"
        );
        // And the sibling's success was recorded — b is (still) healthy.
        assert!(!app.state.model_unhealthy("b", "m"));
    }

    /// A body-matched error rule beats the status ladder: `skip` moves the
    /// walk to the next candidate WITHOUT a cooldown, `passthrough-cooldown`
    /// returns the raw body AND cools the candidate.
    #[tokio::test]
    async fn error_rules_override_the_status_ladder() {
        use axum::routing::post;
        let router = axum::Router::new()
            .route("/a", post(|| async {
                (axum::http::StatusCode::SERVICE_UNAVAILABLE,
                 r#"{"error":{"message":"无可用渠道 (no channel available)"}}"#)
            }))
            .route("/b", post(|| async {
                axum::Json(json!({
                    "id": "x",
                    "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"},
                                "finish_reason": "stop"}],
                    "usage": {"prompt_tokens": 1, "completion_tokens": 1},
                }))
            }));
        let base = mock_server(router).await;
        let cfg_toml = |action: &str| {
            format!(
                r#"
                [server]
                [providers.a]
                base_url = "{base}/a"
                models = ["m"]
                [[providers.a.errors]]
                match = "no channel available"
                action = "{action}"
                [providers.b]
                base_url = "{base}/b"
                models = ["m"]
                [groups.free]
                models = ["a/m", "b/m"]
                "#
            )
        };

        // skip: the 503 is absorbed, b serves, a is NOT cooled (next walk
        // re-probes it — that's what skip means).
        let app = test_app(&cfg_toml("skip"), "error_rules_skip");
        let payload = json!({"model": "free", "messages": [{"role": "user", "content": "hi"}]});
        let out = handle_chat(app.clone(), ClientFormat::Openai, payload, ClientContext::default())
            .await;
        match out {
            Outcome::Json { body, provider, .. } => {
                assert_eq!(body["choices"][0]["message"]["content"], "ok");
                assert_eq!(provider.as_deref(), Some("b/m"));
            }
            Outcome::Stream { .. } => panic!("expected json"),
        }
        assert!(app.state.cooldown("a", "m").is_none(), "skip must not cool");

        // passthrough-cooldown on a SINGLE-model request: raw body passes
        // through unmodified AND the model cools.
        let app = test_app(
            &format!(
                r#"
                [server]
                [providers.a]
                base_url = "{base}/a"
                models = ["m"]
                [[providers.a.errors]]
                match = "no channel available"
                action = "passthrough-cooldown"
                "#
            ),
            "error_rules_passthrough",
        );
        let payload = json!({"model": "a/m", "messages": [{"role": "user", "content": "hi"}]});
        let out = handle_chat(app, ClientFormat::Openai, payload, ClientContext::default()).await;
        match out {
            Outcome::Json { status, body, .. } => {
                assert_eq!(status, 503);
                assert_eq!(body["error"]["message"], "无可用渠道 (no channel available)");
            }
            Outcome::Stream { .. } => panic!("expected json"),
        }
    }

    /// Multi-account: the candidate walk IS the account walk. Account "gh"
    /// 401s (auth = account-wide cooldown under `sub#gh`), so account "g"
    /// serves; the bare provider name still reports `sub/m` for the panels.
    /// `Outcome` had no header field at all, so it was
    /// structurally impossible for an upstream header to reach the client —
    /// `retry-after` and the rate-limit families included, which is what a
    /// harness's own backoff reads. They must now be relayed, minus the
    /// hop-by-hop/entity set (which describes pxy's connection to the upstream,
    /// not the one pxy is writing) and the gateway-telemetry prefixes.
    #[tokio::test]
    async fn upstream_headers_are_relayed_except_the_excluded_set() {
        use axum::response::IntoResponse;
        let router = axum::Router::new().route(
            "/m",
            axum::routing::post(|| async {
                (
                    axum::http::StatusCode::TOO_MANY_REQUESTS,
                    [
                        // Must survive — the client acts on these.
                        ("retry-after", "42"),
                        ("anthropic-ratelimit-unified-reset", "2026-09-01T00:00:00Z"),
                        ("x-ratelimit-remaining-requests", "0"),
                        ("request-id", "req_abc123"),
                        // Must be dropped: describes the upstream hop.
                        ("connection", "close"),
                        // Must be dropped: Claude Code's telemetry reads these
                        // to detect it is behind a gateway.
                        ("x-litellm-version", "1.2.3"),
                        ("helicone-id", "h-1"),
                        // Must be dropped: upstream-connection state, useless
                        // and misleading to a local agent.
                        ("set-cookie", "sid=abc; Path=/"),
                        ("alt-svc", r#"h3=":443""#),
                    ],
                    r#"{"type":"error","error":{"type":"rate_limit_error","message":"slow down"}}"#,
                )
                    .into_response()
            }),
        );
        let base = mock_server(router).await;
        let app = test_app(
            &format!(
                r#"
                [server]
                [providers.only]
                base_url = "{base}/m"
                models = ["m"]
                "#
            ),
            "hdr_relay",
        );

        let payload = json!({"model": "only/m", "messages": [{"role": "user", "content": "hi"}]});
        let out =
            handle_chat(app, ClientFormat::Anthropic, payload, ClientContext::default()).await;
        let Outcome::Json { status, headers, .. } = out else { panic!("expected json") };
        assert_eq!(status, 429);
        let got: std::collections::HashMap<String, String> = headers.into_iter().collect();

        for k in [
            "retry-after",
            "anthropic-ratelimit-unified-reset",
            "x-ratelimit-remaining-requests",
            "request-id",
        ] {
            assert!(got.contains_key(k), "{k} must be relayed; got {got:?}");
        }
        assert_eq!(got.get("retry-after").map(String::as_str), Some("42"));
        for k in [
            "connection",
            "x-litellm-version",
            "helicone-id",
            "content-length",
            "set-cookie",
            "alt-svc",
        ] {
            assert!(!got.contains_key(k), "{k} must NOT be relayed; got {got:?}");
        }
    }

    /// The single-candidate bug: a single-candidate walk replaced the upstream's
    /// real error with a synthetic 429 overloaded_error, so Claude Code's
    /// "usage limit reached, resets at ..." UI never saw the body it reads.
    /// The in-request retry must survive (a transient 429 with a near
    /// Retry-After still recovers) — only the TERMINAL answer changes.
    #[tokio::test]
    async fn exhausted_single_candidate_returns_the_real_upstream_error() {
        use std::sync::atomic::{AtomicUsize, Ordering};
        use axum::response::IntoResponse;
        let calls = Arc::new(AtomicUsize::new(0));
        let seen = calls.clone();
        // A quota 429 naming a window: non-retryable, so the walk gives up at
        // once rather than sleeping on it.
        let router = axum::Router::new().route(
            "/limited",
            axum::routing::post(move || {
                let calls = calls.clone();
                async move {
                    calls.fetch_add(1, Ordering::SeqCst);
                    (
                        axum::http::StatusCode::TOO_MANY_REQUESTS,
                        r#"{"type":"error","error":{"type":"rate_limit_error","message":"You have exceeded your daily quota. Resets at 2026-09-01T00:00:00Z"}}"#,
                    )
                        .into_response()
                }
            }),
        );
        let base = mock_server(router).await;
        let app = test_app(
            &format!(
                r#"
                [server]
                [providers.only]
                base_url = "{base}/limited"
                models = ["m"]
                "#
            ),
            "raw_terminal_error",
        );

        let payload = json!({"model": "only/m", "messages": [{"role": "user", "content": "hi"}]});
        let out =
            handle_chat(app, ClientFormat::Anthropic, payload, ClientContext::default()).await;
        match out {
            Outcome::Json { status, body, .. } => {
                assert_eq!(status, 429, "the upstream's own status");
                // The whole point: the real type and message survive, so the
                // client can say WHEN the limit resets.
                assert_eq!(body["error"]["type"], "rate_limit_error", "got {body}");
                assert!(
                    body["error"]["message"].as_str().unwrap_or("").contains("Resets at"),
                    "reset hint must survive: {body}"
                );
                assert!(
                    !body["error"]["type"].as_str().unwrap_or("").contains("overloaded"),
                    "must not be pxy's synthetic error: {body}"
                );
            }
            Outcome::Stream { .. } => panic!("expected json"),
        }
        assert_eq!(seen.load(Ordering::SeqCst), 1, "a dated quota 429 is not re-fired");
    }

    /// axum's `Json<Value>` accepts any valid JSON, so a scalar or array body
    /// reaches the router. Every downstream step indexes by key, and
    /// serde_json's IndexMut panics on a non-object — which killed the handler
    /// task instead of answering. Both dialects must return a clean 400.
    #[tokio::test]
    async fn non_object_bodies_are_rejected_not_panicked_on() {
        let app = test_app(
            r#"
            [server]
            [providers.p]
            base_url = "https://unused.example/m"
            models = ["m"]
            [groups.free]
            models = ["p/m"]
            "#,
            "non_object_body",
        );
        for body in [json!([]), json!("x"), json!(5), json!(null), json!(true)] {
            for fmt in [ClientFormat::Openai, ClientFormat::Anthropic] {
                let out =
                    handle_chat(app.clone(), fmt, body.clone(), ClientContext::default()).await;
                match out {
                    Outcome::Json { status, .. } => {
                        assert_eq!(status, 400, "body {body} ({fmt:?}) must be a clean 400");
                    }
                    Outcome::Stream { .. } => panic!("expected json for {body}"),
                }
            }
        }
    }

    /// The live bug: `pxy_web_search` was injected by anthropic_to_openai on the
    /// evidence of the client's server tool + `stream: true` alone, while the
    /// interceptor additionally required a configured search provider. With
    /// none configured the model was offered a function nothing would strip,
    /// so the client got a tool_use for a tool it never declared and the turn
    /// wedged. The tool must never reach the wire unless pxy will serve it.
    #[tokio::test]
    async fn web_search_never_reaches_an_upstream_that_pxy_cannot_serve() {
        use axum::response::IntoResponse;
        use axum::routing::post;
        let seen = Arc::new(std::sync::Mutex::new(Vec::<Value>::new()));
        let sink = seen.clone();
        let router = axum::Router::new().route(
            "/m",
            post(move |axum::Json(body): axum::Json<Value>| {
                let sink = sink.clone();
                async move {
                    sink.lock().unwrap().push(body);
                    // The request streams, so the mock must speak SSE — a JSON
                    // body would read as a dead stream and trigger failover.
                    let sse = concat!(
                        "data: {\"choices\":[{\"index\":0,",
                        "\"delta\":{\"content\":\"answered\"}}]}\n\n",
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
        let base = mock_server(router).await;
        // No [[search.providers]] — exactly Saiful's live config.
        let app = test_app(
            &format!(
                r#"
                [server]
                [providers.p]
                base_url = "{base}/m"
                models = ["m"]
                [groups.free]
                models = ["p/m"]
                "#
            ),
            "web_search_guard",
        );

        // Claude Code's shape: the dated server tool, plus a real client tool
        // so `tools` survives the strip and the request stays well-formed.
        let payload = json!({
            "model": "free",
            "stream": true,
            "messages": [{"role": "user", "content": "what happened today?"}],
            "tools": [
                {"type": "web_search_20250305", "name": "web_search", "max_uses": 3},
                {"name": "Read", "description": "read a file",
                 "input_schema": {"type": "object", "properties": {"p": {"type": "string"}}}},
            ],
        });
        let out =
            handle_chat(app.clone(), ClientFormat::Anthropic, payload, ClientContext::default())
                .await;
        // Drain the stream so the upstream call actually happens.
        if let Outcome::Stream { body, .. } = out {
            let _ = axum::body::to_bytes(body, 1 << 20).await.unwrap();
        }

        let bodies = seen.lock().unwrap();
        assert_eq!(bodies.len(), 1, "expected exactly one upstream call");
        let tools = bodies[0]["tools"].as_array().expect("client tool must survive");
        assert!(
            !tools.iter().any(|t| t["function"]["name"] == server_tools::function_name(server_tools::Tool::WebSearch)),
            "pxy_web_search must not be offered when no search provider is configured: {:?}",
            bodies[0]["tools"]
        );
        assert!(
            tools.iter().any(|t| t["function"]["name"] == "Read"),
            "the client's own tool must still be sent: {:?}",
            bodies[0]["tools"]
        );
        drop(bodies);

        // The mirror case: with a provider configured pxy CAN serve the call,
        // so the tool must still be offered — the guard must not over-strip.
        let app = test_app(
            &format!(
                r#"
                [server]
                [providers.p]
                base_url = "{base}/m"
                models = ["m"]
                [groups.free]
                models = ["p/m"]
                [[search.providers]]
                name = "brave"
                kind = "brave"
                api_key = "k"
                "#
            ),
            "web_search_guard_on",
        );
        let payload = json!({
            "model": "free",
            "stream": true,
            "messages": [{"role": "user", "content": "what happened today?"}],
            "tools": [{"type": "web_search_20250305", "name": "web_search", "max_uses": 3}],
        });
        let out =
            handle_chat(app.clone(), ClientFormat::Anthropic, payload, ClientContext::default())
                .await;
        if let Outcome::Stream { body, .. } = out {
            let _ = axum::body::to_bytes(body, 1 << 20).await.unwrap();
        }
        let bodies = seen.lock().unwrap();
        let last = bodies.last().expect("second call recorded");
        assert!(
            last["tools"]
                .as_array()
                .is_some_and(|ts| ts.iter().any(|t| t["function"]["name"] == server_tools::function_name(server_tools::Tool::WebSearch))),
            "pxy_web_search must be offered when a search provider IS configured: {:?}",
            last["tools"]
        );
    }

    /// A NON-streaming turn that asks for web search must not
    /// silently lose it. With a search provider configured, pxy streams
    /// upstream anyway (the interception lives in the stream machinery) and
    /// re-assembles the client dialect's JSON — so the upstream must see
    /// `stream: true` plus the injected function, and the client must get an
    /// ordinary JSON message back.
    #[tokio::test]
    async fn non_streaming_web_search_is_served_via_the_stream_machinery() {
        use axum::response::IntoResponse;
        use axum::routing::post;
        let seen = Arc::new(std::sync::Mutex::new(Vec::<Value>::new()));
        let sink = seen.clone();
        let router = axum::Router::new().route(
            "/m",
            post(move |axum::Json(body): axum::Json<Value>| {
                let sink = sink.clone();
                async move {
                    sink.lock().unwrap().push(body);
                    let sse = concat!(
                        "data: {\"id\":\"c1\",\"choices\":[{\"index\":0,",
                        "\"delta\":{\"content\":\"sunny\"}}]}\n\n",
                        "data: {\"choices\":[{\"index\":0,\"delta\":{},",
                        "\"finish_reason\":\"stop\"}]}\n\n",
                        "data: {\"choices\":[],\"usage\":{\"prompt_tokens\":7,",
                        "\"completion_tokens\":2}}\n\n",
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
        let base = mock_server(router).await;
        let app = test_app(
            &format!(
                r#"
                [server]
                [providers.p]
                base_url = "{base}/m"
                models = ["m"]
                [groups.free]
                models = ["p/m"]
                [[search.providers]]
                name = "brave"
                kind = "brave"
                api_key = "k"
                "#
            ),
            "nonstream_search",
        );
        let payload = json!({
            "model": "free",
            "messages": [{"role": "user", "content": "weather?"}],
            "tools": [{"type": "web_search_20250305", "name": "web_search", "max_uses": 3}],
        });
        let out =
            handle_chat(app.clone(), ClientFormat::Anthropic, payload, ClientContext::default())
                .await;
        let Outcome::Json { status, body, .. } = out else {
            panic!("non-streaming client must get JSON back");
        };
        assert_eq!(status, 200);
        assert_eq!(body["type"], "message", "{body}");
        assert_eq!(body["content"][0]["text"], "sunny", "{body}");
        assert_eq!(body["stop_reason"], "end_turn", "{body}");

        let bodies = seen.lock().unwrap();
        assert_eq!(bodies.len(), 1);
        assert_eq!(bodies[0]["stream"], true, "must stream upstream to run the loop");
        assert!(
            bodies[0]["tools"]
                .as_array()
                .is_some_and(|ts| ts.iter().any(|t| t["function"]["name"] == server_tools::function_name(server_tools::Tool::WebSearch))),
            "the search function must be offered on a non-streaming turn too: {:?}",
            bodies[0]["tools"]
        );
    }

    /// A NON-streaming Messages client whose turn ran a served tool gets the
    /// whole re-assembled message: the server_tool_use pair pxy spliced in
    /// and the model's answer after the continuation.
    #[tokio::test]
    async fn non_streaming_messages_client_keeps_the_served_turn_content() {
        // A reasoning model thinks before it calls, as deepseek does live.
        let first = format!(
            "data: {}\n\n{}",
            json!({"id": "c1", "choices": [{"index": 0, "delta": {"reasoning_content": "Let me search."}}]}),
            calls_sse(&[("call_1", "pxy_web_search", "{\"query\":\"ripgrep\"}")])
        );
        let first: &'static str = Box::leak(first.into_boxed_str());
        let (base, seen) = scripted_upstream(vec![(200, first), (200, ANSWER_SSE)]).await;
        let app = test_app(
            &format!(
                r#"
                [server]
                [providers.p]
                base_url = "{base}/m"
                models = ["m"]
                [providers.q]
                base_url = "{base}/m"
                models = ["m"]
                [groups.free]
                models = ["p/m", "q/m"]
                [[search.providers]]
                name = "brave"
                kind = "brave"
                api_key = "k"
                base_url = "http://127.0.0.1:1/search"
                "#
            ),
            "nonstream_messages_served",
        );
        let payload = json!({
            "model": "free",
            "max_tokens": 100,
            "messages": [{"role": "user", "content": "what is ripgrep?"}],
            "tools": [{"type": "web_search_20250305", "name": "web_search", "max_uses": 2}],
        });
        let out =
            handle_chat(app.clone(), ClientFormat::Anthropic, payload, ClientContext::default())
                .await;
        let Outcome::Json { status, body, .. } = out else { panic!("expected JSON") };
        assert_eq!(status, 200);
        let content = body["content"].as_array().unwrap();
        let types: Vec<&str> = content.iter().filter_map(|b| b["type"].as_str()).collect();
        assert!(types.contains(&"server_tool_use"), "{body}");
        assert!(types.contains(&"web_search_tool_result"), "{body}");
        assert!(content.iter().any(|b| b["text"] == "done"), "the answer must survive: {body}");
        assert_eq!(body["stop_reason"], "end_turn", "{body}");
        assert_eq!(seen.lock().unwrap().len(), 2);
    }

    /// The §2.1 guard, non-streaming edition: with NO search provider the
    /// tool is stripped and the request stays an ordinary JSON round-trip —
    /// it must not be forced through the stream machinery for a tool pxy
    /// isn't going to serve.
    #[tokio::test]
    async fn non_streaming_web_search_without_provider_stays_plain_json() {
        use axum::routing::post;
        let seen = Arc::new(std::sync::Mutex::new(Vec::<Value>::new()));
        let sink = seen.clone();
        let router = axum::Router::new().route(
            "/m",
            post(move |axum::Json(body): axum::Json<Value>| {
                let sink = sink.clone();
                async move {
                    sink.lock().unwrap().push(body);
                    axum::Json(json!({
                        "id": "x",
                        "choices": [{"index": 0, "message": {"role": "assistant",
                                    "content": "dry"}, "finish_reason": "stop"}],
                        "usage": {"prompt_tokens": 3, "completion_tokens": 1},
                    }))
                }
            }),
        );
        let base = mock_server(router).await;
        let app = test_app(
            &format!(
                r#"
                [server]
                [providers.p]
                base_url = "{base}/m"
                models = ["m"]
                [groups.free]
                models = ["p/m"]
                "#
            ),
            "nonstream_search_off",
        );
        let payload = json!({
            "model": "free",
            "messages": [{"role": "user", "content": "weather?"}],
            "tools": [
                {"type": "web_search_20250305", "name": "web_search"},
                {"name": "Read", "input_schema": {"type": "object", "properties": {}}},
            ],
        });
        let out =
            handle_chat(app.clone(), ClientFormat::Anthropic, payload, ClientContext::default())
                .await;
        let Outcome::Json { status, body, .. } = out else { panic!("expected JSON") };
        assert_eq!(status, 200);
        assert_eq!(body["content"][0]["text"], "dry", "{body}");
        let bodies = seen.lock().unwrap();
        assert_eq!(bodies.len(), 1);
        assert!(bodies[0]["stream"].is_null(), "must not stream upstream: {}", bodies[0]);
        let tools = bodies[0]["tools"].as_array().unwrap();
        assert!(!tools.iter().any(|t| t["function"]["name"] == server_tools::function_name(server_tools::Tool::WebSearch)));
        assert!(tools.iter().any(|t| t["function"]["name"] == "Read"));
    }

    /// Declared server tools pxy can't translate (code_execution
    /// et al) must never be silently dropped. On a multi-candidate walk the
    /// OpenAI-format candidate is skipped WITHOUT an upstream call, and the
    /// Anthropic-format peer receives the tools intact.
    #[tokio::test]
    async fn server_tools_skip_openai_candidates_and_pass_through_to_anthropic() {
        use axum::routing::post;
        let oa_calls = Arc::new(std::sync::Mutex::new(0u32));
        let oa = oa_calls.clone();
        let seen = Arc::new(std::sync::Mutex::new(Vec::<Value>::new()));
        let sink = seen.clone();
        let router = axum::Router::new()
            .route(
                "/oa",
                post(move |_body: axum::Json<Value>| {
                    let oa = oa.clone();
                    async move {
                        *oa.lock().unwrap() += 1;
                        axum::Json(json!({"choices": []}))
                    }
                }),
            )
            .route(
                "/anthropic",
                post(move |axum::Json(body): axum::Json<Value>| {
                    let sink = sink.clone();
                    async move {
                        sink.lock().unwrap().push(body);
                        axum::Json(json!({
                            "id": "m1", "type": "message", "role": "assistant",
                            "model": "big", "content": [{"type": "text", "text": "ran it"}],
                            "stop_reason": "end_turn",
                            "usage": {"input_tokens": 5, "output_tokens": 2},
                        }))
                    }
                }),
            );
        let base = mock_server(router).await;
        let app = test_app(
            &format!(
                r#"
                [server]
                [providers.free]
                base_url = "{base}/oa"
                models = ["small"]
                [providers.paid]
                base_url = "{base}/anthropic"
                format = "anthropic"
                models = ["big"]
                [groups.auto]
                models = ["free/small", "paid/big"]
                "#
            ),
            "server_tool_walk",
        );
        let payload = json!({
            "model": "auto",
            "max_tokens": 100,
            "messages": [{"role": "user", "content": "run this"}],
            "tools": [
                {"type": "code_execution_20250522", "name": "code_execution"},
                {"name": "Read", "input_schema": {"type": "object", "properties": {}}},
            ],
        });
        let out =
            handle_chat(app.clone(), ClientFormat::Anthropic, payload, ClientContext::default())
                .await;
        let Outcome::Json { status, body, provider, .. } = out else { panic!("expected JSON") };
        assert_eq!(status, 200, "{body}");
        assert_eq!(provider.as_deref(), Some("paid/big"));
        assert_eq!(*oa_calls.lock().unwrap(), 0, "the OpenAI candidate must be pre-filtered");
        let bodies = seen.lock().unwrap();
        assert!(
            bodies[0]["tools"]
                .as_array()
                .is_some_and(|ts| ts.iter().any(|t| t["type"] == "code_execution_20250522")),
            "the server tool must reach the Anthropic upstream intact: {:?}",
            bodies[0]["tools"]
        );
    }

    /// §2.3's other half: a single-candidate request bound for an OpenAI
    /// upstream gets an honest 400 (no upstream call), and a walk where NO
    /// candidate can serve the tools ends 400, not a retryable 429.
    #[tokio::test]
    async fn server_tools_unservable_everywhere_return_400() {
        use axum::routing::post;
        let calls = Arc::new(std::sync::Mutex::new(0u32));
        let c = calls.clone();
        let router = axum::Router::new().route(
            "/oa",
            post(move |_body: axum::Json<Value>| {
                let c = c.clone();
                async move {
                    *c.lock().unwrap() += 1;
                    axum::Json(json!({"choices": []}))
                }
            }),
        );
        let base = mock_server(router).await;
        let app = test_app(
            &format!(
                r#"
                [server]
                [providers.a]
                base_url = "{base}/oa"
                models = ["m1"]
                [providers.b]
                base_url = "{base}/oa"
                models = ["m2"]
                [groups.auto]
                models = ["a/m1", "b/m2"]
                "#
            ),
            "server_tool_400",
        );
        let tools = json!([
            {"type": "computer_20250124", "name": "computer",
             "display_width_px": 1024, "display_height_px": 768},
        ]);
        // Single candidate: explicit model.
        let payload = json!({
            "model": "a/m1", "max_tokens": 10,
            "messages": [{"role": "user", "content": "click"}],
            "tools": tools,
        });
        let out =
            handle_chat(app.clone(), ClientFormat::Anthropic, payload, ClientContext::default())
                .await;
        let Outcome::Json { status, body, .. } = out else { panic!("expected JSON") };
        assert_eq!(status, 400, "{body}");
        assert!(body["error"]["message"].as_str().unwrap().contains("computer"), "{body}");

        // Whole walk unservable: still a 400, not a back-off-and-retry 429.
        let payload = json!({
            "model": "auto", "max_tokens": 10,
            "messages": [{"role": "user", "content": "click"}],
            "tools": tools,
        });
        let out =
            handle_chat(app.clone(), ClientFormat::Anthropic, payload, ClientContext::default())
                .await;
        let Outcome::Json { status, body, .. } = out else { panic!("expected JSON") };
        assert_eq!(status, 400, "a deterministic no must not read as rate limiting: {body}");
        assert_eq!(*calls.lock().unwrap(), 0, "no upstream call may be spent on it");
    }

    /// A Chat Completions client that declares a served tool has no translator
    /// to swap it for the reserved function, so the router must. The model
    /// calls pxy_datetime, pxy answers it with no provider at all, and the
    /// client sees the model's answer with the reserved call stripped.
    #[tokio::test]
    async fn datetime_is_served_for_a_chat_completions_client() {
        use axum::response::IntoResponse;
        use axum::routing::post;
        let calls = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let seen = Arc::new(std::sync::Mutex::new(Vec::<Value>::new()));
        let counter = calls.clone();
        let sink = seen.clone();
        let router = axum::Router::new().route(
            "/m",
            post(move |axum::Json(body): axum::Json<Value>| {
                let counter = counter.clone();
                let sink = sink.clone();
                async move {
                    sink.lock().unwrap().push(body);
                    let n = counter.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                    // First call: the model asks for the time. Second: it
                    // answers from the tool result pxy fed back.
                    let sse = if n == 0 {
                        concat!(
                            "data: {\"id\":\"c1\",\"choices\":[{\"index\":0,\"delta\":{\"tool_calls\":[",
                            "{\"index\":0,\"id\":\"call_1\",\"type\":\"function\",",
                            "\"function\":{\"name\":\"pxy_datetime\",\"arguments\":\"{}\"}}]}}]}\n\n",
                            "data: {\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"tool_calls\"}]}\n\n",
                            "data: [DONE]\n\n",
                        )
                    } else {
                        concat!(
                            "data: {\"id\":\"c2\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"It is now.\"}}]}\n\n",
                            "data: {\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n",
                            "data: [DONE]\n\n",
                        )
                    };
                    axum::http::Response::builder()
                        .header("content-type", "text/event-stream")
                        .body(axum::body::Body::from(sse))
                        .unwrap()
                        .into_response()
                }
            }),
        );
        let base = mock_server(router).await;
        // No search or fetch providers: datetime needs neither.
        let app = test_app(
            &format!(
                r#"
                [server]
                [providers.p]
                base_url = "{base}/m"
                models = ["m"]
                [groups.free]
                models = ["p/m"]
                "#
            ),
            "datetime_chat",
        );
        let payload = json!({
            "model": "free",
            "stream": true,
            "messages": [{"role": "user", "content": "what time is it?"}],
            "tools": [{"type": "pxy:datetime"}],
            "max_tool_calls": 2,
        });
        let out =
            handle_chat(app.clone(), ClientFormat::Openai, payload, ClientContext::default()).await;
        let Outcome::Stream { body, .. } = out else { panic!("expected a stream") };
        let bytes = axum::body::to_bytes(body, 1 << 20).await.unwrap();
        let text = String::from_utf8_lossy(&bytes);
        assert!(text.contains("It is now."), "{text}");
        assert!(!text.contains("\"tool_calls\""), "the reserved call must be stripped: {text}");
        assert!(!text.contains("pxy_"), "a Chat Completions client must see no pxy_* name: {text}");

        let bodies = seen.lock().unwrap();
        assert_eq!(bodies.len(), 2, "one call to ask, one to answer");
        assert!(
            bodies[0]["tools"]
                .as_array()
                .is_some_and(|ts| ts.iter().any(|t| t["function"]["name"] == "pxy_datetime")),
            "the declared tool must become the reserved function: {:?}",
            bodies[0]["tools"]
        );
        // The replay carries the tool result back to the model.
        let replay = bodies[1]["messages"].to_string();
        assert!(replay.contains("pxy_datetime"), "{replay}");
        assert!(
            bodies[0].get("max_tool_calls").is_none(),
            "the step budget is pxy's, not an upstream field: {}",
            bodies[0]
        );
    }

    /// `[server_tools.defaults.<tool>]` declares the tool on a turn whose
    /// client sent no tools at all: the upstream is offered the reserved
    /// function, the model's call is served, and the client sees one answer
    /// with no `pxy_*` name, exactly as if it had declared the tool itself.
    #[tokio::test]
    async fn default_server_tools_are_declared_on_a_toolless_chat_turn() {
        use axum::response::IntoResponse;
        use axum::routing::post;
        let calls = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let seen = Arc::new(std::sync::Mutex::new(Vec::<Value>::new()));
        let counter = calls.clone();
        let sink = seen.clone();
        let router = axum::Router::new().route(
            "/m",
            post(move |axum::Json(body): axum::Json<Value>| {
                let counter = counter.clone();
                let sink = sink.clone();
                async move {
                    sink.lock().unwrap().push(body);
                    let n = counter.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                    let sse = if n == 0 {
                        concat!(
                            "data: {\"id\":\"c1\",\"choices\":[{\"index\":0,\"delta\":{\"tool_calls\":[",
                            "{\"index\":0,\"id\":\"call_1\",\"type\":\"function\",",
                            "\"function\":{\"name\":\"pxy_datetime\",\"arguments\":\"{}\"}}]}}]}\n\n",
                            "data: {\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"tool_calls\"}]}\n\n",
                            "data: [DONE]\n\n",
                        )
                    } else {
                        concat!(
                            "data: {\"id\":\"c2\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"It is now.\"}}]}\n\n",
                            "data: {\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n",
                            "data: [DONE]\n\n",
                        )
                    };
                    axum::http::Response::builder()
                        .header("content-type", "text/event-stream")
                        .body(axum::body::Body::from(sse))
                        .unwrap()
                        .into_response()
                }
            }),
        );
        let base = mock_server(router).await;
        let app = test_app(
            &format!(
                r#"
                [server]
                [providers.p]
                base_url = "{base}/m"
                models = ["m"]
                [groups.free]
                models = ["p/m"]
                [server_tools.defaults.datetime]
                timezone = "Asia/Dhaka"
                "#
            ),
            "default_tools_chat",
        );
        let payload = json!({
            "model": "free",
            "stream": true,
            "messages": [{"role": "user", "content": "what time is it?"}],
        });
        let out =
            handle_chat(app.clone(), ClientFormat::Openai, payload, ClientContext::default()).await;
        let Outcome::Stream { body, .. } = out else { panic!("expected a stream") };
        let bytes = axum::body::to_bytes(body, 1 << 20).await.unwrap();
        let text = String::from_utf8_lossy(&bytes);
        assert!(text.contains("It is now."), "{text}");
        assert!(!text.contains("pxy_"), "a Chat Completions client must see no pxy_* name: {text}");

        let bodies = seen.lock().unwrap();
        assert_eq!(bodies.len(), 2, "one call to ask, one to answer");
        let names: Vec<&str> = bodies[0]["tools"]
            .as_array()
            .map(|ts| ts.iter().filter_map(|t| t["function"]["name"].as_str()).collect())
            .unwrap_or_default();
        assert_eq!(names, vec!["pxy_datetime"], "{:?}", bodies[0]["tools"]);
        assert!(bodies[1]["messages"].to_string().contains("pxy_datetime"));
    }

    /// A client's own declaration wins over the config default for the same
    /// tool, whatever spelling it used; a default the client did not declare
    /// is appended with its configured parameters.
    #[test]
    fn default_server_tools_yield_to_the_client_declaration() {
        let app = test_app(
            r#"
            [server]
            [providers.p]
            base_url = "http://127.0.0.1:1/m"
            models = ["m"]
            [server_tools.defaults.datetime]
            timezone = "UTC"
            [server_tools.defaults.search_models]
            "#,
            "default_tools_yield",
        );
        let mut payload = json!({"tools": [
            {"type": "function", "function": {"name": "mine", "parameters": {}}},
            {"type": "openrouter:datetime", "parameters": {"timezone": "Asia/Dhaka"}},
        ]});
        inject_default_server_tools(&mut payload, &app);
        let tools = payload["tools"].as_array().unwrap();
        assert_eq!(tools.len(), 3, "{tools:?}");
        assert_eq!(tools[1]["parameters"]["timezone"], "Asia/Dhaka");
        assert_eq!(tools[2], json!({"type": "pxy:search_models"}));
        assert_eq!(
            server_tools::declared_parameters(&payload, server_tools::Tool::Datetime)["timezone"],
            "Asia/Dhaka"
        );
    }

    /// A default is an offer, not a demand: one the app cannot serve (no
    /// search pool) is left out rather than declared and dropped. The client's
    /// dialect decides nothing — every tool is offered on both.
    #[test]
    fn default_server_tools_respect_the_executor_not_the_dialect() {
        let app = test_app(
            r#"
            [server]
            [providers.p]
            base_url = "http://127.0.0.1:1/m"
            models = ["m"]
            [server_tools.defaults.datetime]
            [server_tools.defaults.web_search]
            [server_tools.defaults.advisor]
            model = "p/m"
            "#,
            "default_tools_dialect",
        );
        // An Anthropic client's payload: the advisor has a result block there
        // and datetime has none, and both are declared all the same — a tool
        // without a block is served silently, not withheld.
        let mut anthropic = json!({"messages": [], "max_tokens": 100});
        inject_default_server_tools(&mut anthropic, &app);
        let types: Vec<&str> = anthropic["tools"]
            .as_array()
            .map(|ts| ts.iter().filter_map(|t| t["type"].as_str()).collect())
            .unwrap_or_default();
        assert_eq!(
            types,
            vec!["pxy:advisor", "pxy:datetime"],
            "web_search has no pool: {anthropic}"
        );
    }

    /// An Anthropic-format upstream never sees a `pxy:*` entry: the loop
    /// cannot run there and no upstream knows the spelling. The `openrouter:*`
    /// spelling, native spellings and function tools stay (the upstream may
    /// be OpenRouter itself); an emptied list takes tool_choice with it.
    #[test]
    fn hosted_server_tools_are_dropped_for_anthropic_upstreams() {
        let mut body = json!({"tools": [
            {"type": "pxy:datetime"},
            {"type": "openrouter:web_search", "parameters": {"max_results": 3}},
            {"type": "web_search_20250305", "name": "web_search"},
            {"name": "mine", "input_schema": {"type": "object"}},
        ], "tool_choice": {"type": "auto"}});
        drop_hosted_server_tools(&mut body);
        let tools = body["tools"].as_array().unwrap();
        assert_eq!(tools.len(), 3, "{tools:?}");
        assert_eq!(tools[0]["type"], "openrouter:web_search");
        assert_eq!(tools[1]["type"], "web_search_20250305");
        assert_eq!(tools[2]["name"], "mine");
        assert!(body["tool_choice"].is_object());

        let mut only_hosted = json!({"tools": [{"type": "pxy:datetime"}], "tool_choice": {"type": "any"}});
        drop_hosted_server_tools(&mut only_hosted);
        assert!(only_hosted.get("tools").is_none(), "{only_hosted}");
        assert!(only_hosted.get("tool_choice").is_none(), "{only_hosted}");
    }

    /// A client may declare far more tools than a turn can afford. The ones
    /// marked `defer_loading` are held out of the upstream body when pxy is
    /// offering the search that can reveal them again, and only then: a
    /// client that defers without declaring the search would otherwise lose
    /// them with no way to ask for them back.
    #[test]
    fn deferred_tools_are_held_back_only_when_the_search_can_reveal_them() {
        let search = server_tools::function_name(server_tools::Tool::ToolSearch);
        let thirty = || -> Vec<Value> {
            (0..30)
                .map(|i| {
                    json!({"type": "function", "function": {
                        "name": format!("tool_{i}"),
                        "description": "A client tool.",
                        "defer_loading": true,
                    }})
                })
                .collect()
        };
        let declared = || {
            let mut tools = thirty();
            tools.insert(0, json!({"type": "function", "function": {"name": &search}}));
            json!({"tools": tools, "tool_choice": {"type": "function", "function": {"name": "tool_9"}}})
        };
        // The turn is already mid-call on three of them.
        let payload = json!({"messages": [{"role": "assistant", "tool_calls": [
            {"function": {"name": "tool_1"}},
            {"function": {"name": "tool_2"}},
            {"function": {"name": "tool_3"}},
        ]}]});
        let offered = |body: &Value| -> Vec<String> {
            body["tools"]
                .as_array()
                .unwrap()
                .iter()
                .map(|t| t["function"]["name"].as_str().unwrap().to_string())
                .collect()
        };

        let mut body = declared();
        let deferred = defer_loading_tools(
            &mut body,
            &payload,
            ClientFormat::Openai,
            WireFormat::Openai,
            true,
        );
        // The search itself, the three the model is already calling, and the
        // one `tool_choice` names. The other twenty-six wait to be found.
        assert_eq!(
            offered(&body),
            vec![search.as_str(), "tool_1", "tool_2", "tool_3", "tool_9"],
            "{body}"
        );
        assert_eq!(deferred.len(), 26);
        // `defer_loading` is pxy's own key: an OpenAI upstream 400s on an
        // unknown key, and a revealed definition is replayed as it stands.
        assert!(
            body["tools"].as_array().unwrap().iter().all(|t| t["function"]["defer_loading"].is_null()),
            "{body}"
        );
        assert!(
            deferred.iter().all(|t| t["function"]["defer_loading"].is_null()),
            "{deferred:?}"
        );
        // The choice survives the deferral, because the tool it names did.
        settle_tool_choice(&mut body);
        assert_eq!(body["tool_choice"]["function"]["name"], "tool_9", "{body}");

        // Nothing offers the search: every tool goes up as an ordinary
        // function, minus the key no upstream knows.
        let mut plain = json!({"tools": thirty()});
        let none = defer_loading_tools(
            &mut plain,
            &json!({}),
            ClientFormat::Openai,
            WireFormat::Openai,
            false,
        );
        assert!(none.is_empty());
        assert_eq!(offered(&plain).len(), 30);
        assert!(plain["tools"].as_array().unwrap().iter().all(|t| t["function"]["defer_loading"].is_null()));

        // An Anthropic upstream runs no loop and has deferral of its own, so
        // its body is passed through exactly as built, key included.
        let mut anthropic = declared();
        let untouched = anthropic.clone();
        let nothing = defer_loading_tools(
            &mut anthropic,
            &payload,
            ClientFormat::Anthropic,
            WireFormat::Anthropic,
            true,
        );
        assert!(nothing.is_empty());
        assert_eq!(anthropic, untouched, "an anthropic body is not pxy's to edit");
    }

    /// `tool_choice` must name a function the body offers: a served tool's
    /// own name is rewritten to the reserved function when it is offered, a
    /// name nothing offers is removed, and no tools at all means no choice.
    #[test]
    fn tool_choice_is_settled_against_the_offered_functions() {
        let ws = server_tools::function_name(server_tools::Tool::WebSearch);
        let mut body = json!({
            "tools": [{"type": "function", "function": {"name": ws}}],
            "tool_choice": {"type": "function", "function": {"name": "web_search"}},
        });
        settle_tool_choice(&mut body);
        assert_eq!(body["tool_choice"]["function"]["name"], ws, "{body}");

        let mut dropped = json!({
            "tools": [{"type": "function", "function": {"name": "mine"}}],
            "tool_choice": {"type": "function", "function": {"name": "web_search"}},
        });
        settle_tool_choice(&mut dropped);
        assert!(dropped.get("tool_choice").is_none(), "{dropped}");

        let mut kept = json!({
            "tools": [{"type": "function", "function": {"name": "mine"}}],
            "tool_choice": {"type": "function", "function": {"name": "mine"}},
        });
        settle_tool_choice(&mut kept);
        assert_eq!(kept["tool_choice"]["function"]["name"], "mine");

        let mut none = json!({"tool_choice": "required"});
        settle_tool_choice(&mut none);
        assert!(none.get("tool_choice").is_none(), "{none}");
        let mut auto = json!({"tools": [{"type": "function", "function": {"name": "mine"}}], "tool_choice": "auto"});
        settle_tool_choice(&mut auto);
        assert_eq!(auto["tool_choice"], "auto");
    }

    /// A model asserted `tool_call = false` is offered no reserved function
    /// for a config default, and a toolless client turn still routes to it.
    #[tokio::test]
    async fn default_server_tools_skip_a_model_that_cannot_tool_call() {
        use axum::response::IntoResponse;
        use axum::routing::post;
        let seen = Arc::new(std::sync::Mutex::new(Vec::<Value>::new()));
        let sink = seen.clone();
        let router = axum::Router::new().route(
            "/m",
            post(move |axum::Json(body): axum::Json<Value>| {
                let sink = sink.clone();
                async move {
                    sink.lock().unwrap().push(body);
                    let sse = concat!(
                        "data: {\"id\":\"c1\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"hi\"}}]}\n\n",
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
        let base = mock_server(router).await;
        let app = test_app(
            &format!(
                r#"
                [server]
                [providers.p]
                base_url = "{base}/m"
                models = [{{ id = "m", tool_call = false }}]
                [groups.free]
                models = ["p/m"]
                [server_tools.defaults.datetime]
                "#
            ),
            "default_tools_no_tool_call",
        );
        let payload = json!({
            "model": "free",
            "stream": true,
            "messages": [{"role": "user", "content": "hello"}],
        });
        let out =
            handle_chat(app.clone(), ClientFormat::Openai, payload, ClientContext::default()).await;
        let Outcome::Stream { body, .. } = out else { panic!("expected a stream, the model must still be routed to") };
        let bytes = axum::body::to_bytes(body, 1 << 20).await.unwrap();
        assert!(String::from_utf8_lossy(&bytes).contains("hi"));
        let bodies = seen.lock().unwrap();
        assert_eq!(bodies.len(), 1);
        assert!(bodies[0].get("tools").is_none(), "{}", bodies[0]);
    }

    /// A candidate asserted `vision = false` is sent `[image N]` in place of
    /// every image part — the upstream would 400 on the part, and a model told
    /// an image is there can ask describe_image for it by index. The
    /// continuation replays the placeholders, because the loop takes its copy
    /// of the body after the swap. A model that asserts nothing is unaffected.
    #[tokio::test]
    async fn vision_false_sends_placeholders_instead_of_image_parts() {
        let turn = |base: String, name: &'static str, vision: &'static str| async move {
            let app = test_app(
                &format!(
                    r#"
                    [server]
                    [providers.p]
                    base_url = "{base}/m"
                    models = [{{ id = "m"{vision} }}]
                    "#
                ),
                name,
            );
            let payload = json!({
                "model": "p/m",
                "stream": true,
                "messages": [{"role": "user", "content": [
                    {"type": "text", "text": "what time was this taken?"},
                    {"type": "image_url", "image_url": {"url": "https://e.test/a.png"}},
                ]}],
                "tools": [{"type": "pxy:datetime"}],
            });
            let out = handle_chat(app, ClientFormat::Openai, payload, ClientContext::default())
                .await;
            let Outcome::Stream { body, .. } = out else { panic!("expected a stream") };
            let _ = axum::body::to_bytes(body, 1 << 20).await.unwrap();
        };

        let first = calls_sse(&[("call_1", "pxy_datetime", "{}")]);
        let first: &'static str = Box::leak(first.into_boxed_str());
        let (base, seen) = scripted_upstream(vec![(200, first), (200, ANSWER_SSE)]).await;
        turn(base.clone(), "vision_false", ", vision = false").await;
        let bodies = seen.lock().unwrap().clone();
        assert_eq!(bodies.len(), 2, "the served call must have been replayed");
        for body in &bodies {
            assert_eq!(
                body["messages"][0]["content"],
                json!([
                    {"type": "text", "text": "what time was this taken?"},
                    {"type": "text", "text": "[image 1]"},
                ]),
                "{body}"
            );
        }

        // The mirror case: nothing asserted, so the image goes up as it came.
        seen.lock().unwrap().clear();
        turn(base, "vision_unknown", "").await;
        let bodies = seen.lock().unwrap();
        assert_eq!(
            bodies[0]["messages"][0]["content"][1],
            json!({"type": "image_url", "image_url": {"url": "https://e.test/a.png"}}),
            "{}",
            bodies[0]
        );
    }

    /// The whole point of the capability, end to end: a `vision = false` model
    /// is sent `[image 1]`, calls describe_image with that number, and the leg
    /// carries the real image to a model that can see it. The description comes
    /// back as the tool result and the blind model answers from it.
    #[tokio::test]
    async fn describe_image_gives_a_blind_model_its_picture_back() {
        use std::sync::Mutex;
        use axum::response::IntoResponse;
        use axum::routing::post;
        let chats: Arc<Mutex<Vec<Value>>> = Arc::new(Mutex::new(Vec::new()));
        let eyes: Arc<Mutex<Vec<Value>>> = Arc::new(Mutex::new(Vec::new()));
        let (chat_sink, eyes_sink) = (chats.clone(), eyes.clone());
        let router = axum::Router::new()
            .route(
                "/m",
                post(move |axum::Json(body): axum::Json<Value>| {
                    let sink = chat_sink.clone();
                    async move {
                        let answered = body["messages"].to_string().contains("a red barn");
                        sink.lock().unwrap().push(body);
                        let sse = if answered {
                            ANSWER_SSE.to_string()
                        } else {
                            calls_sse(&[(
                                "call_1",
                                "pxy_describe_image",
                                "{\"image\":\"1\"}",
                            )])
                        };
                        axum::http::Response::builder()
                            .header("content-type", "text/event-stream")
                            .body(axum::body::Body::from(sse))
                            .unwrap()
                            .into_response()
                    }
                }),
            )
            .route(
                "/eyes",
                post(move |axum::Json(body): axum::Json<Value>| {
                    let sink = eyes_sink.clone();
                    async move {
                        sink.lock().unwrap().push(body);
                        axum::Json(json!({"id": "x", "choices": [{"index": 0,
                            "message": {"role": "assistant", "content": "a red barn"},
                            "finish_reason": "stop"}],
                            "usage": {"prompt_tokens": 9, "completion_tokens": 3}}))
                    }
                }),
            );
        let base = mock_server(router).await;
        let app = test_app(
            &format!(
                r#"
                [server]
                [providers.p]
                base_url = "{base}/m"
                models = [{{ id = "m", vision = false }}]
                [providers.eyes]
                base_url = "{base}/eyes"
                models = ["v"]
                "#
            ),
            "describe_image_e2e",
        );
        let payload = json!({
            "model": "p/m",
            "stream": true,
            "messages": [{"role": "user", "content": [
                {"type": "text", "text": "what is in this?"},
                {"type": "image_url", "image_url": {"url": "https://e.test/barn.png"}},
            ]}],
            "tools": [{"type": "pxy:describe_image", "parameters": {"model": "eyes/v"}}],
        });
        let out =
            handle_chat(app.clone(), ClientFormat::Openai, payload, ClientContext::default()).await;
        let Outcome::Stream { body, .. } = out else { panic!("expected a stream") };
        let bytes = axum::body::to_bytes(body, 1 << 20).await.unwrap();
        let text = String::from_utf8_lossy(&bytes);
        assert!(text.contains("done"), "{text}");
        assert!(!text.contains("pxy_"), "the reserved name must not leak: {text}");

        let chats = chats.lock().unwrap();
        assert_eq!(chats.len(), 2, "the call must have been replayed");
        assert_eq!(
            chats[0]["messages"][0]["content"][1],
            json!({"type": "text", "text": "[image 1]"}),
            "{}",
            chats[0]
        );
        let result = chats[1]["messages"].as_array().unwrap().last().unwrap().clone();
        let served: Value = serde_json::from_str(result["content"].as_str().unwrap()).unwrap();
        assert_eq!(served["status"], "ok", "{served}");
        assert_eq!(served["description"], "a red barn", "{served}");

        // The leg, and only the leg, carried the real image.
        let eyes = eyes.lock().unwrap();
        assert_eq!(eyes.len(), 1);
        assert_eq!(eyes[0]["model"], "v", "{}", eyes[0]);
        assert_eq!(
            eyes[0]["messages"][0]["content"][1]["image_url"]["url"],
            "https://e.test/barn.png",
            "{}",
            eyes[0]
        );
    }

    /// search_models and image_generation render through the same Chat
    /// Completions path as datetime: the reserved call is stripped, no `pxy_*`
    /// name reaches the client, and the client sees one answer. The image call
    /// lands on the media walk (the mock provider), not the chat upstream.
    #[tokio::test]
    async fn catalog_and_image_tools_are_served_without_leaking_a_pxy_name() {
        use axum::response::IntoResponse;
        use axum::routing::post;
        let chats = Arc::new(std::sync::Mutex::new(Vec::<Value>::new()));
        let seen = chats.clone();
        let image_calls = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let images = image_calls.clone();
        let router = axum::Router::new()
            .route(
                "/m",
                post(move |axum::Json(body): axum::Json<Value>| {
                    let seen = seen.clone();
                    async move {
                        let n = {
                            let mut v = seen.lock().unwrap();
                            v.push(body);
                            v.len() - 1
                        };
                        let sse = match n {
                            0 => concat!(
                                "data: {\"id\":\"c1\",\"choices\":[{\"index\":0,\"delta\":{\"tool_calls\":[",
                                "{\"index\":0,\"id\":\"call_1\",\"type\":\"function\",",
                                "\"function\":{\"name\":\"pxy_search_models\",\"arguments\":\"{}\"}}]}}]}\n\n",
                                "data: {\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"tool_calls\"}]}\n\n",
                                "data: [DONE]\n\n",
                            ),
                            1 => concat!(
                                "data: {\"id\":\"c2\",\"choices\":[{\"index\":0,\"delta\":{\"tool_calls\":[",
                                "{\"index\":0,\"id\":\"call_2\",\"type\":\"function\",",
                                "\"function\":{\"name\":\"pxy_image_generation\",\"arguments\":\"{\\\"prompt\\\":\\\"a cat\\\"}\"}}]}}]}\n\n",
                                "data: {\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"tool_calls\"}]}\n\n",
                                "data: [DONE]\n\n",
                            ),
                            _ => concat!(
                                "data: {\"id\":\"c3\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"done\"}}]}\n\n",
                                "data: {\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n",
                                "data: [DONE]\n\n",
                            ),
                        };
                        axum::http::Response::builder()
                            .header("content-type", "text/event-stream")
                            .body(axum::body::Body::from(sse))
                            .unwrap()
                            .into_response()
                    }
                }),
            )
            .route(
                "/img",
                post(move || {
                    let images = images.clone();
                    async move {
                        images.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                        axum::Json(json!({"created": 1, "data": [{"url": "https://x/y.png"}]}))
                    }
                }),
            );
        let base = mock_server(router).await;
        let app = test_app(
            &format!(
                r#"
                [server]
                [providers.p]
                base_url = "{base}/m"
                models = ["m"]
                [providers.p.media]
                images_url = "{base}/img"
                image_models = ["im"]
                [media]
                image = ["p/im"]
                [groups.free]
                models = ["p/m"]
                "#
            ),
            "catalog_image_chat",
        );
        let payload = json!({
            "model": "free",
            "stream": true,
            "messages": [{"role": "user", "content": "go"}],
            "tools": [{"type": "pxy:search_models"}, {"type": "pxy:image_generation"}],
            "max_tool_calls": 4,
        });
        let out =
            handle_chat(app.clone(), ClientFormat::Openai, payload, ClientContext::default()).await;
        let Outcome::Stream { body, .. } = out else { panic!("expected a stream") };
        let bytes = axum::body::to_bytes(body, 1 << 20).await.unwrap();
        let text = String::from_utf8_lossy(&bytes);
        assert!(text.contains("done"), "{text}");
        assert!(!text.contains("pxy_"), "a Chat Completions client must see no pxy_* name: {text}");
        assert!(!text.contains("\"tool_calls\""), "reserved calls must be stripped: {text}");
        assert_eq!(
            image_calls.load(std::sync::atomic::Ordering::SeqCst),
            1,
            "the image walk ran once"
        );
        let bodies = chats.lock().unwrap();
        assert_eq!(bodies.len(), 3, "search, image, then the answer");
        let names: Vec<&str> = bodies[0]["tools"]
            .as_array()
            .unwrap()
            .iter()
            .filter_map(|t| t["function"]["name"].as_str())
            .collect();
        assert!(names.contains(&"pxy_search_models"), "{names:?}");
        assert!(names.contains(&"pxy_image_generation"), "{names:?}");
    }

    /// An image_generation served inside a chat turn is a media cost: the
    /// walk counts against `p#media`, the chat counters count only the chat
    /// calls. Nothing about the image eats the model's request budget.
    #[tokio::test]
    async fn image_generation_records_media_usage_not_chat() {
        use axum::response::IntoResponse;
        use axum::routing::post;
        let n = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let calls = n.clone();
        let router = axum::Router::new()
            .route(
                "/m",
                post(move || {
                    let calls = calls.clone();
                    async move {
                        let i = calls.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                        let sse = if i == 0 {
                            concat!(
                                "data: {\"id\":\"c1\",\"choices\":[{\"index\":0,\"delta\":{\"tool_calls\":[",
                                "{\"index\":0,\"id\":\"call_1\",\"type\":\"function\",",
                                "\"function\":{\"name\":\"pxy_image_generation\",\"arguments\":\"{\\\"prompt\\\":\\\"x\\\"}\"}}]}}]}\n\n",
                                "data: {\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"tool_calls\"}]}\n\n",
                                "data: [DONE]\n\n",
                            )
                        } else {
                            concat!(
                                "data: {\"id\":\"c2\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"done\"}}]}\n\n",
                                "data: {\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n",
                                "data: [DONE]\n\n",
                            )
                        };
                        axum::http::Response::builder()
                            .header("content-type", "text/event-stream")
                            .body(axum::body::Body::from(sse))
                            .unwrap()
                            .into_response()
                    }
                }),
            )
            .route(
                "/img",
                post(|| async {
                    axum::Json(json!({"created": 1, "data": [{"url": "https://x/y.png"}]}))
                }),
            );
        let base = mock_server(router).await;
        let app = test_app(
            &format!(
                r#"
                [server]
                [providers.p]
                base_url = "{base}/m"
                models = ["m"]
                [providers.p.media]
                images_url = "{base}/img"
                image_models = ["im"]
                [media]
                image = ["p/im"]
                [groups.free]
                models = ["p/m"]
                "#
            ),
            "image_generation_usage",
        );
        let payload = json!({
            "model": "free",
            "stream": true,
            "messages": [{"role": "user", "content": "draw"}],
            "tools": [{"type": "pxy:image_generation"}],
        });
        let out =
            handle_chat(app.clone(), ClientFormat::Openai, payload, ClientContext::default()).await;
        let Outcome::Stream { body, .. } = out else { panic!("expected a stream") };
        let _ = axum::body::to_bytes(body, 1 << 20).await.unwrap();

        assert_eq!(
            crate::media::used_requests(&app, "p#media", "day"),
            1,
            "the image walk counts once against the media pool"
        );
        assert_eq!(
            crate::media::used_requests(&app, "p", "day"),
            2,
            "only the two chat calls count against the chat budget"
        );
    }

    /// web_fetch is offered only where pxy can run it: a fetch provider is
    /// configured. Without one the injected function is dropped rather than
    /// handed to the model to call into nothing.
    #[tokio::test]
    async fn web_fetch_is_offered_only_with_a_fetch_provider() {
        use axum::response::IntoResponse;
        use axum::routing::post;
        let seen = Arc::new(std::sync::Mutex::new(Vec::<Value>::new()));
        let sink = seen.clone();
        let router = axum::Router::new().route(
            "/m",
            post(move |axum::Json(body): axum::Json<Value>| {
                let sink = sink.clone();
                async move {
                    sink.lock().unwrap().push(body);
                    axum::http::Response::builder()
                        .header("content-type", "text/event-stream")
                        .body(axum::body::Body::from(
                            "data: {\"choices\":[{\"index\":0,\"delta\":{\"content\":\"ok\"}}]}\n\ndata: [DONE]\n\n",
                        ))
                        .unwrap()
                        .into_response()
                }
            }),
        );
        let base = mock_server(router).await;
        let payload = json!({
            "model": "free",
            "stream": true,
            "messages": [{"role": "user", "content": "read this"}],
            "tools": [{"type": "pxy:web_fetch"}],
            "tool_choice": "required",
        });

        let without = test_app(
            &format!(
                r#"
                [server]
                [providers.p]
                base_url = "{base}/m"
                models = ["m"]
                [groups.free]
                models = ["p/m"]
                "#
            ),
            "web_fetch_no_pool",
        );
        let out = handle_chat(
            without,
            ClientFormat::Openai,
            payload.clone(),
            ClientContext::default(),
        )
        .await;
        if let Outcome::Stream { body, .. } = out {
            let _ = axum::body::to_bytes(body, 1 << 20).await.unwrap();
        }
        {
            let bodies = seen.lock().unwrap();
            assert!(
                !bodies[0]["tools"]
                    .as_array()
                    .is_some_and(|ts| ts.iter().any(|t| t["function"]["name"] == "pxy_web_fetch")),
                "no fetch provider: the function must be dropped"
            );
            assert!(
                bodies[0].get("tool_choice").is_none(),
                "a dropped tool must not leave tool_choice dangling: {}",
                bodies[0]
            );
        }

        let with = test_app(
            &format!(
                r#"
                [server]
                [providers.p]
                base_url = "{base}/m"
                models = ["m"]
                [groups.free]
                models = ["p/m"]
                [[fetch.providers]]
                name = "jina"
                kind = "jina-reader"
                api_key = "k"
                "#
            ),
            "web_fetch_pool",
        );
        let out =
            handle_chat(with, ClientFormat::Openai, payload, ClientContext::default()).await;
        if let Outcome::Stream { body, .. } = out {
            let _ = axum::body::to_bytes(body, 1 << 20).await.unwrap();
        }
        let bodies = seen.lock().unwrap();
        assert!(
            bodies[1]["tools"]
                .as_array()
                .is_some_and(|ts| ts.iter().any(|t| t["function"]["name"] == "pxy_web_fetch")),
            "with a fetch provider the function must be offered: {:?}",
            bodies[1]["tools"]
        );
    }

    /// A disabled tool is unservable whatever dialect declared it: on a lone
    /// OpenAI candidate that is an honest 400 up front, not a silent drop.
    #[tokio::test]
    async fn a_disabled_served_tool_is_refused_on_a_single_candidate() {
        let app = test_app(
            r#"
            [server]
            [server_tools]
            enabled = ["web_search"]
            [providers.p]
            base_url = "https://unused.example/m"
            models = ["m"]
            [groups.free]
            models = ["p/m"]
            "#,
            "disabled_tools",
        );
        for ty in ["pxy:web_fetch", "pxy:datetime"] {
            let payload = json!({
                "model": "free",
                "messages": [{"role": "user", "content": "x"}],
                "tools": [{"type": ty}],
            });
            let out =
                handle_chat(app.clone(), ClientFormat::Openai, payload, ClientContext::default())
                    .await;
            let Outcome::Json { status, body, .. } = out else { panic!("expected JSON") };
            assert_eq!(status, 400, "{ty}: {body}");
            assert!(body["error"]["message"].as_str().unwrap().contains(ty), "{body}");
        }
    }

    /// Anthropic Messages is offered every tool pxy can serve, including the
    /// ones with no documented client result block: datetime and web_fetch go
    /// up beside web_search, the round is served silently — nothing is spliced
    /// into the client's stream — and the usage still reports the call.
    #[tokio::test]
    async fn served_turn_on_anthropic_messages_is_silent_without_a_block() {
        let first = calls_sse(&[("call_1", "pxy_datetime", "{}")]);
        let first: &'static str = Box::leak(first.into_boxed_str());
        let (base, seen) = scripted_upstream(vec![(200, first), (200, ANSWER_SSE)]).await;
        // Both executors are configured, so nothing is unservable for lack of
        // a provider: the dialect is the only thing that ever ruled them out.
        let app = test_app(
            &format!(
                r#"
                [server]
                [providers.p]
                base_url = "{base}/m"
                models = ["m"]
                [groups.free]
                models = ["p/m"]
                [[search.providers]]
                name = "brave"
                kind = "brave"
                api_key = "k"
                [[fetch.providers]]
                name = "jina"
                kind = "jina-reader"
                api_key = "k"
                "#
            ),
            "anthropic_silent_round",
        );
        let payload = json!({
            "model": "free",
            "stream": true,
            "max_tokens": 100,
            "messages": [{"role": "user", "content": "x"}],
            "tools": [
                {"type": "pxy:web_fetch"},
                {"type": "pxy:datetime"},
                {"type": "web_search_20250305", "name": "web_search"},
            ],
        });
        let out =
            handle_chat(app, ClientFormat::Anthropic, payload, ClientContext::default()).await;
        let Outcome::Stream { body, .. } = out else { panic!("expected a stream") };
        let bytes = axum::body::to_bytes(body, 1 << 20).await.unwrap();
        let text = String::from_utf8_lossy(&bytes);

        let bodies = seen.lock().unwrap();
        let names: Vec<&str> = bodies[0]["tools"]
            .as_array()
            .expect("every declared tool must survive")
            .iter()
            .filter_map(|t| t["function"]["name"].as_str())
            .collect();
        assert!(names.contains(&"pxy_web_search"), "{names:?}");
        assert!(names.contains(&"pxy_web_fetch"), "{names:?}");
        assert!(names.contains(&"pxy_datetime"), "{names:?}");
        assert_eq!(bodies.len(), 2, "the served call must have been replayed");

        // datetime has no Anthropic result block, so the client sees the
        // answer and nothing else — never a block naming a tool it cannot
        // interpret, and never pxy's own reserved name.
        assert!(text.contains("done"), "{text}");
        assert!(!text.contains("\"type\":\"server_tool_use\""), "{text}");
        assert!(!text.contains("pxy_"), "{text}");
        // The round still happened, and the usage says so.
        assert!(text.contains("\"datetime_requests\":1"), "{text}");
    }

    /// `[server_tools] enabled` and the registry decide which declared server
    /// tools pxy can serve: an `openrouter:advisor` (no registry entry) is
    /// refused on a single OpenAI candidate rather than silently dropped from
    /// the body, and skipped on a walk so an Anthropic peer can take it. A
    /// spelling the registry maps is servable, so the translator injects it.
    /// A meta-tool may not recurse. `tool_depth` is 0 for a client request and
    /// 1 for a meta-tool's sub-request; at 1 the declaration is stripped, so
    /// the upstream is never offered the advisor and the loop can never serve
    /// it. At depth 0 the same declaration becomes the reserved function.
    #[tokio::test]
    async fn depth_one_sub_request_strips_the_meta_tools() {
        use std::sync::Mutex;
        use axum::response::IntoResponse;
        use axum::routing::post;
        let seen: Arc<Mutex<Vec<Value>>> = Arc::new(Mutex::new(Vec::new()));
        let capture = seen.clone();
        let router = axum::Router::new().route(
            "/c",
            post(move |axum::Json(body): axum::Json<Value>| {
                let capture = capture.clone();
                async move {
                    let wants_stream = body["stream"] == true;
                    capture.lock().unwrap().push(body);
                    // A served tool forces the upstream stream, and a
                    // sub-request with none stays plain JSON.
                    if wants_stream {
                        let sse = concat!(
                            "data: {\"id\":\"c1\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"ok\"}}]}\n\n",
                            "data: {\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n",
                            "data: [DONE]\n\n",
                        );
                        axum::http::Response::builder()
                            .header("content-type", "text/event-stream")
                            .body(axum::body::Body::from(sse))
                            .unwrap()
                            .into_response()
                    } else {
                        axum::Json(json!({"id": "x", "choices": [{"index": 0,
                            "message": {"role": "assistant", "content": "ok"},
                            "finish_reason": "stop"}],
                            "usage": {"prompt_tokens": 1, "completion_tokens": 1}}))
                            .into_response()
                    }
                }
            }),
        );
        let base = mock_server(router).await;
        let app = test_app(
            &format!(
                r#"
                [server]
                [providers.a]
                base_url = "{base}/c"
                models = ["m"]
                "#
            ),
            "tool_depth",
        );
        let payload = json!({
            "model": "a/m",
            "messages": [{"role": "user", "content": "hi"}],
            "tools": [{"type": "pxy:advisor"}],
            "stream": false,
        });
        let advisor = server_tools::function_name(server_tools::Tool::Advisor);
        let offers_advisor = |b: &Value| {
            b["tools"].as_array().is_some_and(|ts| {
                ts.iter().any(|t| t["function"]["name"] == advisor.as_str())
            })
        };

        let _ =
            handle_chat(app.clone(), ClientFormat::Openai, payload.clone(), ClientContext::default())
                .await;
        let _ = handle_chat(
            app.clone(),
            ClientFormat::Openai,
            payload.clone(),
            ClientContext { tool_depth: 1, ..ClientContext::default() },
        )
        .await;

        let bodies = seen.lock().unwrap();
        assert_eq!(bodies.len(), 2, "both requests must reach the upstream");
        assert!(offers_advisor(&bodies[0]), "depth 0 offers the advisor: {}", bodies[0]);
        assert!(!offers_advisor(&bodies[1]), "depth 1 must strip it: {}", bodies[1]);
    }

    /// The advisor consults the model its declaration pinned, and the
    /// declaration's `instructions` are the advisor's system turn. The advisor
    /// sees only the prompt: pxy keeps no conversation, so there is nothing
    /// else to forward.
    #[tokio::test]
    async fn advisor_uses_the_declared_model_and_instructions() {
        use std::sync::Mutex;
        use axum::response::IntoResponse;
        use axum::routing::post;
        let strong: Arc<Mutex<Vec<Value>>> = Arc::new(Mutex::new(Vec::new()));
        let sink = strong.clone();
        let router = axum::Router::new()
            .route(
                "/outer",
                post(move |axum::Json(body): axum::Json<Value>| async move {
                    let sse = if body["messages"].to_string().contains("advice text") {
                        concat!(
                            "data: {\"id\":\"c2\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"final answer\"}}]}\n\n",
                            "data: {\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n",
                            "data: [DONE]\n\n",
                        )
                    } else {
                        concat!(
                            "data: {\"id\":\"c1\",\"choices\":[{\"index\":0,\"delta\":{\"tool_calls\":[",
                            "{\"index\":0,\"id\":\"call_1\",\"type\":\"function\",",
                            "\"function\":{\"name\":\"pxy_advisor\",\"arguments\":\"{\\\"prompt\\\":\\\"how?\\\"}\"}}]}}]}\n\n",
                            "data: {\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"tool_calls\"}]}\n\n",
                            "data: [DONE]\n\n",
                        )
                    };
                    axum::http::Response::builder()
                        .header("content-type", "text/event-stream")
                        .body(axum::body::Body::from(sse))
                        .unwrap()
                        .into_response()
                }),
            )
            .route(
                "/strong",
                post(move |axum::Json(body): axum::Json<Value>| {
                    let sink = sink.clone();
                    async move {
                        sink.lock().unwrap().push(body);
                        axum::Json(json!({"id": "x", "choices": [{"index": 0,
                            "message": {"role": "assistant", "content": "advice text"},
                            "finish_reason": "stop"}],
                            "usage": {"prompt_tokens": 3, "completion_tokens": 2}}))
                    }
                }),
            );
        let base = mock_server(router).await;
        let app = test_app(
            &format!(
                r#"
                [server]
                [providers.outer]
                base_url = "{base}/outer"
                models = ["m"]
                [providers.strong]
                base_url = "{base}/strong"
                models = ["s"]
                "#
            ),
            "advisor_spec",
        );
        let payload = json!({
            "model": "outer/m",
            "stream": true,
            "messages": [{"role": "user", "content": "build a pool"}],
            "tools": [{
                "type": "pxy:advisor",
                "parameters": {"model": "strong/s", "instructions": "be terse"},
            }],
        });
        let ctx = ClientContext { agent: Some("pi".into()), ..ClientContext::default() };
        let out = handle_chat(app.clone(), ClientFormat::Openai, payload, ctx).await;
        let Outcome::Stream { body, .. } = out else { panic!("expected a stream") };
        let bytes = axum::body::to_bytes(body, 1 << 20).await.unwrap();
        let text = String::from_utf8_lossy(&bytes);
        assert!(text.contains("final answer"), "{text}");
        assert!(!text.contains("pxy_"), "no reserved name may reach the client: {text}");

        let bodies = strong.lock().unwrap();
        let sub = bodies.first().expect("the advisor sub-request must reach the strong provider");
        assert_eq!(sub["model"], "s", "the declared model, not the outer one: {sub}");
        assert_eq!(sub["messages"][0]["role"], "system", "{sub}");
        assert_eq!(sub["messages"][0]["content"], "be terse", "{sub}");
        assert_eq!(sub["messages"][1]["content"], "how?", "{sub}");
        // The leg is the client's turn too: it bills to the client's agent,
        // not to "other".
        let rows = app.state.model_usage_rows().unwrap();
        assert!(
            rows.iter().any(|r| r.agent == "pi" && r.model == "s"),
            "the advisor leg must bill to the client's agent: {rows:?}"
        );
    }

    /// A Responses client (codex) declares the advisor with its parameters in
    /// the Responses shape. The translator must not lose that declaration on
    /// the way to the loop: the upstream sees exactly one reserved function,
    /// and the consultation goes to the declared model with the declared
    /// instructions — the same as for a Chat Completions client.
    #[tokio::test]
    async fn responses_declaration_reaches_the_loop() {
        use std::sync::Mutex;
        use axum::response::IntoResponse;
        use axum::routing::post;
        let outer_seen: Arc<Mutex<Vec<Value>>> = Arc::new(Mutex::new(Vec::new()));
        let strong: Arc<Mutex<Vec<Value>>> = Arc::new(Mutex::new(Vec::new()));
        let outer_sink = outer_seen.clone();
        let sink = strong.clone();
        let router = axum::Router::new()
            .route(
                "/outer",
                post(move |axum::Json(body): axum::Json<Value>| {
                    let outer_sink = outer_sink.clone();
                    async move {
                        let replay = body["messages"].to_string().contains("advice text");
                        outer_sink.lock().unwrap().push(body);
                        let sse = if replay {
                            concat!(
                                "data: {\"id\":\"c2\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"final answer\"}}]}\n\n",
                                "data: {\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n",
                                "data: [DONE]\n\n",
                            )
                        } else {
                            concat!(
                                "data: {\"id\":\"c1\",\"choices\":[{\"index\":0,\"delta\":{\"tool_calls\":[",
                                "{\"index\":0,\"id\":\"call_1\",\"type\":\"function\",",
                                "\"function\":{\"name\":\"pxy_advisor\",\"arguments\":\"{\\\"prompt\\\":\\\"how?\\\"}\"}}]}}]}\n\n",
                                "data: {\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"tool_calls\"}]}\n\n",
                                "data: [DONE]\n\n",
                            )
                        };
                        axum::http::Response::builder()
                            .header("content-type", "text/event-stream")
                            .body(axum::body::Body::from(sse))
                            .unwrap()
                            .into_response()
                    }
                }),
            )
            .route(
                "/strong",
                post(move |axum::Json(body): axum::Json<Value>| {
                    let sink = sink.clone();
                    async move {
                        sink.lock().unwrap().push(body);
                        axum::Json(json!({"id": "x", "choices": [{"index": 0,
                            "message": {"role": "assistant", "content": "advice text"},
                            "finish_reason": "stop"}],
                            "usage": {"prompt_tokens": 3, "completion_tokens": 2}}))
                    }
                }),
            );
        let base = mock_server(router).await;
        let app = test_app(
            &format!(
                r#"
                [server]
                [providers.outer]
                base_url = "{base}/outer"
                models = ["m"]
                [providers.strong]
                base_url = "{base}/strong"
                models = ["s"]
                "#
            ),
            "responses_declaration",
        );
        // The Responses request as codex sends it, through the same
        // translation /v1/responses applies before routing.
        let payload = crate::translate::responses::request(&json!({
            "model": "outer/m",
            "stream": true,
            "input": [{"type": "message", "role": "user", "content": "build a pool"}],
            "tools": [
                {"type": "function", "name": "mine", "parameters": {"type": "object", "properties": {}}},
                {"type": "openrouter:advisor", "parameters": {"model": "strong/s", "instructions": "be terse"}},
                {"type": "pxy:advisor"},
            ],
        }));
        let ctx = ClientContext { responses: true, ..ClientContext::default() };
        let out = handle_chat(app.clone(), ClientFormat::Openai, payload, ctx).await;
        let Outcome::Stream { body, .. } = out else { panic!("expected a stream") };
        let bytes = axum::body::to_bytes(body, 1 << 20).await.unwrap();
        let text = String::from_utf8_lossy(&bytes);
        assert!(text.contains("final answer"), "{text}");

        let outer = outer_seen.lock().unwrap();
        let names: Vec<&str> = outer[0]["tools"]
            .as_array()
            .map(|ts| ts.iter().filter_map(|t| t["function"]["name"].as_str()).collect())
            .unwrap_or_default();
        assert_eq!(names, vec!["mine", "pxy_advisor"], "{}", outer[0]["tools"]);

        let bodies = strong.lock().unwrap();
        let sub = bodies.first().expect("the advisor sub-request must reach the strong provider");
        assert_eq!(sub["model"], "s", "the declared model, not the outer one: {sub}");
        assert_eq!(sub["messages"][0]["content"], "be terse", "{sub}");
        assert_eq!(sub["messages"][1]["content"], "how?", "{sub}");
    }

    /// An Anthropic Messages client gets the consultation as the official
    /// block shapes: a `server_tool_use` naming the advisor, then an
    /// `advisor_tool_result` carrying the advice. No `pxy_*` name reaches it.
    #[tokio::test]
    async fn advisor_renders_on_the_anthropic_dialect() {
        use std::sync::Mutex;
        use axum::response::IntoResponse;
        use axum::routing::post;
        let strong: Arc<Mutex<Vec<Value>>> = Arc::new(Mutex::new(Vec::new()));
        let sink = strong.clone();
        let router = axum::Router::new()
            .route(
                "/m",
                post(move |axum::Json(body): axum::Json<Value>| async move {
                    let sse = if body["messages"].to_string().contains("advice text") {
                        concat!(
                            "data: {\"id\":\"c2\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"final answer\"}}]}\n\n",
                            "data: {\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n",
                            "data: [DONE]\n\n",
                        )
                    } else {
                        concat!(
                            "data: {\"id\":\"c1\",\"choices\":[{\"index\":0,\"delta\":{\"tool_calls\":[",
                            "{\"index\":0,\"id\":\"call_1\",\"type\":\"function\",",
                            "\"function\":{\"name\":\"pxy_advisor\",\"arguments\":\"{\\\"prompt\\\":\\\"how?\\\"}\"}}]}}]}\n\n",
                            "data: {\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"tool_calls\"}]}\n\n",
                            "data: [DONE]\n\n",
                        )
                    };
                    axum::http::Response::builder()
                        .header("content-type", "text/event-stream")
                        .body(axum::body::Body::from(sse))
                        .unwrap()
                        .into_response()
                }),
            )
            .route(
                "/strong",
                post(move |axum::Json(body): axum::Json<Value>| {
                    let sink = sink.clone();
                    async move {
                        sink.lock().unwrap().push(body);
                        axum::Json(json!({"id": "x", "choices": [{"index": 0,
                            "message": {"role": "assistant", "content": "advice text"},
                            "finish_reason": "stop"}],
                            "usage": {"prompt_tokens": 3, "completion_tokens": 2}}))
                    }
                }),
            );
        let base = mock_server(router).await;
        let app = test_app(
            &format!(
                r#"
                [server]
                [providers.p]
                base_url = "{base}/m"
                models = ["m"]
                [providers.strong]
                base_url = "{base}/strong"
                models = ["s"]
                "#
            ),
            "advisor_anthropic",
        );
        // The Anthropic native shape: the model rides the entry itself.
        let payload = json!({
            "model": "p/m",
            "stream": true,
            "messages": [{"role": "user", "content": "build a pool"}],
            "tools": [{"type": "advisor_20260301", "name": "advisor", "model": "strong/s"}],
        });
        let out =
            handle_chat(app.clone(), ClientFormat::Anthropic, payload, ClientContext::default())
                .await;
        let Outcome::Stream { body, .. } = out else { panic!("expected a stream") };
        let bytes = axum::body::to_bytes(body, 1 << 20).await.unwrap();
        let text = String::from_utf8_lossy(&bytes);
        assert!(text.contains("final answer"), "{text}");
        assert!(text.contains("\"server_tool_use\""), "{text}");
        assert!(text.contains("\"name\":\"advisor\""), "{text}");
        assert!(text.contains("\"advisor_tool_result\""), "{text}");
        assert!(text.contains("advice text"), "{text}");
        assert!(!text.contains("pxy_"), "no reserved name may reach the client: {text}");

        let bodies = strong.lock().unwrap();
        assert_eq!(bodies[0]["model"], "s", "the entry's model picks the advisor");
    }

    /// The whole point of deferring: the model searches, and what it finds is
    /// in the body it is asked to answer from. The first upstream call carries
    /// the reserved function alone; the second carries the tool the search
    /// revealed, and the client never sees a `pxy_` name.
    #[tokio::test]
    async fn tool_search_reveals_its_match_into_the_continuation() {
        use std::sync::Mutex;
        use axum::response::IntoResponse;
        use axum::routing::post;
        let seen: Arc<Mutex<Vec<Value>>> = Arc::new(Mutex::new(Vec::new()));
        let sink = seen.clone();
        let router = axum::Router::new().route(
            "/m",
            post(move |axum::Json(body): axum::Json<Value>| {
                let sink = sink.clone();
                async move {
                    let searched = body["messages"].to_string().contains("get_weather");
                    sink.lock().unwrap().push(body);
                    let sse = if searched {
                        concat!(
                            "data: {\"id\":\"c2\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"final answer\"}}]}\n\n",
                            "data: {\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n",
                            "data: {\"choices\":[],\"usage\":{\"prompt_tokens\":3,\"completion_tokens\":2}}\n\n",
                            "data: [DONE]\n\n",
                        )
                    } else {
                        concat!(
                            "data: {\"id\":\"c1\",\"choices\":[{\"index\":0,\"delta\":{\"tool_calls\":[",
                            "{\"index\":0,\"id\":\"call_1\",\"type\":\"function\",",
                            "\"function\":{\"name\":\"pxy_tool_search\",\"arguments\":\"{\\\"pattern\\\":\\\"weather\\\"}\"}}]}}]}\n\n",
                            "data: {\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"tool_calls\"}]}\n\n",
                            "data: [DONE]\n\n",
                        )
                    };
                    axum::http::Response::builder()
                        .header("content-type", "text/event-stream")
                        .body(axum::body::Body::from(sse))
                        .unwrap()
                        .into_response()
                }
            }),
        );
        let base = mock_server(router).await;
        let app = test_app(
            &format!(
                r#"
                [server]
                [providers.p]
                base_url = "{base}/m"
                models = ["m"]
                "#
            ),
            "tool_search_reveal",
        );
        let deferred = |name: &str, description: &str| {
            json!({"type": "function", "function": {
                "name": name,
                "description": description,
                "parameters": {"type": "object", "properties": {}},
                "defer_loading": true,
            }})
        };
        let payload = json!({
            "model": "p/m",
            "stream": true,
            "messages": [{"role": "user", "content": "what is it like out?"}],
            "tools": [
                {"type": "pxy:tool_search"},
                deferred("get_weather", "Conditions for a city."),
                deferred("send_email", "Deliver a note."),
            ],
        });
        let out = handle_chat(app.clone(), ClientFormat::Openai, payload, ClientContext::default())
            .await;
        let Outcome::Stream { body, .. } = out else { panic!("expected a stream") };
        let text = String::from_utf8_lossy(
            &axum::body::to_bytes(body, 1 << 20).await.unwrap(),
        )
        .to_string();
        assert!(text.contains("final answer"), "{text}");
        assert!(!text.contains("pxy_"), "no reserved name may reach the client: {text}");
        assert!(text.contains("\"tool_search_requests\":1"), "{text}");

        let bodies = seen.lock().unwrap();
        let offered = |body: &Value| -> Vec<String> {
            body["tools"]
                .as_array()
                .unwrap()
                .iter()
                .map(|t| t["function"]["name"].as_str().unwrap().to_string())
                .collect()
        };
        assert_eq!(offered(&bodies[0]), vec!["pxy_tool_search"], "{}", bodies[0]);
        // The match is revealed; the tool the model did not search for waits.
        assert_eq!(
            offered(&bodies[1]),
            vec!["pxy_tool_search", "get_weather"],
            "{}",
            bodies[1]
        );
    }

    /// An Anthropic Messages client gets the search as the official block
    /// shapes. A pattern the engine rejects is an error block the model can
    /// learn from, not a failed turn, so it searches again and gets the
    /// result block.
    #[tokio::test]
    async fn tool_search_renders_both_result_blocks_on_the_messages_dialect() {
        use std::sync::Mutex;
        use axum::response::IntoResponse;
        use axum::routing::post;
        let seen: Arc<Mutex<Vec<Value>>> = Arc::new(Mutex::new(Vec::new()));
        let sink = seen.clone();
        let router = axum::Router::new().route(
            "/m",
            post(move |axum::Json(body): axum::Json<Value>| {
                let sink = sink.clone();
                async move {
                    let history = body["messages"].to_string();
                    sink.lock().unwrap().push(body);
                    let call = |pattern: &str| {
                        format!(
                            concat!(
                                "data: {{\"id\":\"c1\",\"choices\":[{{\"index\":0,\"delta\":{{\"tool_calls\":[",
                                "{{\"index\":0,\"id\":\"call_1\",\"type\":\"function\",",
                                "\"function\":{{\"name\":\"pxy_tool_search\",\"arguments\":\"{{\\\"pattern\\\":\\\"{pattern}\\\"}}\"}}}}]}}}}]}}\n\n",
                                "data: {{\"choices\":[{{\"index\":0,\"delta\":{{}},\"finish_reason\":\"tool_calls\"}}]}}\n\n",
                                "data: [DONE]\n\n",
                            ),
                            pattern = pattern
                        )
                    };
                    let sse = if history.contains("get_weather") {
                        concat!(
                            "data: {\"id\":\"c3\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"final answer\"}}]}\n\n",
                            "data: {\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n",
                            "data: [DONE]\n\n",
                        )
                        .to_string()
                    } else if history.contains("invalid regular expression") {
                        // Second try, with a pattern that compiles.
                        call("weather")
                    } else {
                        // First try: an unclosed group.
                        call("get_(")
                    };
                    axum::http::Response::builder()
                        .header("content-type", "text/event-stream")
                        .body(axum::body::Body::from(sse))
                        .unwrap()
                        .into_response()
                }
            }),
        );
        let base = mock_server(router).await;
        let app = test_app(
            &format!(
                r#"
                [server]
                [providers.p]
                base_url = "{base}/m"
                models = ["m"]
                "#
            ),
            "tool_search_messages",
        );
        let payload = json!({
            "model": "p/m",
            "stream": true,
            "max_tokens": 64,
            "messages": [{"role": "user", "content": "what is it like out?"}],
            "tools": [
                {"type": "tool_search_tool_regex_20251119", "name": "tool_search"},
                {
                    "name": "get_weather",
                    "description": "Conditions for a city.",
                    "input_schema": {"type": "object", "properties": {}},
                    "defer_loading": true,
                },
            ],
        });
        let out =
            handle_chat(app.clone(), ClientFormat::Anthropic, payload, ClientContext::default())
                .await;
        let Outcome::Stream { body, .. } = out else { panic!("expected a stream") };
        let text = String::from_utf8_lossy(
            &axum::body::to_bytes(body, 1 << 20).await.unwrap(),
        )
        .to_string();
        assert!(text.contains("final answer"), "{text}");
        assert!(text.contains("\"server_tool_use\""), "{text}");
        assert!(text.contains("tool_search_tool_regex"), "{text}");
        assert!(text.contains("tool_search_tool_result_error"), "{text}");
        assert!(text.contains("invalid_tool_input"), "{text}");
        assert!(text.contains("tool_search_tool_search_result"), "{text}");
        assert!(text.contains("\"tool_name\":\"get_weather\""), "{text}");
        assert!(!text.contains("pxy_"), "no reserved name may reach the client: {text}");

        let bodies = seen.lock().unwrap();
        assert_eq!(bodies.len(), 3, "a bad pattern costs a round, not the turn");
        let last = bodies[2]["tools"].as_array().unwrap();
        assert!(
            last.iter().any(|t| t["function"]["name"] == "get_weather"),
            "the revealed tool must be callable: {}",
            bodies[2]
        );
    }

    /// An advisor model that does not resolve must not fail the outer turn:
    /// the consultation comes back as a tool error the outer model reads, and
    /// the turn finishes. The outer upstream is charged for both its calls.
    #[tokio::test]
    async fn unavailable_advisor_model_degrades_to_a_tool_error() {
        use std::sync::Mutex;
        use axum::response::IntoResponse;
        use axum::routing::post;
        let seen: Arc<Mutex<Vec<Value>>> = Arc::new(Mutex::new(Vec::new()));
        let sink = seen.clone();
        let router = axum::Router::new().route(
            "/m",
            post(move |axum::Json(body): axum::Json<Value>| {
                let sink = sink.clone();
                async move {
                    let tool_error = body["messages"].to_string().contains("Advisor call failed");
                    sink.lock().unwrap().push(body);
                    let sse = if tool_error {
                        concat!(
                            "data: {\"id\":\"c2\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"final answer\"}}]}\n\n",
                            "data: {\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n",
                            "data: [DONE]\n\n",
                        )
                    } else {
                        concat!(
                            "data: {\"id\":\"c1\",\"choices\":[{\"index\":0,\"delta\":{\"tool_calls\":[",
                            "{\"index\":0,\"id\":\"call_1\",\"type\":\"function\",",
                            "\"function\":{\"name\":\"pxy_advisor\",\"arguments\":\"{\\\"prompt\\\":\\\"how?\\\"}\"}}]}}]}\n\n",
                            "data: {\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"tool_calls\"}]}\n\n",
                            "data: [DONE]\n\n",
                        )
                    };
                    axum::http::Response::builder()
                        .header("content-type", "text/event-stream")
                        .body(axum::body::Body::from(sse))
                        .unwrap()
                        .into_response()
                }
            }),
        );
        let base = mock_server(router).await;
        let app = test_app(
            &format!(
                r#"
                [server]
                [providers.p]
                base_url = "{base}/m"
                models = ["m"]
                "#
            ),
            "advisor_unavailable",
        );
        let payload = json!({
            "model": "p/m",
            "stream": true,
            "messages": [{"role": "user", "content": "build a pool"}],
            "tools": [{
                "type": "pxy:advisor",
                "parameters": {"model": "missing/model"},
            }],
        });
        let out =
            handle_chat(app.clone(), ClientFormat::Openai, payload, ClientContext::default()).await;
        let Outcome::Stream { body, .. } = out else { panic!("expected a stream") };
        let bytes = axum::body::to_bytes(body, 1 << 20).await.unwrap();
        let text = String::from_utf8_lossy(&bytes);
        assert!(text.contains("final answer"), "the turn must continue: {text}");
        assert!(!text.contains("pxy_"), "{text}");

        let bodies = seen.lock().unwrap();
        assert_eq!(bodies.len(), 2, "ask, then answer");
        assert!(
            bodies[1]["messages"].to_string().contains("Advisor call failed"),
            "the model must read the failure: {}",
            bodies[1]["messages"]
        );
        // Cost: the outer upstream is charged for both its calls.
        assert_eq!(app.state.usage_total("p").unwrap().requests, 2);
    }

    /// A fusion panel whose members do not resolve must not fail the outer
    /// turn: the all-failed panel comes back as a tool error the model reads,
    /// and the turn finishes.
    #[tokio::test]
    async fn unavailable_fusion_panel_degrades_to_a_tool_error() {
        use std::sync::Mutex;
        use axum::response::IntoResponse;
        use axum::routing::post;
        let seen: Arc<Mutex<Vec<Value>>> = Arc::new(Mutex::new(Vec::new()));
        let sink = seen.clone();
        let router = axum::Router::new().route(
            "/m",
            post(move |axum::Json(body): axum::Json<Value>| {
                let sink = sink.clone();
                async move {
                    // The follow-up turn carries the served-tool result as a
                    // `role: "tool"` message; the first ask has none. Keying on
                    // the role, not the failure wording, keeps the mock honest
                    // if that wording changes.
                    let tool_error = body["messages"]
                        .as_array()
                        .is_some_and(|ms| ms.iter().any(|m| m["role"] == "tool"));
                    sink.lock().unwrap().push(body);
                    let sse = if tool_error {
                        concat!(
                            "data: {\"id\":\"c2\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"final answer\"}}]}\n\n",
                            "data: {\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n",
                            "data: [DONE]\n\n",
                        )
                    } else {
                        concat!(
                            "data: {\"id\":\"c1\",\"choices\":[{\"index\":0,\"delta\":{\"tool_calls\":[",
                            "{\"index\":0,\"id\":\"call_1\",\"type\":\"function\",",
                            "\"function\":{\"name\":\"pxy_fusion\",\"arguments\":\"{\\\"prompt\\\":\\\"why?\\\"}\"}}]}}]}\n\n",
                            "data: {\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"tool_calls\"}]}\n\n",
                            "data: [DONE]\n\n",
                        )
                    };
                    axum::http::Response::builder()
                        .header("content-type", "text/event-stream")
                        .body(axum::body::Body::from(sse))
                        .unwrap()
                        .into_response()
                }
            }),
        );
        let base = mock_server(router).await;
        let app = test_app(
            &format!(
                r#"
                [server]
                [providers.p]
                base_url = "{base}/m"
                models = ["m"]
                [server_tools]
                fusion_panel = ["missing/one", "missing/two"]
                "#
            ),
            "fusion_unavailable",
        );
        let payload = json!({
            "model": "p/m",
            "stream": true,
            "messages": [{"role": "user", "content": "why is the sky blue?"}],
            "tools": [{"type": "pxy:fusion"}],
        });
        let out =
            handle_chat(app.clone(), ClientFormat::Openai, payload, ClientContext::default()).await;
        let Outcome::Stream { body, .. } = out else { panic!("expected a stream") };
        let bytes = axum::body::to_bytes(body, 1 << 20).await.unwrap();
        let text = String::from_utf8_lossy(&bytes);
        assert!(text.contains("final answer"), "the turn must continue: {text}");
        assert!(!text.contains("pxy_"), "no reserved name may reach the client: {text}");

        let bodies = seen.lock().unwrap();
        assert_eq!(bodies.len(), 2, "ask, then answer");
        assert!(
            bodies[1]["messages"].to_string().contains("all panel models failed"),
            "the model must read the failure: {}",
            bodies[1]["messages"]
        );
        // Neither the panel nor the analyst reached a provider, so only the
        // outer upstream's two calls are charged.
        assert_eq!(app.state.usage_total("p").unwrap().requests, 2);
    }

    /// A meta-tool may not name a meta-tool as its model — a self-reference or
    /// a two-tool loop. The name does not resolve in the catalog, so the
    /// consultation degrades to a tool error and no provider is called for it.
    #[tokio::test]
    async fn advisor_may_not_name_a_meta_tool_as_its_model() {
        use std::sync::Mutex;
        use axum::response::IntoResponse;
        use axum::routing::post;
        let seen: Arc<Mutex<Vec<Value>>> = Arc::new(Mutex::new(Vec::new()));
        let sink = seen.clone();
        let router = axum::Router::new().route(
            "/m",
            post(move |axum::Json(body): axum::Json<Value>| {
                let sink = sink.clone();
                async move {
                    let tool_error = body["messages"].to_string().contains("Advisor call failed");
                    sink.lock().unwrap().push(body);
                    let sse = if tool_error {
                        "data: {\"choices\":[{\"index\":0,\"delta\":{\"content\":\"final answer\"}}]}\n\ndata: [DONE]\n\n"
                    } else {
                        concat!(
                            "data: {\"choices\":[{\"index\":0,\"delta\":{\"tool_calls\":[",
                            "{\"index\":0,\"id\":\"call_1\",\"type\":\"function\",",
                            "\"function\":{\"name\":\"pxy_advisor\",\"arguments\":\"{\\\"prompt\\\":\\\"how?\\\"}\"}}]}}]}\n\n",
                            "data: [DONE]\n\n",
                        )
                    };
                    axum::http::Response::builder()
                        .header("content-type", "text/event-stream")
                        .body(axum::body::Body::from(sse))
                        .unwrap()
                        .into_response()
                }
            }),
        );
        let base = mock_server(router).await;
        let app = test_app(
            &format!(
                r#"
                [server]
                [providers.p]
                base_url = "{base}/m"
                models = ["m"]
                "#
            ),
            "advisor_self_ref",
        );
        let payload = json!({
            "model": "p/m",
            "stream": true,
            "messages": [{"role": "user", "content": "build a pool"}],
            "tools": [{
                "type": "pxy:advisor",
                "parameters": {"model": "openrouter:subagent"},
            }],
        });
        let out =
            handle_chat(app.clone(), ClientFormat::Openai, payload, ClientContext::default()).await;
        let Outcome::Stream { body, .. } = out else { panic!("expected a stream") };
        let bytes = axum::body::to_bytes(body, 1 << 20).await.unwrap();
        let text = String::from_utf8_lossy(&bytes);
        assert!(text.contains("final answer"), "the turn must continue: {text}");

        let bodies = seen.lock().unwrap();
        assert_eq!(bodies.len(), 2, "only the outer upstream is called");
        assert!(
            bodies[1]["messages"].to_string().contains("Advisor call failed"),
            "the meta-model must be refused as a tool error: {}",
            bodies[1]["messages"]
        );
    }

    #[tokio::test]
    async fn server_tool_servability_follows_the_registry() {
        use axum::routing::post;
        let oa_calls = Arc::new(std::sync::Mutex::new(0u32));
        let oa = oa_calls.clone();
        let seen = Arc::new(std::sync::Mutex::new(Vec::<Value>::new()));
        let sink = seen.clone();
        let router = axum::Router::new()
            .route(
                "/oa",
                post(move |_body: axum::Json<Value>| {
                    let oa = oa.clone();
                    async move {
                        *oa.lock().unwrap() += 1;
                        axum::Json(json!({
                            "choices": [{"index": 0,
                                "message": {"role": "assistant", "content": "ok"},
                                "finish_reason": "stop"}],
                            "usage": {"prompt_tokens": 1, "completion_tokens": 1},
                        }))
                    }
                }),
            )
            .route(
                "/anthropic",
                post(move |axum::Json(body): axum::Json<Value>| {
                    let sink = sink.clone();
                    async move {
                        sink.lock().unwrap().push(body);
                        axum::Json(json!({
                            "id": "m1", "type": "message", "role": "assistant",
                            "model": "big", "content": [{"type": "text", "text": "ran it"}],
                            "stop_reason": "end_turn",
                            "usage": {"input_tokens": 5, "output_tokens": 2},
                        }))
                    }
                }),
            );
        let base = mock_server(router).await;
        let app = test_app(
            &format!(
                r#"
                [server]
                [providers.free]
                base_url = "{base}/oa"
                models = ["small"]
                [providers.paid]
                base_url = "{base}/anthropic"
                format = "anthropic"
                models = ["big"]
                [groups.auto]
                models = ["free/small", "paid/big"]
                "#
            ),
            "server_tool_registry",
        );

        // Alone on an OpenAI candidate there is no peer to hand a spelling pxy
        // cannot inject to: 400, with no upstream call spent.
        for ty in ["openrouter:code_execution"] {
            let payload = json!({
                "model": "free/small", "max_tokens": 100,
                "messages": [{"role": "user", "content": "x"}],
                "tools": [{"type": ty}],
            });
            let out =
                handle_chat(app.clone(), ClientFormat::Anthropic, payload, ClientContext::default())
                    .await;
            let Outcome::Json { status, body, .. } = out else { panic!("expected JSON") };
            assert_eq!(status, 400, "{ty}: {body}");
            assert!(body["error"]["message"].as_str().unwrap().contains(ty), "{body}");
        }
        assert_eq!(*oa_calls.lock().unwrap(), 0, "a refused request must spend no call");

        // On a walk the OpenAI candidate is pre-filtered and the Anthropic peer
        // receives the tool intact.
        let payload = json!({
            "model": "auto", "max_tokens": 100,
            "messages": [{"role": "user", "content": "fetch"}],
            "tools": [{"type": "openrouter:code_execution"}],
        });
        let out =
            handle_chat(app.clone(), ClientFormat::Anthropic, payload, ClientContext::default())
                .await;
        let Outcome::Json { status, provider, .. } = out else { panic!("expected JSON") };
        assert_eq!(status, 200);
        assert_eq!(provider.as_deref(), Some("paid/big"));
        assert_eq!(*oa_calls.lock().unwrap(), 0, "the OpenAI candidate must be skipped");
        let bodies = seen.lock().unwrap();
        assert!(
            bodies[0]["tools"]
                .as_array()
                .is_some_and(|ts| ts.iter().any(|t| t["type"] == "openrouter:code_execution")),
            "the unservable tool must reach the Anthropic peer intact: {:?}",
            bodies[0]["tools"]
        );
    }

    /// When every candidate is cooling, the terminal 429 must
    /// carry machine-readable recovery info — Retry-After plus reset fields —
    /// not just prose. A harness can act on a number, not on a sentence.
    /// Non-retryable cooldowns count too: a drained daily tier expires at
    /// reset, and that is exactly what the client should be told to wait for.
    #[tokio::test]
    async fn exhausted_walk_answers_a_structured_cooldown_429() {
        let app = test_app(
            r#"
            [server]
            [providers.a]
            base_url = "http://127.0.0.1:1/a"
            models = ["m1"]
            [providers.b]
            base_url = "http://127.0.0.1:1/b"
            models = ["m2"]
            [groups.free]
            models = ["a/m1", "b/m2"]
            "#,
            "structured_429",
        );
        // A drained daily tier (non-retryable) and a long model cooldown:
        // neither is recoverable inside the request, both expire eventually.
        app.state.set_cooldown("a", None, Some(Duration::from_secs(300)), false, "daily quota");
        app.state.set_cooldown("b", Some("m2"), Some(Duration::from_secs(600)), true, "429");
        let payload = json!({
            "model": "free",
            "messages": [{"role": "user", "content": "hi"}],
        });
        let out =
            handle_chat(app.clone(), ClientFormat::Anthropic, payload, ClientContext::default())
                .await;
        let Outcome::Json { status, body, headers, .. } = out else { panic!("expected JSON") };
        assert_eq!(status, 429, "{body}");
        assert_eq!(body["error"]["code"], "model_cooldown", "{body}");
        let secs = body["error"]["reset_seconds"].as_u64().expect("reset_seconds");
        assert!((290..=300).contains(&secs), "soonest recovery is a's 300s, got {secs}");
        assert!(body["error"]["reset_time"].as_str().is_some_and(|t| t.contains('T')), "{body}");
        let ra = headers.iter().find(|(k, _)| k == "retry-after").expect("retry-after header");
        assert_eq!(ra.1, secs.to_string());
    }

    /// A win under a route pin must not become session affinity: once the
    /// pin is cleared the conversation follows the chain again at once,
    /// not an hour later.
    #[tokio::test]
    async fn a_pinned_win_does_not_bind_the_session() {
        use axum::response::IntoResponse;
        use axum::routing::post;
        let router = axum::Router::new().route(
            "/{p}",
            post(|axum::extract::Path(p): axum::extract::Path<String>| async move {
                axum::Json(json!({"choices": [{"index": 0, "finish_reason": "stop",
                    "message": {"role": "assistant", "content": p}}],
                    "usage": {"prompt_tokens": 1, "completion_tokens": 1}}))
                .into_response()
            }),
        );
        let base = mock_server(router).await;
        let app = test_app(
            &format!(
                r#"
                [server]
                [providers.a]
                base_url = "{base}/a"
                models = ["m1"]
                [providers.b]
                base_url = "{base}/b"
                models = ["m2"]
                [groups.free]
                models = ["a/m1", "b/m2"]
                "#
            ),
            "pinned_no_bind",
        );
        let ask = |app: SharedApp| async move {
            let out = handle_chat(
                app,
                ClientFormat::Openai,
                json!({"model": "free", "user": "s1", "messages": [{"role": "user", "content": "hi"}]}),
                ClientContext::default(),
            )
            .await;
            let Outcome::Json { provider, .. } = out else { panic!("expected json") };
            provider.unwrap()
        };
        app.state.kv_set(&route_pin_key("free"), "b/m2").unwrap();
        assert_eq!(ask(app.clone()).await, "b/m2", "the pin leads");
        assert!(app.state.session_get("user:s1").is_none(), "a pinned win is not bound");
        app.state.kv_delete(&route_pin_key("free")).unwrap();
        assert_eq!(ask(app.clone()).await, "a/m1", "cleared pin: chain order at once");
        assert_eq!(app.state.session_get("user:s1").as_deref(), Some("a/m1"), "a chain win binds");
    }

    /// A 429 that names a spent allowance is the ACCOUNT's quota (opencode
    /// Go's weekly window, a daily free tier): the whole account cools, for
    /// the stated Retry-After even when it is days, and no sibling model pays
    /// its own 429 to find out. The walk moves to the next account.
    #[tokio::test]
    async fn quota_window_429_cools_the_whole_account_for_the_stated_reset() {
        use axum::response::IntoResponse;
        use axum::routing::post;
        let hits = Arc::new(std::sync::Mutex::new(Vec::<(String, String)>::new()));
        let sink = hits.clone();
        let router = axum::Router::new().route(
            "/m",
            post(move |headers: axum::http::HeaderMap, axum::Json(body): axum::Json<Value>| {
                let sink = sink.clone();
                async move {
                    let key = headers
                        .get("authorization")
                        .and_then(|v| v.to_str().ok())
                        .unwrap_or("")
                        .to_string();
                    let model = body["model"].as_str().unwrap_or("").to_string();
                    sink.lock().unwrap().push((key.clone(), model));
                    if key.ends_with("key-gh") {
                        return axum::http::Response::builder()
                            .status(429)
                            .header("retry-after", "172800")
                            .header("content-type", "application/json")
                            .body(axum::body::Body::from(
                                r#"{"type":"error","error":{"type":"GoUsageLimitError","message":"Weekly usage limit reached. Resets in 2 days. To continue using this model now, enable usage from your available balance"},"metadata":{"limitName":"weekly"}}"#,
                            ))
                            .unwrap()
                            .into_response();
                    }
                    axum::Json(json!({"choices": [{"index": 0, "finish_reason": "stop",
                        "message": {"role": "assistant", "content": "ok"}}],
                        "usage": {"prompt_tokens": 1, "completion_tokens": 1}}))
                    .into_response()
                }
            }),
        );
        let base = mock_server(router).await;
        let app = test_app(
            &format!(
                r#"
                [server]
                [providers.go]
                base_url = "{base}/m"
                models = ["m1", "m2"]
                [[providers.go.accounts]]
                name = "gh"
                api_key = "key-gh"
                [[providers.go.accounts]]
                name = "g"
                api_key = "key-g"
                [groups.one]
                models = ["go/m1"]
                [groups.two]
                models = ["go/m2"]
                "#
            ),
            "quota_account_wide",
        );

        // First request on m1: the gh account reports its weekly window
        // spent; the walk lands on the g account and the client is served.
        let out = handle_chat(
            app.clone(),
            ClientFormat::Openai,
            json!({"model": "one", "messages": [{"role": "user", "content": "hi"}]}),
            ClientContext::default(),
        )
        .await;
        let Outcome::Json { status, provider, .. } = out else { panic!("expected json") };
        assert_eq!(status, 200);
        assert_eq!(provider.as_deref(), Some("go/m1"));

        // The cooldown is on the ACCOUNT, for the stated two days, and not
        // something the retry loop should wait on.
        let cd = app.state.cooldown("go#gh", "m2").expect("account-wide cooldown covers m2");
        assert!(!cd.retryable, "{cd:?}");
        let left = cd.until.saturating_duration_since(std::time::Instant::now());
        assert!(left > Duration::from_secs(172_000), "stated reset obeyed: {left:?}");
        assert!(cd.reason.contains("weekly"), "{cd:?}");
        assert!(app.state.cooldown("go#g", "m1").is_none(), "the other account is untouched");

        // A request on the sibling model skips the gh account outright.
        let out = handle_chat(
            app.clone(),
            ClientFormat::Openai,
            json!({"model": "two", "messages": [{"role": "user", "content": "hi"}]}),
            ClientContext::default(),
        )
        .await;
        let Outcome::Json { status, .. } = out else { panic!("expected json") };
        assert_eq!(status, 200);
        let hits = hits.lock().unwrap().clone();
        let gh: Vec<&String> = hits.iter().filter(|(k, _)| k.ends_with("key-gh")).map(|(_, m)| m).collect();
        assert_eq!(gh, vec!["m1"], "gh was probed once, never for m2: {hits:?}");
    }

    /// An OpenAI-dialect client (codex/opencode/fx) routed to an
    /// Anthropic-format provider with `inject_cache_control` gets the standard
    /// breakpoints — its own protocol has no way to set them. Off by default,
    /// and a client-set marker (Anthropic dialect) always wins.
    #[tokio::test]
    async fn cache_control_is_injected_for_openai_clients_when_enabled() {
        use axum::routing::post;
        let seen = Arc::new(std::sync::Mutex::new(Vec::<Value>::new()));
        let sink = seen.clone();
        let router = axum::Router::new().route(
            "/anthropic",
            post(move |axum::Json(body): axum::Json<Value>| {
                let sink = sink.clone();
                async move {
                    sink.lock().unwrap().push(body);
                    axum::Json(json!({
                        "id": "m1", "type": "message", "role": "assistant",
                        "model": "big", "content": [{"type": "text", "text": "ok"}],
                        "stop_reason": "end_turn",
                        "usage": {"input_tokens": 5, "output_tokens": 2},
                    }))
                }
            }),
        );
        let base = mock_server(router).await;
        let mk_app = |inject: bool, name: &str| {
            test_app(
                &format!(
                    r#"
                    [server]
                    [providers.paid]
                    base_url = "{base}/anthropic"
                    format = "anthropic"
                    inject_cache_control = {inject}
                    models = ["big"]
                    "#
                ),
                name,
            )
        };
        let payload = json!({
            "model": "paid/big",
            "messages": [
                {"role": "system", "content": "be brief"},
                {"role": "user", "content": "turn 1"},
                {"role": "user", "content": "turn 2"},
            ],
        });

        let out = handle_chat(
            mk_app(true, "cache_inject_on"),
            ClientFormat::Openai,
            payload.clone(),
            ClientContext::default(),
        )
        .await;
        assert!(matches!(out, Outcome::Json { status: 200, .. }));
        {
            let bodies = seen.lock().unwrap();
            let sent = &bodies[0];
            assert!(
                sent["system"][0]["cache_control"].is_object(),
                "system breakpoint expected: {sent}"
            );
            let msgs = sent["messages"].as_array().unwrap();
            let marked = msgs
                .iter()
                .filter_map(|m| m["content"].as_array())
                .flatten()
                .filter(|b| b["cache_control"].is_object())
                .count();
            assert_eq!(marked, 2, "last two messages marked: {sent}");
        }

        // Default off: the wire stays marker-free.
        let out = handle_chat(
            mk_app(false, "cache_inject_off"),
            ClientFormat::Openai,
            payload,
            ClientContext::default(),
        )
        .await;
        assert!(matches!(out, Outcome::Json { status: 200, .. }));
        let bodies = seen.lock().unwrap();
        let sent = bodies.last().unwrap();
        assert!(sent["system"].as_str().is_some() || !sent["system"][0]["cache_control"].is_object());
        assert!(
            sent["messages"]
                .as_array()
                .unwrap()
                .iter()
                .filter_map(|m| m["content"].as_array())
                .flatten()
                .all(|b| !b["cache_control"].is_object()),
            "no markers without the flag: {sent}"
        );
    }

    #[tokio::test]
    async fn multi_account_walks_accounts_fill_first() {
        use axum::http::HeaderMap;
        use axum::response::IntoResponse;
        use axum::routing::post;
        let ok = axum::Json(json!({
            "id": "x",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"},
                        "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1},
        }));
        let router = axum::Router::new().route(
            "/m",
            post(|headers: HeaderMap| async move {
                let auth = headers.get("authorization").and_then(|v| v.to_str().ok()).unwrap_or("");
                if auth == "Bearer key-gh" {
                    (axum::http::StatusCode::UNAUTHORIZED, r#"{"error":"bad key"}"#)
                        .into_response()
                } else {
                    ok.into_response()
                }
            }),
        );
        let base = mock_server(router).await;
        let app = test_app(
            &format!(
                r#"
                [server]
                [providers.sub]
                base_url = "{base}/m"
                models = ["m"]
                [[providers.sub.accounts]]
                name = "gh"
                api_key = "key-gh"
                [[providers.sub.accounts]]
                name = "g"
                api_key = "key-g"
                [groups.free]
                models = ["sub/m"]
                "#
            ),
            "multi_account",
        );

        // Expansion: the group resolves to BOTH accounts, in config order.
        let ids: Vec<String> = resolve_candidates(&app.catalog, &app.cfg, &app.state, "free", None)
            .iter()
            .map(|c| format!("{}|{}", c.state_provider(), c.full_id()))
            .collect();
        assert_eq!(ids, ["sub#gh|sub/m", "sub#g|sub/m"]);

        let payload = json!({"model": "free", "messages": [{"role": "user", "content": "hi"}]});
        let out = handle_chat(app.clone(), ClientFormat::Openai, payload, ClientContext::default())
            .await;
        match out {
            Outcome::Json { body, provider, .. } => {
                assert_eq!(body["choices"][0]["message"]["content"], "ok");
                // Bare provider name to the client — the panels never see #.
                assert_eq!(provider.as_deref(), Some("sub/m"));
            }
            Outcome::Stream { .. } => panic!("expected json"),
        }
        // The 401 cooled the GH ACCOUNT (account-wide), not the provider's
        // other account and not any other key of the bare provider name.
        assert!(app.state.cooldown("sub#gh", "m").is_some());
        assert!(app.state.cooldown("sub#g", "m").is_none());
        // Usage landed on the SERVING account only.
        assert_eq!(app.state.usage_total("sub#g").unwrap_or_default().requests, 1);
        assert_eq!(app.state.usage_total("sub#gh").unwrap_or_default().requests, 0);

        // Second request: gh is still auth-cooled, g serves again — and a
        // healthy gh would have been preferred (fill-first).
        let payload = json!({"model": "free", "messages": [{"role": "user", "content": "hi"}]});
        let out = handle_chat(app, ClientFormat::Openai, payload, ClientContext::default()).await;
        assert!(matches!(out, Outcome::Json { .. }));
    }

    #[tokio::test]
    async fn context_window_400_fails_over_and_skips_smaller_peers() {
        use std::sync::atomic::{AtomicUsize, Ordering};
        use axum::response::IntoResponse;
        use axum::routing::post;
        let small_calls = Arc::new(AtomicUsize::new(0));
        let counter = small_calls.clone();
        let router = axum::Router::new()
            .route("/small", post(move || {
                let counter = counter.clone();
                async move {
                    counter.fetch_add(1, Ordering::SeqCst);
                    (
                        axum::http::StatusCode::BAD_REQUEST,
                        r#"{"error":{"message":"This model's maximum context length is 8000 tokens"}}"#,
                    )
                        .into_response()
                }
            }))
            .route("/big", post(|| async {
                axum::Json(json!({
                    "id": "x",
                    "choices": [{"index": 0,
                        "message": {"role": "assistant", "content": "fits"},
                        "finish_reason": "stop"}],
                    "usage": {"prompt_tokens": 1, "completion_tokens": 1},
                }))
                .into_response()
            }));
        let base = mock_server(router).await;
        // `tiny` shares small's window: the peer-skip must spare it the call.
        let app = test_app(
            &format!(
                r#"
                [server]
                [providers.small]
                base_url = "{base}/small"
                models = [{{ id = "m", context_length = 8000 }}]
                [providers.tiny]
                base_url = "{base}/small"
                models = [{{ id = "m", context_length = 4000 }}]
                [providers.big]
                base_url = "{base}/big"
                models = [{{ id = "m", context_length = 1000000 }}]
                [groups.free]
                models = ["small/m", "tiny/m", "big/m"]
                "#
            ),
            "ctx_failover",
        );

        let payload = json!({"model": "free", "messages": [{"role": "user", "content": "hi"}]});
        let out = handle_chat(app.clone(), ClientFormat::Openai, payload, ClientContext::default())
            .await;
        match out {
            Outcome::Json { status, body, provider, .. } => {
                assert_eq!(status, 200, "must fail over to the larger window: {body}");
                assert_eq!(provider.as_deref(), Some("big/m"));
            }
            Outcome::Stream { .. } => panic!("expected json"),
        }
        // No cooldown: a smaller request to `small` would work fine.
        assert!(app.state.cooldown("small", "m").is_none());
        // `tiny` (same route, smaller window) must have been peer-skipped:
        // only `small`'s own attempt hit the endpoint.
        assert_eq!(small_calls.load(Ordering::SeqCst), 1, "peer must be spared the call");
    }

    #[tokio::test]
    async fn all_context_failures_return_400_not_429() {
        use axum::response::IntoResponse;
        use axum::routing::post;
        let router = axum::Router::new().route("/small", post(|| async {
            (
                axum::http::StatusCode::BAD_REQUEST,
                r#"{"error":{"message":"context length exceeded"}}"#,
            )
                .into_response()
        }));
        let base = mock_server(router).await;
        let app = test_app(
            &format!(
                r#"
                [server]
                [providers.a]
                base_url = "{base}/small"
                models = [{{ id = "m", context_length = 8000 }}]
                [providers.b]
                base_url = "{base}/small"
                models = [{{ id = "m", context_length = 8000 }}]
                [groups.free]
                models = ["a/m", "b/m"]
                "#
            ),
            "ctx_exhaust",
        );
        let payload = json!({"model": "free", "messages": [{"role": "user", "content": "hi"}]});
        let out = handle_chat(app, ClientFormat::Openai, payload, ClientContext::default()).await;
        match out {
            Outcome::Json { status, body, .. } => {
                assert_eq!(status, 400, "not a synthetic 429: {body}");
                assert_eq!(body["error"]["type"], "invalid_request_error");
            }
            Outcome::Stream { .. } => panic!("expected json"),
        }
    }

    #[tokio::test]
    async fn context_400_plus_rate_limited_peer_stays_retryable() {
        use axum::response::IntoResponse;
        use axum::routing::post;
        // The large-window candidate is only rate limited — the request is
        // NOT invalid, so the terminal error must stay a retryable 429.
        let router = axum::Router::new()
            .route("/limited", post(|| async {
                (
                    axum::http::StatusCode::TOO_MANY_REQUESTS,
                    [("retry-after", "3600")],
                    r#"{"error":{"message":"slow down"}}"#,
                )
                    .into_response()
            }))
            .route("/small", post(|| async {
                (
                    axum::http::StatusCode::BAD_REQUEST,
                    r#"{"error":{"message":"context length exceeded"}}"#,
                )
                    .into_response()
            }));
        let base = mock_server(router).await;
        let app = test_app(
            &format!(
                r#"
                [server]
                [providers.big]
                base_url = "{base}/limited"
                models = [{{ id = "m", context_length = 1000000 }}]
                [providers.small]
                base_url = "{base}/small"
                models = [{{ id = "m", context_length = 8000 }}]
                [groups.free]
                models = ["big/m", "small/m"]
                "#
            ),
            "ctx_mixed",
        );
        let payload = json!({"model": "free", "messages": [{"role": "user", "content": "hi"}]});
        let out = handle_chat(app, ClientFormat::Openai, payload, ClientContext::default()).await;
        match out {
            Outcome::Json { status, body, .. } => {
                assert_eq!(status, 429, "big was merely throttled — not a 400: {body}");
            }
            Outcome::Stream { .. } => panic!("expected json"),
        }
    }

    #[tokio::test]
    async fn tools_request_skips_non_tool_models() {
        use axum::routing::post;
        let router = axum::Router::new().route("/c", post(|| async {
            axum::Json(json!({
                "id": "x",
                "choices": [{"index": 0,
                    "message": {"role": "assistant", "content": "ok"},
                    "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1},
            }))
        }));
        let base = mock_server(router).await;
        let app = test_app(
            &format!(
                r#"
                [server]
                [providers.notools]
                base_url = "{base}/c"
                models = [{{ id = "m", tool_call = false }}]
                [providers.tools]
                base_url = "{base}/c"
                models = [{{ id = "m", tool_call = true }}]
                [groups.free]
                models = ["notools/m", "tools/m"]
                "#
            ),
            "tool_filter",
        );
        let payload = json!({"model": "free",
            "messages": [{"role": "user", "content": "hi"}],
            "tools": [{"type": "function", "function": {"name": "X", "parameters": {}}}]});
        let out = handle_chat(app, ClientFormat::Openai, payload, ClientContext::default()).await;
        match out {
            Outcome::Json { provider, status, .. } => {
                assert_eq!(status, 200);
                assert_eq!(provider.as_deref(), Some("tools/m"), "tool_call=false must be skipped");
            }
            Outcome::Stream { .. } => panic!("expected json"),
        }
    }

    #[tokio::test]
    async fn drop_params_stripped_before_the_wire() {
        use std::sync::Mutex;
        use axum::routing::post;
        let seen: Arc<Mutex<Option<Value>>> = Arc::new(Mutex::new(None));
        let capture = seen.clone();
        let router = axum::Router::new().route(
            "/c",
            post(move |axum::Json(body): axum::Json<Value>| {
                let capture = capture.clone();
                async move {
                    *capture.lock().unwrap() = Some(body);
                    axum::Json(json!({
                        "id": "x",
                        "choices": [{"index": 0,
                            "message": {"role": "assistant", "content": "ok"},
                            "finish_reason": "stop"}],
                        "usage": {"prompt_tokens": 1, "completion_tokens": 1},
                    }))
                }
            }),
        );
        let base = mock_server(router).await;
        let app = test_app(
            &format!(
                r#"
                [server]
                [providers.g]
                base_url = "{base}/c"
                drop_params = ["reasoning_effort"]
                models = [{{ id = "m", drop_params = ["top_k"] }}]
                "#
            ),
            "drop_params",
        );

        let payload = json!({"model": "g/m", "reasoning_effort": "high", "top_k": 40,
            "temperature": 0.5, "plugins": [{"id": "response-healing"}],
            "messages": [{"role": "user", "content": "hi"}]});
        let out = handle_chat(app, ClientFormat::Openai, payload, ClientContext::default()).await;
        match out {
            Outcome::Json { status, body, .. } => assert_eq!(status, 200, "got {body}"),
            Outcome::Stream { .. } => panic!("expected json"),
        }
        let body = seen.lock().unwrap().take().expect("upstream was called");
        assert!(body.get("reasoning_effort").is_none(), "provider-level drop must strip");
        assert!(body.get("top_k").is_none(), "model-level drop must strip");
        assert_eq!(body["temperature"], 0.5, "unlisted params must survive");
        // `plugins` is pxy's own key, consumed here: an upstream that validates
        // its request body 400s on it.
        assert!(body.get("plugins").is_none(), "plugins must never reach the wire");
    }

    /// Fenced JSON is the commonest thing a model returns to a JSON-mode
    /// request, and a client running `JSON.parse` on it fails. Both OpenAI
    /// dialects must get the bare value back.
    #[tokio::test]
    async fn response_healing_unfences_a_json_answer_for_chat_and_responses() {
        use axum::routing::post;
        let router = axum::Router::new().route(
            "/c",
            post(|| async {
                axum::Json(json!({
                    "id": "x",
                    "choices": [{"index": 0,
                        "message": {"role": "assistant",
                            "content": "```json\n{\"ok\": true}\n```"},
                        "finish_reason": "stop"}],
                    "usage": {"prompt_tokens": 1, "completion_tokens": 1},
                }))
            }),
        );
        let base = mock_server(router).await;
        let plain = format!(
            r#"
            [server]
            [providers.g]
            base_url = "{base}/c"
            models = ["m"]
            "#
        );
        let app = test_app(&plain, "healing_request");

        // A Chat Completions client, healing asked for by request plugin.
        let payload = json!({"model": "g/m",
            "response_format": {"type": "json_object"},
            "plugins": [{"id": "response-healing"}],
            "messages": [{"role": "user", "content": "hi"}]});
        let out =
            handle_chat(app.clone(), ClientFormat::Openai, payload, ClientContext::default()).await;
        let Outcome::Json { body, .. } = out else { panic!("expected json") };
        assert_eq!(body["choices"][0]["message"]["content"], "{\"ok\": true}");

        // A Responses client, the same way: the route translates the request,
        // routes it, then rewrites the outcome — the heal must already have
        // happened by the time that rewrite reads `content`.
        let rp = json!({"model": "g/m",
            "text": {"format": {"type": "json_object"}},
            "plugins": [{"id": "response-healing"}],
            "input": [{"role": "user", "content": "hi"}]});
        let chat_payload = crate::translate::responses::request(&rp);
        let out =
            handle_chat(app, ClientFormat::Openai, chat_payload, ClientContext::default()).await;
        let Outcome::Json { body, .. } = out else { panic!("expected json") };
        let rewritten = crate::translate::responses::response(&body, "g/m");
        assert_eq!(rewritten["output"][0]["content"][0]["text"], "{\"ok\": true}");

        // And the config flag reaches a client that asked for nothing.
        let app = test_app(
            &format!("{plain}\n[plugins]\nresponse_healing = true\n"),
            "healing_config",
        );
        let payload = json!({"model": "g/m",
            "response_format": {"type": "json_object"},
            "messages": [{"role": "user", "content": "hi"}]});
        let out = handle_chat(app, ClientFormat::Openai, payload, ClientContext::default()).await;
        let Outcome::Json { body, .. } = out else { panic!("expected json") };
        assert_eq!(body["choices"][0]["message"]["content"], "{\"ok\": true}");
    }

    /// Healing rewrites what the model said, so the gate is narrow on purpose:
    /// the client must have asked for JSON, and a stream has no assembled
    /// content to rewrite.
    #[test]
    fn response_healing_applies_only_to_a_non_streaming_json_mode_openai_turn() {
        let off: Config = toml::from_str("[server]\n").unwrap();
        let on: Config = toml::from_str("[server]\n[plugins]\nresponse_healing = true\n").unwrap();
        let json_mode = json!({"response_format": {"type": "json_object"}});

        assert!(should_heal(&on, ClientFormat::Openai, &json_mode), "config flag");
        assert!(
            should_heal(&off, ClientFormat::Openai, &json!({
                "response_format": {"type": "json_schema"},
                "plugins": [{"id": "response-healing"}]})),
            "request plugin, json_schema"
        );

        assert!(!should_heal(&off, ClientFormat::Openai, &json_mode), "nobody asked");
        assert!(
            !should_heal(&on, ClientFormat::Anthropic, &json_mode),
            "an Anthropic client is never healed"
        );
        assert!(
            !should_heal(&on, ClientFormat::Openai, &json!({"messages": []})),
            "no response_format: the client never promised itself JSON"
        );
        assert!(
            !should_heal(&on, ClientFormat::Openai, &json!({"stream": true,
                "response_format": {"type": "json_object"}})),
            "a stream is not healed"
        );
        assert!(
            !should_heal(&off, ClientFormat::Openai, &json!({
                "response_format": {"type": "json_object"},
                "plugins": [{"id": "web-search"}]})),
            "an unknown id is ignored, not read as consent"
        );
        assert_eq!(
            request_plugins(&json!({"plugins": [{"id": "a"}, {"nope": 1}, {"id": "b"}]})),
            ["a", "b"],
            "an entry with no id is skipped, never a 400"
        );
        assert!(request_plugins(&json!({"messages": []})).is_empty());
    }

    /// The two-page fixture media::pdf is tested on: page 1 typeset, page 2
    /// empty.
    const PDF: &[u8] = include_bytes!("../tests/fixtures/hello.pdf");

    fn pdf_data_url() -> String {
        use base64::Engine as _;
        format!(
            "data:application/pdf;base64,{}",
            base64::engine::general_purpose::STANDARD.encode(PDF)
        )
    }

    /// An upstream that answers anything and keeps every body it was sent.
    async fn recording_upstream(seen: Arc<std::sync::Mutex<Vec<Value>>>) -> String {
        use axum::routing::post;
        let router = axum::Router::new().route(
            "/c",
            post(move |axum::Json(body): axum::Json<Value>| {
                let seen = seen.clone();
                async move {
                    seen.lock().unwrap().push(body);
                    axum::Json(json!({"id": "x", "choices": [{"index": 0,
                        "message": {"role": "assistant", "content": "ok"},
                        "finish_reason": "stop"}]}))
                }
            }),
        );
        mock_server(router).await
    }

    fn pdf_request(file_data: &str, filename: &str) -> Value {
        json!({"model": "p/m", "messages": [{"role": "user", "content": [
            {"type": "file", "file": {"filename": filename, "file_data": file_data}},
            {"type": "text", "text": "summarise"},
        ]}]})
    }

    fn first_part(body: &Value) -> Value {
        body["messages"][0]["content"][0].clone()
    }

    /// The model never sees a file part: it sees the document as text, under
    /// the name the client gave it. The second turn of a conversation resends
    /// the same history, so the parse is read from kv instead of run again —
    /// proved here by leaving a different text under the key.
    #[tokio::test]
    async fn file_parser_swaps_a_pdf_part_and_caches_the_parse() {
        let seen: Arc<std::sync::Mutex<Vec<Value>>> = Default::default();
        let base = recording_upstream(seen.clone()).await;
        let app = test_app(
            &format!("[server]\n[providers.p]\nbase_url = \"{base}/c\"\nmodels = [\"m\"]\n"),
            "file_parser_swap",
        );

        let payload = pdf_request(&pdf_data_url(), "hello.pdf");
        handle_chat(app.clone(), ClientFormat::Openai, payload.clone(), ClientContext::default())
            .await;
        let sent = first_part(&seen.lock().unwrap()[0]);
        assert_eq!(sent["type"], "text", "{sent}");
        let text = sent["text"].as_str().unwrap();
        assert!(text.starts_with("[file hello.pdf]\n"), "{text}");
        assert!(text.contains("Hello, pxy."), "{text}");
        assert!(text.contains("[page 2: no text]"), "{text}");

        let key = format!("file_parse:{}", crate::media::pdf::sha256_hex(PDF));
        app.state.kv_set(&key, "## Page 1\n\nread from kv").unwrap();
        handle_chat(app.clone(), ClientFormat::Openai, payload, ClientContext::default()).await;
        let again = first_part(&seen.lock().unwrap()[1]);
        assert!(
            again["text"].as_str().unwrap().contains("read from kv"),
            "the same bytes must not be parsed twice: {again}"
        );
    }

    /// A file pxy cannot read is one line the model can act on, and the turn
    /// goes on: dropping the part would have the model answer about a
    /// document it was never given.
    #[tokio::test]
    async fn file_parser_reports_what_it_cannot_parse() {
        let seen: Arc<std::sync::Mutex<Vec<Value>>> = Default::default();
        let base = recording_upstream(seen.clone()).await;
        let app = test_app(
            &format!("[server]\n[providers.p]\nbase_url = \"{base}/c\"\nmodels = [\"m\"]\n"),
            "file_parser_errors",
        );
        let ctx = ClientContext::default();

        let cases = [
            ("data:text/plain;base64,aGkK", "notes.txt", "not a PDF"),
            (
                &format!("data:application/pdf;base64,{}", "A".repeat(70 * 1024 * 1024)),
                "huge.pdf",
                "file is too large",
            ),
            // Port 1 is never listening: a fetch that cannot happen at all.
            ("https://127.0.0.1:1/gone.pdf", "gone.pdf", "cannot fetch the file"),
            ("ftp://e.test/x.pdf", "odd.pdf", "data URL or an https URL"),
        ];
        for (i, (data, name, reason)) in cases.iter().enumerate() {
            handle_chat(app.clone(), ClientFormat::Openai, pdf_request(data, name), ctx.clone())
                .await;
            let sent = first_part(&seen.lock().unwrap()[i]);
            let text = sent["text"].as_str().unwrap_or_default();
            assert!(text.starts_with(&format!("[file {name}: ")), "{text}");
            assert!(text.contains(reason), "{text}");
        }
    }

    /// `enabled = false` is the way out: the part reaches the upstream as the
    /// client sent it.
    #[tokio::test]
    async fn file_parser_disabled_leaves_the_part_alone() {
        let seen: Arc<std::sync::Mutex<Vec<Value>>> = Default::default();
        let base = recording_upstream(seen.clone()).await;
        let app = test_app(
            &format!(
                "[server]\n[plugins.file_parser]\nenabled = false\n\
                 [providers.p]\nbase_url = \"{base}/c\"\nmodels = [\"m\"]\n"
            ),
            "file_parser_off",
        );

        handle_chat(
            app,
            ClientFormat::Openai,
            pdf_request(&pdf_data_url(), "hello.pdf"),
            ClientContext::default(),
        )
        .await;
        let sent = first_part(&seen.lock().unwrap()[0]);
        assert_eq!(sent["type"], "file", "{sent}");
        assert_eq!(sent["file"]["filename"], "hello.pdf");
    }

    /// pi's openai-completions client sends OpenAI's newest dialect for a
    /// reasoning model when it cannot tell the real upstream behind pxy:
    /// `developer` (DeepSeek 400s on the literal variant) and
    /// `max_completion_tokens` (opencode-go ignored it, so the output cap did
    /// nothing). The passthrough must rewrite both for a compatible upstream.
    #[tokio::test]
    async fn openai_dialect_is_normalized_before_the_wire() {
        use std::sync::Mutex;
        use axum::routing::post;
        let seen: Arc<Mutex<Option<Value>>> = Arc::new(Mutex::new(None));
        let capture = seen.clone();
        let router = axum::Router::new().route(
            "/c",
            post(move |axum::Json(body): axum::Json<Value>| {
                let capture = capture.clone();
                async move {
                    *capture.lock().unwrap() = Some(body);
                    axum::Json(json!({
                        "id": "x",
                        "choices": [{"index": 0,
                            "message": {"role": "assistant", "content": "ok"},
                            "finish_reason": "stop"}],
                        "usage": {"prompt_tokens": 1, "completion_tokens": 1},
                    }))
                }
            }),
        );
        let base = mock_server(router).await;
        let app = test_app(
            &format!(
                r#"
                [server]
                [providers.p]
                base_url = "{base}/c"
                models = ["m"]
                "#
            ),
            "openai_dialect",
        );

        let payload = json!({"model": "p/m", "max_completion_tokens": 4096, "messages": [
            {"role": "developer", "content": "rules"},
            {"role": "user", "content": "hi"},
        ]});
        let out = handle_chat(app, ClientFormat::Openai, payload, ClientContext::default()).await;
        match out {
            Outcome::Json { status, body, .. } => assert_eq!(status, 200, "got {body}"),
            Outcome::Stream { .. } => panic!("expected json"),
        }
        let body = seen.lock().unwrap().take().expect("upstream was called");
        assert_eq!(body["messages"][0]["role"], "system");
        assert_eq!(body["messages"][0]["content"], "rules");
        assert_eq!(body["messages"][1]["role"], "user");
        assert_eq!(body["max_tokens"], 4096, "cap must survive under the upstream's name");
        assert!(body.get("max_completion_tokens").is_none());
    }

    #[tokio::test]
    async fn openai_native_keeps_its_own_dialect() {
        use std::sync::Mutex;
        use axum::routing::post;
        let seen: Arc<Mutex<Option<Value>>> = Arc::new(Mutex::new(None));
        let capture = seen.clone();
        let router = axum::Router::new().route(
            "/c",
            post(move |axum::Json(body): axum::Json<Value>| {
                let capture = capture.clone();
                async move {
                    *capture.lock().unwrap() = Some(body);
                    axum::Json(json!({
                        "id": "x",
                        "choices": [{"index": 0,
                            "message": {"role": "assistant", "content": "ok"},
                            "finish_reason": "stop"}],
                        "usage": {"prompt_tokens": 1, "completion_tokens": 1},
                    }))
                }
            }),
        );
        let base = mock_server(router).await;
        let app = test_app(
            &format!(
                r#"
                [server]
                [providers.o]
                base_url = "{base}/c"
                openai_native = true
                models = ["m"]
                "#
            ),
            "openai_native",
        );

        // A Claude Code-shaped client sends `system` + `max_tokens`; an
        // OpenAI-native reasoning upstream needs `developer` +
        // `max_completion_tokens`.
        let payload = json!({"model": "o/m", "max_tokens": 512, "messages": [
            {"role": "system", "content": "rules"},
            {"role": "user", "content": "hi"},
        ]});
        let out = handle_chat(app, ClientFormat::Anthropic, payload, ClientContext::default()).await;
        match out {
            Outcome::Json { status, body, .. } => assert_eq!(status, 200, "got {body}"),
            Outcome::Stream { .. } => panic!("expected json"),
        }
        let body = seen.lock().unwrap().take().expect("upstream was called");
        assert_eq!(body["max_completion_tokens"], 512);
        assert!(body.get("max_tokens").is_none());
    }

    #[tokio::test]
    async fn disconnect_still_records_usage() {
        use axum::routing::post;
        // Upstream reports usage in its first chunk; the client then walks
        // away without reading the stream to its end (Ctrl-C'd agent turn).
        let router = axum::Router::new().route("/s", post(|| async {
            "data: {\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"content\":\"hi\"}}],\"usage\":{\"prompt_tokens\":5,\"completion_tokens\":3}}\n\n"
        }));
        let base = mock_server(router).await;
        let app = test_app(
            &format!(
                r#"
                [server]
                [providers.f]
                base_url = "{base}/s"
                models = ["m"]
                "#
            ),
            "disconnect_usage",
        );

        let payload = json!({"model": "f/m", "stream": true,
            "messages": [{"role": "user", "content": "hi"}]});
        let out = handle_chat(app.clone(), ClientFormat::Openai, payload, ClientContext::default())
            .await;
        match out {
            Outcome::Stream { body, .. } => drop(body), // the disconnect
            Outcome::Json { status, body, .. } => panic!("expected stream, got {status}: {body}"),
        }
        let total = app.state.usage_total("f").unwrap();
        assert_eq!(total.requests, 1);
        assert_eq!(total.tokens, 8, "usage seen before the disconnect must be recorded");
    }

    #[tokio::test]
    async fn fatal_stream_error_event_passes_through() {
        use axum::routing::post;
        // 200, then the real error as the only stream event: a 400-class
        // failure must reach the client unmodified, not become a retry storm.
        let router = axum::Router::new().route("/err", post(|| async {
            "data: {\"error\":{\"code\":400,\"message\":\"context length exceeded\"}}\n\n"
        }));
        let base = mock_server(router).await;
        let app = test_app(
            &format!(
                r#"
                [server]
                [providers.e]
                base_url = "{base}/err"
                models = ["m"]
                "#
            ),
            "fatal_stream_error",
        );

        let payload = json!({"model": "e/m", "stream": true,
            "messages": [{"role": "user", "content": "hi"}]});
        let out = handle_chat(app, ClientFormat::Openai, payload, ClientContext::default()).await;
        match out {
            Outcome::Json { status, body, .. } => {
                assert_eq!(status, 400);
                assert_eq!(
                    body["error"]["message"], "context length exceeded",
                    "original error body must pass through: {body}"
                );
            }
            Outcome::Stream { .. } => panic!("expected the raw error, got a stream"),
        }
    }

    #[test]
    fn remaining_zero_cools_with_reset_hint() {
        let s = state("remaining_zero");
        check_quota_exhaustion(
            &s,
            "groq",
            &headers(&[
                ("x-ratelimit-remaining-tokens", "0"),
                ("x-ratelimit-reset-tokens", "7.66s"),
            ]),
        );
        let cd = s.cooldown("groq", "any").expect("cooldown set");
        assert!(cd.reason.contains("x-ratelimit-remaining-tokens"));
        // nonzero remaining leaves the provider alone
        let s2 = state("remaining_ok");
        check_quota_exhaustion(&s2, "groq", &headers(&[("x-ratelimit-remaining", "42")]));
        assert!(s2.cooldown("groq", "any").is_none());
    }
}
