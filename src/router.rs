//! The routing engine: candidate filtering, fallback walk, error
//! classification, usage recording. Synthesis of the OmniRoute + litellm
//! research (docs/03, docs/05).

use std::sync::Arc;
use std::time::Duration;

use bytes::Bytes;
use futures_util::StreamExt;
use jiff::Timestamp;
use serde_json::{json, Value};
use tracing::{info, warn};

use crate::catalog::{Candidate, Catalog};
use crate::config::{Config, ErrorAction, WireFormat};
use crate::secrets::Secrets;
use crate::state::State;
use crate::translate::sse::SseParser;
use crate::translate::think::ThinkFilter;
use crate::translate::tool_text::ToolTextFilter;
use crate::translate::web_search;
use crate::translate::{anthropic_to_openai, estimate_tokens, kiro, openai_to_anthropic, TokenUsage};
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
    /// x-initiator (Copilot billing: "agent" turns are free)
    pub initiator: Option<String>,
    /// anthropic-beta headers, forwarded verbatim to anthropic upstreams
    pub anthropic_beta: Option<String>,
    /// Which coding agent sent this request ("claude", "codex", …), parsed
    /// from the x-pxy-agent header or the api-key suffix `pxy launch` sets.
    /// Only feeds the per-model usage stats — never routing.
    pub agent: Option<String>,
}

/// Outcome handed to the HTTP layer.
pub enum Outcome {
    Json {
        status: u16,
        body: Value,
        provider: Option<String>,
    },
    Stream {
        provider: String,
        body: axum::body::Body,
    },
}

/// kv key holding the route pin, set from `pxy route` / the desktop panel.
/// Read per request so a pin takes effect without a daemon restart.
pub const ROUTE_PIN_KEY: &str = "route_pin";

/// Candidates for a request, honoring the route pin: on a GROUP request the
/// pinned model is walked FIRST, with the group's chain behind it as fallback —
/// pinning must never cost the failover safety a group exists for. One pin
/// covers every group: an agent is launched with a fixed model id, so the pin
/// is the only way to steer a running session. Explicit model requests are
/// untouched, and a pin that no longer resolves (config edit, provider
/// disabled) degrades to the plain chain.
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
    // Manual pin: walked FIRST, ahead of session affinity — `pxy route` is an
    // explicit human decision.
    let mut pinned = None;
    if let Some(pin) = state.kv_get(ROUTE_PIN_KEY).ok().flatten().filter(|p| !p.is_empty()) {
        // is_listed, not just resolves: resolve() fabricates a candidate for
        // any id under an enabled provider, and a pin gone stale (config
        // edit, refresh dropping the model) must degrade to the chain, not
        // put a phantom at the head of every group walk.
        let resolved = catalog.resolve(cfg, &pin);
        if !resolved.is_empty() && resolved.iter().all(|c| catalog.is_listed(&c.full_id())) {
            pinned = Some(resolved);
        } else {
            warn!(pin, "route pin is not in the catalog; using the group chain");
        }
    }
    // Session affinity: the candidate this conversation last won on walks
    // first, so a post-failover conversation keeps its prompt-cache locality
    // instead of bouncing back to the chain head. A stale or unlisted
    // binding is ignored — the walk's winner rebinds it (self-healing).
    let mut affinity = None;
    if pinned.is_none() {
        if let Some(key) = session {
            if let Some(full_id) = state.session_get(key) {
                let bound = catalog.resolve(cfg, &full_id);
                if !bound.is_empty() && bound.iter().all(|c| catalog.is_listed(&c.full_id())) {
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

pub async fn handle_chat(
    app: SharedApp,
    client_format: ClientFormat,
    payload: Value,
    ctx: ClientContext,
) -> Outcome {
    let requested = payload["model"]
        .as_str()
        .filter(|m| !m.is_empty())
        .map(str::to_string)
        .unwrap_or_else(|| app.cfg.default_route());
    let stream = payload["stream"].as_bool().unwrap_or(false);

    // In-band magic prompt: a last user message of exactly "@@usage" is
    // answered locally with the quota report — zero tokens, no upstream.
    if is_usage_magic(&payload) {
        return usage_outcome(usage_report(&app), client_format, stream);
    }

    // Session affinity key (group walks only): keeps a conversation on its
    // last winning candidate for prompt-cache locality.
    let session_key = if app.catalog.is_group(&requested) { session_key(&payload) } else { None };
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

    let mut skipped: Vec<String> = Vec::new();
    let multi = candidates.len() > 1;
    // Set when an upstream 400s with a context-window error: our chars/4
    // estimate under-counted, so every candidate at that context size or
    // below would fail identically — skip them instead of burning calls.
    let mut ctx_too_small: Option<u64> = None;
    // Sticky across walks: ANY non-context obstacle (cooldown, rpm, limits,
    // a 429/5xx attempt…) means the terminal error must stay retryable —
    // only when context was the sole problem is a 400 the honest answer.
    let mut other_failures = false;

    for attempt in 0..=MAX_RETRIES {
        skipped.clear();
        let mut saw_rpm_limit = false;
        for cand in &candidates {
            if ctx_too_small.is_some_and(|c| cand.model.context_length <= c) {
                skipped.push(format!("{}: context window too small", cand.full_id()));
                continue;
            }
            if let Err(reason) = check_candidate(&app, cand, input_estimate, wants_tools, multi) {
                saw_rpm_limit |= reason == "rpm limit";
                // Filter reasons for context start with "context"; everything
                // else (cooldown/rpm/limits/disabled) is a non-context
                // obstacle a later retry might clear.
                other_failures |= !reason.starts_with("context");
                skipped.push(format!("{}: {reason}", cand.full_id()));
                continue;
            }

            match try_candidate(&app, cand, client_format, &payload, stream, input_estimate, &ctx, multi)
                .await
            {
                AttemptResult::Done(outcome) => {
                    // A real success repairs the model's failure-rate record.
                    app.state.model_result(&cand.state_provider(), &cand.model.id, true);
                    // ...and rebinds the conversation's session affinity.
                    if let Some(key) = &session_key {
                        app.state.session_set(key, &cand.full_id());
                    }
                    return outcome;
                }
                AttemptResult::Skip(reason) => {
                    warn!(candidate = %cand.full_id(), %reason, "failover");
                    other_failures = true;
                    // A real attempt failed: feed the failure-rate rule.
                    app.state.model_result(&cand.state_provider(), &cand.model.id, false);
                    skipped.push(format!("{}: {reason}", cand.full_id()));
                }
                AttemptResult::SkipContextWindow(reason) => {
                    // The real tokenizer overruled our estimate. No cooldown
                    // (a smaller request to this model would work fine).
                    warn!(candidate = %cand.full_id(), %reason, "failover (context window)");
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
    error_outcome(
        client_format,
        429,
        "overloaded_error",
        &format!(
            "no provider available for '{requested}' (tried/skipped: {})",
            skipped.join("; ")
        ),
    )
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
        return Outcome::Json { status: 200, body, provider: Some("pxy".into()) };
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
    Outcome::Stream { provider: "pxy".into(), body: axum::body::Body::from(sse) }
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
        .filter_map(|c| app.state.recovery_wait(&c.provider, &c.model.id))
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
fn check_candidate(
    app: &App,
    cand: &Candidate,
    input_estimate: u64,
    wants_tools: bool,
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
    /// Upstream 400'd because the input exceeds THIS model's real context
    /// window (our estimate under-counted): skip it and every candidate
    /// with the same or smaller window, no cooldown.
    SkipContextWindow(String),
    /// Fatal for the whole request: return this to the client.
    Fatal(Outcome),
}

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
    let provider_cfg = app.cfg.providers.get(&cand.provider).unwrap();
    let upstream_format = cand.format(provider_cfg);

    // Build the upstream body.
    let mut body = match (client_format, upstream_format) {
        (ClientFormat::Openai, WireFormat::Openai) => payload.clone(),
        (ClientFormat::Anthropic, WireFormat::Anthropic) => payload.clone(),
        (ClientFormat::Anthropic, WireFormat::Openai) => anthropic_to_openai::request(payload),
        (ClientFormat::Openai, WireFormat::Anthropic) => {
            openai_to_anthropic::request(payload, cand.model.max_output_tokens)
        }
        // Kiro takes neither dialect: normalize to Anthropic first (reusing
        // the existing translator), then build conversationState from it.
        (ClientFormat::Anthropic, WireFormat::Kiro) => {
            let mut a = payload.clone();
            crate::translate::anthropic_sanitize::sanitize(&mut a);
            kiro::request(&a, &cand.model.id, "", &now_iso8601())
        }
        (ClientFormat::Openai, WireFormat::Kiro) => {
            let mut a = openai_to_anthropic::request(payload, cand.model.max_output_tokens);
            crate::translate::anthropic_sanitize::sanitize(&mut a);
            kiro::request(&a, &cand.model.id, "", &now_iso8601())
        }
    };
    // Anthropic validates history strictly (thinking signatures, tool
    // pairing, empty blocks) and the passthrough path replays whatever the
    // client accumulated — repair it at the one choke point.
    if upstream_format == WireFormat::Anthropic {
        crate::translate::anthropic_sanitize::sanitize(&mut body);
    }
    if upstream_format != WireFormat::Kiro {
        body["model"] = json!(cand.model.id);
    }
    // force_stream: the upstream misbehaves without `stream: true` on this
    // model, so stream upstream regardless and re-assemble JSON for a
    // non-streaming client. (Kiro always streams; nothing to force.)
    let force_stream = !stream && cand.model.force_stream && upstream_format != WireFormat::Kiro;
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

    // Keys this upstream 400s on, dropped LAST so they win over anything the
    // translation or the stream_options default put in the body. `model` and
    // `stream` are pxy's own routing keys — dropping them wouldn't disable a
    // param, it would corrupt the request pxy itself built, so they're
    // ignored here. (A provider body_patch — kiro's profileArn — merges
    // later and also can't be stripped.)
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
    let prepared = match crate::providers::prepare(
        &cand.provider,
        provider_cfg,
        &app.secrets,
        &app.state,
        &app.http,
        account,
    )
    .await
    {
        Ok(p) => p,
        Err(e) => return AttemptResult::Skip(format!("prepare failed: {e:#}")),
    };

    // The Anthropic OAuth endpoint rejects requests missing the Claude Code
    // system sentinel; real Claude Code traffic already has it (no-op).
    if provider_cfg.kind == crate::config::ProviderKind::ClaudeOauth {
        crate::providers::claude::ensure_sentinel(&mut body);
    }

    // Providers may need fields inside the body (kiro's profileArn), which is
    // only known after credentials resolve.
    if let Some(patch) = &prepared.body_patch {
        if let (Some(dst), Some(src)) = (body.as_object_mut(), patch.as_object()) {
            for (k, v) in src {
                dst.insert(k.clone(), v.clone());
            }
        }
    }

    // Textual tool-call extraction: OpenAI upstreams only, and only when the
    // request declared tools (otherwise the markup is content, not protocol).
    let tool_names = (upstream_format == WireFormat::Openai)
        .then(|| declared_tool_names(payload))
        .flatten();

    let initiator_override = ctx
        .initiator
        .as_deref()
        .filter(|v| *v == "agent" || *v == "user");

    // The total timeout must NOT apply to client-streaming requests: it
    // spans "connect until the body finishes" (reqwest semantics), so a long
    // agentic turn dies mid-stream at timeout_secs with a clean-looking
    // end_turn. Streams are bounded instead by the headers deadline below
    // and a per-read stall deadline in the unfold. Non-streaming keeps the
    // total timeout (it also bounds force_stream's re-assembly).
    let mut req = app.http.post(&prepared.url).header("content-type", "application/json");
    if !stream {
        req = req.timeout(Duration::from_secs(provider_cfg.timeout_secs));
    }
    for (k, v) in &prepared.headers {
        if k == "x-initiator" && initiator_override.is_some() {
            continue;
        }
        req = req.header(k, v);
    }
    if let Some(init) = initiator_override {
        req = req.header("x-initiator", init);
    }
    if upstream_format == WireFormat::Anthropic {
        if let Some(beta) = &ctx.anthropic_beta {
            req = req.header("anthropic-beta", beta.clone());
        }
        if !prepared.headers.iter().any(|(k, _)| k == "anthropic-version") {
            req = req.header("anthropic-version", "2023-06-01");
        }
    }

    // The hosted web_search tool: pxy runs it for OpenAI upstreams, which have
    // no hosted tools of their own. Whoever built the body decided that —
    // anthropic_to_openai for Claude Code's server tool, responses for
    // `codex --search` — so the marker is the injected function itself.
    // Built before the send so the follow-up call replays this exact request
    // with the results appended.
    let search = (upstream_format == WireFormat::Openai
        && !app.cfg.search.providers.is_empty()
        && body["tools"]
            .as_array()
            .is_some_and(|ts| ts.iter().any(|t| t["function"]["name"] == web_search::TOOL_NAME)))
    .then(|| SearchLoop {
        filter: SearchCallFilter::default(),
        uses_left: web_search::plan(payload).map_or(web_search::DEFAULT_MAX_USES, |p| p.max_uses),
        url: prepared.url.clone(),
        headers: prepared.headers.clone(),
        body: body.clone(),
        timeout: Duration::from_secs(provider_cfg.timeout_secs),
    });

    app.state.rpm_increment(&cand.state_provider());
    // A streaming upstream returns headers as soon as it accepts the request,
    // so silence here means it is not answering at all. With a chain to fall
    // back on, waiting out timeout_secs (600s by default) for that is the
    // wrong trade — one dead provider at the head of a group would strand
    // every request. An explicitly named model still gets the full timeout:
    // there is nothing to fail over to. Non-streaming is exempt because its
    // headers legitimately arrive only once the whole answer is generated.
    let send = req.json(&body).send();
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
    if status >= 400 {
        let retry_after = parse_retry_after(resp.headers());
        let err_body = resp.text().await.unwrap_or_default();
        return classify_error(app, cand, client_format, status, retry_after, err_body, multi);
    }

    // Success: count the request now; tokens follow when usage is known.
    let agent = ctx.agent.as_deref().unwrap_or("");
    record_request(app, agent, &cand.state_provider(), &cand.provider, &cand.model.id);
    app.state.clear_cooldown(&cand.state_provider(), &cand.model.id);
    // After the clear: a success response can still carry "you just used the
    // last of your quota" headers, and that cooldown must survive it.
    check_quota_exhaustion(&app.state, &cand.state_provider(), resp.headers());
    record_free_allowance(&app.state, &cand.state_provider(), resp.headers());
    if !stream {
        info!(candidate = %cand.full_id(), stream, "routed");
    }

    if stream {
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
            search,
        )
        .await
        {
            Ok(outcome) => AttemptResult::Done(outcome),
            Err(StreamFailure::ErrorEvent(data)) => {
                // The stream's first event was the real error: classify it
                // exactly like an HTTP error status would have been, so a
                // fatal 4xx still passes through unmodified instead of
                // becoming a retry storm plus a synthetic 429.
                match error_event_status(&data) {
                    Some(status) => {
                        classify_error(app, cand, client_format, status, None, data, multi)
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
    } else if upstream_format == WireFormat::Kiro {
        // Kiro has no non-streaming mode; collect the eventstream instead.
        let bytes = match resp.bytes().await {
            Ok(b) => b,
            Err(e) => {
                // A 200 whose body can't be read is a misbehaving model: cool
                // it like a dead stream instead of re-probing for free.
                app.state.set_cooldown(
                    &cand.state_provider(),
                    Some(&cand.model.id),
                    None,
                    true,
                    "kiro body read",
                );
                return AttemptResult::Skip(format!("kiro body read: {e}"));
            }
        };
        let (openai_body, usage) =
            kiro::collect_response(&bytes, &cand.model.id, &cand.full_id(), cand.model.context_length);
        record_tokens(app, agent, &cand.state_provider(), &cand.provider, &cand.model.id, usage);
        let client_body = match client_format {
            ClientFormat::Openai => openai_body,
            // kiro::collect_response returns an OpenAI-shaped body, so this is
            // the same conversion an OpenAI upstream would need.
            ClientFormat::Anthropic => {
                anthropic_to_openai::response(&openai_body, &cand.full_id())
            }
        };
        AttemptResult::Done(Outcome::Json {
            status: 200,
            body: client_body,
            provider: Some(cand.full_id()),
        })
    } else {
        let mut upstream_body: Value = if force_stream {
            // Collect the whole upstream stream, then re-assemble the JSON
            // response the client actually asked for.
            let bytes = match resp.bytes().await {
                Ok(b) => b,
                Err(e) => {
                    // Same as the kiro body read: cool the model instead of
                    // re-probing a 200-then-truncate upstream for free.
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
            match upstream_format {
                WireFormat::Openai => crate::translate::aggregate::openai(&events),
                WireFormat::Anthropic => crate::translate::aggregate::anthropic(&events),
                WireFormat::Kiro => unreachable!("kiro handled above"),
            }
        } else {
            match resp.json().await {
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
        let usage = match upstream_format {
            WireFormat::Openai => TokenUsage::from_openai(&upstream_body["usage"]),
            WireFormat::Anthropic => TokenUsage::from_anthropic(&upstream_body["usage"]),
            WireFormat::Kiro => unreachable!("kiro handled above"),
        };
        record_tokens(app, agent, &cand.state_provider(), &cand.provider, &cand.model.id, usage);
        let client_body = match (client_format, upstream_format) {
            (ClientFormat::Openai, WireFormat::Openai)
            | (ClientFormat::Anthropic, WireFormat::Anthropic) => upstream_body,
            (ClientFormat::Anthropic, WireFormat::Openai) => {
                anthropic_to_openai::response(&upstream_body, &cand.full_id())
            }
            (ClientFormat::Openai, WireFormat::Anthropic) => {
                openai_to_anthropic::response(&upstream_body, &cand.full_id())
            }
            (_, WireFormat::Kiro) => unreachable!("kiro handled above"),
        };
        AttemptResult::Done(Outcome::Json {
            status: 200,
            body: client_body,
            provider: Some(cand.full_id()),
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
    status: u16,
    retry_after: Option<Duration>,
    err_body: String,
    multi: bool,
) -> AttemptResult {
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
                passthrough_outcome(client_format, status, &err_body, &cand.full_id())
            }
            ErrorAction::PassthroughCooldown => {
                cool();
                passthrough_outcome(client_format, status, &err_body, &cand.full_id())
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
        // Account-wide problems cool the whole provider; rate limits and
        // upstream errors are usually per-model on aggregators, so they must
        // not sideline the provider's other models.
        let account_wide = matches!(status, 401 | 402 | 403);
        let model_scope = (!account_wide).then_some(cand.model.id.as_str());
        // Auth/credit failures and delisted models don't heal in seconds, so
        // the retry loop must not wait on them (or re-fire a dead key).
        let retryable = !account_wide && status != 404;
        let limits = app.cfg.providers.get(&cand.provider).and_then(|p| p.limits.as_ref());
        // Header wins; else a quota-window body horizon; else a 402 without
        // any hint waits an hour (credits don't reappear in 120s); else the
        // ordinary exponential backoff.
        let (wait, retryable, why) = match retry_after {
            Some(d) => (Some(d), retryable, format!("{status} {reason}")),
            None if status == 429 => match quota_window_cooldown(limits, &err_body) {
                Some((d, why)) => (Some(d), false, format!("429 {why}")),
                None => (None, retryable, format!("{status} {reason}")),
            },
            None if status == 402 => {
                (Some(Duration::from_secs(3600)), false, format!("{status} {reason}"))
            }
            None => (None, retryable, format!("{status} {reason}")),
        };
        app.state.set_cooldown(&cand.state_provider(), model_scope, wait, retryable, &why);
        return AttemptResult::Skip(format!("{status}: {}", truncate(&err_body, 200)));
    }
    passthrough_outcome(client_format, status, &err_body, &cand.full_id())
}

/// Return the upstream error to the client unmodified (the upstream's JSON
/// when it parses, pxy's error shape otherwise). Claude Code's auto-retry
/// depends on unmodified error bodies — this is also what `passthrough`
/// error rules resolve to.
fn passthrough_outcome(
    client_format: ClientFormat,
    status: u16,
    err_body: &str,
    full_id: &str,
) -> AttemptResult {
    let body = serde_json::from_str::<Value>(err_body)
        .unwrap_or_else(|_| error_body(client_format, "api_error", &truncate(err_body, 500)));
    AttemptResult::Fatal(Outcome::Json {
        status,
        body,
        provider: Some(full_id.to_string()),
    })
}

/// kv key holding a provider's last-seen rolling-allowance snapshot.
pub fn free_quota_key(provider: &str) -> String {
    format!("free_quota:{provider}")
}

/// TokenHarbor meters its free tier as a personal rolling 7x24h allowance,
/// priced by the list-price value of the work — pxy cannot compute that, and
/// there is no balance endpoint to poll (every /v1/usage-shaped path 404s).
/// The only readout is a set of undocumented headers on a successful free
/// completion, so remember the last one: `pxy status --remote` reports the
/// snapshot with its age instead of a number nobody can fetch.
///
/// At 100% used the provider is cooled until the window actually rolls: the
/// exhaustion 429 names no window the body classifier recognizes, so the
/// generic ladder would keep re-probing a pool that cannot answer for days.
fn record_free_allowance(state: &State, provider: &str, headers: &reqwest::header::HeaderMap) {
    let get = |name: &str| headers.get(name).and_then(|v| v.to_str().ok()).map(str::trim);
    let Some(pct) =
        get("x-th-free-used-pct").and_then(|s| s.trim_end_matches('%').trim().parse::<f64>().ok())
    else {
        return;
    };
    let resets = get("x-th-free-resets").unwrap_or_default();
    let plan = get("x-th-plan").unwrap_or_default();
    let _ = state.kv_set(
        &free_quota_key(provider),
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
            pct,
            wait_secs = wait.as_secs(),
            "upstream reports the free allowance spent; cooling down until it rolls"
        );
        state.set_cooldown(
            provider,
            None,
            Some(wait),
            false,
            &format!("free allowance spent (resets {resets})"),
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
    // Sanity clamp (litellm): obey only reasonable waits; else exponential backoff.
    if secs > 0 && secs <= 3600 { Some(dur) } else { None }
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
    /// Set for kiro upstreams: binary eventstream is decoded and rewritten as
    /// OpenAI SSE before the normal pipeline sees it.
    kiro: Option<(
        crate::translate::eventstream::EventStreamDecoder,
        kiro::StreamState,
        u64,
    )>,
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
    /// Present when the request carried the web_search server tool and the
    /// upstream speaks OpenAI: pxy runs the search and continues the turn.
    search: Option<SearchLoop>,
}

/// Accumulates the model's calls to `web_search::TOOL_NAME` while stripping
/// them from the chunks, so the client never sees a tool_use it can't run.
#[derive(Default)]
struct SearchCallFilter {
    /// openai tool_call index -> (id, arguments so far). The function name
    /// only rides the first chunk of a call, so later chunks are matched on
    /// the index instead.
    ours: std::collections::HashMap<u64, (String, String)>,
    /// A client tool was called in the same turn. Anthropic's API doesn't run
    /// the search then either — it hands the client tools back first and
    /// searches on the next turn — so pxy leaves the turn alone.
    saw_other: bool,
}

/// The web_search server tool's loop: what the model asked for, what's left of
/// its budget, and the request to replay once results are in.
struct SearchLoop {
    filter: SearchCallFilter,
    uses_left: u64,
    url: String,
    headers: Vec<(String, String)>,
    body: Value,
    timeout: Duration,
}

impl SearchLoop {
    /// Every captured search call this turn, lowest tool-call index first: a
    /// model may issue SEVERAL in one turn (parallel calls), and running only
    /// an arbitrary one silently lost the model's other queries. Capped by
    /// the remaining search budget; empty when another tool call shared the
    /// turn (the continuation can't fake that one) or the budget is spent.
    fn pending(&self) -> Vec<(String, String)> {
        if self.filter.saw_other || self.uses_left == 0 {
            return Vec::new();
        }
        let mut keys: Vec<u64> = self.filter.ours.keys().copied().collect();
        keys.sort_unstable();
        keys.into_iter()
            .take(self.uses_left as usize)
            .filter_map(|k| self.filter.ours.get(&k).cloned())
            .collect()
    }
}

/// Strip calls to the injected search function out of an openai chunk,
/// remembering id + arguments. Returns the rewritten chunk.
fn rewrite_chunk_search(data: &str, f: &mut SearchCallFilter) -> String {
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
            if name != web_search::TOOL_NAME && !f.ours.contains_key(&idx) {
                if !name.is_empty() {
                    f.saw_other = true;
                }
                keep.push(call.clone());
                continue;
            }
            changed = true;
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
        if let Some(kiro_usage) = self.kiro.as_ref().map(|(_, state, ctx_len)| state.usage(*ctx_len))
        {
            self.usage = kiro_usage;
        }
        record_tokens(&self.app, &self.agent, &self.state_provider, &self.provider, &self.model, self.usage);
    }
}

impl StreamCtx {
    /// Process one upstream chunk; returns bytes for the client.
    fn process(&mut self, bytes: &Bytes) -> Bytes {
        // Kiro speaks binary frames; rewrite them as OpenAI SSE so everything
        // below is the ordinary text path.
        let rewritten;
        let bytes = match &mut self.kiro {
            Some((decoder, state, _)) => {
                decoder.push(bytes);
                let frames = decoder.drain();
                if frames.is_empty() {
                    return Bytes::new();
                }
                rewritten = Bytes::from(state.frames_to_sse(frames));
                &rewritten
            }
            None => bytes,
        };
        let Self { parser, kind, think, tooltext, usage, search, .. } = self;
        let events = parser.feed(bytes);
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
                if think.is_none() && tooltext.is_none() && search.is_none() {
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
                        if search.as_ref().is_none_or(|s| s.pending().is_empty()) {
                            out.push_str("data: [DONE]\n\n");
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
                            data = rewrite_chunk_search(&data, &mut s.filter);
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
                        if search.as_ref().is_none_or(|s| s.pending().is_empty()) {
                            out.push_str(&state.on_data(&ev.data));
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
                            data = rewrite_chunk_search(&data, &mut s.filter);
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
                Bytes::from(out)
            }
        }
    }

    /// End-of-stream flush (upstream may end without a terminal marker).
    /// The upstream call ended on a `web_search` call. Run the search, splice
    /// the protocol blocks into the client's stream, and re-issue the request
    /// with the results appended so the model answers from them — all inside
    /// the one message the client is already reading.
    ///
    /// Returns the bytes to emit, or None when there was nothing to search
    /// (then the turn closes normally). A failure to reach a search provider
    /// is reported to the model AND to the client rather than aborting: an
    /// error block is what the real API sends too.
    async fn continue_after_search(&mut self) -> Option<Bytes> {
        let calls = self.search.as_ref()?.pending();
        if calls.is_empty() {
            return None;
        }
        let search = self.search.as_mut()?;
        search.uses_left -= calls.len() as u64;
        search.filter = SearchCallFilter::default();

        // Run every well-formed call, accumulating its client-side block and
        // model-side tool output. A malformed call (unparseable arguments) is
        // skipped: the synthesized assistant message below carries only the
        // calls that were actually served, so the protocol stays valid.
        let mut client_blocks: Vec<(String, String, Value)> = Vec::new();
        let mut tool_calls: Vec<Value> = Vec::new();
        let mut tool_results: Vec<Value> = Vec::new();
        for (call_id, args) in calls {
            let Some(query) = serde_json::from_str::<Value>(&args)
                .ok()
                .and_then(|a| a["query"].as_str().map(String::from))
                .filter(|q| !q.is_empty())
            else {
                continue;
            };
            let found = crate::media::search::run_search(&self.app, &query, 5, None).await;
            let (block, tool_output) = match &found {
                Ok((provider, results)) => {
                    info!(%query, %provider, hits = results.len(), "web_search served");
                    (
                        web_search::result_block(&call_id, results),
                        web_search::results_for_model(results),
                    )
                }
                Err(e) => {
                    warn!(%query, error = %e, "web_search failed");
                    (web_search::error_block(&call_id), format!("Search failed: {e}"))
                }
            };
            client_blocks.push((call_id.clone(), query, block));
            tool_calls.push(json!({
                "id": &call_id,
                "type": "function",
                "function": {"name": web_search::TOOL_NAME, "arguments": &args},
            }));
            tool_results.push(json!({
                "role": "tool", "tool_call_id": &call_id, "content": tool_output,
            }));
        }
        if tool_calls.is_empty() {
            return None;
        }

        // The model's own view of the turn: it called the functions, these
        // are what they returned — one assistant message carrying ALL calls,
        // then one tool message per call (the shape parallel calls require).
        let search = self.search.as_mut()?;
        if let Some(messages) = search.body["messages"].as_array_mut() {
            messages.push(json!({
                "role": "assistant",
                "content": Value::Null,
                "tool_calls": tool_calls,
            }));
            messages.extend(tool_results);
        }

        let mut req = self
            .app
            .http
            .post(&search.url)
            .timeout(search.timeout)
            .header("content-type", "application/json");
        for (k, v) in &search.headers {
            req = req.header(k, v);
        }
        let resp = match req.json(&search.body).send().await {
            Ok(r) if r.status().is_success() => r,
            // No second call means no answer, so the turn ends here rather
            // than hanging: the client still gets the results it can read.
            other => {
                warn!(status = ?other.map(|r| r.status().as_u16()), "web_search continuation failed");
                self.search = None;
                return None;
            }
        };
        self.upstream = resp.bytes_stream().boxed();
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
                // One server_tool_use + result pair per search the model asked for.
                for (call_id, query, block) in &client_blocks {
                    out.push_str(
                        &state.emit_block(web_search::server_tool_use_block(call_id, query)),
                    );
                    out.push_str(&state.emit_block(block.clone()));
                }
            }
            // Passthrough — codex, through /v1/responses. Chat completions has
            // no event for "a search happened", so pxy sends its own marker
            // chunk and translate/responses turns it into the Responses API's
            // `web_search_call` item. Without it the user watches a silent gap
            // for as long as the searches take and assumes it has hung.
            // Shaped as an ordinary empty-choices chunk so a plain chat client
            // (which never asked for this and can't reach here anyway) would
            // skip it the way it skips the usage chunk.
            _ => {
                let spent = std::mem::take(&mut self.usage);
                record_tokens(&self.app, &self.agent, &self.state_provider, &self.provider, &self.model, spent);
                for (call_id, query, _) in &client_blocks {
                    out.push_str(&format!(
                        "data: {}\n\n",
                        json!({
                            "object": "chat.completion.chunk",
                            "choices": [],
                            "pxy_web_search": {"id": call_id, "query": query},
                        })
                    ));
                }
            }
        }
        Some(Bytes::from(out))
    }

    fn finish(&mut self) -> Bytes {
        // Kiro never sends a terminating SSE event: synthesize the closing
        // chunks (incl. buffered tool arguments) and run them through the
        // client-format pipeline before the normal finish.
        if let Some((_, state, ctx_len)) = &mut self.kiro {
            let tail = Bytes::from(state.finish());
            self.usage = state.usage(*ctx_len);
            let translated = {
                let Self { parser, kind, think, usage, .. } = self;
                let events = parser.feed(&tail);
                match kind {
                    StreamKind::OpenaiPass => tail.clone(),
                    StreamKind::ToAnthropic(st) => {
                        let mut out = String::new();
                        for ev in &events {
                            out.push_str(&st.on_data(&ev.data));
                        }
                        // Close the anthropic message properly (message_delta
                        // + message_stop); the kiro tail alone isn't enough.
                        out.push_str(&st.finish());
                        let _ = (think, usage);
                        Bytes::from(out)
                    }
                    _ => tail.clone(),
                }
            };
            record_tokens(&self.app, &self.agent, &self.state_provider, &self.provider, &self.model, self.usage);
            return translated;
        }
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
            _ => match self.tooltext.as_mut().and_then(tooltext_flush_chunk) {
                // Abrupt EOF with buffered text: hand it to the client raw.
                Some(tail) => Bytes::from(format!("data: {tail}\n\n")),
                None => Bytes::new(),
            },
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
    search: Option<SearchLoop>,
) -> Result<Outcome, StreamFailure> {
    let kind = match (client_format, upstream_format) {
        (ClientFormat::Openai, WireFormat::Openai) => StreamKind::OpenaiPass,
        (ClientFormat::Anthropic, WireFormat::Anthropic) => StreamKind::AnthropicPass,
        (ClientFormat::Anthropic, WireFormat::Openai) => StreamKind::ToAnthropic(
            anthropic_to_openai::StreamState::new(&cand.full_id(), input_estimate),
        ),
        (ClientFormat::Openai, WireFormat::Anthropic) => {
            StreamKind::ToOpenai(openai_to_anthropic::StreamState::new(&cand.full_id()))
        }
        // Kiro frames are converted to OpenAI SSE before this point, so the
        // client side behaves exactly as it would for an OpenAI upstream.
        (ClientFormat::Openai, WireFormat::Kiro) => StreamKind::OpenaiPass,
        (ClientFormat::Anthropic, WireFormat::Kiro) => StreamKind::ToAnthropic(
            anthropic_to_openai::StreamState::new(&cand.full_id(), input_estimate),
        ),
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
        kiro: (upstream_format == WireFormat::Kiro).then(|| {
            (
                crate::translate::eventstream::EventStreamDecoder::new(),
                kiro::StreamState::new(&cand.model.id),
                cand.model.context_length,
            )
        }),
        kind,
        think: parse_think.then(ThinkFilter::new),
        tooltext: tool_names.map(ToolTextFilter::new),
        usage: TokenUsage::default(),
        agent: agent.to_string(),
        provider: cand.provider.clone(),
        state_provider: cand.state_provider(),
        model: cand.model.id.clone(),
        app,
        upstream: resp.bytes_stream().boxed(),
        done: false,
        stall,
        search,
    };

    // Pre-commit read: hold processed client bytes until the upstream yields
    // its first complete event. `sniff` re-parses the raw bytes because
    // process() consumes them through translators that don't surface events.
    // Kiro speaks binary frames, not SSE: its proof of life is the first
    // nonempty processed output instead.
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
                let events = if ctx.kiro.is_none() { sniff.feed(&bytes) } else { Vec::new() };
                let out = ctx.process(&bytes);
                if !out.is_empty() {
                    held.push(out);
                }
                if ctx.kiro.is_some() {
                    if !held.is_empty() {
                        break;
                    }
                    continue;
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
                // A queued web_search swaps in a fresh upstream response and
                // keeps the same client stream going.
                if let Some(blocks) = ctx.continue_after_search().await {
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
    })
}

/// Non-streaming: move `<think>` spans in every choice's message.content
/// into message.reasoning_content.
fn extract_think_from_response(body: &mut Value) {
    let Some(choices) = body["choices"].as_array_mut() else { return };
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

/// RFC3339 timestamp for kiro's `[Context: Current time is ...]` prefix.
fn now_iso8601() -> String {
    Timestamp::now().to_string()
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
        let mut f = SearchCallFilter::default();
        let out = rewrite_chunk_search(
            &json!({"choices": [{"index": 0, "delta": {"tool_calls": [{
                "index": 0, "id": "call_1", "type": "function",
                "function": {"name": web_search::TOOL_NAME, "arguments": "{\"query\":"}
            }]}}]})
            .to_string(),
            &mut f,
        );
        assert!(!out.contains("tool_calls"), "{out}");

        // Arguments streamed across chunks: later ones carry no name.
        let out = rewrite_chunk_search(
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

    /// A client tool called in the same turn is forwarded untouched, and its
    /// presence cancels the search: Anthropic's API hands the client tools back
    /// first and searches on the following turn.
    #[test]
    fn client_tool_calls_survive_and_cancel_the_search() {
        let mut f = SearchCallFilter::default();
        let out = rewrite_chunk_search(
            &json!({"choices": [{"index": 0, "delta": {"tool_calls": [
                {"index": 0, "id": "c1", "function": {"name": web_search::TOOL_NAME, "arguments": "{}"}},
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

        let loop_ = SearchLoop {
            filter: f,
            uses_left: 5,
            url: String::new(),
            headers: Vec::new(),
            body: Value::Null,
            timeout: Duration::from_secs(1),
        };
        assert!(loop_.pending().is_empty());
    }

    /// A turn may carry SEVERAL search calls (parallel tool calls): all of
    /// them must be pending, lowest index first, capped by the budget — the
    /// old single-arbitrary-pick silently dropped the model's other queries.
    #[test]
    fn parallel_search_calls_are_all_pending_lowest_index_first() {
        let mut f = SearchCallFilter::default();
        // Insert out of order on purpose.
        f.ours.insert(1, ("call_2".into(), "{\"query\":\"b\"}".into()));
        f.ours.insert(0, ("call_1".into(), "{\"query\":\"a\"}".into()));
        f.ours.insert(2, ("call_3".into(), "{\"query\":\"c\"}".into()));
        let mut loop_ = SearchLoop {
            filter: f,
            uses_left: 3,
            url: String::new(),
            headers: Vec::new(),
            body: Value::Null,
            timeout: Duration::from_secs(1),
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
        // The budget caps how many run this turn.
        loop_.uses_left = 2;
        assert_eq!(loop_.pending().len(), 2);
        // A client tool sharing the turn still suppresses every search.
        loop_.filter.saw_other = true;
        assert!(loop_.pending().is_empty());
    }

    /// The upstream closes a tool turn with a bare `finish_reason` chunk that
    /// carries no delta. It has to be neutralised too, or the client is told
    /// the turn is over while the search is still running.
    #[test]
    fn bare_finish_reason_chunk_is_neutralised() {
        let mut f = SearchCallFilter::default();
        f.ours.insert(0, ("call_1".into(), "{}".into()));
        let out = rewrite_chunk_search(
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
        assert_eq!(rewrite_chunk_search(&data, &mut f), data);

        // So must an ordinary end-of-turn finish.
        f.saw_other = false;
        let data = json!({"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}).to_string();
        assert_eq!(rewrite_chunk_search(&data, &mut f), data);
    }

    /// The trailing usage-only chunk (`choices: []`) must pass through: it
    /// carries the token counts.
    #[test]
    fn usage_only_chunk_passes_through() {
        let mut f = SearchCallFilter::default();
        f.ours.insert(0, ("call_1".into(), "{}".into()));
        let data = json!({"choices": [], "usage": {"prompt_tokens": 5}}).to_string();
        assert_eq!(rewrite_chunk_search(&data, &mut f), data);
    }

    /// `max_uses` is a hard stop: a model that keeps searching runs out of
    /// budget and the turn closes instead of looping on pxy's search quota.
    #[test]
    fn exhausted_budget_stops_the_loop() {
        let mut filter = SearchCallFilter::default();
        filter.ours.insert(0, ("call_1".into(), "{\"query\":\"x\"}".into()));
        let mut loop_ = SearchLoop {
            filter,
            uses_left: 1,
            url: String::new(),
            headers: Vec::new(),
            body: Value::Null,
            timeout: Duration::from_secs(1),
        };
        assert!(!loop_.pending().is_empty());
        loop_.uses_left = 0;
        assert!(loop_.pending().is_empty());
    }

    /// A chunk with no tool calls at all comes back byte-identical.
    #[test]
    fn plain_chunks_pass_through_untouched() {
        let mut f = SearchCallFilter::default();
        let data = json!({"choices": [{"index": 0, "delta": {"content": "hi"}}]}).to_string();
        assert_eq!(rewrite_chunk_search(&data, &mut f), data);
        assert_eq!(rewrite_chunk_search("[DONE]", &mut f), "[DONE]");
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

    #[test]
    fn free_allowance_headers_are_remembered_and_cool_at_100() {
        let s = state("free_quota");
        record_free_allowance(
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

        // Spent: cool until the window actually rolls, and don't retry into it.
        record_free_allowance(
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
        record_free_allowance(&s, "other", &headers(&[("x-quota-5h", "3%")]));
        assert!(s.kv_get(&free_quota_key("other")).unwrap().is_none());
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
            assert_eq!(rewrite_chunk_search(data, &mut SearchCallFilter::default()), data);
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
        app.state.kv_set(ROUTE_PIN_KEY, "b/m2").unwrap();
        let ids: Vec<String> = resolve_candidates(&app.catalog, &app.cfg, &app.state, "free", None)
            .iter()
            .map(|c| c.full_id())
            .collect();
        assert_eq!(ids, ["b/m2", "a/m1"], "pin first, chain as fallback");

        // Explicit model requests ignore the pin entirely.
        let ids: Vec<String> = resolve_candidates(&app.catalog, &app.cfg, &app.state, "a/m1", None)
            .iter()
            .map(|c| c.full_id())
            .collect();
        assert_eq!(ids, ["a/m1"]);

        // A pin that stopped resolving degrades to the plain chain.
        app.state.kv_set(ROUTE_PIN_KEY, "gone/nope").unwrap();
        let ids: Vec<String> = resolve_candidates(&app.catalog, &app.cfg, &app.state, "free", None)
            .iter()
            .map(|c| c.full_id())
            .collect();
        assert_eq!(ids, ["a/m1", "b/m2"]);

        // A pin under a REAL provider but to an unlisted model does too:
        // resolve() fabricates a candidate for it, and without the is_listed
        // gate that phantom would lead every group walk (and a 400 "unknown
        // model" is Fatal — no failover).
        app.state.kv_set(ROUTE_PIN_KEY, "a/ghost").unwrap();
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
            [groups.free]
            models = ["a/m1", "b/m2"]
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
        app.state.kv_set(ROUTE_PIN_KEY, "a/m1").unwrap();
        let ids: Vec<String> =
            resolve_candidates(&app.catalog, &app.cfg, &app.state, "free", Some("uid:u1"))
                .iter()
                .map(|c| c.full_id())
                .collect();
        assert_eq!(ids, ["a/m1", "b/m2"], "pin first, affinity never leads");
        app.state.kv_set(ROUTE_PIN_KEY, "").unwrap();

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
            Outcome::Stream { provider, body } => {
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
            Outcome::Json { status, .. } => assert_eq!(status, 429, "synthetic exhaustion"),
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
            Outcome::Json { status, body, provider } => {
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
            Outcome::Stream { provider, body } => {
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
        // Over the sanity clamp or garbage: fall back to exponential backoff.
        assert_eq!(parse_retry_after(&headers(&[("retry-after", "2h")])), None);
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
            Outcome::Json { status, body, provider } => {
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
            "temperature": 0.5, "messages": [{"role": "user", "content": "hi"}]});
        let out = handle_chat(app, ClientFormat::Openai, payload, ClientContext::default()).await;
        match out {
            Outcome::Json { status, body, .. } => assert_eq!(status, 200, "got {body}"),
            Outcome::Stream { .. } => panic!("expected json"),
        }
        let body = seen.lock().unwrap().take().expect("upstream was called");
        assert!(body.get("reasoning_effort").is_none(), "provider-level drop must strip");
        assert!(body.get("top_k").is_none(), "model-level drop must strip");
        assert_eq!(body["temperature"], 0.5, "unlisted params must survive");
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
