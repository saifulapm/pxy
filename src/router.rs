//! The auto-routing engine: candidate filtering, fallback walk, error
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
use crate::config::{Config, WireFormat};
use crate::secrets::Secrets;
use crate::state::State;
use crate::translate::sse::SseParser;
use crate::translate::think::ThinkFilter;
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

pub async fn handle_chat(
    app: SharedApp,
    client_format: ClientFormat,
    payload: Value,
    ctx: ClientContext,
) -> Outcome {
    let requested = payload["model"].as_str().unwrap_or("auto").to_string();
    let stream = payload["stream"].as_bool().unwrap_or(false);
    let candidates = app.catalog.resolve(&app.cfg, &requested);
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

    let mut skipped: Vec<String> = Vec::new();
    let multi = candidates.len() > 1;

    for attempt in 0..=MAX_RETRIES {
        skipped.clear();
        let mut saw_rpm_limit = false;
        for cand in &candidates {
            if let Err(reason) = check_candidate(&app, cand, input_estimate, multi) {
                saw_rpm_limit |= reason == "rpm limit";
                skipped.push(format!("{}: {reason}", cand.full_id()));
                continue;
            }

            match try_candidate(&app, cand, client_format, &payload, stream, input_estimate, &ctx, multi)
                .await
            {
                AttemptResult::Done(outcome) => return outcome,
                AttemptResult::Skip(reason) => {
                    warn!(candidate = %cand.full_id(), %reason, "failover");
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

/// Filter stage: cooldown, rpm, daily/monthly limits, context window.
fn check_candidate(
    app: &App,
    cand: &Candidate,
    input_estimate: u64,
    multi_candidate: bool,
) -> Result<(), String> {
    let provider = match app.cfg.providers.get(&cand.provider) {
        Some(p) if p.enabled => p,
        _ => return Err("provider disabled".into()),
    };

    // Single-candidate requests skip the cooldown filter (litellm's
    // single-deployment exemption): blocking your only option converts a
    // partial outage into a total one.
    if multi_candidate {
        if let Some(cd) = app.state.cooldown(&cand.provider, &cand.model.id) {
            return Err(format!("cooldown ({})", cd.reason));
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
            if app.state.rpm_effective(&cand.provider) >= rpm as f64 {
                return Err("rpm limit".into());
            }
        }
        // Limit checks fail open on infrastructure errors (litellm rule):
        // a broken tzdb/db must never block routing.
        if let Ok(w) = current_windows(limits, Timestamp::now()) {
            let day = app.state.usage(&cand.provider, "day", w.day_start).unwrap_or_default();
            let month = app
                .state
                .usage(&cand.provider, "month", w.month_start)
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
            let total = app.state.usage_total(&cand.provider).unwrap_or_default();
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
        (ClientFormat::Openai, WireFormat::Openai)
        | (ClientFormat::Anthropic, WireFormat::Anthropic) => payload.clone(),
        (ClientFormat::Anthropic, WireFormat::Openai) => anthropic_to_openai::request(payload),
        (ClientFormat::Openai, WireFormat::Anthropic) => {
            openai_to_anthropic::request(payload, cand.model.max_output_tokens)
        }
        // Kiro takes neither dialect: normalize to Anthropic first (reusing
        // the existing translator), then build conversationState from it.
        (ClientFormat::Anthropic, WireFormat::Kiro) => kiro::request(
            payload,
            &cand.model.id,
            "",
            &now_iso8601(),
        ),
        (ClientFormat::Openai, WireFormat::Kiro) => kiro::request(
            &openai_to_anthropic::request(payload, cand.model.max_output_tokens),
            &cand.model.id,
            "",
            &now_iso8601(),
        ),
    };
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
    if (!provider_cfg.drop_params.is_empty() || !cand.model.drop_params.is_empty())
        && let Some(obj) = body.as_object_mut()
    {
        for key in provider_cfg.drop_params.iter().chain(&cand.model.drop_params) {
            if key == "model" || key == "stream" {
                warn!(candidate = %cand.full_id(), key, "drop_params ignores pxy's own key");
                continue;
            }
            obj.remove(key);
        }
    }

    let prepared = match crate::providers::prepare(
        &cand.provider,
        provider_cfg,
        &app.secrets,
        &app.state,
        &app.http,
    )
    .await
    {
        Ok(p) => p,
        Err(e) => return AttemptResult::Skip(format!("prepare failed: {e:#}")),
    };

    // Providers may need fields inside the body (kiro's profileArn), which is
    // only known after credentials resolve.
    if let Some(patch) = &prepared.body_patch {
        if let (Some(dst), Some(src)) = (body.as_object_mut(), patch.as_object()) {
            for (k, v) in src {
                dst.insert(k.clone(), v.clone());
            }
        }
    }

    let initiator_override = ctx
        .initiator
        .as_deref()
        .filter(|v| *v == "agent" || *v == "user");

    let mut req = app
        .http
        .post(&prepared.url)
        .timeout(Duration::from_secs(provider_cfg.timeout_secs))
        .header("content-type", "application/json");
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

    app.state.rpm_increment(&cand.provider);
    let resp = match req.json(&body).send().await {
        Ok(r) => r,
        Err(e) => {
            // Network failures are our-side/transport, not model-specific.
            app.state.set_cooldown(&cand.provider, None, None, true, "network error");
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
    record_request(app, &cand.provider);
    app.state.clear_cooldown(&cand.provider, &cand.model.id);
    // After the clear: a success response can still carry "you just used the
    // last of your quota" headers, and that cooldown must survive it.
    check_quota_exhaustion(&app.state, &cand.provider, resp.headers());
    if !stream {
        info!(candidate = %cand.full_id(), stream, "routed");
    }

    if stream {
        // A 200 status is not a commitment yet: hold the response until the
        // upstream produces a real first event, so a stream that dies before
        // saying anything fails over instead of reaching the client truncated.
        match stream_outcome(app.clone(), cand, client_format, upstream_format, resp, input_estimate)
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
                            &cand.provider,
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
                    &cand.provider,
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
            Err(e) => return AttemptResult::Skip(format!("kiro body read: {e}")),
        };
        let (openai_body, usage) =
            kiro::collect_response(&bytes, &cand.model.id, &cand.full_id(), cand.model.context_length);
        record_tokens(app, &cand.provider, usage);
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
                Err(e) => return AttemptResult::Skip(format!("stream read: {e}")),
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
                Err(e) => return AttemptResult::Skip(format!("bad upstream json: {e}")),
            }
        };
        if provider_cfg.parse_think_tags && upstream_format == WireFormat::Openai {
            extract_think_from_response(&mut upstream_body);
        }
        let usage = match upstream_format {
            WireFormat::Openai => TokenUsage::from_openai(&upstream_body["usage"]),
            WireFormat::Anthropic => TokenUsage::from_anthropic(&upstream_body["usage"]),
            WireFormat::Kiro => unreachable!("kiro handled above"),
        };
        record_tokens(app, &cand.provider, usage);
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
/// Exception: on a multi-candidate walk (`auto`), a 404 also skips. Free
/// model lists churn, and one delisted id must not kill the whole chain
/// (zenmux delisting glm-5.3-free took `auto` down, 2026-08-25). On a
/// single-candidate request the 404 still passes through raw.
fn classify_error(
    app: &App,
    cand: &Candidate,
    client_format: ClientFormat,
    status: u16,
    retry_after: Option<Duration>,
    err_body: String,
    multi: bool,
) -> AttemptResult {
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
        app.state.set_cooldown(
            &cand.provider,
            model_scope,
            retry_after,
            retryable,
            &format!("{status} {reason}"),
        );
        return AttemptResult::Skip(format!("{status}: {}", truncate(&err_body, 200)));
    }
    let body = serde_json::from_str::<Value>(&err_body)
        .unwrap_or_else(|_| error_body(client_format, "api_error", &truncate(&err_body, 500)));
    AttemptResult::Fatal(Outcome::Json {
        status,
        body,
        provider: Some(cand.full_id()),
    })
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
    let v = headers.get("retry-after")?.to_str().ok()?;
    let secs: u64 = v.trim().parse().ok()?;
    // Sanity clamp (litellm): obey only reasonable waits; else exponential backoff.
    if secs > 0 && secs <= 3600 {
        Some(Duration::from_secs(secs))
    } else {
        None
    }
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
    usage: TokenUsage,
    provider: String,
    app: SharedApp,
    upstream: futures_util::stream::BoxStream<'static, reqwest::Result<Bytes>>,
    done: bool,
}

/// Move `<think>` spans in an openai chunk's delta.content into
/// delta.reasoning_content. Returns the rewritten chunk JSON, or the input
/// unchanged when it isn't a parseable chunk.
fn rewrite_chunk_think(data: &str, filter: &mut ThinkFilter) -> String {
    let Ok(mut v) = serde_json::from_str::<Value>(data) else {
        return data.to_string();
    };
    let delta = &mut v["choices"][0]["delta"];
    if let Some(text) = delta["content"].as_str() {
        let (reasoning, content) = filter.push(text);
        if !reasoning.is_empty() {
            let prior = delta["reasoning_content"].as_str().unwrap_or("");
            delta["reasoning_content"] = json!(format!("{prior}{reasoning}"));
        }
        delta["content"] = json!(content);
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
        record_tokens(&self.app, &self.provider, self.usage);
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
        let Self { parser, kind, think, usage, .. } = self;
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
                let Some(filter) = think else {
                    return bytes.clone();
                };
                // Think extraction forces chunk rewriting even in passthrough.
                let mut out = String::new();
                for ev in events {
                    if ev.data.trim() == "[DONE]" {
                        if let Some(tail) = think_flush_chunk(filter) {
                            out.push_str(&format!("data: {tail}\n\n"));
                        }
                        out.push_str("data: [DONE]\n\n");
                    } else {
                        let data = rewrite_chunk_think(&ev.data, filter);
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
                                    v["message"]["usage"]["input_tokens"].as_u64().unwrap_or(0);
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
                    match think {
                        Some(filter) if ev.data.trim() != "[DONE]" => {
                            // reasoning moved into delta.reasoning_content
                            // becomes a thinking block via on_data
                            out.push_str(&state.on_data(&rewrite_chunk_think(&ev.data, filter)));
                        }
                        Some(filter) => {
                            if let Some(tail) = think_flush_chunk(filter) {
                                out.push_str(&state.on_data(&tail));
                            }
                            out.push_str(&state.on_data(&ev.data));
                        }
                        None => out.push_str(&state.on_data(&ev.data)),
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
            record_tokens(&self.app, &self.provider, self.usage);
            return translated;
        }
        let out = match &mut self.kind {
            StreamKind::ToAnthropic(state) => {
                let mut s = String::new();
                if let Some(filter) = &mut self.think {
                    if let Some(tail) = think_flush_chunk(filter) {
                        s.push_str(&state.on_data(&tail));
                    }
                }
                s.push_str(&state.finish());
                self.usage = state.usage;
                Bytes::from(s)
            }
            _ => Bytes::new(),
        };
        record_tokens(&self.app, &self.provider, self.usage);
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

/// Returns Err when the stream died before producing a first event — the
/// caller treats that as a failed attempt and walks on. Nothing has been
/// sent to the client at that point, so failover is invisible.
async fn stream_outcome(
    app: SharedApp,
    cand: &Candidate,
    client_format: ClientFormat,
    upstream_format: WireFormat,
    resp: reqwest::Response,
    input_estimate: u64,
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
        usage: TokenUsage::default(),
        provider: cand.provider.clone(),
        app,
        upstream: resp.bytes_stream().boxed(),
        done: false,
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
        match ctx.upstream.next().await {
            Some(Ok(bytes)) => {
                let out = ctx.process(&bytes);
                Some((Ok::<Bytes, std::io::Error>(out), ctx))
            }
            Some(Err(e)) => {
                warn!(provider = %ctx.provider, error = %e, "upstream stream error");
                ctx.done = true;
                let tail = ctx.finish();
                Some((Ok(tail), ctx))
            }
            None => {
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

fn record_request(app: &App, provider: &str) {
    record_usage_inner(app, provider, TokenUsage::default(), true);
}

fn record_tokens(app: &App, provider: &str, usage: TokenUsage) {
    if usage.input == 0 && usage.output == 0 {
        return;
    }
    record_usage_inner(app, provider, usage, false);
}

/// Embeddings count as one request + input tokens against the same windows.
pub fn record_embedding_usage(app: &App, provider: &str, tokens: u64) {
    record_usage_inner(app, provider, TokenUsage { input: tokens, output: 0 }, true);
}

fn record_usage_inner(app: &App, provider: &str, usage: TokenUsage, request: bool) {
    let default_limits = crate::config::Limits::default();
    let limits = app
        .cfg
        .providers
        .get(provider)
        .and_then(|p| p.limits.as_ref())
        .unwrap_or(&default_limits);
    if let Ok(w) = current_windows(limits, Timestamp::now()) {
        let res = if request {
            app.state
                .record_usage(provider, w.day_start, w.month_start, usage.input, usage.output)
        } else {
            app.state
                .add_tokens(provider, w.day_start, w.month_start, usage.input, usage.output)
        };
        if let Err(e) = res {
            warn!(provider, error = %e, "usage recording failed");
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
                [auto]
                models = ["a/m", "b/m"]
                "#
            ),
            "dead_stream",
        );

        let payload = json!({"model": "auto", "stream": true,
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
