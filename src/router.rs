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
use crate::translate::{anthropic_to_openai, estimate_tokens, openai_to_anthropic, TokenUsage};
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

    for cand in &candidates {
        if let Err(reason) = check_candidate(&app, cand, input_estimate, multi) {
            skipped.push(format!("{}: {reason}", cand.full_id()));
            continue;
        }

        match try_candidate(&app, cand, client_format, &payload, stream, input_estimate, &ctx)
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
        if let Some(cd) = app.state.cooldown(&cand.provider) {
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
    };
    body["model"] = json!(cand.model.id);
    if stream && upstream_format == WireFormat::Openai && body.get("stream_options").is_none() {
        // Ask OpenAI upstreams to report usage in the final chunk.
        body["stream_options"] = json!({"include_usage": true});
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
            app.state.set_cooldown(&cand.provider, None, "network error");
            return AttemptResult::Skip(format!("network: {e}"));
        }
    };

    let status = resp.status().as_u16();
    if status >= 400 {
        let retry_after = parse_retry_after(resp.headers());
        let err_body = resp.text().await.unwrap_or_default();
        return classify_error(app, cand, client_format, status, retry_after, err_body);
    }

    // Success: count the request now; tokens follow when usage is known.
    record_request(app, &cand.provider);
    app.state.clear_cooldown(&cand.provider);
    info!(candidate = %cand.full_id(), stream, "routed");

    if stream {
        AttemptResult::Done(stream_outcome(
            app.clone(),
            cand,
            client_format,
            upstream_format,
            resp,
            input_estimate,
        ))
    } else {
        let upstream_body: Value = match resp.json().await {
            Ok(v) => v,
            Err(e) => return AttemptResult::Skip(format!("bad upstream json: {e}")),
        };
        let usage = match upstream_format {
            WireFormat::Openai => TokenUsage::from_openai(&upstream_body["usage"]),
            WireFormat::Anthropic => TokenUsage::from_anthropic(&upstream_body["usage"]),
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
fn classify_error(
    app: &App,
    cand: &Candidate,
    client_format: ClientFormat,
    status: u16,
    retry_after: Option<Duration>,
    err_body: String,
) -> AttemptResult {
    let skip = matches!(status, 401 | 403 | 408 | 409 | 429) || status >= 500;
    if skip {
        let reason = match status {
            401 | 403 => "auth error",
            429 => "rate limited",
            _ => "upstream error",
        };
        app.state
            .set_cooldown(&cand.provider, retry_after, &format!("{status} {reason}"));
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
    kind: StreamKind,
    usage: TokenUsage,
    provider: String,
    app: SharedApp,
    upstream: futures_util::stream::BoxStream<'static, reqwest::Result<Bytes>>,
    done: bool,
}

impl StreamCtx {
    /// Process one upstream chunk; returns bytes for the client.
    fn process(&mut self, bytes: &Bytes) -> Bytes {
        let events = self.parser.feed(bytes);
        match &mut self.kind {
            StreamKind::OpenaiPass => {
                for ev in &events {
                    if ev.data.contains("\"usage\"") {
                        if let Ok(v) = serde_json::from_str::<Value>(&ev.data) {
                            if v["usage"].is_object() {
                                let u = TokenUsage::from_openai(&v["usage"]);
                                if u.input > 0 {
                                    self.usage.input = u.input;
                                }
                                if u.output > 0 {
                                    self.usage.output = u.output;
                                }
                            }
                        }
                    }
                }
                bytes.clone()
            }
            StreamKind::AnthropicPass => {
                for ev in &events {
                    if let Ok(v) = serde_json::from_str::<Value>(&ev.data) {
                        match v["type"].as_str() {
                            Some("message_start") => {
                                self.usage.input =
                                    v["message"]["usage"]["input_tokens"].as_u64().unwrap_or(0);
                            }
                            Some("message_delta") => {
                                if let Some(o) = v["usage"]["output_tokens"].as_u64() {
                                    self.usage.output = o;
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
                for ev in &events {
                    out.push_str(&state.on_data(&ev.data));
                }
                self.usage = state.usage;
                Bytes::from(out)
            }
            StreamKind::ToOpenai(state) => {
                let mut out = String::new();
                for ev in &events {
                    out.push_str(&state.on_event(ev.event.as_deref(), &ev.data));
                }
                self.usage = state.usage;
                Bytes::from(out)
            }
        }
    }

    /// End-of-stream flush (upstream may end without a terminal marker).
    fn finish(&mut self) -> Bytes {
        let out = match &mut self.kind {
            StreamKind::ToAnthropic(state) => {
                let s = state.finish();
                self.usage = state.usage;
                Bytes::from(s)
            }
            _ => Bytes::new(),
        };
        record_tokens(&self.app, &self.provider, self.usage);
        out
    }
}

fn stream_outcome(
    app: SharedApp,
    cand: &Candidate,
    client_format: ClientFormat,
    upstream_format: WireFormat,
    resp: reqwest::Response,
    input_estimate: u64,
) -> Outcome {
    let kind = match (client_format, upstream_format) {
        (ClientFormat::Openai, WireFormat::Openai) => StreamKind::OpenaiPass,
        (ClientFormat::Anthropic, WireFormat::Anthropic) => StreamKind::AnthropicPass,
        (ClientFormat::Anthropic, WireFormat::Openai) => StreamKind::ToAnthropic(
            anthropic_to_openai::StreamState::new(&cand.full_id(), input_estimate),
        ),
        (ClientFormat::Openai, WireFormat::Anthropic) => {
            StreamKind::ToOpenai(openai_to_anthropic::StreamState::new(&cand.full_id()))
        }
    };

    let ctx = StreamCtx {
        parser: SseParser::new(),
        kind,
        usage: TokenUsage::default(),
        provider: cand.provider.clone(),
        app,
        upstream: resp.bytes_stream().boxed(),
        done: false,
    };

    let stream = futures_util::stream::unfold(ctx, |mut ctx| async move {
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

    Outcome::Stream {
        provider: cand.full_id(),
        body: axum::body::Body::from_stream(stream),
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
