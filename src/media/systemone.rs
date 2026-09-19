//! `/v1/systemone` — Typesafe's System One model (Jev) in Typesafe's own
//! shape: `{state, model, questions}` in, `{model, answers, usage}` out. Jev
//! is not OpenAI-compatible and not on OpenRouter's chat surface, so this
//! module speaks its wire directly and maps the two gateways that dress it up:
//! Cloudflare wraps the call in `{model, input}` and the answer in a
//! `{result, success, errors}` envelope, and Vercel translates it into the AI
//! SDK's evaluation protocol. `[media] systemone` is the failover chain across
//! them, so no Typesafe key is needed.

use axum::Json;
use axum::extract::State;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use serde_json::{Value, json};

use super::{Attempt, Capability, error_response};
use crate::config::MediaKind;
use crate::router::SharedApp;

pub async fn systemone(State(app): State<SharedApp>, Json(payload): Json<Value>) -> Response {
    let requested = payload["model"].as_str().unwrap_or("").to_string();
    super::run_chain(&app, Capability::SystemOne, &requested, |r| {
        Box::pin(attempt(&app, r, &payload))
    })
    .await
}

/// The same walk without the HTTP layer: `verify` and the web_search reranker
/// ask Jev through here, so all three share one failover and one
/// `provider#media` quota. Returns the canonical body (`model`, `answers`,
/// `usage`); a chain that is empty or wholly down is the `Err` string.
pub async fn ask(
    app: &SharedApp,
    requested: &str,
    state: Value,
    questions: Value,
) -> Result<Value, String> {
    let payload = json!({"state": state, "questions": questions});
    let resp = super::run_chain(app, Capability::SystemOne, requested, |r| {
        Box::pin(attempt(app, r, &payload))
    })
    .await;
    let status = resp.status();
    let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX)
        .await
        .map_err(|e| format!("reading systemone response: {e}"))?;
    let body: Value = serde_json::from_slice(&bytes)
        .map_err(|e| format!("bad systemone response: {e}"))?;
    if !status.is_success() {
        let msg = body["error"]["message"].as_str().unwrap_or("systemone failed");
        return Err(msg.to_string());
    }
    Ok(body)
}

async fn attempt(app: &crate::router::App, r: &super::Resolved<'_>, payload: &Value) -> Attempt {
    let state = payload.get("state").cloned().unwrap_or(Value::Null);
    let questions = payload.get("questions").cloned().unwrap_or_else(|| json!({}));
    let body = match r.media.kind {
        MediaKind::Vercel => json!({"state": state, "questions": vercel_questions(&questions)}),
        MediaKind::Cloudflare => {
            json!({"model": r.model, "input": {"state": state, "questions": questions}})
        }
        _ => {
            let mut b = payload.clone();
            b["model"] = json!(r.model);
            b
        }
    };

    let headers = match super::auth_headers(app, r.cfg) {
        Ok(h) => h,
        Err(e) => {
            return Attempt::Retryable(error_response(StatusCode::BAD_GATEWAY, format!("{e:#}")));
        }
    };
    let mut req = app
        .http
        .post(&r.url)
        .timeout(std::time::Duration::from_secs(r.cfg.timeout_secs))
        .header("content-type", "application/json");
    for (k, v) in &headers {
        req = req.header(k, v);
    }
    // Vercel's evaluation endpoint takes the model out of band; the body it
    // accepts has no `model` key at all. The protocol version is not optional:
    // without it, or with any other value tried, the gateway answers 400
    // "Unsupported gateway protocol version" (measured 2026-09-19).
    if r.media.kind == MediaKind::Vercel {
        req = req
            .header("ai-model-id", &r.model)
            .header("ai-gateway-protocol-version", "0.0.1");
    }
    let resp = match req.json(&body).send().await {
        Ok(resp) => resp,
        Err(e) => return super::network_failure(app, r, e),
    };
    if resp.status().as_u16() >= 400 {
        return super::failed_attempt(app, r, resp).await;
    }
    let upstream: Value = match resp.json().await {
        Ok(v) => v,
        Err(e) => {
            return Attempt::Retryable(error_response(
                StatusCode::BAD_GATEWAY,
                format!("bad upstream json: {e}"),
            ));
        }
    };

    let out = match r.media.kind {
        MediaKind::Vercel => vercel_response(&upstream, &questions, &r.model),
        // Workers AI can answer 200 with `success: false` for an account-level
        // failure, so the envelope, not the status, decides whether this
        // candidate answered. An empty balance sends `result: {}` beside the
        // error, so "answered" means `success` AND a result with something in
        // it — a bare `{}` passed on would be an answer with no answers.
        MediaKind::Cloudflare => {
            let result = upstream.get("result").filter(|r| {
                r.as_object().is_some_and(|o| !o.is_empty()) && upstream["success"] != false
            });
            match result {
                Some(result) => result.clone(),
                None => {
                    let msg = upstream["errors"][0]["message"]
                        .as_str()
                        .unwrap_or("cloudflare returned no result");
                    return Attempt::Retryable(error_response(
                        StatusCode::BAD_GATEWAY,
                        format!("systemone: {msg}"),
                    ));
                }
            }
        }
        _ => upstream,
    };
    let tokens = out["usage"]["input_tokens"].as_u64().unwrap_or(0);
    super::add_units(app, &super::media_key(&r.provider), tokens);
    app.state.clear_cooldown(&super::media_key(&r.provider), &r.model);

    let mut resp = (StatusCode::OK, Json(out)).into_response();
    if let Ok(v) = format!("{}/{}", r.provider, r.model).parse() {
        resp.headers_mut().insert("x-pxy-provider", v);
    }
    Attempt::Ok(resp)
}

/// The AI SDK gateway rejects `noul` with an invalid-discriminator error; the
/// same question typed `boolean` is the one Jev answers. `choice` and `score`
/// go through as they are.
fn vercel_questions(questions: &Value) -> Value {
    let mut out = questions.clone();
    if let Some(map) = out.as_object_mut() {
        for q in map.values_mut() {
            if q["type"] == "noul" {
                q["type"] = json!("boolean");
            }
        }
    }
    out
}

/// Rewrite one AI SDK evaluation answer into Typesafe's shape: `boolean`
/// becomes `noul`, confidence moves in from `providerMetadata`, and a score's
/// `legend` is rebuilt from the criteria the request carried (the gateway
/// drops it).
fn vercel_response(upstream: &Value, questions: &Value, model: &str) -> Value {
    let confidences = &upstream["providerMetadata"]["typesafe"]["confidence"];
    let mut answers = json!({});
    if let Some(map) = upstream["answers"].as_object() {
        for (key, answer) in map {
            let mut a = answer.clone();
            if a["type"] == "boolean" {
                a = json!({"type": "noul", "noul": answer["probability"]});
            } else {
                if let Some(c) = confidences.get(key) {
                    a["confidence"] = c.clone();
                }
                if a["type"] == "score"
                    && let Some(criteria) = questions[key]["criteria"].as_array()
                {
                    let legend: serde_json::Map<String, Value> = criteria
                        .iter()
                        .enumerate()
                        .map(|(i, level)| (i.to_string(), level.clone()))
                        .collect();
                    a["legend"] = Value::Object(legend);
                }
            }
            answers[key] = a;
        }
    }
    json!({
        "model": model,
        "answers": answers,
        "usage": {
            "input_tokens": upstream["usage"]["inputTokens"].as_u64().unwrap_or(0),
            "output_tokens": upstream["usage"]["outputTokens"].as_u64().unwrap_or(0),
        },
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::http::HeaderMap;
    use std::sync::{Arc, Mutex};

    /// The three questions every test asks, one of each type. The score's
    /// criteria are the ordered levels a `legend` is rebuilt from.
    fn questions() -> Value {
        json!({
            "is_urgent": {"type": "noul", "instructions": "Is this urgent?",
                "criteria": {"true": "needs action today", "false": "can wait"}},
            "department": {"type": "choice", "instructions": "Which desk?",
                "criteria": {"billing": "money", "technical": "the product"}},
            "frustration": {"type": "score", "instructions": "How angry?",
                "criteria": ["Calm, just stating facts", "Frustrated but civil",
                             "Very angry, strong language"]},
        })
    }

    /// Vercel sends the id in a header, types a noul question `boolean`, and
    /// answers in the AI SDK's own spelling; pxy has to undo all three.
    #[tokio::test]
    async fn a_vercel_candidate_speaks_the_evaluation_protocol() {
        let seen = Sink::default();
        let addr = upstream(seen.clone()).await;
        let cfg = format!(
            r#"
            [server]
            [providers.vercel]
            [providers.vercel.media]
            kind = "vercel"
            systemone_url = "http://{addr}/vercel"
            systemone_models = ["typesafe-ai/jev"]
            [media]
            systemone = ["vercel/typesafe-ai/jev"]
            "#
        );
        let app = mock_app(&cfg, "vercel");

        let resp = systemone(
            State(app.clone()),
            Json(json!({"model": "auto", "state": "the PDF export hangs", "questions": questions()})),
        )
        .await;

        assert_eq!(resp.status(), StatusCode::OK);
        assert_eq!(resp.headers().get("x-pxy-provider").unwrap(), "vercel/typesafe-ai/jev");

        let (sent, headers) = seen.one();
        assert!(sent.get("model").is_none(), "the id rides the header, not the body: {sent}");
        assert_eq!(headers.get("ai-model-id").unwrap(), "typesafe-ai/jev");
        assert_eq!(
            headers.get("ai-gateway-protocol-version").unwrap(),
            "0.0.1",
            "without it the gateway answers 400 Unsupported gateway protocol version"
        );
        assert_eq!(sent["questions"]["is_urgent"]["type"], "boolean", "noul is rejected as-is");
        assert_eq!(sent["questions"]["is_urgent"]["criteria"]["true"], "needs action today");
        assert_eq!(sent["questions"]["department"]["type"], "choice", "choice is untouched");
        assert_eq!(sent["questions"]["frustration"]["type"], "score", "score is untouched");

        let body = read(resp).await;
        assert_eq!(body["model"], "typesafe-ai/jev");
        assert_eq!(body["answers"]["is_urgent"], json!({"type": "noul", "noul": 0.999}));
        assert_eq!(body["answers"]["department"]["confidence"], 0.596, "confidence moves in");
        assert_eq!(body["answers"]["department"]["choice"], "technical");
        assert_eq!(
            body["answers"]["frustration"]["legend"],
            json!({"0": "Calm, just stating facts", "1": "Frustrated but civil",
                   "2": "Very angry, strong language"}),
            "the legend is rebuilt from the criteria the request carried"
        );
        assert_eq!(body["answers"]["frustration"]["confidence"], 0.842);
        assert_eq!(body["usage"], json!({"input_tokens": 312, "output_tokens": 48}));
        assert_eq!(app.state.usage_total(&super::super::media_key("vercel")).unwrap().tokens, 312);
    }

    /// Cloudflare wraps the call in `input` and the answer in `result`.
    #[tokio::test]
    async fn a_cloudflare_candidate_is_wrapped_in_input_and_unwrapped_from_result() {
        let seen = Sink::default();
        let addr = upstream(seen.clone()).await;
        let cfg = format!(
            r#"
            [server]
            [providers.cloudflare]
            [providers.cloudflare.media]
            kind = "cloudflare"
            systemone_url = "http://{addr}/cloudflare"
            systemone_models = ["typesafe/jev"]
            [media]
            systemone = ["cloudflare/typesafe/jev"]
            "#
        );
        let app = mock_app(&cfg, "cloudflare");

        let resp = systemone(
            State(app.clone()),
            Json(json!({"model": "auto", "state": "s", "questions": questions()})),
        )
        .await;

        assert_eq!(resp.status(), StatusCode::OK);
        assert_eq!(resp.headers().get("x-pxy-provider").unwrap(), "cloudflare/typesafe/jev");
        let (sent, _) = seen.one();
        assert_eq!(sent["model"], "typesafe/jev");
        assert_eq!(sent["input"]["state"], "s");
        assert_eq!(sent["input"]["questions"]["is_urgent"]["type"], "noul", "canonical on the wire");

        let body = read(resp).await;
        assert_eq!(body["answers"]["is_urgent"]["noul"], 0.999, "unwrapped out of `result`");
        assert_eq!(body["usage"]["input_tokens"], 312);
    }

    /// An `openai`-kind gateway takes and answers Typesafe's own shape; only
    /// `model` is rewritten, and the extra keys OpenRouter adds survive.
    #[tokio::test]
    async fn an_openai_candidate_passes_through() {
        let seen = Sink::default();
        let addr = upstream(seen.clone()).await;
        let app = mock_app(&openai_cfg(&addr, "/openai"), "openai");

        let body = ask(&app, "auto", json!("s"), questions()).await.unwrap();

        let (sent, _) = seen.one();
        assert_eq!(sent["model"], "typesafe/jev-1.13", "the candidate's id replaces the request's");
        assert_eq!(sent["questions"]["is_urgent"]["type"], "noul");
        assert_eq!(body["model"], "typesafe/jev-1.13-20260917");
        assert_eq!(body["answers"]["is_urgent"]["noul"], 0.999);
        assert_eq!(body["usage"]["cost"], 0.0000144, "OpenRouter's extra keys pass through");
    }

    /// 429 is a provider-side failure: cool the candidate and walk on.
    #[tokio::test]
    async fn a_429_walks_on_to_the_next_candidate() {
        let seen = Sink::default();
        let addr = upstream(seen.clone()).await;
        let cfg = format!(
            r#"
            [server]
            [providers.busy]
            [providers.busy.media]
            systemone_url = "http://{addr}/429"
            systemone_models = ["typesafe/jev-1.13"]
            [providers.openrouter]
            [providers.openrouter.media]
            systemone_url = "http://{addr}/openai"
            systemone_models = ["typesafe/jev-1.13"]
            [media]
            systemone = ["busy/typesafe/jev-1.13", "openrouter/typesafe/jev-1.13"]
            "#
        );
        let app = mock_app(&cfg, "429");

        let resp = systemone(
            State(app.clone()),
            Json(json!({"model": "auto", "state": "s", "questions": questions()})),
        )
        .await;

        assert_eq!(resp.status(), StatusCode::OK);
        assert_eq!(resp.headers().get("x-pxy-provider").unwrap(), "openrouter/typesafe/jev-1.13");
        assert!(
            app.state.cooldown(&super::super::media_key("busy"), "typesafe/jev-1.13").is_some(),
            "the rate-limited candidate cools down"
        );
    }

    /// Workers AI answers 200 with `success: false`; the envelope, not the
    /// status, has to fail the candidate.
    #[tokio::test]
    async fn a_cloudflare_error_envelope_fails_the_candidate() {
        let seen = Sink::default();
        let addr = upstream(seen.clone()).await;
        let cfg = format!(
            r#"
            [server]
            [providers.cloudflare]
            [providers.cloudflare.media]
            kind = "cloudflare"
            systemone_url = "http://{addr}/cf-broke"
            systemone_models = ["typesafe/jev"]
            [media]
            systemone = ["cloudflare/typesafe/jev"]
            "#
        );
        let app = mock_app(&cfg, "cf-broke");

        let err = ask(&app, "auto", json!("s"), questions()).await.unwrap_err();
        assert!(err.contains("Insufficient balance"), "{err}");
    }

    /// No `[media] systemone` chain anywhere: nothing to walk.
    #[tokio::test]
    async fn an_empty_chain_is_an_error_not_a_call() {
        let app = mock_app("[server]\n", "empty");
        let err = ask(&app, "auto", json!("s"), questions()).await.unwrap_err();
        assert!(err.contains("systemone"), "{err}");
    }

    type Sink = Arc<Mutex<Vec<(Value, HeaderMap)>>>;

    trait One {
        fn one(&self) -> (Value, HeaderMap);
    }
    impl One for Sink {
        fn one(&self) -> (Value, HeaderMap) {
            let calls = self.lock().unwrap();
            assert_eq!(calls.len(), 1, "exactly one upstream call");
            calls[0].clone()
        }
    }

    fn openai_cfg(addr: &str, path: &str) -> String {
        format!(
            r#"
            [server]
            [providers.openrouter]
            [providers.openrouter.media]
            systemone_url = "http://{addr}{path}"
            systemone_models = ["typesafe/jev-1.13"]
            [media]
            systemone = ["openrouter/typesafe/jev-1.13"]
            "#
        )
    }

    async fn read(resp: Response) -> Value {
        let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX).await.unwrap();
        serde_json::from_slice(&bytes).unwrap()
    }

    /// One server with a route per gateway dialect, each recording what it was
    /// sent. The answers are the bodies the wiki recorded live on 2026-09-19.
    async fn upstream(sink: Sink) -> String {
        use axum::routing::post;
        let canonical = json!({
            "model": "typesafe/jev-1.13-20260917",
            "answers": {
                "is_urgent": {"type": "noul", "noul": 0.999},
                "department": {"type": "choice", "choice": "technical",
                    "probabilities": {"billing": 0.159, "technical": 0.84}, "confidence": 0.596},
                "frustration": {"type": "score", "score": 1.035,
                    "legend": {"0": "a", "1": "b", "2": "c"}, "confidence": 0.842},
            },
            "usage": {"input_tokens": 312, "output_tokens": 48, "cost": 0.0000144},
        });
        let vercel = json!({
            "answers": {
                "is_urgent": {"type": "boolean", "probability": 0.999},
                "department": {"type": "choice", "choice": "technical",
                    "probabilities": {"billing": 0.159, "technical": 0.84}},
                "frustration": {"type": "score", "score": 1.035,
                    "probabilities": {"0": 0.2, "1": 0.7, "2": 0.1}},
            },
            "providerMetadata": {"typesafe": {"confidence":
                {"department": 0.596, "frustration": 0.842}}},
            "usage": {"inputTokens": 312, "outputTokens": 48},
        });
        let record = move |sink: Sink, answer: Value| {
            post(move |headers: HeaderMap, axum::Json(body): axum::Json<Value>| {
                let (sink, answer) = (sink.clone(), answer.clone());
                async move {
                    sink.lock().unwrap().push((body, headers));
                    axum::Json(answer)
                }
            })
        };
        let router = axum::Router::new()
            .route("/vercel", record(sink.clone(), vercel))
            .route(
                "/cloudflare",
                record(sink.clone(), json!({"result": canonical, "success": true, "errors": []})),
            )
            .route(
                "/cf-broke",
                record(
                    sink.clone(),
                    // The body the account really sent, `result` an empty
                    // object rather than null (measured 2026-09-19).
                    json!({"result": {}, "success": false, "messages": [], "errors": [
                        {"code": 2021, "message": "Insufficient balance; add money to your gateway or use BYOK"}
                    ]}),
                ),
            )
            .route("/openai", record(sink.clone(), canonical.clone()))
            .route("/429", post(|| async { (StatusCode::TOO_MANY_REQUESTS, "slow down") }));
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move { axum::serve(listener, router).await.unwrap() });
        addr.to_string()
    }

    fn mock_app(cfg_toml: &str, name: &str) -> SharedApp {
        let cfg: crate::config::Config = toml::from_str(cfg_toml).unwrap();
        let catalog = crate::catalog::Catalog::from_config(&cfg);
        let dir = std::env::temp_dir().join(format!("pxy-s1-test-{}-{name}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        Arc::new(crate::router::App {
            catalog,
            secrets: crate::secrets::Secrets::new(),
            state: crate::state::State::open(&dir.join("s.sqlite")).unwrap(),
            http: reqwest::Client::new(),
            cfg,
        })
    }
}
