//! `/v1/search` and `/v1/fetch` — web search and URL->markdown, the two
//! non-model services. Providers are walked in config order (first healthy
//! with quota headroom wins); every dialect normalizes into one result shape:
//! search -> `{provider, query, results: [{title, url, snippet}]}`,
//! fetch  -> `{provider, url, content}`.

use axum::Json;
use axum::extract::{Query, State};
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use serde_json::{Value, json};

use super::error_response;
use crate::config::{ServiceKind, ServiceProvider};
use crate::router::{App, SharedApp};

const BRAVE_URL: &str = "https://api.search.brave.com/res/v1/web/search";
const JINA_SEARCH_URL: &str = "https://s.jina.ai/";
const JINA_READER_URL: &str = "https://r.jina.ai/";
const FIRECRAWL_SEARCH_URL: &str = "https://api.firecrawl.dev/v2/search";
const FIRECRAWL_SCRAPE_URL: &str = "https://api.firecrawl.dev/v2/scrape";

/// Quota + enabled gate; returns the usage key when the provider is usable.
fn service_ready(app: &App, scope: &str, p: &ServiceProvider) -> Option<String> {
    if !p.enabled {
        return None;
    }
    let key = format!("{scope}#{}", p.name);
    if app.state.cooldown(&key, "").is_some() {
        return None;
    }
    if let Some(cap) = p.daily_requests {
        if super::used_requests(app, &key, "day") >= cap {
            return None;
        }
    }
    if let Some(cap) = p.monthly_requests {
        if super::used_requests(app, &key, "month") >= cap {
            return None;
        }
    }
    Some(key)
}

pub async fn search(State(app): State<SharedApp>, Json(payload): Json<Value>) -> Response {
    let Some(query) = payload["query"].as_str().filter(|q| !q.is_empty()) else {
        return error_response(StatusCode::BAD_REQUEST, "'query' is required");
    };
    let n = payload["max_results"].as_u64().unwrap_or(5).clamp(1, 20);
    let only = payload["provider"].as_str();

    let mut errors: Vec<String> = Vec::new();
    // A provider that answered 200 with zero hits is remembered but not
    // punished: another provider may still find something, and if none does
    // the query legitimately has no results.
    let mut empty_from: Option<String> = None;
    for p in &app.cfg.search.providers {
        if only.is_some_and(|o| o != p.name) {
            continue;
        }
        let Some(key) = service_ready(&app, "search", p) else { continue };
        // Count the query up front: an upstream 200 consumed quota even if
        // we fail to read the body.
        super::record(&app, &key);
        match search_one(&app, p, query, n).await {
            Ok(results) => {
                app.state.clear_cooldown(&key, "");
                if results.is_empty() {
                    empty_from.get_or_insert_with(|| p.name.clone());
                    continue;
                }
                return (
                    StatusCode::OK,
                    Json(json!({"provider": p.name, "query": query, "results": results})),
                )
                    .into_response();
            }
            Err(e) => {
                app.state.set_cooldown(&key, None, None, true, &format!("{e:#}"));
                errors.push(format!("{}: {e:#}", p.name));
            }
        }
    }
    if let Some(provider) = empty_from {
        return (
            StatusCode::OK,
            Json(json!({"provider": provider, "query": query, "results": []})),
        )
            .into_response();
    }
    error_response(
        StatusCode::BAD_GATEWAY,
        if errors.is_empty() { "no search provider available".into() } else { errors.join("; ") },
    )
}

async fn search_one(
    app: &App,
    p: &ServiceProvider,
    query: &str,
    n: u64,
) -> anyhow::Result<Vec<Value>> {
    let api_key = app.secrets.resolve_key(&p.api_key)?;
    let timeout = std::time::Duration::from_secs(20);
    let body: Value = match p.kind {
        ServiceKind::Brave => {
            let resp = app
                .http
                .get(format!("{BRAVE_URL}?q={}&count={n}", urlencode(query)))
                .timeout(timeout)
                .header("accept", "application/json")
                .header("x-subscription-token", api_key)
                .send()
                .await?;
            anyhow::ensure!(resp.status().is_success(), "http {}", resp.status());
            resp.json().await?
        }
        ServiceKind::Jina => {
            let resp = app
                .http
                .post(JINA_SEARCH_URL)
                .timeout(timeout)
                .header("accept", "application/json")
                .bearer_auth(api_key)
                .json(&json!({"q": query, "num": n}))
                .send()
                .await?;
            anyhow::ensure!(resp.status().is_success(), "http {}", resp.status());
            resp.json().await?
        }
        ServiceKind::FirecrawlSearch => {
            let resp = app
                .http
                .post(FIRECRAWL_SEARCH_URL)
                .timeout(timeout)
                .bearer_auth(api_key)
                .json(&json!({"query": query, "limit": n}))
                .send()
                .await?;
            anyhow::ensure!(resp.status().is_success(), "http {}", resp.status());
            resp.json().await?
        }
        _ => anyhow::bail!("provider kind is not a search kind"),
    };
    Ok(normalize_search(p.kind, &body))
}

/// Query-string percent-encoding (RFC 3986 unreserved set kept).
fn urlencode(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for b in s.bytes() {
        match b {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' => {
                out.push(b as char)
            }
            _ => out.push_str(&format!("%{b:02X}")),
        }
    }
    out
}

/// Collapse the three response dialects into `[{title, url, snippet}]`.
pub fn normalize_search(kind: ServiceKind, body: &Value) -> Vec<Value> {
    let items: &[Value] = match kind {
        ServiceKind::Brave => body["web"]["results"].as_array().map(|v| &v[..]).unwrap_or(&[]),
        ServiceKind::Jina => body["data"].as_array().map(|v| &v[..]).unwrap_or(&[]),
        ServiceKind::FirecrawlSearch => {
            body["data"]["web"].as_array().map(|v| &v[..]).unwrap_or(&[])
        }
        _ => &[],
    };
    items
        .iter()
        .filter_map(|item| {
            let url = item["url"].as_str().or(item["link"].as_str())?;
            Some(json!({
                "title": item["title"].as_str().unwrap_or(url),
                "url": url,
                "snippet": item["description"].as_str().or(item["snippet"].as_str()).unwrap_or(""),
            }))
        })
        .collect()
}

#[derive(serde::Deserialize)]
pub struct FetchParams {
    url: String,
    provider: Option<String>,
}

pub async fn fetch_get(State(app): State<SharedApp>, Query(q): Query<FetchParams>) -> Response {
    fetch_inner(app, q).await
}

pub async fn fetch(State(app): State<SharedApp>, Json(payload): Json<Value>) -> Response {
    let Some(url) = payload["url"].as_str().filter(|u| !u.is_empty()) else {
        return error_response(StatusCode::BAD_REQUEST, "'url' is required");
    };
    fetch_inner(
        app,
        FetchParams {
            url: url.to_string(),
            provider: payload["provider"].as_str().map(String::from),
        },
    )
    .await
}

async fn fetch_inner(app: SharedApp, params: FetchParams) -> Response {
    if !params.url.starts_with("http://") && !params.url.starts_with("https://") {
        return error_response(StatusCode::BAD_REQUEST, "url must be http(s)");
    }
    let mut errors: Vec<String> = Vec::new();
    for p in &app.cfg.fetch.providers {
        if params.provider.as_deref().is_some_and(|o| o != p.name) {
            continue;
        }
        let Some(key) = service_ready(&app, "fetch", p) else { continue };
        super::record(&app, &key);
        match fetch_one(&app, p, &params.url).await {
            Ok(content) => {
                app.state.clear_cooldown(&key, "");
                return (
                    StatusCode::OK,
                    Json(json!({"provider": p.name, "url": params.url, "content": content})),
                )
                    .into_response();
            }
            Err(e) => {
                app.state.set_cooldown(&key, None, None, true, &format!("{e:#}"));
                errors.push(format!("{}: {e:#}", p.name));
            }
        }
    }
    error_response(
        StatusCode::BAD_GATEWAY,
        if errors.is_empty() { "no fetch provider available".into() } else { errors.join("; ") },
    )
}

async fn fetch_one(app: &App, p: &ServiceProvider, url: &str) -> anyhow::Result<String> {
    let api_key = app.secrets.resolve_key(&p.api_key)?;
    let timeout = std::time::Duration::from_secs(60);
    match p.kind {
        ServiceKind::JinaReader => {
            let resp = app
                .http
                .get(format!("{JINA_READER_URL}{url}"))
                .timeout(timeout)
                .bearer_auth(api_key)
                .header("x-return-format", "markdown")
                .send()
                .await?;
            anyhow::ensure!(resp.status().is_success(), "http {}", resp.status());
            Ok(resp.text().await?)
        }
        ServiceKind::FirecrawlScrape => {
            let resp = app
                .http
                .post(FIRECRAWL_SCRAPE_URL)
                .timeout(timeout)
                .bearer_auth(api_key)
                .json(&json!({"url": url, "formats": ["markdown"]}))
                .send()
                .await?;
            anyhow::ensure!(resp.status().is_success(), "http {}", resp.status());
            let body: Value = resp.json().await?;
            body["data"]["markdown"]
                .as_str()
                .map(String::from)
                .ok_or_else(|| anyhow::anyhow!("no markdown in response"))
        }
        _ => anyhow::bail!("provider kind is not a fetch kind"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalizes_all_three_search_dialects() {
        let brave = json!({"web": {"results": [
            {"title": "T", "url": "https://a", "description": "D"}
        ]}});
        let out = normalize_search(ServiceKind::Brave, &brave);
        assert_eq!(out[0]["title"], "T");
        assert_eq!(out[0]["snippet"], "D");

        let jina = json!({"data": [
            {"title": "J", "url": "https://b", "description": "JD", "content": "long"}
        ]});
        let out = normalize_search(ServiceKind::Jina, &jina);
        assert_eq!(out[0]["url"], "https://b");

        let firecrawl = json!({"data": {"web": [
            {"title": "F", "url": "https://c", "description": "FD"}
        ]}});
        let out = normalize_search(ServiceKind::FirecrawlSearch, &firecrawl);
        assert_eq!(out[0]["snippet"], "FD");

        // Items with no URL are dropped, not emitted half-empty.
        let bad = json!({"web": {"results": [{"title": "no url"}]}});
        assert!(normalize_search(ServiceKind::Brave, &bad).is_empty());
    }
}
