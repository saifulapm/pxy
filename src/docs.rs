//! The Context7 tier behind `find_docs` (wiki:find-docs): resolve a library
//! name to a Context7 id, fetch a topic's snippets as text, and cache both
//! halves in `kv` for a week so a repeated question costs no Context7 call.
//!
//! No key is needed; `[server_tools.context7] api_key` only raises the cap,
//! so the bearer header rides only when that secret resolves.

use serde_json::{Value, json};
use tracing::warn;

use crate::router::SharedApp;
use crate::state::State;

const SEARCH_URL: &str = "https://context7.com/api/v1/search";
const DOCS_BASE: &str = "https://context7.com/api/v1";

/// A cache row older than a week is refetched (wiki:find-docs).
pub const CACHE_TTL_SECS: u64 = 604_800;

/// The kv key holding a library search's results.
pub fn search_key(name: &str) -> String {
    format!("context7:search:{}", name.to_lowercase())
}

/// The kv key holding one topic fetch's text.
pub fn docs_key(id: &str, topic: &str, tokens: u64) -> String {
    format!("context7:docs:{id}:{topic}:{tokens}")
}

/// Which library a search's results meant. The first result whose id ends in
/// `name` wins over Context7's own ranking — a search for "axum" puts the
/// site scrape `/websites/rs_axum` first, and the repo id is what a model
/// asking about axum means. With a `version`, the first of that library's
/// versions containing it is appended to the id.
pub fn pick_library(results: &[Value], name: &str, version: Option<&str>) -> Result<String, String> {
    // Context7's own order stands: for "sqlx" it ranks the Rust site scrape
    // first, and preferring an id that ends in the bare name would hand the
    // Go repo back instead. A model that means another entry pins its id.
    let usable = || results.iter().filter(|r| r["id"].is_string());
    let first = usable().next().ok_or_else(|| format!("no library matched '{name}'"))?;
    let Some(version) = version else { return Ok(first["id"].as_str().unwrap_or_default().to_string()) };

    // Context7 spells a version `axum_v0_8_4`, so "0.8" is the underscored
    // "0_8" carried by the first version that has it, on the first result
    // that publishes one.
    let wanted = version.replace('.', "_");
    fn versions(r: &Value) -> Vec<&str> {
        r["versions"].as_array().map_or(Vec::new(), |v| v.iter().filter_map(Value::as_str).collect())
    }
    for r in usable() {
        if let Some(v) = versions(r).iter().find(|v| v.contains(&wanted)) {
            return Ok(format!("{}/{v}", r["id"].as_str().unwrap_or_default()));
        }
    }
    let seen: Vec<&str> = usable().flat_map(versions).collect();
    Err(format!("no '{name}' entry has a version matching '{version}'; seen: {}", seen.join(", ")))
}

/// Ask Context7 which libraries match a name.
pub async fn search_library(app: &SharedApp, name: &str) -> Result<Vec<Value>, String> {
    let url = format!("{SEARCH_URL}?query={}", urlencode(name));
    let body = get(app, &url).await?;
    let body: Value = serde_json::from_str(&body).map_err(|e| format!("context7 search: {e}"))?;
    Ok(body["results"].as_array().cloned().unwrap_or_default())
}

/// Fetch one library's snippets for a topic, as the text the model reads.
pub async fn fetch_docs(app: &SharedApp, id: &str, topic: &str, tokens: u64) -> Result<String, String> {
    let url = format!("{DOCS_BASE}{id}?topic={}&tokens={tokens}&type=txt", urlencode(topic));
    get(app, &url).await
}

/// A cached value, when the row is younger than `max_age_secs`.
pub fn cache_get(state: &State, key: &str, max_age_secs: u64) -> Option<Value> {
    let row = state.kv_get(key).ok().flatten()?;
    let row: Value = serde_json::from_str(&row).ok()?;
    let at = row["at"].as_u64()?;
    (now_secs().saturating_sub(at) <= max_age_secs).then(|| row["v"].clone())
}

/// Cache a value under `key`, stamped with the time it was fetched.
pub fn cache_put(state: &State, key: &str, value: &Value) {
    let row = json!({"at": now_secs(), "v": value});
    if let Err(e) = state.kv_set(key, &row.to_string()) {
        warn!("context7 cache write failed for {key}: {e:#}");
    }
}

/// One GET against Context7, with the bearer header when a key resolves.
/// `Err` is the sentence the model reads: a rate limit names its reset.
async fn get(app: &SharedApp, url: &str) -> Result<String, String> {
    let mut req = app.http.get(url).timeout(std::time::Duration::from_secs(20));
    if let Some(key) = app
        .cfg
        .server_tools
        .context7
        .as_ref()
        .and_then(|c| app.secrets.resolve_key(&c.api_key).ok())
        .filter(|k| !k.is_empty())
    {
        req = req.bearer_auth(key);
    }
    let resp = req.send().await.map_err(|e| format!("context7: {e}"))?;
    if resp.status().as_u16() == 429 {
        let reset = resp
            .headers()
            .get("ratelimit-reset")
            .and_then(|v| v.to_str().ok())
            .unwrap_or("unknown");
        return Err(format!("context7 rate limit; resets at {reset}"));
    }
    if !resp.status().is_success() {
        return Err(format!("context7: http {}", resp.status()));
    }
    resp.text().await.map_err(|e| format!("context7: {e}"))
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

fn now_secs() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn results() -> Vec<Value> {
        vec![
            json!({"id": "/websites/rs_axum", "title": "Axum", "versions": []}),
            json!({"id": "/tokio-rs/axum", "title": "axum",
                   "versions": ["axum_v0_8_4", "axum_v0_7_9"]}),
        ]
    }

    /// Context7's ranking is the pick: it puts the site scrape first for
    /// "axum", and that is the richer entry.
    #[test]
    fn pick_library_takes_the_first_result() {
        assert_eq!(pick_library(&results(), "axum", None).unwrap(), "/websites/rs_axum");
        assert!(pick_library(&[], "nope", None).is_err(), "no results is no library");
        let unusable = vec![json!({"title": "no id"})];
        assert!(pick_library(&unusable, "axum", None).is_err(), "an entry with no id is no library");
    }

    /// A version's dots are the ids' underscores, and a partial one picks the
    /// first version that carries it, on the first result that publishes one:
    /// the site scrape lists none, so the repo id answers.
    #[test]
    fn pick_library_appends_a_matching_version() {
        assert_eq!(
            pick_library(&results(), "axum", Some("0.8")).unwrap(),
            "/tokio-rs/axum/axum_v0_8_4"
        );
        let err = pick_library(&results(), "axum", Some("9.9")).unwrap_err();
        assert!(err.contains("axum_v0_8_4"), "the error lists what there is: {err}");
        assert!(err.contains("axum_v0_7_9"), "across every result: {err}");
    }

    fn state(name: &str) -> State {
        let dir = std::env::temp_dir().join(format!("pxy-docs-{}-{name}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        State::open(&dir.join("s.sqlite")).unwrap()
    }

    #[test]
    fn cache_serves_a_fresh_row_and_drops_a_stale_one() {
        let st = state("cache");
        cache_put(&st, "context7:search:axum", &json!({"results": 1}));
        assert_eq!(cache_get(&st, "context7:search:axum", 60), Some(json!({"results": 1})));
        assert_eq!(cache_get(&st, "context7:search:other", 60), None, "no row at all");

        // A row written a fortnight ago is past the week the tool serves.
        let stale = json!({"at": now_secs() - 2 * CACHE_TTL_SECS, "v": "old"});
        st.kv_set("context7:docs:/x:y:4000", &stale.to_string()).unwrap();
        assert_eq!(cache_get(&st, "context7:docs:/x:y:4000", CACHE_TTL_SECS), None);
    }
}
