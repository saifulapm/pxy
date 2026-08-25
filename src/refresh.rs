//! `pxy refresh` — discover live provider catalogs and report drift.
//!
//! Two catalogs answer two different questions and neither answers both:
//!   * a provider's `/models` says what THIS ACCOUNT may call (ids, availability,
//!     and — for the richer gateways — pricing that proves free-ness);
//!   * models.dev says what a model DOES (tool calling, context, cost) and covers
//!     ~94% of what we configure, including every entry currently in `auto`.
//! Joining them leaves only a small remainder that needs a live probe, so probing
//! is a fallback rather than the mechanism.
//!
//! Hard rules, each of which is somebody's post-mortem:
//!   * **Absence is not death.** A model missing from a listing is REPORTED, never
//!     removed. Cloudflare's `id` is a UUID and its real id is `name`; reading the
//!     wrong field made five working models look deleted. Only a failing call is
//!     evidence.
//!   * **Capabilities are tri-state.** `Unknown` must never collapse into `No` or
//!     into an optimistic `Yes` — OmniRoute shipped `tools: bool` with `false`
//!     doubling as unknown and had to bump their schema to undo it, and their
//!     optimistic tool-calling default caused a real routing bug.
//!   * **A failed fetch is not an empty catalog.** They are separate outcomes;
//!     conflating them once made an expired credential look like "no new models".
//!   * **Billing safety is never inferred.** Whether a provider hard-stops instead
//!     of charging is a curated fact about the vendor, so it stays hand-written.

use std::collections::{BTreeMap, BTreeSet, HashMap};

use anyhow::{Context, Result};
use serde_json::Value;

use crate::config::{Config, ProviderConfig};
use crate::secrets::Secrets;

const MODELS_DEV_URL: &str = "https://models.dev/api.json";
/// Reject an upstream catalog that looks truncated rather than overwriting a
/// good one with it (litellm's poison-pill guard).
const MIN_MODELS_DEV_ENTRIES: usize = 500;

/// Tri-state. `Unknown` is a real answer and must survive as one.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Tri {
    Yes,
    No,
    Unknown,
}

impl Tri {
    fn from_opt(b: Option<bool>) -> Self {
        match b {
            Some(true) => Tri::Yes,
            Some(false) => Tri::No,
            None => Tri::Unknown,
        }
    }
    fn mark(self) -> &'static str {
        match self {
            Tri::Yes => "yes",
            Tri::No => "no",
            Tri::Unknown => "?",
        }
    }
}

/// Capability facts for one canonical model name.
#[derive(Debug, Clone, Default)]
pub struct Caps {
    pub tool_call: Option<bool>,
    pub context: Option<u64>,
}

/// One model as offered by one provider.
#[derive(Debug, Clone)]
pub struct Discovered {
    pub id: String,
    pub canonical: String,
    pub free: Tri,
    pub tool_call: Tri,
    pub context: Option<u64>,
}

/// Outcome of asking one provider for its catalog. `Failed` is deliberately
/// distinct from an empty `Ok` so a broken credential can never read as
/// "this provider has no models".
pub enum ProviderCatalog {
    Ok(Vec<Discovered>),
    Skipped(String),
    Failed(String),
}

/// Normalize a provider-specific model id to a comparable name.
/// Conservative on purpose: it drops vendor prefixes and free/variant suffixes,
/// but never rewrites the stem, so two genuinely different models can't collide.
pub fn canonical(id: &str) -> String {
    let mut t = id.trim().to_ascii_lowercase();
    if let Some(p) = t.rfind('/') {
        t = t[p + 1..].to_string(); // "anthropic/claude-x" | "@cf/openai/y"
    }
    for suf in [":free", "-free", "-latest", "-preview", "-instruct", "-discounted"] {
        if let Some(s) = t.strip_suffix(suf) {
            t = s.to_string();
        }
    }
    for pre in ["coding-", "claude-"] {
        if let Some(s) = t.strip_prefix(pre) {
            t = s.to_string();
        }
    }
    t
}

/// Fetch models.dev and index it by canonical name.
pub async fn fetch_capabilities(http: &reqwest::Client) -> Result<HashMap<String, Caps>> {
    let resp = http
        .get(MODELS_DEV_URL)
        .send()
        .await
        .context("fetching models.dev")?;
    if !resp.status().is_success() {
        anyhow::bail!("models.dev returned {}", resp.status());
    }
    let root: Value = resp.json().await.context("parsing models.dev")?;
    let obj = root.as_object().context("models.dev: expected an object")?;

    let mut out: HashMap<String, Caps> = HashMap::new();
    for provider in obj.values() {
        let Some(models) = provider["models"].as_object() else {
            continue;
        };
        for (mid, rec) in models {
            let caps = Caps {
                tool_call: rec["tool_call"].as_bool(),
                context: rec["limit"]["context"].as_u64(),
            };
            // First writer wins: providers are iterated in a stable order and
            // the facts are about the MODEL, not the reseller.
            out.entry(canonical(mid)).or_insert(caps);
        }
    }
    if out.len() < MIN_MODELS_DEV_ENTRIES {
        anyhow::bail!(
            "models.dev returned only {} models (<{MIN_MODELS_DEV_ENTRIES}) — refusing \
             to treat a truncated catalog as authoritative",
            out.len()
        );
    }
    Ok(out)
}

/// Where a provider's model list lives. Explicit override wins; otherwise
/// derive it from the chat endpoint, which works for the large majority.
fn models_url(cfg: &ProviderConfig) -> Option<String> {
    if let Some(u) = &cfg.models_url {
        return Some(u.clone());
    }
    let base = cfg.base_url.as_ref()?;
    if let Some(stem) = base.strip_suffix("/chat/completions") {
        return Some(format!("{stem}/models"));
    }
    None
}

/// Free-ness from three independent signals, OR'd. Pricing is exact where the
/// provider reports it; the id suffix is the fallback for gateways that return
/// nothing but ids. Neither firing means Unknown, not No.
fn free_of(rec: &Value, id: &str) -> Tri {
    let zero = |v: &Value| match v {
        Value::Number(n) => n.as_f64() == Some(0.0),
        Value::String(s) => s.parse::<f64>() == Ok(0.0),
        _ => false,
    };
    let p = &rec["pricing"];
    if !p.is_null() && zero(&p["prompt"]) && zero(&p["completion"]) {
        return Tri::Yes;
    }
    if !p.is_null() && (p["prompt"].is_number() || p["prompt"].is_string()) {
        return Tri::No;
    }
    if rec["isFree"].as_bool() == Some(true) {
        return Tri::Yes;
    }
    let low = id.to_ascii_lowercase();
    if low.ends_with(":free") || low.ends_with("-free") {
        return Tri::Yes;
    }
    Tri::Unknown
}

/// Ask one provider for its catalog and join in models.dev capabilities.
pub async fn discover(
    cfg: &ProviderConfig,
    secrets: &Secrets,
    http: &reqwest::Client,
    caps: &HashMap<String, Caps>,
) -> ProviderCatalog {
    if !cfg.discover {
        return ProviderCatalog::Skipped("discover = false".into());
    }
    let Some(url) = models_url(cfg) else {
        return ProviderCatalog::Skipped("no models_url and base_url is not /chat/completions".into());
    };

    let mut req = http.get(&url).header("accept", "application/json");
    if let Some(sref) = cfg.api_key.as_ref().or(cfg.credentials.as_ref()) {
        match secrets.resolve_key(sref) {
            Ok(k) => req = req.header("authorization", format!("Bearer {k}")),
            Err(e) => return ProviderCatalog::Failed(format!("credential: {e:#}")),
        }
    }
    for (k, v) in &cfg.headers {
        req = req.header(k, v);
    }

    let resp = match req.send().await {
        Ok(r) => r,
        Err(e) => return ProviderCatalog::Failed(format!("request: {e}")),
    };
    let status = resp.status();
    let body = match resp.text().await {
        Ok(b) => b,
        Err(e) => return ProviderCatalog::Failed(format!("body: {e}")),
    };
    if !status.is_success() {
        return ProviderCatalog::Failed(format!("HTTP {status}: {}", snippet(&body)));
    }
    let root: Value = match serde_json::from_str(&body) {
        Ok(v) => v,
        Err(e) => return ProviderCatalog::Failed(format!("json: {e}")),
    };

    // data[] | models[] | result[] | bare array — covers every gateway we use.
    let arr = ["data", "models", "result"]
        .iter()
        .find_map(|k| root[*k].as_array())
        .or_else(|| root.as_array());
    let Some(arr) = arr else {
        return ProviderCatalog::Failed(format!("unrecognized shape: {}", snippet(&body)));
    };

    let id_field = cfg.id_field.as_deref().unwrap_or("id");
    let mut out = Vec::new();
    for rec in arr {
        // Fall back across the known id spellings, but honour an explicit
        // id_field first: Cloudflare's `id` is a UUID and `name` is the real id.
        let id = rec[id_field]
            .as_str()
            .or_else(|| rec["id"].as_str())
            .or_else(|| rec["modelId"].as_str())
            .or_else(|| rec["name"].as_str());
        let Some(id) = id else { continue };
        let canon = canonical(id);
        let known = caps.get(&canon);
        out.push(Discovered {
            id: id.to_string(),
            free: free_of(rec, id),
            tool_call: Tri::from_opt(known.and_then(|c| c.tool_call)),
            context: rec["context_length"]
                .as_u64()
                .or_else(|| rec["tokenLimits"]["maxInputTokens"].as_u64())
                .or_else(|| known.and_then(|c| c.context)),
            canonical: canon,
        });
    }
    ProviderCatalog::Ok(out)
}

fn snippet(s: &str) -> String {
    let t = s.trim().replace('\n', " ");
    if t.chars().count() > 120 {
        format!("{}…", t.chars().take(120).collect::<String>())
    } else {
        t
    }
}

/// Run discovery across all providers and print a drift report. Read-only.
pub async fn run(cfg: &Config, secrets: &Secrets) -> Result<()> {
    let http = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(30))
        .build()?;

    println!("fetching models.dev …");
    let caps = match fetch_capabilities(&http).await {
        Ok(c) => {
            println!("  {} models with capability data\n", c.len());
            c
        }
        Err(e) => {
            // Degrade loudly: discovery still works, capabilities just go Unknown.
            println!("  WARNING: {e:#}\n  continuing without capability data\n");
            HashMap::new()
        }
    };

    let mut stale: Vec<String> = Vec::new();
    let mut failures: Vec<(String, String)> = Vec::new();
    let mut new_free: BTreeMap<String, Vec<Discovered>> = BTreeMap::new();
    let mut pools: BTreeMap<String, BTreeSet<String>> = BTreeMap::new();
    let (mut n_ok, mut n_models) = (0usize, 0usize);

    for (name, pcfg) in &cfg.providers {
        if !pcfg.enabled {
            continue;
        }
        let configured: BTreeSet<String> = pcfg
            .models
            .iter()
            .map(|m| m.spec().id)
            .collect();

        match discover(pcfg, secrets, &http, &caps).await {
            ProviderCatalog::Skipped(why) => {
                println!("{name:<22} skipped   ({why})");
            }
            ProviderCatalog::Failed(why) => {
                println!("{name:<22} FAILED    {why}");
                failures.push((name.clone(), why));
            }
            ProviderCatalog::Ok(models) => {
                n_ok += 1;
                n_models += models.len();
                // Match on the exact id first, then on the canonical form:
                // Google lists "models/gemini-3-flash-preview" but accepts the
                // bare id, and a spelling difference is not a missing model.
                let live: BTreeSet<&str> = models.iter().map(|m| m.id.as_str()).collect();
                let live_canon: BTreeSet<&str> =
                    models.iter().map(|m| m.canonical.as_str()).collect();
                let missing: Vec<&String> = configured
                    .iter()
                    .filter(|c| {
                        !live.contains(c.as_str()) && !live_canon.contains(canonical(c).as_str())
                    })
                    .collect();
                println!(
                    "{name:<22} {:>4} live, {:>2} configured{}",
                    models.len(),
                    configured.len(),
                    if missing.is_empty() {
                        String::new()
                    } else {
                        format!("  ⚠ {} not listed", missing.len())
                    }
                );
                for m in &missing {
                    stale.push(format!("{name}/{m}"));
                }
                for m in &models {
                    if m.free == Tri::Yes && m.tool_call == Tri::Yes {
                        pools
                            .entry(m.canonical.clone())
                            .or_default()
                            .insert(name.clone());
                    }
                }
                // Free + tool-capable + not already configured = candidates.
                let cands: Vec<Discovered> = models
                    .into_iter()
                    .filter(|m| {
                        m.free == Tri::Yes
                            && m.tool_call == Tri::Yes
                            && !configured.contains(&m.id)
                    })
                    .collect();
                if !cands.is_empty() {
                    new_free.insert(name.clone(), cands);
                }
            }
        }
    }

    println!("\n── summary ──");
    println!("{n_ok} providers discovered, {n_models} models listed");

    if !failures.is_empty() {
        println!("\ndiscovery failed ({}) — these were NOT checked for drift:", failures.len());
        for (n, why) in &failures {
            println!("  {n:<20} {why}");
        }
    }

    println!("\nconfigured but not listed upstream ({}):", stale.len());
    if stale.is_empty() {
        println!("  none");
    } else {
        for s in &stale {
            println!("  {s}");
        }
        println!(
            "  NOTE: not proof of removal — a listing can omit a working model.\n\
             \x20       Verify with a real call before deleting anything."
        );
    }

    // Which canonical models are served by more than one provider. This is the
    // ground truth a bare-name preference list resolves against, so it belongs
    // in the report: it shows what "kimi-k3" would actually match.
    if !pools.is_empty() {
        let mut multi: Vec<(&String, &BTreeSet<String>)> =
            pools.iter().filter(|(_, v)| v.len() > 1).collect();
        multi.sort_by_key(|(name, provs)| (std::cmp::Reverse(provs.len()), (*name).clone()));
        println!("\nfree models served by multiple providers ({}):", multi.len());
        for (name, provs) in multi.iter().take(12) {
            println!(
                "  {:<26} {} pools: {}",
                name,
                provs.len(),
                provs.iter().cloned().collect::<Vec<_>>().join(", ")
            );
        }
        if multi.len() > 12 {
            println!("  … and {} more", multi.len() - 12);
        }
    }

    let total_new: usize = new_free.values().map(|v| v.len()).sum();
    println!("\nfree + tool-capable, not yet configured ({total_new}):");
    for (prov, models) in &new_free {
        let mut sorted = models.clone();
        sorted.sort_by_key(|m| std::cmp::Reverse(m.context.unwrap_or(0)));
        for m in sorted.iter().take(6) {
            println!(
                "  {:<20} {:<44} ctx={:<9} tools={}",
                prov,
                m.id,
                m.context.map(|c| c.to_string()).unwrap_or_else(|| "?".into()),
                m.tool_call.mark()
            );
        }
        if sorted.len() > 6 {
            println!("  {:<20} … and {} more", prov, sorted.len() - 6);
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn canonical_collapses_vendor_prefixes_and_free_suffixes() {
        for (input, want) in [
            ("moonshotai/kimi-k3", "kimi-k3"),
            ("minimax/minimax-m3:free", "minimax-m3"),
            ("@cf/openai/gpt-oss-120b", "gpt-oss-120b"),
            ("MiniMax-M3", "minimax-m3"),
            ("coding-glm-5.2-free", "glm-5.2"),
            ("claude-haiku-4.5", "haiku-4.5"),
            ("meituan/longcat-2.0-free", "longcat-2.0"),
        ] {
            assert_eq!(canonical(input), want, "canonical({input})");
        }
    }

    #[test]
    fn canonical_does_not_merge_distinct_models() {
        assert_ne!(canonical("glm-4.7-flash"), canonical("glm-5.2"));
        assert_ne!(canonical("minimax-m3"), canonical("minimax-m2.7"));
        assert_ne!(canonical("gpt-oss-120b"), canonical("gpt-oss-20b"));
    }

    #[test]
    fn free_detection_prefers_pricing_over_suffix() {
        // exact: zero price
        assert_eq!(
            free_of(&json!({"pricing": {"prompt": "0", "completion": "0"}}), "x"),
            Tri::Yes
        );
        // exact: non-zero price beats a misleading name
        assert_eq!(
            free_of(&json!({"pricing": {"prompt": "0.0000003"}}), "something-free"),
            Tri::No
        );
        // no pricing at all -> suffix heuristic
        assert_eq!(free_of(&json!({}), "minimax-m3-free"), Tri::Yes);
        assert_eq!(free_of(&json!({}), "z-ai/glm-5.3:free"), Tri::Yes);
        // kilocode's explicit flag
        assert_eq!(free_of(&json!({"isFree": true}), "x"), Tri::Yes);
        // nothing to go on must stay Unknown, never a guess
        assert_eq!(free_of(&json!({}), "gpt-5.6-sol"), Tri::Unknown);
    }

    #[test]
    fn tri_state_never_collapses_unknown() {
        assert_eq!(Tri::from_opt(None), Tri::Unknown);
        assert_eq!(Tri::from_opt(Some(false)), Tri::No);
        assert_eq!(Tri::from_opt(Some(true)), Tri::Yes);
        assert_ne!(Tri::Unknown, Tri::No);
    }

    #[test]
    fn models_url_derives_from_chat_endpoint_and_honours_override() {
        let mut c = ProviderConfig::test_default();
        c.base_url = Some("https://api.example.com/v1/chat/completions".into());
        assert_eq!(models_url(&c).unwrap(), "https://api.example.com/v1/models");

        c.models_url = Some("https://custom/list".into());
        assert_eq!(models_url(&c).unwrap(), "https://custom/list");

        // A non-derivable endpoint must yield None rather than a guessed URL.
        let mut d = ProviderConfig::test_default();
        d.base_url = Some("https://codewhisperer.amazonaws.com/generateAssistantResponse".into());
        assert!(models_url(&d).is_none());
    }
}
