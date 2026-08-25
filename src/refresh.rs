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

use crate::config::{Config, ProviderConfig, Tier};
use crate::secrets::Secrets;
use crate::state::State;

const MODELS_DEV_URL: &str = "https://models.dev/api.json";
/// Reject an upstream catalog that looks truncated rather than overwriting a
/// good one with it (litellm's poison-pill guard).
const MIN_MODELS_DEV_ENTRIES: usize = 500;
/// How long a "does not support tools" verdict is trusted before re-probing.
const NEGATIVE_PROBE_TTL_SECS: u64 = 7 * 24 * 60 * 60;

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

/// Discover, report, and (when `write`) generate `generated.toml`.
pub async fn run(cfg: &Config, secrets: &Secrets, write: bool, out_path: &std::path::Path) -> Result<()> {
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
    let mut found: BTreeMap<String, Vec<Discovered>> = BTreeMap::new();
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
                let all = models.clone();
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
                // Keep the FULL list: a model already in config.toml still needs
                // its discovered capability data, or generation would re-probe
                // something models.dev already answered.
                found.insert(name.clone(), all);
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

    if !write {
        println!("\n(dry run — nothing written. Use `pxy refresh --write` to generate.)");
        return Ok(());
    }
    // Generating from degraded discovery would shrink the chain to whatever
    // happened to work this minute, and the shrunken file would then be loaded
    // as truth. Credential failures (a locked gpg agent takes out EVERY
    // provider at once) or a high failure rate abort the write; the previous
    // generated.toml stays in force.
    let cred_failures = failures
        .iter()
        .filter(|(_, why)| why.starts_with("credential:"))
        .count();
    if cred_failures > 0 || failures.len() > n_ok / 2 {
        anyhow::bail!(
            "refusing to write: {} discovery failure(s), {} credential-related \
             (locked gpg agent?). Fix access and rerun; the existing \
             generated.toml is untouched.",
            failures.len(),
            cred_failures
        );
    }
    generate(cfg, secrets, &http, found, out_path).await
}

/// Build and write `generated.toml`.
async fn generate(
    cfg: &Config,
    secrets: &Secrets,
    http: &reqwest::Client,
    discovered: BTreeMap<String, Vec<Discovered>>,
    out_path: &std::path::Path,
) -> Result<()> {
    let state = State::open(&crate::config::data_dir().join("state.sqlite"))?;
    let today = today();
    let prefs = &cfg.preferences;
    let rank_of = |canon: &str| prefs.models.iter().position(|p| canonical(p) == canon);

    println!("\n── generating ──");

    let mut per_provider: BTreeMap<String, Vec<(String, u64, Option<bool>)>> = BTreeMap::new();
    let mut candidates: Vec<Candidate> = Vec::new();
    let mut probed = 0usize;
    let mut dropped_promo: Vec<String> = Vec::new();

    for (name, pcfg) in &cfg.providers {
        if !pcfg.enabled {
            continue;
        }
        // Start from what config.toml already has. Never lose a hand-added
        // model: discovery can omit one that works.
        let mut models: BTreeMap<String, u64> = pcfg
            .models
            .iter()
            .map(|m| {
                let s = m.spec();
                (s.id, s.context_length)
            })
            .collect();
        // Hand-asserted capability wins over both discovery and probing.
        let asserted: HashMap<String, bool> = pcfg
            .models
            .iter()
            .filter_map(|m| {
                let s = m.spec();
                s.tool_call.map(|t| (s.id, t))
            })
            .collect();

        // Drop expired promo models wherever they came from.
        if let Some(promo) = &pcfg.promo {
            if promo.is_expired(&today) {
                for id in &promo.models {
                    if models.remove(id).is_some() {
                        dropped_promo.push(format!("{name}/{id}"));
                    }
                }
            }
        }

        for d in discovered.get(name).into_iter().flatten() {
            // Only free models are auto-added. The full discovery list is kept
            // for capability lookups, but unioning it wholesale would pull a
            // provider's entire paid catalogue into the exposed model list.
            if d.free != Tri::Yes {
                continue;
            }
            if pcfg
                .promo
                .as_ref()
                .is_some_and(|p| p.is_expired(&today) && p.models.contains(&d.id))
            {
                continue;
            }
            models.insert(d.id.clone(), d.context.unwrap_or(crate::config::default_context()));
        }

        // Candidates for `auto`: reserve tiers never qualify, whatever their
        // ranking — a preference list must not be able to start spending money.
        if pcfg.tier != Tier::Reserve {
            for (id, ctx) in &models {
                let canon = canonical(id);
                let rank = rank_of(&canon);
                // Only rank-worthy models are worth a probe.
                let known = discovered
                    .get(name)
                    .and_then(|v| v.iter().find(|d| &d.id == id))
                    .map(|d| d.tool_call)
                    .unwrap_or(Tri::Unknown);
                let known = match asserted.get(id) {
                    Some(true) => Tri::Yes,
                    Some(false) => Tri::No,
                    None => known,
                };
                let tools = match known {
                    Tri::Unknown if rank.is_some() => {
                        probed += 1;
                        probe_tool_calling(name, pcfg, id, secrets, &state, http).await
                    }
                    other => other,
                };
                // Unknown is NOT eligible. An optimistic default here is
                // exactly the bug OmniRoute shipped.
                if tools != Tri::Yes {
                    continue;
                }
                candidates.push(Candidate {
                    provider: name.clone(),
                    id: id.clone(),
                    canonical: canon,
                    tier: pcfg.tier,
                    context: *ctx,
                    rank,
                });
            }
        }
        per_provider.insert(
            name.clone(),
            models
                .into_iter()
                .map(|(id, ctx)| {
                    let a = asserted.get(&id).copied();
                    (id, ctx, a)
                })
                .collect(),
        );
    }

    // Tier first (free pools before paid), preference order within a tier,
    // then widest context. Ranked models always precede unranked ones.
    candidates.sort_by(|a, b| {
        a.tier
            .cmp(&b.tier)
            .then(a.rank.is_none().cmp(&b.rank.is_none()))
            .then(a.rank.cmp(&b.rank))
            .then(b.context.cmp(&a.context))
            .then(a.provider.cmp(&b.provider))
    });

    // Cap pools per model so one popular model can't crowd out the chain, and
    // bound the unranked tail so `auto` stays preference-driven.
    let denied: BTreeSet<String> = prefs.deny.iter().map(|d| canonical(d)).collect();
    let mut seen: HashMap<String, usize> = HashMap::new();
    let mut unranked = 0usize;
    let auto: Vec<Candidate> = candidates
        .into_iter()
        .filter(|c| {
            if denied.contains(&c.canonical) || prefs.deny.contains(&c.id) {
                return false;
            }
            if c.rank.is_none() {
                unranked += 1;
                if unranked > prefs.max_unranked {
                    return false;
                }
            }
            let n = seen.entry(c.canonical.clone()).or_insert(0);
            *n += 1;
            *n <= prefs.max_pools_per_model
        })
        .collect();

    let unmatched: Vec<&String> = prefs
        .models
        .iter()
        .filter(|p| {
            let c = canonical(p);
            !auto.iter().any(|a| a.canonical == c)
        })
        .collect();

    let body = render_generated(&per_provider, &auto, &today);
    std::fs::write(out_path, &body)
        .with_context(|| format!("writing {}", out_path.display()))?;

    let total: usize = per_provider.values().map(|v| v.len()).sum();
    println!("probed {probed} model(s) for tool calling (results cached)");
    if !dropped_promo.is_empty() {
        println!("dropped {} expired promo model(s):", dropped_promo.len());
        for d in &dropped_promo {
            println!("  {d}");
        }
    }
    if !unmatched.is_empty() {
        println!("preferences with no eligible pool ({}):", unmatched.len());
        for u in &unmatched {
            println!("  {u}");
        }
    }
    println!(
        "wrote {} — {} models across {} providers, auto chain of {}",
        out_path.display(),
        total,
        per_provider.len(),
        auto.len()
    );
    Ok(())
}

// ---------------------------------------------------------------------------
// Stage 2: probing the gap models.dev can't answer
// ---------------------------------------------------------------------------

/// Ask a model to call a trivial tool. Only used when capability data is
/// missing AND the model is a candidate for `auto`, so the cost stays tiny.
/// A result is cached forever: a model's tool support doesn't change.
async fn probe_tool_calling(
    prov: &str,
    cfg: &ProviderConfig,
    model_id: &str,
    secrets: &Secrets,
    state: &State,
    http: &reqwest::Client,
) -> Tri {
    let key = format!("probe:tools:{prov}/{model_id}");
    if let Ok(Some(v)) = state.kv_get(&key) {
        let (verdict, at) = v.split_once('@').unwrap_or((v.as_str(), "0"));
        let age = now_secs().saturating_sub(at.parse().unwrap_or(0));
        match verdict {
            // Capability is intrinsic, so a YES never needs rechecking.
            "yes" => return Tri::Yes,
            // A NO expires. Free pools degrade and recover (aihubmix's
            // gemini-3.7-flash tool-called in the morning and stopped by the
            // afternoon), so a permanent negative would bury a model that
            // came back.
            "no" if age < NEGATIVE_PROBE_TTL_SECS => return Tri::No,
            _ => {}
        }
    }
    let Some(url) = cfg.base_url.clone() else {
        return Tri::Unknown;
    };
    let mut req = http.post(&url).header("content-type", "application/json");
    if let Some(sref) = cfg.api_key.as_ref().or(cfg.credentials.as_ref()) {
        match secrets.resolve_key(sref) {
            Ok(k) => req = req.header("authorization", format!("Bearer {k}")),
            Err(_) => return Tri::Unknown,
        }
    }
    for (k, v) in &cfg.headers {
        req = req.header(k, v);
    }
    let body = serde_json::json!({
        "model": model_id,
        // Generous budget on purpose: a reasoning model can spend a small
        // allowance entirely on thinking and emit no tool call, which would
        // read as "no tool support".
        "max_tokens": 512,
        "messages": [{"role": "user", "content": "What is the weather in Dhaka? Use the tool."}],
        "tools": [{"type": "function", "function": {
            "name": "get_weather",
            "description": "Get current weather for a city",
            "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}
        }}],
    });
    let Ok(resp) = req.json(&body).send().await else {
        return Tri::Unknown; // transport failure proves nothing; don't cache
    };
    if !resp.status().is_success() {
        return Tri::Unknown;
    }
    let Ok(v) = resp.json::<Value>().await else {
        return Tri::Unknown;
    };
    let choice = &v["choices"][0];
    let called = choice["message"]["tool_calls"]
        .as_array()
        .is_some_and(|a| !a.is_empty());
    if called {
        let _ = state.kv_set(&key, &format!("yes@{}", now_secs()));
        return Tri::Yes;
    }
    // Ran out of room before it could answer: that is not evidence of
    // anything, so record nothing.
    if choice["finish_reason"].as_str() == Some("length") {
        return Tri::Unknown;
    }
    let _ = state.kv_set(&key, &format!("no@{}", now_secs()));
    Tri::No
}

fn now_secs() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

// ---------------------------------------------------------------------------
// Stage 3: generating the model lists and the auto chain
// ---------------------------------------------------------------------------

/// A model eligible for the generated auto chain.
struct Candidate {
    provider: String,
    id: String,
    canonical: String,
    tier: Tier,
    context: u64,
    /// Index into the preference list; None = unranked (sorts after ranked).
    rank: Option<usize>,
}

/// Today as YYYY-MM-DD, for promo expiry.
fn today() -> String {
    jiff::Zoned::now().date().to_string()
}

/// Build `generated.toml`: per-provider model lists (union of hand-configured
/// and newly discovered) plus the auto chain.
fn render_generated(
    per_provider: &BTreeMap<String, Vec<(String, u64, Option<bool>)>>,
    auto: &[Candidate],
    stamp: &str,
) -> String {
    let mut out = String::new();
    out.push_str(&format!(
        "# AUTO-GENERATED by `pxy refresh --write` on {stamp}.\n"
    ));
    out.push_str(concat!(
        "# Do not edit: rerun the command instead. Hand-written provider settings\n",
        "# (credentials, limits, headers, quirks) live in config.toml and are never\n",
        "# touched by generation; only model lists and the auto chain come from here.\n",
        "# Model lists are a UNION with config.toml: a provider's /models can omit a\n",
        "# model that works (zai/glm-4.7-flash is absent from Z.AI's own listing).\n\n",
    ));
    for (prov, models) in per_provider {
        out.push_str(&format!("[providers.{prov}]\nmodels = [\n"));
        for (id, ctx, asserted) in models {
            let extra = match asserted {
                Some(t) => format!(", tool_call = {t}"),
                None => String::new(),
            };
            out.push_str(&format!(
                "  {{ id = \"{id}\", context_length = {ctx}{extra} }},\n"
            ));
        }
        out.push_str("]\n\n");
    }
    out.push_str("[auto]\nmodels = [\n");
    let mut last_tier: Option<Tier> = None;
    for c in auto {
        if last_tier != Some(c.tier) {
            out.push_str(&format!("  # --- {:?} ---\n", c.tier).to_lowercase());
            last_tier = Some(c.tier);
        }
        let note = match c.rank {
            Some(r) => format!("preference #{}", r + 1),
            None => "unranked".to_string(),
        };
        out.push_str(&format!(
            "  \"{}/{}\",{}# {note}, ctx={}\n",
            c.provider,
            c.id,
            " ".repeat(46usize.saturating_sub(c.provider.len() + c.id.len())),
            c.context
        ));
    }
    out.push_str("]\n");
    out
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

    fn cand(provider: &str, id: &str, tier: Tier, rank: Option<usize>, ctx: u64) -> Candidate {
        Candidate {
            provider: provider.into(),
            id: id.into(),
            canonical: canonical(id),
            tier,
            context: ctx,
            rank,
        }
    }

    /// The ordering contract: cost class outranks preference, so a ranking can
    /// never move a paid pool above a free one.
    fn sort_for_test(mut v: Vec<Candidate>) -> Vec<String> {
        v.sort_by(|a, b| {
            a.tier
                .cmp(&b.tier)
                .then(a.rank.is_none().cmp(&b.rank.is_none()))
                .then(a.rank.cmp(&b.rank))
                .then(b.context.cmp(&a.context))
                .then(a.provider.cmp(&b.provider))
        });
        v.into_iter().map(|c| format!("{}/{}", c.provider, c.id)).collect()
    }

    #[test]
    fn tier_outranks_preference() {
        // "best" is the top preference but only on a finite pool; a lower-ranked
        // model on a free pool must still come first.
        let order = sort_for_test(vec![
            cand("finite-pool", "best", Tier::Finite, Some(0), 100),
            cand("free-pool", "second", Tier::Free, Some(1), 100),
        ]);
        assert_eq!(order, ["free-pool/second", "finite-pool/best"]);
    }

    #[test]
    fn preference_orders_within_a_tier_and_ranked_beats_unranked() {
        let order = sort_for_test(vec![
            cand("p", "unranked-huge", Tier::Free, None, 1_000_000),
            cand("p", "third", Tier::Free, Some(2), 100),
            cand("p", "first", Tier::Free, Some(0), 100),
        ]);
        assert_eq!(order, ["p/first", "p/third", "p/unranked-huge"]);
    }

    #[test]
    fn reserve_tier_is_ordered_last() {
        let order = sort_for_test(vec![
            cand("money", "top-model", Tier::Reserve, Some(0), 100),
            cand("free", "meh", Tier::Free, None, 100),
        ]);
        assert_eq!(order, ["free/meh", "money/top-model"]);
    }

    #[test]
    fn pool_cap_limits_duplicates_of_one_model() {
        let cands = vec![
            cand("a", "kimi-k3", Tier::Free, Some(0), 100),
            cand("b", "kimi-k3:free", Tier::Free, Some(0), 100),
            cand("c", "moonshotai/kimi-k3", Tier::Free, Some(0), 100),
            cand("d", "kimi-k3-free", Tier::Free, Some(0), 100),
        ];
        let mut seen: HashMap<String, usize> = HashMap::new();
        let kept: Vec<_> = cands
            .into_iter()
            .filter(|c| {
                let n = seen.entry(c.canonical.clone()).or_insert(0);
                *n += 1;
                *n <= 2
            })
            .collect();
        assert_eq!(kept.len(), 2, "all four are the same canonical model");
    }

    #[test]
    fn promo_expiry_is_date_ordered_and_fails_closed() {
        let p = crate::config::Promo {
            models: vec!["m".into()],
            expires: "2026-09-06".into(),
        };
        assert!(!p.is_expired("2026-09-06"), "expiry day is inclusive");
        assert!(!p.is_expired("2026-08-25"));
        assert!(p.is_expired("2026-09-07"));
        // An unparseable date must drop the model, not keep spending on it.
        let bad = crate::config::Promo {
            models: vec!["m".into()],
            expires: "".into(),
        };
        assert!(bad.is_expired("2026-08-25"));
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
