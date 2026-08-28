//! `pxy refresh` — discover live provider catalogs and report drift.
//!
//! Two catalogs answer two different questions and neither answers both:
//!   * a provider's `/models` says what THIS ACCOUNT may call (ids, availability,
//!     and — for the richer gateways — pricing that proves free-ness);
//!   * models.dev says what a model DOES (tool calling, context) and covers ~94%
//!     of what we configure.
//! Joining them is the whole mechanism: `--generate` writes `models.toml` with
//! EVERY model of every provider, free and paid alike, and the groups that
//! decide what actually gets routed stay hand-written in config.toml.
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
    /// Back to an Option for the generated file. `Unknown` writes NOTHING —
    /// the key is absent, not `false`: OmniRoute shipped `tools: bool` with
    /// `false` doubling as unknown and had to bump their schema to undo it.
    fn as_opt(self) -> Option<bool> {
        match self {
            Tri::Yes => Some(true),
            Tri::No => Some(false),
            Tri::Unknown => None,
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
            context: discovered_context(rec, known),
            canonical: canon,
        });
    }
    ProviderCatalog::Ok(out)
}

/// Context window for one discovered record, across the id spellings gateways
/// use. A ZERO window is not a window, it is a missing one: aihubmix lists
/// image models (gpt-image-2-free) with a null `context_length` and the
/// models.dev fallback answers 0. Left as `Some(0)` that reaches models.toml
/// verbatim, and pi rejects the whole pxy provider over it ("invalid
/// contextWindow"). Reporting `None` instead lets the caller apply
/// `default_context()`, exactly as for a provider that omits the field.
fn discovered_context(rec: &Value, known: Option<&Caps>) -> Option<u64> {
    rec["context_length"]
        .as_u64()
        .or_else(|| rec["tokenLimits"]["maxInputTokens"].as_u64())
        .or_else(|| known.and_then(|c| c.context))
        .filter(|c| *c > 0)
}

fn snippet(s: &str) -> String {
    let t = s.trim().replace('\n', " ");
    if t.chars().count() > 120 {
        format!("{}…", t.chars().take(120).collect::<String>())
    } else {
        t
    }
}

/// Discover, report, and (when `write`) generate `models.toml`.
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
    let mut ungrouped: BTreeMap<String, Vec<Discovered>> = BTreeMap::new();
    let mut pools: BTreeMap<String, BTreeSet<String>> = BTreeMap::new();
    let mut found: BTreeMap<String, Vec<Discovered>> = BTreeMap::new();
    let (mut n_ok, mut n_models) = (0usize, 0usize);

    // Every "provider/model" that some group already routes to. Discovery's
    // job is to surface what ISN'T in a chain yet — a free, tool-capable model
    // nobody put in a group is the one actionable finding this command has.
    let grouped: BTreeSet<&str> = cfg
        .groups
        .values()
        .flat_map(|g| g.models.iter().map(String::as_str))
        .collect();

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
                // Free + tool-capable + in no group yet = worth adding to one.
                let all = models.clone();
                let cands: Vec<Discovered> = models
                    .into_iter()
                    .filter(|m| {
                        m.free == Tri::Yes
                            && m.tool_call == Tri::Yes
                            && !grouped.contains(format!("{name}/{}", m.id).as_str())
                    })
                    .collect();
                if !cands.is_empty() {
                    ungrouped.insert(name.clone(), cands);
                }
                // Keep the FULL list — free and paid alike: it is what gets
                // written to models.toml.
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

    // Which canonical models are served by more than one provider — the pools
    // to interleave when hand-writing a group, so one provider's 429 doesn't
    // stall the chain.
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

    let total_new: usize = ungrouped.values().map(|v| v.len()).sum();
    println!("\nfree + tool-capable, in no group ({total_new}):");
    for (prov, models) in &ungrouped {
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
        println!("\n(dry run — nothing written. Use `pxy refresh --generate` to write models.toml.)");
        return Ok(());
    }
    // Generating from degraded discovery would shrink the catalog to whatever
    // happened to work this minute, and the shrunken file would then be loaded
    // as truth. Credential failures (a locked gpg agent takes out EVERY
    // provider at once) or a high failure rate abort the write; the previous
    // models.toml stays in force.
    let cred_failures = failures
        .iter()
        .filter(|(_, why)| why.starts_with("credential:"))
        .count();
    if cred_failures > 0 || failures.len() > n_ok / 2 {
        anyhow::bail!(
            "refusing to write: {} discovery failure(s), {} credential-related \
             (locked gpg agent?). Fix access and rerun; the existing \
             models.toml is untouched.",
            failures.len(),
            cred_failures
        );
    }
    generate(cfg, found, out_path)
}

/// One generated model row — everything `models.toml` records about a model.
struct Row {
    id: String,
    context: u64,
    tool_call: Option<bool>,
    free: Option<bool>,
}

/// Build and write `models.toml`: for every enabled provider, the UNION of its
/// hand-written models and everything discovery listed — free and paid alike.
/// Groups are never touched. What actually gets routed is a hand-written
/// decision, and a generator able to rewrite a chain is a generator able to
/// spend money.
fn generate(
    cfg: &Config,
    discovered: BTreeMap<String, Vec<Discovered>>,
    out_path: &std::path::Path,
) -> Result<()> {
    let today = today();

    println!("\n── generating ──");

    let mut per_provider: BTreeMap<String, (Tier, Vec<Row>)> = BTreeMap::new();
    let mut dropped_promo: Vec<String> = Vec::new();

    for (name, pcfg) in &cfg.providers {
        if !pcfg.enabled {
            continue;
        }
        // Start from what config.toml already has. Never lose a hand-added
        // model: discovery can omit one that works.
        let mut rows: BTreeMap<String, Row> = pcfg
            .models
            .iter()
            .map(|m| {
                let s = m.spec();
                let row = Row {
                    id: s.id.clone(),
                    context: s.context_length,
                    tool_call: s.tool_call,
                    free: s.free,
                };
                (s.id, row)
            })
            .collect();

        // Drop expired promo models wherever they came from.
        let promo = pcfg.promo.as_ref().filter(|p| p.is_expired(&today));
        for id in promo.iter().flat_map(|p| &p.models) {
            if rows.remove(id).is_some() {
                dropped_promo.push(format!("{name}/{id}"));
            }
        }

        for d in discovered.get(name).into_iter().flatten() {
            if promo.is_some_and(|p| p.models.contains(&d.id)) {
                continue;
            }
            match rows.get_mut(&d.id) {
                // A hand-written entry is authoritative at load time
                // (Config::apply_generated takes the curated spec WHOLESALE),
                // so discovery may only fill in the facts it left blank.
                // Overwriting the window with the discovered one would record a
                // number the router never uses: aihubmix lists
                // coding-kimi-k3-free at 1M, the curated entry pins it to 262k,
                // and 262k is what a request gets.
                Some(row) => {
                    row.tool_call = row.tool_call.or(d.tool_call.as_opt());
                    row.free = row.free.or(d.free.as_opt());
                }
                None => {
                    rows.insert(
                        d.id.clone(),
                        Row {
                            id: d.id.clone(),
                            context: d.context.unwrap_or(crate::config::default_context()),
                            tool_call: d.tool_call.as_opt(),
                            free: d.free.as_opt(),
                        },
                    );
                }
            }
        }
        per_provider.insert(name.clone(), (pcfg.tier, rows.into_values().collect()));
    }

    let body = render_generated(&per_provider, &today);
    std::fs::write(out_path, &body)
        .with_context(|| format!("writing {}", out_path.display()))?;

    let rows = || per_provider.values().flat_map(|(_, v)| v.iter());
    let total = rows().count();
    let free = rows().filter(|r| r.free == Some(true)).count();
    if !dropped_promo.is_empty() {
        println!("dropped {} expired promo model(s):", dropped_promo.len());
        for d in &dropped_promo {
            println!("  {d}");
        }
    }
    println!(
        "wrote {} — {total} models across {} providers ({free} priced at zero). \
         Groups in config.toml are unchanged.",
        out_path.display(),
        per_provider.len(),
    );
    Ok(())
}

// ---------------------------------------------------------------------------
// Rendering
// ---------------------------------------------------------------------------

/// Today as YYYY-MM-DD, for promo expiry.
fn today() -> String {
    jiff::Zoned::now().date().to_string()
}

/// Build `models.toml`: per-provider model lists (union of hand-configured and
/// discovered), each with the facts a picker needs.
fn render_generated(per_provider: &BTreeMap<String, (Tier, Vec<Row>)>, stamp: &str) -> String {
    let mut out = String::new();
    out.push_str(&format!(
        "# AUTO-GENERATED by `pxy refresh --generate` on {stamp}.\n"
    ));
    out.push_str(concat!(
        "# Do not edit: rerun the command instead. Hand-written provider settings\n",
        "# (credentials, limits, headers, quirks) and the [groups] chains live in\n",
        "# config.toml and are never touched by generation; only model lists come\n",
        "# from here.\n",
        "# Every model each provider lists is here, free and paid alike. Model lists\n",
        "# are a UNION with config.toml: a provider's /models can omit a model that\n",
        "# works (zai/glm-4.7-flash is absent from Z.AI's own listing), and for a\n",
        "# model config.toml also declares, the hand-written spec wins wholesale.\n",
        "# `free` is a DISPLAY fact (provider pricing as discovery saw it); routing\n",
        "# never reads it. `tier` echoes config.toml so this file reads on its own.\n\n",
    ));
    for (prov, (tier, rows)) in per_provider {
        out.push_str(&format!(
            "[providers.{prov}]\ntier = \"{}\"\nmodels = [\n",
            tier.as_str()
        ));
        for r in rows {
            let mut extra = String::new();
            if let Some(t) = r.tool_call {
                extra.push_str(&format!(", tool_call = {t}"));
            }
            if let Some(f) = r.free {
                extra.push_str(&format!(", free = {f}"));
            }
            out.push_str(&format!(
                "  {{ id = \"{}\", context_length = {}{extra} }},\n",
                r.id, r.context
            ));
        }
        out.push_str("]\n\n");
    }
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
    fn zero_context_reads_as_unknown_not_as_a_window() {
        // aihubmix's gpt-image-2-free: null upstream, 0 from the caps fallback.
        // Some(0) used to reach models.toml and break pi's models.json.
        let caps = Caps {
            context: Some(0),
            ..Default::default()
        };
        assert_eq!(
            discovered_context(&json!({"id": "gpt-image-2-free", "context_length": null}), Some(&caps)),
            None
        );
        // A provider reporting 0 directly is equally meaningless.
        assert_eq!(
            discovered_context(&json!({"context_length": 0}), None),
            None
        );
        // Real windows still come through, from either spelling.
        assert_eq!(
            discovered_context(&json!({"context_length": 262144}), None),
            Some(262144)
        );
        assert_eq!(
            discovered_context(&json!({"tokenLimits": {"maxInputTokens": 131072}}), None),
            Some(131072)
        );
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

    /// Generation writes what discovery found for a NEW model, and only fills
    /// blanks on one config.toml already declares — the curated spec is
    /// authoritative at load time, so a discovered window written over a pinned
    /// one would be a number the router never uses.
    #[test]
    fn generated_rows_keep_curated_specs_and_absorb_discovery() {
        let mut rows: BTreeMap<String, Row> = BTreeMap::from([(
            "pinned".to_string(),
            Row { id: "pinned".into(), context: 262_144, tool_call: Some(true), free: None },
        )]);
        let discovered = [
            Discovered { id: "pinned".into(), canonical: "pinned".into(), free: Tri::Yes,
                         tool_call: Tri::No, context: Some(1_000_000) },
            Discovered { id: "new".into(), canonical: "new".into(), free: Tri::No,
                         tool_call: Tri::Unknown, context: Some(400_000) },
        ];
        for d in &discovered {
            match rows.get_mut(&d.id) {
                Some(row) => {
                    row.tool_call = row.tool_call.or(d.tool_call.as_opt());
                    row.free = row.free.or(d.free.as_opt());
                }
                None => {
                    rows.insert(d.id.clone(), Row {
                        id: d.id.clone(),
                        context: d.context.unwrap_or(crate::config::default_context()),
                        tool_call: d.tool_call.as_opt(),
                        free: d.free.as_opt(),
                    });
                }
            }
        }
        let pinned = &rows["pinned"];
        assert_eq!(pinned.context, 262_144, "curated window survives discovery");
        assert_eq!(pinned.tool_call, Some(true), "curated assertion survives");
        assert_eq!(pinned.free, Some(true), "blank display fact is filled in");
        // Paid models are generated too now, and an unknown capability writes
        // no key at all rather than a `false`.
        let new = &rows["new"];
        assert_eq!(new.free, Some(false));
        assert_eq!(new.tool_call, None);
    }

    #[test]
    fn render_omits_unknown_capabilities_and_stamps_the_tier() {
        let out = render_generated(
            &BTreeMap::from([(
                "p".to_string(),
                (Tier::Subscription, vec![
                    Row { id: "known".into(), context: 8192, tool_call: Some(true), free: Some(false) },
                    Row { id: "unknown".into(), context: 128_000, tool_call: None, free: None },
                ]),
            )]),
            "2026-08-29",
        );
        assert!(out.contains("tier = \"subscription\""), "{out}");
        assert!(out.contains(r#"{ id = "known", context_length = 8192, tool_call = true, free = false }"#), "{out}");
        assert!(out.contains(r#"{ id = "unknown", context_length = 128000 }"#), "{out}");
        // The generated file must parse back into the overlay struct.
        let parsed: crate::config::Generated = toml::from_str(&out).unwrap();
        assert_eq!(parsed.providers["p"].models.len(), 2);
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
