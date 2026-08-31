//! `pxy doctor` — prove the installation works (config, daemon, credentials,
//! agent binaries), and `pxy explain <model>` — why each candidate of a model
//! id would or wouldn't be routed to right now.
//!
//! Doctor's rule (docs/09 §6.2): prove a credential by RESOLVING it (a real
//! `pass show`), not by checking that a key is configured. No live provider
//! calls — probes must not burn quota or mutate routing state.

use anyhow::Result;
use jiff::Timestamp;

use crate::catalog::Catalog;
use crate::config::Config;
use crate::secrets::Secrets;
use crate::state::State;

fn state() -> Result<State> {
    State::open(&crate::config::data_dir().join("state.sqlite"))
}

// ---------------------------------------------------------------------------
// pxy explain <model>
// ---------------------------------------------------------------------------

pub fn explain(cfg: &Config, requested: &str, json: bool) -> Result<()> {
    let catalog = Catalog::from_config(cfg);
    // Persisted cooldowns rehydrate at open; rpm windows are daemon-memory
    // only and reported as unknown.
    let st = state()?;
    // Same resolution the daemon runs, pin included: explain must describe
    // the walk a request would actually take, not the config-only chain.
    let candidates = crate::router::resolve_candidates(&catalog, cfg, &st, requested, None);
    let pin = st
        .kv_get(crate::router::ROUTE_PIN_KEY)
        .ok()
        .flatten()
        .filter(|p| !p.is_empty());
    let pinned_ids: std::collections::HashSet<String> = pin
        .as_deref()
        .map(|p| catalog.resolve(cfg, p).iter().map(|c| c.full_id()).collect())
        .unwrap_or_default();
    let is_group = catalog.is_group(requested);
    // A stale pin (resolve_candidates ignored it) must not be reported as
    // steering the walk: when active, the pin leads — so check the head.
    let pin_active = is_group
        && candidates.first().is_some_and(|c| pinned_ids.contains(&c.full_id()));
    if candidates.is_empty() {
        if json {
            println!(
                "{}",
                serde_json::json!({"requested": requested, "routePin": pin, "routePinActive": false, "candidates": []})
            );
        } else {
            println!("'{requested}' resolves to nothing (not in any provider's model list).");
        }
        return Ok(());
    }
    let default_limits = crate::config::Limits::default();
    let now = Timestamp::now();

    let mut records: Vec<serde_json::Value> = Vec::new();
    if !json {
        match (&pin, pin_active) {
            (Some(p), true) => println!(
                "'{requested}' -> {} candidate(s) (pin '{p}' first), walked in order:\n",
                candidates.len()
            ),
            (Some(p), false) if is_group => println!(
                "'{requested}' -> {} candidate(s); pin '{p}' is STALE (not in the catalog), walked in chain order:\n",
                candidates.len()
            ),
            _ => println!("'{requested}' -> {} candidate(s), walked in order:\n", candidates.len()),
        }
    }
    for (i, cand) in candidates.iter().enumerate() {
        let mut skips: Vec<String> = Vec::new();
        let mut notes: Vec<String> = Vec::new();

        let provider = cfg.providers.get(&cand.provider);
        match provider {
            Some(p) if p.enabled => {
                let limits = p.limits.as_ref().unwrap_or(&default_limits);
                if let Some(cd) = st.cooldown(&cand.state_provider(), &cand.model.id) {
                    let left = cd.until.saturating_duration_since(std::time::Instant::now());
                    skips.push(format!(
                        "cooldown: {} ({}s left{})",
                        cd.reason,
                        left.as_secs(),
                        if cd.retryable { "" } else { ", non-retryable" },
                    ));
                }
                if let Ok(w) = crate::usage::current_windows(limits, now) {
                    let day = st.usage(&cand.state_provider(), "day", w.day_start).unwrap_or_default();
                    let month =
                        st.usage(&cand.state_provider(), "month", w.month_start).unwrap_or_default();
                    let mut gate = |used: u64, cap: Option<u64>, what: &str| match cap {
                        Some(c) if used >= c => skips.push(format!("{what}: {used}/{c} EXHAUSTED")),
                        Some(c) => notes.push(format!("{what}: {used}/{c}")),
                        None => {}
                    };
                    gate(day.requests, limits.daily_requests, "daily requests");
                    gate(day.tokens, limits.daily_tokens, "daily tokens");
                    gate(month.requests, limits.monthly_requests, "monthly requests");
                    gate(month.tokens, limits.monthly_tokens, "monthly tokens");
                    if limits.total_requests.is_some() || limits.total_tokens.is_some() {
                        let total = st.usage_total(&cand.state_provider()).unwrap_or_default();
                        gate(total.requests, limits.total_requests, "total requests");
                        gate(total.tokens, limits.total_tokens, "total tokens");
                    }
                }
                if let Some(rpm) = limits.rpm {
                    notes.push(format!("rpm cap {rpm} (live window unknown outside the daemon)"));
                }
                if cand.model.tool_call == Some(false) {
                    notes.push("tool_call=false: skipped for tools requests in a group".into());
                }
                notes.push(format!(
                    "ctx {}k, max_out {}k{}",
                    cand.model.context_length / 1000,
                    cand.model.max_output_tokens / 1000,
                    match cand.model.free {
                        Some(true) => ", free",
                        Some(false) => ", paid",
                        None => "",
                    }
                ));
            }
            Some(_) => skips.push("provider disabled".into()),
            None => skips.push("provider missing from config".into()),
        }

        let eligible = skips.is_empty();
        let pinned = pin_active && pinned_ids.contains(&cand.full_id());
        if json {
            records.push(serde_json::json!({
                "position": i + 1,
                "id": cand.full_id(),
                "provider": cand.provider,
                "account": cand.account,
                "model": cand.model.id,
                "free": cand.model.free,
                "pinned": pinned,
                "eligible": eligible,
                "skips": skips,
                "notes": notes,
            }));
            continue;
        }
        let verdict = if eligible { "ELIGIBLE" } else { "would skip" };
        let pin_mark = if pinned { "  [pinned]" } else { "" };
        let acct = cand
            .account
            .as_ref()
            .map(|a| format!("  [account {a}]"))
            .unwrap_or_default();
        println!("{:>2}. {}{acct}  [{verdict}]{pin_mark}", i + 1, cand.full_id());
        for s in &skips {
            println!("      ✗ {s}");
        }
        for n in &notes {
            println!("      · {n}");
        }
    }
    if json {
        println!(
            "{}",
            serde_json::json!({"requested": requested, "routePin": pin, "routePinActive": pin_active, "candidates": records})
        );
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// pxy doctor
// ---------------------------------------------------------------------------

struct Report {
    fails: u32,
    warns: u32,
}

impl Report {
    fn ok(&mut self, name: &str, detail: impl std::fmt::Display) {
        println!("  ok   {name} — {detail}");
    }
    fn warn(&mut self, name: &str, detail: impl std::fmt::Display) {
        self.warns += 1;
        println!("  warn {name} — {detail}");
    }
    fn fail(&mut self, name: &str, detail: impl std::fmt::Display) {
        self.fails += 1;
        println!("  FAIL {name} — {detail}");
    }
}

pub async fn doctor(cfg_path: &std::path::Path) -> Result<()> {
    let mut r = Report { fails: 0, warns: 0 };
    println!("pxy doctor\n");

    // 1. Config parses.
    let cfg = match Config::load(cfg_path) {
        Ok(c) => {
            r.ok("config", format!("{} parses, {} providers", cfg_path.display(), c.providers.len()));
            Some(c)
        }
        Err(e) => {
            r.fail("config", format!("{e:#}"));
            None
        }
    };

    // 2. State db opens (also rehydrates cooldowns).
    match state() {
        Ok(st) => {
            let cooling = st.active_cooldowns().len();
            r.ok("state", format!("sqlite opens; {cooling} active cooldown(s)"));
        }
        Err(e) => r.fail("state", format!("{e:#}")),
    }

    // 3. Daemon answering?
    if let Some(cfg) = &cfg {
        let url = format!("http://127.0.0.1:{}/healthz", cfg.server.port);
        let http = reqwest::Client::new();
        match http
            .get(&url)
            .timeout(std::time::Duration::from_secs(2))
            .send()
            .await
        {
            Ok(resp) if resp.status().is_success() => {
                // Model count through the daemon proves the catalog loaded.
                let models = http
                    .get(format!("http://127.0.0.1:{}/v1/models", cfg.server.port))
                    .timeout(std::time::Duration::from_secs(3))
                    .send()
                    .await
                    .ok();
                let count = match models {
                    Some(m) => m
                        .json::<serde_json::Value>()
                        .await
                        .ok()
                        .and_then(|v| v["data"].as_array().map(|a| a.len()))
                        .unwrap_or(0),
                    None => 0,
                };
                r.ok("daemon", format!("healthz ok on :{}, {count} models exposed", cfg.server.port));
            }
            Ok(resp) => r.fail("daemon", format!("healthz answered {}", resp.status())),
            Err(_) => r.fail(
                "daemon",
                format!("not answering on :{} — systemctl --user restart pxy", cfg.server.port),
            ),
        }

        // 4. Credentials actually RESOLVE (pass show / file read). A locked
        // gpg agent fails every pass-backed provider at once — that pattern
        // IS the diagnosis.
        let secrets = Secrets::new();
        let mut resolved = 0u32;
        for (name, p) in cfg.providers.iter().filter(|(_, p)| p.enabled) {
            let sref = p.credentials.as_ref().or(p.api_key.as_ref());
            match sref {
                None => {} // keyless providers exist (opencode-zen)
                Some(sref) => match secrets.resolve_key(sref) {
                    Ok(k) if !k.is_empty() => resolved += 1,
                    Ok(_) => r.fail(format!("cred {name}").as_str(), "resolved EMPTY"),
                    Err(e) => r.fail(format!("cred {name}").as_str(), format!("{e:#}")),
                },
            }
        }
        r.ok("credentials", format!("{resolved} provider credential(s) resolve"));

        // 5. Providers that will serve nothing. config.toml is the whole
        // catalog, so a provider with an empty `models` list is simply
        // invisible — no error anywhere, it just never shows up.
        let mute: Vec<&str> = cfg
            .providers
            .iter()
            .filter(|(name, p)| {
                p.enabled
                    && cfg.provider_allowed(name)
                    && p.models.is_empty()
                    && p.embedding_models.is_empty()
                    && p.media.is_none()
            })
            .map(|(name, _)| name.as_str())
            .collect();
        if mute.is_empty() {
            r.ok("models", "every enabled provider declares models in config.toml");
        } else {
            r.warn(
                "models",
                format!(
                    "no models in config.toml, so nothing is served: {} — \
                     `pxy refresh --generate` lists what each one offers",
                    mute.join(", ")
                ),
            );
        }

        // 6. models.toml is a REPORT — pxy never loads it, so its age only
        // says how stale the list you copy rows from is.
        let gen_path = crate::config::models_path(cfg_path);
        match std::fs::metadata(&gen_path).and_then(|m| m.modified()) {
            Ok(t) => {
                let age = t.elapsed().map(|d| d.as_secs() / 86_400).unwrap_or(0);
                r.ok("catalog report", format!("models.toml {age} day(s) old (reference only)"));
            }
            Err(_) => r.ok("catalog report", "no models.toml — `pxy refresh --generate` writes one"),
        }
    }

    // 7. Agent binaries on PATH.
    for agent in ["claude", "opencode", "pi", "codex"] {
        if on_path(agent) {
            r.ok(format!("agent {agent}").as_str(), "on PATH");
        } else {
            r.warn(format!("agent {agent}").as_str(), "not found on PATH");
        }
    }

    println!();
    if r.fails > 0 {
        anyhow::bail!("{} check(s) FAILED, {} warning(s)", r.fails, r.warns);
    }
    println!("all checks passed ({} warning(s))", r.warns);
    Ok(())
}

fn on_path(bin: &str) -> bool {
    std::env::var_os("PATH")
        .map(|paths| {
            std::env::split_paths(&paths).any(|dir| {
                let p = dir.join(bin);
                p.is_file()
            })
        })
        .unwrap_or(false)
}
