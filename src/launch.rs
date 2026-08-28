//! `pxy launch <agent>` — wire a coding agent to the local proxy.
//! Mechanisms verified per agent in docs/02-agent-wiring.md:
//! claude = env vars only; opencode = OPENCODE_CONFIG_CONTENT inline JSON;
//! pi = additive merge into ~/.pi/agent/models.json.

use std::os::unix::process::CommandExt;
use std::process::Command;

use anyhow::{Context, Result};
use serde_json::{json, Map, Value};

use crate::catalog::Catalog;
use crate::config::Config;

pub fn launch(
    cfg: &Config,
    agent: &str,
    model: Option<&str>,
    dry_run: bool,
    extra_args: &[String],
) -> Result<()> {
    let catalog = Catalog::from_config(cfg);
    let model = model.map(String::from).unwrap_or_else(|| cfg.default_route());
    if model.is_empty() {
        anyhow::bail!("no model to launch with: pass --model, or set [launch] model in config.toml");
    }

    match agent {
        "claude" => launch_claude(cfg, &catalog, &model, dry_run, extra_args),
        "opencode" => launch_opencode(cfg, &catalog, &model, dry_run, extra_args),
        "pi" => launch_pi(cfg, &catalog, &model, dry_run, extra_args),
        "codex" => launch_codex(cfg, &model, dry_run, extra_args),
        "fx" => launch_fx(cfg, &model, dry_run, extra_args),
        other => {
            anyhow::bail!("unknown agent '{other}' (supported: claude, opencode, pi, codex, fx)")
        }
    }
}

/// The api key with the agent's name smuggled on as a `:agent` suffix. The
/// server never validates the key (soft gate, loopback only) but does parse
/// the suffix back out in client_ctx(), which is how per-model usage stats
/// know WHICH agent asked for a group. One mechanism for every agent — they
/// all send the key, while only some can be taught a custom header.
fn tagged_key(cfg: &Config, agent: &str) -> String {
    format!("{}:{agent}", cfg.server.api_key)
}

fn exec_or_print(mut cmd: Command, dry_run: bool, note: &str) -> Result<()> {
    if dry_run {
        println!("would exec: {:?}", cmd.get_program());
        let args: Vec<_> = cmd.get_args().map(|a| a.to_string_lossy().into_owned()).collect();
        if !args.is_empty() {
            println!("  args: {args:?}");
        }
        // Env var NAMES only — never values (may hold tokens).
        let envs: Vec<_> = cmd
            .get_envs()
            .filter_map(|(k, v)| v.map(|_| k.to_string_lossy().into_owned()))
            .collect();
        println!("  env set: {envs:?}");
        if !note.is_empty() {
            println!("  {note}");
        }
        return Ok(());
    }
    // exec() replaces this process: signals, exit codes, terminal all belong
    // to the agent — no forwarding machinery needed.
    let err = cmd.exec();
    Err(anyhow::Error::new(err).context("exec failed (is the agent installed?)"))
}

// ---------------------------------------------------------------------------
// Claude Code
// ---------------------------------------------------------------------------

fn launch_claude(
    cfg: &Config,
    catalog: &Catalog,
    model: &str,
    dry_run: bool,
    extra_args: &[String],
) -> Result<()> {
    let mut cmd = Command::new("claude");
    // Claude Code applies a settings file's `env` block OVER the process
    // environment, so a user-level CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC
    // re-disables the model discovery enabled below no matter what pxy puts in
    // the child env. The --settings scope outranks user settings and an empty
    // value reads as unset. Before extra_args so a caller's own --settings wins.
    cmd.arg("--settings")
        .arg(r#"{"env":{"CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC":""}}"#);
    cmd.args(extra_args);

    // Delete every inherited ANTHROPIC_* var: a stale shell token must not
    // shadow the injected one (OmniRoute buildClaudeEnv).
    for (key, _) in std::env::vars() {
        if key.starts_with("ANTHROPIC_") {
            cmd.env_remove(&key);
        }
    }

    // Base URL WITHOUT /v1 — Claude Code appends /v1/messages itself.
    cmd.env("ANTHROPIC_BASE_URL", cfg.base_url());
    // Must be non-empty or Claude Code stops at its login gate.
    cmd.env("ANTHROPIC_AUTH_TOKEN", tagged_key(cfg, "claude"));
    cmd.env("ANTHROPIC_MODEL", model);
    let small = cfg.launch.small_model.clone().unwrap_or_else(|| model.to_string());
    cmd.env("ANTHROPIC_DEFAULT_HAIKU_MODEL", &small);
    // NOT CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC: it also switches off
    // gateway model discovery (docs/en/llm-gateway-protocol, "When discovery
    // runs"), which would cancel the flag below — inherited copies too, hence
    // the removal. These two are the telemetry half of it, all pxy wants off.
    cmd.env_remove("CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC");
    cmd.env("DISABLE_TELEMETRY", "1");
    cmd.env("DISABLE_ERROR_REPORTING", "1");
    // In-session /model switching across every pxy provider: the picker
    // reads /v1/models, which mirrors all ids under a "claude/" prefix.
    cmd.env("CLAUDE_CODE_ENABLE_GATEWAY_MODEL_DISCOVERY", "1");

    // Claude Code assumes a 200K context for model ids it doesn't recognize —
    // wrong in both directions, and it says so at startup. Declare the real
    // window instead (docs/en/model-config, "Correct the window for a gateway
    // or custom model ID"), which also silences that warning. min() over the
    // chain: any member may serve the request, so the declared window has to be
    // one they all satisfy. The compact bound keeps 5% of it as headroom.
    let min_ctx = catalog
        .resolve(cfg, model)
        .iter()
        .map(|c| c.model.context_length)
        .min();
    if let Some(ctx) = min_ctx {
        cmd.env("CLAUDE_CODE_MAX_CONTEXT_TOKENS", ctx.to_string());
        cmd.env("CLAUDE_CODE_AUTO_COMPACT_WINDOW", ((ctx * 95) / 100).to_string());
    }

    exec_or_print(cmd, dry_run, "claude wired via ANTHROPIC_* env vars")
}

// ---------------------------------------------------------------------------
// opencode
// ---------------------------------------------------------------------------

fn launch_opencode(
    cfg: &Config,
    catalog: &Catalog,
    model: &str,
    dry_run: bool,
    extra_args: &[String],
) -> Result<()> {
    let mut models_map = Map::new();
    for (name, group) in catalog.groups() {
        let (ctx, max_out) = crate::catalog::chain_limits(&group.chain);
        models_map.insert(
            name.clone(),
            json!({"name": group.label, "limit": {"context": ctx, "output": max_out}}),
        );
    }
    for cand in catalog.models() {
        models_map.insert(
            cand.full_id(),
            json!({
                "name": cand.model.name.clone().unwrap_or_else(|| cand.full_id()),
                "limit": {
                    "context": cand.model.context_length,
                    "output": cand.model.max_output_tokens,
                },
            }),
        );
    }

    let config_content = json!({
        "$schema": "https://opencode.ai/config.json",
        "provider": {
            "pxy": {
                "npm": "@ai-sdk/openai-compatible",
                "name": "pxy",
                "options": {
                    "baseURL": format!("{}/v1", cfg.base_url()),
                    // env indirection keeps the key out of the serialized JSON
                    "apiKey": "{env:PXY_API_KEY}",
                },
                "models": models_map,
            }
        },
        "model": format!("pxy/{model}"),
    });

    let mut cmd = Command::new("opencode");
    cmd.args(extra_args);
    cmd.env_remove("OPENCODE_CONFIG_CONTENT");
    cmd.env("OPENCODE_CONFIG_CONTENT", config_content.to_string());
    cmd.env("PXY_API_KEY", tagged_key(cfg, "opencode"));

    exec_or_print(cmd, dry_run, "opencode wired via OPENCODE_CONFIG_CONTENT")
}

// ---------------------------------------------------------------------------
// codex
// ---------------------------------------------------------------------------

/// codex is wired entirely through `-c` config overrides (parsed as TOML), so
/// ~/.codex/config.toml is never touched. wire_api = "responses" points it at
/// pxy's /v1/responses endpoint.
fn launch_codex(cfg: &Config, model: &str, dry_run: bool, extra_args: &[String]) -> Result<()> {
    let mut cmd = Command::new("codex");
    for (k, v) in [
        ("model_provider", "\"pxy\"".to_string()),
        ("model_providers.pxy.name", "\"pxy\"".to_string()),
        (
            "model_providers.pxy.base_url",
            format!("\"{}/v1\"", cfg.base_url()),
        ),
        ("model_providers.pxy.env_key", "\"PXY_API_KEY\"".to_string()),
        ("model_providers.pxy.wire_api", "\"responses\"".to_string()),
    ] {
        cmd.arg("-c").arg(format!("{k}={v}"));
    }
    cmd.arg("-m").arg(model);
    cmd.args(extra_args);
    cmd.env("PXY_API_KEY", tagged_key(cfg, "codex"));

    exec_or_print(cmd, dry_run, "codex wired via -c model_providers.pxy overrides")
}

// ---------------------------------------------------------------------------
// fx (vercel-labs/fx)
// ---------------------------------------------------------------------------

/// fx talks to Vercel's AI Gateway in the AI SDK LanguageModel dialect; pxy
/// serves that at /v3/ai/language-model (translate/aisdk).
///
/// Two overrides are needed, not one: `FX_GATEWAY_BASE_URL` redirects the
/// catalog/credits GETs, while the generation POST reads its own
/// `FX_GATEWAY_CHAT_URL`. fx silently ignores either unless the URL is
/// loopback HTTP with an explicit port (the base URL carries the bearer
/// token), which pxy's 127.0.0.1:<port> satisfies.
///
/// `AI_GATEWAY_API_KEY` short-circuits fx's credential chain: no Vercel
/// login, no token refresh, no team lookup — zero traffic leaves the machine.
fn launch_fx(cfg: &Config, model: &str, dry_run: bool, extra_args: &[String]) -> Result<()> {
    let mut cmd = Command::new("fx");
    // A stale Vercel session would otherwise outrank the injected key.
    for (key, _) in std::env::vars() {
        if key.starts_with("FX_") || key.starts_with("AI_GATEWAY_") || key == "VERCEL_OIDC_TOKEN" {
            cmd.env_remove(&key);
        }
    }
    cmd.env("AI_GATEWAY_API_KEY", tagged_key(cfg, "fx"));
    cmd.env("FX_GATEWAY_BASE_URL", cfg.base_url());
    cmd.env("FX_GATEWAY_CHAT_URL", format!("{}/v3/ai/language-model", cfg.base_url()));
    cmd.env("FX_MODEL", model);
    cmd.args(extra_args);

    exec_or_print(cmd, dry_run, "fx wired via FX_GATEWAY_* + AI_GATEWAY_API_KEY")
}

// ---------------------------------------------------------------------------
// pi
// ---------------------------------------------------------------------------

fn launch_pi(
    cfg: &Config,
    catalog: &Catalog,
    model: &str,
    dry_run: bool,
    extra_args: &[String],
) -> Result<()> {
    let models_path = crate::config::home_dir().join(".pi/agent/models.json");
    merge_pi_models(cfg, catalog, &models_path, dry_run)?;

    let mut cmd = Command::new("pi");
    cmd.arg("--provider").arg("pxy").arg("--model").arg(model);
    cmd.args(extra_args);
    cmd.env("PXY_API_KEY", tagged_key(cfg, "pi"));

    exec_or_print(
        cmd,
        dry_run,
        &format!("pi provider merged into {}", models_path.display()),
    )
}

/// Idempotent additive merge: only the `providers.pxy` key is touched;
/// everything else in the user's models.json is preserved.
fn merge_pi_models(
    cfg: &Config,
    catalog: &Catalog,
    path: &std::path::Path,
    dry_run: bool,
) -> Result<()> {
    let mut root: Value = match std::fs::read_to_string(path) {
        Ok(s) => serde_json::from_str(&s)
            .with_context(|| format!("{} is not valid JSON", path.display()))?,
        Err(_) => json!({}),
    };

    let mut models: Vec<Value> = Vec::new();
    for (name, group) in catalog.groups() {
        let (ctx, max_out) = crate::catalog::chain_limits(&group.chain);
        models.push(json!({
            "id": name,
            "name": group.label,
            "reasoning": false,
            "input": ["text"],
            "contextWindow": ctx,
            "maxTokens": max_out,
            "cost": {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0},
        }));
    }
    for cand in catalog.models() {
        models.push(json!({
            "id": cand.full_id(),
            "name": cand.model.name.clone().unwrap_or_else(|| cand.full_id()),
            "reasoning": false,
            "input": ["text"],
            "contextWindow": cand.model.context_length,
            "maxTokens": cand.model.max_output_tokens,
            "cost": {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0},
        }));
    }

    let provider = json!({
        "baseUrl": format!("{}/v1", cfg.base_url()),
        "api": "openai-completions",
        "apiKey": "$PXY_API_KEY",
        "models": models,
    });

    if !root.is_object() {
        root = json!({});
    }
    if !root["providers"].is_object() {
        root["providers"] = json!({});
    }
    root["providers"]["pxy"] = provider;

    if dry_run {
        println!("would write providers.pxy into {}", path.display());
        return Ok(());
    }
    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir)?;
    }
    std::fs::write(path, serde_json::to_string_pretty(&root)?)
        .with_context(|| format!("writing {}", path.display()))?;
    Ok(())
}
