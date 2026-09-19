//! `pxy launch <agent>` — wire a coding agent to the local proxy.
//! Mechanisms verified per agent (wiki:launch):
//! claude = env vars only; opencode = OPENCODE_CONFIG_CONTENT inline JSON;
//! pi = an extension in ~/.pi/agent/extensions that registers the provider.

use std::os::unix::process::CommandExt;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{Duration, Instant};

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
        "pi" => launch_pi(cfg, &model, dry_run, extra_args),
        "codex" => launch_codex(cfg, &catalog, &model, dry_run, extra_args),
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
    // Claude Code switches tool search off when ANTHROPIC_BASE_URL is not a
    // first-party Anthropic host (litellm cli/commands/agents.py, read
    // 2026-09-19), which would silently disable the tool_search pxy serves
    // (wiki:tool-search). An inherited value is the user's call.
    if std::env::var_os("ENABLE_TOOL_SEARCH").is_none() {
        cmd.env("ENABLE_TOOL_SEARCH", "true");
    }

    // Claude Code assumes a 200K context for model ids it doesn't recognize —
    // wrong in both directions, and it says so at startup. Declare the real
    // window instead (docs/en/model-config, "Correct the window for a gateway
    // or custom model ID"), which also silences that warning. min() over the
    // chain: any member may serve the request, so the declared window has to be
    // one they all satisfy.
    //
    // Deliberately NOT CLAUDE_CODE_AUTO_COMPACT_WINDOW: it is the first branch
    // of resolveAutoCompactWindow, so it pins ONE window for the entire session
    // and no in-session /model switch can widen it — that is what froze a 1M
    // model at the launch group's window. Claude Code already subtracts its own
    // summary buffer, so reserving headroom here only double-discounted.
    // Per-model windows after a switch come from the "[1m]" marker instead
    // (server::models).
    let min_ctx = catalog
        .resolve(cfg, model)
        .iter()
        .map(|c| c.model.context_length)
        .min();
    if let Some(ctx) = min_ctx {
        cmd.env("CLAUDE_CODE_MAX_CONTEXT_TOKENS", ctx.to_string());
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
fn launch_codex(
    cfg: &Config,
    catalog: &Catalog,
    model: &str,
    dry_run: bool,
    extra_args: &[String],
) -> Result<()> {
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
    // -m picks the model to start on; the picker behind /model is a separate
    // list codex ships, so without this it offers OpenAI's slugs alone.
    let mut note = "codex wired via -c model_providers.pxy overrides".to_string();
    match codex_catalog_file(cfg, catalog, model) {
        Ok(path) => {
            cmd.arg("-c").arg(codex_catalog_override(&path));
            note.push_str(&format!("; /model picker from {}", path.display()));
        }
        Err(e) => eprintln!("pxy: /model shows codex's own models: {e:#}"),
    }
    cmd.arg("-m").arg(model);
    cmd.args(extra_args);
    cmd.env("PXY_API_KEY", tagged_key(cfg, "codex"));

    exec_or_print(cmd, dry_run, &note)
}

/// One row pxy wants in codex's picker: a group name or a `provider/model`
/// id, the label the other launchers show for it, and the window it serves.
pub struct CodexEntry {
    pub id: String,
    pub name: String,
    pub window: u64,
}

/// Every id pxy serves, in the order launch_opencode lists them: the groups
/// first, then every provider/model.
fn codex_entries(catalog: &Catalog) -> Vec<CodexEntry> {
    let mut entries = Vec::new();
    for (name, group) in catalog.groups() {
        let (ctx, _) = crate::catalog::chain_limits(&group.chain);
        entries.push(CodexEntry { id: name.clone(), name: group.label.clone(), window: ctx });
    }
    for cand in catalog.models() {
        entries.push(CodexEntry {
            id: cand.full_id(),
            name: cand.model.name.clone().unwrap_or_else(|| cand.full_id()),
            window: cand.model.context_length,
        });
    }
    entries
}

/// pxy's catalog in the shape codex reads from `model_catalog_json`. A stock
/// entry is reused whole when the slug matches, so codex keeps everything it
/// knows about that model; every other id is cloned off the first stock entry,
/// which is how base_instructions and the tool flags stay codex's own. Only
/// context_window carries pxy's number — max_context_window is left as the
/// template has it, because codex reads the pair together.
fn codex_catalog(stock: &Value, entries: &[CodexEntry]) -> Value {
    let stock_models = stock["models"].as_array().map(Vec::as_slice).unwrap_or_default();
    let Some(template) = stock_models.first().filter(|t| t.is_object()) else {
        return json!({ "models": [] });
    };
    let models: Vec<Value> = entries
        .iter()
        .enumerate()
        .map(|(i, entry)| {
            let mut row = match stock_models.iter().find(|m| m["slug"] == entry.id.as_str()) {
                Some(stock) => stock.clone(),
                None => {
                    let mut row = template.clone();
                    row["slug"] = json!(entry.id);
                    row["display_name"] = json!(entry.name);
                    row["context_window"] = json!(entry.window);
                    // No effort picker for a pxy id: the chain decides, and the
                    // "-high" suffix ids route on their own.
                    row["supported_reasoning_levels"] = json!([]);
                    // The template's retirement notice and first-run blurb
                    // belong to its model, not to this one.
                    row["upgrade"] = Value::Null;
                    row["availability_nux"] = Value::Null;
                    row
                }
            };
            row["priority"] = json!(i + 1);
            row["visibility"] = json!("list");
            row
        })
        .collect();
    json!({ "models": models })
}

fn codex_home() -> PathBuf {
    std::env::var_os("CODEX_HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|| crate::config::home_dir().join(".codex"))
}

fn codex_catalog_override(file: &Path) -> String {
    format!("model_catalog_json=\"{}\"", file.display())
}

/// Write pxy's catalog where codex can read it and hand the path back — but
/// only after codex has read the file itself and listed the launch model. A
/// file codex rejects would leave the picker empty and the launch model with
/// it, so a preflight that fails anywhere is worth less than doing nothing.
fn codex_catalog_file(cfg: &Config, catalog: &Catalog, model: &str) -> Result<PathBuf> {
    let stock = codex_debug_models(cfg, None).context("reading codex's own catalog")?;
    let built = codex_catalog(&stock, &codex_entries(catalog));
    let path = codex_home().join("pxy-models.json");
    crate::config::write_atomic(&path, &serde_json::to_vec(&built)?)
        .with_context(|| format!("writing {}", path.display()))?;
    let read_back = codex_debug_models(cfg, Some(&path)).context("reading the catalog back")?;
    let listed = read_back["models"]
        .as_array()
        .is_some_and(|models| models.iter().any(|m| m["slug"] == model));
    anyhow::ensure!(listed, "codex read {} without '{model}' in it", path.display());
    Ok(path)
}

/// codex 0.147.0 prints its catalog with `codex debug models` and replaces it
/// with `-c model_catalog_json=<path>` (probed 2026-09-19). Neither invocation
/// touches the network, so the only way this hangs is codex itself.
const CODEX_PROBE_TIMEOUT: Duration = Duration::from_secs(10);

fn codex_debug_models(cfg: &Config, catalog_file: Option<&Path>) -> Result<Value> {
    let out = std::env::temp_dir().join(format!("pxy-codex-models.{}.json", std::process::id()));
    let models = run_codex_debug_models(cfg, catalog_file, &out);
    let _ = std::fs::remove_file(&out);
    models
}

fn run_codex_debug_models(cfg: &Config, catalog_file: Option<&Path>, out: &Path) -> Result<Value> {
    let mut cmd = Command::new("codex");
    if let Some(file) = catalog_file {
        cmd.arg("-c").arg(codex_catalog_override(file));
    }
    cmd.arg("debug").arg("models");
    cmd.env("PXY_API_KEY", tagged_key(cfg, "codex"));
    // The catalog runs to a few hundred KB — past a pipe's buffer, and nothing
    // drains that pipe until the child exits, so stdout goes to a file.
    cmd.stdout(std::fs::File::create(out).with_context(|| format!("creating {}", out.display()))?);
    cmd.stderr(std::process::Stdio::null());
    let mut child = cmd.spawn().context("running codex debug models")?;

    let deadline = Instant::now() + CODEX_PROBE_TIMEOUT;
    let status = loop {
        if let Some(status) = child.try_wait()? {
            break status;
        }
        if Instant::now() >= deadline {
            let _ = child.kill();
            let _ = child.wait();
            anyhow::bail!("codex debug models ran past {}s", CODEX_PROBE_TIMEOUT.as_secs());
        }
        std::thread::sleep(Duration::from_millis(25));
    };
    anyhow::ensure!(status.success(), "codex debug models exited with {status}");
    Ok(serde_json::from_slice(&std::fs::read(out)?)?)
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

fn launch_pi(cfg: &Config, model: &str, dry_run: bool, extra_args: &[String]) -> Result<()> {
    let ext_path = crate::config::home_dir().join(".pi/agent/extensions/pxy.ts");
    install_pi_extension(cfg, &ext_path, dry_run)?;

    let mut cmd = Command::new("pi");
    cmd.arg("--provider").arg("pxy").arg("--model").arg(model);
    cmd.args(extra_args);

    exec_or_print(cmd, dry_run, &format!("pi provider registered by {}", ext_path.display()))
}

/// The extension is pi's whole view of pxy, so it is rewritten on every launch
/// — an edit to the installed copy goes, and a moved port or a reinstalled
/// binary heals by itself. Only this one file is touched: the catalog it
/// registers is read live from `pxy models --json`, so nothing about the model
/// list is baked in here and models.json stays the user's alone.
///
/// (An earlier version merged `providers.pxy` into ~/.pi/agent/models.json.
/// Delete that key if it is still there: models.json overrides compose ABOVE
/// registered providers, so a stale copy silently shadows this one.)
fn install_pi_extension(cfg: &Config, path: &std::path::Path, dry_run: bool) -> Result<()> {
    let pxy_bin = std::env::current_exe()
        .context("locating the running pxy binary for the pi extension")?;
    let source = include_str!("../contrib/pi-pxy.ts")
        .replace("__PXY_BASE_URL__", &format!("{}/v1", cfg.base_url()))
        .replace("__PXY_BIN__", &pxy_bin.to_string_lossy());

    if dry_run {
        println!("would write {}", path.display());
        return Ok(());
    }
    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir)?;
    }
    crate::config::write_atomic(path, source.as_bytes())
        .with_context(|| format!("writing {}", path.display()))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `codex debug models` as codex 0.147.0 prints it, three entries kept and
    /// the long instruction strings trimmed.
    fn stock() -> serde_json::Value {
        serde_json::from_str(include_str!("../tests/fixtures/codex-debug-models.json")).unwrap()
    }

    #[test]
    fn codex_catalog_reuses_stock_and_clones_unknown() {
        let entries = vec![
            CodexEntry { id: "aaa".into(), name: "aaa chain".into(), window: 200_000 },
            CodexEntry { id: "gpt-5.5".into(), name: "pxy name".into(), window: 12 },
            CodexEntry { id: "gpt-5.4".into(), name: "pxy name".into(), window: 12 },
        ];
        let built = codex_catalog(&stock(), &entries);
        let models = built["models"].as_array().unwrap();
        assert_eq!(models.len(), 3);

        // Unknown slug: cloned off the first stock entry, so codex's own
        // instructions and tool flags come along untouched.
        let cloned = &models[0];
        assert_eq!(cloned["slug"], "aaa");
        assert_eq!(cloned["display_name"], "aaa chain");
        assert_eq!(cloned["context_window"], 200_000);
        assert_eq!(cloned["base_instructions"], stock()["models"][0]["base_instructions"]);
        assert_eq!(cloned["apply_patch_tool_type"], stock()["models"][0]["apply_patch_tool_type"]);
        // max_context_window stays the template's (ruling 7), the picker's
        // effort list goes, and neither an upgrade notice nor a first-run
        // message belongs on a pxy id.
        assert_eq!(cloned["max_context_window"], stock()["models"][0]["max_context_window"]);
        assert_eq!(cloned["supported_reasoning_levels"], serde_json::json!([]));
        assert_eq!(cloned["upgrade"], serde_json::Value::Null);
        assert_eq!(cloned["availability_nux"], serde_json::Value::Null);

        // Matching slug: the stock entry itself, its own name and window kept.
        let reused = &models[1];
        assert_eq!(reused["slug"], "gpt-5.5");
        assert_eq!(reused["display_name"], "GPT-5.5");
        assert_eq!(reused["context_window"], 272_000);
        assert_eq!(reused["supported_reasoning_levels"].as_array().unwrap().len(), 4);

        // Every row is listed, in pxy's order — a stock entry codex hides
        // (gpt-5.4) included, or a pxy id would be missing from the picker.
        assert_eq!(cloned["priority"], 1);
        assert_eq!(reused["priority"], 2);
        assert_eq!(cloned["visibility"], "list");
        assert_eq!(reused["visibility"], "list");
        assert_eq!(models[2]["slug"], "gpt-5.4");
        assert_eq!(models[2]["visibility"], "list");
    }

    #[test]
    fn codex_catalog_orders_by_pxy_listing() {
        let cfg: Config = toml::from_str(
            r#"
            [server]
            [providers.openai]
            base_url = "https://o.example/chat"
            models = ["gpt-5.4", { id = "o-mini", context_length = 64000 }]
            [providers.zai]
            base_url = "https://z.example/chat"
            models = ["glm-4.7-flash"]
            [groups.muse]
            models = ["zai/glm-4.7-flash"]
            [groups.aaa]
            models = ["openai/gpt-5.4"]
            "#,
        )
        .unwrap();
        let catalog = Catalog::from_config(&cfg);
        let entries = codex_entries(&catalog);
        let built = codex_catalog(&stock(), &entries);
        let slugs: Vec<&str> =
            built["models"].as_array().unwrap().iter().map(|m| m["slug"].as_str().unwrap()).collect();
        // Groups first (alphabetical, as the catalog holds them), then every
        // provider/model id — the order launch_opencode lists.
        assert_eq!(slugs, ["aaa", "muse", "openai/gpt-5.4", "openai/o-mini", "zai/glm-4.7-flash"]);
        let priorities: Vec<u64> = built["models"]
            .as_array()
            .unwrap()
            .iter()
            .map(|m| m["priority"].as_u64().unwrap())
            .collect();
        assert_eq!(priorities, [1, 2, 3, 4, 5]);
        // Each id carries its own window: a model its context_length, a group
        // the narrowest window its chain can serve.
        let by_slug = |s: &str| {
            built["models"].as_array().unwrap().iter().find(|m| m["slug"] == s).unwrap().clone()
        };
        assert_eq!(by_slug("openai/o-mini")["context_window"], 64_000);
        assert_eq!(by_slug("muse")["context_window"], by_slug("zai/glm-4.7-flash")["context_window"]);
    }
}
