mod catalog;
mod config;
mod diagnose;
mod launch;
mod media;
mod providers;
mod refresh;
mod router;
mod secrets;
mod server;
mod state;
mod translate;
mod usage;

use anyhow::Result;
use clap::{Parser, Subcommand};

#[derive(Parser)]
#[command(name = "pxy", version, about = "Tiny multi-provider LLM proxy")]
struct Cli {
    /// Path to config file (default: ~/.config/pxy/config.toml)
    #[arg(long, global = true)]
    config: Option<std::path::PathBuf>,

    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Run the proxy server
    Serve,
    /// Launch a coding agent wired to pxy
    Launch {
        /// Agent: claude | opencode | pi | codex | fx
        agent: String,
        /// Model to use (a group name, or "provider/model")
        #[arg(long, short)]
        model: Option<String>,
        /// Print what would be done without launching
        #[arg(long)]
        dry_run: bool,
        /// Extra arguments passed through to the agent
        #[arg(trailing_var_arg = true, allow_hyphen_values = true)]
        args: Vec<String>,
    },
    /// List available models (group names first, then every provider/model)
    Models {
        /// Emit one JSON array with each entry's details instead of bare ids
        #[arg(long)]
        json: bool,
    },
    /// Diagnose the installation: config, daemon, credentials, agent binaries
    Doctor,
    /// Explain how a model id resolves and why candidates would be skipped
    Explain {
        /// Model id (a group name, "provider/model", or a bare id)
        model: String,
        /// Emit one JSON object instead of the text report
        #[arg(long)]
        json: bool,
    },
    /// Pin one model ahead of every group chain, or show the current pin
    Route {
        /// Model id to pin ("provider/model" or a bare id); omit to show the
        /// current pin
        model: Option<String>,
        /// Remove the pin — group requests follow their configured chain again
        #[arg(long)]
        clear: bool,
    },
    /// Show provider status (cooldowns, usage)
    Status {
        /// Also query providers' remote billing endpoints (balance_url)
        #[arg(long)]
        remote: bool,
        /// Emit one JSON object (providers, modelUsage, remote) instead of
        /// the table — this is what the desktop usage panel consumes
        #[arg(long)]
        json: bool,
        /// Restrict the report (and remote fetches) to these providers;
        /// repeatable. Model-usage rows are never filtered.
        #[arg(long = "provider")]
        providers: Vec<String>,
    },
    /// Discover live provider catalogs; report drift and optionally regenerate
    Refresh {
        /// Write models.toml (every model of every provider). Without this the
        /// command only reports.
        #[arg(long)]
        generate: bool,
    },
    /// Web search via the configured search providers
    Search {
        query: String,
        /// Number of results
        #[arg(long, short, default_value_t = 5)]
        n: u64,
        /// Force a specific search provider
        #[arg(long)]
        provider: Option<String>,
        /// Print the raw JSON response
        #[arg(long)]
        json: bool,
    },
    /// Fetch a URL as markdown (jina-reader / firecrawl)
    Fetch {
        url: String,
        #[arg(long)]
        provider: Option<String>,
    },
    /// Transcribe an audio file (speech-to-text)
    Transcribe {
        file: std::path::PathBuf,
        /// Model ("provider/model" or bare id; default from [media])
        #[arg(long, short)]
        model: Option<String>,
    },
    /// Text-to-speech into an audio file
    Say {
        text: String,
        /// Output file
        #[arg(long, short, default_value = "say.mp3")]
        output: std::path::PathBuf,
        #[arg(long, short)]
        model: Option<String>,
        /// Voice name (mapped per provider) or a raw provider voice id
        #[arg(long)]
        voice: Option<String>,
    },
    /// Generate an image
    Image {
        prompt: String,
        /// Output file
        #[arg(long, short, default_value = "image.png")]
        output: std::path::PathBuf,
        #[arg(long, short)]
        model: Option<String>,
    },
    /// Generate a video (blocks until the upstream job finishes)
    Video {
        prompt: String,
        /// Output file
        #[arg(long, short, default_value = "video.mp4")]
        output: std::path::PathBuf,
        #[arg(long, short)]
        model: Option<String>,
    },
}

fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "pxy=info".into()),
        )
        // stderr, not stdout: `pxy status --json` (and `models`) are piped
        // into JSON parsers, and one stray info! line ahead of the payload
        // silently kills the desktop panel's scanners. journald captures
        // stderr just the same for the daemon.
        .with_writer(std::io::stderr)
        .init();

    let cli = Cli::parse();
    let cfg_path = cli
        .config
        .unwrap_or_else(config::default_config_path);

    match cli.command {
        Command::Serve => {
            let cfg = config::Config::load(&cfg_path)?;
            tokio::runtime::Builder::new_multi_thread()
                .enable_all()
                .build()?
                .block_on(server::serve(cfg))
        }
        Command::Launch { agent, model, dry_run, args } => {
            let cfg = config::Config::load(&cfg_path)?;
            launch::launch(&cfg, &agent, model.as_deref(), dry_run, &args)
        }
        Command::Models { json } => {
            let cfg = config::Config::load(&cfg_path)?;
            models(&cfg, json)
        }
        Command::Doctor => block_on_current(diagnose::doctor(&cfg_path)),
        Command::Explain { model, json } => {
            let cfg = config::Config::load(&cfg_path)?;
            diagnose::explain(&cfg, &model, json)
        }
        Command::Route { model, clear } => {
            let cfg = config::Config::load(&cfg_path)?;
            route(&cfg, model.as_deref(), clear)
        }
        Command::Refresh { generate } => {
            // Baseline only: generation must never consume its own output.
            let cfg = config::Config::load_base(&cfg_path)?;
            let secrets = secrets::Secrets::new();
            let out = config::models_path(&cfg_path);
            tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()?
                .block_on(refresh::run(&cfg, &secrets, generate, &out))
        }
        Command::Status { remote, json, providers } => {
            let cfg = config::Config::load(&cfg_path)?;
            tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()?
                .block_on(server::print_status(&cfg, remote, json, &providers))
        }
        Command::Search { query, n, provider, json } => {
            let cfg = config::Config::load(&cfg_path)?;
            block_on_current(media::cli::search(&cfg, &query, n, provider.as_deref(), json))
        }
        Command::Fetch { url, provider } => {
            let cfg = config::Config::load(&cfg_path)?;
            block_on_current(media::cli::fetch(&cfg, &url, provider.as_deref()))
        }
        Command::Transcribe { file, model } => {
            let cfg = config::Config::load(&cfg_path)?;
            block_on_current(media::cli::transcribe(&cfg, &file, model.as_deref()))
        }
        Command::Say { text, output, model, voice } => {
            let cfg = config::Config::load(&cfg_path)?;
            block_on_current(media::cli::say(
                &cfg,
                &text,
                model.as_deref(),
                voice.as_deref(),
                &output,
            ))
        }
        Command::Image { prompt, output, model } => {
            let cfg = config::Config::load(&cfg_path)?;
            block_on_current(media::cli::image(&cfg, &prompt, model.as_deref(), &output))
        }
        Command::Video { prompt, output, model } => {
            let cfg = config::Config::load(&cfg_path)?;
            block_on_current(media::cli::video(&cfg, &prompt, model.as_deref(), &output))
        }
    }
}

/// `pxy models [--json]` — group names first, then every "provider/model". The
/// JSON form carries what a picker wants to show next to an id (provider, tier,
/// window, tool calling, free-ness); the desktop panel reads it.
fn models(cfg: &config::Config, json: bool) -> Result<()> {
    use std::io::Write;
    let catalog = catalog::Catalog::from_config(cfg);
    let stdout = std::io::stdout();
    let mut out = stdout.lock();
    if !json {
        for m in catalog.model_ids() {
            // stop quietly when the pipe closes (e.g. `pxy models | head`)
            if writeln!(out, "{m}").is_err() {
                break;
            }
        }
        return Ok(());
    }
    let mut rows: Vec<serde_json::Value> = catalog
        .groups()
        .map(|(name, group)| {
            let (ctx, max_out) = catalog::chain_limits(&group.chain);
            serde_json::json!({
                "id": name, "kind": "group", "label": group.label,
                "size": group.chain.len(),
                "contextLength": ctx, "maxOutputTokens": max_out,
                "members": group.chain.iter().map(|c| c.full_id()).collect::<Vec<_>>(),
            })
        })
        .collect();
    // Which groups route to a model is the fact a picker most wants: it is the
    // difference between "one of my free pools" and "real money per token".
    let membership = |full_id: &str| -> Vec<String> {
        catalog
            .groups()
            .filter(|(_, g)| g.chain.iter().any(|c| c.full_id() == full_id))
            .map(|(name, _)| name.clone())
            .collect()
    };
    for cand in catalog.models() {
        let id = cand.full_id();
        let tier = cfg.providers.get(&cand.provider).map(|p| p.tier.as_str());
        rows.push(serde_json::json!({
            "id": id,
            "kind": "model",
            "provider": cand.provider,
            "model": cand.model.id,
            "name": cand.model.name,
            "tier": tier,
            "contextLength": cand.model.context_length,
            "maxOutputTokens": cand.model.max_output_tokens,
            "toolCall": cand.model.tool_call,
            "free": cand.model.free,
            "groups": membership(&id),
        }));
    }
    let _ = writeln!(out, "{}", serde_json::Value::Array(rows));
    Ok(())
}

/// `pxy route [MODEL] [--clear]` — the route pin: one model walked ahead of
/// whichever group chain a request asks for. The pin lives in the daemon's
/// sqlite kv and is read per request, so no restart is needed; the desktop pxy
/// panel drives this same verb.
fn route(cfg: &config::Config, model: Option<&str>, clear: bool) -> Result<()> {
    let st = state::State::open(&config::data_dir().join("state.sqlite"))?;
    let catalog = catalog::Catalog::from_config(cfg);
    // Pinning a GROUP means "no single model" — pinning would grab whatever
    // happens to lead that chain today and freeze it, so it clears instead.
    if clear || model.is_some_and(|m| catalog.is_group(m)) {
        st.kv_delete(router::ROUTE_PIN_KEY)?;
        println!("route unpinned — group chain priority restored");
        return Ok(());
    }
    let Some(model) = model else {
        match st.kv_get(router::ROUTE_PIN_KEY)?.filter(|p| !p.is_empty()) {
            Some(pin) => println!("route pinned to: {pin} (the group chain is the fallback)"),
            None => println!("route unpinned (configured group chain priority)"),
        }
        return Ok(());
    };
    let resolved = catalog.resolve(cfg, model);
    let Some(cand) = resolved.first() else {
        anyhow::bail!("'{model}' resolves to nothing — see `pxy models`");
    };
    // resolve() fabricates a candidate for any id under a known provider;
    // refuse to store one, or a typo'd pin leads every group walk to a model
    // that doesn't exist.
    if !catalog.is_listed(&cand.full_id()) {
        anyhow::bail!("'{model}' is not in the catalog — see `pxy models`");
    }
    // Store what the user typed, resolved to its canonical pair: a bare id
    // pins ONE provider's copy, deterministically, not whichever bare match
    // wins on a later config.
    st.kv_set(router::ROUTE_PIN_KEY, &cand.full_id())?;
    println!(
        "route pinned to: {} (the group chain is the fallback)",
        cand.full_id()
    );
    Ok(())
}

fn block_on_current<F: std::future::Future<Output = Result<()>>>(fut: F) -> Result<()> {
    tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()?
        .block_on(fut)
}
