mod catalog;
mod config;
mod launch;
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
        /// Agent: claude | opencode | pi
        agent: String,
        /// Model to use (pxy model id, e.g. "auto" or "provider/model")
        #[arg(long, short)]
        model: Option<String>,
        /// Print what would be done without launching
        #[arg(long)]
        dry_run: bool,
        /// Extra arguments passed through to the agent
        #[arg(trailing_var_arg = true, allow_hyphen_values = true)]
        args: Vec<String>,
    },
    /// List available models
    Models,
    /// Show provider status (cooldowns, usage)
    Status,
    /// Discover live provider catalogs and report drift (read-only)
    Refresh {
        /// Report only; never write. Currently the only supported mode.
        #[arg(long, default_value_t = true)]
        dry_run: bool,
    },
}

fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "pxy=info".into()),
        )
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
        Command::Models => {
            let cfg = config::Config::load(&cfg_path)?;
            use std::io::Write;
            let stdout = std::io::stdout();
            let mut out = stdout.lock();
            for m in catalog::Catalog::from_config(&cfg).model_ids() {
                // stop quietly when the pipe closes (e.g. `pxy models | head`)
                if writeln!(out, "{m}").is_err() {
                    break;
                }
            }
            Ok(())
        }
        Command::Refresh { dry_run } => {
            if !dry_run {
                anyhow::bail!("only --dry-run is implemented so far");
            }
            let cfg = config::Config::load(&cfg_path)?;
            let secrets = secrets::Secrets::new();
            tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()?
                .block_on(refresh::run(&cfg, &secrets))
        }
        Command::Status => {
            let cfg = config::Config::load(&cfg_path)?;
            tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()?
                .block_on(server::print_status(&cfg))
        }
    }
}
