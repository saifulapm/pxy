use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use serde::Deserialize;

pub fn default_config_path() -> PathBuf {
    let base = std::env::var_os("XDG_CONFIG_HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|| home_dir().join(".config"));
    base.join("pxy").join("config.toml")
}

pub fn home_dir() -> PathBuf {
    PathBuf::from(std::env::var_os("HOME").expect("HOME not set"))
}

pub fn data_dir() -> PathBuf {
    let base = std::env::var_os("XDG_DATA_HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|| home_dir().join(".local").join("share"));
    base.join("pxy")
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Config {
    pub server: ServerConfig,
    #[serde(default)]
    pub providers: BTreeMap<String, ProviderConfig>,
    #[serde(default)]
    pub auto: AutoConfig,
    #[serde(default)]
    pub launch: LaunchConfig,
}

impl Config {
    pub fn load(path: &Path) -> Result<Self> {
        let raw = std::fs::read_to_string(path)
            .with_context(|| format!("reading config {}", path.display()))?;
        let cfg: Config = toml::from_str(&raw)
            .with_context(|| format!("parsing config {}", path.display()))?;
        cfg.validate()?;
        Ok(cfg)
    }

    fn validate(&self) -> Result<()> {
        for (name, p) in &self.providers {
            if p.kind == ProviderKind::OpenaiCompat
                && p.base_url.is_none()
                && p.embeddings_url.is_none()
            {
                anyhow::bail!("provider '{name}': base_url (or embeddings_url) is required");
            }
        }
        for entry in &self.auto.models {
            let (prov, _) = entry
                .split_once('/')
                .with_context(|| format!("auto model '{entry}' must be provider/model"))?;
            if !self.providers.contains_key(prov) {
                anyhow::bail!("auto model '{entry}' references unknown provider '{prov}'");
            }
        }
        Ok(())
    }

    pub fn base_url(&self) -> String {
        format!("http://127.0.0.1:{}", self.server.port)
    }
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ServerConfig {
    #[serde(default = "default_port")]
    pub port: u16,
    /// Key clients must send. Loopback-only bind, so this is a soft gate.
    #[serde(default = "default_api_key")]
    pub api_key: String,
}

fn default_port() -> u16 {
    4100
}
fn default_api_key() -> String {
    "pxy-local".into()
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum WireFormat {
    /// OpenAI chat completions
    Openai,
    /// Anthropic messages
    Anthropic,
    /// AWS CodeWhisperer conversationState + vnd.amazon.eventstream
    Kiro,
}

impl Default for WireFormat {
    fn default() -> Self {
        WireFormat::Openai
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Default)]
#[serde(rename_all = "kebab-case")]
pub enum ProviderKind {
    /// Plain HTTP endpoint with a static credential
    #[default]
    OpenaiCompat,
    /// GitHub Copilot: 2-stage token mint + header profile
    GithubCopilot,
    /// Kimi coding tier: rotating refresh tokens + device-id header profile
    KimiCoding,
    /// Kiro / Amazon Q: CodeWhisperer conversationState + eventstream
    Kiro,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProviderConfig {
    #[serde(default)]
    pub kind: ProviderKind,
    /// Wire format of `base_url` (what the upstream speaks)
    #[serde(default)]
    pub format: WireFormat,
    /// COMPLETE endpoint URL (e.g. https://api.deepseek.com/chat/completions)
    pub base_url: Option<String>,
    /// COMPLETE embeddings endpoint URL, if the provider offers one
    pub embeddings_url: Option<String>,
    /// Embedding model ids served via embeddings_url (kept out of the chat catalog)
    #[serde(default)]
    pub embedding_models: Vec<String>,
    pub api_key: Option<SecretRef>,
    /// OAuth credential blob (JSON in pass) for kinds that need it
    pub credentials: Option<SecretRef>,
    #[serde(default)]
    pub headers: BTreeMap<String, String>,
    /// Header used for the credential. Default: authorization bearer.
    #[serde(default)]
    pub auth_header: AuthHeader,
    #[serde(default)]
    pub models: Vec<ModelEntry>,
    pub limits: Option<Limits>,
    #[serde(default = "default_true")]
    pub enabled: bool,
    /// Request timeout in seconds (default 600)
    #[serde(default = "default_timeout")]
    pub timeout_secs: u64,
    /// Extract literal <think>...</think> tags from responses into proper
    /// reasoning (for models like MiniMax/Qwen that inline CoT as text)
    #[serde(default)]
    pub parse_think_tags: bool,
}

fn default_true() -> bool {
    true
}
fn default_timeout() -> u64 {
    600
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Default)]
#[serde(rename_all = "kebab-case")]
pub enum AuthHeader {
    #[default]
    Bearer,
    XApiKey,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
pub enum SecretRef {
    Pass { pass: String },
    Env { env: String },
    Cmd { cmd: String },
    Literal(String),
}

#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
pub enum ModelEntry {
    Id(String),
    Full(ModelSpec),
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ModelSpec {
    pub id: String,
    /// Display name
    pub name: Option<String>,
    #[serde(default = "default_context")]
    pub context_length: u64,
    #[serde(default = "default_max_output")]
    pub max_output_tokens: u64,
    /// Override wire format for this model (e.g. copilot claude models)
    pub format: Option<WireFormat>,
}

pub fn default_context() -> u64 {
    128_000
}
pub fn default_max_output() -> u64 {
    16_384
}

impl ModelEntry {
    pub fn spec(&self) -> ModelSpec {
        match self {
            ModelEntry::Id(id) => ModelSpec {
                id: id.clone(),
                name: None,
                context_length: default_context(),
                max_output_tokens: default_max_output(),
                format: None,
            },
            ModelEntry::Full(s) => s.clone(),
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Limits {
    /// Requests per minute (sliding window)
    pub rpm: Option<u32>,
    pub daily_requests: Option<u64>,
    pub daily_tokens: Option<u64>,
    pub monthly_requests: Option<u64>,
    pub monthly_tokens: Option<u64>,
    /// All-time budget (e.g. a prepaid credit pack). Never resets.
    pub total_requests: Option<u64>,
    pub total_tokens: Option<u64>,
    /// Daily reset time "HH:MM" in `reset_tz` (default "00:00")
    #[serde(default = "default_reset")]
    pub reset: String,
    /// IANA timezone for daily/monthly windows (default "UTC")
    #[serde(default = "default_tz")]
    pub reset_tz: String,
}

fn default_reset() -> String {
    "00:00".into()
}
fn default_tz() -> String {
    "UTC".into()
}

impl Default for Limits {
    fn default() -> Self {
        Limits {
            rpm: None,
            daily_requests: None,
            daily_tokens: None,
            monthly_requests: None,
            monthly_tokens: None,
            total_requests: None,
            total_tokens: None,
            reset: default_reset(),
            reset_tz: default_tz(),
        }
    }
}

#[derive(Debug, Clone, Deserialize, Default)]
#[serde(deny_unknown_fields)]
pub struct AutoConfig {
    /// Ordered candidates: "provider/model". First healthy with headroom wins.
    #[serde(default)]
    pub models: Vec<String>,
}

#[derive(Debug, Clone, Deserialize, Default)]
#[serde(deny_unknown_fields)]
pub struct LaunchConfig {
    /// Default model for launched agents (default: "auto")
    pub model: Option<String>,
    /// Cheap/background model for agents that want one (claude haiku slot)
    pub small_model: Option<String>,
}
