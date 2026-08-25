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

/// `generated.toml` lives beside the config it augments.
pub fn generated_path(config_path: &Path) -> PathBuf {
    config_path
        .parent()
        .unwrap_or_else(|| Path::new("."))
        .join("generated.toml")
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
    /// Model-quality ranking used to generate the `auto` chain.
    #[serde(default)]
    pub preferences: Preferences,
    /// Phase 2: web search providers (ordered; first healthy wins).
    #[serde(default)]
    pub search: ServiceConfig,
    /// Phase 2: URL -> markdown providers (ordered; first healthy wins).
    #[serde(default)]
    pub fetch: ServiceConfig,
    /// Phase 2: default model per media capability (used when a request
    /// omits `model` or asks for "auto").
    #[serde(default)]
    pub media: MediaDefaults,
}

/// A non-model HTTP service pool (search, fetch). Array of tables so config
/// order is priority order.
#[derive(Debug, Clone, Default, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ServiceConfig {
    #[serde(default)]
    pub providers: Vec<ServiceProvider>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ServiceProvider {
    pub name: String,
    pub kind: ServiceKind,
    pub api_key: SecretRef,
    /// Free-quota guards, enforced locally against pxy's own counters.
    pub daily_requests: Option<u64>,
    pub monthly_requests: Option<u64>,
    #[serde(default = "default_true")]
    pub enabled: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum ServiceKind {
    /// Brave web search (GET, X-Subscription-Token)
    Brave,
    /// Jina s.jina.ai search (POST {q, num})
    Jina,
    /// Firecrawl /v2/search
    FirecrawlSearch,
    /// Jina r.jina.ai reader (GET, markdown out)
    JinaReader,
    /// Firecrawl /v2/scrape
    FirecrawlScrape,
}

#[derive(Debug, Clone, Default, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MediaDefaults {
    pub image: Option<ModelChain>,
    pub transcription: Option<ModelChain>,
    pub speech: Option<ModelChain>,
    pub rerank: Option<ModelChain>,
    pub video: Option<ModelChain>,
}

/// One model id or an ordered failover chain (first healthy wins). A bare
/// string stays valid so existing configs don't change.
#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
pub enum ModelChain {
    One(String),
    Many(Vec<String>),
}

impl ModelChain {
    pub fn as_slice(&self) -> &[String] {
        match self {
            ModelChain::One(s) => std::slice::from_ref(s),
            ModelChain::Many(v) => v,
        }
    }
}

/// Bare model names (no provider), best first. Ordering INSIDE a tier only:
/// a preferred model on a paid pool still sits below the free tier, so a
/// ranking can never quietly start spending money.
#[derive(Debug, Clone, Default, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Preferences {
    #[serde(default)]
    pub models: Vec<String>,
    /// Cap on how many providers may serve the same model in `auto`. Without
    /// it, one popular model with 4 pools crowds out everything below it.
    #[serde(default = "default_max_pools")]
    pub max_pools_per_model: usize,
    /// How many models NOT on the preference list may ride along as a
    /// resilience tail. They sit below every ranked model in their tier.
    #[serde(default = "default_max_unranked")]
    pub max_unranked: usize,
    /// Model ids (or bare names) that must never enter `auto`, whatever
    /// discovery says. For catalogue entries that list but don't work.
    #[serde(default)]
    pub deny: Vec<String>,
}

fn default_max_unranked() -> usize {
    12
}

fn default_max_pools() -> usize {
    3
}

/// Cost class. Decides `auto` ordering, and whether a provider may be in
/// `auto` at all. Never inferred — a vendor's billing behaviour is a curated
/// fact, so `metered`/`reserve` must be set by hand.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Deserialize, Default)]
#[serde(rename_all = "kebab-case")]
pub enum Tier {
    /// Free and renewing; the default home of the auto chain.
    #[default]
    Free,
    /// A subscription already paid for, with a usage allowance.
    Subscription,
    /// A finite grant that does not renew — spend after renewables.
    Finite,
    /// Real money per token, or a balance that can be drained. NEVER in `auto`.
    Reserve,
}

impl Config {
    pub fn load(path: &Path) -> Result<Self> {
        let mut cfg = Self::load_base(path)?;
        let gen_path = generated_path(path);
        if gen_path.exists() {
            let graw = std::fs::read_to_string(&gen_path)
                .with_context(|| format!("reading {}", gen_path.display()))?;
            let generated: Generated = toml::from_str(&graw)
                .with_context(|| format!("parsing {}", gen_path.display()))?;
            cfg.apply_generated(generated);
        }
        cfg.validate()?;
        Ok(cfg)
    }

    /// The hand-written config alone, without the generated overlay.
    /// `pxy refresh` MUST use this: generation reading its own previous
    /// output is a feedback loop — the first symptom was curated `tool_call`
    /// marks vanishing because the overlay had replaced the model lists
    /// before the generator could read them.
    pub fn load_base(path: &Path) -> Result<Self> {
        let raw = std::fs::read_to_string(path)
            .with_context(|| format!("reading config {}", path.display()))?;
        let cfg: Config = toml::from_str(&raw)
            .with_context(|| format!("parsing config {}", path.display()))?;
        Ok(cfg)
    }

    /// Overlay generated model lists and auto chain onto the hand-written
    /// baseline. Generation only ever produces these two things, so nothing
    /// hand-curated (credentials, limits, headers, quirks) can be clobbered.
    /// A generated block for an unknown provider is ignored rather than
    /// creating a half-configured provider with no credentials.
    ///
    /// The generated list decides WHICH models exist; for a model the
    /// hand-written config also lists, the hand-written spec wins wholesale.
    /// generated.toml only carries id/context/tool_call, so taking its entry
    /// for a curated model silently dropped max_output_tokens, per-model
    /// format overrides, and pinned context lengths (groq's 8192).
    fn apply_generated(&mut self, generated: Generated) {
        for (name, g) in generated.providers {
            if let Some(p) = self.providers.get_mut(&name) {
                if !g.models.is_empty() {
                    let curated: BTreeMap<String, ModelEntry> = p
                        .models
                        .iter()
                        .map(|m| (m.spec().id, m.clone()))
                        .collect();
                    p.models = g
                        .models
                        .into_iter()
                        .map(|m| curated.get(&m.spec().id).cloned().unwrap_or(m))
                        .collect();
                }
            }
        }
        if !generated.auto.models.is_empty() {
            self.auto.models = generated.auto.models;
        }
    }

    fn validate(&self) -> Result<()> {
        for (name, p) in &self.providers {
            if p.kind == ProviderKind::OpenaiCompat
                && p.base_url.is_none()
                && p.embeddings_url.is_none()
                && p.media.is_none()
            {
                anyhow::bail!(
                    "provider '{name}': base_url (or embeddings_url / media) is required"
                );
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

/// The subset `pxy refresh --write` produces. Deliberately tiny: anything not
/// in this struct cannot be generated, and so cannot be lost to generation.
#[derive(Debug, Clone, Default, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Generated {
    #[serde(default)]
    pub providers: BTreeMap<String, GeneratedProvider>,
    #[serde(default)]
    pub auto: AutoConfig,
}

#[derive(Debug, Clone, Default, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GeneratedProvider {
    #[serde(default)]
    pub models: Vec<ModelEntry>,
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

    // ---- `pxy refresh` (discovery) ----
    /// Include this provider in catalog discovery. Default ON: an opt-in
    /// allowlist is how OmniRoute ended up silently serving stale catalogs
    /// for dozens of providers.
    #[serde(default = "default_true")]
    pub discover: bool,
    /// Complete models-list URL. Only needed when it can't be derived from
    /// base_url (i.e. base_url doesn't end in /chat/completions).
    pub models_url: Option<String>,
    /// Field holding the usable model id in the discovery response. Default
    /// "id"; Cloudflare needs "name" because its `id` is a UUID.
    pub id_field: Option<String>,
    /// Cost class for `auto` generation.
    #[serde(default)]
    pub tier: Tier,
    /// Time-limited promotional models, dropped from generation once expired.
    /// Upstream expiry metadata is not trustworthy (OpenRouter carries
    /// `expiration_date` on 8 of 419 models, and not on the promo we use), so
    /// the deadline is declared here.
    pub promo: Option<Promo>,
    /// Remote billing endpoint for `pxy status --remote`. Two shapes are
    /// recognized: OpenRouter's `data.{total_credits,total_usage}` (dollars)
    /// and the OpenAI/new-api `total_usage` (cents).
    pub balance_url: Option<String>,
    /// Phase 2 media capabilities (images, audio, rerank, video).
    pub media: Option<MediaConfig>,
}

/// Non-chat capabilities of a provider. URLs may contain `{model}` (and, for
/// speech, `{voice}`) placeholders. `run_url` is the fallback for every
/// capability URL — cloudflare serves all media through one run template.
#[derive(Debug, Clone, Default, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MediaConfig {
    /// Wire dialect. `openai` = canonical shapes pass through untouched.
    #[serde(default)]
    pub kind: MediaKind,
    pub run_url: Option<String>,
    pub images_url: Option<String>,
    #[serde(default)]
    pub image_models: Vec<String>,
    /// Fields merged into image requests when absent (agnes requires `size`).
    #[serde(default)]
    pub image_defaults: BTreeMap<String, serde_json::Value>,
    pub transcription_url: Option<String>,
    #[serde(default)]
    pub transcription_models: Vec<String>,
    pub speech_url: Option<String>,
    #[serde(default)]
    pub speech_models: Vec<String>,
    /// Voice-name -> provider voice id (elevenlabs). Key "default" is used
    /// when the request has no `voice`; an unmapped name passes through.
    #[serde(default)]
    pub voices: BTreeMap<String, String>,
    pub rerank_url: Option<String>,
    #[serde(default)]
    pub rerank_models: Vec<String>,
    pub video_url: Option<String>,
    /// Poll URL template with `{id}` (agnes job flow).
    pub video_status_url: Option<String>,
    #[serde(default)]
    pub video_models: Vec<String>,
    /// Media-only daily request cap, counted separately from chat usage.
    /// Billing safety for providers where overage costs money (cloudflare).
    pub daily_requests: Option<u64>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Default)]
#[serde(rename_all = "kebab-case")]
pub enum MediaKind {
    /// OpenAI images/audio + Cohere rerank shapes, passed through.
    #[default]
    Openai,
    /// Workers AI run endpoint: JSON-or-binary results per task type.
    Cloudflare,
    /// ElevenLabs native TTS/STT.
    Elevenlabs,
    /// Voyage rerank (top_k / data[] instead of top_n / results[]).
    Voyage,
    /// Agnes: OpenAI-ish images + async video job (submit then poll).
    Agnes,
    /// DashScope (Alibaba) multimodal-generation: images/TTS/ASR on one URL.
    Dashscope,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Promo {
    /// Model ids that are only free until `expires`.
    #[serde(default)]
    pub models: Vec<String>,
    /// Last day the promo is valid, `YYYY-MM-DD` (inclusive).
    pub expires: String,
}

impl Promo {
    /// True once `today` is past `expires`. An unparseable date is treated as
    /// expired: failing closed drops a model, failing open spends money.
    pub fn is_expired(&self, today: &str) -> bool {
        self.expires.as_str() < today
    }
}

#[cfg(test)]
impl ProviderConfig {
    /// Minimal instance for unit tests.
    pub fn test_default() -> Self {
        Self {
            kind: ProviderKind::default(),
            format: WireFormat::default(),
            base_url: None,
            embeddings_url: None,
            embedding_models: Vec::new(),
            api_key: None,
            credentials: None,
            headers: BTreeMap::new(),
            auth_header: AuthHeader::default(),
            models: Vec::new(),
            limits: None,
            enabled: true,
            timeout_secs: default_timeout(),
            parse_think_tags: false,
            discover: true,
            models_url: None,
            id_field: None,
            tier: Tier::default(),
            promo: None,
            balance_url: None,
            media: None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn overlay_preserves_curated_model_specs() {
        let mut cfg_provider = ProviderConfig::test_default();
        cfg_provider.models = vec![ModelEntry::Full(ModelSpec {
            id: "m1".into(),
            name: None,
            context_length: 8192, // pinned by hand (groq-style)
            max_output_tokens: 64000,
            format: Some(WireFormat::Anthropic),
            tool_call: Some(true),
            force_stream: true,
        })];
        let mut cfg = Config {
            server: ServerConfig { port: 1, api_key: "k".into() },
            providers: BTreeMap::from([("p".to_string(), cfg_provider)]),
            auto: AutoConfig::default(),
            launch: LaunchConfig::default(),
            preferences: Preferences::default(),
            search: ServiceConfig::default(),
            fetch: ServiceConfig::default(),
            media: MediaDefaults::default(),
        };
        let generated: Generated = toml::from_str(
            r#"
            [providers.p]
            models = [
              { id = "m1", context_length = 131072 },
              { id = "m2", context_length = 32768 },
            ]
            "#,
        )
        .unwrap();
        cfg.apply_generated(generated);
        let models: Vec<ModelSpec> =
            cfg.providers["p"].models.iter().map(|m| m.spec()).collect();
        // Curated spec wins wholesale for m1…
        assert_eq!(models[0].context_length, 8192);
        assert_eq!(models[0].max_output_tokens, 64000);
        assert_eq!(models[0].format, Some(WireFormat::Anthropic));
        assert!(models[0].force_stream);
        // …and discovered m2 still comes through.
        assert_eq!(models[1].id, "m2");
        assert_eq!(models[1].context_length, 32768);
    }
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
    /// ElevenLabs' `xi-api-key` (Bearer is rejected with 401)
    XiApiKey,
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
    /// Asserted tool-calling support, skipping discovery and probing. Set this
    /// only from a real verified call — it exists because probing can't reach
    /// some models (Z.AI allows one concurrent request, so a refresh sweep
    /// gets 429s and would exclude a model that works).
    pub tool_call: Option<bool>,
    /// Always request streaming upstream, even for a non-streaming client
    /// call; pxy collects the stream and returns ordinary JSON. For upstreams
    /// that error or time out without `stream: true` on some models
    /// (agentrouter's deepseek-v4f on its Anthropic route).
    #[serde(default)]
    pub force_stream: bool,
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
                tool_call: None,
                force_stream: false,
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
