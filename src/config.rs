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

/// Write a file atomically: tmp + rename, so an interrupt mid-write can never
/// leave a truncated file that a later startup (or another tool) parses as
/// truth. Used for every file pxy regenerates.
pub(crate) fn write_atomic(path: &Path, bytes: &[u8]) -> Result<()> {
    use std::io::Write;
    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir)?;
    }
    let tmp = path.with_extension("pxy-tmp");
    {
        let mut f = std::fs::File::create(&tmp)
            .with_context(|| format!("creating {}", tmp.display()))?;
        f.write_all(bytes)?;
        f.sync_all().ok();
    }
    std::fs::rename(&tmp, path).with_context(|| format!("renaming into {}", path.display()))?;
    Ok(())
}


/// `models.toml` lives beside the config it augments.
pub fn models_path(config_path: &Path) -> PathBuf {
    config_path
        .parent()
        .unwrap_or_else(|| Path::new("."))
        .join("models.toml")
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
    /// Named failover chains — "free", "subscription", "payperuse", whatever
    /// the config declares. A group name is itself a routable model id.
    #[serde(default)]
    pub groups: BTreeMap<String, GroupConfig>,
    /// Provider allowlist. Empty (the default) exposes every enabled provider.
    /// Non-empty exposes ONLY these: their models are the entire catalog a
    /// picker sees, group chains keep only members that survive the filter, and
    /// a group with nothing left stops being advertised at all. An entry
    /// matches by exact name or as a FAMILY PREFIX — "opencode-go" covers
    /// opencode-go-github and opencode-go-google, while never matching an
    /// unrelated name that merely starts with the same letters.
    #[serde(default)]
    pub providers_whitelist: Vec<String>,
    #[serde(default)]
    pub launch: LaunchConfig,
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

/// One named failover chain. Config order inside `models` is walk order: pxy
/// skips candidates in cooldown / over limits / with too small a context, then
/// takes the first survivor. Entirely hand-written — generation never touches
/// a group, so what you put in a chain is what it walks.
#[derive(Debug, Clone, Default, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GroupConfig {
    /// Ordered candidates: "provider/model".
    #[serde(default)]
    pub models: Vec<String>,
    /// What pickers show instead of the raw key. Optional because the key is
    /// usually enough once title-cased — but "payperuse" can only become
    /// "Pay Per Use" if somebody says so.
    pub name: Option<String>,
}

impl GroupConfig {
    /// Display label: the configured `name`, else the key title-cased across
    /// `-` and `_` ("free" -> "Free", "pay-per-use" -> "Pay Per Use").
    pub fn label(&self, key: &str) -> String {
        if let Some(n) = self.name.as_ref().filter(|n| !n.is_empty()) {
            return n.clone();
        }
        key.split(['-', '_'])
            .filter(|w| !w.is_empty())
            .map(|w| {
                let mut c = w.chars();
                match c.next() {
                    Some(f) => f.to_uppercase().collect::<String>() + c.as_str(),
                    None => String::new(),
                }
            })
            .collect::<Vec<_>>()
            .join(" ")
    }
}

/// Cost class of a provider. Display metadata: it rides into `models.toml` and
/// the pickers so a model's price is visible before it is chosen. Never
/// inferred — a vendor's billing behaviour is a curated fact, so anything but
/// `free` must be set by hand.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Deserialize, Default)]
#[serde(rename_all = "kebab-case")]
pub enum Tier {
    /// Free and renewing.
    #[default]
    Free,
    /// A subscription already paid for, with a usage allowance (opencode Go).
    Subscription,
    /// A finite grant that does not renew.
    Finite,
    /// Real money per token, or a promotional balance that drains
    /// (agentrouter, gorouter).
    Reserve,
}

impl Tier {
    pub fn as_str(self) -> &'static str {
        match self {
            Tier::Free => "free",
            Tier::Subscription => "subscription",
            Tier::Finite => "finite",
            Tier::Reserve => "reserve",
        }
    }
}

impl Config {
    pub fn load(path: &Path) -> Result<Self> {
        let mut cfg = Self::load_base(path)?;
        let gen_path = models_path(path);
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

    /// Overlay generated model lists onto the hand-written baseline. Generation
    /// only ever produces model lists, so nothing hand-curated (credentials,
    /// limits, headers, quirks, groups) can be clobbered. A generated block for
    /// an unknown provider is ignored rather than creating a half-configured
    /// provider with no credentials.
    ///
    /// The generated list decides WHICH models exist; for a model the
    /// hand-written config also lists, the hand-written spec wins wholesale.
    /// models.toml only carries id/context/tool_call/free, so taking its entry
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
    }

    /// Whether this provider is exposed at all. The one gate every catalog
    /// view goes through, so the picker and the router can never disagree
    /// about which providers exist.
    pub fn provider_allowed(&self, name: &str) -> bool {
        self.providers_whitelist.is_empty()
            || self.providers_whitelist.iter().any(|entry| {
                entry == name
                    || name
                        .strip_prefix(entry.as_str())
                        .is_some_and(|rest| rest.starts_with('-'))
            })
    }

    pub(crate) fn validate(&self) -> Result<()> {
        // A whitelist entry that matches nothing is a typo, and a typo in an
        // allowlist fails SILENTLY into a smaller catalog — the one failure
        // mode worth a hard error.
        for entry in &self.providers_whitelist {
            if !self.providers.keys().any(|name| {
                entry == name
                    || name
                        .strip_prefix(entry.as_str())
                        .is_some_and(|rest| rest.starts_with('-'))
            }) {
                anyhow::bail!(
                    "providers_whitelist entry '{entry}' matches no configured provider"
                );
            }
        }
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
            // Numeric sanity: these fail SILENTLY downstream — a zero window
            // disables agent auto-compaction (the advertised min goes to 0),
            // rpm 0 skips every candidate of the provider, and a duplicate id
            // lists and walks twice.
            if p.limits.as_ref().and_then(|l| l.rpm).is_some_and(|r| r == 0) {
                anyhow::bail!("provider '{name}': rpm = 0 would skip every candidate");
            }
            let mut seen = std::collections::BTreeSet::new();
            for entry in &p.models {
                let spec = entry.spec();
                if spec.context_length == 0 {
                    anyhow::bail!(
                        "provider '{name}': model '{}' has context_length = 0 (omit it for \
                         the default, or set the real window)",
                        spec.id
                    );
                }
                if !seen.insert(spec.id.clone()) {
                    anyhow::bail!("provider '{name}': duplicate model id '{}'", spec.id);
                }
            }
            // Accounts: a separate credential dimension, exclusive with the
            // top-level credential fields. claude-oauth reads a SHARED
            // credentials FILE (not a per-account secret ref), so it has no
            // account dimension yet.
            if !p.accounts.is_empty() {
                if p.kind == ProviderKind::ClaudeOauth {
                    anyhow::bail!(
                        "provider '{name}': accounts are not supported for claude-oauth \
                         (it reads the Claude Code credentials file, not per-account secrets)"
                    );
                }
                if p.api_key.is_some() || p.credentials.is_some() {
                    anyhow::bail!(
                        "provider '{name}': top-level api_key/credentials are mutually \
                         exclusive with accounts (each account carries its own)"
                    );
                }
                let mut names = std::collections::BTreeSet::new();
                for a in &p.accounts {
                    if a.name.is_empty()
                        || !a
                            .name
                            .bytes()
                            .all(|b| b.is_ascii_lowercase() || b.is_ascii_digit() || b == b'-')
                    {
                        anyhow::bail!(
                            "provider '{name}': account name '{}' must be non-empty and use \
                             only [a-z0-9-] (it is part of the state key)",
                            a.name
                        );
                    }
                    if !names.insert(a.name.clone()) {
                        anyhow::bail!("provider '{name}': duplicate account name '{}'", a.name);
                    }
                    if a.credential().is_none() {
                        anyhow::bail!(
                            "provider '{name}': account '{}' needs api_key or credentials",
                            a.name
                        );
                    }
                }
            }
        }
        for (group, g) in &self.groups {
            // A group name IS a model id on the wire, so it must not collide
            // with the "provider/model" spelling or with a provider name — a
            // group that shadowed a provider would swallow every request for
            // that provider's models.
            if group.contains('/') {
                anyhow::bail!("group '{group}': a group name must not contain '/'");
            }
            if self.providers.contains_key(group) {
                anyhow::bail!("group '{group}' collides with the provider of the same name");
            }
            for entry in &g.models {
                let (prov, _) = entry.split_once('/').with_context(|| {
                    format!("group '{group}': model '{entry}' must be provider/model")
                })?;
                if !self.providers.contains_key(prov) {
                    anyhow::bail!(
                        "group '{group}': model '{entry}' references unknown provider '{prov}'"
                    );
                }
            }
        }
        Ok(())
    }

    /// Model id for a request that names none. Every agent pxy wires sends one,
    /// so this is a safety net: the launch default, else the first group.
    pub fn default_route(&self) -> String {
        self.launch
            .model
            .clone()
            .or_else(|| self.groups.keys().next().cloned())
            .unwrap_or_default()
    }

    pub fn base_url(&self) -> String {
        format!("http://127.0.0.1:{}", self.server.port)
    }
}

/// The subset `pxy refresh --generate` produces. Deliberately tiny: anything
/// not in this struct cannot be generated, and so cannot be lost to generation.
#[derive(Debug, Clone, Default, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Generated {
    #[serde(default)]
    pub providers: BTreeMap<String, GeneratedProvider>,
}

#[derive(Debug, Clone, Default, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GeneratedProvider {
    /// Echo of config.toml's `tier`, written so the file reads on its own.
    /// Declared but never read: config.toml is where a tier lives, and the
    /// field exists so `deny_unknown_fields` accepts the key we emit.
    #[allow(dead_code)]
    pub tier: Option<Tier>,
    #[serde(default)]
    pub models: Vec<ModelEntry>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ServerConfig {
    #[serde(default = "default_port")]
    pub port: u16,
    /// Informational only — pxy is deliberately loopback-only and NO endpoint
    /// checks this today (single-user design, HANDOFF "no multi-tenant"). It
    /// rides `pxy launch` / media CLI auth headers and is what `@@usage`
    /// agent-tagging parses; it does not gate anything.
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
    /// Anthropic Claude subscription via the Claude Code CLI's OAuth
    /// credential file (rotating refresh tokens, written back to the file)
    ClaudeOauth,
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
    /// Path to a credential FILE shared with a local CLI (claude-oauth kind;
    /// default ~/.claude/.credentials.json). pxy reads it fresh per request
    /// and writes refreshed tokens back.
    pub credentials_file: Option<String>,
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
    /// Top-level request-body keys this upstream 400s on (e.g.
    /// `reasoning_effort`, `top_k`, `stream_options`); removed after
    /// translation, just before the wire. Per-model `drop_params` adds to it.
    #[serde(default)]
    pub drop_params: Vec<String>,
    /// Body-matched error overrides (CLIProxyAPI's request-scoped errors):
    /// absorb aggregator/WAF error text without code changes. FIRST matching
    /// rule (case-insensitive substring on the error body) wins over the
    /// status ladder; the context-window carve-out still runs before rules.
    #[serde(default)]
    pub errors: Vec<ErrorRule>,

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
    /// Remote billing endpoint for `pxy status --remote`. Recognized shapes:
    /// OpenRouter's `data.{total_credits,total_usage}` (dollars), new-api's
    /// `data.{quota,used_quota}` (units of 1/500000 USD), and the OpenAI
    /// dashboard's `total_usage` (cents).
    pub balance_url: Option<String>,
    /// Credential for `balance_url` when it is NOT the chat credential, sent
    /// RAW in `Authorization` (no `Bearer`). new-api gateways gate billing
    /// behind a separate console-issued key — aihubmix calls it a Manage Key —
    /// and answer 401 to the `sk-` inference key.
    pub balance_key: Option<SecretRef>,
    /// Separate subscription accounts behind ONE provider entry (CLIProxyAPI's
    /// multi-credential model): the walk tries them in config order
    /// (fill-first), and cooldowns/usage/limits are tracked per account under
    /// the `provider#account` state key. Mutually exclusive with top-level
    /// `api_key`/`credentials`. `balance_url` stays provider-level.
    #[serde(default)]
    pub accounts: Vec<Account>,
    /// Phase 2 media capabilities (images, audio, rerank, video).
    pub media: Option<MediaConfig>,
}

/// One credential of a multi-account provider. `name` is the state-key suffix
/// (`provider#name`) — lowercase letters, digits and dashes only.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Account {
    pub name: String,
    pub api_key: Option<SecretRef>,
    /// OAuth credential blob (JSON in pass) for kinds that need it.
    pub credentials: Option<SecretRef>,
    /// Extra headers this account sends (device identity, org id), overriding
    /// the provider's same-named headers.
    #[serde(default)]
    pub headers: BTreeMap<String, String>,
}

impl Account {
    /// The credential this account authenticates with.
    pub fn credential(&self) -> Option<&SecretRef> {
        self.credentials.as_ref().or(self.api_key.as_ref())
    }
}

/// One body-matched error override for a provider. Actions mirror
/// CLIProxyAPI's request-scoped errors, in pxy's vocabulary: `skip*` moves
/// the walk to the next candidate, `passthrough*` returns the raw upstream
/// error to the client (Claude Code's auto-retry depends on unmodified
/// bodies). The `-cooldown` variants additionally set a model-scoped
/// retryable cooldown.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ErrorRule {
    /// Case-insensitive substring matched against the upstream error body.
    /// TOML key is `match` (`matches` would read as a verb here).
    #[serde(rename = "match")]
    pub matches: String,
    pub action: ErrorAction,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum ErrorAction {
    Skip,
    SkipCooldown,
    Passthrough,
    PassthroughCooldown,
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
    /// expired: failing closed drops a model, failing open spends money. The
    /// parse must be STRICT — a bare string compare let "2026-9-6" (or any
    /// typo sorting after today) keep a paid model in the free chains forever.
    pub fn is_expired(&self, today: &str) -> bool {
        use std::str::FromStr;
        match (
            jiff::civil::Date::from_str(&self.expires),
            jiff::civil::Date::from_str(today),
        ) {
            (Ok(expires), Ok(today)) => today > expires,
            _ => true,
        }
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
            credentials_file: None,
            headers: BTreeMap::new(),
            auth_header: AuthHeader::default(),
            models: Vec::new(),
            limits: None,
            enabled: true,
            timeout_secs: default_timeout(),
            parse_think_tags: false,
            drop_params: Vec::new(),
            errors: Vec::new(),
            accounts: Vec::new(),
            discover: true,
            models_url: None,
            id_field: None,
            tier: Tier::default(),
            promo: None,
            balance_url: None,
            balance_key: None,
            media: None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// These fail SILENTLY downstream (zero window disables agent compaction,
    /// rpm 0 skips every candidate, duplicates walk twice) — so they must be
    /// hard errors at load, not runtime surprises.
    #[test]
    fn numeric_sanity_is_validated() {
        let cases = [
            (
                "rpm zero",
                r#"
                [server]
                [providers.p]
                base_url = "https://p.example/chat"
                models = ["m"]
                [providers.p.limits]
                rpm = 0
                "#,
                "rpm = 0",
            ),
            (
                "zero context",
                r#"
                [server]
                [providers.p]
                base_url = "https://p.example/chat"
                models = [{ id = "m", context_length = 0 }]
                "#,
                "context_length = 0",
            ),
            (
                "duplicate id",
                r#"
                [server]
                [providers.p]
                base_url = "https://p.example/chat"
                models = ["m", "m"]
                "#,
                "duplicate model id 'm'",
            ),
        ];
        for (name, toml_src, expect) in cases {
            let err = toml::from_str::<Config>(toml_src)
                .map_err(|e| e.to_string())
                .and_then(|c| c.validate().map_err(|e| e.to_string()))
                .unwrap_err();
            assert!(err.contains(expect), "{name}: {err}");
        }
        // And a sane config still validates.
        toml::from_str::<Config>(
            r#"
            [server]
            [providers.p]
            base_url = "https://p.example/chat"
            models = [{ id = "m", context_length = 128000 }]
            [providers.p.limits]
            rpm = 5
            "#,
        )
        .map_err(|e| e.to_string())
        .and_then(|c| c.validate().map_err(|e| e.to_string()))
        .unwrap();
    }

    /// Accounts: exclusive with top-level credentials, unique [a-z0-9-] names,
    /// each carrying a credential; claude-oauth has no account dimension.
    #[test]
    fn account_configs_are_validated() {
        let cases = [
            (
                "top-level credential conflict",
                r#"
                [server]
                [providers.p]
                base_url = "https://p.example/chat"
                models = ["m"]
                api_key = "k"
                [[providers.p.accounts]]
                name = "a"
                api_key = "k1"
                "#,
                "mutually exclusive",
            ),
            (
                "duplicate account name",
                r#"
                [server]
                [providers.p]
                base_url = "https://p.example/chat"
                models = ["m"]
                [[providers.p.accounts]]
                name = "a"
                api_key = "k1"
                [[providers.p.accounts]]
                name = "a"
                api_key = "k2"
                "#,
                "duplicate account name 'a'",
            ),
            (
                "bad account charset",
                r#"
                [server]
                [providers.p]
                base_url = "https://p.example/chat"
                models = ["m"]
                [[providers.p.accounts]]
                name = "Main Account"
                api_key = "k1"
                "#,
                "must be non-empty and use",
            ),
            (
                "account without credential",
                r#"
                [server]
                [providers.p]
                base_url = "https://p.example/chat"
                models = ["m"]
                [[providers.p.accounts]]
                name = "a"
                "#,
                "needs api_key or credentials",
            ),
            (
                "claude-oauth accounts",
                r#"
                [server]
                [providers.p]
                kind = "claude-oauth"
                format = "anthropic"
                models = ["m"]
                [[providers.p.accounts]]
                name = "a"
                api_key = "k1"
                "#,
                "not supported for claude-oauth",
            ),
        ];
        for (name, toml_src, expect) in cases {
            let err = toml::from_str::<Config>(toml_src)
                .map_err(|e| e.to_string())
                .and_then(|c| c.validate().map_err(|e| e.to_string()))
                .unwrap_err();
            assert!(err.contains(expect), "{name}: {err}");
        }
        // And a valid accounts config parses + validates, each account
        // carrying its own credential.
        let cfg: Config = toml::from_str(
            r#"
            [server]
            [providers.p]
            base_url = "https://p.example/chat"
            models = ["m"]
            [[providers.p.accounts]]
            name = "gh"
            api_key = "k1"
            [providers.p.accounts.headers]
            x-org = "org-1"
            [[providers.p.accounts]]
            name = "g"
            credentials = { pass = "AI/sub/g" }
            "#,
        )
        .unwrap();
        cfg.validate().unwrap();
        let accounts = &cfg.providers["p"].accounts;
        assert_eq!(accounts.len(), 2);
        assert!(accounts[0].credential().is_some());
        assert!(accounts[1].credential().is_some());
    }

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
            free: None,
            force_stream: true,
            drop_params: Vec::new(),
        })];
        let mut cfg = Config {
            server: ServerConfig { port: 1, api_key: "k".into() },
            providers: BTreeMap::from([("p".to_string(), cfg_provider)]),
            groups: BTreeMap::new(),
            providers_whitelist: Vec::new(),
            launch: LaunchConfig::default(),
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
    /// Asserted tool-calling support, skipping discovery. Set this only from a
    /// real verified call.
    pub tool_call: Option<bool>,
    /// Whether the provider prices this model at zero, as discovery saw it.
    /// DISPLAY METADATA — routing never reads it, so a wrong value costs a
    /// misleading picker row and nothing else. `None` = nobody knows.
    pub free: Option<bool>,
    /// Always request streaming upstream, even for a non-streaming client
    /// call; pxy collects the stream and returns ordinary JSON. For upstreams
    /// that error or time out without `stream: true` on some models
    /// (agentrouter's deepseek-v4f on its Anthropic route).
    #[serde(default)]
    pub force_stream: bool,
    /// Top-level request-body keys THIS model's upstream 400s on; removed
    /// after translation, just before the wire. Adds to the provider-level
    /// list.
    #[serde(default)]
    pub drop_params: Vec<String>,
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
                free: None,
                force_stream: false,
                drop_params: Vec::new(),
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
pub struct LaunchConfig {
    /// Default model for launched agents — usually a group name.
    pub model: Option<String>,
    /// Cheap/background model for agents that want one (claude haiku slot)
    pub small_model: Option<String>,
}
