use std::collections::BTreeMap;

use crate::config::{Config, ModelSpec, ProviderConfig, WireFormat};

/// The marker pxy appends to an advertised id whose window is >= 1M. It
/// exists purely to be read back by Claude Code, which resolves any id
/// matching /\[1m\]/i to a 1,000,000-token window; it is never part of a
/// provider's real id, so resolve() takes it off again before routing.
const CTX_1M_MARKER: &str = "[1m]";

/// The marker for an id, given its window: appended only at >= 1M (rounding
/// 1048576 down to the safe direction), and never doubled on an id that
/// already carries one (which is what makes a trailing marker safe to strip).
/// Public because the /v1/models listing needs it for the "[1m]" variants of
/// claude-containing ids (see server::models).
pub fn ctx_1m_marker(id: &str, context_length: u64) -> &'static str {
    if context_length >= 1_000_000 && !id.to_ascii_lowercase().contains(CTX_1M_MARKER) {
        CTX_1M_MARKER
    } else {
        ""
    }
}

/// The id a Claude Code mirror is advertised under (server::models).
///
/// Claude Code DISCARDS the `context_length` in that listing — its gateway
/// discovery schema is `{id, display_name?}` followed by `.strip()` — so the
/// only window it knows for a mirror is the launch-time
/// `CLAUDE_CODE_MAX_CONTEXT_TOKENS`, stale the moment `/model` picks something
/// else. A `[1m]` anywhere in the id is the one per-model signal it honours:
/// getModelContextWindow tests `/\[1m\]/i` ahead of every env var, and
/// re-resolves on each model switch. An id that already carries the marker is
/// left as is, which is what lets resolve() treat a trailing one as
/// unambiguously pxy's.
pub fn claude_mirror_id(id: &str, context_length: u64) -> String {
    format!("claude/{id}{}", ctx_1m_marker(id, context_length))
}

/// A concrete (provider, model) pair a request can be routed to. For
/// multi-account providers this is one ACCOUNT of the pair — resolve()
/// expands a bare candidate into its accounts (config order = fill-first
/// priority); single-credential providers yield themselves unchanged
/// (`account: None`).
#[derive(Debug, Clone)]
pub struct Candidate {
    pub provider: String,
    pub model: ModelSpec,
    pub account: Option<String>,
}

impl Candidate {
    /// The bare wire id (`provider/model`): what x-pxy-provider reports, what
    /// pins and session bindings store. Panels parse it — never scoped.
    pub fn full_id(&self) -> String {
        format!("{}/{}", self.provider, self.model.id)
    }
    /// The provider scope for STATE keys (cooldowns, usage windows, limits,
    /// failure-rate record): per account, so each account of a subscription
    /// gets its own quota buckets and its own cooldowns. `provider#account`,
    /// same convention as the media keys `provider#media`.
    pub fn state_provider(&self) -> String {
        match &self.account {
            Some(a) => format!("{}#{}", self.provider, a),
            None => self.provider.clone(),
        }
    }
    /// Wire format for this candidate (model override beats provider default).
    pub fn format(&self, provider: &ProviderConfig) -> WireFormat {
        self.model.format.unwrap_or(provider.format)
    }
}

/// A routable group: its display label and the order it walks.
pub struct Group {
    pub label: String,
    pub chain: Vec<Candidate>,
    /// Whether clients that declare capabilities up front (pi) may offer a
    /// thinking level for this group: the config's assertion when it made one,
    /// else the per-member rule.
    pub reasoning: bool,
}

pub struct Catalog {
    models: Vec<Candidate>,
    /// Group name -> its walk order. Config order inside a group is priority;
    /// the map is keyed by name, so groups themselves list alphabetically.
    groups: BTreeMap<String, Group>,
    /// Alias name -> its target (a group, a "provider/model" id or
    /// "auto/free"), straight from the config. Alphabetical like the groups,
    /// which every listing puts them after.
    aliases: BTreeMap<String, String>,
}

/// The bare id a requested spelling routes on: the `claude/` mirror prefix and
/// the `[1m]` window marker taken off, in that order.
fn bare_id(requested: &str) -> &str {
    let bare = requested.strip_prefix("claude/").unwrap_or(requested);
    bare.strip_suffix(CTX_1M_MARKER).unwrap_or(bare)
}

impl Catalog {
    pub fn from_config(cfg: &Config) -> Self {
        let mut models = Vec::new();
        for (name, p) in &cfg.providers {
            if !p.enabled || !cfg.provider_allowed(name) {
                continue;
            }
            for entry in &p.models {
                models.push(Candidate { provider: name.clone(), model: entry.spec(), account: None });
            }
        }
        let groups = cfg
            .groups
            .iter()
            .map(|(name, g)| {
                let chain: Vec<Candidate> = g
                    .models
                    .iter()
                    .filter_map(|entry| {
                        let (prov, model_id) = entry.split_once('/')?;
                        let pc = cfg.providers.get(prov)?;
                        if !pc.enabled || !cfg.provider_allowed(prov) {
                            return None;
                        }
                        // Use the declared spec when listed, else defaults.
                        let spec = pc
                            .models
                            .iter()
                            .map(|e| e.spec())
                            .find(|s| s.id == model_id)
                            .unwrap_or_else(|| {
                                let mut s =
                                    crate::config::ModelEntry::Id(model_id.to_string()).spec();
                                s.id = model_id.to_string();
                                s
                            });
                        Some(Candidate { provider: prov.to_string(), model: spec, account: None })
                    })
                    .collect();
                let reasoning = g.reasoning.unwrap_or_else(|| chain_reasoning(&chain));
                (name.clone(), Group { label: g.label(name), chain, reasoning })
            })
            .collect();
        Self { models, groups, aliases: cfg.aliases.clone() }
    }

    /// All exposed model ids: group names first, then every "provider/model".
    pub fn model_ids(&self) -> Vec<String> {
        let mut ids: Vec<String> = self.group_names().map(str::to_string).collect();
        ids.extend(self.models.iter().map(|c| c.full_id()));
        ids
    }

    pub fn models(&self) -> &[Candidate] {
        &self.models
    }

    /// Non-empty groups only: a group whose every member sits on a disabled
    /// or non-whitelisted provider would advertise an id that resolves to
    /// nothing.
    pub fn groups(&self) -> impl Iterator<Item = (&String, &Group)> {
        self.groups.iter().filter(|(_, g)| !g.chain.is_empty())
    }

    pub fn group_names(&self) -> impl Iterator<Item = &str> {
        self.groups().map(|(name, _)| name.as_str())
    }

    /// Every declared alias, name -> target, alphabetical: the listings put
    /// them after the groups.
    pub fn aliases(&self) -> impl Iterator<Item = (&str, &str)> {
        self.aliases.iter().map(|(name, target)| (name.as_str(), target.as_str()))
    }

    /// The target a requested id aliases (the `claude/` mirror and the `[1m]`
    /// marker stripped), or None when it is not an alias. Aliases are one
    /// level deep — validation refuses a target naming another alias — so the
    /// answer is a group name, a "provider/model" id or "auto/free", never a
    /// second alias.
    pub fn alias_target(&self, requested: &str) -> Option<&str> {
        self.aliases.get(bare_id(requested)).map(String::as_str)
    }

    /// Is this id a routable group (bare, or behind the "claude/" mirror)?
    /// An alias of a group is one; an alias of a model id or of `auto/free`
    /// is not, and routes like the explicit id it names.
    pub fn is_group(&self, requested: &str) -> bool {
        self.group_name(requested).is_some()
    }

    /// The bare group name a requested id routes on (the `claude/` mirror and
    /// the `[1m]` marker stripped, an alias replaced by its target), or None
    /// when it is not a routable group. The route pin is keyed on this, so
    /// every spelling of one group — mirror, marker, alias — shares one pin.
    pub fn group_name(&self, requested: &str) -> Option<&str> {
        let bare = bare_id(requested);
        let bare = self.aliases.get(bare).map(String::as_str).unwrap_or(bare);
        self.groups
            .get_key_value(bare)
            .filter(|(_, g)| !g.chain.is_empty())
            .map(|(k, _)| k.as_str())
    }

    /// The GroupConfig a requested id names, accepting the `claude/` mirror,
    /// the `[1m]` window marker and an alias — the policy lookups (headroom,
    /// paid-reserve) must see the same group `resolve` routed on.
    pub fn group_config<'a>(
        &self,
        cfg: &'a Config,
        requested: &str,
    ) -> Option<&'a crate::config::GroupConfig> {
        let bare = bare_id(requested);
        let bare = self.aliases.get(bare).map(String::as_str).unwrap_or(bare);
        cfg.groups.get(bare)
    }

    /// Whether this exact provider/model pair is actually cataloged — listed
    /// on its provider, or a member of some group. resolve() deliberately
    /// fabricates a spec for any id under a known provider (an explicit
    /// request should still route); the route pin must be stricter, or a
    /// typo'd/stale pin becomes a phantom that every group walk hits first.
    pub fn is_listed(&self, full_id: &str) -> bool {
        self.models
            .iter()
            .chain(self.groups.values().flat_map(|g| &g.chain))
            .any(|c| c.full_id() == full_id)
    }

    /// Every listed candidate whose spec is marked free, in catalog order:
    /// the `auto/free` virtual id. The ordinary walk still applies per-provider
    /// limits and cooldowns, so "free quota left" falls out of the existing
    /// machinery rather than a second accounting path.
    pub fn free_chain(&self) -> Vec<Candidate> {
        self.models
            .iter()
            .filter(|c| c.model.free == Some(true))
            .cloned()
            .collect()
    }

    /// Resolve a requested model id to an ordered candidate list.
    ///
    /// - a group name -> that group's chain (config order = priority)
    /// - "claude/<anything>" -> Claude Code discovery alias: the picker only
    ///   shows ids starting "claude"/"anthropic", so /v1/models mirrors every
    ///   id under a "claude/" prefix. Stripped here — but only when the
    ///   stripped base actually resolves, so models on the real `claude`
    ///   provider keep working (never strip blindly).
    /// - "provider/model" -> that pair (split on FIRST slash; model ids may
    ///   contain slashes themselves, e.g. openrouter's vendor-prefixed ids)
    /// - bare id -> first provider (BTreeMap = alphabetical) listing that model
    ///
    /// An alias is its target here, mirror and marker included: one lookup up
    /// front, and every path below sees the id the alias names.
    pub fn resolve(&self, cfg: &Config, requested: &str) -> Vec<Candidate> {
        let requested = self.alias_target(requested).unwrap_or(requested);
        if let Some(g) = self.groups.get(requested) {
            return g.chain.clone();
        }
        // Virtual id: any free model, across providers.
        if requested == "auto/free" {
            return self.free_chain();
        }
        // A trailing "[1m]" is pxy's own window marker (see claude_mirror_id):
        // the listing appends it to every >= 1M id — claude-containing ones
        // INCLUDED, since the subscription's 1M models are exactly the ids the
        // mirror filter skips. It must come off before routing on every path,
        // so strip it once, up front. (A provider id that genuinely ends in
        // "[1m]" was already broken by the mirror-path strip; this does not
        // change that.)
        let requested = requested.strip_suffix(CTX_1M_MARKER).unwrap_or(requested);
        if let Some(g) = self.groups.get(requested) {
            return g.chain.clone();
        }
        // Mirrors are ALWAYS "claude/<provider>/<model>" or "claude/<group>" —
        // a slashless rest ("claude/claude-opus-5") is a REAL model on the
        // `claude` provider and must never be stripped: the bare-id fallback
        // would hand the subscription's model to whichever provider sorts
        // first (agentrouter hijack, caught in review).
        if let Some(rest) = requested.strip_prefix("claude/") {
            if let Some(g) = self.groups.get(rest) {
                // An empty chain -> empty candidates -> clean local 404, same
                // as the bare group name; never a literal group id upstream.
                return g.chain.clone();
            }
            if rest == "auto/free" {
                return self.free_chain();
            }
            if rest.contains('/') {
                let stripped = self.resolve_concrete(cfg, rest);
                if !stripped.is_empty() {
                    return stripped;
                }
                // A slashed rest that resolves to nothing (stale mirror after
                // a delisting, a whitelisted-away provider) must 404 cleanly:
                // falling through would fabricate the whole slashed string as
                // a model on the real `claude` subscription.
                return Vec::new();
            }
        }
        self.resolve_concrete(cfg, requested)
    }

    fn resolve_concrete(&self, cfg: &Config, requested: &str) -> Vec<Candidate> {
        if let Some((prov, model_id)) = requested.split_once('/') {
            if let Some(pc) = cfg.providers.get(prov) {
                // The whitelist gates routing too, not just the listing: a
                // catalog that hides a provider while still serving it is an
                // allowlist in name only.
                if pc.enabled && cfg.provider_allowed(prov) {
                    let spec = pc
                        .models
                        .iter()
                        .map(|e| e.spec())
                        .find(|s| s.id == model_id)
                        .unwrap_or_else(|| {
                            crate::config::ModelEntry::Id(model_id.to_string()).spec()
                        });
                    return vec![Candidate { provider: prov.to_string(), model: spec, account: None }];
                }
                return Vec::new();
            }
            // No such provider: fall through — the whole string may be a bare
            // model id containing a slash (e.g. "deepseek-ai/DeepSeek-V3").
        }
        self.models
            .iter()
            .filter(|c| c.model.id == requested)
            .take(1)
            .cloned()
            .collect()
    }
}

/// (context_length, max_output_tokens) to advertise for a chain: the MINIMUM
/// over its members, since any member may serve the request. A window a member
/// can't honour breaks the agent's auto-compaction rather than the request.
pub fn chain_limits(chain: &[Candidate]) -> (u64, u64) {
    (
        chain.iter().map(|c| c.model.context_length).min().unwrap_or(0),
        chain.iter().map(|c| c.model.max_output_tokens).min().unwrap_or(0),
    )
}

/// Whether a chain may be advertised as a thinking model: only when EVERY
/// member is asserted to reason, for the same reason chain_limits takes a
/// min() — any member may serve the request, and a client told to expect
/// thinking from one that has none is a client shown a broken capability.
/// An empty chain reasons about nothing.
/// The thinking efforts a whole chain accepts: the INTERSECTION over its
/// members, in canonical order. Same logic as `chain_reasoning` and for the
/// same reason — failover picks the member, so a level only one of them takes
/// is a level the group cannot promise. A member that declares nothing knows
/// nothing and is skipped rather than counted as accepting everything; an
/// empty result means nobody knew, and the client keeps its own defaults.
pub fn chain_effort(chain: &[Candidate]) -> Vec<String> {
    let mut known = chain.iter().filter(|c| !c.model.effort.is_empty()).peekable();
    if known.peek().is_none() {
        return Vec::new();
    }
    crate::refresh::EFFORTS
        .iter()
        .filter(|e| known.clone().all(|c| c.model.effort.iter().any(|m| m == *e)))
        .map(|e| e.to_string())
        .collect()
}

pub fn chain_reasoning(chain: &[Candidate]) -> bool {
    !chain.is_empty() && chain.iter().all(|c| c.model.reasoning == Some(true))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg() -> Config {
        toml::from_str(
            r#"
            [server]
            # Sorts BEFORE "claude" and lists the same bare id — the hijack
            # trap the review caught: a stripped bare id must never win.
            [providers.agentrouter]
            base_url = "https://a.example/chat"
            models = ["claude-opus-5"]
            [providers.claude]
            base_url = "https://api.anthropic.com/v1/messages"
            format = "anthropic"
            models = ["claude-opus-5"]
            [providers.zai]
            base_url = "https://z.example/chat"
            models = ["glm-4.7-flash", { id = "glm-5.3-flash", context_length = 1048576 }]
            [groups.free]
            models = ["zai/glm-4.7-flash"]
            [groups.subscription]
            models = ["claude/claude-opus-5"]
            "#,
        )
        .unwrap()
    }

    #[test]
    fn discovery_alias_strips_only_when_base_resolves() {
        let c = cfg();
        let cat = Catalog::from_config(&c);
        // Mirror id -> the real provider, never the claude provider.
        let r = cat.resolve(&c, "claude/zai/glm-4.7-flash");
        assert_eq!(r[0].provider, "zai");
        assert_eq!(r[0].model.id, "glm-4.7-flash");
        // Real claude-provider model keeps working — even though agentrouter
        // (alphabetically earlier) lists the same bare id. A slashless rest
        // is never stripped, so the subscription cannot be hijacked.
        let r = cat.resolve(&c, "claude/claude-opus-5");
        assert_eq!(r[0].provider, "claude");
        assert_eq!(r[0].model.id, "claude-opus-5");
        // Unresolvable stripped base falls through to the claude provider
        // (explicit unlisted id still routable there).
        let r = cat.resolve(&c, "claude/claude-nonexistent-model");
        assert_eq!(r[0].provider, "claude");
        assert_eq!(r[0].model.id, "claude-nonexistent-model");
    }

    /// The marker exists only so Claude Code reads a 1M window off the id; a
    /// provider must never see it. Round-trips against the id the listing
    /// actually advertises, so the append and the strip cannot drift apart.
    #[test]
    fn the_1m_window_marker_comes_off_before_routing() {
        let c = cfg();
        let cat = Catalog::from_config(&c);

        let mirrored = claude_mirror_id("zai/glm-5.3-flash", 1_048_576);
        assert_eq!(mirrored, "claude/zai/glm-5.3-flash[1m]");
        let r = cat.resolve(&c, &mirrored);
        assert_eq!(r.len(), 1);
        assert_eq!(r[0].provider, "zai");
        assert_eq!(r[0].model.id, "glm-5.3-flash");

        // Sub-1M ids are untouched, and an id that already ends in the marker
        // is not doubled — that is what makes a trailing one safe to strip.
        assert_eq!(
            claude_mirror_id("zai/glm-4.7-flash", 200_000),
            "claude/zai/glm-4.7-flash"
        );
        assert_eq!(
            claude_mirror_id("vercel/some-model[1m]", 1_000_000),
            "claude/vercel/some-model[1m]"
        );

        // Groups carry it the same way, and stay recognisable as groups.
        assert!(cat.is_group("claude/free[1m]"));
        assert_eq!(cat.resolve(&c, "claude/free[1m]")[0].provider, "zai");
    }

    /// The subscription's 1M models are exactly the ids the mirror filter
    /// skips (they contain "claude"), so their listing variant carries the
    /// marker directly on the bare id — and resolve() must strip it on that
    /// path too, or the literal "[1m]" would ride to the upstream.
    #[test]
    fn the_1m_marker_comes_off_claude_ids_too() {
        let c = cfg();
        let cat = Catalog::from_config(&c);
        let r = cat.resolve(&c, "claude/claude-opus-5[1m]");
        assert_eq!(r.len(), 1);
        assert_eq!(r[0].provider, "claude");
        assert_eq!(r[0].model.id, "claude-opus-5");
        // The plain id still resolves to the same model.
        let r = cat.resolve(&c, "claude/claude-opus-5");
        assert_eq!(r[0].model.id, "claude-opus-5");
        // And a genuine 1M claude id is not marker-doubled by ctx_1m_marker.
        assert_eq!(ctx_1m_marker("claude/claude-opus-5[1m]", 1_000_000), "");
        assert_eq!(ctx_1m_marker("claude/claude-opus-5", 1_000_000), "[1m]");
        assert_eq!(ctx_1m_marker("claude/claude-haiku-4-5", 200_000), "");
    }

    /// A group advertises thinking only when every member does. A client that
    /// registers models up front (pi) offers the level for the group id, and
    /// the group id is what an agent is normally launched with — so one silent
    /// member is a thinking level offered for a turn that cannot think.
    #[test]
    fn a_group_reasons_only_when_every_member_does() {
        let c: Config = toml::from_str(
            r#"
            [server]
            [providers.zai]
            base_url = "https://z.example/chat"
            models = [
              { id = "thinker", reasoning = true },
              { id = "quiet" },
              { id = "denied", reasoning = false },
            ]
            [groups.all-think]
            models = ["zai/thinker"]
            [groups.one-unknown]
            models = ["zai/thinker", "zai/quiet"]
            [groups.one-denied]
            models = ["zai/thinker", "zai/denied"]
            [groups.asserted]
            models = ["zai/thinker", "zai/quiet"]
            reasoning = true
            [groups.disowned]
            models = ["zai/thinker"]
            reasoning = false
            "#,
        )
        .unwrap();
        let cat = Catalog::from_config(&c);
        let group = |name: &str| cat.groups().find(|(n, _)| *n == name).unwrap().1.reasoning;
        assert!(group("all-think"));
        // Unset is not a quiet yes: nobody has verified this model thinks.
        assert!(!group("one-unknown"));
        assert!(!group("one-denied"));
        // An empty chain reasons about nothing.
        assert!(!chain_reasoning(&[]));
        // The config's own assertion wins both ways: it is the only way to
        // claim a chain whose members cannot be verified (provider with
        // `discover = false` and an upstream that is down or out of budget),
        // and the only way to disclaim one the per-member rule would grant.
        assert!(group("asserted"));
        assert!(!group("disowned"));
    }

    /// Failover picks the member, so the group may only offer a level every
    /// member takes — one that only some accept is a 400 waiting for the turn
    /// the chain falls through.
    #[test]
    fn a_group_offers_only_the_efforts_its_whole_chain_takes() {
        let c: Config = toml::from_str(
            r#"
            [server]
            [providers.p]
            base_url = "https://p.example/chat"
            models = [
              { id = "wide", reasoning = true, effort = ["none", "low", "medium", "high"] },
              { id = "narrow", reasoning = true, effort = ["low", "high", "max"] },
              { id = "mute", reasoning = true },
            ]
            [groups.mixed]
            models = ["p/wide", "p/narrow"]
            [groups.with-unknown]
            models = ["p/narrow", "p/mute"]
            [groups.nobody-knows]
            models = ["p/mute"]
            "#,
        )
        .unwrap();
        let cat = Catalog::from_config(&c);
        let effort = |name: &str| chain_effort(&cat.groups().find(|(n, _)| *n == name).unwrap().1.chain);
        // "medium" is dropped (narrow rejects it) and so is "max" and "none".
        assert_eq!(effort("mixed"), ["low", "high"]);
        // A member that declares nothing knows nothing: it must not veto the
        // levels a measured member reported, nor invent any.
        assert_eq!(effort("with-unknown"), ["low", "high", "max"]);
        // Nobody measured anything -> say nothing, and let the client decide.
        assert!(effort("nobody-knows").is_empty());
        assert!(chain_effort(&[]).is_empty());
    }

    /// A mirrored id whose stripped base contains a slash but resolves to
    /// nothing (provider disabled or unknown) must 404 cleanly — falling
    /// through would fabricate it as a model on the real `claude`
    /// subscription and burn an upstream call on a guaranteed 404.
    #[test]
    fn an_unresolvable_slashed_mirror_is_a_clean_404() {
        let c: Config = toml::from_str(
            r#"
            [server]
            [providers.claude]
            base_url = "https://api.anthropic.com/v1/messages"
            format = "anthropic"
            models = ["claude-opus-5"]
            [providers.off]
            base_url = "https://o.example/chat"
            enabled = false
            models = ["dead"]
            "#,
        )
        .unwrap();
        let cat = Catalog::from_config(&c);
        assert!(cat.resolve(&c, "claude/off/dead").is_empty());
        assert!(cat.resolve(&c, "claude/none-such/m").is_empty());
        // Slashless rests keep their fallthrough: an explicit unlisted id on
        // the claude provider stays routable (tested above).
    }

    #[test]
    fn group_names_resolve_to_their_chain_bare_and_mirrored() {
        let c = cfg();
        let cat = Catalog::from_config(&c);
        for id in ["free", "claude/free"] {
            let r = cat.resolve(&c, id);
            assert_eq!(r.len(), 1, "{id}");
            assert_eq!(r[0].provider, "zai", "{id}");
        }
        // A group whose name collides with nothing still beats the bare-id
        // fallback, and every group is advertised ahead of the models.
        assert_eq!(
            cat.model_ids()[..2],
            ["free".to_string(), "subscription".to_string()]
        );
        assert!(cat.is_group("free") && cat.is_group("claude/subscription"));
        assert!(!cat.is_group("zai/glm-4.7-flash"));
    }

    #[test]
    fn a_group_member_on_a_disabled_provider_drops_out_of_the_chain() {
        let c: Config = toml::from_str(
            r#"
            [server]
            [providers.zai]
            base_url = "https://z.example/chat"
            models = ["glm-4.7-flash"]
            [providers.off]
            base_url = "https://off.example/chat"
            enabled = false
            models = ["dead"]
            [groups.free]
            models = ["off/dead", "zai/glm-4.7-flash"]
            [groups.empty]
            models = ["off/dead"]
            "#,
        )
        .unwrap();
        let cat = Catalog::from_config(&c);
        let r = cat.resolve(&c, "free");
        assert_eq!(r.len(), 1);
        assert_eq!(r[0].provider, "zai");
        // A group left with nothing is not advertised at all — an id that
        // resolves to nothing is worse in a picker than an absent one.
        assert!(!cat.is_group("empty"));
        assert!(!cat.model_ids().contains(&"empty".to_string()));
    }

    #[test]
    fn whitelist_hides_providers_from_the_catalog_and_from_routing() {
        let c: Config = toml::from_str(
            r#"
            providers_whitelist = ["opencode-go", "zai"]
            [server]
            [providers.zai]
            base_url = "https://z.example/chat"
            models = ["glm-4.7-flash"]
            [providers.opencode-go-github]
            base_url = "https://g.example/chat"
            models = ["hy3"]
            [providers.openrouter]
            base_url = "https://o.example/chat"
            models = ["ox-alpha"]
            [groups.free]
            models = ["openrouter/ox-alpha", "zai/glm-4.7-flash"]
            [groups.paid]
            models = ["openrouter/ox-alpha"]
            "#,
        )
        .unwrap();
        let cat = Catalog::from_config(&c);
        // Family prefix: "opencode-go" covers opencode-go-github.
        let ids = cat.model_ids();
        assert_eq!(
            ids,
            ["free", "opencode-go-github/hy3", "zai/glm-4.7-flash"]
                .map(String::from)
                .to_vec()
        );
        // The chain keeps only the members that survived the filter…
        let chain = cat.resolve(&c, "free");
        assert_eq!(chain.len(), 1);
        assert_eq!(chain[0].provider, "zai");
        // …and a group left with nothing stops being advertised.
        assert!(!cat.is_group("paid"));
        // An explicit request for a hidden provider routes nowhere: hiding a
        // provider while still serving it would be an allowlist in name only.
        assert!(cat.resolve(&c, "openrouter/ox-alpha").is_empty());
    }

    #[test]
    fn group_labels_title_case_the_key_unless_config_names_one() {
        let c: Config = toml::from_str(
            r#"
            [server]
            [providers.p]
            base_url = "https://p.example/chat"
            models = ["m"]
            [groups.free]
            models = ["p/m"]
            [groups.pay-per-use]
            models = ["p/m"]
            [groups.payperuse]
            name = "Pay Per Use"
            models = ["p/m"]
            "#,
        )
        .unwrap();
        let cat = Catalog::from_config(&c);
        let labels: Vec<&str> = cat.groups().map(|(_, g)| g.label.as_str()).collect();
        // Separators are enough for most names; "payperuse" is exactly the
        // case that needs the explicit override.
        assert_eq!(labels, ["Free", "Pay Per Use", "Pay Per Use"]);
        // The label is display only — the routable id is still the key.
        assert!(cat.is_group("payperuse") && !cat.is_group("Pay Per Use"));
    }

    #[test]
    fn whitelist_prefix_never_matches_an_unrelated_name() {
        let c: Config = toml::from_str(
            r#"
            providers_whitelist = ["go"]
            [server]
            [providers.go]
            base_url = "https://a.example/chat"
            [providers.google]
            base_url = "https://b.example/chat"
            "#,
        )
        .unwrap();
        // "go" covers "go" and would cover "go-anything", never "google".
        assert!(c.provider_allowed("go"));
        assert!(c.provider_allowed("go-cloud"));
        assert!(!c.provider_allowed("google"));
    }

    fn alias_cfg() -> Config {
        toml::from_str(
            r#"
            [server]
            [providers.p]
            base_url = "https://p.example/chat"
            models = ["large", { id = "gratis", free = true }]
            [providers.q]
            base_url = "https://q.example/chat"
            models = ["large"]
            [providers.off]
            base_url = "https://o.example/chat"
            enabled = false
            models = ["dead"]
            [groups.daily]
            models = ["p/large", "q/large"]
            headroom = true
            [groups.gone]
            models = ["off/dead"]
            [aliases]
            chat = "daily"
            cheap = "auto/free"
            one = "p/large"
            stale = "gone"
            "#,
        )
        .unwrap()
    }

    /// An alias is a name for its target and nothing else, in every spelling
    /// a group id is accepted in: the client config that pins one never has
    /// to learn the catalog moved underneath it.
    #[test]
    fn alias_resolves_to_its_target_chain() {
        let c = alias_cfg();
        c.validate().unwrap();
        let cat = Catalog::from_config(&c);
        for id in ["chat", "claude/chat", "chat[1m]", "claude/chat[1m]"] {
            let r = cat.resolve(&c, id);
            assert_eq!(r.len(), 2, "{id}");
            assert_eq!(r[0].full_id(), "p/large", "{id}");
            assert_eq!(r[1].full_id(), "q/large", "{id}");
            assert_eq!(cat.alias_target(id), Some("daily"), "{id}");
        }
        // Alphabetical, and listed after the groups they name.
        assert_eq!(
            cat.aliases().collect::<Vec<_>>(),
            [("chat", "daily"), ("cheap", "auto/free"), ("one", "p/large"), ("stale", "gone")]
        );
        // A target left empty by a disabled or whitelisted-away provider
        // resolves to nothing, exactly as the group itself does.
        assert!(cat.resolve(&c, "stale").is_empty());
        assert!(!cat.is_group("stale"));
        assert!(cat.alias_target("daily").is_none());
    }

    /// Ruling 1: an alias of a group IS that group for every policy lookup.
    /// Otherwise `pxy route daily <model>` would not steer a conversation
    /// launched on `chat`, and that conversation would bind its own affinity
    /// key — two walks where the config declared one.
    #[test]
    fn alias_to_a_group_shares_its_pin_key_and_policy() {
        let c = alias_cfg();
        let cat = Catalog::from_config(&c);
        for id in ["chat", "claude/chat", "chat[1m]"] {
            assert!(cat.is_group(id), "{id}");
            assert_eq!(cat.group_name(id), Some("daily"), "{id}");
            assert_eq!(
                crate::router::route_pin_key(cat.group_name(id).unwrap()),
                crate::router::route_pin_key("daily"),
                "{id}"
            );
            assert_eq!(cat.group_config(&c, id).unwrap().headroom, Some(true), "{id}");
        }
    }

    /// Ruling 2: an alias of a model id or of `auto/free` is not a group. It
    /// resolves to what the target resolves to, which is what leaves the
    /// router's explicit-id path (and its account expansion) in charge.
    #[test]
    fn alias_to_a_model_or_auto_free_is_not_a_group() {
        let c = alias_cfg();
        let cat = Catalog::from_config(&c);
        for (alias, target) in [("one", "p/large"), ("cheap", "auto/free")] {
            for id in [alias.to_string(), format!("claude/{alias}"), format!("{alias}[1m]")] {
                assert!(!cat.is_group(&id), "{id}");
                assert!(cat.group_name(&id).is_none(), "{id}");
                assert!(cat.group_config(&c, &id).is_none(), "{id}");
                let got: Vec<String> =
                    cat.resolve(&c, &id).iter().map(|x| x.full_id()).collect();
                let want: Vec<String> =
                    cat.resolve(&c, target).iter().map(|x| x.full_id()).collect();
                assert_eq!(got, want, "{id}");
                assert!(!got.is_empty(), "{id}");
            }
        }
        assert_eq!(cat.resolve(&c, "cheap")[0].full_id(), "p/gratis");
    }

    #[test]
    fn a_whitelist_entry_matching_nothing_is_a_hard_error() {
        let err = toml::from_str::<Config>(
            r#"
            providers_whitelist = ["zai", "typo"]
            [server]
            [providers.zai]
            base_url = "https://z.example/chat"
            "#,
        )
        .map_err(|e| e.to_string())
        .and_then(|c: Config| c.validate().map_err(|e| e.to_string()))
        .unwrap_err();
        assert!(err.contains("'typo' matches no configured provider"), "{err}");
    }
}
