use std::collections::BTreeMap;

use crate::config::{Config, ModelSpec, ProviderConfig, WireFormat};

/// A concrete (provider, model) pair a request can be routed to.
#[derive(Debug, Clone)]
pub struct Candidate {
    pub provider: String,
    pub model: ModelSpec,
}

impl Candidate {
    pub fn full_id(&self) -> String {
        format!("{}/{}", self.provider, self.model.id)
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
}

pub struct Catalog {
    models: Vec<Candidate>,
    /// Group name -> its walk order. Config order inside a group is priority;
    /// the map is keyed by name, so groups themselves list alphabetically.
    groups: BTreeMap<String, Group>,
}

impl Catalog {
    pub fn from_config(cfg: &Config) -> Self {
        let mut models = Vec::new();
        for (name, p) in &cfg.providers {
            if !p.enabled || !cfg.provider_allowed(name) {
                continue;
            }
            for entry in &p.models {
                models.push(Candidate { provider: name.clone(), model: entry.spec() });
            }
        }
        let groups = cfg
            .groups
            .iter()
            .map(|(name, g)| {
                let chain = g
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
                        Some(Candidate { provider: prov.to_string(), model: spec })
                    })
                    .collect();
                (name.clone(), Group { label: g.label(name), chain })
            })
            .collect();
        Self { models, groups }
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

    /// Is this id a routable group (bare, or behind the "claude/" mirror)?
    pub fn is_group(&self, requested: &str) -> bool {
        let bare = requested.strip_prefix("claude/").unwrap_or(requested);
        self.groups.get(bare).is_some_and(|g| !g.chain.is_empty())
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

    /// Resolve a requested model id to an ordered candidate list.
    ///
    /// - a group name -> that group's chain (config order = priority)
    /// - "claude/<anything>" -> Claude Code discovery alias: the picker only
    ///   shows ids starting "claude"/"anthropic", so /v1/models mirrors every
    ///   id under a "claude/" prefix. Stripped here — but only when the
    ///   stripped base actually resolves, so models on the real `claude`
    ///   provider keep working (never strip blindly: docs/09 §5.1 rule).
    /// - "provider/model" -> that pair (split on FIRST slash; model ids may
    ///   contain slashes themselves, e.g. openrouter's vendor-prefixed ids)
    /// - bare id -> first provider (BTreeMap = alphabetical) listing that model
    pub fn resolve(&self, cfg: &Config, requested: &str) -> Vec<Candidate> {
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
            if rest.contains('/') {
                let stripped = self.resolve_concrete(cfg, rest);
                if !stripped.is_empty() {
                    return stripped;
                }
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
                    return vec![Candidate { provider: prov.to_string(), model: spec }];
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
            kind = "claude-oauth"
            format = "anthropic"
            models = ["claude-opus-5"]
            [providers.zai]
            base_url = "https://z.example/chat"
            models = ["glm-4.7-flash"]
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
