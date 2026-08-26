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

pub struct Catalog {
    models: Vec<Candidate>,
    auto: Vec<Candidate>,
}

impl Catalog {
    pub fn from_config(cfg: &Config) -> Self {
        let mut models = Vec::new();
        for (name, p) in &cfg.providers {
            if !p.enabled {
                continue;
            }
            for entry in &p.models {
                models.push(Candidate { provider: name.clone(), model: entry.spec() });
            }
        }
        let auto = cfg
            .auto
            .models
            .iter()
            .filter_map(|entry| {
                let (prov, model_id) = entry.split_once('/')?;
                let pc = cfg.providers.get(prov)?;
                if !pc.enabled {
                    return None;
                }
                // Use the declared spec when listed, else defaults.
                let spec = pc
                    .models
                    .iter()
                    .map(|e| e.spec())
                    .find(|s| s.id == model_id)
                    .unwrap_or_else(|| {
                        let mut s = crate::config::ModelEntry::Id(model_id.to_string()).spec();
                        s.id = model_id.to_string();
                        s
                    });
                Some(Candidate { provider: prov.to_string(), model: spec })
            })
            .collect();
        Self { models, auto }
    }

    /// All exposed model ids ("provider/model", plus "auto" when configured).
    pub fn model_ids(&self) -> Vec<String> {
        let mut ids: Vec<String> = Vec::new();
        if !self.auto.is_empty() {
            ids.push("auto".to_string());
        }
        ids.extend(self.models.iter().map(|c| c.full_id()));
        ids
    }

    pub fn models(&self) -> &[Candidate] {
        &self.models
    }

    /// Whether this exact provider/model pair is actually cataloged — listed
    /// on its provider, or a member of the auto chain. resolve() deliberately
    /// fabricates a spec for any id under a known provider (an explicit
    /// request should still route); the auto-route pin must be stricter, or a
    /// typo'd/stale pin becomes a phantom that every auto request walks first.
    pub fn is_listed(&self, full_id: &str) -> bool {
        self.models.iter().chain(self.auto.iter()).any(|c| c.full_id() == full_id)
    }

    /// Resolve a requested model id to an ordered candidate list.
    ///
    /// - "auto" -> the configured auto chain (config order = priority)
    /// - "claude/<anything>" -> Claude Code discovery alias: the picker only
    ///   shows ids starting "claude"/"anthropic", so /v1/models mirrors every
    ///   id under a "claude/" prefix. Stripped here — but only when the
    ///   stripped base actually resolves, so models on the real `claude`
    ///   provider keep working (never strip blindly: docs/09 §5.1 rule).
    /// - "provider/model" -> that pair (split on FIRST slash; model ids may
    ///   contain slashes themselves, e.g. openrouter's vendor-prefixed ids)
    /// - bare id -> first provider (BTreeMap = alphabetical) listing that model
    pub fn resolve(&self, cfg: &Config, requested: &str) -> Vec<Candidate> {
        if requested == "auto" {
            return self.auto.clone();
        }
        // Mirrors are ALWAYS "claude/<provider>/<model>" or "claude/auto" —
        // a slashless rest ("claude/claude-opus-5") is a REAL model on the
        // `claude` provider and must never be stripped: the bare-id fallback
        // would hand the subscription's model to whichever provider sorts
        // first (agentrouter hijack, caught in review).
        if let Some(rest) = requested.strip_prefix("claude/") {
            if rest == "auto" {
                // Empty chain -> empty candidates -> clean local 404, same
                // as bare "auto"; never a literal model:"auto" upstream.
                return self.auto.clone();
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
                if pc.enabled {
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
            [auto]
            models = ["zai/glm-4.7-flash"]
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
        // Mirrored auto -> the auto chain.
        let r = cat.resolve(&c, "claude/auto");
        assert_eq!(r[0].provider, "zai");
        // Unresolvable stripped base falls through to the claude provider
        // (explicit unlisted id still routable there).
        let r = cat.resolve(&c, "claude/claude-nonexistent-model");
        assert_eq!(r[0].provider, "claude");
        assert_eq!(r[0].model.id, "claude-nonexistent-model");
    }
}
