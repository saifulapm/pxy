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

    /// Resolve a requested model id to an ordered candidate list.
    ///
    /// - "auto" -> the configured auto chain (config order = priority)
    /// - "provider/model" -> that pair (split on FIRST slash; model ids may
    ///   contain slashes themselves, e.g. openrouter's vendor-prefixed ids)
    /// - bare id -> first provider (config order) listing that model
    pub fn resolve(&self, cfg: &Config, requested: &str) -> Vec<Candidate> {
        if requested == "auto" {
            return self.auto.clone();
        }
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
