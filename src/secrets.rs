use std::collections::HashMap;
use std::process::Command;
use std::sync::Mutex;

use anyhow::{Context, Result};

use crate::config::SecretRef;

/// Resolves secret references, caching results for the process lifetime.
/// `pass` invocations shell out to gpg, so cache aggressively.
pub struct Secrets {
    cache: Mutex<HashMap<String, String>>,
}

impl Secrets {
    pub fn new() -> Self {
        Self { cache: Mutex::new(HashMap::new()) }
    }

    /// Resolve a secret. For pass entries the FULL entry body is returned
    /// (API keys are single-line; OAuth entries are JSON blobs).
    pub fn resolve(&self, sref: &SecretRef) -> Result<String> {
        let key = cache_key(sref);
        if let Some(v) = self.cache.lock().unwrap().get(&key) {
            return Ok(v.clone());
        }
        let value = match sref {
            SecretRef::Pass { pass } => {
                let out = Command::new("pass")
                    .arg("show")
                    .arg(pass)
                    .output()
                    .context("running pass")?;
                if !out.status.success() {
                    anyhow::bail!(
                        "pass show {pass} failed: {}",
                        String::from_utf8_lossy(&out.stderr).trim()
                    );
                }
                String::from_utf8(out.stdout)?.trim_end().to_string()
            }
            SecretRef::Env { env } => std::env::var(env)
                .with_context(|| format!("env var {env} not set"))?,
            SecretRef::Cmd { cmd } => {
                let out = Command::new("sh")
                    .arg("-c")
                    .arg(cmd)
                    .output()
                    .with_context(|| format!("running secret cmd"))?;
                if !out.status.success() {
                    anyhow::bail!(
                        "secret cmd failed: {}",
                        String::from_utf8_lossy(&out.stderr).trim()
                    );
                }
                String::from_utf8(out.stdout)?.trim_end().to_string()
            }
            SecretRef::Literal(v) => v.clone(),
        };
        // API-key style: first line is the secret; but OAuth JSON blobs span
        // lines. Callers that want just the key use `first_line()`.
        self.cache.lock().unwrap().insert(key, value.clone());
        Ok(value)
    }

    /// Resolve and take only the first line (the pass convention for API keys).
    pub fn resolve_key(&self, sref: &SecretRef) -> Result<String> {
        let full = self.resolve(sref)?;
        Ok(full.lines().next().unwrap_or_default().trim().to_string())
    }

    /// Write back an updated pass entry (used for rotated OAuth tokens).
    #[allow(dead_code)] // wired up when OAuth providers with rotating tokens land
    pub fn write_pass(&self, entry: &str, content: &str) -> Result<()> {
        use std::io::Write;
        let mut child = Command::new("pass")
            .args(["insert", "-m", "-f", entry])
            .stdin(std::process::Stdio::piped())
            .stdout(std::process::Stdio::null())
            .spawn()
            .context("running pass insert")?;
        child
            .stdin
            .as_mut()
            .unwrap()
            .write_all(content.as_bytes())?;
        let status = child.wait()?;
        if !status.success() {
            anyhow::bail!("pass insert {entry} failed");
        }
        self.cache
            .lock()
            .unwrap()
            .insert(format!("pass:{entry}"), content.trim_end().to_string());
        Ok(())
    }
}

fn cache_key(sref: &SecretRef) -> String {
    match sref {
        SecretRef::Pass { pass } => format!("pass:{pass}"),
        SecretRef::Env { env } => format!("env:{env}"),
        SecretRef::Cmd { cmd } => format!("cmd:{cmd}"),
        SecretRef::Literal(v) => format!("lit:{v}"),
    }
}
