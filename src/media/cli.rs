//! CLI verbs for the Phase 2 endpoints: `pxy search/fetch/transcribe/say/image`.
//! Thin HTTP clients against the running daemon so quota accounting and
//! cooldowns stay in one place. The daemon must be up (systemd).

use anyhow::{Context, Result, bail};
use base64::Engine as _;
use serde_json::{Value, json};

use crate::config::Config;

fn client() -> reqwest::Client {
    reqwest::Client::builder()
        .connect_timeout(std::time::Duration::from_secs(3))
        .build()
        .expect("building http client")
}

async fn post_json(cfg: &Config, path: &str, body: Value) -> Result<reqwest::Response> {
    let url = format!("{}{path}", cfg.base_url());
    let resp = client()
        .post(&url)
        .header("authorization", format!("Bearer {}", cfg.server.api_key))
        .json(&body)
        .send()
        .await
        .with_context(|| format!("connecting to {url} — is `pxy serve` running?"))?;
    Ok(resp)
}

/// Bail with the upstream error message when the daemon answered non-2xx.
async fn expect_ok(resp: reqwest::Response) -> Result<reqwest::Response> {
    if resp.status().is_success() {
        return Ok(resp);
    }
    let status = resp.status();
    let body: Value = resp.json().await.unwrap_or_default();
    let msg = body["error"]["message"].as_str().unwrap_or("").to_string();
    bail!("{status}: {msg}");
}

pub async fn search(cfg: &Config, query: &str, n: u64, provider: Option<&str>, raw: bool) -> Result<()> {
    let mut body = json!({"query": query, "max_results": n});
    if let Some(p) = provider {
        body["provider"] = json!(p);
    }
    let resp = expect_ok(post_json(cfg, "/v1/search", body).await?).await?;
    let out: Value = resp.json().await?;
    if raw {
        println!("{}", serde_json::to_string_pretty(&out)?);
        return Ok(());
    }
    let provider = out["provider"].as_str().unwrap_or("?");
    for (i, r) in out["results"].as_array().into_iter().flatten().enumerate() {
        println!("{}. {}", i + 1, r["title"].as_str().unwrap_or(""));
        println!("   {}", r["url"].as_str().unwrap_or(""));
        let snippet = r["snippet"].as_str().unwrap_or("");
        if !snippet.is_empty() {
            println!("   {snippet}");
        }
    }
    eprintln!("[{provider}]");
    Ok(())
}

pub async fn fetch(cfg: &Config, url: &str, provider: Option<&str>) -> Result<()> {
    let mut body = json!({"url": url});
    if let Some(p) = provider {
        body["provider"] = json!(p);
    }
    let resp = expect_ok(post_json(cfg, "/v1/fetch", body).await?).await?;
    let out: Value = resp.json().await?;
    println!("{}", out["content"].as_str().unwrap_or(""));
    Ok(())
}

pub async fn transcribe(cfg: &Config, file: &std::path::Path, model: Option<&str>) -> Result<()> {
    let bytes = std::fs::read(file).with_context(|| format!("reading {}", file.display()))?;
    let filename = file
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("audio")
        .to_string();
    let part = reqwest::multipart::Part::bytes(bytes).file_name(filename);
    let form = reqwest::multipart::Form::new()
        .part("file", part)
        .text("model", model.unwrap_or("auto").to_string());
    let url = format!("{}/v1/audio/transcriptions", cfg.base_url());
    let resp = client()
        .post(&url)
        .header("authorization", format!("Bearer {}", cfg.server.api_key))
        .multipart(form)
        .send()
        .await
        .with_context(|| format!("connecting to {url} — is `pxy serve` running?"))?;
    let resp = expect_ok(resp).await?;
    let out: Value = resp.json().await?;
    println!("{}", out["text"].as_str().unwrap_or(""));
    Ok(())
}

pub async fn say(
    cfg: &Config,
    text: &str,
    model: Option<&str>,
    voice: Option<&str>,
    output: &std::path::Path,
) -> Result<()> {
    let mut body = json!({"input": text, "model": model.unwrap_or("auto")});
    if let Some(v) = voice {
        body["voice"] = json!(v);
    }
    let resp = expect_ok(post_json(cfg, "/v1/audio/speech", body).await?).await?;
    let bytes = resp.bytes().await?;
    std::fs::write(output, &bytes).with_context(|| format!("writing {}", output.display()))?;
    println!("{} ({} bytes)", output.display(), bytes.len());
    Ok(())
}

pub async fn video(
    cfg: &Config,
    prompt: &str,
    model: Option<&str>,
    output: &std::path::Path,
) -> Result<()> {
    eprintln!("submitting video job (blocks until rendered)…");
    let body = json!({"prompt": prompt, "model": model.unwrap_or("auto")});
    let resp = expect_ok(post_json(cfg, "/v1/videos/generations", body).await?).await?;
    let out: Value = resp.json().await?;
    let Some(url) = out["data"][0]["url"].as_str() else {
        bail!("no video url in response: {out}");
    };
    let bytes = client()
        .get(url)
        .timeout(std::time::Duration::from_secs(300))
        .send()
        .await?
        .error_for_status()?
        .bytes()
        .await?;
    std::fs::write(output, &bytes).with_context(|| format!("writing {}", output.display()))?;
    println!("{} ({} bytes)", output.display(), bytes.len());
    Ok(())
}

pub async fn image(
    cfg: &Config,
    prompt: &str,
    model: Option<&str>,
    output: &std::path::Path,
) -> Result<()> {
    let body = json!({"prompt": prompt, "model": model.unwrap_or("auto")});
    let resp = expect_ok(post_json(cfg, "/v1/images/generations", body).await?).await?;
    let out: Value = resp.json().await?;
    let Some(first) = out["data"].as_array().and_then(|d| d.first()) else {
        bail!("no image in response: {out}");
    };
    let bytes = if let Some(b64) = first["b64_json"].as_str() {
        base64::engine::general_purpose::STANDARD
            .decode(b64)
            .context("decoding image base64")?
    } else if let Some(url) = first["url"].as_str() {
        client().get(url).send().await?.error_for_status()?.bytes().await?.to_vec()
    } else {
        bail!("image response has neither b64_json nor url: {first}");
    };
    std::fs::write(output, &bytes).with_context(|| format!("writing {}", output.display()))?;
    println!("{} ({} bytes)", output.display(), bytes.len());
    Ok(())
}
