//! CLI verbs for the Phase 2 endpoints: `pxy search/fetch/transcribe/say/image`
//! and `pxy ask`. Thin HTTP clients against the running daemon so quota
//! accounting and cooldowns stay in one place. The daemon must be up (systemd).

use std::collections::BTreeMap;
use std::io::Read as _;
use std::path::Path;

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

// --------------------------------------------------------------------------
// `pxy ask` — Jev, through the `[media] systemone` chain.
//
// Jev cannot abstain: it must return one of the options it was given, and it
// returns it with a confidence that reads the same whether or not the right
// answer was among them. So the two guards a caller would otherwise write for
// itself are this verb's exit code instead — `--min` on the winning
// probability, `--margin` on its lead over the runner-up — and `pick` adds an
// explicit `none_of_these` option so "not on this page" has somewhere to land.
//
// Exit codes: 0 answered · 1 error · 2 `check` said no · 3 abstained. Both
// failure codes are falsy, so `pxy ask check … && act` declines to act when
// the answer is no *and* when the chain is down.

/// The option `pick` adds so a missing target is expressible.
const ESCAPE: &str = "none_of_these";

const ANSWERED: i32 = 0;
const NO: i32 = 2;
const ABSTAINED: i32 = 3;

#[derive(clap::Subcommand)]
pub enum AskCmd {
    /// Pick one of N labelled options; prints the chosen id
    Pick {
        /// What to decide ("which element is the cart link")
        instructions: String,
        /// An option, as `id=description`; repeatable
        #[arg(long = "option", value_name = "ID=DESC")]
        options: Vec<String>,
        /// Options from a file: a JSON object {id: description}, or `id=description` lines
        #[arg(long, value_name = "PATH")]
        options_file: Option<std::path::PathBuf>,
        /// Leave a missing target unsayable — drop the `none_of_these` option
        #[arg(long)]
        no_escape: bool,
        #[command(flatten)]
        common: AskCommon,
    },
    /// Yes/no probability; exits 0 when it clears --min and 2 when it does not
    Check {
        /// The proposition to judge, stated as a fact ("the cart is empty")
        instructions: String,
        #[command(flatten)]
        common: AskCommon,
    },
    /// Position on an ordered rubric; prints the score
    Rate {
        /// What to rate ("how severe is this failure")
        instructions: String,
        /// Ordered levels, lowest first, separated by `|`
        #[arg(long, value_name = "A|B|C")]
        levels: String,
        #[command(flatten)]
        common: AskCommon,
    },
    /// Send a whole `{state, questions}` body on stdin; prints the answers
    Raw,
}

#[derive(clap::Args)]
pub struct AskCommon {
    /// State to judge; omit to read it from stdin
    #[arg(long, value_name = "PATH")]
    state_file: Option<std::path::PathBuf>,
    /// Abstain below this probability (`check`: answer no)
    #[arg(long, default_value_t = 0.5)]
    min: f64,
    /// Abstain when the winner leads the runner-up by less than this
    #[arg(long, default_value_t = 0.0)]
    margin: f64,
    /// Print the whole answer object instead of the bare value
    #[arg(long)]
    json: bool,
    /// Model id (default: walk the `[media] systemone` chain)
    #[arg(long, short)]
    model: Option<String>,
}

pub async fn ask(cfg: &Config, what: AskCmd) -> Result<i32> {
    match what {
        AskCmd::Raw => {
            let mut body = String::new();
            std::io::stdin().read_to_string(&mut body).context("reading the body from stdin")?;
            let body: Value = serde_json::from_str(&body).context("the body is not JSON")?;
            let out = post_systemone(cfg, body).await?;
            println!("{}", serde_json::to_string_pretty(&out["answers"])?);
            Ok(ANSWERED)
        }

        AskCmd::Pick { instructions, options, options_file, no_escape, common } => {
            let mut criteria = parse_options(&options, options_file.as_deref())?;
            if criteria.len() < 2 {
                bail!("pick needs at least two options (--option ID=DESC, or --options-file)");
            }
            if !no_escape {
                // Strictly absence, never doubt. An escape that also meant
                // "unsure" won on pages where the answer was plainly listed:
                // with little state to go on, "the state does not say which"
                // is arguable about almost anything, and it took 0.59 off a
                // correct option sitting at 0.36. Uncertainty is what --min
                // and --margin are for.
                criteria.insert(
                    ESCAPE.into(),
                    "The thing being asked for is not among the options above at \
                     all. Choose this only when none of them is it — not when \
                     you are unsure which one it is."
                        .into(),
                );
            }
            let answer = one(
                cfg,
                &common,
                json!({"type": "choice", "instructions": instructions, "criteria": criteria}),
            )
            .await?;

            let choice = answer["choice"].as_str().unwrap_or_default().to_string();
            let (top, runner_up) = top_two(&answer["probabilities"]);
            if common.json {
                println!("{}", serde_json::to_string(&answer)?);
            }
            if choice == ESCAPE {
                eprintln!("abstained: no option fits (p={top:.2})");
                return Ok(ABSTAINED);
            }
            if let Some(why) = shortfall(top, top - runner_up, &common) {
                eprintln!("abstained: {why}");
                return Ok(ABSTAINED);
            }
            if !common.json {
                println!("{choice}");
            }
            Ok(ANSWERED)
        }

        AskCmd::Check { instructions, common } => {
            // A noul answers with a bare probability and no confidence, so
            // `--min` is the whole gate and `--margin` has nothing to compare.
            let answer =
                one(cfg, &common, json!({"type": "noul", "instructions": instructions})).await?;
            let p = answer["noul"].as_f64().unwrap_or(0.0);
            if common.json {
                println!("{}", serde_json::to_string(&answer)?);
            } else {
                println!("{p:.3}");
            }
            Ok(if p >= common.min { ANSWERED } else { NO })
        }

        AskCmd::Rate { instructions, levels, common } => {
            let levels: Vec<&str> =
                levels.split('|').map(str::trim).filter(|l| !l.is_empty()).collect();
            if levels.len() < 2 {
                bail!("rate needs at least two levels, lowest first: --levels 'a|b|c'");
            }
            let answer = one(
                cfg,
                &common,
                json!({"type": "score", "instructions": instructions, "criteria": levels}),
            )
            .await?;
            let score = answer["score"].as_f64().unwrap_or(0.0);
            if common.json {
                println!("{}", serde_json::to_string(&answer)?);
            }
            // A score has a confidence but no winner to lead, so only `--min`
            // applies, and it applies to that confidence.
            let confidence = answer["confidence"].as_f64().unwrap_or(1.0);
            if confidence < common.min {
                eprintln!("abstained: confidence {confidence:.2} below --min {:.2}", common.min);
                return Ok(ABSTAINED);
            }
            if !common.json {
                println!("{score:.3}");
            }
            Ok(ANSWERED)
        }
    }
}

/// Ask one question and return its answer object.
async fn one(cfg: &Config, common: &AskCommon, question: Value) -> Result<Value> {
    let body = json!({
        "model": common.model.clone().unwrap_or_else(|| "auto".into()),
        "state": read_state(common.state_file.as_deref())?,
        "questions": {"q": question},
    });
    let out = post_systemone(cfg, body).await?;
    match out["answers"].get("q") {
        Some(answer) => Ok(answer.clone()),
        None => bail!("no answer in the response: {out}"),
    }
}

async fn post_systemone(cfg: &Config, body: Value) -> Result<Value> {
    let resp = expect_ok(post_json(cfg, "/v1/systemone", body).await?).await?;
    Ok(resp.json().await?)
}

/// State is JSON when it parses as JSON and a plain string otherwise, so a
/// page snapshot pipes in as readily as a built-up object.
fn read_state(path: Option<&Path>) -> Result<Value> {
    let raw = match path {
        Some(p) => std::fs::read_to_string(p).with_context(|| format!("reading {}", p.display()))?,
        None => {
            let mut buf = String::new();
            std::io::stdin().read_to_string(&mut buf).context("reading the state from stdin")?;
            buf
        }
    };
    if raw.trim().is_empty() {
        bail!("the state is empty — pass --state-file, or pipe it in");
    }
    Ok(serde_json::from_str(&raw).unwrap_or_else(|_| json!(raw)))
}

/// `--option id=desc` flags and an options file, merged; a flag wins a tie.
fn parse_options(flags: &[String], file: Option<&Path>) -> Result<BTreeMap<String, String>> {
    let mut out = BTreeMap::new();
    if let Some(path) = file {
        let raw =
            std::fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
        match serde_json::from_str::<BTreeMap<String, String>>(&raw) {
            Ok(map) => out.extend(map),
            Err(_) => {
                for line in raw.lines() {
                    let line = line.trim();
                    if line.is_empty() || line.starts_with('#') {
                        continue;
                    }
                    let (id, desc) = split_option(line)?;
                    out.insert(id, desc);
                }
            }
        }
    }
    for flag in flags {
        let (id, desc) = split_option(flag)?;
        out.insert(id, desc);
    }
    if out.contains_key(ESCAPE) {
        bail!("`{ESCAPE}` is the escape option pick adds — name the option something else");
    }
    Ok(out)
}

fn split_option(s: &str) -> Result<(String, String)> {
    match s.split_once('=') {
        Some((id, desc)) if !id.trim().is_empty() => {
            Ok((id.trim().to_string(), desc.trim().to_string()))
        }
        _ => bail!("options are `id=description`, not {s:?}"),
    }
}

/// The two highest probabilities, highest first. A distribution with one entry
/// has no runner-up, so the winner's lead is the whole of it.
fn top_two(probabilities: &Value) -> (f64, f64) {
    let mut ps: Vec<f64> =
        probabilities.as_object().into_iter().flatten().filter_map(|(_, p)| p.as_f64()).collect();
    ps.sort_by(|a, b| b.total_cmp(a));
    (ps.first().copied().unwrap_or(0.0), ps.get(1).copied().unwrap_or(0.0))
}

/// Why this answer does not clear the caller's guards, if it does not.
fn shortfall(top: f64, lead: f64, common: &AskCommon) -> Option<String> {
    if top < common.min {
        return Some(format!("top probability {top:.2} below --min {:.2}", common.min));
    }
    if common.margin > 0.0 && lead < common.margin {
        return Some(format!("lead {lead:.2} below --margin {:.2}", common.margin));
    }
    None
}

#[cfg(test)]
mod ask_tests {
    use super::*;

    fn guards(min: f64, margin: f64) -> AskCommon {
        AskCommon { state_file: None, min, margin, json: false, model: None }
    }

    fn write(name: &str, body: &str) -> std::path::PathBuf {
        let path = std::env::temp_dir().join(format!("pxy-ask-{}-{name}", std::process::id()));
        std::fs::write(&path, body).unwrap();
        path
    }

    #[test]
    fn an_options_file_is_read_as_json_or_as_lines() {
        let json = write("json", r#"{"e1": "the search box", "e2": "the home link"}"#);
        assert_eq!(parse_options(&[], Some(&json)).unwrap()["e1"], "the search box");

        // The same options as lines, with a comment and a blank the parser
        // has to skip rather than read as a malformed row.
        let lines = write("lines", "# refs\ne1=the search box\n\ne2=the home link\n");
        let parsed = parse_options(&[], Some(&lines)).unwrap();
        assert_eq!(parsed["e1"], "the search box");
        assert_eq!(parsed.len(), 2);
    }

    #[test]
    fn a_flag_overrides_the_same_id_in_the_file() {
        let file = write("override", r#"{"e1": "from the file", "e2": "kept"}"#);
        let parsed =
            parse_options(&["e1=from the flag".to_string()], Some(&file)).unwrap();
        assert_eq!(parsed["e1"], "from the flag");
        assert_eq!(parsed["e2"], "kept");
    }

    #[test]
    fn the_escape_id_cannot_be_used_for_a_real_option() {
        // Otherwise `pick` would silently read a real target as an abstention.
        let err = parse_options(&[format!("{ESCAPE}=a real element")], None).unwrap_err();
        assert!(err.to_string().contains(ESCAPE), "{err}");
    }

    #[test]
    fn an_option_without_an_equals_sign_is_refused() {
        let err = parse_options(&["justanid".to_string()], None).unwrap_err();
        assert!(err.to_string().contains("id=description"), "{err}");
    }

    #[test]
    fn top_two_sorts_regardless_of_key_order() {
        let (top, runner_up) = top_two(&json!({"a": 0.1, "b": 0.7, "c": 0.2}));
        assert_eq!((top, runner_up), (0.7, 0.2));
    }

    #[test]
    fn a_lone_option_has_no_runner_up_so_its_lead_is_the_whole_of_it() {
        assert_eq!(top_two(&json!({"only": 0.9})), (0.9, 0.0));
    }

    #[test]
    fn min_and_margin_are_separate_reasons_to_abstain() {
        // Confident and clear: neither guard fires.
        assert!(shortfall(0.84, 0.70, &guards(0.5, 0.15)).is_none());
        // High enough, but the runner-up is right behind it — the near-tie
        // that reads as a confident answer unless the margin catches it.
        assert!(shortfall(0.49, 0.02, &guards(0.4, 0.15)).unwrap().contains("lead"));
        // Clear winner of a weak field.
        assert!(shortfall(0.30, 0.25, &guards(0.5, 0.15)).unwrap().contains("--min"));
    }

    #[test]
    fn margin_is_off_at_zero() {
        // The default, so an exact tie still answers unless a caller asks for
        // the guard.
        assert!(shortfall(0.50, 0.0, &guards(0.5, 0.0)).is_none());
    }

    #[test]
    fn state_is_json_when_it_parses_and_a_string_when_it_does_not() {
        let obj = write("state-json", r#"{"total": "$72.00"}"#);
        assert_eq!(read_state(Some(&obj)).unwrap()["total"], "$72.00");

        // A page snapshot is YAML-ish text, not JSON, and must reach Jev whole
        // rather than as a parse error.
        let snap = write("state-text", "- button \"Search\" [ref=e16]\n");
        assert!(read_state(Some(&snap)).unwrap().as_str().unwrap().contains("[ref=e16]"));
    }

    #[test]
    fn an_empty_state_is_refused_rather_than_asked_about() {
        let empty = write("state-empty", "   \n");
        let err = read_state(Some(&empty)).unwrap_err();
        assert!(err.to_string().contains("empty"), "{err}");
    }
}
