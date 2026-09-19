//! PDF text for the file-parser plugin (wiki:plugins): poppler extracts the
//! text, and pages it finds empty go through the same OCR leg describe_image
//! uses (wiki:vision).

use base64::Engine as _;
use serde_json::json;
use sha2::{Digest, Sha256};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use tokio::process::Command;
use tracing::warn;

use crate::router::{ClientContext, SharedApp};
use crate::translate::server_tools::{OCR_PROMPT, run_internal_chat};

/// How a page's text is obtained.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Engine {
    /// poppler's `pdftotext` only; a page with no text stays empty.
    Pdftotext,
    /// Rasterise every page and read it with the OCR model.
    Ocr,
    /// `pdftotext`, then OCR for the pages that came back all but blank.
    Auto,
}

impl Engine {
    /// The `engine` a request or config spells; anything else is the default.
    pub fn from_name(name: &str) -> Engine {
        match name {
            "pdftotext" => Engine::Pdftotext,
            "ocr" => Engine::Ocr,
            _ => Engine::Auto,
        }
    }
}

/// A page poppler answered with less than this many non-blank characters is
/// scanned rather than typeset, so `auto` sends it to the OCR model.
const THIN_PAGE: usize = 20;

/// The SHA-256 of a file as lowercase hex: the `file_parse:<hex>` cache key.
pub fn sha256_hex(bytes: &[u8]) -> String {
    // sha2 0.11's digest has no LowerHex, so the hex is built per byte.
    Sha256::digest(bytes).iter().map(|b| format!("{b:02x}")).collect()
}

/// A parsed PDF. `complete` is false when an OCR leg failed and poppler's
/// text (or the no-text note) stands in for that page: the Markdown is
/// still the best answer for this turn, but not one to cache under the
/// file's hash forever, or a transient failure would silence OCR for these
/// bytes for good (seen live 2026-09-19).
pub struct Parsed {
    pub markdown: String,
    pub complete: bool,
}

/// Parse a PDF into Markdown, one section per page. `Err` is a file pxy could
/// not read at all, which the caller turns into the one-line part that
/// replaces the file; a page that failed on its own is reported in place.
pub async fn pdf_to_markdown(
    app: &SharedApp,
    bytes: &[u8],
    engine: Engine,
    ocr_model: Option<&str>,
    max_pages: u64,
    caller: &ClientContext,
) -> Result<Parsed, String> {
    let dir = scratch_dir()?;
    let parsed = parse(app, &dir, bytes, engine, ocr_model, max_pages, caller).await;
    let _ = std::fs::remove_dir_all(&dir);
    parsed
}

async fn parse(
    app: &SharedApp,
    dir: &Path,
    bytes: &[u8],
    engine: Engine,
    ocr_model: Option<&str>,
    max_pages: u64,
    caller: &ClientContext,
) -> Result<Parsed, String> {
    let pdf = dir.join("in.pdf");
    std::fs::write(&pdf, bytes).map_err(|e| format!("cannot stage the file: {e}"))?;
    let pages = pdftotext(&pdf).await?;
    let kept = pages.len().min(max_pages as usize);

    let mut out = Vec::new();
    let mut complete = true;
    for (i, text) in pages.iter().take(kept).enumerate() {
        let n = i + 1;
        let (body, ok) = page(app, &pdf, n, text, engine, ocr_model, caller).await;
        complete &= ok;
        out.push(format!("## Page {n}\n\n{body}"));
    }
    if pages.len() > kept {
        out.push(format!("[{} more pages not parsed]", pages.len() - kept));
    }
    Ok(Parsed { markdown: out.join("\n\n"), complete })
}

/// One page's Markdown: poppler's text, what the OCR model read, or the note
/// that says the page held neither. The flag is false when an OCR leg was
/// wanted and failed.
async fn page(
    app: &SharedApp,
    pdf: &Path,
    n: usize,
    text: &str,
    engine: Engine,
    ocr_model: Option<&str>,
    caller: &ClientContext,
) -> (String, bool) {
    let thin = text.chars().filter(|c| !c.is_whitespace()).count() < THIN_PAGE;
    let wants_ocr = match engine {
        Engine::Pdftotext => false,
        Engine::Ocr => true,
        Engine::Auto => thin,
    };
    let mut ok = true;
    if wants_ocr && let Some(model) = ocr_model {
        // A leg that failed leaves poppler's text standing: half a page beats
        // none, and the model is never told a page was skipped.
        match ocr(app, pdf, n, model, caller).await {
            Ok(markdown) => return (markdown, true),
            Err(e) => {
                warn!(page = n, error = %e, "pdf ocr leg failed");
                ok = false;
            }
        }
    }
    let body = match text.trim() {
        "" => format!("[page {n}: no text]"),
        text => text.to_string(),
    };
    (body, ok)
}

/// poppler's text, split into pages.
async fn pdftotext(pdf: &Path) -> Result<Vec<String>, String> {
    let out = Command::new("pdftotext")
        .args(["-layout", "-enc", "UTF-8"])
        .arg(pdf)
        .arg("-")
        .output()
        .await
        .map_err(|e| missing_binary("pdftotext", &e))?;
    if !out.status.success() {
        return Err(format!("pdftotext failed: {}", first_line(&out.stderr)));
    }
    let text = String::from_utf8_lossy(&out.stdout);
    // pdftotext ends EVERY page with a form feed, the last one included, so
    // the split leaves a final empty segment that is not a page.
    let mut pages: Vec<String> = text.split('\x0c').map(str::to_string).collect();
    if pages.last().is_some_and(String::is_empty) {
        pages.pop();
    }
    Ok(pages)
}

/// Rasterise one page and read it with the OCR model: the same single user
/// message describe_image sends, because DeepSeek 400s on an image anywhere
/// but a user turn (wiki:vision).
async fn ocr(
    app: &SharedApp,
    pdf: &Path,
    n: usize,
    model: &str,
    caller: &ClientContext,
) -> Result<String, String> {
    let prefix = pdf.with_file_name(format!("page{n}"));
    let out = Command::new("pdftoppm")
        .args(["-r", "150", "-png", "-f", &n.to_string(), "-l", &n.to_string()])
        .arg(pdf)
        .arg(&prefix)
        .output()
        .await
        .map_err(|e| missing_binary("pdftoppm", &e))?;
    if !out.status.success() {
        return Err(format!("pdftoppm failed: {}", first_line(&out.stderr)));
    }
    let png = std::fs::read(rendered(&prefix)?).map_err(|e| format!("cannot read the page: {e}"))?;
    let url = format!(
        "data:image/png;base64,{}",
        base64::engine::general_purpose::STANDARD.encode(&png)
    );
    let messages = vec![json!({"role": "user", "content": [
        {"type": "text", "text": OCR_PROMPT},
        {"type": "image_url", "image_url": {"url": url}},
    ]})];
    run_internal_chat(app, model, messages, json!({}), caller).await
}

/// pdftoppm zero-pads the page number to the page-count width, so the file it
/// wrote is found by listing, never by formatting the number.
fn rendered(prefix: &Path) -> Result<PathBuf, String> {
    let (dir, stem) = match (prefix.parent(), prefix.file_name().and_then(|s| s.to_str())) {
        (Some(dir), Some(stem)) => (dir, stem),
        _ => return Err("no page to read".to_string()),
    };
    std::fs::read_dir(dir)
        .map_err(|e| format!("cannot list the pages: {e}"))?
        .flatten()
        .map(|e| e.path())
        .find(|p| {
            p.file_name()
                .and_then(|s| s.to_str())
                .is_some_and(|s| s.starts_with(&format!("{stem}-")) && s.ends_with(".png"))
        })
        .ok_or_else(|| "pdftoppm wrote no page".to_string())
}

/// A scratch directory of our own: two turns parsing the same file at once
/// must not delete each other's pages.
fn scratch_dir() -> Result<PathBuf, String> {
    static NEXT: AtomicU64 = AtomicU64::new(0);
    let dir = std::env::temp_dir().join(format!(
        "pxy-pdf-{}-{}",
        std::process::id(),
        NEXT.fetch_add(1, Ordering::Relaxed)
    ));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).map_err(|e| format!("cannot make a scratch directory: {e}"))?;
    Ok(dir)
}

/// poppler is a runtime requirement checked here, not at startup, so a pxy on
/// a machine without it serves every request that carries no file.
fn missing_binary(binary: &str, e: &std::io::Error) -> String {
    match e.kind() {
        std::io::ErrorKind::NotFound => format!("{binary} not installed"),
        _ => format!("{binary} failed to start: {e}"),
    }
}

fn first_line(stderr: &[u8]) -> String {
    String::from_utf8_lossy(stderr).lines().next().unwrap_or("no output").trim().to_string()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::router::App;
    use serde_json::{Value, json};
    use std::sync::{Arc, Mutex};

    /// Two pages: the first carries text, the second carries nothing at all.
    const FIXTURE: &[u8] = include_bytes!("../../tests/fixtures/hello.pdf");

    #[test]
    fn sha256_hex_matches_a_known_digest() {
        assert_eq!(
            sha256_hex(b""),
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        );
        assert_eq!(
            sha256_hex(FIXTURE),
            "7b7f4f7070adb9e56eaf0adedad5bd0e38b9b50988182b67d8a917a5cbebdc23"
        );
    }

    #[test]
    fn engine_names_fall_back_to_auto() {
        assert_eq!(Engine::from_name("pdftotext"), Engine::Pdftotext);
        assert_eq!(Engine::from_name("ocr"), Engine::Ocr);
        assert_eq!(Engine::from_name(""), Engine::Auto);
        assert_eq!(Engine::from_name("magic"), Engine::Auto);
    }

    /// Every page gets a section, and a page poppler found empty says so
    /// rather than vanishing: a model that is told nothing about page 2
    /// answers as if the document ended.
    #[tokio::test]
    async fn every_page_gets_a_section() {
        let app = mock_app("[server]\n", "pages");
        let md = pdf_to_markdown(&app, FIXTURE, Engine::Pdftotext, None, 20, &ctx())
            .await
            .unwrap()
            .markdown;
        assert!(md.contains("## Page 1"), "{md}");
        assert!(md.contains("Hello, pxy."), "{md}");
        assert!(md.contains("## Page 2"), "{md}");
        assert!(md.contains("[page 2: no text]"), "{md}");
    }

    #[tokio::test]
    async fn max_pages_cuts_with_a_trailing_line() {
        let app = mock_app("[server]\n", "cut");
        let md = pdf_to_markdown(&app, FIXTURE, Engine::Pdftotext, None, 1, &ctx())
            .await
            .unwrap()
            .markdown;
        assert!(md.contains("Hello, pxy."), "{md}");
        assert!(!md.contains("## Page 2"), "{md}");
        assert!(md.ends_with("[1 more pages not parsed]"), "{md}");
    }

    /// `auto` sends only the page that came back blank, as one user message
    /// holding the OCR prompt and the rasterised page.
    #[tokio::test]
    async fn only_a_blank_page_runs_an_ocr_leg() {
        let seen: Arc<Mutex<Vec<Value>>> = Default::default();
        let addr = ocr_upstream(seen.clone(), "scanned page two").await;
        let app = mock_app(
            &format!("[server]\n[providers.p]\nbase_url = \"http://{addr}/c\"\nmodels = [\"reads\"]\n"),
            "ocr",
        );

        let parsed = pdf_to_markdown(&app, FIXTURE, Engine::Auto, Some("p/reads"), 20, &ctx())
            .await
            .unwrap();
        assert!(parsed.complete);
        let md = parsed.markdown;

        assert!(md.contains("Hello, pxy."), "page 1 keeps poppler's text: {md}");
        assert!(md.contains("scanned page two"), "{md}");
        let bodies = seen.lock().unwrap();
        assert_eq!(bodies.len(), 1, "only the blank page is worth a model call");
        let msgs = bodies[0]["messages"].as_array().unwrap();
        assert_eq!(msgs.len(), 1, "one user message: {:?}", bodies[0]);
        assert_eq!(msgs[0]["role"], "user");
        let parts = msgs[0]["content"].as_array().unwrap();
        assert_eq!(parts[0]["text"], OCR_PROMPT);
        let url = parts[1]["image_url"]["url"].as_str().unwrap();
        assert!(url.starts_with("data:image/png;base64,"), "{url}");
    }

    /// Without an OCR model a blank page is reported, never sent anywhere.
    #[tokio::test]
    async fn a_blank_page_without_an_ocr_model_is_reported() {
        let app = mock_app("[server]\n", "noocr");
        let parsed = pdf_to_markdown(&app, FIXTURE, Engine::Auto, None, 20, &ctx())
            .await
            .unwrap();
        assert!(parsed.complete, "no OCR wanted means nothing failed");
        assert!(parsed.markdown.contains("[page 2: no text]"), "{}", parsed.markdown);
    }

    /// An OCR leg that fails leaves the note standing for this turn and marks
    /// the parse incomplete, so the caller does not cache it under the file's
    /// hash forever.
    #[tokio::test]
    async fn a_failed_ocr_leg_marks_the_parse_incomplete() {
        let app = mock_app(
            "[server]\n[providers.p]\nbase_url = \"http://127.0.0.1:1/c\"\nmodels = [\"reads\"]\n",
            "ocrfail",
        );
        let parsed = pdf_to_markdown(&app, FIXTURE, Engine::Auto, Some("p/reads"), 20, &ctx())
            .await
            .unwrap();
        assert!(!parsed.complete);
        assert!(parsed.markdown.contains("Hello, pxy."), "{}", parsed.markdown);
        assert!(parsed.markdown.contains("[page 2: no text]"), "{}", parsed.markdown);
    }

    fn ctx() -> ClientContext {
        ClientContext::default()
    }

    /// A chat upstream that answers every call with `answer` and keeps the
    /// bodies it was sent.
    async fn ocr_upstream(sink: Arc<Mutex<Vec<Value>>>, answer: &'static str) -> String {
        use axum::routing::post;
        let router = axum::Router::new().route(
            "/c",
            post(move |axum::Json(body): axum::Json<Value>| {
                let sink = sink.clone();
                async move {
                    sink.lock().unwrap().push(body);
                    axum::Json(json!({"id": "x", "choices": [{"index": 0,
                        "message": {"role": "assistant", "content": answer},
                        "finish_reason": "stop"}]}))
                }
            }),
        );
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move { axum::serve(listener, router).await.unwrap() });
        addr.to_string()
    }

    /// A minimal app, mirroring `server_tools`' test app.
    fn mock_app(cfg_toml: &str, name: &str) -> Arc<App> {
        let cfg: crate::config::Config = toml::from_str(cfg_toml).unwrap();
        let catalog = crate::catalog::Catalog::from_config(&cfg);
        let dir = std::env::temp_dir().join(format!("pxy-pdf-test-{}-{name}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        Arc::new(App {
            catalog,
            secrets: crate::secrets::Secrets::new(),
            state: crate::state::State::open(&dir.join("s.sqlite")).unwrap(),
            http: reqwest::Client::new(),
            cfg,
        })
    }
}
