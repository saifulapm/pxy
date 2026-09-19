//! The memory tool's store (wiki:memory-tool): Anthropic's `memory_20250818`
//! command set over one directory on disk. Every path the model sends starts
//! with `/memories`, which maps to the store root, and the strings it reads
//! back — success and error alike — are that tool's own, so a model trained
//! on it needs no explanation.

use serde_json::Value;
use std::path::{Component, Path, PathBuf};

/// The virtual root. A path that does not start here never reaches the disk.
const MEMORIES: &str = "/memories";

/// The one path error, for every way out of the root: a `..`, a path
/// somewhere else, or a symlink pointing out.
const ESCAPE: &str = "path must stay under /memories";

/// Caps (wiki:memory-tool). The file cap bounds what one `create` can put on
/// disk, the line cap keeps the six-column numbering honest, and the view cap
/// keeps one `view` from swallowing the context window.
const MAX_FILE_BYTES: usize = 1024 * 1024;
const MAX_LINES: usize = 999_999;
const MAX_VIEW_CHARS: usize = 16_000;

/// How many lines of context `str_replace` shows around the region it edited.
const SNIPPET_CONTEXT: usize = 4;

/// One agent's memory directory. `run` executes one command against it and
/// answers with the text the model sees.
pub struct MemoryStore {
    root: PathBuf,
}

impl MemoryStore {
    /// The store over `root`. The directory itself is made on first use, so a
    /// store nobody has written to still answers `view /memories`.
    pub fn open(root: PathBuf) -> MemoryStore {
        MemoryStore { root }
    }

    /// Run one command. `Ok` and `Err` are both text for the model: `Err`
    /// carries the `Error: ` prefix the tool puts on every refusal.
    pub fn run(&self, args: &Value, read_only: bool) -> Result<String, String> {
        let command = args["command"].as_str().unwrap_or_default();
        if read_only && command != "view" {
            return Err(err("memory is read-only"));
        }
        match command {
            "view" => self.view(args),
            "create" => self.create(args),
            "str_replace" => self.str_replace(args),
            "insert" => self.insert(args),
            "delete" => self.delete(args),
            "rename" => self.rename(args),
            other => Err(err(&format!("Unknown memory command: {other}"))),
        }
    }

    fn view(&self, args: &Value) -> Result<String, String> {
        let (path, real) = self.path_arg(args, "path", "view")?;
        if real.is_dir() {
            return Ok(listing(&real, &path));
        }
        let text = self.read(&real, &path)?;
        let lines: Vec<&str> = text.lines().collect();
        let (first, shown) = match args.get("view_range") {
            Some(Value::Array(range)) if range.len() == 2 => {
                let start = range[0].as_i64().unwrap_or(0);
                let end = range[1].as_i64().unwrap_or(-1);
                let last = if end == -1 { lines.len() as i64 } else { end };
                if start < 1 || last < start || last > lines.len() as i64 {
                    return Err(err(&format!(
                        "Invalid `view_range` parameter: [{start}, {end}]. It should be within the range of lines of the file: [1, {}]",
                        lines.len()
                    )));
                }
                (start as usize, lines[start as usize - 1..last as usize].join("\n"))
            }
            _ => (1, text),
        };
        let body = numbered(&shown, first);
        Ok(format!("Here's the content of {path} with line numbers:\n{}", cut(body)))
    }

    fn create(&self, args: &Value) -> Result<String, String> {
        let (path, real) = self.path_arg(args, "path", "create")?;
        let text = str_arg(args, "file_text", "create")?;
        check_size(&text)?;
        if let Some(parent) = real.parent() {
            std::fs::create_dir_all(parent).map_err(|e| err(&e.to_string()))?;
        }
        std::fs::write(&real, text).map_err(|e| err(&e.to_string()))?;
        Ok(format!("File created successfully at: {path}"))
    }

    fn str_replace(&self, args: &Value) -> Result<String, String> {
        let (path, real) = self.path_arg(args, "path", "str_replace")?;
        let old = str_arg(args, "old_str", "str_replace")?;
        let new = args["new_str"].as_str().unwrap_or("");
        let text = self.read(&real, &path)?;
        let hits: Vec<usize> = text.match_indices(&old).map(|(at, _)| at).collect();
        match hits.len() {
            0 => {
                return Err(err(&format!(
                    "No replacement was performed, old_str `{old}` did not appear verbatim in {path}."
                )));
            }
            1 => {}
            _ => {
                let lines: Vec<String> =
                    hits.iter().map(|at| line_of(&text, *at).to_string()).collect();
                return Err(err(&format!(
                    "No replacement was performed. Multiple occurrences of old_str `{old}` in lines: {}. Please ensure it is unique",
                    lines.join(", ")
                )));
            }
        }
        let edited = text.replacen(&old, new, 1);
        check_size(&edited)?;
        std::fs::write(&real, &edited).map_err(|e| err(&e.to_string()))?;
        let from = line_of(&text, hits[0]).saturating_sub(SNIPPET_CONTEXT).max(1);
        let to = (line_of(&text, hits[0]) + new.lines().count() + SNIPPET_CONTEXT)
            .min(edited.lines().count().max(1));
        let region: Vec<&str> = edited.lines().skip(from - 1).take(to + 1 - from).collect();
        Ok(format!("The memory file has been edited.\n{}", numbered(&region.join("\n"), from)))
    }

    fn insert(&self, args: &Value) -> Result<String, String> {
        let (path, real) = self.path_arg(args, "path", "insert")?;
        let at = args["insert_line"]
            .as_i64()
            .ok_or_else(|| missing("insert_line", "insert"))?;
        let text = str_arg(args, "insert_text", "insert")?;
        let body = self.read(&real, &path)?;
        let mut lines: Vec<&str> = body.lines().collect();
        if at < 0 || at as usize > lines.len() {
            return Err(err(&format!(
                "Invalid `insert_line` parameter: {at}. It should be within the range of lines of the file: [0, {}]",
                lines.len()
            )));
        }
        let inserted: Vec<&str> = text.lines().collect();
        for (i, line) in inserted.iter().enumerate() {
            lines.insert(at as usize + i, line);
        }
        let edited = format!("{}\n", lines.join("\n"));
        check_size(&edited)?;
        std::fs::write(&real, edited).map_err(|e| err(&e.to_string()))?;
        Ok(format!("The file {path} has been edited."))
    }

    fn delete(&self, args: &Value) -> Result<String, String> {
        let (path, real) = self.path_arg(args, "path", "delete")?;
        if path == MEMORIES {
            return Err(err("The memory directory itself cannot be deleted."));
        }
        if real.is_dir() {
            std::fs::remove_dir_all(&real).map_err(|e| err(&e.to_string()))?;
        } else if real.is_file() {
            std::fs::remove_file(&real).map_err(|e| err(&e.to_string()))?;
        } else {
            return Err(err(&not_found(&path)));
        }
        Ok(format!("Successfully deleted {path}"))
    }

    fn rename(&self, args: &Value) -> Result<String, String> {
        let (old_path, old_real) = self.path_arg(args, "old_path", "rename")?;
        let (new_path, new_real) = self.path_arg(args, "new_path", "rename")?;
        if old_path == MEMORIES {
            return Err(err("The memory directory itself cannot be renamed."));
        }
        if !old_real.exists() {
            return Err(err(&not_found(&old_path)));
        }
        if new_real.exists() {
            return Err(err(&format!("The destination {new_path} already exists")));
        }
        if let Some(parent) = new_real.parent() {
            std::fs::create_dir_all(parent).map_err(|e| err(&e.to_string()))?;
        }
        std::fs::rename(&old_real, &new_real).map_err(|e| err(&e.to_string()))?;
        Ok(format!("Successfully renamed {old_path} to {new_path}"))
    }

    fn read(&self, real: &Path, path: &str) -> Result<String, String> {
        if !real.is_file() {
            return Err(err(&not_found(path)));
        }
        std::fs::read_to_string(real).map_err(|e| err(&e.to_string()))
    }

    /// The named argument as a `/memories` path, plus where it lands on disk.
    fn path_arg(&self, args: &Value, name: &str, command: &str) -> Result<(String, PathBuf), String> {
        let path = str_arg(args, name, command)?;
        let real = self.resolve(&path)?;
        Ok((path, real))
    }

    /// Map a path the model sent onto the root, checked twice as the page
    /// requires: the spelling first, then the canonical path, so a symlink
    /// pointing out of the root is refused as squarely as a `..`.
    fn resolve(&self, path: &str) -> Result<PathBuf, String> {
        let rest = path
            .strip_prefix(MEMORIES)
            .filter(|rest| rest.is_empty() || rest.starts_with('/'))
            .ok_or_else(|| err(ESCAPE))?;
        let relative = Path::new(rest.trim_start_matches('/'));
        if relative.components().any(|c| !matches!(c, Component::Normal(_))) {
            return Err(err(ESCAPE));
        }
        // The root is made here rather than on the first write: `view
        // /memories` is the first call of nearly every turn, and canonicalising
        // needs a directory that exists.
        std::fs::create_dir_all(&self.root).map_err(|e| err(&e.to_string()))?;
        let root = self.root.canonicalize().map_err(|e| err(&e.to_string()))?;
        let real = canonical(&root.join(relative))?;
        if real == root || real.starts_with(&root) { Ok(real) } else { Err(err(ESCAPE)) }
    }
}

/// The path with every existing component resolved. A file that does not
/// exist yet is canonicalised through its nearest existing ancestor, so the
/// answer still tells the caller where the write would land.
fn canonical(path: &Path) -> Result<PathBuf, String> {
    let mut tail = Vec::new();
    let mut probe = path.to_path_buf();
    let mut real = loop {
        match probe.canonicalize() {
            Ok(real) => break real,
            Err(_) => match (probe.file_name(), probe.parent()) {
                (Some(name), Some(parent)) => {
                    tail.push(name.to_os_string());
                    probe = parent.to_path_buf();
                }
                _ => return Err(err(ESCAPE)),
            },
        }
    };
    for name in tail.iter().rev() {
        real.push(name);
    }
    Ok(real)
}

/// The directory's own line first, then everything two levels down, hidden
/// items and node_modules left out.
fn listing(real: &Path, path: &str) -> String {
    let mut lines = vec![entry_line(real, path, true)];
    walk(real, path.trim_end_matches('/'), 2, &mut lines);
    format!(
        "Here're the files and directories up to 2 levels deep in {path}, excluding hidden items and node_modules:\n{}",
        lines.join("\n")
    )
}

fn walk(dir: &Path, prefix: &str, depth: usize, out: &mut Vec<String>) {
    if depth == 0 {
        return;
    }
    let Ok(read) = std::fs::read_dir(dir) else { return };
    let mut entries: Vec<_> = read.flatten().collect();
    entries.sort_by_key(|e| e.file_name());
    for entry in entries {
        let name = entry.file_name().to_string_lossy().to_string();
        if name.starts_with('.') || name == "node_modules" {
            continue;
        }
        let path = format!("{prefix}/{name}");
        let is_dir = entry.path().is_dir();
        out.push(entry_line(&entry.path(), &path, is_dir));
        if is_dir {
            walk(&entry.path(), &path, depth - 1, out);
        }
    }
}

/// `{size}\t{path}`, a directory marked by a trailing slash.
fn entry_line(real: &Path, path: &str, is_dir: bool) -> String {
    let size = std::fs::metadata(real).map(|m| m.len()).unwrap_or(0);
    let slash = if is_dir { "/" } else { "" };
    format!("{size}\t{path}{slash}")
}

/// Lines numbered from `first`, right-aligned in six columns.
fn numbered(text: &str, first: usize) -> String {
    text.lines()
        .enumerate()
        .map(|(i, line)| format!("{:>6}\t{}", first + i, line))
        .collect::<Vec<_>>()
        .join("\n")
}

/// The 1-based line a byte offset falls on.
fn line_of(text: &str, at: usize) -> usize {
    text[..at].matches('\n').count() + 1
}

fn cut(body: String) -> String {
    match body.char_indices().nth(MAX_VIEW_CHARS) {
        Some((at, _)) => format!(
            "{}\n[Cut at {MAX_VIEW_CHARS} characters. Use `view_range` to read the rest.]",
            &body[..at]
        ),
        None => body,
    }
}

fn check_size(text: &str) -> Result<(), String> {
    if text.len() > MAX_FILE_BYTES {
        return Err(err("The file is larger than the 1 MiB memory limit."));
    }
    if text.lines().count() > MAX_LINES {
        return Err(err("The file has more than 999,999 lines."));
    }
    Ok(())
}

fn str_arg(args: &Value, name: &str, command: &str) -> Result<String, String> {
    match args[name].as_str() {
        Some(value) => Ok(value.to_string()),
        None => Err(missing(name, command)),
    }
}

fn missing(name: &str, command: &str) -> String {
    err(&format!("Parameter `{name}` is required for command: {command}."))
}

fn not_found(path: &str) -> String {
    format!("The path {path} does not exist. Please provide a valid path.")
}

fn err(message: &str) -> String {
    format!("Error: {message}")
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use std::sync::atomic::{AtomicU64, Ordering};

    fn store(name: &str) -> MemoryStore {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let dir = std::env::temp_dir().join(format!(
            "pxy-memory-{}-{name}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        let _ = std::fs::remove_dir_all(&dir);
        MemoryStore::open(dir)
    }

    fn run(store: &MemoryStore, args: Value) -> String {
        match store.run(&args, false) {
            Ok(out) => out,
            Err(out) => out,
        }
    }

    #[test]
    fn view_of_an_untouched_store_lists_the_directory_itself() {
        let store = store("empty");
        let out = run(&store, json!({"command": "view", "path": "/memories"}));
        let mut lines = out.lines();
        assert_eq!(
            lines.next().unwrap(),
            "Here're the files and directories up to 2 levels deep in /memories, excluding hidden items and node_modules:"
        );
        assert!(lines.next().unwrap().ends_with("\t/memories/"), "{out}");
        assert_eq!(lines.next(), None, "{out}");
    }

    #[test]
    fn view_lists_two_levels_and_skips_hidden_items() {
        let store = store("listing");
        run(&store, json!({"command": "create", "path": "/memories/notes.md", "file_text": "hi\n"}));
        run(&store, json!({"command": "create", "path": "/memories/a/b/deep.md", "file_text": "d\n"}));
        run(&store, json!({"command": "create", "path": "/memories/.hidden", "file_text": "x\n"}));
        let out = run(&store, json!({"command": "view", "path": "/memories"}));
        let paths: Vec<&str> =
            out.lines().skip(1).map(|l| l.split_once('\t').unwrap().1).collect();
        assert_eq!(paths, vec!["/memories/", "/memories/a/", "/memories/a/b/", "/memories/notes.md"]);
    }

    #[test]
    fn create_overwrites_and_view_numbers_the_lines() {
        let store = store("create");
        let out = run(
            &store,
            json!({"command": "create", "path": "/memories/notes.md", "file_text": "one\ntwo\n"}),
        );
        assert_eq!(out, "File created successfully at: /memories/notes.md");
        let out =
            run(&store, json!({"command": "create", "path": "/memories/notes.md", "file_text": "new\n"}));
        assert_eq!(out, "File created successfully at: /memories/notes.md");
        let out = run(&store, json!({"command": "view", "path": "/memories/notes.md"}));
        assert_eq!(out, "Here's the content of /memories/notes.md with line numbers:\n     1\tnew");
    }

    #[test]
    fn view_range_pages_and_refuses_a_range_past_the_end() {
        let store = store("range");
        run(&store, json!({"command": "create", "path": "/memories/n.md", "file_text": "a\nb\nc\n"}));
        let out =
            run(&store, json!({"command": "view", "path": "/memories/n.md", "view_range": [2, -1]}));
        assert_eq!(
            out,
            "Here's the content of /memories/n.md with line numbers:\n     2\tb\n     3\tc"
        );
        let out =
            run(&store, json!({"command": "view", "path": "/memories/n.md", "view_range": [2, 9]}));
        assert_eq!(
            out,
            "Error: Invalid `view_range` parameter: [2, 9]. It should be within the range of lines of the file: [1, 3]"
        );
    }

    #[test]
    fn a_view_over_sixteen_thousand_characters_is_cut_with_the_paging_line() {
        let store = store("cut");
        let text = "0123456789\n".repeat(3000);
        run(&store, json!({"command": "create", "path": "/memories/big.md", "file_text": text}));
        let out = run(&store, json!({"command": "view", "path": "/memories/big.md"}));
        assert!(
            out.ends_with("\n[Cut at 16000 characters. Use `view_range` to read the rest.]"),
            "{}",
            &out[out.len() - 120..]
        );
    }

    #[test]
    fn str_replace_edits_once_and_shows_the_region() {
        let store = store("replace");
        run(&store, json!({"command": "create", "path": "/memories/n.md", "file_text": "a\nold\nc\n"}));
        let out = run(
            &store,
            json!({"command": "str_replace", "path": "/memories/n.md", "old_str": "old", "new_str": "new"}),
        );
        assert_eq!(out, "The memory file has been edited.\n     1\ta\n     2\tnew\n     3\tc");
    }

    #[test]
    fn str_replace_refuses_a_missing_and_a_repeated_old_str() {
        let store = store("replace-bad");
        run(&store, json!({"command": "create", "path": "/memories/n.md", "file_text": "x\ny\nx\n"}));
        let out = run(
            &store,
            json!({"command": "str_replace", "path": "/memories/n.md", "old_str": "z", "new_str": "q"}),
        );
        assert_eq!(
            out,
            "Error: No replacement was performed, old_str `z` did not appear verbatim in /memories/n.md."
        );
        let out = run(
            &store,
            json!({"command": "str_replace", "path": "/memories/n.md", "old_str": "x", "new_str": "q"}),
        );
        assert_eq!(
            out,
            "Error: No replacement was performed. Multiple occurrences of old_str `x` in lines: 1, 3. Please ensure it is unique"
        );
    }

    #[test]
    fn insert_takes_line_zero_and_refuses_a_line_past_the_end() {
        let store = store("insert");
        run(&store, json!({"command": "create", "path": "/memories/n.md", "file_text": "b\n"}));
        let out = run(
            &store,
            json!({"command": "insert", "path": "/memories/n.md", "insert_line": 0, "insert_text": "a"}),
        );
        assert_eq!(out, "The file /memories/n.md has been edited.");
        let out = run(&store, json!({"command": "view", "path": "/memories/n.md"}));
        assert_eq!(
            out,
            "Here's the content of /memories/n.md with line numbers:\n     1\ta\n     2\tb"
        );
        let out = run(
            &store,
            json!({"command": "insert", "path": "/memories/n.md", "insert_line": 9, "insert_text": "z"}),
        );
        assert_eq!(
            out,
            "Error: Invalid `insert_line` parameter: 9. It should be within the range of lines of the file: [0, 2]"
        );
    }

    #[test]
    fn delete_is_recursive_and_rename_never_overwrites() {
        let store = store("delete-rename");
        run(&store, json!({"command": "create", "path": "/memories/a/b.md", "file_text": "b\n"}));
        run(&store, json!({"command": "create", "path": "/memories/keep.md", "file_text": "k\n"}));
        let out = run(
            &store,
            json!({"command": "rename", "old_path": "/memories/a/b.md", "new_path": "/memories/c.md"}),
        );
        assert_eq!(out, "Successfully renamed /memories/a/b.md to /memories/c.md");
        let out = run(
            &store,
            json!({"command": "rename", "old_path": "/memories/c.md", "new_path": "/memories/keep.md"}),
        );
        assert_eq!(out, "Error: The destination /memories/keep.md already exists");
        let out = run(&store, json!({"command": "delete", "path": "/memories/a"}));
        assert_eq!(out, "Successfully deleted /memories/a");
        let out = run(&store, json!({"command": "delete", "path": "/memories/a"}));
        assert_eq!(
            out,
            "Error: The path /memories/a does not exist. Please provide a valid path."
        );
    }

    #[test]
    fn every_way_out_of_the_root_is_the_same_path_error() {
        let store = store("escape");
        for path in ["/memories/../etc/passwd", "/etc/passwd", "memories/x", "/memoriesx/y"] {
            assert_eq!(
                run(&store, json!({"command": "view", "path": path})),
                format!("Error: {ESCAPE}"),
                "{path}"
            );
        }
    }

    #[test]
    fn a_symlink_pointing_out_of_the_root_is_refused() {
        let store = store("symlink");
        run(&store, json!({"command": "create", "path": "/memories/seed.md", "file_text": "s\n"}));
        let outside = std::env::temp_dir().join(format!("pxy-memory-out-{}", std::process::id()));
        std::fs::create_dir_all(&outside).unwrap();
        std::fs::write(outside.join("secret.md"), "s\n").unwrap();
        std::os::unix::fs::symlink(&outside, store.root.join("out")).unwrap();
        assert_eq!(
            run(&store, json!({"command": "view", "path": "/memories/out/secret.md"})),
            format!("Error: {ESCAPE}")
        );
        let _ = std::fs::remove_dir_all(&outside);
    }

    #[test]
    fn the_root_itself_is_never_deleted_or_renamed() {
        let store = store("root");
        assert_eq!(
            run(&store, json!({"command": "delete", "path": "/memories"})),
            "Error: The memory directory itself cannot be deleted."
        );
        assert_eq!(
            run(&store, json!({"command": "rename", "old_path": "/memories", "new_path": "/memories/x"})),
            "Error: The memory directory itself cannot be renamed."
        );
    }

    #[test]
    fn a_file_over_one_mebibyte_is_refused() {
        let store = store("big");
        let text = "x".repeat(MAX_FILE_BYTES + 1);
        assert_eq!(
            run(&store, json!({"command": "create", "path": "/memories/big.md", "file_text": text})),
            "Error: The file is larger than the 1 MiB memory limit."
        );
        assert!(!store.root.join("big.md").exists());
    }

    #[test]
    fn read_only_refuses_every_command_but_view() {
        let store = store("read-only");
        run(&store, json!({"command": "create", "path": "/memories/n.md", "file_text": "a\n"}));
        for command in ["create", "str_replace", "insert", "delete", "rename"] {
            let args = json!({"command": command, "path": "/memories/n.md", "file_text": "b\n"});
            assert_eq!(store.run(&args, true), Err("Error: memory is read-only".to_string()));
        }
        let args = json!({"command": "view", "path": "/memories/n.md"});
        assert!(store.run(&args, true).is_ok());
    }
}
