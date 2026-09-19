//! JSON repair for the `response-healing` plugin (the page is wiki:plugins;
//! rustdoc would read a `plugins.md` link as an intra-doc one and warn, since
//! the repo carries no such file).
//!
//! A model asked for JSON answers with JSON wrapped in a Markdown fence, or
//! prefaced by a sentence, or with a trailing comma its training data was full
//! of — and a client that runs `JSON.parse` on `content` fails on all three.
//! The repair is deliberately conservative: it returns `Some` only when the
//! rebuilt text parses, so a body no closer set rescues (a `max_tokens`
//! truncation mid-token, say) reaches the client exactly as the upstream sent
//! it.

/// Candidate starts tried from each of the two ranks below. Every one costs a
/// scan of the text after it, so an unbounded search is quadratic in the
/// length of the answer. The cap is applied per rank, after ranking, so it
/// bites on the runs in prose and never on the value itself.
const MAX_CANDIDATES: usize = 64;

/// Repair a JSON answer wrapped in prose, a fence or its own mistakes.
///
/// Every `{` or `[` in the text starts a candidate, rebuilt up to the close
/// that matches it while quoting bare keys, dropping trailing commas, and
/// closing whatever strings and containers were left open. Picking the right
/// candidate is the whole problem: a model's prose is full of runs that parse
/// on their own, and serving one of those instead of the answer is silent,
/// because nothing in the response says a heal happened.
///
/// Where a candidate sits decides it, never how big it is. A model puts its
/// answer at the start of a line — alone, inside a fence, or after a preamble
/// that ended with a newline — while a `[1]` citation marker, a bare `[]` or a
/// list in a sentence sits partway into one. So the earliest candidate that
/// starts a line and parses is the answer, whether the prose around it is
/// longer or shorter. Only when nothing starts a line does reach decide, and
/// there it is measured to the same origin for every candidate.
///
/// `None` when nothing parses, and `None` when the winner is just the input
/// again, so a caller can treat `Some` as "this needed healing and the healed
/// form is valid".
pub fn heal_json(text: &str) -> Option<String> {
    let chars: Vec<char> = text.chars().collect();
    let (mut opens_line, mut mid_line) = (Vec::new(), Vec::new());
    // True while nothing but whitespace has been seen since the last newline.
    let mut fresh = true;
    for (i, c) in chars.iter().enumerate() {
        if *c == '{' || *c == '[' {
            if fresh { &mut opens_line } else { &mut mid_line }.push(i);
        }
        match c {
            '\n' => fresh = true,
            c if c.is_whitespace() => {}
            _ => fresh = false,
        }
    }

    let mut healed = None;
    for start in opens_line.into_iter().take(MAX_CANDIDATES) {
        let (candidate, _) = rebuild(&chars[start..]);
        if parses(&candidate) {
            healed = Some(candidate);
            break;
        }
    }
    if healed.is_none() {
        // Nothing on a line of its own: the furthest into the text a candidate
        // reaches is the best evidence left of which one is the value.
        let mut furthest = 0;
        for start in mid_line.into_iter().take(MAX_CANDIDATES) {
            let (candidate, consumed) = rebuild(&chars[start..]);
            if parses(&candidate) && start + consumed > furthest {
                furthest = start + consumed;
                healed = Some(candidate);
            }
        }
    }

    let healed = healed?;
    (healed != text.trim()).then_some(healed)
}

fn parses(candidate: &str) -> bool {
    serde_json::from_str::<serde_json::Value>(candidate).is_ok()
}

/// Copy `src` up to the close that matches its first container, repairing as
/// it goes, and report how much of `src` that took. Everything outside a
/// string is understood; everything inside one is passed through byte for
/// byte, so escapes and braces in string values survive untouched.
fn rebuild(src: &[char]) -> (String, usize) {
    let mut out = String::new();
    // One entry per open container: the closer it wants, and — for an object —
    // whether the next token is a key, which is the only place a bare word may
    // be quoted.
    let mut stack: Vec<(char, bool)> = Vec::new();
    let mut in_string = false;
    let mut escaped = false;
    let mut i = 0;
    while i < src.len() {
        let c = src[i];
        i += 1;
        if in_string {
            out.push(c);
            if escaped {
                escaped = false;
            } else if c == '\\' {
                escaped = true;
            } else if c == '"' {
                in_string = false;
            }
            continue;
        }
        match c {
            '"' => {
                in_string = true;
                out.push(c);
            }
            '{' => {
                stack.push(('}', true));
                out.push(c);
            }
            '[' => {
                stack.push((']', false));
                out.push(c);
            }
            '}' | ']' => {
                out.push(c);
                stack.pop();
                // The value is complete: whatever the model wrote after it is
                // commentary, not JSON.
                if stack.is_empty() {
                    break;
                }
            }
            ',' => {
                // A comma with nothing but a closer (or the end of a truncated
                // body) after it is the trailing comma no JSON parser accepts.
                if matches!(
                    src[i..].iter().find(|c| !c.is_whitespace()),
                    None | Some('}') | Some(']')
                ) {
                    continue;
                }
                out.push(c);
                if let Some((closer, expect_key)) = stack.last_mut()
                    && *closer == '}'
                {
                    *expect_key = true;
                }
            }
            ':' => {
                out.push(c);
                if let Some((_, expect_key)) = stack.last_mut() {
                    *expect_key = false;
                }
            }
            _ if c.is_whitespace() => out.push(c),
            _ => {
                let bare_key = stack.last().is_some_and(|(cl, key)| *cl == '}' && *key);
                let word = src[i - 1..]
                    .iter()
                    .take_while(|c| c.is_alphanumeric() || "_$-.".contains(**c))
                    .count();
                if bare_key && word > 0 {
                    out.push('"');
                    out.extend(&src[i - 1..i - 1 + word]);
                    out.push('"');
                    i += word - 1;
                } else {
                    out.push(c);
                }
            }
        }
    }
    if in_string {
        if escaped {
            // A lone trailing backslash would escape the quote we are about to
            // add, reopening the string it is meant to close.
            out.pop();
        }
        out.push('"');
    }
    while let Some((closer, _)) = stack.pop() {
        out.push(closer);
    }
    (out, i)
}

#[cfg(test)]
mod tests {
    use super::heal_json;

    /// The five shapes wiki:plugins documents, each with the exact
    /// text a client must end up able to parse.
    #[test]
    fn the_documented_shapes_heal() {
        for (name, input, want) in [
            ("missing closer", "{\"ok\": true", "{\"ok\": true}"),
            (
                "fenced",
                "```json\n{\"ok\": true}\n```",
                "{\"ok\": true}",
            ),
            (
                "prose before",
                "Here is the JSON you asked for:\n{\"ok\": true}",
                "{\"ok\": true}",
            ),
            (
                "trailing comma",
                "{\"a\": 1, \"b\": 2,}",
                "{\"a\": 1, \"b\": 2}",
            ),
            ("bare keys", "{ok: true}", "{\"ok\": true}"),
        ] {
            assert_eq!(heal_json(input).as_deref(), Some(want), "{name}");
        }
    }

    #[test]
    fn a_truncated_object_closes() {
        // Cut off mid-value: braces and brackets appended in the right order
        // make it parse, and the trailing comma before EOF goes with them.
        assert_eq!(
            heal_json("{\"a\": {\"b\": [1, 2").as_deref(),
            Some("{\"a\": {\"b\": [1, 2]}}")
        );
        assert_eq!(
            heal_json("{\"a\": [1,").as_deref(),
            Some("{\"a\": [1]}")
        );
        // An open string is closed too.
        assert_eq!(
            heal_json("{\"a\": \"hel").as_deref(),
            Some("{\"a\": \"hel\"}")
        );
    }

    /// A citation marker, a bare `[]` or any short list in the model's own
    /// prose parses on its own, and taking the FIRST bracket would serve it as
    /// the answer — silently, since nothing in the response says a heal
    /// happened. The value that accounts for the most of what the model wrote
    /// wins instead.
    #[test]
    fn a_bracketed_run_in_the_prose_never_beats_the_real_value() {
        assert_eq!(
            heal_json("Per [1] and [2], here is the answer:\n{\"value\": 42}").as_deref(),
            Some("{\"value\": 42}")
        );
        assert_eq!(
            heal_json("Options [] were empty, so:\n[{\"id\": 7}]").as_deref(),
            Some("[{\"id\": 7}]")
        );
        // A trailing run must not displace the value whatever its length, and
        // a preamble's run must not win by being the longer of the two: the
        // deciding fact is where each one sits, never how big it is.
        assert_eq!(
            heal_json("{\"a\": 1}\n\nHope that helps [1]").as_deref(),
            Some("{\"a\": 1}")
        );
        assert_eq!(
            heal_json("{\"ok\": true}\n\nNotes: [\"alpha\", \"beta\", \"gamma\", \"delta\"]")
                .as_deref(),
            Some("{\"ok\": true}"),
            "a long trailing list must not displace a short value"
        );
        assert_eq!(
            heal_json("The keys [\"a\", \"b\", \"c\"] are covered below:\n{\"a\": 1}").as_deref(),
            Some("{\"a\": 1}"),
            "a long preamble list must not displace a short value"
        );
        // With no candidate on a line of its own, reach is all there is to go
        // on, and it still has to pass over the citation.
        assert_eq!(
            heal_json("See [1]: the value is {\"a\": 1}.").as_deref(),
            Some("{\"a\": 1}")
        );
    }

    #[test]
    fn valid_json_and_hopeless_text_are_left_alone() {
        assert_eq!(heal_json("{\"a\": 1}"), None, "already valid");
        assert_eq!(heal_json("[1, 2, 3]"), None, "already valid");
        assert_eq!(heal_json("  {\"a\": 1}\n"), None, "valid but for whitespace");
        assert_eq!(heal_json("I could not answer that."), None, "no JSON at all");
        assert_eq!(heal_json(""), None, "empty");
        // Truncated mid-key: no closer set makes this parse, so it stands.
        assert_eq!(heal_json("{\"a\": 1, \"b"), None, "key with no value");
    }
}
