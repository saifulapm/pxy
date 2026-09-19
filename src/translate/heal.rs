//! JSON repair for the `response-healing` plugin (see [plugins](plugins.md)).
//!
//! A model asked for JSON answers with JSON wrapped in a Markdown fence, or
//! prefaced by a sentence, or with a trailing comma its training data was full
//! of — and a client that runs `JSON.parse` on `content` fails on all three.
//! The repair is deliberately conservative: it returns `Some` only when the
//! rebuilt text parses, so a body no closer set rescues (a `max_tokens`
//! truncation mid-token, say) reaches the client exactly as the upstream sent
//! it.

/// Repair a JSON answer wrapped in prose, a fence or its own mistakes.
///
/// Scans from the first `{` or `[` to the close that matches it — which drops
/// a fence, a preamble and any commentary after the value in one move — while
/// quoting bare keys, dropping trailing commas, and closing whatever strings
/// and containers the text left open. `None` when the result does not parse,
/// and `None` when it is just the input again, so a caller can treat `Some` as
/// "this needed healing and the healed form is valid".
pub fn heal_json(text: &str) -> Option<String> {
    let chars: Vec<char> = text.chars().collect();
    let start = chars.iter().position(|c| *c == '{' || *c == '[')?;
    let healed = rebuild(&chars[start..]);
    if healed == text.trim() || serde_json::from_str::<serde_json::Value>(&healed).is_err() {
        return None;
    }
    Some(healed)
}

/// Copy `src` up to the close that matches its first container, repairing as
/// it goes. Everything outside a string is understood; everything inside one
/// is passed through byte for byte, so escapes and braces in string values
/// survive untouched.
fn rebuild(src: &[char]) -> String {
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
    out
}

#[cfg(test)]
mod tests {
    use super::heal_json;

    /// The five shapes [plugins](plugins.md) documents, each with the exact
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
