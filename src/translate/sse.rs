/// Incremental server-sent-events parser: feed byte chunks, get whole events.
#[derive(Default)]
pub struct SseParser {
    /// Partial event text (always valid UTF-8): bytes after the last complete
    /// event, awaiting the event's terminator.
    text: String,
    /// Undecodable tail: the start of a multi-byte char whose remaining bytes
    /// arrive in a later chunk. Never more than 3 bytes in the common case;
    /// genuinely invalid bytes are consumed lossily rather than stalling.
    tail: Vec<u8>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct SseEvent {
    pub event: Option<String>,
    pub data: String,
}

impl SseParser {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn feed(&mut self, bytes: &[u8]) -> Vec<SseEvent> {
        self.tail.extend_from_slice(bytes);
        // Decode only up to the last valid UTF-8 boundary: a multi-byte char
        // split across chunk boundaries must stay buffered until its
        // remaining bytes arrive, not become U+FFFD. `error_len() == None`
        // means an incomplete (possibly split) char — hold it; `Some(_)`
        // means genuinely invalid bytes — consume them lossily so a corrupt
        // stream can never wedge the parser. Loop until no progress: one
        // chunk can carry several invalid bytes in a row.
        while !self.tail.is_empty() {
            let decodable = match std::str::from_utf8(&self.tail) {
                Ok(_) => self.tail.len(),
                Err(e) => match e.error_len() {
                    None => e.valid_up_to(),
                    Some(bad) => e.valid_up_to() + bad,
                },
            };
            if decodable == 0 {
                break;
            }
            let decoded = String::from_utf8_lossy(&self.tail[..decodable]).into_owned();
            self.tail.drain(..decodable);
            self.text.push_str(&decoded);
        }

        let mut events = Vec::new();
        // Events are separated by a blank line. Handle \r\n by trimming \r.
        while let Some(pos) = find_event_end(&self.text) {
            let raw = self.text[..pos.start].to_string();
            self.text.drain(..pos.end);
            if let Some(ev) = parse_event(&raw) {
                events.push(ev);
            }
        }
        events
    }
}

struct EventEnd {
    start: usize, // length of the event text (before the separator)
    end: usize,   // length including the separator
}

fn find_event_end(buf: &str) -> Option<EventEnd> {
    let lf = buf.find("\n\n").map(|i| EventEnd { start: i, end: i + 2 });
    let crlf = buf.find("\r\n\r\n").map(|i| EventEnd { start: i, end: i + 4 });
    match (lf, crlf) {
        (Some(a), Some(b)) => Some(if a.start <= b.start { a } else { b }),
        (a, b) => a.or(b),
    }
}

fn parse_event(raw: &str) -> Option<SseEvent> {
    let mut event = None;
    let mut data_lines: Vec<&str> = Vec::new();
    for line in raw.lines() {
        let line = line.trim_end_matches('\r');
        if let Some(rest) = line.strip_prefix("event:") {
            event = Some(rest.trim().to_string());
        } else if let Some(rest) = line.strip_prefix("data:") {
            data_lines.push(rest.strip_prefix(' ').unwrap_or(rest));
        }
        // comments (":") and other fields ignored
    }
    if event.is_none() && data_lines.is_empty() {
        return None;
    }
    Some(SseEvent { event, data: data_lines.join("\n") })
}

/// Serialize an SSE event in Anthropic style (`event:` + `data:`).
pub fn format_event(event: &str, data: &serde_json::Value) -> String {
    format!("event: {event}\ndata: {data}\n\n")
}

/// Serialize an OpenAI-style data-only SSE chunk.
pub fn format_data(data: &serde_json::Value) -> String {
    format!("data: {data}\n\n")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_split_chunks() {
        let mut p = SseParser::new();
        assert!(p.feed(b"event: message_start\nda").is_empty());
        let evs = p.feed(b"ta: {\"a\":1}\n\ndata: [DONE]\n\n");
        assert_eq!(evs.len(), 2);
        assert_eq!(evs[0].event.as_deref(), Some("message_start"));
        assert_eq!(evs[0].data, "{\"a\":1}");
        assert_eq!(evs[1].data, "[DONE]");
    }

    #[test]
    fn handles_crlf() {
        let mut p = SseParser::new();
        let evs = p.feed(b"data: x\r\n\r\n");
        assert_eq!(evs.len(), 1);
        assert_eq!(evs[0].data, "x");
    }

    /// A multi-byte char split across chunk boundaries must survive intact:
    /// TCP chunks split at arbitrary byte offsets, and the old lossy decode
    /// turned each half into U+FFFD in the client-visible text.
    #[test]
    fn multibyte_char_split_across_chunks_survives() {
        // 3-byte CJK char, split 1 / 2.
        let mut p = SseParser::new();
        assert!(p.feed(b"data: \xe4").is_empty());
        assert!(p.feed(b"\xbd").is_empty());
        let evs = p.feed(b"\xa0 hello\n\n");
        assert_eq!(evs[0].data, "\u{4f60} hello"); // 你
        // 4-byte emoji, split 2 / 2, mid-event with text after.
        let mut p = SseParser::new();
        assert!(p.feed(b"data: a\xf0\x9f").is_empty());
        let evs = p.feed(b"\x98\x80b\n\n");
        assert_eq!(evs[0].data, "a\u{1f600}b");
    }

    /// Genuinely invalid bytes must not wedge the parser: consume them
    /// lossily (U+FFFD) the way the old per-chunk lossy decode did.
    #[test]
    fn invalid_utf8_is_replaced_not_stalled() {
        let mut p = SseParser::new();
        let evs = p.feed(b"data: \xff\xfe ok\n\n");
        assert_eq!(evs.len(), 1);
        assert_eq!(evs[0].data, "\u{fffd}\u{fffd} ok");
        // And the parser still works afterwards.
        let evs = p.feed(b"data: \xc3\xa9\n\n");
        assert_eq!(evs[0].data, "é");
    }

    /// The undecodable tail must not swallow a following complete event:
    /// a split char followed by more events in the same chunk.
    #[test]
    fn split_char_then_complete_event_in_one_chunk() {
        let mut p = SseParser::new();
        assert!(p.feed(b"data: \xe4\xbd").is_empty());
        let evs = p.feed(b"\xa0\n\ndata: two\n\n");
        assert_eq!(evs.len(), 2);
        assert_eq!(evs[0].data, "\u{4f60}");
        assert_eq!(evs[1].data, "two");
    }
}
