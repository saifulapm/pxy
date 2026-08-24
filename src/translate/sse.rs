/// Incremental server-sent-events parser: feed byte chunks, get whole events.
#[derive(Default)]
pub struct SseParser {
    buf: String,
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
        self.buf.push_str(&String::from_utf8_lossy(bytes));
        let mut events = Vec::new();
        // Events are separated by a blank line. Handle \r\n by trimming \r.
        while let Some(pos) = find_event_end(&self.buf) {
            let raw = self.buf[..pos.start].to_string();
            self.buf.drain(..pos.end);
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
}
