//! AWS `application/vnd.amazon.eventstream` frame decoder.
//!
//! Wire format (identical across Kinesis / Transcribe / Bedrock / CodeWhisperer):
//!
//! ```text
//! +----------------+----------------+---------------+
//! | total_len u32  | headers_len u32| prelude_crc32 |   12-byte prelude
//! +----------------+----------------+---------------+
//! | headers (headers_len bytes)                     |
//! +-------------------------------------------------+
//! | payload (total_len - headers_len - 16 bytes)    |
//! +-------------------------------------------------+
//! | message_crc32                                   |
//! +-------------------------------------------------+
//! ```
//!
//! All integers big-endian. Each header is
//! `name_len u8 | name | value_type u8 | value`, and CodeWhisperer only ever
//! sends type 7 (string: `len u16 | utf8`) — the other 8 types are decoded far
//! enough to skip them correctly.
//!
//! CRCs are not verified: the transport is TLS, a mismatch would mean a
//! framing bug we cannot recover from anyway, and skipping them keeps this
//! dependency-free (no crc32 crate).

/// One decoded frame. `event_type` comes from the `:event-type` header.
#[derive(Debug, Clone, PartialEq)]
pub struct Frame {
    pub event_type: String,
    pub message_type: String,
    pub payload: Vec<u8>,
}

/// Incremental decoder: feed arbitrary byte chunks, pull whole frames out.
#[derive(Default)]
pub struct EventStreamDecoder {
    buf: Vec<u8>,
}

impl EventStreamDecoder {
    pub fn new() -> Self {
        Self { buf: Vec::new() }
    }

    pub fn push(&mut self, bytes: &[u8]) {
        self.buf.extend_from_slice(bytes);
    }

    /// Pop every complete frame currently buffered. A trailing partial frame
    /// stays buffered for the next chunk.
    pub fn drain(&mut self) -> Vec<Frame> {
        let mut out = Vec::new();
        loop {
            match self.next_frame() {
                Some(f) => out.push(f),
                None => break,
            }
        }
        out
    }

    fn next_frame(&mut self) -> Option<Frame> {
        if self.buf.len() < 12 {
            return None;
        }
        let total_len = be_u32(&self.buf[0..4]) as usize;
        let headers_len = be_u32(&self.buf[4..8]) as usize;
        // Guard against a corrupt length driving an enormous allocation/wait.
        if total_len < 16 || headers_len > total_len - 16 {
            // Unrecoverable framing error: drop everything rather than spin.
            self.buf.clear();
            return None;
        }
        if self.buf.len() < total_len {
            return None;
        }

        let headers = &self.buf[12..12 + headers_len];
        let payload = self.buf[12 + headers_len..total_len - 4].to_vec();
        let (mut event_type, mut message_type) = (String::new(), String::new());
        for (name, value) in parse_headers(headers) {
            match name.as_str() {
                ":event-type" => event_type = value,
                ":message-type" => message_type = value,
                // :exception-type marks error frames; surface it as the event
                // type so callers can react without a second lookup.
                ":exception-type" if event_type.is_empty() => event_type = value,
                _ => {}
            }
        }

        self.buf.drain(..total_len);
        Some(Frame {
            event_type,
            message_type,
            payload,
        })
    }
}

/// Decode `name -> value` pairs; non-string values are skipped (never sent by
/// CodeWhisperer) but still consume the right number of bytes.
fn parse_headers(mut h: &[u8]) -> Vec<(String, String)> {
    let mut out = Vec::new();
    while h.len() >= 2 {
        let name_len = h[0] as usize;
        if h.len() < 1 + name_len + 1 {
            break;
        }
        let name = String::from_utf8_lossy(&h[1..1 + name_len]).to_string();
        let vtype = h[1 + name_len];
        let rest = &h[2 + name_len..];
        let (value, consumed) = match vtype {
            // 0 = true, 1 = false: value is implicit in the type
            0 | 1 => (String::new(), 0),
            2 => (String::new(), 1),  // byte
            3 => (String::new(), 2),  // short
            4 => (String::new(), 4),  // integer
            5 => (String::new(), 8),  // long
            6 | 7 => {
                // 6 = byte array, 7 = string; both are u16-length-prefixed
                if rest.len() < 2 {
                    break;
                }
                let len = ((rest[0] as usize) << 8) | rest[1] as usize;
                if rest.len() < 2 + len {
                    break;
                }
                (
                    String::from_utf8_lossy(&rest[2..2 + len]).to_string(),
                    2 + len,
                )
            }
            8 => (String::new(), 8),   // timestamp
            9 => (String::new(), 16),  // uuid
            _ => break,                // unknown type: cannot skip safely
        };
        if rest.len() < consumed {
            break;
        }
        out.push((name, value));
        h = &rest[consumed..];
    }
    out
}

fn be_u32(b: &[u8]) -> u32 {
    u32::from_be_bytes([b[0], b[1], b[2], b[3]])
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a frame the way AWS does, so the decoder is tested against the
    /// spec rather than against itself.
    fn frame(event_type: &str, payload: &[u8]) -> Vec<u8> {
        let mut headers = Vec::new();
        for (name, value) in [(":event-type", event_type), (":message-type", "event")] {
            headers.push(name.len() as u8);
            headers.extend_from_slice(name.as_bytes());
            headers.push(7); // string
            headers.extend_from_slice(&(value.len() as u16).to_be_bytes());
            headers.extend_from_slice(value.as_bytes());
        }
        let total = 12 + headers.len() + payload.len() + 4;
        let mut out = Vec::new();
        out.extend_from_slice(&(total as u32).to_be_bytes());
        out.extend_from_slice(&(headers.len() as u32).to_be_bytes());
        out.extend_from_slice(&0u32.to_be_bytes()); // prelude crc (unverified)
        out.extend_from_slice(&headers);
        out.extend_from_slice(payload);
        out.extend_from_slice(&0u32.to_be_bytes()); // message crc (unverified)
        out
    }

    #[test]
    fn decodes_a_single_frame() {
        let mut d = EventStreamDecoder::new();
        d.push(&frame("assistantResponseEvent", br#"{"content":"hi"}"#));
        let frames = d.drain();
        assert_eq!(frames.len(), 1);
        assert_eq!(frames[0].event_type, "assistantResponseEvent");
        assert_eq!(frames[0].message_type, "event");
        assert_eq!(frames[0].payload, br#"{"content":"hi"}"#);
    }

    #[test]
    fn reassembles_frames_split_across_chunks() {
        let bytes = frame("assistantResponseEvent", br#"{"content":"split"}"#);
        let (a, b) = bytes.split_at(7);
        let mut d = EventStreamDecoder::new();
        d.push(a);
        assert!(d.drain().is_empty(), "partial prelude must not yield a frame");
        d.push(b);
        let frames = d.drain();
        assert_eq!(frames.len(), 1);
        assert_eq!(frames[0].payload, br#"{"content":"split"}"#);
    }

    #[test]
    fn decodes_multiple_frames_in_one_chunk() {
        let mut bytes = frame("assistantResponseEvent", b"a");
        bytes.extend(frame("toolUseEvent", b"b"));
        bytes.extend(frame("assistantResponseEvent", b"c"));
        let mut d = EventStreamDecoder::new();
        d.push(&bytes);
        let frames = d.drain();
        assert_eq!(frames.len(), 3);
        assert_eq!(frames[1].event_type, "toolUseEvent");
        assert_eq!(frames[2].payload, b"c");
    }

    /// Real bytes captured from CodeWhisperer (claude-haiku-4.5, "Reply with
    /// exactly: OK") — the decoder is only trustworthy if it handles the
    /// actual wire output, not just our own encoder.
    #[test]
    fn decodes_a_real_codewhisperer_response() {
        let bytes = include_bytes!("testdata_kiro_response.bin");
        let mut d = EventStreamDecoder::new();
        d.push(bytes);
        let frames = d.drain();

        let types: Vec<&str> = frames.iter().map(|f| f.event_type.as_str()).collect();
        assert_eq!(
            types,
            ["assistantResponseEvent", "contextUsageEvent", "meteringEvent"]
        );

        let text = String::from_utf8_lossy(&frames[0].payload);
        let v: serde_json::Value = serde_json::from_str(&text).unwrap();
        assert_eq!(v["content"], "OK");
        assert_eq!(v["modelId"], "claude-haiku-4.5");

        // metering carries the credit cost — the only real usage signal
        let m: serde_json::Value =
            serde_json::from_slice(&frames[2].payload).unwrap();
        assert_eq!(m["unit"], "credit");
        assert!(m["usage"].as_f64().unwrap() > 0.0);
    }

    /// The same real response, delivered one byte at a time.
    #[test]
    fn decodes_a_real_response_byte_by_byte() {
        let bytes = include_bytes!("testdata_kiro_response.bin");
        let mut d = EventStreamDecoder::new();
        let mut frames = Vec::new();
        for b in bytes.iter() {
            d.push(&[*b]);
            frames.extend(d.drain());
        }
        assert_eq!(frames.len(), 3);
        assert_eq!(frames[0].event_type, "assistantResponseEvent");
    }

    #[test]
    fn empty_payload_frame_is_valid() {
        let mut d = EventStreamDecoder::new();
        d.push(&frame("messageMetadataEvent", b""));
        let frames = d.drain();
        assert_eq!(frames.len(), 1);
        assert!(frames[0].payload.is_empty());
    }
}
