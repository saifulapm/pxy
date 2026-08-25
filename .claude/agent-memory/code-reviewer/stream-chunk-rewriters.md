---
name: stream-chunk-rewriters
description: router.rs chunk rewriters and ThinkFilter-pattern copies harbor panics — serde_json mut-index on empty choices, and dropped is_char_boundary guards
metadata:
  type: project
---

Two confirmed panic patterns in pxy's per-chunk stream rewriters (both reproduced by probe in the 2026-08-26 review):

1. `&mut v["choices"][0]` — serde_json mut-indexing PANICS on an empty
   array ("cannot access index 0 of JSON array of length 0"). pxy itself
   requests `stream_options.include_usage` on every streamed OpenAI upstream,
   and the spec-compliant usage-only final chunk is `{"choices":[],...}`.
   `rewrite_chunk_think` (router.rs) shipped with this latent; any new
   rewriter copying the pattern (e.g. `rewrite_chunk_tools`) inherits it.
2. Held-prefix logic copied from ThinkFilter tends to drop think.rs's
   `is_char_boundary` guard (think.rs:74). Any `&s[s.len()-take..]` byte
   slice over delta.content panics on multi-byte text (CJK/emoji) — i.e.
   on essentially any non-English stream.

**How to apply:** any new per-chunk rewriter in router.rs or translate/ must
be checked against (a) the `{"choices":[]}` usage chunk and (b) multi-byte
content at every slice point. Same hand-copy-drift family as
[[media-mirrors-chat]].

Pattern 1 recurred in the fx round (2026-08-26): `translate/aisdk.rs
response()` used `out["choices"][0][..].as_array_mut()` — panics on a 200
body with empty OR missing `choices` (probe-verified: "cannot access index 0
of JSON null"). It hits NON-streaming passthroughs too, not just chunk
rewriters. Also check every new dialect's finish-reason mapping in BOTH
directions: fx/codex-style clients enforce closed sets and
consistency rules (fx: `stop`/`other` + tool_calls = invalid_completion
hard error), so a passthrough that skips normalization on the non-streaming
path is a latent turn-killer even when the streaming path normalizes.
