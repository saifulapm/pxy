---
name: router-invariants
description: Reviews of src/router.rs must be checked against the invariant list in HANDOFF.md (cooldown scopes, raw error passthrough, single-candidate exemption, Kiro binary eventstream)
metadata:
  type: reference
---

HANDOFF.md at the repo root is the authoritative invariant list for pxy's routing engine. Any review of src/router.rs should verify against it, especially:
- Cooldown scope: 401/402/403 provider-wide; 429/408/409/5xx provider/model only (classify_error).
- Fatal error bodies must pass through unmodified (Claude Code auto-retry parses them).
- record_tokens only from real usage fields, exactly once per stream (StreamCtx::finish, guarded by `done`).
- Single-candidate requests bypass the cooldown filter in check_candidate (multi flag).
- Kiro is binary eventstream, not SSE — SSE sniffing/parsing logic must exclude it.

Confirmed pitfall: streaming reads in router.rs are bounded only by the per-request reqwest timeout (provider timeout_secs, default 600s) — any new "read upstream before responding" logic holds the client with zero bytes (no headers) for up to that long, since axum handlers await handle_chat before sending anything.
