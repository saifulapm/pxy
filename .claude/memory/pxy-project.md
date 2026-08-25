---
name: pxy-project
description: "pxy = tiny Rust LLM proxy (OmniRoute replacement), feature-complete and in production; read HANDOFF.md first, then docs/09 for the gap audit; creds in pass under AI/"
metadata: 
  node_type: memory
  type: project
  originSessionId: 624ccdf0-7b6b-465c-874f-f3a18da125c7
  modified: 2026-08-25T21:23:45.762Z
---

Saiful's `pxy` (repo `~/Sites/github/omniroute`, github.com/saifulapm/pxy) replaced OmniRoute:
one local endpoint (:4100) over ~30 providers, free-first `auto` routing with failover/retries,
`pxy launch claude|opencode|pi|codex`. **Read `HANDOFF.md` first in any session** — it is the
living doc (state, invariants, catalog, next steps). `docs/09-omniroute-litellm-gap-audit.md`
holds the 2026-08-26 deep comparison against OmniRoute/litellm; its prioritized queue is DONE
(bug round + claude-oauth provider + textual tool-call extraction + 429 classification +
@@usage), leaving only optional DX items. Credentials live in `pass` under `AI/<provider>/…`;
the `claude` provider borrows `~/.claude/.credentials.json` (rotating tokens, write-back).
Hard rule from Saiful: **Anthropic models never go in `auto`** — reserve tier, manual only.
`references/` (OmniRoute, litellm, opencode checkouts) is gitignored; verify against it, never
guess APIs. Restart pxy after config/pass changes. Calendar: Sep 6 2026 — remove openrouter
GMI promo entries.
