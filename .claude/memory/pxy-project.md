---
name: pxy-project
description: "pxy = tiny Rust LLM proxy replacing OmniRoute; design + decisions live in docs/, credentials in pass under AI/"
metadata: 
  node_type: memory
  type: project
  originSessionId: dd64f805-74b1-4224-a4bf-4b27cf9ed21e
  modified: 2026-08-24T11:48:51.818Z
---

Saiful is replacing OmniRoute (heavy Node proxy at localhost:20128) with `pxy`, a tiny Rust
CLI/daemon: single endpoint over ~25 providers, `auto` model with limit-aware fallback,
`pxy launch claude|opencode|pi`. All research + decisions are in `docs/` of
`~/Sites/github/omniroute` (read `docs/07-pxy-design.md` first — it has the decision table D1-D5
and links the five research docs). Key decisions (2026-08-24): OAuth v1 = github copilot only;
web-cookie providers dropped; non-chat endpoints are Phase 2; source lives in this folder (git
repo, crate at root, `references/` gitignored). All provider credentials were decrypted from
OmniRoute's sqlite and saved in `pass` under `AI/<provider>/<name>` (API keys = first line;
OAuth entries = pure JSON with access_token/refresh_token/expires_at).
