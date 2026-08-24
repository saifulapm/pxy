---
name: feedback-use-find-docs
description: Saiful - never guess APIs/schemas/config formats; verify via find-docs skill or reference source code first
metadata: 
  node_type: memory
  type: feedback
  originSessionId: dd64f805-74b1-4224-a4bf-4b27cf9ed21e
  modified: 2026-08-24T11:58:19.646Z
---

Saiful: "Never guess anything, instead use find-docs skills."

**Why:** Training-data API knowledge goes stale (crate APIs, provider endpoints, agent config
schemas); guessed details cause silent breakage.

**How to apply:** Before writing code against any external surface (Rust crate APIs, provider
HTTP endpoints, Claude Code/opencode/pi config schemas), verify with the find-docs skill or by
reading the actual source in `references/` ([[pxy-project]]). Applies to version numbers,
feature flags, header names, JSON shapes.
