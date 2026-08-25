---
name: catalog-bare-id-order
description: catalog.rs bare-id fallback is BTreeMap-alphabetical (docs say "config order" — wrong); many providers in the live config list identical claude-* bare ids, so any new resolve path must be traced against ~/.config/pxy/config.toml
metadata:
  type: project
---

`cfg.providers` is a `BTreeMap` — provider iteration (and therefore
`Catalog::from_config`'s `self.models` order and the bare-id first-match
fallback in `resolve_concrete`) is ALPHABETICAL, not config order, despite
doc comments claiming "config order".

**Why:** Confirmed defect in the 2026-08-26 review: the "claude/" discovery-alias
strip turned `claude/claude-opus-5` into a bare-id lookup, and `agentrouter`
(sorts before `claude`, lists bare `claude-opus-5`) hijacked the Claude-Max
subscription entry — verified live with `pxy explain`. The unit-test config had
single-owner ids, so the test could not see it (see [[vacuous-test-assertions]]).

**How to apply:** Any change to `Catalog::resolve`/`resolve_concrete` that adds
or reorders a path reaching the bare-id fallback must be traced against the
real config (`~/.config/pxy/config.toml` + `generated.toml`), where
agentrouter/gorouter/tabitoken/kiro/github all list overlapping `claude-*`
bare ids. Cheap live check: `cargo run -- explain <id>`.
