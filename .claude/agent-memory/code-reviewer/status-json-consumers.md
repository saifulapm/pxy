---
name: status-json-consumers
description: pxy status --json is parsed by ~/.dotfiles/bin/{codex,opencode}-usage-scan; review dedupe rules and stdout purity whenever status output or model_usage changes
metadata:
  type: project
---

`pxy status --json` (server.rs print_status) is a machine contract consumed by
`~/.dotfiles/bin/codex-usage-scan`, `opencode-usage-scan`, and
`pxy-panel-scan` (json.loads on the WHOLE stdout; panel-scan also parses
`pxy explain auto --json` and `pxy models`), feeding the Quickshell panels
(AiModel.js, Pxy.qml/PxyModel.js). Stdout purity was fixed 2026-08-26
(tracing → stderr in main.rs) — keep it that way.

**Why:** confirmed a real double-count defect at review time (2026-08-26): the
codex scanner dedupes pxy-routed turns only by `turn_context.model == "auto"`,
but pxy records model_usage for ALL agent=codex traffic — `pxy launch codex
<concrete-model>` sessions (live sessions had model "github-free/gpt-5-mini")
get counted natively AND merged from pxy. The opencode scanner's rule
(`providerID == "pxy"`, catches every pxy message regardless of model) is the
correct pattern.

**How to apply:** when reviewing changes to print_status, model_usage, or the
scanners: (1) check each scanner's skip rule covers ALL pxy-routed traffic,
not just "auto"; (2) stdout must stay pure JSON — tracing_subscriber::fmt()
writes to STDOUT by default, so any info!/warn! reachable from the status path
(e.g. claude.rs oauth-refresh info!, or RUST_LOG in the panel's env) silently
breaks both scanners; (3) day strings must stay local-date on both sides
(jiff Zoned::now().date() vs Python datetime.now()); (4) pxy model ids are
bare/per-provider ("gpt-5-mini", "stealth/ox-alpha") while agents log
prefixed ids — same tokens can split across two model rows.
