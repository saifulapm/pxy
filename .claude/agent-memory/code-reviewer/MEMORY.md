# Memory index

- [Router invariants](router-invariants.md) — check router.rs changes against HANDOFF.md invariants; stream reads bounded only by 600s reqwest timeout
- [Media mirrors chat](media-mirrors-chat.md) — src/media hand-copies classify_error semantics and drifts; diff status/cooldown handling against chat every time
- [Cooldown mirror tests](state-cooldown-mirror.md) — lazy expiry masks sqlite-mirror bugs; test via db rows/escalation level, never cooldown() output
- [Vacuous test assertions](vacuous-test-assertions.md) — "skips X" tests with shared mock routes pass without the feature; demand call counters
- [Stream chunk rewriters](stream-chunk-rewriters.md) — mut-index panics on empty choices (include_usage chunk); copied hold logic drops char-boundary guard
- [Catalog bare-id order](catalog-bare-id-order.md) — provider map is alphabetical BTreeMap not config order; trace resolve changes against live config's colliding claude-* ids
- [status --json consumers](status-json-consumers.md) — dotfiles scanners parse whole stdout; check pxy-dedupe rules (auto-only skip double-counts) and tracing-to-stdout purity
