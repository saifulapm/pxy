# Memory index

- [Router invariants](router-invariants.md) — check router.rs changes against HANDOFF.md invariants; stream reads bounded only by 600s reqwest timeout
- [Media mirrors chat](media-mirrors-chat.md) — src/media hand-copies classify_error semantics and drifts; diff status/cooldown handling against chat every time
- [Cooldown mirror tests](state-cooldown-mirror.md) — lazy expiry masks sqlite-mirror bugs; test via db rows/escalation level, never cooldown() output
