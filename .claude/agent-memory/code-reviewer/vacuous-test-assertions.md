---
name: vacuous-test-assertions
description: pxy tests repeatedly assert routing/skip behavior their mocks cannot distinguish — demand a call counter or distinct routes before trusting a "skips X" test
metadata:
  type: project
---

Recurring pattern (confirmed twice): a pxy test's name/comment claims a
skip/ordering behavior, but the mock setup makes the assertion pass even with
the feature deleted.

**Why:** Second confirmed instance in the 2026-08-26 review:
`context_window_400_fails_over_and_skips_smaller_peers` (src/router.rs) points
`small` and `tiny` at the same mock route and only asserts the final provider +
no cooldown — without the peer-skip, `tiny` would 400 identically and the test
still lands on `big` and passes. First instance: the cooldown-persistence test
documented in [[state-cooldown-mirror]]. Third instance (2026-08-26): the
catalog "claude/" alias test used a config where every model id has a single
owner, so the strip-precedence hijack ([[catalog-bare-id-order]]) was
invisible — sparse test *configs* are the same trap as shared mock routes:
give colliding ids to two providers when testing resolution precedence.

**How to apply:** Any test claiming "candidate X was skipped / not called"
must assert on a per-route call counter (the `drop_params` test's `Mutex`
capture pattern) or give each candidate a distinct route whose invocation
changes the observable outcome. Otherwise treat the claimed behavior as
unverified and re-derive it from the code.
