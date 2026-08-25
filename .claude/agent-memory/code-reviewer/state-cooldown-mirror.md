---
name: state-cooldown-mirror
description: src/state.rs cooldown sqlite mirror — lazy expiry masks persistence bugs in tests; assert on db rows or escalation level, not cooldown() output
metadata:
  type: project
---

The cooldown sqlite mirror (src/state.rs, added 2026-08) is write-behind and pruned only at `State::open`. The in-memory map's lazy expiry (`cd.until > now` on read) hides unpruned/stale db rows: an expired row that wrongly rehydrates still returns `None` from `cooldown()` and is invisible to `recovery_wait()`.

**Why:** Confirmed in the 2026-08 review: the `cooldowns_survive_restart` test's "expired blip must not resurrect" assertion is vacuous — deleting the prune `DELETE` entirely still passes the whole suite, because lazy expiry masks the row and a `clear_cooldown` earlier in the test deletes it anyway.

**How to apply:** Any test of the cooldown persistence path must assert on something lazy expiry cannot mask: raw `cooldowns` table contents, or the escalation `level` after a follow-up failure (unpruned expired rows resurrect levels; pruned ones reset to 0). Also note the intended lock invariant: map lock and db lock are never held together (set/clear drop the map guard before taking db).
