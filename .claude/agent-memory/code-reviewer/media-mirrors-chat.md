---
name: media-mirrors-chat
description: src/media re-implements chat failure semantics by hand — always diff media classification against router::classify_error (confirmed drift on 404)
metadata:
  type: project
---

The media module (src/media/mod.rs: `failed_attempt`, `cool_on_failure`, `network_failure`) duplicates chat's failure classification (src/router.rs `classify_error`) rather than sharing code, and its comments claim it mirrors chat "exactly". It does not stay in sync.

**Why:** Confirmed drift in the 2026-08 media-failover review: chat skips 404 when multi-candidate (`multi && status == 404`, delisted-model case) and sets a non-retryable cooldown; media treats 404 as Fatal always and `cool_on_failure` sets no cooldown for 404 — so a delisted model in a media chain blocks failover forever.

**How to apply:** Any review touching src/media failure paths must line-by-line diff the status classification and cooldown scopes against router.rs `classify_error`, including the `multi` special cases. See [[router-invariants]] for the chat-side invariant list.
