# OmniRoute routing / fallback / limits — research (2026-08-24)

Source: deep exploration of `references/OmniRoute`. This is the blueprint for pxy's `auto`
routing. Paths below are relative to the OmniRoute repo.

## Core structural insight (copy this)

`auto` is **not** "pick one provider". It scores the whole candidate pool once and emits the
full descending-sorted list as an ordered fallback chain, then walks it. Selection and fallback
are the *same data structure*. (`scoreAutoTargets()` in `open-sse/services/combo/autoStrategy.ts:351`,
walked by `handleComboChat` in `open-sse/services/combo.ts`.)

## Candidate pool

- Zero-config `auto`: virtual combo built per request (`open-sse/services/autoCombo/virtualFactory.ts`)
  — all active connections with usable credentials × available models. Adding a provider expands
  the pool with no config edit.
- Filters (resilience-blocked, exclusions, category/tier) apply in sequence; each returns the
  input array unchanged when nothing filtered (no hot-path allocation).
- Pool maxima computed **once per scoring pass**, not per candidate (they had a real O(n²) OOM).

## Scoring

15 factors, weights sum to exactly 1.0 (health .1605, quota .1429, costInv .1429, latencyInv
.1143, taskFit .0762, …). Every factor `clamp01`'d — NaN→0 because NaN sorts nondeterministically.
Health = breaker state (CLOSED 1.0 / HALF_OPEN 0.5 / OPEN 0.0). Unknown quota defaults to 100
(fail-open) — so exhausted-but-unknown providers get **multiplicative post-sum penalties**
instead of extra factors: ×0.7 quota-soft-overage, ×0.5 transient-unavailable.
Mode packs = full weight overrides per variant (coding/fast/cheap/…).

## Failover loop

- Per target: `maxRetries = 1` (2 attempts). Same-model retry only for transient errors
  `[408, 429, 500, 502, 503, 504]`.
- **Account rotation is implicit**: failed connection gets `rateLimitedUntil` written; retry
  re-enters credential selection which filters it out. One mechanism (cooldown write + selection
  filter) does the work of two.
- Context-length-exceeded: if all remaining targets share the same model → abort immediately
  (same context fails identically); if heterogeneous → advance. No cooldown, no breaker increment
  (the account is healthy).

## Error classification — three independent scopes (critical design)

1. **Provider circuit breaker** (whole provider): trips only on 408/500/502/503/504 — NOT
   401/403/429 (those are account problems).
2. **Connection cooldown** (one key/account): `rateLimitedUntil` + exponential backoff
   `base * 2^min(failCount-1, maxLevel)`, base 5s OAuth / 3s API-key, cap 2 min.
3. **Model lockout** (provider+connection+model): in-memory, one bad model doesn't kill the
   connection.

Terminal states (`banned`, `expired`, `credits_exhausted`) never time-expire — only operator
action clears them. Internal contract violations (our own bugs) never punish the provider.
Client aborts / stream lifecycle errors excluded from breaker counting.

A single "provider is down" flag is wrong in both directions — this 3-scope decomposition is
cheap up front and painful to retrofit.

## Breaker + cooldown mechanics

- Breaker: CLOSED → DEGRADED → OPEN → HALF_OPEN with **lazy recovery on read** — no background
  timer; every read refreshes expired OPEN → HALF_OPEN. Eliminates timer leaks + stale state;
  restart rehydration needs no reconciliation. Escalation `resetTimeout * 2^cycles`, capped.
  Live thresholds: 8 fails/60s reset (OAuth), 12/30s (API-key). (AGENTS.md documents a different,
  disabled-by-default layer — code is the truth.)
- No jitter; instead per-connection mutex + dedup windows (5s connection-failure dedup, 10s
  network-error dedup — one VPN drop hitting N targets counts once).
- `Retry-After` honored exactly and **resets backoff to 0**. Parser handles integer secs,
  HTTP-date, epoch s/ms (disambiguate via `> 1e10`), Go durations, free text ("resets in 92h"),
  capped 30 days.
- Self-healing: score <0.2 → excluded 5→30 min; HALF_OPEN probes, full re-admission after 3
  successes; incident mode (>50% breakers open) disables exploration.

## Quota tracking

- Buckets keyed `(pool, unit, window)`; units `percent|requests|tokens|usd`; windows
  `5h|hourly|daily|weekly|monthly`.
- **Two-bucket sliding window computed on read**, no cron:
  `effective = prev * (1 - elapsed/window) + curr`, `bucketIndex = floor(now/windowMs)`,
  epoch-aligned UTC. Single atomic UPSERT per write. Trivially portable to Rust+SQLite.
- `percent` units never accrue local writes — they mirror provider-reported saturation.
- Real reset instants learned from two evidence sources: rate-limit headers
  (`anthropic-ratelimit-*`, `x-ratelimit-*`) AND observed usage *drops* in polling (don't trust a
  reset timestamp that didn't move). Append-only reset-event log, idempotent via UNIQUE +
  INSERT OR IGNORE.
- Headroom = **pessimistic min across windows** (most-exhausted window wins).
- 30s TTL cache in front of usage polling is load-bearing (Anthropic usage endpoint 429s
  under naive polling).
- Anti-pattern found: a hardcoded UTC-3 daily rollover (maintainer's timezone). pxy: make reset
  timezone per-provider config.

## Usage counting

- Tokens from provider `usage` field only, normalized across shapes (`normalizeUsage()`);
  streaming reads terminal SSE events (Claude `message_start`+`message_delta`, OpenAI
  `response.completed`).
- **No tokenizer fallback for accounting** — if usage absent, token dims just don't accrue
  (requests still count). chars/4 estimator used only pre-request for affordability. Don't write
  guessed numbers into a ledger.
- Pricing: layered sources (defaults → litellm sync → models.dev → manual override); subtract
  cache-read/creation tokens from prompt tokens before pricing.

## Session affinity

Three separate mechanisms:
1. Durable pin `(sessionKey, provider) → connectionId`. Session key: headers → body fields →
   **SHA-256 of first user message** (the fallback that makes it work — most clients send no id).
   TTL opt-in, default disabled. On failover, evict pin by exact connection match only.
2. **Prompt-cache affinity: stateless rendezvous hashing** — `argmax over sha256(cache_key + id)`;
   same key → same connection, zero storage, minimal reshuffle on pool change. ~20 lines in Rust;
   cache-locality routing for free.
3. Session model history (analytics only).

## What pxy should steal (ranked by value-per-line)

1. Score-once → ordered-list = fallback chain.
2. Lazy breaker recovery on read (no timers).
3. Implicit account rotation (cooldown write + selection filter).
4. Stateless rendezvous hashing for cache affinity.
5. Two-bucket sliding window counters, atomic UPSERT.
6. Pessimistic min-across-windows headroom.
7. Multiplicative post-sum penalties for soft states.
8. Dedup windows instead of jitter.
9. Asymmetric failure posture: fail-open on infra errors, fail-closed on unknown enum values.
10. Three-scope failure separation (provider / connection / model).

## What pxy should skip

19 routing strategies (weighted scoring + last-known-good covers reality), quota
pool/fair-share/DRR engine (multi-tenant), manifest factors (tierAffinity/specificityMatch),
dead tables (`session_account_affinity`, `tier_assignments` are unused in OmniRoute itself).
