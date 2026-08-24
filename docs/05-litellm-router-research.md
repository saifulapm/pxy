# litellm router design — research (2026-08-24)

Source: exploration of `references/litellm`. Note: `litellm-rust/` in that repo is NOT a routing
reference (provider transforms + axum server only; Python owns routing).

## Architecture

Filter pipeline + picker, in two nested loops: inner retry loop over deployments within a model
group, outer fallback loop across model groups. **Filtering and picking are separate** — every
constraint (cooldown, budget, context fit, tags) narrows the candidate list; the strategy picks
only from survivors. Keeps each constraint independently testable and "why did it pick that"
answerable.

## Error classification — four-stage cascade (THE thing to copy, ~80 lines of Rust)

**Stage 1 — status → error kind:** 400 BadRequest (body sniff promotes to
ContextWindowExceeded / ContentPolicyViolation), 401 Auth, 403 Permission, 404 NotFound,
408 Timeout, 413→BadRequest, 422 Unprocessable, 429 RateLimit, 500/529 Internal, 502 BadGateway,
503 ServiceUnavailable, 504 Timeout. Context/ContentPolicy subclass BadRequest — check
subclasses first.

**Stage 2 — retryable at all?** Retry on 408, 409, 429, ≥500. Nothing else (OpenAI SDK rule).

**Stage 3 — should the router retry this error?** (ordered)
1. ContextWindowExceeded + context fallbacks configured → abort to fallback
2. ContentPolicyViolation + policy fallbacks configured → abort to fallback
3. non-retryable status AND not 401/403 → abort (401/403 carve-out: a bad key must not kill a
   multi-deployment group — retry on a *different* deployment)
4. NotFound → abort
5. RateLimit + zero healthy + fallbacks exist → abort to fallback
6. Auth error + single-deployment group → abort
7. Zero healthy → abort; else retry

**Stage 4 — cool down this deployment?** APIConnectionError (our side) → no; 429/401/408/404 →
yes; other 4xx → no; 5xx/unparseable → yes.

Three-way summary: **retryable** 408/409/429/5xx · **fatal** 400/422/404 · **skip deployment,
keep going** 401/403 (cooldown + try another).

## Retry / backoff

- Default 2 retries.
- **Sleep inversion (3 lines, most important retry insight):** if another healthy deployment
  exists → sleep 0 and switch immediately; back off only when out of options.
- Backoff: obey `Retry-After` (int secs or HTTP-date) **only if in (0, 60]**; else
  `0.5 * 2^n` clamped to 8s; + jitter `0.75 * random()`.
- Per-exception-type retry counts (RetryPolicy) exist; note they bypass stage 3 entirely
  (quirk — a policy can retry a "fatal" 400).

## Cooldown

- Storage: `{timestamp, cooldown_time}`; expiry computed from stored timestamp on read, TTL is
  cleanup only. Default cooldown 5s, allowed_fails 3.
- Default algorithm = error-rate circuit breaker over a 60s window:
  cool down if (429 && group>1) OR (fail rate == 100% && total ≥ 1000) OR
  (fail rate > 50% && total ≥ 5 && group>1) OR fatal status.
  Minimum request count (5) prevents one unlucky failure from removing a deployment.
- **Single-deployment groups are exempt from cooldown** — cooling your only backend turns a
  partial outage into a total one. Hardcoded in three branches.

## Fallbacks

- Three lists: `fallbacks` (general), `context_window_fallbacks`, `content_policy_fallbacks`,
  + `default_fallbacks`; per-model or `"*"` wildcard.
- **Local context-window pre-check**: count input tokens vs `max_input_tokens` per deployment
  BEFORE any network call; if all fail → raise ContextWindowExceeded locally. Cheap, saves a
  round trip.
- Fallback walk recurses (each hop gets its own retry budget) with two guards, both required:
  `max_fallbacks` depth cap (5) AND a shared-by-reference attempted-targets set across the whole
  walk (cycle guard — without it a cyclic fallback graph is exponential).
- Fallback equal to original group is skipped; provider-scoped resources (file ids) refuse
  cross-group fallback.

## Rate limits / budgets

- Per-deployment tpm/rpm: pre-call admission (optimistic local check `>=`, authoritative
  post-increment check `>`), raises 429 with `retry-after: 60`. **Limiter failures fail OPEN**
  (Redis outage must never fail requests).
- Their minute-bucket keys (`HH-MM`, no date) collide daily — port the better scheme:
  `window_start` + counter, reset when `now - window_start >= window`.
- Budgets: `max_budget` + `budget_duration` (s/m/h/d/w/mo + hourly/daily/weekly/monthly words).
  Spend keys carry an explicit window start; increments' TTL shortened to window remainder.
  Over-budget → filtered from candidates (should raise 429-shaped error, not ValueError — their
  bug).
- Token reservation pattern: reserve estimate pre-call (chars/4 + output budget), reconcile
  post-call with `actual - reserved` (can be negative), carrying the reservation's window-start
  so refunds can't land in a fresh window.

## Strategies

Default `simple-shuffle` (weighted random by weight→rpm→tpm, 40 lines) + rate-limit-aware
`usage-based-routing-v2` (drop deployments over tpm/rpm headroom, pick min-TPM with random
tiebreak). **Skip**: least-busy (leaky gauge), cost-based (flawed metric), latency-based,
semantic/adaptive auto-routers. The complexity-router idea (local heuristic scoring → tier →
model, zero API calls) is the only content-aware one worth considering later.

## Aliases / wildcards

- Alias map: flat rewrite, optional `hidden` flag (kept out of /v1/models).
- Wildcards: `re.escape(pattern).replace("\*", "(.*)")`; captured group substituted into target
  model; most-specific-first ordering by (length, meta-char count) — copy this, declaration
  order surprises people.

## Port checklist for pxy (small + load-bearing)

1. Four-stage error cascade as enum + predicates.
2. Retry on 408/409/429/5xx only.
3. 401/403: skip deployment, not request.
4. Sleep-zero-and-switch when alternatives exist.
5. Retry-After (0,60] clamp; else 0.5*2^n cap 8s + jitter.
6. Error-rate breaker with minimum-sample guard; single-deployment exemption.
7. Cooldown state = timestamp + duration, computed on read.
8. Fallback depth cap + shared attempted-set.
9. Local context-window pre-check.
10. Filter-then-pick separation.
11. Limiter infrastructure errors fail open.
