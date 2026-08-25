# OmniRoute + litellm gap audit — 2026-08-26

Deep comparison sweeps of `references/OmniRoute` (11 parallel agents) and
`references/litellm` (re-verification pass) against pxy as of commit `b59b71e`.
Everything below is agent-reported with file/line citations — **verify each
claim against the code before acting on it** (this same sweep caught our own
research docs 03/04/06 being wrong repeatedly, see §9).

Earlier rounds already shipped from these comparisons: in-request retries,
mid-stream failover (pre-first-event commit), disconnect accounting, media
failover chains, cooldown persistence, drop_params, multi-candidate 404
carve-outs (chat + media).

## 1. Probable BUGS in pxy (outrank all features)

- **B1 — empty thinking-signature poison.** `translate/anthropic_to_openai.rs`
  (~:292 response, ~:445 stream) emits `{"type":"thinking","signature":""}`
  for OpenAI-upstream reasoning. Claude Code stores + replays the block; the
  Anthropic→Anthropic passthrough (`router.rs` `payload.clone()`) sends it to
  a real Anthropic-format upstream → `400 Invalid signature in thinking
  block`, forever (block lives in history). In OmniRoute this manifested as
  silent permanent pinning to non-Anthropic legs. Fix both ends: never mint
  `signature: ""`; strip thinking blocks with empty signature on the
  Anthropic request path. (OmniRoute: `openai-to-claude.ts:605-720`,
  `defaultThinkingSignature.ts`.)
- **L1 — context-window 400s are fatal and kill the `auto` chain.** Our
  chars/4 `estimate_tokens` under-counts code/tool-schema/base64 payloads; a
  128k candidate passes the filter, upstream 400s "maximum context length",
  `classify_error` treats 400 as Fatal → chain dies while 1M-ctx candidates
  sit unused. litellm: `ExceptionCheckers.is_error_str_context_window_exceeded`
  (`litellm_core_utils/exception_mapping_utils.py:74-99`) — nine substrings,
  two false-positive exclusions (`string_above_max_length`, `invalid 'user'`);
  always fails over. Same shape as our zenmux 404 carve-out. Refinement: on
  such a 400, also skip remaining candidates with equal/smaller context.
- **L2 — no tool-pair repair in the OpenAI→Anthropic direction** (the strict
  one). `repair_tool_pairs` exists only in `anthropic_to_openai.rs`. Anthropic
  hard-400s a `tool_use` without a following `tool_result` — live whenever
  opencode/codex route onto kiro/kimi/tabitoken/gorouter/agentrouter-anthropic.
  Trigger: Ctrl-C'd turns (the reason Drop-accounting exists). Compounds with
  L1 (the 400 is fatal). litellm: `factory.py:2037-2067` `_add_missing_tool_results`.
- **L3 — three more Anthropic 400 triggers on the same path**: final assistant
  message trailing whitespace (rstrip needed); first message must be `user`;
  empty `{"type":"text","text":""}` array parts pass through (we only guard a
  bare empty string). litellm `factory.py:2350-2372` (empty-text rewrite is
  unconditional because agent frameworks routinely emit them).
- **L4 — Anthropic cache tokens never counted.** `TokenUsage::from_anthropic`
  and both streaming taps read only `input_tokens`/`output_tokens`, which
  EXCLUDE `cache_creation_input_tokens`/`cache_read_input_tokens`. Claude Code
  caches aggressively → most real input on Anthropic-native providers is
  invisible to budgets and `pxy status`. Likely why tabitoken's ~7k hidden
  tokens/call are hard to see. litellm: `backfill_missing_cache_usage_fields`
  (fills only when None — never clobber real values with zero).
  Accounting trap the other way (OmniRoute `claude-to-openai.ts:190-243`):
  when *translating* usage to OpenAI shape, `prompt_tokens = input +
  cache_read`, exclude cache_creation (Anthropic pads creation to a
  1024-token minimum → 250× inflation on tiny calls); OpenAI `prompt_tokens`
  INCLUDES cache while Anthropic `input_tokens` excludes it — reading the
  wrong field double-counts.
- **B2 — `tool_call` capability is probed then ignored.** `ModelSpec.tool_call`
  populated by `pxy refresh`, never read by `check_candidate`; a tools request
  can route to a text-only model (burns the call, returns prose). No vision
  field at all. OmniRoute rule: vision compatibility is `=== true` only.
- **B4 — `parse_retry_after` only parses bare integers.** `Retry-After: 5m`
  (groq style) falls through to default backoff. Parse duration forms first.
- **B5 — no quota-exhausted cooldown class.** Backoff caps at 120s, so a
  drained daily tier with no header is re-probed every 2 min until midnight
  (OmniRoute measured 285×429 in 48h on one account). Proper fix is §2.3
  (429 body classification with window-sized cooldowns); a naive substring
  match is dangerous ("exceeded your current quota" collides with Gemini's
  transient 429 boilerplate).
- **B6 — `/v1/models` lists `auto` without `context_length`.** A missing/zero
  context window disables opencode auto-compaction entirely. Rule: min() over
  chain members — `launch.rs` already computes exactly this for
  CLAUDE_CODE_AUTO_COMPACT_WINDOW.
- **B7 — `drop_params` is top-level only.** Can't strip
  `thinking.budget_tokens` etc.; needs dotted paths + empty-parent pruning
  (sending `{"thinking":{}}` is itself a 400).
- **B3 — video handler speaks only the Agnes job shape** (`video_id`,
  `status=="completed"`, `metadata.url`). DashScope uses `output.task_id` /
  `output.task_status=="SUCCEEDED"` / `output.video_url` → alibaba video
  cannot work despite the dashscope dialect. OmniRoute makes the four paths
  config (`videoGeneration/job.ts::VIDEO_JOB_PRESETS`) — four dot-path TOML
  keys unlock DashScope/Sora/MuAPI/kie.
- **Chain-exhaustion status is always 429 `overloaded_error`** regardless of
  cause (router.rs). OmniRoute classifies attempt kinds: all-model-4xx →
  preserve that 4xx; all-same-kind → that status; mixed → 504 if any timeout
  else 502; only attach retry-after when surfacing 429/503.
- **L7 — `/api/event_logging/batch` stub**: Claude Code telemetry POSTs 404
  against pxy; a `{"status":"ok"}` discard stub stops retry churn. Trivial.

## 2. Free-quota features (HIGH)

1. **Codex rate-limit reset credits** — `src/lib/usage/codexResetCredits.ts`:
   `GET chatgpt.com/backend-api/wham/rate-limit-reset-credits`, `POST
   …/consume {redeem_request_id, credit_id}`; banked credits that INSTANTLY
   reset an exhausted window; redeem soonest-expiry-first. The only quota
   *recovery* mechanism found anywhere.
2. **Quota-window opening** — `warmupScheduler.ts`: Claude 5h windows start on
   first use; a max_tokens:1 ping on a cron places the boundary. Codex
   variant (`quotaAutoPing.ts`): reset-slide-driven, pings only when resetAt
   drifts ≥30s, must FULLY drain the SSE body (window starts on stream
   completion).
3. **429 body classification + window-sized cooldowns** —
   `classify429.ts` + `quotaTextCooldowns.ts`: `rate_limit|quota_exhausted|
   transient` from ~25 regexes; decision order matters (terminal-quota wins;
   upstream delay <3600s forces rate_limit even if quota keywords matched);
   cooldown horizon = window size (1h/5h/24h/weekly/until-midnight). Fixes B5.
4. **Headroom-aware routing** — pxy's `balance_url` data feeds only `pxy
   status`; OmniRoute's `headroomRanking.ts` is a stable single-key sort:
   `1 − max(util_5h, util_7d)` desc, ties by config index, original order on
   error. 32 quota-endpoint definitions in `open-sse/services/usage/*.ts`
   worth taking verbatim — incl. Anthropic `GET /api/oauth/usage`
   (`anthropic-beta: oauth-2025-04-20`, per-model-family 7d windows).
   Mandatory companions: 250ms fetch throttle (burst got a Codex token
   REVOKED) and separate cooldown for the usage endpoint itself.
5. **Meta/background request detection** — `backgroundTaskDetector.ts`:
   header `x-initiator: background`; max_tokens ∈ (0,50); or 18 system-prompt
   substrings AND ≤3 user messages. pxy covers only Claude Code via launch
   env; opencode/codex title calls unhandled. Should select a cheaper chain.
6. **Claude Code security-classifier short-circuit** —
   `claudeClassifierCompat.ts`: in auto permission mode CC fires a
   `/v1/messages` call ("You are a security monitor for autonomous AI coding
   agents…") needing `<block>no</block>`-style replies; anything else fails
   closed and blocks the tool call. Answer locally (opt-in — technically
   auto-approve). Saves one call per tool use.
7. **Prompt-cache preservation set**: strip Anthropic's rotating
   `x-anthropic-billing-header:` line from system prompts before non-Anthropic
   upstreams (2 lines, kills every cache hit otherwise); OpenAI/Codex do
   implicit prefix caching with NO cache_control marker (never gate prefix
   care on its presence); inject anything before the last user turn, never at
   top; session→account affinity pin with failure-class-specific eviction.
8. **cache_control breakpoint injection** (`claudeHelper.ts:299-535`): exactly
   4 breakpoints (system-last, 2nd-to-last user, last assistant block, last
   non-defer tool); re-anchor when hoisting moves a client's marker; Anthropic
   requires longer-TTL-first ordering in system[].
9. **Free providers**: treat OmniRoute's ~40-entry list SKEPTICALLY — it
   contradicts our live testing (docs/08 verified OVHcloud dead; Cerebras is
   8K-ctx/5RPM for us; SambaNova 20 req/day). Credible: NVIDIA NIM (already
   on our list), maybe llm7 / morph (fast-apply edit model) / ant-ling —
   re-verify each. Flags worth copying into our catalog: `hardStopGuaranteed`,
   `trainsOnPrompts`.
10. **Keyless non-chat**: Context7 fetch (anonymous,
    `GET context7.com/api/v1/{lib}?type=llms.txt&topic=` — library docs as
    markdown, high value for agents); DuckDuckGo Lite scraper as terminal
    search fallback (~150 lines); gtts / edge-tts; aihorde images (anonymous
    key `"0000000000"`).
11. **Free-tier accounting method** (`freeModelCatalog.ts`): daily×30;
    RPD×~800tok×30; RPM/TPM-only = `recurring-uncapped`, never summed;
    pool-dedupe by poolKey taking max (variant double-count inflated 60M→462M);
    signup credits = 0 in the headline.

## 3. Agentic reliability — request pipeline

- **Commit on first CONTENT, not first event** (`validateQuality.ts`): pxy's
  pre-commit gate accepts a bare `message_start`. OmniRoute peeks until real
  content (tool_use block start, input_json_delta even empty, non-empty
  text/thinking delta) and treats as failures: "streaming empty content
  block" (DeepSeek/GLM tool-heavy mode), truncation without finish_reason,
  empty completion, exhaustion marker inside a 200 body (scanned in
  error.* fields only, NEVER message content). Non-streaming 200 bodies get
  zero inspection in pxy today.
- **Early keepalive / heartbeat** (`earlyStreamKeepalive.ts`,
  `sseHeartbeat.ts`): pxy holds fresh streams up to 10s sending ZERO bytes
  (headers included); codex's reqwest drops idle connections ~5s. Heartbeat
  shape must match client dialect (`event: ping` anthropic / empty-delta
  openai / response.in_progress); `:` comment keepalives crash strict OpenAI
  clients — off by default.
- **`finish_reason` normalization**: force `"tool_calls"` when tool_calls
  non-empty but upstream said `"stop"` — 3 lines; clients gate tool execution
  on it (`passthroughToolNames.ts`).
- **Empty-response rejection** (`streamEmptyChoices.ts`): all-empty-choices
  200 stream ends as a clean empty turn → clients retry to their cap with no
  error to stop on. Companion throughput watchdog counts only assistant text
  (any reasoning/tool event suspends judgement).
- **Terminal-status policy** for chain failure (see bugs list) +
  `unavailableRetryGate.ts` (retry-after only on 429/503).
- **Cooldown details**: cap backoff independently of Retry-After; never
  cooldown on self-inflicted timeout (our own deadline); third lock namespace
  "model missing here (404)" vs "model quota spent (429)"; never shrink an
  active cooldown; IP-bucketed 429 → cool sibling accounts (single egress IP
  applies to us); success HALVES failureCount (decay path — pxy has none).
- **Claude Code constraint enforcement** (`claudeCodeConstraints.ts`):
  thinking → temperature=1 + DELETE top_p; tool_choice forcing a tool →
  delete thinking AND context_management; ≤4 cache_control breakpoints
  counted across system/messages/tools; TTL ordering; default TTL "1h" on
  native OAuth path.
- **Per-account concurrency semaphore** (`accountSemaphore.ts`): FIFO queue
  keyed provider:account, queues instead of failing; for silent-limit
  providers (NVIDIA NIM has no headers, no quota API) 429s come from burst
  parallelism — Claude Code fans out subagents. Lease with 120s expiry so an
  aborted request can't wedge a credential.

## 4. Agentic reliability — translation

- **Text-embedded tool-call extraction** — biggest translation gap. Free
  Qwen/GLM/DeepSeek/NVIDIA emit tool calls as text; pxy passes prose, Claude
  Code stalls. Three dialects (`<invoke name=…>`, `<tool_call>{json}
  </tool_call>`, `TOOL_CALL name: {json}` with brace-depth scanner), matched
  earliest-first, buffered across chunks, emitted at finish_reason time with
  finish_reason overridden to tool_calls. Model the streaming state machine
  on `textualToolCall.ts` (three-valued complete/partial/null with
  character-exact partial detection).
- **Tool-arg streaming**: snapshot-vs-delta disambiguation (`next ===
  existing` keep / startsWith replace / else append — NO fuzzy overlap);
  args arriving as parsed objects; literal 0x0A/0x09 inside JSON strings
  (Gemini) with escape state persisted across deltas per tool index; clamp
  absurd numerics (`limit: 25999999999999999` → CC reject-retry loop).
- **Tool-schema sanitization** (pxy has none outside kiro): `[MaxDepth]`
  placeholders → `{}` only in subschema slots; index-keyed objects → arrays;
  KEEP boolean schemas (coercing `additionalProperties:false` to `{}`
  invites hallucinated args); strip regex-lookaround `pattern` for
  OpenAI/Codex; Gemini stripper keeps pattern/min/max (removing pattern broke
  glob/grep tools); `sanitizeToolId` (`^[A-Za-z0-9_-]+$`, applied to BOTH
  tool_use.id and tool_result.tool_use_id).
- **Tool-name case-insensitive restore**: upstreams return `bash` for `Bash`;
  Gemini always lowercases. Rewrite all four sites or history desyncs.
- **Forward-direction message repair** (`fixToolUseOrdering`,
  `toolResultAdjacency.ts`): consecutive same-role merge with tool_result
  hoisted front; unpaired tool_result → visible text (not dropped);
  tool_result in assistant msg → previous user turn; media-only user turns
  count as real content (else vision input deleted); empty messages[] →
  synthesize "." user turn.
- **Reasoning replay cache** (`reasoningCache.ts`): DeepSeek V4/Kimi/GLM
  thinking models 400 on turn 2 without `reasoning_content` passed back;
  cache keyed by tool_call_id, else sha256 of transcript prefix. Gate via
  models.dev `interleavedField` (we already join models.dev).
- **SSE parser robustness**: strip ANSI/VT100 escapes BEFORE field detection
  (gemini-cli prefixes frames with terminal-redraw escapes → silent stall);
  flush on new `event:` without blank line; `jsonToSse` when a provider
  ignores stream:true.
- **Usage translation trap** — see L4 above.
- **Thinking-budget fitting** (`thinkingBudget.ts`): responseRoom =
  max(max_tokens,1024); target = min(room+budget, modelCap); fitted = target
  − room; <1024 → retry with room=1024; still short → DISABLE thinking rather
  than send invalid. Model caps: sonnet-4.5/4.6 + haiku-4.5 62000, opus-4.5
  32000, sonnet-5/opus-4.6+/fable-5 120000, qwen3-max + glm-5.x 38912,
  gemini-3-flash 0. Delete temperature when thinking active.
- **Reasoning field aliases**: 6 names (`reasoning_content`, `reasoning`,
  `reasoning_text`, `thinking`, `thought`, `reasoning_details[]`) — providers
  using the others lose all thinking output in pxy. `<think>` variants:
  4 tag names, attributes in open tag, newline-terminated unclosed, gated
  per-model (tags can be legit user content).
- **Response sanitizer**: strip `x_groq`/`service_tier` etc. (break OpenAI
  SDK Pydantic); excise leaked Harmony `to=functions.X {…}` envelopes; scrub
  zero-width chars from text fields incl. tool args (corrupt file paths).

## 5. Credentials / OAuth

- **Import from local agent CLIs** — cheapest win: `~/.claude/.credentials.json`
  (Claude Pro/Max) and `~/.codex/auth.json` (ChatGPT) give inference WITHOUT
  implementing OAuth. Also kiro-cli sqlite, `~/.aws/sso/cache`, Cursor
  state.vscdb, Zed keychain, `~/.grok/auth.json`. Readers:
  `src/lib/oauth/utils/*Import.ts`.
- **Write refreshed tokens BACK** to those files (0600, .bak, read-modify-
  write preserving foreign keys like mcpOAuth) — otherwise pxy and the CLI
  fight over rotating refresh tokens.
- **Flows ranked**: Anthropic PKCE (client `9d1c250a-e61b-44d9-88ed-5944d1962f5e`,
  hosted redirect at platform.claude.com — no loopback needed); Codex (client
  `app_EMoamEEZ73f0CkXaXp7hrann`, port 1455 or their custom deviceauth);
  Antigravity (no PKCE, no openid scope — openid hangs consent).
- **Rotation safety**: per-connection mutex with DB write inside; staleness
  re-read; rotation map keyed pbkdf2(oldToken) 60s TTL; rotation GROUPS
  (`codex+openai` share auth0) at concurrency 1; CAS persist guard.
  Counter-intuitive: `prompt=login` on authorize is load-bearing (session
  inheritance → Auth0 revokes first account's token family); sending `scope`
  on Codex refresh REVOKES siblings; refresh as LATE as possible (5 min lead
  for rotating providers).
- **Multiple accounts per provider** — most direct quota multiplier; pxy's
  cooldowns need an account dimension. Kiro: registerClient() PER connection
  (one OIDC registration = one active session).
- **Probe isolation invariant** (`probeOrigin.ts`): health probes must never
  mutate routing state (no cooldowns/lockouts/refresh); fail-safe ON.
  Search-provider "validation" fires real billed queries — beware.
- **Ban tables**: anchor longest unambiguous phrase; Cloudflare `1010` /
  `browser_signature_banned` is a TLS/UA fingerprint rejection NOT a ban
  (curl 200 vs urllib 403 on identical body).

## 6. Routing mechanics

- **Model-variant grammar**: `no-think/<id>` (OpenAI lane: reasoning_effort
  ="none", NOT a delete), `-low/-medium/-high/-xhigh/-max`, `-thinking`,
  `[1m]` (+ anthropic-beta context-1m-2025-08-07), `auto/free` ("any provider
  with free quota left" — highest-value virtual id), `auto/<family>`. Rule:
  never strip a suffix unless the stripped base resolves
  (`registeredEffortVariants.ts` ~35 lines). Strip reserved prefixes BEFORE
  first-slash split.
- **CC discovery aliases**: Claude Code's model picker only lists ids starting
  `claude|anthropic`; mirror catalog as `claude/<id>`, strip server-side;
  set `CLAUDE_CODE_ENABLE_GATEWAY_MODEL_DISCOVERY=1` in launch (pxy doesn't).
  Gives in-session `/model` switching across all free providers.
- **Context filter fixes**: never compare maxInputTokens against
  input+output (double-counts); contextWindow=total vs maxInputTokens=
  input-only (models.dev has both); unknown never removes a candidate.
- **`fallback_only_on_quota_exhaustion` per-step flag**: stop the chain at a
  paid step unless every prior failure was genuine quota exhaustion (~40
  lines; exactly right for free-first + paid reserves).
- **Terminal cooldown class**: banned/expired are NOT cooldowns (never
  expire, never overwritten by transient writes); credits_exhausted is NOT
  terminal (30-min re-probe); re-admission needs 3 successful probes.
- **Reset-aware early release**: fresh quota fetch showing capacity releases
  a cooldown early (CAS; remaining<=0 with no parseable reset stays locked).
  Window-START learning from observed transitions (providers only tell you
  when windows end).
- **Anthropic rate-limit headers missing**: `anthropic-ratelimit-*` family
  unread by check_quota_exhaustion; Anthropic sends RFC3339 dates (our
  parse_reset only does epochs/Go durations); bare-integer heuristic:
  >1.7e9 = epoch else duration.
- **provider/* wildcard chain steps** (keep original entry when no match) +
  startup chain invariants (family/provider allowlists fail at load).
- **Strip-and-retry-once on 400** (`providerFieldStrips.ts`): extract
  offending param from error text, strip, retry once; persist the learned
  rule via pxy refresh into generated.toml. Self-healing drop_params.
  Same pattern for thinking caps (`learnedThinkingCaps.ts`).
- **Per-lane strips + clamps** (`paramSupport.ts` + `maxTokensHelper.ts`):
  regex-on-model-id rules (opus-4 drops temperature; nvidia drops
  prompt_cache_key — codex injects it, NIM 400s); RAISE max_tokens to 32000
  floor when tools present (truncated tool args); port both or neither (the
  azure 16384 clamp exists because the floor trips it).
- **Role normalization**: `developer`→`system` except {openai,azure,github};
  system→first-user for ERNIE/GLM≤5.0.

## 7. Diagnostics / DX

- **`@@om-usage` magic prompt**: last user message exactly that token →
  answer locally with quota summary in the client's protocol shape; zero
  tokens. Best agentic-UX idea in the repo (~150 lines).
- **`pxy doctor`**: ordered checks, ok/warn/fail, JSON mode; PROVE the
  credential by using it, don't check presence.
- **`pxy explain <model>`**: skip-reason vocabulary
  (provider_circuit_open / connection_cooldown / connection_terminal_status /
  model_excluded / model_lockout / no_active_connection) + per-candidate
  decision trace with a CLOSED reason allowlist that throws on unknown.
- **Request artifact capture**: summary in sqlite, bodies on disk; capture
  all 6 stages (client raw → openai hub → provider request/response → client
  response + all three SSE chunk streams); 512KB bound, chunks degrade first;
  mask secrets BEFORE buffering (3 ReDoS-bounded regexes); /proc/net/tcp
  port→PID attribution labels requests claude/opencode/codex with zero
  client config; HAR 1.2 export ~190 lines.
- **Cache-health analyzer**: classify calls warm/cold/rewrite/uncached;
  relative outlier threshold max(median×10, 1024); their motivating case:
  aggregate ratio looked fine while 18% of calls carried 94% of cache-write.
- **Launch hardening**: --dry-run plan with auth SOURCE (option|env|context)
  and env-diff key names only; codex needs OPENAI_API_KEY/BASE_URL/API_BASE/
  ORG_ID/CODEX_API_KEY all DELETED; gemini/qwen need mkdtemp isolated HOME
  (stored Google OAuth silently overrides proxy launch).
- **Per-model Claude Code profiles** via CLAUDE_CONFIG_DIR
  (`~/.claude/profiles/<name>/settings.json`, never write the token).
- **CLI output contract**: exit codes 0/2/3(offline)/4(auth)/5(quota)/124
  (timeout); stdout=data stderr=progress — pxy's verbs are agent-invoked via
  the skill, machine-readable failure classes are cheap and on-mission.

## 8. Medium tier (compressed)

Context editing = pure param delegation to Anthropic (NOT local emulation);
reusable: edits[] ordering (clear_thinking before clear_tool_uses), 400-
fallback strip-and-retry, `clear_thinking_20251015 keep:"all"` is DEFAULT-ON
for Claude Code clients (CLI-fingerprint emulation). Superseded-Read collapse
(replace earlier Read tool-results when the path is re-read/written — largest
pure-waste category, lossless). Per-image token accounting (1200/image, not
base64-as-text). Timeout tiers with keepAliveTimeout pinned (upstream
Keep-Alive header must not stretch it). Undici lessons: N single-connection
agents (SSE queues behind trailers on shared conns); keep-alive-disabled
dispatcher for connection-error retries. SSRF: re-validate every redirect
hop, reject private DNS per hop, pin connection to validated IP. Search cache
WITH request coalescing (parallel subagents duplicate searches; each dup
burns a Brave credit today). Embeddings dimension-conflict guard (refuse
failover across vector dimensions — we deliberately have no embeddings
failover, keep it that way). More search backends: serper 2500/mo, google-pse
3000/mo, searxng self-hosted. `/v1/images/edits`, `/v1/ocr`,
`/v1/audio/translations` (~30 lines each given existing multipart plumbing).
Config hot-reload with per-section diff. Call-log rotation with BOUNDED
paging (unbounded SELECT on 170MB db crashed their daemon at startup). WAF
burst guard (500ms min gap). Catalog-overlay semantics: per-field local
override beats feed, but feed enabled:false beats even local true (safety
direction). Signed-feed verify over exact wire bytes.

## 9. Corrections to docs/03/04/06 (research-doc drift)

- `bin/aliasResolver.mjs` = Node ESM loader hook, NOT model aliases.
- `src/lib/headroom/` = third-party Python sidecar, NOT quota headroom
  (that's `combo/headroomRanking.ts`).
- `chaos/` is a fan-out feature, not fault injection; `proxySubscription/` is
  a Clash/V2Ray egress manager, not billing.
- Context editing is not local emulation.
- **OmniRoute has NO capability probing** — pxy's tool-call probing is
  strictly ahead; nothing to port there.
- No cross-provider output diff, no local response cache exist anywhere.
- Model-metadata precedence is per-field and INVERTS by field (context
  window: curated beats models.dev; everything else: models.dev beats
  curated).
- OmniRoute's own docs drift badly (FEATURE_FLAGS.md claims 37 flags vs 51 in
  code; free-tier catalogs disagree). Code is truth.

## 10. litellm re-verification — confirmed non-gaps

stream_options include_usage injection, anthropic-beta forwarding, post-
finish_reason usage chunks, cache_control leak (translator rebuilds blocks),
single-candidate cooldown exemption, soonest-recovery = get_min_cooldown,
no empty-response retry in litellm either, response_headers.py is empty,
encrypted_content_affinity moot (our Responses translator synthesizes ids).

## 11. Non-goals confirmed and skipped

Web-cookie executors, TLS-MITM AgentBridge, dashboards/Electron/PWA,
multi-tenant keys/budgets, MCP/A2A/ACP, compression engines, semantic/
scoring/bandit routing, quota pools/DRR, i18n, Docker/k8s/tunnels,
enterprise RBAC/audit, gamification, integrations, batches/files, music,
hedging (burns two free accounts per answer), shadow routing (actively
harmful), and the stealth/ban-evasion material (JA3/JA4 impersonation,
zero-width client-name obfuscation) — noted only for awareness.
