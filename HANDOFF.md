# pxy — handoff (as of 2026-08-25)

Read this first in a new session, then `docs/07-pxy-design.md` for design rationale.

## What pxy is

A tiny Rust proxy replacing OmniRoute (a heavy Node router that used ~800 MB RAM).
**pxy uses ~12.5 MB.** One local endpoint over 30 providers, an `auto` model that routes by
priority + quota with automatic failover, and `pxy launch claude|opencode|pi` to wire coding
agents to it. Repo: `github.com/saifulapm/pxy` (private).

## Current state — WORKING and in production use

- **Installed**: `~/.local/bin/pxy`, systemd user unit `pxy.service` (enabled, running).
- **Config**: `~/.config/pxy/config.toml` (not in git; `config.example.toml` mirrors it).
- **Secrets**: `pass` under `AI/<provider>/<name>`, API key on the first line.
  Read-only since 2026-08-31 — pxy never writes a credential back.
- **State**: `~/.local/share/pxy/state.sqlite` (usage counters survive restarts).
- **Verified end-to-end**: Claude Code ran a real session through pxy on the `auto` chain.

### Commands
```sh
pxy serve                     # daemon (systemd runs this)
pxy launch claude|opencode|pi|codex|fx # spawn an agent wired to pxy (--dry-run shows the plan)
pxy models                    # 146 models exposed
pxy status [--remote] [--json] [--provider X]... # usage vs limits; --remote live balances; --json for the panel
pxy route [MODEL|--clear]     # pin the auto route to one model (chain stays as fallback); no arg shows it
pxy explain <model> [--json]  # why each candidate would (not) be routed; --json for the panel
pxy doctor                    # config/daemon/credentials/agents health, exit 1 on FAIL
pxy explain <model>           # why each candidate would (not) be routed right now
pxy refresh [--generate]      # discover catalogs; report drift / write models.toml (a report; pxy never reads it)
pxy search "query" [-n N]     # web search (brave -> jina -> firecrawl)
pxy fetch <url>               # URL -> markdown (jina-reader -> firecrawl)
pxy transcribe <file>         # STT (groq whisper default)
pxy say "text" [-o f.mp3]     # TTS (cloudflare aura default; --voice nova etc.)
pxy image "prompt" [-o f.png] # image gen (cloudflare flux default)
pxy video "prompt" [-o f.mp4] # video gen (agnes v2.0; blocks minutes)
journalctl --user -u pxy -f   # watch routing decisions ("routed" / "failover" lines)
systemctl --user restart pxy  # REQUIRED after any config or pass change (secrets are cached)
```

### Endpoints served
`POST /v1/chat/completions` (OpenAI) · `POST /v1/messages` + `/v1/messages/count_tokens`
(Anthropic) · `POST /v1/responses` (OpenAI Responses API — codex) · `POST /v1/embeddings` ·
`GET /v1/models` · `GET /healthz`.
Both chat protocols translate in both directions, streaming included. /v1/responses
wraps the OpenAI path (translate/responses.rs), so routing/accounting apply unchanged.

**Phase 2 (added 2026-08-25):** `POST /v1/images/generations` (OpenAI shape) ·
`POST /v1/audio/transcriptions` (multipart) · `POST /v1/audio/speech` (binary streamed) ·
`POST /v1/rerank` (Cohere shape) · `POST /v1/videos/generations` (blocking submit+poll) ·
`POST /v1/search` + `POST/GET /v1/fetch` (custom minimal shapes). Handlers live in
`src/media/`; model resolution mirrors embeddings (`provider/model`, bare id, or the
`[media]` default per capability). Media usage + cooldowns are isolated under
`<provider>#media` / `search#<name>` / `fetch#<name>` keys (own rows in `pxy status`) so
they never touch chat budgets; cloudflare's media pool has a hard `daily_requests = 30`
cap because Workers AI overage bills real money.

**Per-agent model accounting (added 2026-08-26):** the desktop usage panel needs
"tokens by model" per agent, but auto-routed agents only ever log "auto" —
pxy is the only party that knows which model answered. So:
- `pxy launch <agent>` tags requests by suffixing the api key
  (`pxy-local:claude` etc., `tagged_key` in launch.rs — one mechanism for all
  five agents; only some can be taught a custom header). server.rs
  `client_agent()` parses it back (explicit `x-pxy-agent` header wins), and
  the agent rides `ClientContext` into the router.
- Every chat record_request/record_tokens also upserts
  `model_usage(day, agent, provider, model, requests, input_tokens,
  output_tokens)` in state.sqlite (day = LOCAL date — the panel groups by
  local days). Enforcement windows are untouched; embeddings/media stay out.
- `pxy status --json [--provider X ...]` emits `{providers, modelUsage,
  remote}`; `--provider` limits the table and the remote HTTP but never
  modelUsage (readers slice by agent). fetch_balance now also returns the raw
  balance body, so `remote.<name>.data` carries e.g. opencode Go's
  rolling/weekly/monthly percent+resetsAt for the panel's meters.
- Consumers: `~/.dotfiles/bin/opencode-usage-scan` (new OpenCode panel tab:
  local opencode.db + pxy agent=opencode rows + both Go accounts' windows as
  extraLimits) and `codex-usage-scan` (native turns whose session_meta says
  `model_provider: "pxy"` count prompts/sessions only; their TOKENS come from
  pxy agent=codex rows — matching by model=="auto" would double-count
  concrete `pxy launch codex -m X` sessions). Claude Code needs nothing: it
  logs the RESOLVED model from pxy's response echo already. agent=pi/fx rows
  are recorded but have no panel consumer yet. CLI logging goes to stderr so
  `status --json` stdout stays parseable.

**Group pin (added 2026-08-26):** `pxy route <model>` pins a group walk to one
model — the pin is walked FIRST on every group request and the configured
chain stays behind it as fallback, so pinning never costs the failover safety
groups exist for. (Originally the "auto-route pin" for the generated `auto`
chain; since e75a49c it applies to hand-written group requests.) The pin lives
in state.sqlite kv
(`route_pin`, canonical `provider/model`), is read per request
(router::resolve_candidates — no daemon restart to take effect), and degrades
to the plain chain when it stops resolving. Explicit model requests ignore it.
`pxy explain` reflects it (`[pinned]` marker; `--json` for machines);
`pxy status` reports it plus active cooldowns. The desktop **pxy panel**
(`~/.dotfiles/shell/Modules/Bar/widgets/Pxy*.qml` + `bin/pxy-panel-scan`)
drives the same verb: route picker over the live walk order with
per-candidate verdicts, cooldown list, per-provider limit meters (remote
balances included), daemon health + restart.

## Architecture (src/)

| file | role |
|---|---|
| `main.rs` | clap CLI: serve / launch / models / status |
| `config.rs` | TOML config types; `SecretRef` (pass/env/cmd/literal) |
| `secrets.rs` | resolves + caches secrets; `pass show` shelling |
| `state.rs` | sqlite usage counters, two-scope cooldowns, rpm sliding window |
| `catalog.rs` | model resolution (`provider/model`, first-slash split, `auto`) |
| `router.rs` | the engine: filter → attempt → classify → failover; streaming tap |
| `server.rs` | axum routes |
| `providers.rs` | URL + auth headers per provider/account (API keys only; sync) |
| `translate/` | anthropic↔openai (request + streaming response), SSE parser, `<think>` filter |
| `translate/responses.rs` | OpenAI Responses API (codex) ↔ chat completions, streaming incl. |
| `translate/aggregate.rs` | SSE stream → complete JSON body (the `force_stream` re-assembly) |
| `refresh.rs` | catalog discovery: provider /models x models.dev join, drift report |
| `launch.rs` | per-agent env/config injection |
| `media/` | Phase 2: images, audio STT/TTS, rerank, video, search, fetch + CLI verbs |

Key invariants worth preserving (learned the hard way, see docs/03 + docs/05):
- **Cooldown scopes**: 401/402/403 → provider-wide; 429/408/409/5xx → `provider/model` only.
  One flaky model must never sideline an account's other models. Cooldowns also carry a
  `retryable` flag: transient (429/5xx/network/stream-death) yes, auth/credits/404 no —
  the in-request retry loop keys off it (a revoked key must never be re-fired).
- **In-request retries (added 2026-08-25)**: when a whole chain walk comes up empty,
  `handle_chat` re-walks up to 2 more times, sleeping until the soonest RETRYABLE cooldown
  expires (max of both scopes per candidate; +2s hint when rpm-limited). It gives up
  immediately when nothing can recover by waiting (hard daily caps, dead keys) or the wait
  exceeds 10s — agents have their own retry logic, fail fast for them. Switching candidates
  still never sleeps (litellm rule).
- **Pre-first-event stream commit (added 2026-08-25)**: a 200 on a streaming request is not
  a commitment. The response is held until the first complete SSE event;
  EOF/transport error/bare `[DONE]` before that → model-scoped cooldown +
  failover, invisible to the client (held bytes are prepended on commit). An error-shaped
  first event is classified by its embedded status via the normal ladder, so a 400-class
  error still passes through raw. Deadline 10s: past it pxy commits and streams as-is
  (openrouter queues free models behind `: PROCESSING` keepalives — alive ≠ dead; failover
  before the deadline needs affirmative evidence of death).
- **404s fail over on multi-candidate walks only** (added 2026-08-25 after zenmux delisted
  glm-5.3-free and its 404 killed the whole `auto` chain): in `auto`, an upstream 404 skips
  the candidate with a model-scoped cooldown + failover log; an explicit single-model
  request still passes the raw 404 through (Claude Code needs the unmodified body).
- **Token accounting** comes only from real `usage` fields; never guess. A client
  disconnect mid-stream (Ctrl-C'd agent turn) still records whatever real usage the
  tap saw — `Drop for StreamCtx` (added 2026-08-25); the upstream billed those tokens.
- **Media failover chains (added 2026-08-25)**: the `[media]` default per capability is
  one id OR an ordered list (`image = ["cloudflare/…", "agnes/…"]` — bare string stays
  valid). "auto"/omitted-model requests walk it: cooldown/cap-gated per candidate,
  provider-side failures (401/402/403/408/409/429/5xx/network) fail over, fatal 4xx
  passes through raw. Upstream 404 on a multi-candidate chain gets the same carve-out
  as chat (non-retryable model cooldown + walk on). A bare id walks every provider
  listing it (alphabetical — BTreeMap); explicit `provider/model` stays single-candidate
  with raw error passthrough. Embeddings deliberately have NO cross-model failover:
  different embedding models produce incompatible vector spaces.
- Error bodies pass through unmodified (Claude Code's auto-retry depends on it).
- 175 tests: `cargo test` (integration tests against local mock upstreams: dead-stream failover, retry-after recovery, auth
  fail-fast, fatal stream-error passthrough, disconnect accounting, media chain failover,
  cooldown persistence, drop_params, context-window failover, tool-capability filtering,
  Anthropic history sanitizing).
- **fx agent support (2026-08-26)**: `pxy launch fx` (vercel-labs/fx). fx speaks a THIRD
  dialect — the Vercel AI SDK LanguageModel spec v4 — at `POST /v3/ai/language-model`:
  `prompt[]` not `messages[]`, model id + streaming as HEADERS, typed SSE parts. pxy
  implements that gateway API locally (translate/aisdk.rs wraps the OpenAI path like
  responses.rs does for codex) and also serves `/coding-agent/v1/{models,credits}`. fx source is
  cloned to gitignored `references/fx` (Zig) — verify protocol claims there.
  Launch needs BOTH `FX_GATEWAY_BASE_URL` (catalog/credits) and `FX_GATEWAY_CHAT_URL`
  (generation — its own var; a base-URL-only override still sends the real token to
  Vercel), plus `AI_GATEWAY_API_KEY` which short-circuits fx's credential chain: no
  login, no refresh, no team lookup, zero traffic off the machine. Overrides are
  loopback-http-with-port only.
  Rules fx's parser enforces (violating any kills the turn): `data: ` needs the space;
  `finishReason` is an OBJECT with `unified` from a closed set; stop/other/error
  TOGETHER WITH tool calls is an invalid completion, so finish reasons are forced to
  `tool-calls` when calls were emitted; usage is nested (`inputTokens.total`);
  duplicate toolCallIds wedge the next turn (ids carry the call index).
  Known gap: pxy's catalog advertises only the `tool-use` tag because ModelSpec has no
  vision/reasoning metadata — so fx never sends image parts or reasoning options
  through pxy (the file-part mapping in aisdk.rs is currently unreachable via fx).
- **DX round (2026-08-26)**: `pxy doctor`, `pxy explain <model>`, and Claude Code
  discovery aliases — /v1/models mirrors every id as `claude/<id>` (stripped in
  catalog.resolve, ONLY when the rest contains a slash or is "auto": a slashless rest
  is a REAL claude-provider model and stripping it would hand the subscription's ids
  to whichever provider sorts first). `pxy launch claude` sets
  CLAUDE_CODE_ENABLE_GATEWAY_MODEL_DISCOVERY=1 → in-session /model switching across
  all providers. sqlite opens carry a 5s busy_timeout so CLI reads never abort
  on a busy daemon. From inside any agent: "@@usage" for quota.
- **Feature round from docs/09 audit (2026-08-26)**:
  - ~~**`claude` provider** (`kind = "claude-oauth"`)~~ — **REMOVED 2026-08-31**
    (see NEXT STEPS §0). It borrowed the Claude Code CLI's own OAuth credential;
    the subscription is now spent by running `claude` natively, beside pxy. A
    provider *named* `claude` is still legal as an ordinary Anthropic **Console**
    API key, and `catalog.resolve` still special-cases the `claude/<id>` mirror.
  - **429 body classification**: window-naming quota text gets a window-sized
    non-retryable cooldown (daily→until provider reset, weekly 4h, monthly 6h,
    credits 1h); header always wins; Gemini transient boilerplate deliberately
    excluded. 402 with no hint waits 1h.
  - **`@@usage` magic prompt**: final user message of exactly `@@usage` answered
    locally (both dialects, streaming included) with day/month usage + cooldowns.
    Zero tokens, no upstream.
  - **Textual tool-call extraction** (translate/tool_text.rs): `<tool_call>{json}
    </tool_call>` and `<invoke name=…><parameter…>` spans in OpenAI-upstream text
    become REAL tool_calls (streaming state machine with cross-chunk assembly,
    finish_reason stop→tool_calls override). Guards: only when the request declared
    tools; extracted names must match a declared tool or the span re-emits
    byte-for-byte; 16KB hold cap. Multi-byte-safe; `choices: []` usage chunks safe
    (both were review-caught panics).
- **Bug round from docs/09 audit (2026-08-26)** — read docs/09 §1 for full context:
  - Never mint `{"type":"thinking","signature":""}` (poisoned client history → permanent
    Anthropic 400s); `translate/anthropic_sanitize.rs` now repairs EVERY Anthropic-bound
    body at one choke point (unverifiable thinking blocks stripped, tool pairs repaired
    both directions, empty text blocks, first-message-user, trailing whitespace, empty
    messages[]). Known residual: a final assistant turn with tool_use whose thinking
    block was stripped can still 400 when the request has `thinking` enabled.
  - Anthropic cache tokens (cache_creation/cache_read) now count as input everywhere.
  - Context-window 400/413/422 on a multi-candidate walk fails over (NO cooldown),
    peer-skips candidates with the same-or-smaller window, and when context was the
    ONLY problem the terminal error is an honest 400 (else stays retryable 429).
  - `tool_call = false` models are skipped for tools requests on multi-candidate walks
    (single-candidate exempt, upstream answers for itself).
  - `Retry-After: 5m`-style durations parse; `drop_params` takes dotted paths with
    empty-parent pruning; `/v1/models` advertises min-over-chain context_length for
    `auto` (a missing window disables opencode auto-compaction); Claude Code telemetry
    POSTs to /api/event_logging/batch get a discard stub instead of 404s.
- **Cooldowns persist across restarts (added 2026-08-25)**: the in-memory map stays
  authoritative at runtime, but every set/clear is mirrored to a sqlite `cooldowns`
  table; `State::open` prunes expired rows and rehydrates the rest (remaining wait,
  level, retryable flag). A deploy mid-day no longer re-probes providers sitting on a
  six-hour quota cooldown. Media/search keys (`p#media`, `search#name`) ride along.
- **`drop_params` (added 2026-08-25)**: per-provider and/or per-model list of top-level
  request-body keys the upstream 400s on (`reasoning_effort`, `top_k`, …), removed after
  translation just before the wire — a picky new free provider needs config, not code.
  `model`/`stream` are pxy's own keys and are ignored if listed. Like every model fact (tool_call,
  force_stream, pinned contexts) it lives in config.toml and nothing overwrites it.

**Routing feature round (added 2026-08-30):**
- **Failure-rate cooldown** (litellm's rule): `state.model_health` tracks
  per-`provider/model` request/failure two-bucket windows (60s); ≥50% of ≥5
  attempts failing skips the model on multi-candidate walks even after each
  individual error cooldown expires (flapping 200/500 upstreams never trip the
  per-error ladder). Real attempts only — pre-filter skips and
  context-window skips don't count; a success repairs the record. In-memory
  only; cooldown persistence already covers the decisive failures.
- **Request-scoped error rules**: `[[providers.X.errors]]` with `match`
  (case-insensitive substring on the error body) + `action` = `skip` |
  `skip-cooldown` | `passthrough` | `passthrough-cooldown`. Checked in
  `classify_error` AFTER the context-window carve-out, BEFORE the status
  ladder; first match wins. Absorbs aggregator/WAF error text
  (agentrouter's 无可用渠道 503s…) without code changes.
- **Session affinity**: conversations stick to their last WINNING candidate
  for prompt-cache locality instead of bouncing back to the chain head after
  a failover. Key ladder: `metadata.user_id` (Claude Code) → `user` (OpenAI
  shape) → FNV-1a of the first message (fixed hash — DefaultHasher is keyed
  per process, which would invalidate every stored binding on restart).
  Binding lives in kv `session:<key>` as `{candidate, seen}`, 1h TTL enforced
  on read; walks first when fresh+listed, BELOW a manual pin, and rebinds on
  every `Done` (self-healing after failover).
- **Multi-account providers**: `[[providers.X.accounts]]` — each entry has
  `name` ([a-z0-9-], part of the state key), `api_key` or `credentials`, and
  optional per-account `headers` that override the provider's same-named
  ones. Mutually exclusive with top-level api_key/credentials.
  `resolve_candidates` expands a bare candidate into per-account candidates
  (config order = fill-first: one account burns before the next starts), so
  the ordinary walk/cooldown machinery works per account unchanged. State
  keys are scoped per account — cooldowns, usage windows, limit enforcement,
  rpm and the failure-rate record all live under `provider#account` (the
  media-key convention) — while `x-pxy-provider` and the `model_usage` table
  keep the BARE provider name so the desktop panel and usage-scan consumers
  never see `#`. `pxy status` sums accounts into the provider row;
  `pxy explain` shows `[account n]` per candidate; refresh discovery
  authenticates as the FIRST account. A provider without `accounts` behaves
  identically to before (implicit default account,
  `state_provider == provider`).
- **Estimator**: `estimate_tokens` counts ASCII codepoints /4 + every
  non-ASCII codepoint ×1 — CJK at chars/4 was the big under-count (400 CJK
  chars ≈ 400 tokens, not 300). The `/v1/messages/count_tokens` endpoint
  inherits the accuracy.

## Provider catalog (27 active; 4 OAuth entries removed 2026-08-31)

**Paid subscriptions (already yours):**
- ~~`github`~~ / ~~`github-free`~~ — **REMOVED 2026-08-31** (docs/10 §0.2: GitHub
  ToS-blocked the copilot-gpt4-service proxy and its suspension emails name "use of
  unsupported clients"; pxy impersonated VS Code). Sanctioned path if ever wanted:
  Copilot SDK behind your own registered OAuth App. Cost of removal: 300 premium
  req/month + the unlimited 0x `gpt-5-mini`.
- `opencode-go` — $10/mo Go plan per account; usage-dollar
  limits $12/5h (rolling), $30/wk, $60/mo, enforced upstream (429 → pxy cooldown +
  account rotation; pxy does no dollar accounting itself). Live utilization:
  `GET /zen/go/v1/usage` with the API key (found in the opencode source, wired into
  `pxy status --remote` 2026-08-25 — shows percent per window + reset time).
  **Merged 2026-08-30 into ONE multi-account provider** (was
  `opencode-go-github` + `opencode-go-google`): two `[[providers.opencode-go.accounts]]`
  (github, google), fill-first walk, per-account cooldowns/usage/limits under
  `opencode-go#<account>` state keys, `status --remote` reports both accounts'
  windows. `ox-alpha-free` started answering "Model not supported" on 2026-08-30
  (upstream churn — both accounts, same error) — re-verify before re-adding.
  **Per-model allowances vary 400×** — table in docs/07. Only high-allowance models are in
  `subscription`; kimi-k3 (110/5h), grok-4.5 (120), qwen3.8-max (160), glm-5.3 (220) are excluded.
  - `muse-spark-1.2-contributor` (45,300/5h): data-collection opt-in was enabled on both
    accounts 2026-08-24, which cleared the `DataPolicyError` — but the model then returned
    **HTTP 500 on every call** (both accounts, all three routes, streaming and not, while
    `hy3` on the same key worked). That's an opencode-side outage, not our config. Retry
    later; keep it out of `auto` until it answers. Note it trains on prompts/completions.
    (Retested 2026-08-25 evening and 2026-08-26: still 500.)

**Free, renewable:**
- ~~`kiro`~~ — **REMOVED 2026-08-31**: kiro.dev's FAQ prohibits third-party
  harnesses verbatim (docs/10 §0.2). Cost: the free 50 credits/month.
  `amazon-q` was already ruled out — same AWS account, same credit pool.
- ~~`kimi-coding`~~ — **REMOVED 2026-08-31**, but NOT for legality: Moonshot
  explicitly permits third-party clients via forward proxy. Removed because its
  credits were dead (bare 500s since activation) and it carried the last of the
  rotating-refresh machinery. If re-added, it must send an honest `pxy/<version>`
  UA — the old code forged `kimi-code-cli/0.26.0`, breaking the one rule Moonshot
  actively screens for.
- `kilocode` — Kilo Code gateway (added 2026-08-25, **Phase 3 #1 — zero Rust needed**):
  the archived device-flow token is a long-lived JWT (exp ~2031, no refresh), so it's a
  plain bearer + `X-KILOCODE-EDITORNAME` header, with a `cmd` secret extracting the token
  from the OAuth JSON in pass. 17 `:free` models; hy3:free + minimax-m3:free in `auto`.
  Paid models refuse cleanly at $0. `kilo-auto/free` router id is broken — skip it.
  ⚠️ Kilo was acquired by Anaconda (announced ~2026-08) — free tiers often shrink
  post-acquisition; if kilocode starts erroring, deprioritize rather than debug.
  (Not to be confused with **Kiro** = AWS's IDE, the Phase 3 amazon-q item.)
- `google` — AI Studio Gemini free tier (added 2026-08-25) via Google's **OpenAI compat
  layer** (`/v1beta/openai/` — docs/08's "needs a Gemini translator" was WRONG, 3rd
  correction). gemini-3-flash-preview (in `auto`), 3.1-flash-lite, 2.5-flash; all 1M ctx,
  tools verified. No billing on the project → hard 429 stop. Unpublished free limits,
  rpm=8 client-side. ⚠️ trains on prompts. Key: `AI/google/main` (project 416746239844).
- `zai` — Z.AI direct (added 2026-08-25): permanently free `glm-4.7-flash` (59.2 SWE-bench,
  agentic-tuned, tool calling verified), `glm-4.5-flash`, `glm-4.6v-flash` (vision).
  ONE concurrent request on the free tier (rpm = 5 in config). Key: `AI/zai/main`.
- `zenmux` — free models; `z-ai/glm-5.3-free` was DELISTED ~2026-08-25 (their free list
  churns — a stale id breaks `auto` because their 404 passes through non-retryably).
  Current: `deepseek/deepseek-v4-flash-vision-exp-free` (1M ctx, in `auto`),
  `z-ai/glm-4.7-flash-free`, `z-ai/glm-4.6v-flash-free`. Needs balance > $0 (anti-abuse;
  never deducted). $5 topped up.
- `openrouter` — `:free` models + `stealth/ox-alpha`; 20 rpm, 1000/day (because ≥$10 lifetime
  credits purchased). ⏰ **GMI Cloud promo until Sep 6 2026**: `minimax/minimax-m3:free`
  (1M ctx, in `auto`) + `minimax-m2.7:free` (196k) — unlimited, tool calling verified,
  $0 cost confirmed per-request. **Remove both + the `auto` entry after Sep 6** (they go paid).
  No safety net: the `promo`/`expires` config key was deleted 2026-08-30, so this is a
  fully manual edit of config.toml.
  Free models never touch the credit balance.
- `aihubmix` — aggregator, free catalog opened ~2026-08 (added 2026-08-25): 50 `-free`
  models on one key, no card. In `auto`: `gemini-3.7-flash-free` (only Gemini in the stack,
  1M ctx), `minimax-m3-free`, `ox-alpha` (plain id, free while in stealth). Kim-series free
  pool usually exhausted ("insufficient promotional resources" — retry); `gpt-5.5-free`
  doesn't route. No published rate limits. Key: `AI/aihubmix/main`.
- `tokenharbor` — TokenHarbor aggregator (added 2026-08-26): free tier is the three `:free`
  ids only (mimo-v2.5, deepseek-v4-flash, qwen3.8-27b — tools verified). The allowance is a
  personal **rolling 7×24h window started by your first free call**, metered by the
  list-price value of the work, so pxy cannot compute it and there is no billing endpoint
  to poll (every `/v1/usage`-shaped path 404s). The only readout is a set of undocumented
  headers on a successful free completion — `x-th-plan`, `x-th-free-used-pct`,
  `x-th-free-resets` — captured by `record_free_allowance` (router.rs) into state.sqlite kv
  `free_quota:<provider>`. `pxy status --remote` reports that snapshot **with its age**
  ("seen 3h ago"): it is only as current as pxy's last call there, so traffic sent outside
  pxy (their connect CLI, the web chat) reads low — the Copilot-counter trap again.
  `--json` carries it in `remote.<name>.data` (`usedPct`/`resetsAt`), which the desktop
  panel meters via `remote_meter` in `bin/pxy-panel-scan`. At 100% the provider is cooled
  non-retryably until `x-th-free-resets`, because the exhaustion 429 names no window the
  body classifier recognizes and the generic ladder would re-probe a dead pool for days.
  ⚠️ The key is live-billing-capable — paid ids are deliberately absent from the config.
- `opencode-zen` — 9 free models, served anonymously (`Bearer public` works!). Shares the Go key.
- `ollama` — free plan: gpt-oss:120b/20b, nemotron-3-*, gemma4:31b only. Flagships need Pro.
- `mistral` (free Experiment tier, phone-verified, opts into data training) + `codestral`
  (separate free key/endpoint).
- `agnes` — flash tier free at $0 balance. Also has image/video models (Phase 2).
- `tokenrouter` — 2 free models of 128. No rate-limit headers; throttles with 503s.
  ⏳ Plus a **50M-token Kimi K3 grant** (2026-08-25 promo): plain `moonshotai/kimi-k3`
  is free-routed on this $0-balance account, in `auto`'s finite tier. No remaining-grant
  readout — when it quota-errors, remove it from `auto`.
- `bai` — exactly 1 free model (`mimo-v2.5`), heavily throttled; other 41 need a deposit.
- `openadapter` — 50/day, 200/month; only 5 small models on the free plan. Counts *failed*
  requests against quota too.
- `groq` — short-call sidecar (added 2026-08-25), **NEVER in `auto`**: gpt-oss-120b/20b +
  qwen3.6-27b at 1000 req/day but 8K TPM (context_length pinned to 8192 in config on
  purpose); `groq/compound(-mini)` at 250/day, 70K TPM with built-in web search — but it
  REJECTS external tools, so it's a question-answering/search utility, not an agent model.
  ~3000 tok/s. No card = hard 429 stop. Whisper models on the same key (Phase 2).
- `cloudflare` — Workers AI on the PAID Workers plan (added 2026-08-25): 10k neurons/day
  free, **overage bills automatically with no block switch**, so pxy enforces
  `daily_tokens = 50000` over cheap models only (worst case ~6k neurons ≈ 60% of free).
  5 tool-verified models incl. deepseek-v4-flash (1.3M ctx). kimi-k2.7-code/glm-5.2
  excluded ($4+/M output). **NEVER put in `auto`** — the cap dies instantly under agentic
  load, and past it, real money. Account id (in base_url) came from the old OmniRoute db.

**Free but finite (use before expiry):**
- `inception` — Mercury 2 diffusion LLM (>1000 tok/s), 100M-token signup grant (added
  2026-08-25), tool calling verified, in `auto`'s finite tier. `mercury-coder` is gated
  to pre-2026-02 accounts. Training-data setting: checked by Saiful in Account
  Settings 2026-08-25 — already in the desired state, nothing to flip. CLOSED.
- ~~`scaleway`~~ — DISABLED same day it was added (2026-08-25): docs/08 was WRONG about
  the "1M free tokens, no card" claim. Signup REQUIRED a card, and the billing API shows
  zero discounts — no free tier exists; every call bills. Key verified working and kept
  in `AI/scaleway/main` (secret key = Bearer); block commented out in config. Re-enable
  only deliberately as paid. ⚠️ docs/08 has now been wrong twice (Vercel, Scaleway) —
  re-verify its remaining claims (Ant Ling, Morph, NVIDIA) against reality before use.
- `tencent` — TokenHub intl; 5 activated models × 1M tokens, expire **2027-08**.
  Full activation sweep 2026-08-25: the five chat grants are exactly hy3, glm-5.3,
  kimi-k3, deepseek-v4-pro-0813, minimax-m3 (all configured). All four kinfra
  embeddings are active — the VL pair (`kinfra-vl-embedding-2b/8b`, image+text fused
  into ONE embedding, 2048/4096-dim) lives on a separate path
  `/v1/embeddings/multimodal` with content-part input, served via the `tencent-vl`
  provider entry. **Dead without paid billing**: hy-mt2-* translation
  (INSUFFICIENT_BALANCE), hy-3d-3.1 (BILLING_PROVISION_FAILED on real calls — a
  max_tokens=1 probe deceptively succeeds), hy-3d-* pipeline stages (upstream
  unreachable), tripo-3d (no trial quota).
- `alibaba` — Model Studio; ~5M tokens across Qwen models, expire **~Nov 2026** (90-day).
  ⚠️ Enable "Free Quota Only" in the console or it silently switches to pay-as-you-go.
  Added 2026-08-25 (all verified live via the native DashScope
  multimodal-generation endpoint, `kind = "dashscope"` media dialect):
  `qwen-image-3.0` + `z-image-turbo` (images), `qwen3-tts-flash` (TTS, answers a
  signed OSS wav URL that pxy fetches and relays), `qwen3-asr-flash` (ASR, accepts
  base64 data-URIs so multipart translates cleanly), `qwen3-vl-flash` (vision chat,
  OpenAI-compat), `text-embedding-v4` (2nd embedding pool). ⚠️ Free-quota status of
  the media models is UNVERIFIED in the console — media daily cap held at 20/day
  until Saiful checks Model Studio per-model quotas + the Free Quota Only switch.

**Paid reserves (deliberately NOT in `auto`):**
- `agentrouter` / `agentrouter-openai` — $125 balance, Opus 5 / Opus 4.8 / gpt-5.6-sol /
  deepseek-v4f. Their gateway rejects requests that do not carry a `claude-cli/...`
  User-Agent, so config sets one (see the residual note below).
  deepseek-v4f now also sits on the Anthropic route with `force_stream = true`
  (their route rejects it without stream) — but as of 2026-08-25 evening (and still
  2026-08-26) agentrouter has NO deepseek-v4f channel on EITHER route (503 "无可用渠道");
  upstream churn, retry later.
- `tabitoken` — $120 referral credits (added 2026-08-25), Opus 5 / 4.8 (+ `-thinking`
  variants). Same User-Agent requirement; speaks Anthropic natively; fronts Kiro/Amazon-Q
  accounts (usage leaks `kiro_credits`). ⚠️ injects ~7k hidden prompt tokens per call,
  billed to us — use for real sessions, not one-liners. Remote usage IS readable
  (earlier "no balance endpoint" note was wrong: /v1/dashboard/billing/usage answers
  with that UA + Bearer — wired into `pxy status --remote`, used $2.40).
- `gorouter` — $70 referral credits (added 2026-08-25). Same operator as tabitoken
  (identical gateway, models, injection, `kiro_credits`) — same caveats. Combined Opus
  reserve across both: ~$190.

⚠️ **Residual, worth revisiting**: those three resale gateways are the ONLY place left
where pxy sends a client identity that is not its own — a `claude-cli/...` User-Agent
in `config.example.toml`, because their gateway rejects other values. It is config, not
code, and they are paid accounts we hold credits with. But it sits against the rule the
2026-08-31 round settled on (identify honestly everywhere), so: try an honest
`pxy/<version>` UA against each and drop the override wherever it still works.
- `fireworks` — pay-per-token, $1 signup credit only.

**Commented out (dead):** `deepseek` ($0 balance, no free tier), `v0-vercel` (API plan-gated,
404s), plus `freemodel-dev` (insufficient balance).

**Service/media-only providers (Phase 2, 2026-08-25):** `elevenlabs` (STT/TTS, xi-api-key
auth), `voyage` (rerank + embeddings), `jina` (rerank; same key as jina-reader), plus
`[[search.providers]]` brave/jina/firecrawl and `[[fetch.providers]]` jina-reader/firecrawl.
Media capabilities on existing providers: cloudflare (images/STT/TTS via run endpoint),
groq + mistral (STT), agnes (images/video).

## Where we left off — NEXT STEPS

0. **API-KEY-ONLY — OAuth removal DONE 2026-08-31 (branch `remove-oauth-providers`).**
   Saiful's call: subscriptions are used natively, *beside* pxy, so pxy keeps API-key
   providers only. `docs/11-api-key-only-roadmap.md` is the queue this opened; it
   supersedes docs/10's sequencing and CANCELS its Tier 1 (no codex/ChatGPT OAuth path
   will be built). docs/10 §0.1/§0.2 remain the compliance record for WHY.
   Removed: `ProviderKind` entirely (claude-oauth / github-copilot / kimi-coding /
   kiro), `WireFormat::Kiro`, `providers/{claude,copilot,kimi,kiro}.rs`,
   `translate/{kiro,eventstream}.rs` + its binary fixture, `RefreshLock`,
   `Secrets::write_pass`, the `credentials_file` config key, and the `x-initiator`
   plumbing (Copilot billing only). **~2,940 lines net deleted**; `src/providers/`
   collapsed to a flat `src/providers.rs` (58 lines, sync — it no longer mints
   anything). Consequences worth knowing:
   - **pxy holds no rotating credential any more.** Secrets are read-only: nothing is
     written back to `pass`, no flock, no refresh mutexes, no token kv rows. `libc`
     stays only for `server.rs` umask.
   - `WireFormat` is a 2-variant enum, so every `!= Kiro` guard and `unreachable!`
     arm in router.rs is gone; the non-streaming eventstream branch and
     `StreamCtx.kiro` with it.
   - `kind = "..."` is **no longer a provider config key** — a leftover line fails
     startup (`unknown field kind`), same precedent as the `tier`/`promo` removals.
     A provider named `claude` is still legal and still special-cased in
     `catalog.resolve` (the `claude/<id>` discovery mirror): it just has to be an
     ordinary API-key provider now, which is what an Anthropic **Console** key is.
   - Cost: Copilot's 300 premium req/month and kiro's 50 free credits/month. Both
     configs had all four blocks commented out already, so nothing live changed.
     `pxy doctor` on the real config: 9 providers, 7 credentials, 134 models.
   - Verified: `cargo test` 171/171, clippy 58 warnings (was 79, none added), real
     streaming + non-streaming requests in BOTH dialects through a test daemon on
     :4199 against the live config. Review also caught a PRE-EXISTING dead test:
     `context_window_400_fails_over_and_skips_smaller_peers` had no `#[tokio::test]`
     attribute and had never run — attribute added, it passes, hence 171 not 170.
     (It is not vacuous: `small` and `tiny` deliberately share one mock route so the
     call counter proves the smaller-window peer was skipped, not merely that the
     walk reached `big`.)
   **docs/11 §2 bugs — first two FIXED 2026-08-31** (same branch):
   - **`pxy_web_search` was offered when nothing could intercept it.** The
     translators injected it on their own dialect's evidence, but interception
     lives entirely in `StreamCtx` and also needs a configured search provider —
     and every `[[search.providers]]` block in the live config is commented out.
     Claude Code with web search on therefore got a `tool_use` for a tool it never
     declared, wedging the turn. `attempt()` now owns the invariant: it builds the
     `SearchLoop` only for an OpenAI upstream + configured provider + genuinely
     streaming turn, and STRIPS the tool from the outbound body otherwise. Also
     closes the `codex --search` variant, which injected pre-routing and could
     send the tool to an Anthropic upstream.
   - **Non-object request bodies panicked the handler task.** axum's `Json<Value>`
     accepts `[]`/`"x"`/`5`, and serde_json's IndexMut panics on them. Guarded in
     `handle_chat`. A live endpoint sweep then found a SECOND defect the audit
     missed: `/v1/responses` answered `5` with a 200 **and a real upstream call**,
     because `responses::request` manufactures a valid object before `handle_chat`
     ever sees the body — so that handler needed its own edge check. All 12
     malformed chat requests now 400 with zero upstream calls.
   - **`<think>` parsing now defaults ON** (OpenAI-format upstreams only, opt out
     with `parse_think_tags = false`). Only 3 live providers set it, so google,
     zai, aihubmix, openrouter, kilocode, tokenharbor and inception were leaking
     literal `<think>` tags into Claude Code as assistant text — which the agent
     then replays as history, burning context every turn. The failure modes are
     asymmetric: a false positive only reclassifies text as reasoning, where
     clients still show it. Redundant `= true` lines dropped from both configs.
   - **docs/11 §3.1 / docs/10's "worst bug": single-candidate walks now return the
     REAL upstream error.** A synthetic `429 overloaded_error` used to replace it,
     so Claude Code's "usage limit reached, resets at …" UI and its
     status-specific retry never saw the body they read. NOT fixed the way
     docs/10 proposed (returning raw straight out of `classify_error`) — that
     would have killed the in-request retry that recovers a transient 429 with a
     near `Retry-After`. Instead `AttemptResult::SkipRaw` carries the upstream
     status+body up the walk, and `handle_chat` returns it verbatim only when a
     SINGLE-candidate walk exhausts its retries. Cooldowns, cooldown scoping and
     retries are untouched; multi-candidate walks keep the aggregate (N failures
     don't reduce to one). Verified live: an explicit model with a dead key
     answers `401 Unauthorized` where it used to answer a synthetic 429.
     `auth_failure_never_retried` updated exactly as docs/10 predicted.
   All four carry regression tests verified to fail without the fix (175 total).
   **Still open in §2**: non-streaming search silently dropped, other server tools
   silently dropped. Then the rest of docs/11 §3 (response headers,
   `count_tokens` forwarding, `/v1/models` negotiation) and §4.1 `cache_control`
   injection on the paid Anthropic reserves.

1. **Add more free providers** — research is DONE: see `docs/08-free-provider-candidates.md`.
   Ready-to-use config blocks are already staged (commented) at the bottom of
   `config.example.toml` / `~/.config/pxy/config.toml`. Each needs a signup + a pass entry,
   then uncomment + restart. Signup order by value:
   1. ~~**Z.AI**~~ — DONE 2026-08-25, active as `zai` (see catalog above).
   2. ~~**Inception Labs**~~ — DONE 2026-08-25, active as `inception` (see catalog above).
   3. **NVIDIA NIM** — ~40 RPM, 100+ models incl. devstral-2-123b.
   4. ~~**Vercel AI Gateway**~~ — REJECTED 2026-08-25: docs/08 was wrong. The $5/month
      grant requires the **Vercel Pro plan ($20/mo)**, not just a card (verified: Hobby
      account with valid key 403s "customer_verification_required" on every call,
      including `-free` models). $20/mo for $5 of credits is a non-starter. Key kept
      in `AI/vercel-gateway/main` in case the account ever goes Pro for other reasons.
   5. **Groq / Scaleway / Ant Ling / Morph / Cerebras** — see docs/08 for the trade-offs.
   Verified DEAD, don't chase: Chutes, Targon, Together, OVHcloud, Hyperbolic, Nebius,
   DeepInfra, Predibase, MonsterAPI, Phind, HuggingFace. **SambaNova is 20 requests/DAY.**
   **Cerebras free tier caps context at 8,192 tokens at 5 RPM** — fatal for agents, so it's
   staged as a short-call-only provider and must never go in `auto`.
   Always test tool calling before putting anything in `auto`.
   Note: Pollinations' keyless tier is gone (401 as of 2026-08-24); it needs a free key now.
2. **Phase 2 — non-chat endpoints: DONE 2026-08-25** (see "Endpoints served" above; research
   basis: OmniRoute's registries + litellm's per-capability adapters, both studied from
   `references/`). Everything below verified live end-to-end:
   - **Search**: brave (2k q/mo free, capped 1900) → jina `s.jina.ai` → firecrawl
     (500 credits/mo, capped 200/scope). **Fetch**: jina-reader → firecrawl scrape.
   - **STT**: groq whisper-large-v3(-turbo) default (⚠️ groq needs a real file EXTENSION),
     mistral voxtral, cloudflare whisper (base64 JSON dialect), elevenlabs scribe_v1.
   - **TTS**: cloudflare aura-2 default, elevenlabs (10k chars/mo; ⚠️ library voices 402
     on free — premade ids mapped from OpenAI voice names in config; usage counted in
     characters). melotts was 500ing upstream — retest.
   - **Images**: cloudflare flux-1-schnell default (b64 JSON; SDXL answers raw bytes —
     handled), agnes-image-2.x (`size` required, config default "1K").
   - **Video**: agnes-video-v2.0 (submit `/v1/videos` → poll `/agnesapi?video_id=`,
     ~75s, upstream limit ~2/min). ⚠️ agnes 2.5 video models require an undocumented
     `mode` value (std/pro/fast/t2v all rejected) — revisit when Agnes documents it.
     Fun fact: agnes's video_id decodes as a litellm-encoded job handle — they run litellm.
   - **Rerank**: voyage rerank-2.5(-lite) default (top_k/data[] dialect mapped to Cohere
     shape), jina reranker (Cohere-compatible passthrough). Voyage embeddings also added
     to `/v1/embeddings` (5th pool).
   - **Dead ends verified**: Google AI Studio image gen is billing-gated (free-tier limit
     is literally 0); aihubmix `gpt-image-2-free` allows 10 lifetime calls and answers in
     chat shape. Not wired: tencent `hy-3d-*`/tripo 3D (niche — revisit on demand),
     groq has no TTS models on this key.
3. ~~**Phase 3 — OAuth providers**~~ — **ALL REVERSED 2026-08-31.** kilocode, kiro
   and kimi-coding were built here; kiro and kimi were deleted in the API-key-only
   round (§0), and kilocode survives because its "OAuth" token is really a
   long-lived JWT used as a plain bearer. `antigravity/agy` was already **DEAD**
   (verified 2026-08-25, no code written): its OAuth client lost free-tier access —
   `loadCodeAssist` returns `UNSUPPORTED_CLIENT` and `streamGenerateContent` 403s
   `SUBSCRIPTION_REQUIRED`. Free Gemini is covered by the `google` (AI Studio)
   provider instead. **Do not reopen this phase** — new providers get API keys.

4. **Catalog automation (`pxy refresh`) — REWRITTEN 2026-08-29 (commit e75a49c).**
   `pxy refresh` = dry-run report; `pxy refresh --generate` = write
   `~/.config/pxy/models.toml`.
   - **models.toml IS NOT CONFIG — nothing loads it (changed 2026-08-30).**
     `Config::load` reads config.toml and nothing else, so a model is served
     exactly when config.toml declares it. models.toml is a REPORT of what each
     provider's /models listed (discovery only — no merge with config.toml, and
     discovery's own numbers, so verify a window before pinning it); you copy
     rows out of it into a provider's `models = [...]` and restart pxy. Before
     this, it was overlaid onto config.toml at startup, which made "why is this
     model served / at this context length?" a two-file question.
   - **Groups replaced the auto route.** The generated `[auto]` chain — context
     buckets, `max_unranked`, open-weights ordering, the `[preferences]`
     tie-breaker, `deny`, `max_pools_per_model` — is ALL GONE. `[groups.*]`
     chains are hand-written in config.toml, and generation writes nothing but a
     per-provider MODEL LIST report (free and paid alike; `free` is a display
     fact, routing never reads it).
   - **Probes are removed.** Capabilities come from the models.dev join
     (7285 models; `tool_call`, `context_length`) plus hand-written overrides.
     Pre-rewrite artifacts — `~/.config/pxy/generated.toml` and the
     `probe:tools:*` kv rows — are dead weight; deleted 2026-08-30.
   - Per-provider: `discover`, `models_url`, `id_field`.
   - **`[providers.X.promo]` is GONE as a config key (2026-08-30).** It dropped
     expired promo ids from the generated report so a now-paid model couldn't be
     pasted into a free chain. The report is read by a human who curates rows by
     hand anyway, so the machinery bought nothing. config.toml is
     `deny_unknown_fields`: a leftover `[providers.X.promo]` table now fails
     startup — delete it.
   - **`tier` is GONE as a config key (2026-08-30).** It was the cost class the
     dead `auto` generator ranked by; afterwards it only echoed into models.toml
     and `pxy models --json` (the desktop panel copied it, rendered it nowhere).
     config.toml is `deny_unknown_fields`, so a leftover `tier = "…"` line now
     fails the load — delete it.
   - **Write guard**: --generate ABORTS if any discovery failure is
     credential-shaped (secret resolution failure, or HTTP 401/402/403 from a
     revoked-but-present key) or >half of providers fail; the previous
     models.toml is kept. A report that silently lost half its providers is
     worse than a stale one — it is what you decide config.toml from. Writes
     are atomic (tmp+rename).
   - **`pxy doctor` warns** when an enabled, allowlisted provider declares no
     models (and isn't embeddings/media-only): with config.toml as the whole
     catalog, that provider is simply invisible — no error anywhere.
   - **Rules that must not be relaxed** (each is somebody's post-mortem):
     absence from a listing is REPORTED, never auto-deleted; capabilities are
     tri-state (Unknown never collapses to No or to an optimistic Yes); a
     failed fetch is distinct from an empty catalog; billing-safety stays
     hand-curated.
   - **Proof the no-auto-delete rule is load-bearing**: `zai/glm-4.7-flash`
     (our best free coding model, 59.2 SWE-bench) does NOT appear in Z.AI's
     own /models listing but works fine — verified live. Auto-deletion would
     have removed it.
5. **Nice-to-haves — ALL RESOLVED 2026-08-25** (one commit each, see git log):
   - ~~Quota headers~~ DONE: on a SUCCESS response, openadapter's `X-Quota-*` (≥100%)
     and `x-ratelimit-remaining-*: 0` (reset parsed from plain-secs / Go-duration /
     epoch dialects) set a provider-wide cooldown. Copilot dropped from the item —
     it does not report premium quota in response headers (checked the reference).
   - ~~Remote balances~~ DONE: `pxy status --remote` + per-provider `balance_url`
     (agentrouter, tokenrouter, tabitoken, gorouter = new-api cents-used shape;
     openrouter = real credits). new-api gateways return dummy hard limits, so only
     spend is visible, not remaining balance.
     **deepseek added 2026-08-30** (`GET /user/balance`, chat key as Bearer):
     amounts are STRINGS, one entry per currency, and `is_available` is a
     separate flag that the line must never omit — an expired grant still counts
     inside `total_balance`, so the money can read healthy on an account that
     refuses every call. A missing flag is reported as unusable, not assumed OK.
     Verified live against the real account: `0.00 USD left · ⚠ NOT usable`.
   - ~~Per-model `force_stream`~~ DONE: streams upstream, re-assembles JSON via
     `translate/aggregate.rs`. Also fixed the generated-overlay bug it exposed:
     the overlay was replacing curated ModelSpecs (max_output_tokens, format,
     pinned contexts) with bare generated entries. (Moot since 2026-08-30 —
     the overlay itself is gone, see §4.)
   - Live model discovery — covered by `pxy refresh` (stage 1-3, done earlier).
   - ~~`/v1/responses`~~ DONE: full Responses API endpoint + `pxy launch codex`
     (wired via `-c` overrides, config.toml untouched). Verified: codex exec ran a
     real shell-tool round-trip through pxy. **Gemini protocol deliberately NOT
     built**: no Gemini client installed here, and free Gemini upstream is already
     served via Google's OpenAI-compat layer — revisit only if gemini-cli lands.
5. **Housekeeping**: `~/.config/pxy/config.toml` and the systemd unit are not chezmoi-managed
   yet — consider adding them to `~/.dotfiles` (config has no secrets, only pass references).

## Decisions already made (don't relitigate)

- **API keys only (2026-08-31).** No OAuth provider kinds, no borrowed
  subscription credentials, no token refresh. Subscriptions (Claude, ChatGPT)
  are used natively, beside pxy. See NEXT STEPS §0 and docs/11; the compliance
  reasoning is docs/10 §0.1-§0.2. A new provider that needs OAuth is out of scope.
- Free-first routing: paid subscriptions and balances are reserves, not defaults.
- No web-cookie/browser providers, ever (that's the Playwright weight pxy exists to shed).
- No MCP/A2A servers, no dashboard, no multi-tenant quota pools.
- Model ids are `provider/model`, split on the FIRST slash (ids may contain slashes).
- Never guess API/library details — verify against docs or the `references/` sources
  (`references/` holds OmniRoute, litellm, and opencode checkouts; gitignored).
- Rejected: Tencent Token Plan, Alibaba Token Plan, b.ai deposit, Fireworks Fire Pass — all
  buy capacity that free pools already cover. Revisit only when `pxy status` shows real
  exhaustion.
