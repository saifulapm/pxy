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
- **Secrets**: `pass` under `AI/<provider>/<name>`. API keys = first line; OAuth = pure JSON.
- **State**: `~/.local/share/pxy/state.sqlite` (usage counters survive restarts).
- **Verified end-to-end**: Claude Code ran a real session through pxy on the `auto` chain.

### Commands
```sh
pxy serve                     # daemon (systemd runs this)
pxy launch claude|opencode|pi|codex # spawn an agent wired to pxy (--dry-run shows the plan)
pxy models                    # 146 models exposed
pxy status [--remote]         # per-provider usage vs limits; --remote adds live balances
pxy refresh [--write]         # discover catalogs; report drift / regenerate generated.toml
journalctl --user -u pxy -f   # watch routing decisions ("routed" / "failover" lines)
systemctl --user restart pxy  # REQUIRED after any config or pass change (secrets are cached)
```

### Endpoints served
`POST /v1/chat/completions` (OpenAI) · `POST /v1/messages` + `/v1/messages/count_tokens`
(Anthropic) · `POST /v1/responses` (OpenAI Responses API — codex) · `POST /v1/embeddings` ·
`GET /v1/models` · `GET /healthz`.
Both chat protocols translate in both directions, streaming included. /v1/responses
wraps the OpenAI path (translate/responses.rs), so routing/accounting apply unchanged.

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
| `providers/copilot.rs` | GitHub Copilot two-stage token mint + header profile |
| `providers/kimi.rs` | Kimi coding: rotating-refresh OAuth, X-Msh-* identity |
| `providers/kiro.rs` | Kiro/CodeWhisperer: social refresh, ARN->region, profileArn body patch |
| `translate/eventstream.rs` | AWS vnd.amazon.eventstream binary frame decoder |
| `translate/kiro.rs` | anthropic<->conversationState; frames -> OpenAI SSE; sha1/uuidv5 |
| `translate/` | anthropic↔openai (request + streaming response), SSE parser, `<think>` filter |
| `translate/responses.rs` | OpenAI Responses API (codex) ↔ chat completions, streaming incl. |
| `translate/aggregate.rs` | SSE stream → complete JSON body (the `force_stream` re-assembly) |
| `refresh.rs` | catalog discovery: provider /models x models.dev join, drift report |
| `launch.rs` | per-agent env/config injection |

Key invariants worth preserving (learned the hard way, see docs/03 + docs/05):
- **Cooldown scopes**: 401/402/403 → provider-wide; 429/408/409/5xx → `provider/model` only.
  One flaky model must never sideline an account's other models.
- **404s fail over on multi-candidate walks only** (added 2026-08-25 after zenmux delisted
  glm-5.3-free and its 404 killed the whole `auto` chain): in `auto`, an upstream 404 skips
  the candidate with a model-scoped cooldown + failover log; an explicit single-model
  request still passes the raw 404 through (Claude Code needs the unmodified body).
- **Token accounting** comes only from real `usage` fields; never guess.
- Error bodies pass through unmodified (Claude Code's auto-retry depends on it).
- 47 unit tests: `cargo test` (incl. real-capture kiro eventstream fixtures).

## Provider catalog (30 active)

**Paid subscriptions (already yours):**
- `github` — Copilot Pro, 300 premium req/month (resets 1st, UTC). `github-free/gpt-5-mini` is
  the 0x-multiplier model: unlimited, never consumes premium requests.
  Live quota in `pxy status --remote` via `GET api.github.com/copilot_internal/user`
  (needs the LONG-LIVED GitHub token, `token <gh>` — not the minted Copilot bearer).
  ⚠️ pxy's local 300-counter only sees pxy traffic — GitHub's number is authoritative
  (2026-08-25: GitHub said 89 used vs pxy's 14, the rest was VS Code / pre-pxy).
  ⚠️ `overage_permitted: true` on the account — premium requests past 300 BILL
  ($0.04 each); the status line flags it. Disable in github.com → Settings → Copilot
  if surprise billing is unwanted.
- `opencode-go-github` + `opencode-go-google` — $10/mo Go plan per account; usage-dollar
  limits $12/5h (rolling), $30/wk, $60/mo, enforced upstream (429 → pxy cooldown +
  account rotation; pxy does no dollar accounting itself). Live utilization:
  `GET /zen/go/v1/usage` with the API key (found in the opencode source, wired into
  `pxy status --remote` 2026-08-25 — shows percent per window + reset time).
  **Per-model allowances vary 400×** — table in docs/07. Only high-allowance models are in
  `auto`; kimi-k3 (110/5h), grok-4.5 (120), qwen3.8-max (160), glm-5.3 (220) are excluded.
  - `muse-spark-1.2-contributor` (45,300/5h): data-collection opt-in was enabled on both
    accounts 2026-08-24, which cleared the `DataPolicyError` — but the model then returned
    **HTTP 500 on every call** (both accounts, all three routes, streaming and not, while
    `hy3` on the same key worked). That's an opencode-side outage, not our config. Retry
    later; keep it out of `auto` until it answers. Note it trains on prompts/completions.
    (Retested 2026-08-25 evening: still 500 on both zen and Go routes.)

**Free, renewable:**
- `kiro` — AWS CodeWhisperer via Kiro (added 2026-08-25, **Phase 3 #3**, the big one:
  `providers/kiro.rs` + `translate/kiro.rs` + `translate/eventstream.rs`). **KIRO FREE
  plan: 50 credits/month, resets the 1st**; overage DISABLED + OVERAGE_INCAPABLE, so it
  cannot bill. Credits scale by `rateMultiplier`, so cheap models go far: qwen3-coder-next
  0.05x, minimax-m2.1 0.15x, deepseek-3.2/minimax-m2.5 0.25x, haiku-4.5 0.4x, glm-5 0.5x,
  sonnet-4.5 1.3x (a small haiku call metered 0.005 credits). Tool calling verified
  streaming + non-streaming. Model ids MUST match ListAvailableModels or Kiro 400s —
  the account has NO claude-sonnet-5/gpt-5.6 despite OmniRoute's registry listing them.
  Not in `auto` (scarcest free pool). Usage is ESTIMATED (kiro sends no token counts,
  only contextUsagePercentage + a credit figure).
  - **`amazon-q` deliberately NOT added** (checked 2026-08-25): its archived credential is
    byte-identical to kiro's — same refresh token, same `profileArn`, so the same AWS
    account and the SAME 50-credit pool. A second provider entry would add zero quota and
    would actively break accounting by splitting one pool across two counters. Only worth
    adding if a genuinely separate AWS account is connected.
  - Kiro's SOCIAL refresh tokens do **not** rotate (verified: the endpoint returns the
    presented token unchanged, and the pre-existing one still worked after a refresh).
    The one-time-use warning in docs/06 applies to the AWS SSO-OIDC builder-id path, not
    this one — so this credential is durable and shouldn't need re-login.
- `kimi-coding` — Moonshot Kimi coding tier (added 2026-08-25, **Phase 3 #2**, real Rust:
  `providers/kimi.rs`). Rotating refresh tokens (serialized, persisted to kv BEFORE use —
  losing one kills the session), 900s access tokens, X-Msh-* device identity, Anthropic
  native. ⚠️ credits exhausted at activation (masked as bare 500s); NOT in `auto` — retest
  after quota reset (retested 2026-08-25 evening: still 500s, no reset yet).
  Re-login = curl device flow (see config.example comment).
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
  Free models never touch the credit balance.
- `aihubmix` — aggregator, free catalog opened ~2026-08 (added 2026-08-25): 50 `-free`
  models on one key, no card. In `auto`: `gemini-3.7-flash-free` (only Gemini in the stack,
  1M ctx), `minimax-m3-free`, `ox-alpha` (plain id, free while in stealth). Kim-series free
  pool usually exhausted ("insufficient promotional resources" — retry); `gpt-5.5-free`
  doesn't route. No published rate limits. Key: `AI/aihubmix/main`.
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
  to pre-2026-02 accounts. TODO (needs Saiful in a browser, still pending): flip the
  training opt-out in Account Settings on platform.inceptionlabs.ai.
- ~~`scaleway`~~ — DISABLED same day it was added (2026-08-25): docs/08 was WRONG about
  the "1M free tokens, no card" claim. Signup REQUIRED a card, and the billing API shows
  zero discounts — no free tier exists; every call bills. Key verified working and kept
  in `AI/scaleway/main` (secret key = Bearer); block commented out in config. Re-enable
  only deliberately as paid. ⚠️ docs/08 has now been wrong twice (Vercel, Scaleway) —
  re-verify its remaining claims (Ant Ling, Morph, NVIDIA) against reality before use.
- `tencent` — TokenHub intl; 5 activated models × 1M tokens, expire **2027-08**.
- `alibaba` — Model Studio; ~5M tokens across Qwen models, expire **~Nov 2026** (90-day).
  ⚠️ Enable "Free Quota Only" in the console or it silently switches to pay-as-you-go.

**Paid reserves (deliberately NOT in `auto`):**
- `agentrouter` / `agentrouter-openai` — $125 balance, Opus 5 / Opus 4.8 / gpt-5.6-sol /
  deepseek-v4f. WAF requires the `claude-cli/...` User-Agent (already set).
  deepseek-v4f now also sits on the Anthropic route with `force_stream = true`
  (their route rejects it without stream) — but as of 2026-08-25 evening agentrouter
  has NO deepseek-v4f channel on EITHER route (503 "无可用渠道"); upstream churn,
  retry later.
- `tabitoken` — $120 referral credits (added 2026-08-25), Opus 5 / 4.8 (+ `-thinking`
  variants). Same claude-cli UA WAF; speaks Anthropic natively; fronts Kiro/Amazon-Q
  accounts (usage leaks `kiro_credits`). ⚠️ injects ~7k hidden prompt tokens per call,
  billed to us — use for real sessions, not one-liners. Remote usage IS readable
  (earlier "no balance endpoint" note was wrong: /v1/dashboard/billing/usage answers
  with the claude-cli UA + Bearer — wired into `pxy status --remote`, used $2.40).
- `gorouter` — $70 referral credits (added 2026-08-25). Same operator as tabitoken
  (identical WAF, models, injection, `kiro_credits`) — same caveats. Combined Opus
  reserve across both: ~$190.
- `fireworks` — pay-per-token, $1 signup credit only.

**Commented out (dead):** `deepseek` ($0 balance, no free tier), `v0-vercel` (API plan-gated,
404s), plus `freemodel-dev` (insufficient balance).

## Where we left off — NEXT STEPS

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
2. **Phase 2 — non-chat endpoints** (design already validated; these are simple pass-through
   handlers, architecturally separate): images generations/edits, audio transcription + TTS,
   video generation, web search, rerank. Credentials already in pass for: `voyage-ai`,
   `jina-reader`, `elevenlabs`, `brave-search`, `firecrawl`, `cloudflare-ai`, plus agnes
   image/video and Tencent's `hy-3d-*` models. `/v1/embeddings` already exists (4 pools:
   tencent kinfra, alibaba qwen, fireworks qwen3-8b, voyage pending).
3. **Phase 3 — OAuth providers**, one at a time, easiest first (research in docs/06):
   ~~kilocode~~ (DONE 2026-08-25 — token turned out long-lived, plain config, no Rust) →
   ~~kiro/amazon-q~~ (kiro DONE 2026-08-25 — see catalog; amazon-q is the same
   protocol under a different connection, add when wanted) →
   ~~kimi-coding~~ (DONE 2026-08-25 — `kind = "kimi-coding"` in providers/kimi.rs: serialized
   rotation-safe refresh persisted to kv, X-Msh-* identity profile, Anthropic endpoint.
   The archived refresh token was dead; re-logged-in via curl device flow. ⚠️ Account
   credits currently EXHAUSTED — Kimi masks this as bare 500s on chat/models; auth verified,
   retest for quota reset before adding k3 to [auto]) → ~~antigravity/agy~~
   (**DEAD — verified 2026-08-25, do not build**). **Phase 3 is COMPLETE.**

   **antigravity/agy — dead, no code written.** Both archived credentials
   (`AI/antigravity/saifulapm@gmail.com`, `AI/agy/…`, same project `avid-rex-p83hx`, both
   labelled "Antigravity Starter Quota" / free-tier) still REFRESH fine — Google refresh
   tokens don't rotate, so auth was never the problem. The account has simply lost access:
   - `POST cloudcode-pa.googleapis.com/v1internal:loadCodeAssist` now lists `free-tier`
     under **ineligibleTiers** with `UNSUPPORTED_CLIENT`: *"This client is no longer
     supported for Gemini Code Assist for individuals. To continue using Gemini, please
     migrate to the Antigravity suite of products."* Only paid `standard-tier` is allowed.
   - `POST …:streamGenerateContent?alt=sse` → **403 SUBSCRIPTION_REQUIRED** ("You do not
     have a valid license of this product") on BOTH the `ide` and `cli` profiles.
   Google retired this OAuth client's free path, so the Cloud Code envelope translator
   (the last hard piece of work in Phase 3) would buy nothing. Re-verify with those two
   calls BEFORE writing any code if Antigravity ships a new client/API. Free Gemini is
   already covered by the `google` (AI Studio) provider in `auto`.
4. **Catalog automation (`pxy refresh`) — ALL THREE STAGES DONE 2026-08-25.**
   `pxy refresh` = dry-run report; `pxy refresh --write` = regenerate
   `~/.config/pxy/generated.toml` (model lists + auto chain), which Config::load
   overlays onto config.toml at startup. Restart pxy after a --write.
   - `[preferences]` in config.toml: bare model names best-first, `max_pools_per_model`,
     `max_unranked` tail, `deny` list for listed-but-broken ids. **Tier-first ordering**
     (free -> subscription -> finite; reserve NEVER generated), preference order within
     a tier — a ranking can never start spending money.
   - Per-provider: `tier`, `discover`, `models_url`, `id_field`, `[providers.X.promo]`
     (`expires = "YYYY-MM-DD"`, fails closed on bad dates).
   - Per-model: `tool_call = true|false` — a CURATED fact that beats discovery and
     probing (zai's 1-concurrent throttle makes probing it unreliable; its glm-4.7-flash
     is also absent from Z.AI's own /models yet works).
   - Probes: only for ranked models with unknown capability; YES cached forever, NO for
     7 days (aihubmix's gemini tool-called in the morning and stopped by afternoon —
     free pools degrade and recover), truncated answers (finish=length) cache nothing.
   - **Write guard**: --write ABORTS if any discovery failure is credential-shaped or
     >half of providers fail (a locked gpg agent takes out every provider at once;
     generating from that would shrink the chain and then be loaded as truth).
   - **Feedback-loop guard**: refresh reads `Config::load_base` (baseline WITHOUT the
     generated overlay) — generation consuming its own output erased curated marks.
   - Stage-1 history (research + rules):
   Design research: OmniRoute + litellm + our own config (see the commit message).
   - **Stage 1 (done)**: `pxy refresh` discovers every provider's live `/models`
     (default-ON, seed fallback — an opt-in allowlist is how OmniRoute silently served
     stale catalogs), joins **models.dev** (7285 models, 100% carry `tool_call`; covers
     94% of our config and 27/27 of `auto`), and prints drift + free-and-tool-capable
     candidates + cross-provider pools. Read-only.
   - **Stage 2 (todo)**: probe cache in sqlite for the ~6% models.dev can't answer;
     generate per-provider `models` lists into `generated.toml` (merged at load, so
     hand-written auth/limits/quirks are never touched).
   - **Stage 3 (todo)**: `[preferences]` list of bare model names + per-provider `tier`;
     generate the `auto` chain. **Open decision**: tier-first (free pools before paid,
     preference orders within a tier — recommended) vs literal preference-first.
     Also: `[providers.X.promo] expires = "…"` to auto-drop promos (openrouter's own
     `expiration_date` is present on only 8/419 models and absent on the GMI promo, so
     upstream expiry data can NOT be relied on).
   - **Rules that must not be relaxed** (each is somebody's post-mortem):
     absence from a listing is REPORTED, never auto-deleted; capabilities are tri-state
     (Unknown never collapses to No or to an optimistic Yes); a failed fetch is distinct
     from an empty catalog; billing-safety stays hand-curated.
   - **Proof the no-auto-delete rule is load-bearing**: `zai/glm-4.7-flash` (our best free
     coding model, 59.2 SWE-bench) does NOT appear in Z.AI's own `/models` listing but
     works fine — verified live. Auto-deletion would have removed it.
5. **Nice-to-haves — ALL RESOLVED 2026-08-25** (one commit each, see git log):
   - ~~Quota headers~~ DONE: on a SUCCESS response, openadapter's `X-Quota-*` (≥100%)
     and `x-ratelimit-remaining-*: 0` (reset parsed from plain-secs / Go-duration /
     epoch dialects) set a provider-wide cooldown. Copilot dropped from the item —
     it does not report premium quota in response headers (checked the reference).
   - ~~Remote balances~~ DONE: `pxy status --remote` + per-provider `balance_url`
     (agentrouter, tokenrouter, tabitoken, gorouter = new-api cents-used shape;
     openrouter = real credits). new-api gateways return dummy hard limits, so only
     spend is visible, not remaining balance.
   - ~~Per-model `force_stream`~~ DONE: streams upstream, re-assembles JSON via
     `translate/aggregate.rs`. Also fixed the generated-overlay bug it exposed:
     the overlay was replacing curated ModelSpecs (max_output_tokens, format,
     pinned contexts) with bare generated entries — curated spec now wins by id.
   - Live model discovery — covered by `pxy refresh` (stage 1-3, done earlier).
   - ~~`/v1/responses`~~ DONE: full Responses API endpoint + `pxy launch codex`
     (wired via `-c` overrides, config.toml untouched). Verified: codex exec ran a
     real shell-tool round-trip through pxy. **Gemini protocol deliberately NOT
     built**: no Gemini client installed here, and free Gemini upstream is already
     served via Google's OpenAI-compat layer — revisit only if gemini-cli lands.
5. **Housekeeping**: `~/.config/pxy/config.toml` and the systemd unit are not chezmoi-managed
   yet — consider adding them to `~/.dotfiles` (config has no secrets, only pass references).

## Decisions already made (don't relitigate)

- Free-first routing: paid subscriptions and balances are reserves, not defaults.
- No web-cookie/browser providers, ever (that's the Playwright weight pxy exists to shed).
- No MCP/A2A servers, no dashboard, no multi-tenant quota pools.
- Model ids are `provider/model`, split on the FIRST slash (ids may contain slashes).
- Never guess API/library details — verify against docs or the `references/` sources
  (`references/` holds OmniRoute, litellm, and opencode checkouts; gitignored).
- Rejected: Tencent Token Plan, Alibaba Token Plan, b.ai deposit, Fireworks Fire Pass — all
  buy capacity that free pools already cover. Revisit only when `pxy status` shows real
  exhaustion.
