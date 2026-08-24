# pxy — handoff (as of 2026-08-24)

Read this first in a new session, then `docs/07-pxy-design.md` for design rationale.

## What pxy is

A tiny Rust proxy replacing OmniRoute (a heavy Node router that used ~800 MB RAM).
**pxy uses ~12.5 MB.** One local endpoint over 19 providers, an `auto` model that routes by
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
pxy launch claude|opencode|pi # spawn an agent wired to pxy (--dry-run shows the plan)
pxy models                    # 95 models exposed
pxy status                    # per-provider usage vs limits
journalctl --user -u pxy -f   # watch routing decisions ("routed" / "failover" lines)
systemctl --user restart pxy  # REQUIRED after any config or pass change (secrets are cached)
```

### Endpoints served
`POST /v1/chat/completions` (OpenAI) · `POST /v1/messages` + `/v1/messages/count_tokens`
(Anthropic) · `POST /v1/embeddings` · `GET /v1/models` · `GET /healthz`.
Both chat protocols translate in both directions, streaming included.

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
| `translate/` | anthropic↔openai (request + streaming response), SSE parser, `<think>` filter |
| `launch.rs` | per-agent env/config injection |

Key invariants worth preserving (learned the hard way, see docs/03 + docs/05):
- **Cooldown scopes**: 401/402/403 → provider-wide; 429/408/409/5xx → `provider/model` only.
  One flaky model must never sideline an account's other models.
- **Token accounting** comes only from real `usage` fields; never guess.
- Error bodies pass through unmodified (Claude Code's auto-retry depends on it).
- 24 unit tests: `cargo test`.

## Provider catalog (19 active)

**Paid subscriptions (already yours):**
- `github` — Copilot Pro, 300 premium req/month (resets 1st, UTC). `github-free/gpt-5-mini` is
  the 0x-multiplier model: unlimited, never consumes premium requests.
- `opencode-go-github` + `opencode-go-google` — $12/5h, $30/wk, $60/mo per account.
  **Per-model allowances vary 400×** — table in docs/07. Only high-allowance models are in
  `auto`; kimi-k3 (110/5h), grok-4.5 (120), qwen3.8-max (160), glm-5.3 (220) are excluded.
  - `muse-spark-1.2-contributor` (45,300/5h): data-collection opt-in was enabled on both
    accounts 2026-08-24, which cleared the `DataPolicyError` — but the model then returned
    **HTTP 500 on every call** (both accounts, all three routes, streaming and not, while
    `hy3` on the same key worked). That's an opencode-side outage, not our config. Retry
    later; keep it out of `auto` until it answers. Note it trains on prompts/completions.

**Free, renewable:**
- `zenmux` — free models incl. `z-ai/glm-5.3-free` (1M ctx). Needs balance > $0 (anti-abuse;
  never deducted). $5 topped up.
- `openrouter` — `:free` models + `stealth/ox-alpha`; 20 rpm, 1000/day (because ≥$10 lifetime
  credits purchased). Free models never touch the credit balance.
- `opencode-zen` — 9 free models, served anonymously (`Bearer public` works!). Shares the Go key.
- `ollama` — free plan: gpt-oss:120b/20b, nemotron-3-*, gemma4:31b only. Flagships need Pro.
- `mistral` (free Experiment tier, phone-verified, opts into data training) + `codestral`
  (separate free key/endpoint).
- `agnes` — flash tier free at $0 balance. Also has image/video models (Phase 2).
- `tokenrouter` — 2 free models of 128. No rate-limit headers; throttles with 503s.
- `bai` — exactly 1 free model (`mimo-v2.5`), heavily throttled; other 41 need a deposit.
- `openadapter` — 50/day, 200/month; only 5 small models on the free plan. Counts *failed*
  requests against quota too.

**Free but finite (use before expiry):**
- `tencent` — TokenHub intl; 5 activated models × 1M tokens, expire **2027-08**.
- `alibaba` — Model Studio; ~5M tokens across Qwen models, expire **~Nov 2026** (90-day).
  ⚠️ Enable "Free Quota Only" in the console or it silently switches to pay-as-you-go.

**Paid reserves (deliberately NOT in `auto`):**
- `agentrouter` / `agentrouter-openai` — $125 balance, Opus 5 / Opus 4.8 / gpt-5.6-sol /
  deepseek-v4f. WAF requires the `claude-cli/...` User-Agent (already set).
- `fireworks` — pay-per-token, $1 signup credit only.

**Commented out (dead):** `deepseek` ($0 balance, no free tier), `v0-vercel` (API plan-gated,
404s), plus `freemodel-dev` (insufficient balance).

## Where we left off — NEXT STEPS

1. **Add more free providers** — research is DONE: see `docs/08-free-provider-candidates.md`.
   Ready-to-use config blocks are already staged (commented) at the bottom of
   `config.example.toml` / `~/.config/pxy/config.toml`. Each needs a signup + a pass entry,
   then uncomment + restart. Signup order by value:
   1. **Z.AI** (`api.z.ai/api/paas/v4`) — permanently free `glm-4.7-flash`, 59.2 SWE-bench,
      agentic-coding tuned, clean ToS. Caveat: ONE concurrent request on free tier.
   2. **Inception Labs** — 100M free tokens on signup, no card.
   3. **NVIDIA NIM** — ~40 RPM, 100+ models incl. devstral-2-123b.
   4. **Vercel AI Gateway** — $5/month RECURRING on frontier models (never top up: buying
      credits permanently ends the free grant).
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
   kilocode (easy) → kimi-coding (device flow + persistent device id) → antigravity/agy
   (Google auth-code + Cloud Code envelope) → kiro/amazon-q (AWS eventstream binary parser).
   Credentials for all of these are already archived in `pass`.
4. **Nice-to-haves identified but not built**:
   - Read upstream quota headers (`X-Quota-5h/Week/Month` on openadapter, Copilot's, etc.) and
     cool a provider down when it self-reports exhaustion.
   - `pxy status` showing remote balances (agentrouter/tokenrouter expose billing endpoints).
   - Per-model `force_stream` flag (agentrouter's deepseek-v4f needs streaming on its
     Anthropic route; worked around by routing it via the OpenAI endpoint).
   - Live model discovery (`pxy models --refresh`) — free model lists churn weekly.
   - Gemini protocol + `/v1/responses` (would let `pxy launch codex` work).
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
