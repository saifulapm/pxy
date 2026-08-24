# pxy — design (v1) — 2026-08-24

Tiny, fast, low-memory Rust replacement for OmniRoute covering exactly what Saiful uses:
one local endpoint over ~25 providers, an `auto` model with limit-aware fallback, and
`pxy launch claude|opencode|pi`. Config in TOML, secrets in `pass`.

## Decisions (2026-08-24, confirmed with Saiful)

| # | Decision |
|---|---|
| D1 | OAuth in v1: **github copilot only**. kilocode, kimi-coding, antigravity, agy, kiro, amazon-q added later one by one (research for each is in `06-omniroute-provider-layer.md`). |
| D2 | **Drop web-session providers**: zai-web, zenmux-free (credentials stay archived in pass). Also drop mimocode (no credentials, gone from OmniRoute). |
| D3 | Non-chat endpoints (embeddings, images gen/edit, audio stt/tts, video, web search, rerank) = **Phase 2**, immediately after chat is solid. They are separate simple pass-through handlers — deferring costs nothing (verified in OmniRoute's architecture). |
| D4 | Source lives **in this folder** (`~/Sites/github/omniroute`), git init here, crate at repo root alongside `docs/` and `references/` (gitignored). Rename repo later. |
| D5 | Secrets: pass entries under `AI/` (see `01-provider-inventory.md`). API keys = first line; OAuth = pure JSON. pxy reads via `pass show`, caches in memory; refreshed OAuth tokens are written back to pass. |

## Shape

- Single binary `pxy`. Subcommands: `serve` (the daemon), `launch <agent>`, `models`, `status`,
  `auth <provider>` (OAuth device/code flows), later `usage`.
- Always-on via a **systemd user unit** (`pxy.service`, WantedBy=default.target). No socket
  activation needed — idle cost is near zero for a Rust process.
- State: `~/.local/share/pxy/state.sqlite` (usage counters, cooldowns worth persisting, OAuth
  token cache). Config: `~/.config/pxy/config.toml`.
- Client-facing endpoints (v1): `POST /v1/chat/completions`, `POST /v1/messages`,
  `POST /v1/messages/count_tokens`, `GET /v1/models` (fast — Claude Code discovery has a 3s
  timeout), `GET /healthz`.

## Config sketch

```toml
[server]
port = 4100
api_key = "pxy-local"            # sentinel; loopback-only bind

[providers.openrouter]
format = "openai"                 # openai | openai-responses | anthropic | (later: gemini)
base_url = "https://openrouter.ai/api/v1/chat/completions"   # COMPLETE endpoint (OmniRoute lesson)
api_key = { pass = "AI/openrouter/main" }
headers = { HTTP-Referer = "https://pxy.local", X-Title = "pxy" }
models = ["deepseek/deepseek-chat-v3.1:free", "qwen/qwen3-coder:free"]

  [providers.openrouter.limits]
  rpm = 20                        # requests/min
  daily_requests = 1000
  daily_tokens = 2000000
  monthly_requests = 0            # 0 = unlimited
  reset = "00:00Z"                # per-provider reset time+zone (OmniRoute hardcoded the
                                  # maintainer's TZ — we make it config)

[providers.github]
format = "openai"                 # copilot also has native /v1/messages for claude models
kind = "oauth-github-copilot"     # selects the executor: 2-stage token mint + header profile
credentials = { pass = "AI/github/main" }
models = ["gpt-5.2", "claude-sonnet-4.6", "gemini-2.5-pro"]

[auto]
# Ordered = priority. First entry with headroom + healthy wins; on failure/limit walk down.
models = [
  "github/claude-sonnet-4.6",
  "openrouter/deepseek/deepseek-chat-v3.1:free",
  "deepseek/deepseek-chat",
]
```

Model ids are `<provider>/<model>`, split on FIRST slash with exact-id escape for ids that
contain slashes themselves (OmniRoute's resolver rule). `auto` is a virtual model.

## Router (synthesis of research 03 + 05)

1. **Filter-then-pick** (litellm): candidates = auto list order; drop those in cooldown, over
   any limit window (pessimistic min-across-windows), circuit-open, or context-window-too-small
   (local pre-check vs model contextLength). Pick the first survivor — config order IS the
   priority. The ordered survivor list is the fallback chain (OmniRoute: selection and fallback
   are one data structure).
2. **Error classification** (litellm's cascade): retryable = 408/409/429/5xx; fatal for the
   request = 400/422/404; 401/403 = skip this provider (cooldown), try next. Context-window
   errors: skip same-model candidates, advance to different-model ones; no cooldown.
3. **Retry**: if another candidate exists → switch immediately, sleep 0. Backoff only when out
   of options: honor Retry-After in (0,60], else 0.5*2^n capped 8s + jitter.
4. **Cooldowns**: per-provider-connection `rate_limited_until`, exponential 3s base (5s OAuth),
   cap 2min; Retry-After resets backoff level. Lazy expiry on read — no timers. Terminal states
   (banned/expired/credits_exhausted) never auto-expire.
5. **Three failure scopes**: provider breaker (only 408/5xx trip it), connection cooldown,
   per-model lockout. Single-provider `auto` lists are exempt from breaker removal (litellm's
   single-deployment exemption).
6. **Limits/usage**: token counts only from real `usage` fields (streaming: terminal SSE
   events); requests always count. Two-bucket sliding window for rpm; fixed daily/monthly
   windows anchored at the provider's configured `reset`. Persisted in sqlite via atomic UPSERT
   so restarts don't forget the day's spend.
7. `x-pxy-provider`/`x-pxy-model` response headers for observability.

## Protocol layer (research 04)

- `/v1/messages` (Anthropic) and `/v1/chat/completions` (OpenAI) both client-facing; hub format
  = OpenAI. Translators: anthropic→openai request + openai→anthropic streaming response first
  (that's Claude Code), then the reverse pair (for anthropic-format upstreams like
  agentrouter/deepseek-alternate/copilot-claude).
- Streaming traps to implement from day one: defer tool_use `content_block_start` until the
  tool name arrives; snapshot-style arg deltas → emit suffix only; close text/thinking blocks
  before tool blocks; `length`→`max_tokens`; tool-message regrouping + orphan repair;
  strip Claude Code beta fields (`thinking:{type:"adaptive"}`, output_config, tool `strict`)
  for non-Anthropic upstreams; pass upstream error bodies through unmodified.
- count_tokens: count EVERY block type (text+tool_use+tool_result), chars/4 estimate is fine.

## Launch (research 02 + 04)

- `pxy launch claude [--model X]`: health-check self, then spawn with clean env:
  delete inherited `ANTHROPIC_*`; set `ANTHROPIC_BASE_URL` (no /v1), `ANTHROPIC_AUTH_TOKEN`,
  `ANTHROPIC_MODEL`, `ANTHROPIC_DEFAULT_HAIKU_MODEL` (cheap model from config),
  `CLAUDE_CODE_AUTO_COMPACT_WINDOW` (from model's context length). Model picker: expose
  `claude-*`-prefixed aliases in /v1/models (strip server-side) OR skip discovery — v1 skips
  discovery, uses env vars.
- `pxy launch opencode`: inject `OPENCODE_CONFIG_CONTENT` JSON (provider `pxy`,
  `@ai-sdk/openai-compatible`, baseURL with /v1, `{env:PXY_API_KEY}`), models map generated
  from config.
- `pxy launch pi`: idempotent merge of provider `pxy` into `~/.pi/agent/models.json`
  (current pi 0.84.1 schema — NOT OmniRoute's old ~/.pi/config.json), then
  `pi --provider pxy --model ...`. apiKey via literal env ref.
- All: `stdio: inherit`, forward signals, exit codes 130/143/129, ENOENT→127,
  `--dry-run` prints plan without secrets.

## Phases

1. **Phase 1 (v1)**: config + pass secrets; default OpenAI executor (covers all API-key
   providers); anthropic↔openai translators; `auto` router + limits; github copilot executor;
   `/v1/models`; `pxy launch claude|opencode|pi`; systemd unit.
2. **Phase 2**: embeddings, images generations/edits, audio transcriptions/speech, video
   generations, web search, rerank as pass-through handlers (voyage-ai, jina, elevenlabs,
   brave-search, firecrawl, agnes, cloudflare-ai...).
3. **Phase 3+**: OAuth providers one by one (kilocode → kimi-coding → antigravity/agy →
   kiro/amazon-q), gemini protocol, openai-responses client endpoint (codex support),
   `no-think`/effort-suffix variants, usage dashboard (`pxy usage`).

## Non-goals

Web-cookie providers, MITM agent bridge, MCP/A2A servers, dashboards/web UI, multi-tenant
quota pools, semantic routing, compression — all the OmniRoute weight pxy exists to shed.

## Auto-chain design (revised 2026-08-24)

Ordering rules, in priority order:
1. **Quality × throughput first**, but never spend a scarce quota when an
   equal-quality cheap pool exists.
2. **Interleave providers** — consecutive entries from one provider all die
   together when that provider rate-limits.
3. **Every entry must do tool calling** (verified by probe; agentic coding
   depends on it).
4. **Renewable before finite**: daily/5-hourly pools refill; free packs expire.
5. **Scarcest paid last**: Copilot's 300 premium requests/month is the most
   precious resource in the stack; `github-free/gpt-5-mini` (0x multiplier,
   unlimited) is the floor that can never run out.

### opencode Go per-model allowances (from the Go dashboard, per account, ×2)

| model | req/5h | note |
|---|---:|---|
| ox-alpha-free | ∞ | limited-time; intermittent 503s |
| muse-spark-1.2-contributor | 45,300 | requires data-collection opt-in — skipped |
| mimo-v2.5 | 30,100 | |
| hy3 | 4,300 | counts 8× → ~34,400 effective |
| longcat-2.0 | 11,400 | |
| deepseek-v4-flash | 7,600 | |
| qwen3.7-plus | 4,300 | |
| minimax-m3 | 3,200 | |
| kimi-k2.7-code | 1,350 | |
| glm-5.2 | 880 | |
| **glm-5.3** | **220** | too scarce for `auto` |
| **qwen3.8-max** | **160** | too scarce for `auto` |
| **grok-4.5** | **120** | too scarce for `auto` |
| **kimi-k3** | **110** | too scarce for `auto` |

The bottom four are configured (manual use) but deliberately excluded from
`auto`: at 110 req/5h a single agent session would drain the window.

### Cooldown scoping (implemented after a real failure)

`ox-alpha-free` 503s intermittently. With provider-wide cooldowns that would
have sidelined `hy3` on the same account. Cooldowns are now two-scoped:
- **401/402/403 → provider-wide** (auth/credits are account problems)
- **429/408/409/5xx → `provider/model` only** (aggregators rate-limit per model)
