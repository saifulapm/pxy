# How `pxy launch <agent>` must wire each agent (researched 2026-08-24)

Verified against local installs: Claude Code 2.1.241, opencode 1.18.21, pi 0.84.1.
Canonical pattern (LiteLLM, claude-code-router do exactly this): launcher-owned local
Anthropic-format endpoint + **per-process env vars / config injection** — never mutate the
user's global config.

## Claude Code (env vars only — perfectly session-scoped)

- `ANTHROPIC_BASE_URL=http://localhost:PORT` — expects **Anthropic Messages format**.
- `ANTHROPIC_AUTH_TOKEN=<pxy key>` — sent as `Authorization: Bearer`; overrides claude.ai login
  with **no prompt** (unlike `ANTHROPIC_API_KEY`, which prompts once in interactive mode). Use this.
- Model selection:
  - `ANTHROPIC_MODEL` — session model (overridden only by `--model`).
  - `ANTHROPIC_DEFAULT_MODEL` — default for new sessions (v2.1.236+, only when settings don't set `model`).
  - `ANTHROPIC_DEFAULT_HAIKU_MODEL` — `haiku` alias + background tasks (`ANTHROPIC_SMALL_FAST_MODEL` is deprecated).
  - `ANTHROPIC_DEFAULT_SONNET_MODEL` / `ANTHROPIC_DEFAULT_OPUS_MODEL` / `ANTHROPIC_DEFAULT_FABLE_MODEL` — alias remaps.
  - `CLAUDE_CODE_SUBAGENT_MODEL` — subagent model.
- `API_TIMEOUT_MS` (default 600000; values > 2147483647 overflow and break).
- `CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC=1` recommended with gateways — but it also disables
  gateway model discovery.

### Endpoints Claude Code calls on the base URL
| Endpoint | Required? | Notes |
|---|---|---|
| `POST /v1/messages?beta=true` | yes | match path only, query varies; MUST stream SSE unbuffered; stream silent >300s aborted (forward pings) |
| `POST /v1/messages/count_tokens` | optional | fallback exists if absent |
| `HEAD /api/hello` | no | warm-up probe, safe to reject |
| `GET /v1/models?limit=1000` | opt-in | only with `CLAUDE_CODE_ENABLE_GATEWAY_MODEL_DISCOVERY=1`; 3s timeout; no redirects; cached at `~/.claude/cache/gateway-models.json` |

**Model-picker filter gotcha:** only model IDs containing `claude` or `anthropic` (substring,
case-insensitive, v2.1.223+) appear in the `/model` picker ("From gateway"). Non-matching IDs are
invisible — alias them (e.g. `pxy/claude-auto`) or set via `ANTHROPIC_MODEL`/`ANTHROPIC_DEFAULT_*`.

Fast-mode check + WebFetch domain-safety go directly to api.anthropic.com (ignore base URL) — fine.

### Gotchas serving Claude Code from OpenAI-backed upstreams
- Beta capabilities ship as header+body pairs (`context_management`, `output_config`, tool
  `strict`/`defer_loading`) → upstream 400 "Extra inputs are not permitted" unless the proxy strips
  them. Client-side kill switch: `CLAUDE_CODE_DISABLE_EXPERIMENTAL_BETAS=1`.
- Adaptive thinking: sends `thinking:{"type":"adaptive"}` for Claude ≥4.6 **and any unrecognized
  model name** (i.e. our aliases). Strip proxy-side or set `CLAUDE_CODE_DISABLE_ADAPTIVE_THINKING=1`.
  Claude Code auto-retries thinking/signature rejections only if the upstream error body passes
  through unmodified (or wrapped with a `capability_rejected:` token).
- Interleaved thinking / extended context are header-only betas — forward `anthropic-beta` verbatim.
- Fine-grained tool streaming is off by default via custom base URL (`CLAUDE_CODE_ENABLE_FINE_GRAINED_TOOL_STREAMING=1` to opt in).
- Attribution system-prompt block is cache-stable since v2.1.181; `CLAUDE_CODE_ATTRIBUTION_HEADER=0` omits it.

## opencode (config injection via env — merges, never clobbers)

- `OPENCODE_CONFIG_CONTENT='{"provider":{...},"model":"pxy/model-id"}'` — inline JSON merged near
  the top of the config stack (above project config). **Ideal for the launcher; no temp file.**
- Alternative: `OPENCODE_CONFIG=/path/to/file` (merged lower in the stack).
- Provider shape (`npm: "@ai-sdk/openai-compatible"` for OpenAI format, `@ai-sdk/anthropic` for Anthropic):

```json
{
  "provider": {
    "pxy": {
      "npm": "@ai-sdk/openai-compatible",
      "name": "pxy",
      "options": { "baseURL": "http://localhost:PORT/v1", "apiKey": "{env:PXY_KEY}" },
      "models": { "model-id": { "name": "Display", "limit": { "context": 128000, "output": 65536 } } }
    }
  }
}
```

- Model select: `opencode -m pxy/model-id` or `"model"` key in the injected JSON.
- Custom providers are **static-config only** — no dynamic /v1/models fetch. pxy must generate the
  models map at launch time from its own catalog.

## pi (v0.84.1, package renamed to @earendil-works/pi-coding-agent; repo badlogic/pi-mono)

- `~/.pi/agent/models.json` — **additive** over the built-in catalog, hot-reloaded whenever /model
  opens. Schema (packages/coding-agent/docs/models.md):

```json
{
  "providers": {
    "pxy": {
      "baseUrl": "http://localhost:PORT/v1",
      "api": "openai-completions",
      "apiKey": "$PXY_KEY",
      "models": [
        { "id": "model-id", "name": "Display", "reasoning": false,
          "contextWindow": 128000, "maxTokens": 16384,
          "cost": {"input":0,"output":0,"cacheRead":0,"cacheWrite":0} }
      ]
    }
  }
}
```

- `api` can be `openai-completions`, `openai-responses`, `anthropic-messages`,
  `google-generative-ai`; per-model override allowed. `apiKey` accepts `$ENV`, `!cmd`, or literal
  (`!pass show ...` works!). `compat` flags for quirky upstreams.
- Launch: `pi --provider pxy --model <pattern>`. (`PI_MODEL`/`PI_PROVIDER` env are *outputs* for
  bash subprocesses, not selectors.)
- `PI_CODING_AGENT_DIR` relocates the whole agent dir (settings+auth+sessions) — too blunt; prefer
  merging our provider block into the user's `models.json` (additive by design).

## Consequences for pxy design

1. pxy must serve **both** protocols natively: Anthropic Messages (`/v1/messages`, streaming,
   count_tokens) for Claude Code, and OpenAI chat completions for opencode/pi.
2. Model IDs exposed to Claude Code should contain "claude" (picker filter) — or we skip discovery
   and drive selection purely via env vars.
3. `pxy launch` per agent:
   - claude → spawn with env vars only.
   - opencode → spawn with `OPENCODE_CONFIG_CONTENT` (generated from pxy catalog).
   - pi → merge `pxy` provider into `~/.pi/agent/models.json` (idempotent), then
     `pi --provider pxy --model ...`.
4. pxy needs to strip/translate Claude Code beta fields when the upstream isn't Anthropic, and
   pass through error bodies unmodified so client-side auto-retry works.
