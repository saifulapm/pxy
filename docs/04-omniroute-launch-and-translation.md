# OmniRoute agent launching + Anthropic↔OpenAI translation — research (2026-08-24)

Source: exploration of `references/OmniRoute`. Paths relative to that repo.

## Two separate systems — pxy only needs one

- **`omniroute launch` / `run <target>`** (`bin/cli/commands/launch.mjs`, `run.mjs`,
  `cli-manifest.mjs`): env-var + ephemeral-config injection. This is the `pxy launch` analogue.
- **AgentBridge** = TLS MITM (:443, /etc/hosts spoof, root CA) for agents that hardcode their
  endpoint (Cursor, Copilot, Kiro IDE...). The `agent_bridge_*` tables belong to this. **pxy
  skips this entirely.**

## Launch flow to copy (run.mjs:361-454)

resolve base URL + token → health-check proxy (3s timeout, abort if down) → build clean child
env (**delete inherited conflicting vars first**) → optional ephemeral config dir → spawn with
`stdio: inherit`, forward SIGINT/SIGTERM/SIGHUP, exit codes 130/143/129, ENOENT → 127.
`--dry-run --json` prints the plan with env var *names* only, never token values.

## Per-agent contracts

### Claude Code (env only; buildClaudeEnv, launch.mjs:27)
```
ANTHROPIC_BASE_URL=<root url, NO /v1>        # Claude Code appends /v1/messages itself
ANTHROPIC_AUTH_TOKEN=<token or sentinel>     # must be set or CC stops at login gate
CLAUDE_CODE_ENABLE_GATEWAY_MODEL_DISCOVERY=1
CLAUDE_CODE_AUTO_COMPACT_WINDOW=190000       # CC assumes 200K ctx for unknown model ids
CLAUDE_CONFIG_DIR=<dir>                      # optional; only profile mechanism CC has
ANTHROPIC_MODEL=<model>                      # optional
```
- Delete ALL inherited `ANTHROPIC_*` first (stale shell tokens shadow).
- `ANTHROPIC_SMALL_FAST_MODEL` removed upstream → use `ANTHROPIC_DEFAULT_HAIKU_MODEL`;
  per-tier remaps `ANTHROPIC_DEFAULT_{FABLE,OPUS,SONNET,HAIKU}_MODEL`.
- **Model picker filter**: discovery only shows ids starting `claude`/`anthropic`. OmniRoute
  mirrors models under a `claude/` prefix (`ccDiscoveryAliases.ts`) and strips it server-side
  before routing (`ccDiscoveryAliasStrip.ts`), never re-targeting a genuine claude model.
- `/v1/models` must answer < 3s or the picker is silently empty.

### opencode (genericEnv, run.mjs:246) — best pattern, nothing on disk
```
OMNIROUTE_API_KEY=<token>
OPENCODE_CONFIG_CONTENT={"$schema":...,"provider":{"pxy":{"npm":"@ai-sdk/openai-compatible",
  "options":{"baseURL":"<base>/v1","apiKey":"{env:PXY_API_KEY}"},"models":{...}}}}
```
`{env:...}` indirection keeps the literal key out of the config. Model: `--model pxy/<model>`.

### pi — DISCREPANCY with current pi
OmniRoute merges `{baseUrl (WITH /v1), apiKey, model, _managedBy}` into `~/.pi/config.json` —
that targets an **older pi**. Current pi 0.84.1 (verified separately, see
`02-agent-wiring.md`) uses `~/.pi/agent/models.json` with a `providers` map. pxy should use the
models.json mechanism. The `_managedBy` marker + backup-before-write ideas are worth keeping.

### codex — `-c` TOML flag injection (launch-codex.mjs:173)
```
-c model_provider="pxy" -c model_providers.pxy.base_url="<base>/v1"
-c model_providers.pxy.env_key="PXY_API_KEY" -c model_providers.pxy.wire_api="responses"
```
Codex speaks the OpenAI **Responses** API, not chat completions. Strips all OPENAI_*/CODEX_* env.

### Others (future targets)
- aider: `OPENAI_API_BASE`+`OPENAI_API_KEY`, `--model openai/<m>`
- goose: `GOOSE_PROVIDER=openai`, `OPENAI_HOST`, `GOOSE_MODEL`
- qwen/gemini: throwaway HOME dir (`mkdtemp`, 0600 settings, rm -rf on exit) to neutralize
  stored OAuth without touching real config — good pattern.
- crush: `~/.config/crush/crush.json` `type: "openai-compat"`, `api_key: "$PXY_API_KEY"` env ref.

## Model naming / resolution

- `/v1/models` ids are `<provider-alias>/<model>`; resolver splits on FIRST slash with an
  exact-id escape hatch (`deepseek-ai/DeepSeek-V3` stays whole).
- Accepts: provider/model, alias/model, bare id (alias map → glob → provider inference),
  combo names, `auto`, `no-think/<id>`, effort suffixes `-low/-medium/-high/-xhigh`, `[1m]` suffix.

## Anthropic ↔ OpenAI translation (the hard 20%)

Hub-and-spoke: Claude → OpenAI → target. `/v1/messages` is a thin shim over the same core
handler as `/v1/chat/completions`; format is a tag derived from URL path.

### Request (claude-to-openai.ts)
- `system` block array → single `\n`-joined string (unless upstream honors cache_control).
- `thinking` blocks → message-level `reasoning_content`; `redacted_thinking` dropped.
- `tool_result` → `{role:"tool", tool_call_id, content}`; images inside tool results lifted to a
  following user turn (OpenAI tool messages can't carry images).
- **`regroupToolMessages()`**: reorder tool replies to follow their assistant turn, drop orphans;
  **`fixMissingToolResponses()`**: inject "[No response received]" for unanswered calls.
  Skip these → OpenAI upstreams 400.
- `input_schema`→`parameters` (force empty `properties:{}` for strict mode).
- `thinking.budget_tokens` → `reasoning_effort` buckets: ≤1024 low, ≤10240 medium, <131072 high,
  else xhigh.
- Raise `max_tokens` floor when tools present (arg truncation).

### Streaming response (openai-to-claude.ts:166)
- Per-chunk translator returning Anthropic SSE events (`event: X\ndata: {...}\n\n`).
- **Defer `content_block_start` for tool_use until the tool name arrives** (some providers split
  id/name across chunks; Anthropic can't patch a started block).
- Handle snapshot-style upstreams (full args resent each chunk): emit only the suffix as
  `partial_json`.
- Shared block-index counter; close text/thinking blocks before opening tool blocks.
- Map `stop`→`end_turn`, `length`→`max_tokens`, `tool_calls`→`tool_use` (OmniRoute's
  non-streaming path forgets `length`→`max_tokens` — a bug, don't copy).
- Text-embedded tool-call extraction (`<tool_call>{json}</tool_call>` etc.) exists; optional
  for pxy v1.
- Known OmniRoute bug: `tool_choice:{type:"none"}` falls through to `"auto"` — handle `none`.

### count_tokens
`/v1/messages/count_tokens`: pass through to Anthropic-format upstreams; else local estimate.
**Count every block type** — counting only text blocks broke Claude Code auto-compaction.

## Claude Code specifics

- NOT needed: `/v1/me`, `/api/oauth/*`, statsig/telemetry. Base URL + token is the whole contract.
- **Tool-name remapping**: send tool names lowercased upstream, restore PascalCase in responses
  (`claudeCodeToolRemapper.ts`) or Claude Code errors "No such tool available". (Reason:
  Anthropic fingerprints tool names.) Verify whether this applies to pxy's non-Anthropic
  upstreams too — it's mainly for routing CC traffic to Anthropic-account providers.
- Haiku models 400 on `thinking:{type:"adaptive"}` and `output_config.effort` — strip for
  `/haiku/i` (claudeHaikuConstraints.ts).
- Meta-request bypass (bypassHandler.ts): answer locally by shape — "Warmup", single "count"
  message, `max_tokens===1` + "quota", topic-title extraction. Saves quota + latency; optional.
