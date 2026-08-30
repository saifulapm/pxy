# pxy

Tiny Rust proxy that puts all your LLM providers behind one local endpoint, with named
**groups** that route by priority + limits and fall over automatically. Built to
replace a heavyweight Node router (OmniRoute) for personal/side-project use.

```
pxy serve                  # the daemon (or: systemctl --user enable --now pxy)
pxy launch claude          # Claude Code wired to pxy (env vars only)
pxy launch opencode        # opencode via OPENCODE_CONFIG_CONTENT
pxy launch pi              # pi via ~/.pi/agent/models.json merge
pxy models                 # group names, then every provider/model (--json for details)
pxy refresh --generate     # report every provider's live catalog into models.toml
pxy status                 # per-provider usage vs limits
```

## Setup

```sh
cargo build --release
cp target/release/pxy ~/.local/bin/
mkdir -p ~/.config/pxy && cp config.example.toml ~/.config/pxy/config.toml  # then edit
cp contrib/pxy.service ~/.config/systemd/user/
systemctl --user daemon-reload && systemctl --user enable --now pxy
```

Secrets never live in the config — they're `pass` references:

```toml
[providers.openrouter]
base_url = "https://openrouter.ai/api/v1/chat/completions"   # COMPLETE endpoint URL
api_key = { pass = "AI/openrouter/main" }                     # or {env=...}, {cmd=...}, literal
models = ["z-ai/glm-5.2:free"]

[providers.openrouter.limits]
rpm = 20
daily_requests = 1000        # daily window anchored at reset/reset_tz
daily_tokens = 2000000
reset = "00:00"
reset_tz = "UTC"
```

## Groups

A group is a named failover chain, and its name is itself a model id:

```toml
[groups.free]
models = ["zai/glm-4.7-flash", "openrouter/minimax/minimax-m3:free"]

[groups.subscription]
models = ["opencode-go-github/hy3", "claude/claude-opus-5"]
```

A request for model `free` walks that list: skip providers in cooldown / over any limit
window / too small a context; first survivor gets the request; retryable failures
(429/5xx/timeouts, auth errors) cool the provider down and move to the next. `Retry-After`
is honored. Usage (requests + real token counts from responses) persists in sqlite across
restarts; `x-pxy-provider` on every response tells you who served it.

`pxy route <provider/model>` pins one model ahead of every group's chain (the chain stays
as fallback) — that's how you switch model mid-session in an agent launched with a fixed
group id. `pxy route --clear` (or `pxy route <group>`) unpins.

**config.toml is the whole catalog.** A model is served exactly when config.toml declares
it, under the provider that declares it — providers, credentials, limits, model lists and
the group chains all live there, and pxy reads no other file.

**models.toml** is a report, not config: `pxy refresh --generate` writes every model each
provider's `/models` currently lists — free and paid, with context window, tool-calling
support and price class — and **pxy never reads it**. Copy the rows you want into a
provider's `models = [...]` in config.toml and restart. Nothing is auto-added and nothing
is auto-removed: what spends money stays a decision, not a heuristic.

`providers_whitelist` (top-level, above `[server]`) is an allowlist: non-empty, and only
those providers exist at all — for the pickers, for the group chains, and for routing.
Entries match by exact name or family prefix (`opencode-go` covers `opencode-go-github`).

## Endpoints (v1)

- `POST /v1/chat/completions` (OpenAI) and `POST /v1/messages` + `count_tokens`
  (Anthropic) — both translate to whatever each upstream speaks, streaming included.
- `GET /v1/models`, `GET /healthz`.
- Providers: any OpenAI- or Anthropic-compatible endpoint via config, plus GitHub
  Copilot (built-in two-stage token mint). More OAuth providers land one by one.

Hosted **web search** works on models that have never heard of it — Anthropic's
`web_search` server tool (Claude Code's WebSearch) and the Responses API's
(`codex --search`). An OpenAI upstream is offered a plain function instead; pxy
intercepts the call, runs it through `[[search.providers]]`, feeds the results back to
the model, and continues the same streamed turn. Claude clients get the
`server_tool_use` + `web_search_tool_result` blocks they expect.

Design/research notes live in `docs/` (start with `docs/07-pxy-design.md`).
