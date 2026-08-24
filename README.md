# pxy

Tiny Rust proxy that puts all your LLM providers behind one local endpoint, with an
`auto` model that routes by priority + limits and falls over automatically. Built to
replace a heavyweight Node router (OmniRoute) for personal/side-project use.

```
pxy serve                  # the daemon (or: systemctl --user enable --now pxy)
pxy launch claude          # Claude Code wired to pxy (env vars only)
pxy launch opencode        # opencode via OPENCODE_CONFIG_CONTENT
pxy launch pi              # pi via ~/.pi/agent/models.json merge
pxy models                 # everything exposed, incl. "auto"
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

## The auto model

`[auto].models` is an ordered `provider/model` list. A request for model `auto` walks it:
skip providers in cooldown / over any limit window / too small a context; first survivor
gets the request; retryable failures (429/5xx/timeouts, auth errors) cool the provider
down and move to the next. `Retry-After` is honored. Usage (requests + real token counts
from responses) persists in sqlite across restarts; `x-pxy-provider` on every response
tells you who served it.

## Endpoints (v1)

- `POST /v1/chat/completions` (OpenAI) and `POST /v1/messages` + `count_tokens`
  (Anthropic) — both translate to whatever each upstream speaks, streaming included.
- `GET /v1/models`, `GET /healthz`.
- Providers: any OpenAI- or Anthropic-compatible endpoint via config, plus GitHub
  Copilot (built-in two-stage token mint). More OAuth providers land one by one.

Design/research notes live in `docs/` (start with `docs/07-pxy-design.md`).
