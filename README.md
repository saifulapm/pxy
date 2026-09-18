# pxy

One local endpoint in front of every LLM provider I use. Ask for a group name
like `daily`; pxy walks that chain, skips anything cooling down or out of
quota, and fails over until a provider answers. It replaced OmniRoute, a Node
router that ran to hundreds of megabytes. pxy is one binary and one sqlite
file.

```sh
cargo build --release
install -Dm755 target/release/pxy ~/.local/bin/pxy
mkdir -p ~/.config/pxy && cp config.example.toml ~/.config/pxy/config.toml
$EDITOR ~/.config/pxy/config.toml
install -Dm644 contrib/pxy.service ~/.config/systemd/user/pxy.service
systemctl --user daemon-reload && systemctl --user enable --now pxy
```

`config.example.toml` documents every key. The short version:

```toml
[server]
port = 4100

[providers.example]
base_url = "https://api.example.com/v1/chat/completions"  # the whole endpoint, not a host
api_key = { pass = "AI/example/main" }
models = ["large", "small"]

[providers.example.limits]
rpm = 20
daily_requests = 1000
reset = "00:00"
reset_tz = "UTC"

[groups.daily]
models = ["example/large", "other/large"]
```

Secrets stay in `pass`, or in an environment variable or a command. pxy reads
them and never writes back, so restart the daemon after you change one.

## Groups

A group name is itself a model id. `pxy launch claude --model daily` hands the
group to Claude Code, and every request for `daily` walks its chain in order. A
provider in cooldown, over a limit window, or with too small a context is
skipped. Real token counts from response bodies land in sqlite, so limits
survive a restart, and `x-pxy-provider` on the response says who served it.

`pxy route <provider/model>` pins one model to the front of every group walk,
with the chain still behind it as fallback. `pxy route --clear` undoes it.

`config.toml` is the whole catalog: a model is served when a provider lists it,
and not otherwise. `models.toml` is a report that `pxy refresh --generate`
writes and pxy never reads. Copy rows out of it by hand. Nothing is added or
removed automatically, because anything that spends money stays a decision.

## Launch

```sh
pxy launch claude      # ANTHROPIC_* env vars
pxy launch opencode    # OPENCODE_CONFIG_CONTENT
pxy launch codex       # -c model_providers overrides
pxy launch pi          # writes ~/.pi/agent/extensions/pxy.ts
pxy launch fx          # FX_GATEWAY_* env vars
```

Every agent sees the whole catalog, so in-session model switching works.

## Endpoints

Chat in both dialects, streaming included, at `POST /v1/chat/completions` and
`POST /v1/messages` (plus `count_tokens`). `POST /v1/responses` serves codex and
`POST /v3/ai/language-model` serves fx. Embeddings have their own passthrough.
`GET /v1/models` and `/healthz` round it out.

Non-chat work runs through `pxy search`, `fetch`, `transcribe`, `say`, `image`
and `video`, or the matching `/v1/...` endpoints. Media usage is counted on its
own, so it never eats a chat budget.

Web search works on models that never learned it. pxy offers the model a plain
function, runs the call through the search providers you configure, and feeds
the results back into the same streamed turn. Claude Code gets the
`server_tool_use` blocks it expects. The `[server_tools]` table in
`config.toml` decides which tools pxy serves this way and how many tool-call
steps one turn may take. A model can also convene a `fusion` panel of several
models answering the same question at once, with one analyst comparing their
answers; name the panel in `fusion_panel`.

The living spec is this project's mem wiki (`mem wiki` lists the pages). Design
history is in the git log.
