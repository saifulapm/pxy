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

`pxy route <group> <provider/model>` pins one model to the front of that
group's walk, with the chain still behind it as fallback; other groups are
untouched. `pxy route <group> --clear` undoes it, `pxy route` lists the pins.

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

Server tools work on models that never learned them. A client declares one
by its type (`pxy:web_search`, or OpenRouter's `openrouter:web_search`), pxy
offers the model a plain function, runs the call itself, and feeds the result
back into the same streamed turn. Claude Code gets the `server_tool_use`
blocks it expects. Twelve tools are served: web_search, web_fetch, datetime,
search_models, image_generation, advisor, subagent, fusion, tool_search,
describe_image, memory and find_docs. Each takes the parameters OpenRouter
documents for it, bar the last three, which are pxy's own and take pxy's. A client with more tools
than one turn can afford marks the spare ones `defer_loading` and declares
tool_search: pxy keeps them out of the upstream body, and the model gets back
the ones it asks for by searching. A model that cannot take an image at all is
marked `vision = false`: on an OpenAI-format upstream pxy puts `[image 1]` in
the transcript where the image was, and the model hands that number to
describe_image, which asks a model that can see. memory gives the model a
directory it keeps between conversations, under
`~/.local/share/pxy/memory/<store>/`, and the same six commands Anthropic's
memory tool uses; each agent gets its own store unless the config points two
at one. find_docs is how a model stops guessing at an API it half remembers:
it names a library and a topic, and pxy fetches that topic's snippets from
Context7, whose free tier needs no account at all. Both halves of the lookup
are cached for a week, so the same question twice costs one call. The
`[server_tools]` table in
`config.toml` decides which are served, how many tool-call steps one turn
may take, and, through `[server_tools.defaults.<tool>]`, which ride every
client turn without the harness declaring them.

Separate from the tools are plugins: request-time transforms that cost no
model call. A client asks for one with `plugins: [{"id": "..."}]`, or
`[plugins]` in `config.toml` turns it on for every request; pxy consumes the
key, so it never reaches an upstream, and an id pxy has not implemented is
ignored rather than refused. Two exist. `response-healing` comes first. Ask a
model for JSON and it will hand you a Markdown fence, a preamble or a trailing
comma often enough to matter, and `JSON.parse` fails on all three. Healing
rebuilds each `{` or `[` up to its matching close, quoting bare keys, dropping
trailing commas and closing what was left open, then picks between them. That
choice is the hard part: prose is full of runs that parse on their own, and a
`[1]` citation marker served in place of the answer would be silent, since
nothing in the response says a heal happened. A model puts its answer at the
start of a line, so a candidate that opens one outranks a candidate inside a
sentence; within a rank the one accounting for the most of its own text wins.
The answer is replaced only when the repair parses, so one truncated by
`max_tokens` reaches the client as the model left it. Streaming turns and
Anthropic clients are never healed.

The second is `file-parser`, and it is the one plugin on by default. Attach a
PDF — a chat `file` part, an Anthropic `document` block, a Responses
`input_file` item — and pxy parses it before it picks a model, so the document
reaches every model as text, not just the few with a file API. Pages poppler
finds blank are scanned rather than typeset; set `ocr_model` and they are
rasterised and read back as Markdown, otherwise such a page says so. Each
parse is cached by the file's SHA-256, so the history a client resends every
turn costs one parse per document, not one per message. A file that cannot be
read becomes a single line saying why and the turn goes on. This one needs
`poppler-utils` on the machine: pdftotext and pdftoppm are what does the
reading.

The living spec is this project's mem wiki (`mem wiki` lists the pages). Design
history is in the git log.
