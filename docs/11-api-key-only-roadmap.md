# pxy roadmap — API-key-only, harness-native (2026-08-31)

Supersedes the *sequencing* of docs/10 (its findings all stand; its Tier 1 is
cancelled). Written after a three-way sweep of `references/CLIProxyAPI`,
`references/litellm`, `references/OmniRoute` against pxy at `c94b521`.

Saiful's goal, restated: **any model from any provider, through pxy, feels the
same to Claude Code / codex / pi / opencode / fx as that harness's native
model** — while the Claude and ChatGPT subscriptions stay on their own native
clients, beside pxy rather than inside it.

---

## 0. The decision that reframes the whole queue: drop OAuth

> **STATUS: DONE 2026-08-31**, branch `remove-oauth-providers`. ~2,940 lines net
> deleted, 171/171 tests green, clippy 58 warnings (was 79, none added), real
> streaming + non-streaming requests verified in both dialects against the live
> config on a test daemon. Both config files had all four blocks commented out
> already, so nothing live changed. Details in HANDOFF NEXT STEPS §0.
> Actual removals matched the estimate below, plus three the estimate missed:
> the `credentials_file` config key, the `x-initiator` plumbing (Copilot-only),
> and `ProviderKind` itself — with one variant left it was dead weight, so the
> `kind` key is gone entirely (a leftover line now fails startup, same precedent
> as the `tier`/`promo` removals). `src/providers/` collapsed to a flat
> `src/providers.rs`; `prepare()` is now synchronous, since nothing mints tokens.

Saiful's call, 2026-08-31: keep API-key providers only, because the OAuth ones
are not legal. docs/10 §0.2 already established the compliance matrix; this
section is only about what removal *costs and buys*, since the legal side is
settled.

**Assumption being made explicit:** "use pxy beside my claude subscription"
means the Claude subscription is spent by running `claude` natively, not
through pxy. Under that reading the whole `claude` provider is redundant, not
merely risky. If instead you wanted Claude Code → pxy → Anthropic to keep
working, docs/10 Tier 0 items 3-5 come back and §0.1's gray zone comes with
them. Everything below assumes the first reading.

### Per-provider, precisely

| kind | legal status (docs/10 §0.2) | verdict |
|---|---|---|
| `ClaudeOauth` | prohibited for non-CC clients; gray for CC itself | **remove** — subscription goes native |
| `GithubCopilot` | effectively prohibited; enforcement record is real; impersonates VS Code at `providers/copilot.rs:20-49` | **remove** — highest risk in the stack, and it's a *paid* sub |
| `Kiro` | flatly prohibited, kiro.dev FAQ verbatim | **remove** |
| `KimiCoding` | **explicitly permitted** (forward proxy OK) — but `providers/kimi.rs:32` forges `kimi-code-cli/0.26.0`, breaking their one named rule | not a legal removal. Remove on *other* grounds: credits dead since 2026-08-25, and it carries the rotating-refresh machinery |
| `kilocode` | permitted, and **not OAuth code at all** — a long-lived JWT via a `cmd` secret | **keep**, nothing to do |

So the legal deletions are three; kimi is a judgement call that happens to land
the same way.

### What removal costs

- `github` — 300 premium req/month + unlimited `gpt-5-mini` (the 0x model). The
  real loss. Sanctioned replacement exists: Copilot SDK behind your own OAuth
  App, which is a build, not a config change.
- `kiro` — 50 credits/month free, scaled by `rateMultiplier` (cheap models went
  far). Meaningful but free-tier.
- `claude` — nothing, under the assumption above.
- `kimi-coding` — nothing today (500s since activation).

### What removal buys — and it is a lot more than compliance

From the source audit, deleting all four kinds removes roughly **2,540 LOC**
plus ~150 lines of router surgery:

| kind | scope |
|---|---|
| `Kiro` | `providers/kiro.rs` (316) + `translate/kiro.rs` (973) + `translate/eventstream.rs` (264) + binary fixture; `WireFormat::Kiro`; two request-build arms `router.rs:660-669`, the non-streaming eventstream branch `:936-968`, three `unreachable!` arms, `StreamCtx.kiro` and its five handling sites, two `stream_outcome` arms |
| `ClaudeOauth` | `providers/claude.rs` (295); `ensure_sentinel` at `router.rs:735-737`; accounts-forbidden validation `config.rs:265-274`; doctor check `diagnose.rs:262-285` |
| `KimiCoding` | `providers/kimi.rs` (311) minus two helpers that must be relocated first — `claude.rs:161` uses `kimi::form_encode`, `kiro.rs:272` uses `kimi::iso8601` |
| `GithubCopilot` | `providers/copilot.rs` (179) + the copilot-only `fetch_quota` branch `server.rs:1009-1035`; leaves `x-initiator` handling (`router.rs:44`, `:755-758`, `:770-778`) dead |

Structural consequences worth having for their own sake:

- **`WireFormat` collapses to two variants** (`Openai | Anthropic`). Every
  `!= Kiro` guard and `unreachable!` arm in the router goes with it.
- **`RefreshLock` (`providers/mod.rs:26-46`) and `Secrets::write_pass`
  (`secrets.rs:73`) become dead** — their only callers are claude/kimi/kiro.
  No credential in pxy is written back to `pass` any more.
- **No rotating credentials at all.** Every "losing a refresh token kills the
  session" hazard, the flock, the three per-provider process mutexes, the
  staleness re-reads, the atomic write-back with `.bak` — all gone. Secrets
  become read-only.
- `libc` stays (used by `server.rs:26` for umask), so no dep churn.
- docs/10's **Tier 1 is cancelled**: no `codex-oauth`, no `app-server` driving,
  no `chatgptAuthTokens`. `WireFormat::Responses` loses its only strong
  motivation (see §3.4).

This is the biggest single simplification available to pxy, and it happens to
be the compliant one. That is the same lesson docs/10 §0.2 closed on.

### Sequencing note

Do the removal **first**, before any fidelity work. Every item in §2-§5 is
cheaper against a two-variant `WireFormat` and a router with no eventstream
branch, and three of the four kinds are already non-functional or unused.

---

## 1. The fidelity target changes shape

docs/10 asked: *on the path where pxy translates nothing (CC → Anthropic OAuth),
how close is the wire to native?* With OAuth gone that path does not exist.

The question becomes: **on paths where pxy translates everything (CC → an
OpenAI-compat free provider), what makes the harness misbehave versus a native
model?** Different question, mostly the same answers — the Tier 0 items survive,
but their justification moves from "don't mutate a valid request" to "don't
swallow what the harness needs to see".

Two items from docs/10 Tier 0 **drop out** with OAuth:

- item 3 (skip `anthropic_sanitize` for native CC) — sanitizing is now always
  correct, because no upstream is real Anthropic-with-a-subscription. The
  prompt-cache-invalidation worry evaporates on free providers that don't cache.
  *Except* on the paid Anthropic-format reserves — see §4.1.
- item 4 (forward `x-app` / `x-stainless-*` / real UA) — these mean nothing to
  an OpenAI-compat upstream. Forwarding them to a random aggregator leaks the
  client identity for zero benefit. **Actively don't do this now.**

Everything else stands, and §2-§5 add what the three-way sweep turned up.

---

## 2. Tier 0 — bugs that break a harness today

### 2.1 `pxy_web_search` is offered when nothing can intercept it — **FIXED 2026-08-31**

`anthropic_to_openai.rs:101-106` injects the `pxy_web_search` function whenever
`stream == true` and the client sent an Anthropic `web_search*` server tool. The
interceptor, though, is only constructed when search providers exist:
`router.rs:794-798` requires `!app.cfg.search.providers.is_empty()`.

In `~/.dotfiles/home/dot_config/pxy/config.toml` **every `[[search.providers]]`
block is commented out** (lines 1000, 1006, 1011). So today: Claude Code with web
search enabled, streaming, routed to any OpenAI-format provider → the model is
offered a tool, calls it, nothing strips the call, and Claude Code receives a
`tool_use` for a tool it never declared. That is exactly the failure the module's
own comment (`web_search.rs:8-12`, and the comment at the injection site) says it
exists to prevent — the guard just isn't wired to the config.

**Fixed** by enforcing the invariant at the one place that can see all of it.
The translators inject on their own dialect's evidence; only `attempt()` knows
the upstream format, whether a search provider exists, and whether the turn
genuinely streams (interception lives entirely in `StreamCtx`, so `force_stream`
never runs the loop either). So the router now computes `has_search_tool`,
builds `SearchLoop` only when all three hold, and **strips the tool from the
outbound body whenever it won't**. That also closes the `codex --search` variant
for free: `responses.rs:329` injects before routing, so it could previously send
`pxy_web_search` to an *Anthropic* upstream, which no gate covered.

Regression test `web_search_never_reaches_an_upstream_that_pxy_cannot_serve`
covers both directions — tool absent with no provider configured (and the
client's own tool preserved), tool present when one is. Verified to fail
without the guard.

### 2.2 Non-streaming requests silently lose web search

Same injection site: no injection when `stream == false`
(`anthropic_to_openai.rs:718-725` locks this in as a test). A non-streaming
Claude Code turn that asks for search just gets no search, with no error. Either
run the search loop on the non-streaming path too, or return an honest error —
silently dropping a declared capability is the thing that makes a harness feel
subtly broken.

### 2.3 Non-`web_search` server tools vanish

`code_execution_20250522`, `bash_20250124`, `text_editor_20250124`, `computer_*`
are dropped by the tool filter at `anthropic_to_openai.rs:79` with no error
(test at `:691-714` asserts the drop). Both references do better in the same
cheap way: **recognize them by prefix so they are never mangled, and fail
loudly** rather than silently, when the target can't serve them.

- CLIProxyAPI: `IsClaudeServerToolType` prefix list,
  `helps/claude_builtin_tools.go:27`.
- litellm: explicit drop-with-warning for `computer_use`/`image_generation`/
  `shell`, `responses/litellm_completion_transformation/transformation.py:1870-1879`.

Recommendation: a declared server tool that pxy cannot fulfil on the chosen
upstream should **skip that candidate on a multi-candidate walk** (same shape as
the existing `tool_call = false` rule), and return a real 400 on a
single-candidate one. Never a silent drop.

### 2.4 `<think>` tags leak into Claude Code as visible text

`translate/think.rs` is correct and streaming-safe, but it is opt-in per provider
(`parse_think_tags`, default `false` at `config.rs:609`) and in the live config
only three providers set it (opencode-zen, opencode-go, commandcode — lines 194,
214, 245). Every other OpenAI-format provider serving a reasoning model —
google, zai, aihubmix, openrouter, kilocode, tokenharbor, inception — will send
literal `<think>…</think>` straight through into an Anthropic text block.

litellm parses `<think>` unconditionally on every OpenAI-compat response
(`common_utils.py:1532`, wired into `convert_dict_to_response.py:671`). It is
strictly safer as a default: the filter only fires on a well-formed tag pair, and
a false positive costs formatting, while a false negative costs the harness's
whole reasoning display.

Recommendation: **flip the default to on for OpenAI-format upstreams**, keep the
flag as an escape hatch.

### 2.5 Kiro's unfiltered tool schema — moot after §0

`translate/kiro.rs:249-269` doesn't filter on `input_schema`, so a server tool
became `"inputSchema": {"json": null}`. Deleted along with the provider.

### 2.6 Panic guard — **FIXED 2026-08-31**

`anthropic_sanitize.rs:23` and `router.rs` `IndexMut` a body that axum's
`Json<Value>` will happily hand over as `[]`, `"x"` or `5` → handler task panic
instead of a 400. Confirmed against serde_json 1.0.151 (`cannot access key
"messages" in JSON string`). Guarded in `handle_chat`, which both chat dialects
funnel through.

A live sweep of every JSON endpoint then turned up a second, separate defect the
audit had missed: **`/v1/responses` answered `5` with a 200 and a real upstream
call.** `responses::request` builds a fresh object from whatever it is handed, so
a scalar became an empty-but-valid chat request and pxy spent quota asking the
model what the user meant. `handle_chat`'s guard cannot see it — the body is an
object by then — so the `responses` handler needed its own edge check. All 12
malformed chat requests now return 400 with **zero** upstream calls.

---

## 3. Tier 1 — the "native feel" plumbing

These are docs/10's surviving Tier 0 items plus what the sweep added. All are
plumbing; none is impersonation.

### 3.1 Single-candidate errors must pass through raw (docs/10 §2.1)

Unchanged and now *more* important, not less: with OAuth gone, every provider's
error body is the only thing a harness has to work with.
`classify_error` (`router.rs:1217-1252`) turns `401|402|403|408|409|429|≥500`
into `Skip` regardless of `multi`, and an exhausted walk returns a synthetic
`429 overloaded_error` (`router.rs:327-336`) with the real body truncated to 200
chars inside a prose string. Generalize the existing `multi && status == 404`
carve-out to the whole ladder. Test `auth_failure_never_retried`
(`router.rs:3203-3242`) locks in current behavior and must change with it.

### 3.2 Forward upstream response headers (docs/10 §2.2)

`Outcome` (`router.rs:55-65`) has no header field, so it is structurally
impossible for any upstream header to reach the client. Add the map; forward
minus litellm's exclusion set (`transfer-encoding, content-encoding,
content-length, server, date, connection, keep-alive` —
`pass_through_endpoints.py:312-345`) plus CLIProxyAPI's gateway-telemetry prefix
strip (`x-litellm-`, `helicone-`, `x-portkey-`, `cf-aig-`, `x-kong-`, `x-bt-` —
`sdk/api/handlers/header_filter.go:8-18`), whose stated reason is that Claude
Code's telemetry detects those and reports the gateway type.

`retry-after` and `x-ratelimit-*` reaching the client are what let a harness's
own backoff work instead of fighting pxy's.

### 3.3 Generalize passive quota-header observation

pxy parses only `x-th-free-*`, `x-quota-*` and generic `x-ratelimit-*`
(`router.rs:1291-1389`). CLIProxyAPI's `ObserveResponseHeadersForProvider`
(`sdk/cliproxy/auth/quota_signals.go:37`) snapshots a per-provider allowlist from
*successful* responses — free health signal, no probe traffic. Two details worth
copying: the snapshot **replaces** rather than merges (`:26-34`) so a watermark
can't go stale, and values are CRLF-rejected (`:137`) because they land in logs.

The `anthropic-ratelimit-unified-*` family (docs/10 §4.2) still matters for the
Anthropic-format *aggregators* (agentrouter, tabitoken, gorouter), which relay
Anthropic's headers. The scope rule — `-5h/-7d Status: rejected` → credential
scope, a model-specific `7d_oi` rejection → model scope, deadline from `-Reset`
plus `Retry-After` with **1-30s random fuzz** so restarts don't thundering-herd
the reset instant (`helps/claude_ratelimit.go:22-48`) — is worth taking whole.

### 3.4 `count_tokens`, `/v1/models` negotiation, keepalive

- **`count_tokens`** (docs/10 §2.3): `server.rs:447-452` takes no `State`, so it
  physically cannot forward. Claude Code drives auto-compaction off this number,
  so a wrong answer makes compaction fire early or late versus native. Forward to
  `format = "anthropic"` upstreams; keep the local estimate as the fallback for
  OpenAI-format ones (they have no such endpoint). The estimator is at least
  honest now (`estimate_tokens`, ASCII/4 + non-ASCII×1).
- **Content-negotiated `/v1/models`**: pxy always answers OpenAI-shaped
  (`server.rs:540-605`); it works only because CC's discovery schema is loose.
  Both references negotiate on `Anthropic-Version` header or a `claude-cli` UA
  (`server_routes.go:568-608`). Cheap correctness.
- **Keepalive on long non-streaming requests**: CLIProxyAPI's
  `nonstream-keepalive-interval` (`internal/api/server_keepalive.go`) exists
  because slow upstreams trip client read timeouts. pxy's `force_stream`
  aggregation path (`translate/aggregate.rs`) has exactly this exposure — it can
  hold a socket silent for the whole generation.

### 3.5 Structured cooldown errors

When every candidate is cooling, pxy returns prose. CLIProxyAPI returns a `429`
carrying `Retry-After` plus a JSON body with `code: model_cooldown`,
`reset_time`, `reset_seconds` (`sdk/cliproxy/auth/selector.go:82-149`). A harness
can act on that; it cannot act on a sentence.

### 3.6 `WireFormat::Responses` — demoted, not dropped

With no ChatGPT subscription path, codex traffic through pxy always lands on an
OpenAI-compat upstream that doesn't speak Responses either, so the triple
translation (docs/10 §3) costs nothing real. Keep the item on the list only as
polish for the field losses (`prompt_cache_key`, `reasoning` items,
`encrypted_content`); it is no longer blocking anything.

---

## 4. Tier 2 — token economy (the compression question, answered)

Saiful asked for a compression option, globally or per provider. Three separate
things hide under that word, and they rank very differently.

### 4.1 Prompt caching is the real lever — and pxy does nothing here

**There is no `cache_control` handling anywhere in `src/`.** Grep finds only the
usage-accounting fields. Concretely:

- Anthropic → Anthropic: markers survive by accident (the payload is cloned,
  `router.rs:653`, and `anthropic_sanitize` rebuilds nothing but orphan
  `tool_result` blocks). Fine.
- Anthropic → OpenAI: stripped implicitly by reconstruction — every block is
  rebuilt field-by-field, and the catch-all at `anthropic_to_openai.rs:198`
  documents it: *"document, cache markers etc: dropped for openai upstreams"*.
  Correct, nothing to do.
- **OpenAI → Anthropic: nothing is ever injected**
  (`openai_to_anthropic.rs:73-162`). This is the gap.

After §0, the Anthropic-format upstreams left are the **paid reserves** —
agentrouter, tabitoken, gorouter (~$190 of Opus credits between them), plus
deepseek's Anthropic route. Those are the only places in pxy where tokens cost
real money, and they are exactly where no breakpoint is ever set.

Why this beats text compression, plainly: in an agentic coding session the
dominant input cost is the transcript replayed on every turn. A cache hit prices
that prefix at ~10% and is **lossless**. A 30% lossy squeeze of the same text is
both smaller and destructive. Caching wins on arithmetic before you even reach
the risk argument.

Both references implement the same shape, and it is small:
- CLIProxyAPI `ensureCacheControl` (`claude_executor_cloaking.go:1093`) — last
  system block, last cacheable message, last tool only when there is no system
  prompt; `injectMessagesCacheControl` (`:1526`) skips assistant turns ending in
  a thinking-like block, which cannot host a marker; `enforceCacheControlLimit`
  (`:1347`) evicts lowest-value-first to respect Anthropic's **4-breakpoint cap**.
- litellm `anthropic_cache_control_hook.py` — same cap enforcement (`:52`), plus
  a **yield-to-client rule** (`:231`, `:537`): if the client set any
  `cache_control` itself, injection stops and the client's TTL is preserved.
  That rule is mandatory, not optional — it is what keeps injection from
  fighting a harness that already knows what it's doing.
- Provider support must be checked before injecting (`:596`); several providers
  400 on the field (`dashscope/chat/transformation.py:15`,
  `cometapi/chat/transformation.py:55`).

**Verify per aggregator before shipping**: whether agentrouter / tabitoken /
gorouter actually relay `cache_control` upstream is unknown and must be measured
(the test is `cache_read_input_tokens` on turn 2+), not assumed. Note tabitoken
injects ~7k hidden prompt tokens per call, which interacts badly with a stable
cached prefix.

Also cheap and adjacent: **`cache_control` currently rides through
`anthropic_sanitize` intact only because the sanitizer is conservative**. When
touching that file, the `strip_invalid_blocks` path (`:36-55`) drops empty text
blocks *including any `cache_control` attached to them* — docs/10 §2.4 flagged
this and it stays true for the paid Anthropic reserves.

### 4.2 HTTP compression — already done upstream, pointless downstream

- **Upstream: already on.** `Cargo.toml:16` enables reqwest's `gzip` feature and
  nothing calls `.no_gzip()`, so every upstream request already advertises
  `accept-encoding: gzip` and transparently decodes.
- **Downstream: not implemented, and shouldn't be.** No `tower-http`, no
  `CompressionLayer`. pxy serves `127.0.0.1` to a process on the same machine —
  gzipping there spends CPU to save loopback bytes. Neither reference does it
  either: litellm's proxy registers six middlewares and none compress
  (`proxy_server.py:2066-2107`), and CLIProxyAPI has no client-side compression.

litellm also has a warning worth heeding if this is ever revisited: it
**deliberately refuses to forward the client's `accept-encoding`**
(`passthrough/utils.py:21,73-75`) because relaying `br` to a caller whose decoder
is absent hands back undecodable bytes.

So: the HTTP reading of "add a compression option" is already satisfied on the
side that matters, and a no-op on the other.

### 4.3 Prompt/context compression — recommend against as a default

OmniRoute had a large engine here: `open-sse/services/compression/` (51 files),
modes `off|lite|standard|aggressive|ultra|rtk|omniglyph|stacked`, engines
Caveman (prose condensation), RTK (terminal/tool output), CCR (content-addressed
block references), session-dedup, plus a fidelity gate and a compression
analytics dashboard. litellm has a smaller, sharper version:
`litellm/compression/compress.py:333` — BM25 + optional embedding scoring against
the last user message, protected indices, atomic Anthropic tool_use/tool_result
spans (dropping half a pair 400s), messages **stubbed rather than deleted**
(`message_stubbing.py:21`), and a `litellm_content_retrieve` tool injected so the
model can pull back anything it needs (`retrieval_tool.py:6`).
**CLIProxyAPI has none of this** — no trimming, no compression, no reasoning
stripping for cost.

Three reasons to say no for now:

1. **It is the weight pxy exists to shed.** OmniRoute's compression subsystem is
   a meaningful share of the ~800 MB / Node footprint that motivated the rewrite.
   litellm's needs BM25 plus optionally an embedding model in the request path.
2. **Lossy in the one place lossiness is unaffordable.** An agentic coding
   transcript is mostly file contents, diffs and tool output. Caveman-class
   condensation on that is how an agent starts editing a file it can no longer
   see correctly, and the failure is silent and downstream.
3. **It is redundant with what the harnesses already do.** Claude Code
   auto-compacts, opencode compacts, and `launch.rs:118-140` already enables
   that. Compressing under a compactor means two systems editing the same
   history with different models of what matters.

If it is wanted later, the shape to copy is litellm's, not OmniRoute's, and the
non-negotiable parts are: stub-don't-delete, a retrieval tool so nothing is
truly lost, tool-pair spans kept atomic, and a hard trigger threshold so short
sessions are untouched. Gate it exactly as asked — a `[compression]` block with
a global default and a per-provider override — but default `off`.

### 4.4 Delegated context editing — the middle path, if anything

Anthropic's own `context_management.edits[]` with
`clear_tool_uses_20250919` asks the *provider* to drop stale tool blocks against
its real tokenizer, rather than pxy guessing. OmniRoute scoped it to Claude-only
for a good reason (others 400 on the field) with a 400-fallback, per
`docs/compression/CONTEXT_EDITING.md`. pxy already passes the field through
untouched to Anthropic-format upstreams, so this is opt-in *config*, not code —
worth trying on the paid reserves before writing any compression engine.

litellm goes further and **polyfills** it for non-Anthropic providers
(`context_management/editors/clear_tool_uses.py`, `compact.py` — the latter calls
a separately-configured summarization model). That is the compression engine
again wearing a different hat; same recommendation, same reasons.

---

## 5. Tier 3 — reasoning fidelity

pxy is in better shape here than expected, and better than docs/10 implied.
What already works:

- `thinking.budget_tokens` → `reasoning_effort` buckets
  (`anthropic_to_openai.rs:132-144`) and `reasoning_effort` →
  `thinking.budget_tokens` with a `max_tokens` bump to `budget + 4096`
  (`openai_to_anthropic.rs:144-160`). That `max_tokens` bump is a real footgun
  handled correctly — litellm needs the same code
  (`base_llm/chat/transformation.py:103`) and CLIProxyAPI has it as
  `normalizeClaudeBudget` (`provider/claude/apply.go:171`).
- Upstream `reasoning_content` **or** `reasoning` (the z-ai/GLM spelling) is read
  (`anthropic_to_openai.rs:323-328`) and surfaced to Anthropic clients as real
  `thinking` blocks, streaming included (`:457-458`, `:477-512`), **with no
  fabricated `signature`** — deliberate, documented, and correct.
- The reverse (Anthropic thinking → OpenAI `reasoning_content`) is also wired
  (`openai_to_anthropic.rs:225`, `:355-357`), including into the Responses and
  AI-SDK shapes.

The gaps, in order:

1. **The `<think>` default** — §2.4. Highest impact, one-line default change.
2. **Effort levels are coarse and unclamped per model.** pxy has three buckets;
   both references have a canonical table and per-model clamping (CLIProxyAPI
   `internal/thinking/convert.go:11` with `none/auto/minimal/low/medium/high/
   xhigh/max`; litellm `constants.py:172,191-197`, env-overridable, plus
   per-model capability read from the cost map). pxy handles the "provider 400s
   on this param" case with `drop_params` config, which is adequate — but a
   model whose minimum budget is higher than pxy's `low` bucket silently gets a
   worse answer than native.
3. **`thinking` is dropped for Kiro** — moot after §0.
4. **Anthropic-dialect history loses thinking on OpenAI upstreams** — assistant
   `thinking` blocks hit the catch-all at `anthropic_to_openai.rs:251-252`, and
   pxy-synthesized blocks can never survive a second turn (stripped by
   `anthropic_sanitize.rs:39-48` if they later land on Anthropic, since they
   carry no signature). **This is correct and should stay.** Both references
   agree on the rule — litellm's `_is_unsignable_thinking_block`
   (`factory.py:2330`) drops rather than repairs, because Anthropic verifies the
   signature cryptographically. pxy already never mints an empty signature
   (HANDOFF's poisoned-history invariant).
5. **The heavyweight thing pxy lacks: reasoning replay.** CLIProxyAPI caches
   signed/encrypted reasoning server-side per session and re-injects it
   (`internal/cache/codex_reasoning_replay_cache.go`, four peers), with a
   cross-provider signature compatibility engine (`internal/signature/`, ~5.6k
   LOC, `DecideSignatureCompatibility` at `provider_compatibility.go:242`
   returning preserve/drop_block/drop_signature/bypass). litellm has the
   equivalent scar tissue: INTERLEAVED vs SEQUENTIAL thinking-block replay
   ordering because *"Anthropic verifies thinking block signatures based on
   position"* (`factory.py:2536-2690`), and a 400-triggered strip-and-retry
   (`common_utils.py:947,962`). **Don't build this.** It exists to keep a
   subscription-grade model's reasoning verifiable across turns and across
   provider switches — the free-provider chain has no signatures to preserve,
   and after §0 pxy has no first-party reasoning upstream at all.

One cheap borrow worth taking: CLIProxyAPI's **model-name suffix**
(`internal/thinking/suffix.go:23`) — `zai/glm-4.7-flash(high)`,
`.../model(16384)`, `(none)`. It gives per-request effort control through any
harness that only lets you type a model name, which is all of them.

---

## 6. Tier 4 — routing and cooldown

pxy is already competitive here; this is the shortest list in the document.
Present and matching the references: fill-first multi-account walks, session
affinity with a fixed FNV hash, failure-rate cooldowns, config-driven
request-scoped error rules, persistent cooldowns across restarts, in-request
retries that only sleep on retryable cooldowns, context-window peer-skipping.

Worth taking:

1. **Round-based retry caps.** CLIProxyAPI bounds both the number of distinct
   credentials tried per round (`max-retry-credentials`) and how long a round
   will wait for a cooldown (`max-retry-interval`), with per-credential opt-in
   depth (`conductor_execution.go:56,318,496,679`). pxy's equivalent is a fixed
   "2 more walks, ≤10s" (HANDOFF). Making it configurable is small.
2. **Structured cooldown 429** — §3.5.
3. **Passive quota observation** — §3.3.

Deliberately not taking: weighted/smooth round-robin and least-busy (fill-first
is the correct strategy for stacked free tiers, and is already implemented),
latency-based and cost-based routing (free-first makes cost routing moot),
litellm's budget/tag/complexity/bandit routers and per-key/team budget
enforcement (all multi-tenant features; pxy has one user).

One litellm idea that is genuinely tempting and still a no: `complexity_router`
(`router_strategy/complexity_router/`) — rule-based, zero API calls, <1ms, scores
7 dimensions to pick a tier. It would let cheap models take easy turns. It is
also a quality lottery inside an agent loop, where one bad turn costs more than
the tokens saved. Revisit only if `pxy status` shows real exhaustion.

---

## 7. Explicitly rejected (carried forward)

From docs/09 §11 and docs/10 §0, all still rejected and now largely moot after
§0: uTLS/JA3 spoofing, `cch=` body signing, MCP tool-name cloaking, zero-width
obfuscation, synthetic device identities, plugin FFI ABI, WebSocket/Realtime
surfaces, cluster mode, management REST API + TUI.

Newly rejected in this round: prompt/context compression engines (§4.3),
reasoning replay caches and signature compatibility engines (§5.5), downstream
HTTP compression (§4.2), multi-tenant routing strategies and budget enforcement
(§6), forwarding client identity headers to third-party upstreams (§1).

Also worth recording: **neither CLIProxyAPI nor litellm implements MCP
passthrough** (`mcp_servers`, `mcp_list_tools`, `mcp_call`). CLIProxyAPI's only
MCP code is tool-name *cloaking*, which is impersonation. pxy passes
`mcp_servers` through untouched to Anthropic-format upstreams and drops it for
OpenAI ones — which is the same behavior, arrived at by not implementing
anything. Fine.

---

## 8. Order of work

1. ~~**§0 removal.**~~ **DONE 2026-08-31** — four provider kinds deleted,
   `WireFormat` collapsed, `RefreshLock` / `write_pass` / `ProviderKind` gone.
2. **§2 bugs.** ~~Web-search injection guard (live)~~ DONE, ~~`is_object()`
   guard~~ DONE (plus the `/v1/responses` edge it exposed). Remaining:
   non-streaming search, server-tool skip/error, `<think>` default.
3. **§3 plumbing.** Raw single-candidate errors → response headers →
   `count_tokens` forwarding → `/v1/models` negotiation → structured cooldown
   429 → keepalive.
4. **§4.1 cache_control injection** on the paid Anthropic reserves, gated on the
   yield-to-client rule and the 4-breakpoint cap, verified by
   `cache_read_input_tokens` on turn 2+.
5. **§5.2 / §5 suffix** and **§6** items as polish.

## 9. Verification gate

docs/10's gate, adapted — it stays a diff, not a vibe:

- Run the same prompt through each harness twice: native model versus pxy-routed
  model. Compare error bodies on a forced 429, the header set the client
  receives, and `cache_read_input_tokens` on turn 2+ of a long session.
- Per-harness conformance: Claude Code (`web_search` + streaming + compaction
  threshold), codex (`/v1/responses` tool round-trip), opencode (auto-compaction
  needs a real `context_length`), pi, fx (the finish-reason and duplicate
  tool-call-id rules from HANDOFF).
- The existing mock-upstream integration harness covers every §2 and §3 item
  without touching a real provider.
