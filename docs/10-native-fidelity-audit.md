# Native-fidelity audit — 2026-08-31

Goal set by Saiful: *"when I use Claude Code with my Claude subscription through
pxy, I want 100% the same performance as native. Same for codex with a ChatGPT
subscription. So I can use pxy everywhere without worrying."*

Sweep of `references/CLIProxyAPI` (HEAD `f0de1d0`, 2026-08-29) and
`references/litellm` (HEAD `f005afa`) against pxy as of `c94b521`, plus a
line-by-line trace of pxy's own Anthropic→Anthropic path. Claims below carry
file:line — **verify before acting** (docs/09 §9 is the standing reminder that
research docs drift).

docs/09 covered routing/translation/quota breadth. This one covers a different
axis: **on the paths where pxy is not translating anything — Claude Code →
`claude` provider → api.anthropic.com — how close is the wire to native?**

---

## 0. The framing that matters

CLIProxyAPI spends roughly half its Claude code on *impersonation*: uTLS
ClientHello spoofing pinned to Claude Code 2.1.220, HTTP header name-casing and
ordering reconstruction, a reverse-engineered `cch=` xxHash body signature,
synthetic per-credential device identities, MCP tool-name cloaking, zero-width
obfuscation of sensitive words. It needs all of that because it re-shapes
foreign clients (a Gemini CLI request, an OpenAI SDK request) into something
Anthropic's OAuth endpoint will accept from "Claude Code".

**pxy's target case does not need any of it.** When the client genuinely *is*
Claude Code and the credential is Saiful's own Claude subscription, the request
already has native shape and a legitimate token. Fidelity here is achieved by
**subtraction, not addition** — stop mutating a request that was already
correct. Everything in §2 is something pxy currently *does to* a valid request.

This also keeps docs/09 §11's non-goal intact: the stealth/ban-evasion material
(JA3/JA4 impersonation, `cch` signing, zero-width cloaking) stays rejected. It
is not needed for the stated goal, and it rots — `helps/utls_client.go:160-163`
says to re-capture packets whenever the Claude Code version changes.

---

## 0.1 Compliance — checked 2026-08-31, and it constrains the design

**Why CLIProxyAPI impersonates Claude Code**: not for performance. Anthropic
added a *server-side* authenticity check. Around 2026-01-09 a server-side switch
started rejecting subscription OAuth credentials presented by non-Claude-Code
clients ("this credential is only authorized for use with Claude Code"). The
whole impersonation stack — uTLS ClientHello, header casing/ordering, the `cch=`
body hash, synthetic device identity, the `claude-code-20250219` beta, the
system sentinel — exists to defeat that check. It is circumvention of a
technical restriction, by construction.

**Authoritative policy text** (code.claude.com/docs/en/legal-and-compliance,
"Authentication and credential use"):

> OAuth authentication is intended **exclusively** for purchasers of Claude
> Free, Pro, Max, Team, and Enterprise subscription plans and is designed to
> support **ordinary use of Claude Code and other native Anthropic
> applications**.
>
> Anthropic does not permit third-party developers to offer Claude.ai login
> into their own applications, or to **route requests through Free, Pro, or Max
> plan credentials on behalf of their users**. Moreover, developers may not
> collect, store, or **intermediate Claude.ai credentials or session tokens**.
>
> Anthropic reserves the right to take measures to enforce these restrictions
> and may do so **without prior notice**.

Same page, Acceptable use: *"Advertised usage limits for Pro and Max plans
assume **ordinary, individual usage** of Claude Code and the Agent SDK."*

Consumer Terms §3 adds: no accessing the Services *"through automated or
non-human means, whether through a bot, script, or otherwise"* **except via an
Anthropic API key**.

Timeline: server-side switch ~2026-01-09 → docs updated 2026-02-19/20 →
enforcement 2026-04-04, at which point Pro/Max/Team subscriptions stopped
covering third-party harnesses (OpenCode and OpenClaw named in reporting).

### What this means for pxy's two cases

| | risk |
|---|---|
| **A. Claude Code → pxy → Anthropic**, Saiful's own credential, his own use | **Gray, low-moderate.** The client is the unmodified CC binary, one individual, own credential, ordinary usage — the substance the policy protects. Friction: pxy reads/refreshes/writes the credential file, which is literally "intermediating … session tokens" (though that sentence targets developers acting *on behalf of their users*), and the localhost hop is arguably "a script". Detection risk is mostly fingerprint drift (§2.5). |
| **B. opencode / codex / pi / fx → pxy → Anthropic** on the Claude credential | **High — this is the banned pattern by name.** HANDOFF's "non-CC clients get the CC system sentinel auto-prepended" is functionally the same trick OpenCode was called out for. This is the account-suspension vector. |

### Design consequences (binding)

1. **The `claude` provider must serve Claude Code only.** The sentinel
   auto-prepend for non-CC clients should be **removed**, not improved — that
   feature *is* the violation. Everything else can route freely.
2. **§2.5 header forwarding must be forward-only, never synthesize.** Passing
   through the headers the real Claude Code sent is subtraction-of-mutation and
   compliant. Attaching those same headers to an opencode request would be
   spoofing. The implementation must copy what the client sent and never
   fabricate `x-app` / `x-stainless-*` / a UA for a client that didn't send one.
3. HANDOFF's existing "Anthropic models never in `auto`" rule now has a
   compliance rationale on top of the cost one. Keep it.
4. No amount of engineering closes the remaining gap. The Claude-subscription
   part of "use pxy everywhere without worries" has a **policy ceiling**, not a
   technical one. The only contractually grounded path for non-CC clients is an
   Anthropic Console API key.

### Codex / ChatGPT (bears on §1 and Tier 1) — researched 2026-08-31

Opposite posture, and materially better than Anthropic's. **Every binding
OpenAI document is SILENT** on third-party clients and plan credentials (Terms
of Use eff. 2026-01-01, ROW Terms, Service Terms, Usage Policies — full-text
verified). Decisively, OpenAI has **no "access the Services only through
interfaces we provide" clause** — the exact clause Anthropic has and OpenAI
lacks. The words "interface", "third-party", "harness", "proxy" appear nowhere
in the restrictions. Nearest applicable clauses: no sharing account credentials
(Registration), no "automatically or programmatically extract data or Output"
(anti-scraping/distillation in context — OpenAI ships `codex exec` and an SDK,
so a literal reading is untenable), no circumventing rate limits.

The one on-point official statement — **@thsottiaux (Thibault Sottiaux, OpenAI
Codex/ChatGPT product lead), 2026-08-21**, on reports of changed usage limits:

> Converting a subscription into api traffic to then **re-serve or share across
> many users** is not something we support and this type of usage gets flagged
> by our fraud-prevention systems.
>
> **You are completely fine if you use your subscription through Sign in With
> ChatGPT, either through the official clients or through one of the many OSS
> clients (Pi, OpenCode, …)** that support signing in with your account and
> using your included usage.

A single-user localhost proxy straddles this: blessed on "own credential, own
usage", uncomfortable on "converting a subscription into api traffic". The
qualifier *"to then re-serve or share across many users"* is subordinate to
"converting", so the objectionable act is the sharing — which puts pxy on the
safe side of the sentence, unverified.

**Corrections to earlier assumptions in this doc's first draft:** the
2026-08-20 "Codex as a platform" post does **not** authorize subscription auth
— it explicitly says *"the open-source layer is the harness and integration
surface; **model access and managed services remain separate**."* And the
enforcement reports don't survive scrutiny: the OpenClaw 429 issue (#54615)
was closed as a client-side stale-auth-profile bug fixed in v2026.4.22, and the
"Pro account banned" thread is attributed to a shared hospital egress IP.
**No credible evidence exists of a ban for third-party-client OAuth use.**

**Design consequence for §1/Tier 1 — this changes the build.** OpenAI documents
an official surface for exactly this case: `codex app-server` accepts
**`chatgptAuthTokens`**, described as *"intended for host apps that already own
the user's ChatGPT auth lifecycle"*, taking `accessToken`, `chatgptAccountId`
and `chatgptPlanType` directly, with a documented
`account/chatgptAuthTokens/refresh` request
(learn.chatgpt.com/docs/app-server). Codex's own Rust code treats a non-first-
party `originator` as an expected category (`is_first_party_originator`,
feature-gated with "Only support first-party clients for now"), and the
override var is named `CODEX_INTERNAL_ORIGINATOR_OVERRIDE` — so spoofing a
first-party originator is clearly not intended.

**So Tier 1 should drive `codex app-server` rather than reimplement the wire
protocol against `chatgpt.com/backend-api/codex`.** Same capability, inside an
OpenAI-shipped harness on a documented API, instead of hand-rolled traffic that
looks like sub2api to a fraud classifier. Caveat: the permission rests on a
revocable X post, not contract — Anthropic went silent → prohibited → enforced
in ~3 months.

---

## 0.2 Per-provider compliance matrix — researched 2026-08-31

Primary sources only; every verdict below traces to vendor text, not commentary.
The axis that matters is **not** free-vs-paid, it is *where the credential came
from* and *whether pxy misrepresents the client*.

Two red lines are common to every vendor checked: **don't share the credential
with other people**, and **don't misrepresent your client**. Only Moonshot
states the permitted side positively.

| provider | credential | verdict | action |
|---|---|---|---|
| `claude` | borrowed Claude Code OAuth | **prohibited** for non-CC clients (§0.1) | close the non-CC path |
| `github` / `github-free` | `copilot_internal/v2/token` mint | **effectively prohibited — now the highest risk in the stack** | see below |
| `kiro` | Kiro OAuth | **explicitly prohibited**, verbatim | remove or accept ban risk |
| `kimi-coding` | device-flow OAuth | **explicitly permitted** — but pxy violates a named rule | fix the UA, then fine |
| `kilocode` | device-flow token | **permitted** | fine; console key is cleaner |
| `zai` (plain key) | console API key | **permitted** | fine — never add a Coding Plan key |
| `opencode-go` / `opencode-zen` | console API key | **permitted** ("use with any agent") | fine |
| codex/ChatGPT (unbuilt) | ChatGPT OAuth | **silent + blessed for OSS clients** (§0.1) | build via app-server |
| all other providers | console API keys | ordinary intended use | fine |

### GitHub Copilot — the finding that should change something

GitHub's *written terms are nearly silent*: Copilot Pro routes to ToS §J (AI
Features), which covers ownership/training/indemnity and says nothing about
clients or proxying. The nearest clause is AUP §6: *"You will not reproduce,
duplicate, copy, sell, resell or exploit any portion of the Service, use of the
Service, or **access to the Service** without our express written permission."*

But the **enforcement record is primary and strong**:

1. GitHub **TOS-blocked** `aaamoon/copilot-gpt4-service`, the original
   Copilot→OpenAI proxy. Independently re-verified 2026-08-31:
   `GET api.github.com/repos/aaamoon/copilot-gpt4-service` → 403
   `{"message":"Repository access blocked","block":{"reason":"tos",…}}`.
   A ToS block, not DMCA.
2. GitHub's abuse-team suspension email names the conduct verbatim (reproduced
   by suspended users in community discussions #186400, #190535):
   *"...use of Copilot via scripted interactions, an otherwise deliberately
   unusual or strenuous nature, or **use of unsupported clients** or multiple
   accounts to circumvent billing and usage limits. Due to this, we have
   suspended your access to Copilot."*
3. A **sanctioned path now exists**, which removes the "merely undocumented"
   defence: GitHub officially supports OpenCode with Copilot subscriptions via
   *formal partnership* (changelog 2026-01-16), and the Copilot SDK documents
   third-party access as *your own registered OAuth App → user authorizes →
   pass their `gho_`/`ghu_` token to the SDK*, billed to the user's subscription.

**pxy is squarely in the named conduct.** `providers/copilot.rs:20-49` sends
`copilot-integration-id: vscode-chat`, `editor-version: vscode/1.126.0`,
`editor-plugin-version: copilot-chat/0.54.0`,
`user-agent: GitHubCopilotChat/0.54.0`. That is impersonating VS Code, not
merely using an undocumented endpoint. This is a **paid subscription Saiful
relies on** (300 premium req/month), and the realistic downside is losing
Copilot access, not a warning.

### AWS Kiro — the only flat prohibition

Verified verbatim today by direct fetch of kiro.dev/faq (also on /pricing and
/cli), under *"With which tools can I use my Kiro subscription?"*:

> Kiro subscriptions can be used with Kiro IDE, Kiro CLI, Kiro on the web, Kiro
> Crew, ACP compatible IDEs, and automation in software development (ex:
> reviews during CI/CD). **Use through third-party automation harnesses (such
> as OpenClaw) that route requests outside of Kiro's native interfaces is not
> permitted.**

Added silently ~2026-04-10 (no changelog); the original wording was blunter
("Use with OpenClaw and similar tools that leverage third-party harnesses is
prohibited"). Backed by AWS Customer Agreement §6.4 ("...in any manner or for
any purpose other than as expressly permitted"). There is no standalone Kiro
ToS; Kiro appears in AWS Service Terms §50 only.

Enforcement is real but **opaque**: ~87 "unusual user activity" suspension
issues on `kirodotdev/Kiro`, a wave of 13 on 2026-08-16, but AWS never states a
reason and no kiro2api-family repo has been taken down. The proxy→ban causal
link is community inference, not vendor-confirmed. Mitigating for pxy: this is
the *free* 50-credit tier, so the loss exposure is the AWS account standing,
not money already spent.

### Moonshot Kimi — permitted, and pxy breaks the one rule that matters

Kimi Code's Community Guidelines welcome third-party clients by name:

> Kimi Code subscriptions are for interactive use only. We're compatible with
> mainstream coding tools and agent frameworks (Kimi CLI, VS Code, Claude Code,
> OpenCode, OpenClaw, etc.), so you can call Kimi Code's AI capabilities from
> the tools you already use.

Four conditions attach; the on-point one for a proxy is FAQ Q2:

> 通过代理访问 Kimi（**正向代理**）没问题；把自己的账号给别人转发（**反向代理**），属于违规。
> *Accessing Kimi through a proxy (**forward proxy**) is fine; forwarding your
> own account to other people (**reverse proxy**) is a violation.*

That is exactly the distinction pxy needs, and pxy is on the permitted side —
**except for rule 3, 不伪造或篡改客户端身份信息 ("don't spoof or alter client
identity information"), which explicitly names User-Agent forgery.**
`providers/kimi.rs:32` sends `user-agent: kimi-code-cli/0.26.0`.

This is gratuitous: Moonshot permits third-party clients outright, so pxy gains
nothing by forging the CLI's identity and takes on the one violation the vendor
actively screens for. Moonshot deployed a UA-based risk-control rule in
May 2026 that false-flagged paying subscribers with `access_terminated_error`
and was rolled back (staff `yuikns`, forum.moonshot.ai/t/…/421). **Send an
honest `pxy/<version>` UA instead.** Note the X-Msh-* device identity is a
different thing — it is the device profile minted at login and must be reused,
not forged.

### The pattern worth internalizing

pxy's own impersonation is concentrated in exactly the two providers that
punish it (`copilot.rs`, `kimi.rs`) and absent from the ones that don't care.
Same lesson as §0: **the honest request is usually also the compliant one**,
and where a vendor permits third-party clients, forging identity converts a
permitted use into a violation for zero benefit.

---

## 1. Structural gap: there is no ChatGPT/Codex subscription path at all

- `ProviderKind` = `OpenaiCompat | GithubCopilot | KimiCoding | Kiro |
  ClaudeOauth` (`src/config.rs:386-399`). No ChatGPT/Codex OAuth kind.
- No `~/.codex/auth.json` on this machine; no `openai`/`chatgpt` entry under
  `pass AI/`. The subscription credential does not exist yet either.
- `pxy launch codex` wires codex as a **client** of pxy; its traffic then lands
  on free/aggregator providers. That can never match native ChatGPT-subscription
  quality, because it is not the same upstream.

So "use my codex subscription through pxy" is blocked on two independent
things: acquiring/logging in the credential, and building a provider kind that
spends it.

CLIProxyAPI has the complete recipe (`internal/auth/codex/openai_auth.go:24-30`,
`sdk/auth/codex_device.go:27-33`):

| | |
|---|---|
| authorize | `https://auth.openai.com/oauth/authorize` |
| token/refresh | `https://auth.openai.com/oauth/token` (form-encoded) |
| client id | `app_EMoamEEZ73f0CkXaXp7hrann` |
| redirect | `http://localhost:1455/auth/callback` |
| auth scope | `openid email profile offline_access` |
| refresh scope | `openid profile email` (**differs** — docs/09 §5 warns sending the wrong scope on refresh revokes siblings) |
| extra params | `prompt=login`, `id_token_add_organizations=true`, `codex_cli_simplified_flow=true` |
| account id | ID-token claim `https://api.openai.com/auth` → `chatgpt_account_id` (`jwt_parser.go:22,42-52`) |
| inference | `POST https://chatgpt.com/backend-api/codex/responses` |
| headers | `Chatgpt-Account-Id`, `Originator: codex-tui`, `Session-Id` = `prompt_cache_key`, UA `codex-tui/0.146.0 (…)` |

A device-code flow exists (`…/api/accounts/deviceauth/usercode` → poll
`…/deviceauth/token`), which is the headless/SSH-friendly path — the same shape
as the kimi login already in `providers/kimi.rs`.

---

## 2. Claude path: what pxy does to a request that was already native

Body handling is **field-preserving** — everything is `serde_json::Value`, there
are no typed request structs, so `cache_control`, `betas`, `mcp_servers`,
`context_management` and any future field survive (`server.rs:342-350`,
`router.rs:652-653`). The damage is elsewhere.

### 2.1 Single-candidate errors are replaced with a synthetic 429 — **worst bug**

`classify_error` (`router.rs:1217-1252`) treats `401 | 402 | 403 | 408 | 409 |
429 | ≥500` as `Skip` **regardless of `multi`**. When the walk exhausts,
`handle_chat` returns (`router.rs:327-336`):

```rust
error_outcome(client_format, 429, "overloaded_error",
    &format!("no provider available for '{requested}' (tried/skipped: {})", …))
```

So an explicit `claude/claude-fable-5` request that hits Anthropic's real
`{"type":"error","error":{"type":"rate_limit_error",…}}` reaches Claude Code as
pxy's `overloaded_error`, with the real body only truncated to 200 chars inside
a prose string. The 404-carve-out precedent (`multi && status == 404`,
HANDOFF invariant) is exactly the rule that should apply to the whole ladder.

Consequence: Claude Code's "usage limit reached, resets at …" UI and its
status-specific retry logic both read the real error type — neither works
through pxy today. Test `auth_failure_never_retried` (`router.rs:3203-3242`)
locks in the current behavior, so this is a deliberate-looking change that needs
its test updated.

### 2.2 Every upstream response header is dropped

`Outcome` (`router.rs:55-65`) has no header field — it is structurally
impossible for an upstream header to reach the client. Lost:
`anthropic-ratelimit-unified-status`, `-unified-reset`, the
`anthropic-ratelimit-{requests,tokens}-*` family, `retry-after`, `request-id`,
`anthropic-organization-id`.

pxy does not even *read* them: `grep -rn "anthropic-ratelimit" src/` → zero
hits. Its cooldown header parsing (`router.rs:1291-1389`) only knows
`x-th-free-*`, `x-quota-*`, generic `x-ratelimit-*`. So Anthropic's own
authoritative quota signal is invisible to both the client and the router.

Both reference implementations solve this the same way:
- litellm forwards all upstream headers minus a hop-by-hop/encoding exclusion
  set `{transfer-encoding, content-encoding, content-length, server, date,
  connection, keep-alive}` (`pass_through_endpoints.py:312-345`).
- CLIProxyAPI gates on `passthrough-headers: true` and additionally strips
  **AI-gateway telemetry prefixes** `x-litellm-`, `helicone-`, `x-portkey-`,
  `cf-aig-`, `x-kong-`, `x-bt-` (`sdk/api/handlers/header_filter.go:8-18`),
  with the comment that Claude Code's client-side telemetry detects these and
  reports the gateway type. Worth copying verbatim — one match list.

### 2.3 `count_tokens` is answered locally, always

`server.rs:447-452`: the handler takes no `State`, so it *cannot* reach an
upstream. It returns `estimate_tokens(messages)+system+tools` (ASCII/4 +
non-ASCII×1, `translate/mod.rs:48-70`) and omits
`cache_creation_input_tokens`/`cache_read_input_tokens` entirely.

This contradicts pxy's own research note (`docs/04:106` — "pass through to
Anthropic-format upstreams; else local estimate"). Claude Code drives
auto-compaction off this number, so compaction fires early or late relative to
native. CLIProxyAPI forwards to the real
`POST {base}/v1/messages/count_tokens?beta=true` for first-party credentials —
and uses a **different, much smaller `anthropic-beta` profile** there
(`claude_code-20250219, [oauth], interleaved-thinking, context-management,
token-counting-2024-11-01`, "verified identical across 37 captured calls",
`claude_executor_request.go:176-195`), stripping `metadata`,
`context_management` and `diagnostics` which Anthropic rejects on that path.

### 2.4 `anthropic_sanitize` runs on clean Claude Code traffic

`router.rs:674-676` gates on `upstream_format == WireFormat::Anthropic`, **not**
on provider kind or client dialect. The repair rules were written for foreign
clients (docs/09 L2/L3) but they run on native CC transcripts too:

| rule | file:line | what it can do to valid CC traffic |
|---|---|---|
| `strip_invalid_blocks` | `anthropic_sanitize.rs:36-55` | drops `{"type":"text","text":""}` blocks **including any `cache_control` breakpoint attached to them** |
| `repair_tool_pairs` | `:59-138` | `open_calls` is cleared by *any* user message (`:132`); a `tool_result` answering an earlier-than-immediately-previous `tool_use` is rewritten into a text block `"[Tool result]: …"` (`:121-129`) and a synthetic `"[No response received]"` result is injected (`:83-92`) |
| `trim_trailing_assistant` | `:172-197` | trims trailing whitespace off the last assistant text block — breaks intentional prefill |
| `ensure_first_user` / empty-messages | `:166-170`, `:28-31` | injects a `{"role":"user","content":"."}` turn |

The `healthy_history_untouched` test (`:322-336`) only proves a 4-message toy
transcript is a no-op.

**Why this is the "performance" answer.** Any edit inside the cached prefix
invalidates the prompt cache from that point forward. On a subscription that is
not just latency — cache misses are billed against the quota. This is the most
plausible mechanism by which a long pxy session would feel measurably worse
than native, and it is silent.

CLIProxyAPI's equivalent rule: cloaking and cache-control placement are both
disabled unconditionally for a *confirmed native client*
(`resolveClaudeWirePolicy`, `claude_executor_cloaking.go:984-990`;
`shouldEnsureCacheControl`, `:1199-1201`) — "native owns its own breakpoints".
Its detector requires **all four** signals (`x-app: cli` + plausible
`claude-cli/…` UA + a `claude-code-*` beta + a valid `metadata.user_id`) plus an
entrypoint allowlist (`helps/claude_client_detection.go:66-70,128-152`).

### 2.5 Request-header fingerprint diverges from native

`server.rs:82-94` captures only `x-initiator` and `anthropic-beta` from the
client; the upstream request is rebuilt from scratch (`router.rs:766-786`).

| header | Claude Code sends | pxy sends |
|---|---|---|
| `anthropic-beta` | yes | forwarded, `oauth-2025-04-20` appended as a second header line |
| `anthropic-version` | `2023-06-01` | dropped; hardcoded `2023-06-01` (`router.rs:783-785`) |
| `user-agent` | real `claude-cli/<version>` | replaced by static `claude-cli/2.1.220 (external, cli)` from `config.example.toml:132` — a pinned, aging fingerprint |
| `x-app: cli` | yes | **dropped** |
| `x-stainless-*` (9 headers) | yes | **all dropped** |
| `x-client-request-id` | yes | dropped |
| `anthropic-dangerous-direct-browser-access` | yes | dropped |

Forwarding these on the claude-oauth path is pure subtraction-of-mutation: the
client's own values, unchanged. It is *not* impersonation — CC really is the
client. (CLIProxyAPI additionally reconstructs header **casing and ordering**
via `applyClaudeWireHeaderCasing`, `claude_executor_request.go:988-1028`, since
Go canonicalizes and sorts. Lower priority; `hyper` gives more control than Go
does, but nothing observed suggests Anthropic gates on it for a real token.)

### 2.6 The 128k default context gate can refuse a long native session

An id **not listed** under `[providers.claude].models` gets a fabricated spec
with `default_context() = 128_000` (`catalog.rs:236-238`, `config.rs:901-906`),
and `check_candidate` (`router.rs:566-571`) rejects locally and
**unconditionally** — not gated on `multi`. The result is §2.1's synthetic 429,
never reaching Anthropic, whose real window is 200k/1M.

Scope check: the five configured ids carry explicit `context_length`
(1M for fable/opus/sonnet-5, 200k for haiku) and gateway model discovery means
the picker only offers what pxy advertises — so this bites on a **dated or
newly-released id** (`claude-sonnet-4-5-20250929`, next month's model), not on
the everyday path. Still wrong: on a single-candidate request the upstream
should decide.

### 2.7 Panic hazard

`anthropic_sanitize.rs:23` and `router.rs:678` use `IndexMut` on the body.
A `POST /v1/messages` with a scalar/array body (`[]`, `"x"`, `5`) passes axum's
`Json<Value>` extractor and **panics the handler task** instead of returning
400. One `is_object()` guard.

### 2.8 Not a bug — corrected

The `drop_params = ["temperature","top_p","top_k"]` on the three Claude 5 ids
(`config.example.toml:136-138`) is **deliberate and correct**: those models 400
on non-default sampling params. Leave it.

Streaming responses are already **byte-verbatim** SSE passthrough
(`router.rs:1838-1856`, `StreamKind::AnthropicPass`), including `event:` lines
and `: ping` comments, with held chunks re-emitted in order. Non-streaming 200
bodies pass through as verbatim `Value`. That half is right.

---

## 3. `/v1/responses` is always a triple translation

`WireFormat` = `Openai | Anthropic | Kiro` (`config.rs:369-376`). There is no
Responses variant, so `server.rs:172-235` always down-converts codex traffic to
chat-completions and re-synthesizes the Responses envelope. Routed onto an
Anthropic upstream that becomes Responses → chat → Anthropic → chat → Responses.

Lost on the request side (`translate/responses.rs:21-203`): `previous_response_id`,
`store`, `background`, `include`, `metadata`, `prompt_cache_key`,
`safety_identifier`, `service_tier`, `truncation`, `conversation`,
`text.verbosity`, `reasoning.summary`; **all `reasoning` input items including
`encrypted_content`** (catch-all `:151-154`); `image_generation`,
`code_interpreter`, `file_search`, `mcp`, `computer_use_preview` tools (`:331`).

Lost on the response side: `model` echo in the stream skeleton (`:462-472`),
`output_text`, `instructions`/`tools` echo, annotations, `response.error` shape,
real `cached_tokens`/`reasoning_tokens`.

For today's free-provider usage this is acceptable — those upstreams do not
speak Responses either. It becomes the blocking issue the moment a ChatGPT
subscription is added (§1), because `chatgpt.com/backend-api/codex/responses`
speaks Responses natively and codex↔ChatGPT should be a passthrough, not a
double translation. CLIProxyAPI registers **no** codex→codex translator at all
for exactly this reason (`sdk/translator/registry.go:66-107` falls through to
the original bytes).

---

## 4. Worth stealing from CLIProxyAPI (beyond the above)

1. **Request-scoped vs credential-scoped errors** as a first-class concept
   (`sdk/cliproxy/auth/conductor_cooldown.go`). Concrete case: a Claude 429
   whose body mentions "fast mode … usage credits" is *request*-scoped, so one
   `speed:"fast"` request does not cool the whole credential
   (`claude_executor_request.go:284-323`). pxy's config-driven
   `[[providers.X.errors]]` rules (HANDOFF, 2026-08-30) are the right hook —
   this is a ruleset, not new machinery.
2. **`Anthropic-Ratelimit-Unified-*` parsing** (`helps/claude_ratelimit.go:22-48`):
   `-5h/-7d-Status: rejected` → credential-scoped; a Fable-only `7d_oi`
   rejection while 5h/7d are `allowed` stays **model**-scoped. Deadline from
   `…-Reset` + `Retry-After`, latest wins, **plus 1–30s crypto-random fuzz**
   so a restart fleet doesn't thundering-herd the reset instant. Pairs directly
   with §2.2.
3. **Refresh hardening** (`internal/auth/claude/anthropic_auth.go:44,70-116,
   493-499,579-581`): singleflight keyed on the refresh token;
   `context.WithoutCancel` + 30s timeout so a client disconnect cannot abort a
   refresh mid-flight; a per-refresh-token 429 block map (clamped 5s–5m) so a
   throttled refresh endpoint is not re-hit; preserve the old refresh token when
   the response omits one. pxy has flock + mutex + staleness re-read
   (`providers/mod.rs:26-46`) — the disconnect-abort and 429-block cases are the
   real additions. Refresh lead is 4h there vs pxy's 5min.
4. **Per-credential HTTP transport pools** (`helps/transport_cache.go`) — one
   connection pool per OAuth identity, "matching the native client's
   one-credential process model". Relevant once multi-account Claude exists.
5. **Content-negotiated `/v1/models`** (`server_routes.go:568-608`):
   `Anthropic-Version` header or `claude-cli` UA → Anthropic-shaped catalog.
   pxy always answers OpenAI-shaped (`server.rs:540-605`); it works only because
   CC's discovery schema is loose. Cheap correctness.
6. **Codex stream-bootstrap buffering** (`codex_executor_stream.go:136-256`) —
   a capacity rejection smuggled inside an HTTP 200 stream becomes a transparent
   failover. pxy's pre-first-event commit is the same idea; theirs additionally
   pattern-matches the bootstrap failure body.

Deliberately **not** taking: uTLS/JA3 spoofing, `cch=` body signing, MCP
tool-name cloaking, zero-width obfuscation, synthetic device identities, the
plugin FFI ABI, WebSocket/Realtime surfaces, cluster mode, the management REST
API + TUI. All are either docs/09 §11 non-goals or scale features for a
multi-tenant deployment pxy explicitly is not.

---

## 5. Prioritized queue

**Tier 0 — "native passthrough" (the whole ask).** One coherent change: when
client dialect == upstream dialect **and** the candidate is the single explicit
one, pxy should be a conduit, not a translator.

1. Single-candidate walks pass upstream errors through raw at **every** status
   (§2.1). Generalize the existing `multi && status == 404` carve-out.
2. Add a header map to `Outcome`; forward upstream response headers minus the
   litellm exclusion set, plus CLIProxyAPI's gateway-telemetry prefix strip
   (§2.2).
3. Skip `anthropic_sanitize` + `drop_params` when the client is Claude Code and
   the upstream is claude-oauth; gate on a native-client detector, not on wire
   format (§2.4).
4. Forward the client's `x-app`, `x-stainless-*`, `x-client-request-id`,
   `anthropic-version`, and real `user-agent` on the claude-oauth path (§2.5).
5. Forward `count_tokens` to `format = "anthropic"` upstreams, with the reduced
   beta profile and the `metadata`/`context_management`/`diagnostics` strip
   (§2.3).
6. Drop the local context gate on single-candidate requests (§2.6); guard
   `is_object()` (§2.7).
7. Parse `anthropic-ratelimit-unified-*` into cooldowns with scope rules and
   reset fuzz (§4.2).

Tier 0 is entirely subtraction plus plumbing — no new protocol, no
impersonation, and every item is independently testable against a mock upstream
using the existing integration-test harness.

Compliance rider on Tier 0 (see §0.1): item 3 should *also* stop the `claude`
provider serving non-Claude-Code clients, and item 4 must forward only headers
the client actually sent — never synthesize them.

**Tier 1 — reach the ChatGPT subscription.** Policy-cleared (§0.1: every
binding OpenAI document is silent, and the product lead explicitly blessed OSS
clients on Sign-in-with-ChatGPT), but resting on a revocable X post rather than
contract. Blocked on Saiful actually having/logging in a ChatGPT subscription —
there is no `~/.codex/auth.json` on this machine today.
8. **Prefer driving `codex app-server` with `chatgptAuthTokens`** over a raw
   `codex-oauth` provider kind (§0.1). It is OpenAI's documented surface for a
   host app that owns the user's ChatGPT auth lifecycle, so it stays inside a
   shipped harness instead of emitting sub2api-shaped traffic. A hand-rolled
   `chatgpt.com/backend-api/codex` client (the CLIProxyAPI recipe in §1) is the
   fallback only if app-server can't be driven headlessly.
   Never set `CODEX_INTERNAL_ORIGINATOR_OVERRIDE` — spoofing a first-party
   originator is exactly the line §0.1 says not to cross.
9. `WireFormat::Responses` so codex↔ChatGPT is passthrough rather than triple
   translation (§3). Only worth it with #8, and only on the raw-client fallback
   (app-server owns the protocol itself).

**Tier 2 — hardening.** Refresh singleflight + `WithoutCancel` + 429 block map
(§4.3); per-credential transport pools (§4.4); content-negotiated `/v1/models`
(§4.5); request-scoped error rules for the Claude fast-mode case (§4.1).

**Verification gate for "is it native yet?"** — the honest test is a diff, not a
vibe: run the same Claude Code prompt natively and through pxy with request
logging on both sides, and compare (a) the request bodies byte-for-byte after
key-sort normalization, (b) the header sets, (c) `cache_read_input_tokens` on
turn 2+ of a long session. (c) is the number that decides whether pxy costs more
quota than native.
