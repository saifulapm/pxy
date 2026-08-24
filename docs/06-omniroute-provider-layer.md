# OmniRoute provider layer — research (2026-08-24)

Source: exploration of `references/OmniRoute`. Paths relative to that repo.

## Headline

Two registries; ~72% of providers are pure data. The dashboard catalog (349 entries) is UI-only —
ignore it. The real wire config is 267 `RegistryEntry` records
(`open-sse/config/providers/registry/<id>/index.ts`), of which ~192 are just
"base URL + Bearer + OpenAI chat JSON" through one generic executor. Complexity concentrates in:
format translators (~250 KB TS), ~75 dedicated executors, OAuth token lifecycle.

## RegistryEntry shape worth porting (open-sse/config/providers/shared.ts:114)

- Identity: `id`, `alias`
- Protocol: `format` (openai | openai-responses | claude | gemini | antigravity | kiro | ...),
  `executor`
- Endpoints: **`baseUrl` is the COMPLETE endpoint URL** (kills path-joining bugs), `baseUrls[]`
  fallback chain, `messagesUrl`, `modelsUrl`
- Auth: `authType`, `authHeader` (bearer | x-api-key | ...), `authPrefix`, `oauth {...}`
- Models: `models[]` with per-model capabilities (toolCalling, reasoning, vision, contextLength,
  maxOutputTokens, `targetFormat`, `strip[]`, `unsupportedParams[]`)
- Behavior: `timeoutMs`, `forceStream`, `requestDefaults`, `unsupportedParams[]`,
  **`alternateFormats[]`** — one provider, second protocol on another URL/auth (DeepSeek: primary
  openai-responses at `/responses`, alternate claude at `/anthropic/v1/messages`). Cheap, add early.

Distribution (267 entries): authType apikey 182 / oauth 22 / optional 10 / none 9 / cookie 1.
format openai 200 / claude 15 / openai-responses 6 / gemini 4. Executor "default" for ~192.

## Translation architecture

Hub-and-spoke with OpenAI chat completions as pivot; direct paths added where the pivot loses
fidelity (gemini↔claude thinking blocks). Dispatch = two maps keyed `"<from>:<to>"`. Streaming
translation is stateful per stream (tool-call map, block indices, thinking flags).
Biggest files: openai-responses response (55KB), openai-to-kiro (43KB), openai-to-gemini (36KB),
openai-to-claude (34KB). Target format resolution: per-model targetFormat → connection override →
registry format.

## OAuth providers — difficulty ranking for a Rust port

Shared infra needed once: device-code/auth-code flows, token refresh w/ per-provider lead time,
per-connection refresh mutex, embedded public client creds, lazy per-request `needsRefresh`.

| Provider | Difficulty | Flow | Upstream | Key quirks |
|---|---|---|---|---|
| kilocode | **easy** | custom device flow, no client_id | OpenAI chat at api.kilo.ai/api/openrouter/chat/completions | NO refresh (long-lived bearer); `X-KILOCODE-EDITORNAME` header; anonymous free tier |
| github (copilot) | **medium** | RFC 8628 device code | api.githubcopilot.com `/chat/completions`, `/responses`, native `/v1/messages`; `/models` | second-stage mint: `GET api.github.com/copilot_internal/v2/token` with `Authorization: token <gh>` → short-lived (~30min) token; fixed copilot header profile; `X-Initiator` billing (agent turns free); ~10 small body mutations |
| kimi-coding | **medium** | RFC 8628 at auth.kimi.com | dual: Anthropic `api.kimi.com/coding/v1/messages?beta=true` (x-api-key) or OpenAI `/v1/chat/completions` (Bearer) | refresh tokens ROTATE (persist immediately!); persistent device id file + 6 `X-Msh-*` headers on every call (anti-bot) |
| antigravity / agy | **hard** (medium once Gemini translator exists) | Google auth-code, NO PKCE, no openid scope, embedded client id+secret, localhost:8080 callback | `cloudcode-pa.googleapis.com/v1internal:streamGenerateContent?alt=sse` — Gemini JSON in Cloud Code envelope | `:loadCodeAssist` project bootstrap + `:onboardUser` loop (missing project = 422); tiny systemInstruction (system → first user msg); strip thinking/reasoning_effort; sessionId = negative decimal; header scrub + pinned darwin/arm64 UA; Google RTs don't rotate; agy = same client, different profile |
| kiro | **hard** | AWS SSO-OIDC device code (JSON bodies, fresh OIDC client per login) OR kiro social device flow; refresh tokens one-time-use | CodeWhisperer `POST /generateAssistantResponse`, `X-Amz-Target` header, **no SigV4** (plain bearer) | needs: conversationState builder (alternating turns, no system role, uuidv5 conversationId for AWS prompt cache) + AWS `vnd.amazon.eventstream` binary frame parser (few hundred lines, well-specified); no token counts → estimate from contextUsagePercentage |
| amazon-q | **free once kiro done** | same object as kiro internally | same | different connection namespace only |
| zai-web | **skip** | none — pasted localStorage JWT, no refresh | signed-HTTP path needs CAPTCHA proof + HMAC sig + scraped X-FE-Version + ~35 fingerprint params; default transport is Playwright driving chat.z.ai | tool calling disabled; alternatives: `zai`/`glm` API-key providers reach same GLM models |

Web-cookie is a 35-provider category with a headless-browser subsystem — excluding it removes
Playwright from a Rust port entirely.

## API-key providers (our 24 connections)

19/24 are plain OpenAI-compatible Bearer endpoints. Non-obvious ones:

- agentrouter: PRIMARY is Anthropic messages (`agentrouter.org/v1/messages`, x-api-key);
  alternates chat/completions + responses. Misreports quota exhaustion as 403/400.
- deepseek: primary **openai-responses** at `/responses`; alternate claude at
  `/anthropic/v1/messages`.
- cloudflare-ai: needs account id in URL; content arrays flattened to text.
- fireworks: `modelIdPrefix: accounts/fireworks/models/`.
- opencode-go/zen: `opencode.ai/zen/go/v1` / `zen/v1`; per-model targetFormat (x-api-key for
  claude-targeted models).
- openrouter: static HTTP-Referer + X-Title headers; passthroughModels (per-model 404 ≠ cooldown).
- zenmux-free: actually web-cookie tier (Anthropic body + `ctoken` from cookie) — skip with
  web-cookie category.
- Non-chat services: brave-search (GET, X-Subscription-Token), elevenlabs (TTS, xi-api-key,
  voice in path), firecrawl (search/scrape), jina-reader (GET r.jina.ai/{url}), voyage-ai
  (embeddings + rerank, own shapes: top_k, {data:[]}).
- mimocode: not in OmniRoute snapshot anymore (and we have no credentials) — drop.

## Endpoint surface (client-facing)

- OpenAI: `/v1/chat/completions`, `/v1/completions`, `/v1/responses`, `/v1/models`
- Anthropic: `/v1/messages`, `/v1/messages/count_tokens`
- Gemini: `/v1beta/models/{m}:generateContent` / `:streamGenerateContent?alt=sse` (streaming from
  URL suffix)
- Non-chat: `/v1/embeddings`, `/v1/images/generations`, `/v1/images/edits`,
  `/v1/audio/transcriptions`, `/v1/audio/speech`, `/v1/videos/generations`, `/v1/search`,
  `/v1/web/fetch`, `/v1/rerank`, `/v1/moderations`, ...

**Non-chat is architecturally separate in OmniRoute**: per-modality flat registries + own
handlers that fetch directly — bypassing executor/breaker/translator; only credentials shared.
Deferring modalities costs nothing architecturally.

## Suggested build order (from the research)

1. Core: RegistryEntry-shaped config + default executor (base-URL-verbatim + Bearer + OpenAI
   passthrough + SSE relay) + `alternateFormats` → ~all API-key providers.
2. Translators: OpenAI↔Anthropic (client-facing /v1/messages is mandatory for Claude Code),
   then OpenAI↔Gemini. Biggest real work.
3. OAuth in order: kilocode → github → kimi-coding → antigravity/agy → kiro/amazon-q.
4. Non-chat modalities as simple separate handlers.
5. Drop: web-cookie category (incl. zai-web, zenmux-free), mimocode.
