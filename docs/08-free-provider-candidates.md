# Free-tier providers worth adding — research 2026-08-24

Mined from the OmniRoute registry (`references/OmniRoute/docs/reference/FREE_TIERS.md` is the
best local artifact — it has a per-provider ToS audit table) plus web verification against
official docs. litellm contributed nothing useful here (its zero-cost rows are local Ollama +
Gemini duplicates).

Each of these needs **you to sign up and put a key in pass** — none can be added unattended.
Config blocks are staged (commented) in `config.example.toml` and `~/.config/pxy/config.toml`.

## ⚠️ CORRECTIONS (2026-08-25) — this doc has been wrong twice; re-verify before acting

- **§6 Vercel AI Gateway: WRONG.** The $5/month grant requires the Vercel **Pro plan
  ($20/mo)** — a Hobby account with a valid key 403s ("customer_verification_required")
  on every call, even `-free` models. Rejected.
- **§8 Scaleway: WRONG.** Signup **required a card**, and the billing API shows **zero
  discounts** on the fresh account — no "1M free tokens" tier exists; every generative-API
  call bills the card. Disabled the same day it was added.
- **§3 Google AI Studio: WRONG about needing a Gemini translator.** Google's OpenAI
  compatibility layer (`generativelanguage.googleapis.com/v1beta/openai/chat/completions`,
  Bearer auth) handles chat + tool calling fine — added as a plain provider 2026-08-25,
  zero Rust work. (A native Gemini translator would still be needed only for
  Gemini-exclusive features like cached content or the Live API.)
- **§5 Groq: CORRECT** — live `x-ratelimit-*` headers matched this doc exactly
  (8K TPM on gpt-oss/qwen, 70K TPM + 250 RPD on compound). Also: compound REJECTS
  external tool definitions (built-in tools only).
- Lesson: for any remaining candidate (Ant Ling, Morph, NVIDIA), verify the free
  grant **from the account itself** (billing/credits endpoint or a $0-balance test call)
  before wiring it into pxy — don't trust this doc's quota claims.

## Ranked adds

### 1. Z.AI — GLM free models  ★ highest value
- `https://api.z.ai/api/paas/v4/chat/completions`, Bearer, OpenAI-compatible.
- **Permanently free models** (not trial credits): `glm-4.7-flash` (30B-A3B MoE,
  **59.2 SWE-bench Verified**, tuned for agentic coding / long-horizon planning / tool use),
  `glm-4.5-flash`, `glm-4.6v-flash` (vision).
- Signup: email, no card. Key: z.ai console.
- **Gotchas**: free tier allows **1 concurrent request** → set `rpm` low, never fan out.
  Endpoint split matters: general keys use `/api/paas/v4`, Coding-Plan keys use
  `/api/coding/paas/v4` (mixing them is the #1 auth error). Use `api.z.ai`, not
  `open.bigmodel.cn` (China).
- ToS rated **ok** for personal proxy use.

### 2. Inception Labs — largest no-card grant still alive
- `https://api.inceptionlabs.ai/v1/chat/completions`, Bearer, OpenAI-compatible.
- **100M free tokens on signup** (officially raised from 10M; some of their pages still say 10M).
- Models: `mercury-2`, `mercury-coder` — diffusion-based, >1,000 tok/s.
- Signup: platform.inceptionlabs.ai, no card. Flip the training opt-out in Account Settings.
- One-time grant, not recurring.

### 3. Google AI Studio (Gemini) — best free tool calling
- `https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent`
- Auth header is **`x-goog-api-key`**, and the wire format is **Gemini-native, not OpenAI** —
  pxy needs a Gemini translator first (not yet built; see HANDOFF next steps).
- Free (recurring, no card): `gemini-3-flash`, `gemini-3.1-flash-lite`, `gemini-2.5-flash(-lite)`
  — all tool calling + vision, 1M context.
- Gotchas: Pro models left the free tier 2026-04-01. Google no longer publishes per-model free
  limits (visible only in AI Studio after sign-in) — don't hardcode quotas. Limits are
  per-project. Free-tier prompts are used for training. RPD resets midnight Pacific.

### 4. NVIDIA NIM — best model variety
- `https://integrate.api.nvidia.com/v1/chat/completions`, Bearer.
- ~40 RPM free (upgradeable to ~200 on request), no card.
- 100+ models: GLM 5.2, Kimi K2.6, Qwen3.5-397B, `mistralai/devstral-2-123b-instruct-2512`,
  Nemotron 3 Ultra 550B, Mistral Large 3 675B.
- Gotchas: ToS restricts to prototyping/dev/research (not production). `openai/gpt-oss-*` there
  is flagged `toolCalling: false`. Treat as passthrough (multiplexes 9 vendors) so one stale
  model 404 doesn't cool the whole provider.

### 5. Groq — fast, but 8K TPM is the real ceiling
- `https://api.groq.com/openai/v1/chat/completions`, Bearer.
- Free tier (from the official rate-limit table): `openai/gpt-oss-120b`, `gpt-oss-20b`,
  `qwen/qwen3.6-27b` → 30 RPM / 1K RPD / **8K TPM** / 200K TPD; `groq/compound(-mini)`
  (agentic, built-in web search + code execution) → 30 RPM / 250 RPD / 70K TPM.
- **OmniRoute's registry model list is STALE** — Llama 3.3 70B / Llama 4 Scout / Qwen3-32B are
  no longer free. Limits are org-wide (extra keys don't multiply). Adding a card (still free)
  unlocks 10× Developer tier.

### 6. Vercel AI Gateway — recurring credit on frontier models
- `https://ai-gateway.vercel.sh/v1/chat/completions`, Bearer.
- **$5/month recurring**, 0% markup, spendable on Claude/GPT/Gemini.
- ⚠️ **One-way door**: buying any credits permanently ends the monthly free grant. Never top up.

### 7. Ant Ling (InclusionAI) — 500K tokens/day recurring
- `https://api.ant-ling.com/v1/chat/completions`, Bearer.
- 500K free tokens/day (input+output), resets 02:00 UTC+8, no rollover; plus a 100M signup promo.
- Models: Ling-3.0-flash, Ling/Ring-2.6-1T (strong long-context programming), Ming omni.
- **Open risk**: docs don't say whether signup needs a Chinese phone/ID; top-ups go via Alipay.
  Not geo-blocked, English docs exist — worth an attempt.

### 8. Scaleway — cleanest ToS of the lot
- `https://api.scaleway.ai/v1/chat/completions`, Bearer. EU/Paris (latency from BD is the cost).
- 1M free tokens one-time + 60 min audio transcription, ~60 RPM, no card under the limit.
- Models: qwen3-235b-a22b, gpt-oss-120b, deepseek-v3, mistral-small-3.2.

### 9. Morph — specialty: fast-apply edit merging
- `https://api.morphllm.com/v1/chat/completions`, Bearer. 250K credits (~$2.50) / ~200 req/mo.
- `morph-v3-fast|large|auto` — applies diffs at ~10,500 tok/s, 98% accuracy, 262K ctx.
- Small quota but high value: lets a weak free model sketch a diff and Morph apply it cleanly.

### 10. Reka — rare recurring credit
- **$10/month recurring** (+$20 signup), no billing info. Not a coding leader; the draw is the
  recurring credit and their web-research API.

### 11. Cerebras — ONLY for short non-agentic calls
- `https://api.cerebras.ai/v1/chat/completions`, Bearer. Free: `gpt-oss-120b`, `gemma-4-31b`.
- **Disqualified for agents**: free-tier context is capped at **8,192 tokens** and the rate is
  **5 RPM** (not the 30 RPM third-party guides quote). An agent's system prompt + tool defs
  alone blow 8K. Use for commit messages / classification / quick rewrites at ~3,000 tok/s.
- The "$5 signup credit" requires a verified payment method — not card-free.

### 12. GitHub Models — distinct product from Copilot
- `https://models.github.ai/inference/chat/completions`, Bearer with a PAT (`models:read`).
- 45+ models; rate tier scales with your Copilot subscription.
- **Hard 8K input / 4K output per request on ALL tiers** → unusable for agentic coding; fine as
  a classification/summarization sidecar.

## Cheap extra slots
- **Nous Research Portal** — 50 RPM / 500K TPM, rotating Hermes catalog (their 248-model list is
  OpenRouter underneath; only Hermes runs on their backend).
- **Pollinations** — `https://gen.pollinations.ai/v1/chat/completions`. **Anonymous access is
  gone** (verified 2026-08-24: returns 401 "A valid API key is required"); free key at
  enter.pollinations.ai.
- **Cohere** — `https://api.cohere.com/compatibility/v1/chat/completions` (use the compat layer,
  not native `/v2/chat`). 1,000 calls/month. **ToS forbids production/commercial use.**
- **ModelScope** — 2,000 calls/day across 900+ models, but Alibaba real-name verification
  usually demands a Chinese ID/phone, and it's China-hosted.

Small gateways advertising huge quotas (Bazaarlink 4M/day, Nara 5M/day, AnyAPI, Navy, AINative)
have unknown provenance — lottery tickets, don't build on them.

## Confirmed DEAD (don't waste time)
Chutes (free tier retired 2026-03-15) · Targon (never free) · Together AI ($25 credit retired,
$5 min prepay) · OVHcloud (trial needs payment method; anonymous tier 2 RPM/IP) · Hugging Face
($0.10/mo) · Hyperbolic / Nebius / DeepInfra / nScale (one-time $1–5 trials) · Predibase (gone) ·
MonsterAPI (shut 2026-06-30) · Phind (shut 2026-01) · Featherless / AIMLAPI / Yi / StepFun
(ended or paused) · **SambaNova** (alive but 20 requests per DAY — useless for agents).

## ToS note
Groq (§6.3), Cerebras, SambaNova (§1.5(c)), Fireworks, Nebius (§5f) all have clauses prohibiting
reselling/sublicensing/proxying API access. A single-user self-hosted proxy is a grey area, not
a green light. Cleanest for exactly this use case: **Scaleway, Z.AI, Morph**.

## Suggested signup order for Saiful
1. **Z.AI** — permanent free agentic coding model, clean ToS, 2-minute signup.
2. **Inception Labs** — 100M tokens, no card.
3. **NVIDIA NIM** — biggest model variety, 40 RPM.
4. **Vercel AI Gateway** — $5/mo recurring on frontier models (never top up).
5. **Google AI Studio** — needs a Gemini translator in pxy first.
