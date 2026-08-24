# Provider inventory (from OmniRoute, 2026-08-24)

Source of truth: `~/.omniroute/storage.sqlite` → `provider_connections` (34 connections, 31 providers).
All credentials were decrypted (AES-256-GCM, key in `~/.omniroute/.env`) and saved to **pass** under
`AI/<provider>/<connection-name>`:

- API-key providers: first line of the pass entry = the API key.
- OAuth providers: entry body is a **pure JSON object** (`pass show AI/kiro/main | jq` works)
  with `access_token`, `refresh_token`, `expires_at`, and where relevant `project_id`,
  `provider_specific_data`. These tokens rotate — pxy must refresh and persist its own token
  cache; pass holds the bootstrap refresh token (snapshot from 2026-08-24).

## API-key providers (24 connections)

| pass entry | provider | notes |
|---|---|---|
| AI/agentrouter/main | agentrouter | aggregator |
| AI/agnes/main | agnes | |
| AI/alibaba/china | alibaba (dashscope CN) | |
| AI/alibaba/global | alibaba (dashscope intl) | |
| AI/bai/main | bai | 74 models in omniroute |
| AI/brave-search/main | brave-search | web search |
| AI/cloudflare-ai/main | cloudflare-ai | workers AI |
| AI/codestral/main | codestral | mistral code endpoint |
| AI/deepseek/main | deepseek | |
| AI/elevenlabs/main | elevenlabs | TTS/STT |
| AI/firecrawl/main | firecrawl | scraping/search |
| AI/fireworks/main | fireworks | |
| AI/freemodel-dev/main | freemodel-dev | |
| AI/jina-reader/main | jina-reader | reader/embeddings |
| AI/mistral/main | mistral | |
| AI/openadapter/main | openadapter | aggregator, 101 models |
| AI/opencode-go/github | opencode-go | |
| AI/opencode-go/google | opencode-go | |
| AI/opencode-zen/main | opencode-zen | 96 models |
| AI/openrouter/main | openrouter | 960 models |
| AI/tokenrouter/main | tokenrouter | aggregator, 172 models |
| AI/v0-vercel/main | v0-vercel | |
| AI/voyage-ai/main | voyage-ai | embeddings/rerank |
| AI/zai-web/main | zai-web | 3273-char JWT-ish token — likely web-derived; candidate to DROP (no web-cookie providers in pxy) |
| AI/zenmux/main | zenmux | 198 models |
| AI/zenmux-free/main | zenmux-free | 2325-char token |

## OAuth providers (8 connections)

| pass entry | provider | refresh token? | expiry seen |
|---|---|---|---|
| AI/agy/saifulapm@gmail.com | agy | yes | short-lived AT |
| AI/amazon-q/main | amazon-q | yes | short-lived AT |
| AI/antigravity/saifulapm@gmail.com | antigravity | yes | short-lived AT |
| AI/github/main | github (copilot) | no (long-lived gho_ token) | — |
| AI/kilocode/saiful.apm@gmail.com | kilocode | no (long-lived) | — |
| AI/kimi-coding/main | kimi-coding | yes | short-lived AT |
| AI/kiro/main | kiro | yes | short-lived AT |

## Skipped

- `mimocode` — connection exists but has no credentials stored.

## OmniRoute runtime facts

- Running instance: `http://localhost:20128/v1`, dashboard same port.
- `/v1/models` currently lists **3045 models** (includes per-provider aliases like `gh/`=`github/`,
  `zm/`=`zenmux/`, plus `auto/*` combo models and `no-think/*` variants).
- OmniRoute stores everything in sqlite; encryption format `enc:v1:<iv>:<ct>:<tag>`,
  key derivation `scrypt(STORAGE_ENCRYPTION_KEY, "omniroute-field-encryption-v1", 32)`.
