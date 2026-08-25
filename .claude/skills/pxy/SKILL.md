---
name: pxy
description: Free-quota web search, URL fetch, and media generation through the local pxy daemon. Use when a task needs web search results, a web page as markdown, audio transcribed, text spoken aloud, or an image or video generated.
---

# pxy — local search, fetch, and media

The pxy daemon (systemd unit `pxy.service`, http://127.0.0.1:4100) routes every
call to free-quota providers with failover and local billing caps. `pxy <verb>
--help` is the authoritative flag reference; this table is the map:

| verb | does | worth knowing |
|---|---|---|
| `pxy search "query" [-n N] [--json]` | web search (brave → jina → firecrawl) | titles + urls + snippets; `--provider` forces one |
| `pxy fetch <url>` | page as markdown (jina-reader → firecrawl) | the clean way to read a page from the shell |
| `pxy transcribe <file>` | speech-to-text (groq whisper default) | the file needs a real audio extension (`.mp3`, `.wav`, …) — groq infers format from the name |
| `pxy say "text" [-o out.mp3] [--voice nova]` | text-to-speech (cloudflare aura default) | premium voices: `-m elevenlabs/eleven_turbo_v2_5` (10k chars/month — spend on real audio, not tests) |
| `pxy image "prompt" [-o out.png]` | image gen (cloudflare flux default, fast) | higher quality: `-m agnes/agnes-image-2.1-flash` or `-m alibaba/qwen-image-3.0` |
| `pxy video "prompt" [-o out.mp4]` | video gen (agnes) | blocks 1–3 minutes while rendering; upstream allows ~2 submits/min |

For scripts, the same capabilities over HTTP: `POST /v1/search {query}`,
`POST/GET /v1/fetch {url}`, plus OpenAI-shaped `/v1/images/generations`,
`/v1/audio/transcriptions` (multipart), `/v1/audio/speech`, `/v1/rerank`
(Cohere shape), `/v1/videos/generations`.

Quotas are enforced locally — `pxy status` shows the `#media` / `search#` rows.
A 429 saying "cooling down" or "cap reached" means that pool's quota is spent:
switch model/provider (`-m`, `--provider`) or move on without it.
