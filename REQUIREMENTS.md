# Requirements

## 1. Context

This is a small, containerised RAG-based chat application over a single Wikipedia article, built with a local LLM and an AI-agent-driven development workflow. The product is intentionally narrow — depth on a small surface area is the point. Planning artefacts (this file plus DESIGN.md and TASKS.md) are first-class deliverables alongside the code; the AI-assisted workflow itself is part of what is being demonstrated.

This document is my restatement of the project brief in concrete, testable terms. Where the brief is ambiguous I close the question explicitly and explain the reasoning. Section 7 captures every unilateral call I made.

## 2. Functional Requirements

### 2.1 Article ingestion

- **FR-1.** The user pastes a Wikipedia article URL into a single text input on the home page and submits it.
- **FR-2.** The backend accepts canonical (`en.wikipedia.org/wiki/...`) and mobile (`en.m.wikipedia.org/wiki/...`) URLs. Other Wikipedia language subdomains are out of scope (§5). Non-Wikipedia URLs are rejected at the API boundary, before any LLM or vector calls.
- **FR-3.** The backend extracts article body text — paragraphs and section headings — and discards navigation chrome, infoboxes, the references list, image captions, and citation footnotes inside paragraphs. The goal is signal density at chunk time, not source fidelity.
- **FR-4.** Wikipedia redirects are followed transparently.
- **FR-5.** Disambiguation pages are detected and rejected with a friendly message asking the user to pick a specific article. Indexing a list of links produces a useless chat experience.
- **FR-6.** Articles whose extracted body text falls below a small threshold (~200 characters) are rejected as "too thin to chat about". The brief flags empty articles as a sensible-error-handling concern.

### 2.2 Summary

- **FR-7.** After scraping, the backend generates a concise summary using the local LLM (one inference call) and returns it to the frontend. Target length: 5–8 sentences (~150–250 words).
- **FR-8.** The summary is grounded in the scraped article text, not the model's prior knowledge of the topic. The system prompt instructs the model to summarise the supplied text and nothing else.

### 2.3 Indexing

- **FR-9.** After scraping, the article body is chunked, embedded, and upserted into the vector store. Concrete chunk size, overlap, and embedding model live in [DESIGN.md](DESIGN.md) — they are deliberate engineering choices, not contract terms.
- **FR-10.** The system holds one indexed article at a time. When the user submits a new URL, the previous article's vectors are cleared (or a fresh collection is used) before the new one is indexed.

### 2.4 Chat (RAG)

- **FR-11.** Below the summary, a chat box lets the user ask free-text questions about the indexed article.
- **FR-12.** For each question, the backend embeds the query, retrieves the top-k most similar chunks, and supplies them as context to the local LLM along with the question. The LLM produces an answer grounded **only** in the retrieved chunks. Concrete top-k and prompt template are pinned in DESIGN.md.
- **FR-13.** Multi-turn conversation within a session is supported — the user can ask follow-ups about the same article without re-pasting the URL. No persistence across page reloads.
- **FR-14. Refusal-on-no-evidence (rubric-critical).** If the retrieved chunks do not contain information that answers the question, the model must say it cannot find the answer in the article rather than fabricate from prior knowledge. This is the whole point of grounded RAG.
- **FR-15.** The chat API response carries enough information for grounding to be verifiable — at minimum the chunk text or chunk IDs that informed the answer. Whether the frontend renders citations visibly is a UX call settled in DESIGN.md.

### 2.5 Errors and edge cases

- **FR-16.** Bad / malformed URLs are rejected at the API boundary with HTTP 400 and a human-readable message; the frontend surfaces it inline.
- **FR-17.** Wikipedia 404s, network failures, and rate-limit responses are caught and surfaced with a message that distinguishes "article not found" from "Wikipedia unreachable".
- **FR-18.** Ollama cold-start latency is expected on the first call after stack startup. The backend does not begin serving until the LLM runtime is reachable (healthcheck-gated).
- **FR-19.** Qdrant transient unavailability during stack startup is handled the same way — the backend waits on a vector-store healthcheck before serving.

## 3. Non-Functional Requirements

- **NFR-1. Local-only inference (hard requirement).** The running application makes zero hosted-LLM and zero hosted-embedding calls. Both summarisation and chat run against an Ollama-served model. Embeddings are produced by a local Ollama-served embedding model. Nothing in the runtime stack reaches OpenAI / Anthropic / Gemini / build.nvidia.com / any other hosted endpoint.
- **NFR-2. Single-command startup (hard requirement).** A clean machine with Docker and Docker Compose v2 installed must be able to clone the repo, run a single command, and reach a working app at a documented localhost URL. Models are pulled by the Ollama service on first start.
- **NFR-3. Containerised stack.** Backend, frontend, vector store, and LLM runtime each run as a service in `docker-compose.yml`. Healthchecks gate startup order so the backend does not begin serving until both the vector store and the LLM runtime are reachable.
- **NFR-4. Test coverage (hard requirement).** Backend application code achieves ≥85% line coverage with meaningful tests, not coverage padding. Coverage exclusions (entrypoint, framework glue, pure data classes) are declared explicitly in `pyproject.toml` / `.coveragerc` and called out in DESIGN.md. A coverage report is committed.
- **NFR-5. LLM and vector store are swappable.** Business logic depends on small Python interfaces, not on Ollama or Qdrant directly. Swapping in vLLM or Chroma requires writing a new adapter, not rewriting the pipeline. Concrete adapters are wired in at app startup via dependency injection; route handlers stay thin.
- **NFR-6. No secrets in the repo.** A `.env.example` documents required configuration; no real `.env` is committed. There are no API keys today (everything is local), but this stays true going forward.
- **NFR-7. Modest hardware target.** The stack must run on a developer laptop with 16 GB RAM and a CPU-only inference path. Model selection (`qwen2.5:3b` ~2.0 GB quantised, `nomic-embed-text` ~270 MB) is calibrated to that. First inference will be slow due to model load; subsequent calls must be acceptable for an interactive demo. The contract is "interactive enough to demo", not a specific p95.
- **NFR-8. Honest authorship.** Every commit and every planning artefact records what was AI-drafted vs human-written. NOTES.md captures things the AI got wrong that I corrected. The brief explicitly flags lying about authorship as the only disqualifying behaviour.
- **NFR-9. Documentation.** README.md explains prerequisites, the single startup command, the URL to open, and known caveats. DESIGN.md justifies every non-obvious choice with the alternative considered.

## 4. In Scope

- Single-page web app: URL input → summary → chat box.
- One indexed article at a time, replaced when a new URL is submitted.
- Multi-turn chat against the current article; no cross-session persistence.
- English Wikipedia article URLs only (canonical and mobile subdomains).
- Containerised stack: backend, frontend, vector store, LLM runtime.
- Local LLM for both summarisation and chat. Local embedding model.
- Backend tests with ≥85% line coverage; at least one integration test that wires up the real services through compose.
- A small set of frontend tests for chat-box interaction logic.
- Three planning artefacts (REQUIREMENTS, DESIGN, TASKS), README, NOTES, screen recording or screenshots.

## 5. Out of Scope (Explicit)

- Authentication, accounts, user profiles, sessions across reloads.
- Article history, multi-article support, tabs, bookmarking.
- Analytics, telemetry, usage tracking.
- Non-English Wikipedia, non-Wikipedia sources, PDFs, file uploads.
- Streaming token rendering in the UI. Non-streaming is sufficient; we revisit only if it costs almost nothing to add.
- Citations as clickable deep links back to specific Wikipedia anchors. Surfacing *which* chunks grounded an answer is in scope; deep-linking is not.
- Production concerns: rate limiting, abuse handling, multi-tenant isolation, autoscaling, GPU support.
- Visual polish beyond "functional and unembarrassing" (per the brief's own FAQ).
- Automated answer-quality evaluation harness. Manual verification plus the no-evidence-refusal test is the bar.

## 6. Assumptions

- The reviewer runs `docker compose up` on a Mac, Linux, or WSL2 box with at least 16 GB RAM and Docker Desktop or equivalent. We do not target Windows-native Docker quirks.
- First-run model download happens over the reviewer's internet connection. This is acceptable; the brief's "single command" expectation accommodates a one-time pull on first start.
- The reviewer has working internet during evaluation only for (a) initial model pull and (b) Wikipedia article fetches at runtime. Hosted inference is still off-limits for the running app.
- Wikipedia's REST/HTML endpoints remain available for unauthenticated GETs at demo volume, with a polite User-Agent.
- A small top-k (target 4–6, pinned in DESIGN.md) is sufficient grounding for the chat model's context budget.

## 7. Open Questions I Closed Unilaterally

| Question | My answer | Why |
| --- | --- | --- |
| Which Wikipedia editions? | English only (`en.wikipedia.org`, `en.m.wikipedia.org`). | Brief says "Wikipedia article URL". Multi-language multiplies scrape, prompt, and tokenisation surface for no demo value. |
| Disambiguation pages? | Detect and reject. | Link lists, not article body. Indexing them produces meaningless RAG. |
| Multi-article state? | One at a time, new URL replaces old. | Brief explicitly forbids multi-article history. Avoids "which article is the user asking about?" routing. |
| Chat persistence across reloads? | No. Refresh = restart. | Adds storage and identity concerns the brief does not ask for. |
| Streaming tokens? | Not for v1. Single fully-formed response. | Brief does not require it. We may revisit if Ollama streaming is trivial to wire up. |
| What is "concise" for the summary? | 5–8 sentences, ~150–250 words. | Brief does not specify. Shorter loses substance; longer competes with the chat for attention. |
| Show evidence with chat answers? | Backend returns chunk evidence; frontend at minimum signals that grounding occurred. | Grounding is the entire premise of RAG. Hiding it makes "is the model fabricating?" unverifiable. |
| What when retrieval supports nothing? | Model says it cannot find this in the article. Tested explicitly. | Rubric-critical. Hallucination on retrieval miss is the most damaging failure mode. |
| Wikipedia scrape mechanism? | Pinned in DESIGN.md. Leaning toward the Wikipedia REST API (`/api/rest_v1/page/html/{title}`) because Parsoid HTML is much cleaner than rendered page HTML. | Engineering choice, not a requirement. |
| Coverage exclusions? | Entrypoint (`main.py`), pure Pydantic schemas, DI wiring. Declared in `pyproject.toml`. | Brief explicitly permits this. Must be called out, not hidden. |
| Hardware target? | 16 GB RAM laptop, CPU-only inference. | The brief explicitly accommodates a 3B-class model on a developer laptop. |

## 8. Acceptance Criteria

A submission satisfies these requirements if and only if:

1. `docker compose up` on a clean machine with Docker installed brings the full stack up. README documents prerequisites, command, and URL.
2. Pasting a valid English Wikipedia URL produces a summary and an active chat box within a reasonable time after first model pull.
3. Asking a question whose answer is in the article produces an answer derived from the retrieved chunks, with grounding visible in the response payload.
4. Asking a question whose answer is **not** in the article produces an "I cannot find this in the article" response — verified by an automated test.
5. Pasting a non-Wikipedia URL, a malformed URL, a disambiguation page, or a thin article produces the corresponding friendly error rather than a 500.
6. Backend test coverage is ≥85% line coverage on application code, with a coverage report committed. Tests are meaningful, not coverage padding.
7. Both the LLM client and the vector store client sit behind Python interfaces; at least one route handler depends on the interface, not a concrete adapter.
8. The repo contains REQUIREMENTS.md, DESIGN.md, TASKS.md, README.md, NOTES.md, `docker-compose.yml`, source, tests, coverage report, and a short screen recording or screenshots.
9. No hosted-LLM or hosted-embedding calls occur in the running application. Demonstrated by absence of API keys and by `docker-compose.yml` configuration.
10. No secrets are committed; `.env.example` documents required configuration.
