# Tasks

Execution log. Status moves `pending` → `in_progress` → `done` as commits land. The **Authorship** column records honestly what was AI-drafted vs human-written for each task — filled in when the task closes.

## Conventions

- One task = one commit. Commit messages reference the task ID (e.g. `T05: scraper + Wikipedia REST fixtures`).
- Tests land alongside the feature, never at the end.
- A task is `done` only when its tests pass and coverage on the touched module is meaningful (not padding).
- `Authorship` values:
  - `AI` — the AI agent drafted the substantive content; I reviewed and shipped with at most light edits.
  - `Mixed` — substantive back-and-forth, both contributed material edits.
  - `Human` — I wrote it; AI assist was minor or absent.
  - `Pending` — task not started yet, authorship TBD.
- Tasks are sized to roughly fit one focused agent run (15–45 minutes of focused work).

## Phase 0 — Bootstrap

| ID | Task | Status | Authorship | Notes |
| --- | --- | --- | --- | --- |
| T01 | Repo bootstrap: `.gitignore`, `.env.example`, README skeleton, license. Initial commit. | done | AI | Plain text only at this stage. |
| T02 | Backend project scaffold: `backend/` with `pyproject.toml`, `src/` layout, `pytest`, `pytest-cov`, `ruff`, `httpx`, `pydantic`, `fastapi`. Coverage config with the three exclusions from DESIGN.md §12. Hello-world test that runs green. | done | AI | Python pinned to ≥3.11. Verified `pytest` runs green in conda env `ai` (1 passed, 100% coverage). |

## Phase 1 — Backend core (test-first)

| ID | Task | Status | Authorship | Notes |
| --- | --- | --- | --- | --- |
| T03 | Core interfaces and dataclasses: `LLMClient`, `EmbeddingClient`, `VectorStore` `Protocol`s; `Chunk`, `RetrievedChunk` frozen dataclasses. Smoke tests on the dataclasses. | done | AI | Async signatures (DESIGN §5 updated to match). `@runtime_checkable` so adapters can be sanity-checked structurally. 12 passed, 100% coverage. |
| T04 | Chunker: recursive character splitter (paragraph → sentence → word → char), defaults `chunk_size=800`, `overlap=120`. Unit tests covering happy path, empty string, single-paragraph, boundary-exact, overlap correctness. | done | AI | Pure function. 21 tests, 100% line coverage. |
| T05 | Scraper: Wikipedia REST API fetch (`/api/rest_v1/page/html/{title}`), Parsoid HTML cleaner (drop infobox, references, navboxes, captions, footnotes), disambiguation detection, thin-article check (<200 chars). Tests against committed HTML fixtures: canonical article, redirect, disambiguation, stub. | done | AI | 28 tests via `httpx.MockTransport`; created `core/errors.py` with the 5 scraper-related exceptions (T06 adds LLM + vector ones + handlers). 100% coverage. |
| T06 | Typed exception hierarchy (`InvalidUrlError`, `ArticleNotFoundError`, `WikipediaUnreachableError`, `DisambiguationError`, `ThinArticleError`, `LLMUnavailableError`, `VectorStoreUnavailableError`) plus FastAPI exception handlers mapping each to the HTTP status from DESIGN.md §10. Tests verify mapping. | done | AI | Renamed the base class to `AppError` (was `ScraperError` from T05). Single table-driven handler. 9 tests. |
| T07 | Ollama LLM adapter (`OllamaLLMClient`): wraps `/api/generate`. Tests with `httpx.MockTransport`. | done | AI | 12 tests covering payload shape, max_tokens default, base-url normalisation, and every failure path mapping to `LLMUnavailableError`. |
| T08 | Ollama embedding adapter (`OllamaEmbeddingClient`): wraps `/api/embeddings`. Tests with `httpx.MockTransport`. | done | AI | One HTTP call per text (Ollama's endpoint isn't batchable). 10 tests. Empty-input short-circuits with no HTTP call. |
| T09 | Qdrant vector store adapter (`QdrantVectorStore`): `reset`, `upsert`, `search` against `qdrant-client`. Tests with the client patched at the module boundary. | done | AI | 14 tests via `unittest.mock.AsyncMock`. Converts our 16-hex-char chunk ids to uint64 Qdrant point ids; original id round-trips through point payload. Added `qdrant-client>=1.11` dep. |
| T10 | Summariser use case: composes `LLMClient.generate` with the summary prompt from DESIGN.md §7. Tests with a `FakeLLMClient` verifying prompt content and output passthrough. | done | AI | 9 tests; prompt invariants pinned (grounding constraint, no-invention, 5–8 sentence length). |
| T11 | RAG pipeline use case: embed → search(k=5) → render prompt → generate. Tests with fake Embedding/VectorStore/LLM. **Includes the no-evidence refusal test (FR-14): when fakes return chunks unrelated to the question, the pipeline returns `"I cannot find that in the article."`** | done | AI | 13 tests. The `REFUSAL_PHRASE` constant pins the verbatim contract; the system prompt embeds it; a unit test pins the pipeline's pass-through. Live-stack verification deferred to T17. |
| T12 | FastAPI app: `POST /article`, `POST /chat`, DI wiring in `deps.py`, app instantiation in `main.py`. Route tests via `httpx.AsyncClient` with `dependency_overrides` injecting fakes. | done | AI | 12 route tests. Lifespan creates one shared `httpx.AsyncClient` for both Ollama adapters and a separate one for Wikipedia. CORS allow-all (frontend uses same-origin in compose; permissive for dev). `/health` for the container probe. |

## Phase 2 — Containerisation

The brief recommends getting the happy path working on bare metal before containerising. Phase 2 follows that order: stand up the data services in compose, run the backend on the host against them, confirm the end-to-end flow, and only then containerise the backend.

| ID | Task | Status | Authorship | Notes |
| --- | --- | --- | --- | --- |
| T13 | Compose stack v1: `ollama` and `qdrant` services with named volumes and healthchecks. Ollama entrypoint script that pulls the chat and embedding models on first start before `ollama serve`. | done | AI | `ollama/entrypoint.sh` runs serve in background, waits for the API, pulls both models (idempotent), then `wait`s on serve. Healthcheck verifies both models are listed. Defaults: `qwen2.5:3b` + `nomic-embed-text`. |
| T14 | Bare-metal end-to-end smoke: backend on host, services from T13's compose. Paste a real Wikipedia URL via the API, confirm summary returns, ask one in-article and one out-of-article question, confirm grounding and refusal behaviour. No code change task — verification gate before T15. | manual | Human | User-driven verification: `docker compose up -d ollama qdrant` then run backend on host. Marked manual; outcome should be recorded in NOTES.md if anything blocked. |
| T15 | Backend Dockerfile: multi-stage (build → slim runtime), `uvicorn` entrypoint, healthcheck. | done | AI | Single-stage `python:3.11-slim` build (deliberate trade-off: simpler than multi-stage, ~250 MB final). `pip install .` from pyproject. Healthcheck calls `/health` via httpx. |
| T16 | Compose stack v2: wire `backend` into compose with `depends_on: condition: service_healthy` for both ollama and qdrant. Re-run the smoke from T14 against the fully containerised stack. | done | AI | Backend service in same compose file as T13 (one file, not two). `depends_on` with `condition: service_healthy` gates startup. Env vars passed through with sensible defaults. |
| T17 | Integration test: `@pytest.mark.integration` test that hits the live compose stack, indexes a small fixture Wikipedia article, asks one in-article and one out-of-article question, asserts the refusal phrase appears for the latter. | done | AI | Skipped by default via `addopts = -m "not integration"`. Substring match for the refusal phrase (allowing the model to paraphrase slightly while still proving grounding). |

## Phase 3 — Frontend

| ID | Task | Status | Authorship | Notes |
| --- | --- | --- | --- | --- |
| T18 | Frontend Vite scaffold + page layout: URL input, submit button, summary panel, chat box, error region. Plain CSS. Wired against `/api/article` and `/api/chat`. | done | AI | React 18 + Vite 5, `App.jsx` + `ChatBox.jsx`, plain CSS, no Tailwind, no component library. Multi-turn chat with evidence disclosure per assistant turn. |
| T19 | Frontend tests with Vitest: chat-box state (multi-turn append, optimistic input clear), error rendering, evidence indicator. | done | AI | 8 Vitest tests using `@testing-library/react` and `@testing-library/user-event`: rendering, send-disabled, multi-turn append, input-clear after send, evidence rendering, API error path, whitespace guard. |
| T20 | Frontend Dockerfile (Node build → nginx serve) with `/api/*` reverse-proxy config. Wire `frontend` service into compose so `docker compose up` exposes the app at `http://localhost:8080`. | done | AI | Multi-stage: `node:20-alpine` build → `nginx:alpine` serve. `nginx.conf` with SPA fallback and `/api/*` proxy to `backend:8000` with 600s read timeout for LLM cold starts. Compose service on `:8080` with healthcheck via `wget --spider`. |

## Phase 4 — Deliverables

| ID | Task | Status | Authorship | Notes |
| --- | --- | --- | --- | --- |
| T21 | Coverage report generation: run `pytest --cov` against the backend, commit the HTML report under `coverage/` (or a screenshot if HTML is unwieldy). Verify ≥85% line coverage on application code with the declared exclusions. | done | AI | HTML report generated at `coverage/`. **165 unit tests, 398 statements, 100% line coverage on application code** (above the 85% bar). The integration test is skipped from the default run. |
| T22 | README.md: prerequisites (Docker + Compose v2, ~16 GB RAM, ~3 GB disk for models), single-command startup, the localhost URL, known caveats (first-run model download, cold-start latency). | done | AI | Replaced T01 skeleton. Sections: prerequisites, one-command run, stack table, usage, refusal-behaviour callout, caveats, dev workflow for both backend and frontend, testing & coverage, files of interest. |
| T23 | NOTES.md: honest retrospective. What I would change with another two days (reranker, citation deep-links, streaming, multi-article). What the AI got wrong during this build that I had to correct. | done | Mixed | AI drafted; I curated the AI-correction list against what actually happened (DRAFT/Claude headers, company name, Python pin, git reflex, venv vs conda, ScraperError rename, cwd drift). |
| T24 | Screen recording or screenshots showing the end-to-end flow: paste URL, see summary, ask in-article question (answered), ask out-of-article question (refused). Committed under `docs/`. | manual | Human | User-driven: requires a live stack and a screen capture. Commit to `docs/screenshots/` or as a Loom link in README. |
| T25 | Final smoke: clean machine simulation — `git clone` into a fresh directory, `docker compose up`, walk through the README. Catch any "works on my machine" gaps. Push to a public Git repository (GitHub / GitLab / Bitbucket) per the submission format. | manual | Human | User-driven: clean-clone smoke and `git push` to a public repo. |

## AI vs human contribution — running summary

Filled in as tasks close. Format: `T## — one-sentence honest description of who did what.`

| Task | Authorship summary |
| --- | --- |
| Planning: REQUIREMENTS.md | Mixed — AI drafted the structure and prose; I directed the unilateral closures (English-only, disambig-reject, thin-article threshold, evidence-on-API), tightened scope, and signed off the wording. |
| Planning: DESIGN.md | Mixed — AI drafted the architecture, the ASCII diagram, and the trade-off framing; I locked the stack, the RAG numbers (chunk 800 / overlap 120 / top_k 5), and the prompt structure. Final wording mine. |
| Planning: TASKS.md | Mixed — AI proposed the task decomposition; I adjust granularity and order as we execute. |
| T01 — bootstrap | AI — generated standard `.gitignore`, `.env.example`, README skeleton, MIT LICENSE; nothing project-specific beyond the env defaults that match DESIGN.md §11. |
| T02 — backend scaffold | AI — pyproject with hatchling build, src layout, pytest+cov+asyncio, ruff config, the three coverage exclusions tagged 1:1 to the brief's permitted categories. Smoke test verifies `import app` works. Adjusted `requires-python` from 3.12 to 3.11 to match the local conda env. |
| T03 — interfaces | AI — three `@runtime_checkable` Protocols (`LLMClient`, `EmbeddingClient`, `VectorStore`) with async methods and two frozen `slots=True` dataclasses (`Chunk`, `RetrievedChunk`). Tests cover field access, frozen behaviour, equality, and protocol structural conformance via fakes. Updated DESIGN §5 to match (async signatures + `@runtime_checkable` + design note 3 explaining why async). |
| T04 — chunker | AI — recursive splitter with sliding-window backward search for paragraph/sentence/word boundaries in the second half of the window. Stable 16-hex-char sha256 ids derived from `(article_title, chunk_index)`. 21 tests, 100% coverage. |
| T05 — scraper | AI — Wikipedia REST API fetch via `httpx.AsyncClient`, Parsoid HTML cleaner via `bs4.BeautifulSoup` + `html.parser`, redirect-aware title resolution. 28 tests with `httpx.MockTransport` and 3 committed HTML fixtures (canonical, disambiguation, stub). Added `beautifulsoup4` to runtime deps. Created `core/errors.py` with 5 scraper exceptions; T06 will extend it. |
| T06 — error handlers | AI — table-driven handler maps each typed error to its HTTP status from DESIGN §10. Renamed base from `ScraperError` to `AppError` for clarity. Added `LLMUnavailableError` and `VectorStoreUnavailableError` for the adapter layer. |
| T07 — Ollama LLM adapter | AI — async POST against `/api/generate` with `stream: false` and `options.num_predict`. 12 tests verify payload shape, response stripping, and that every failure mode (5xx, 4xx, network error, invalid JSON) raises `LLMUnavailableError`. |
| T08 — Ollama embedding adapter | AI — sequential POSTs against `/api/embeddings` (Ollama doesn't accept batches today). 10 tests verify per-text payloads, empty-input short-circuit, float coercion, and mapped failures. |
| T09 — Qdrant adapter | AI — async client with `recreate_collection` for `reset`, `PointStruct` upsert, `search` with cosine. 14 tests via `unittest.mock.AsyncMock`. Convert chunk-id hex to uint64 for Qdrant point ids; round-trip through payload. |
| T10 — summariser | AI — prompt verbatim from DESIGN §7, single LLM call, 9 tests pinning prompt invariants (grounding clause, no-invention clause, 5–8 sentence target). |
| T11 — RAG pipeline | AI — embed → search(k=5) → numbered-excerpt prompt → generate. `REFUSAL_PHRASE = "I cannot find that in the article."` is pinned in three places (constant, system prompt, unit test). 13 tests including the FR-14 refusal contract. |
| T12 — FastAPI app + routes | AI — thin route handlers, three Pydantic schemas (excluded from coverage per DESIGN §12), DI providers in `deps.py` (excluded), `main.py` lifespan owns the shared `httpx.AsyncClient` and `AsyncQdrantClient`. 12 route tests via `dependency_overrides`. |
| T13/T15/T16 — compose + Dockerfile | AI — single `docker-compose.yml` with all four services (ollama, qdrant, backend, frontend); ollama entrypoint script pulls models on first start; backend Dockerfile is single-stage `python:3.11-slim`; healthcheck-gated startup order. |
| T17 — integration test | AI — `@pytest.mark.integration`, skipped by default. Indexes a real Wikipedia article via the live stack, asks an in-article and an off-article question, asserts the refusal substring on the latter. |
| T18 — frontend scaffold | AI — Vite + React 18, `App.jsx` for URL form + summary, `ChatBox.jsx` for multi-turn chat with evidence disclosure. Plain CSS, no Tailwind, no component library. |
| T19 — frontend tests | AI — 8 Vitest + Testing Library tests for the chat box state model (multi-turn append, input clear, evidence rendering, error path, whitespace guard). |
| T20 — frontend container | AI — multi-stage Dockerfile (Node 20 → nginx alpine), `nginx.conf` with SPA fallback and `/api/*` reverse-proxy to `backend:8000`. Compose service on `:8080`. |
| T21 — coverage report | AI — generated HTML at `coverage/index.html`. **100% line coverage on application code, 398 statements, 165 unit tests.** |
| T22 — README | AI — full run instructions, prerequisites, caveats (first-run pull time, CPU cold start), dev workflow for backend and frontend, refusal-behaviour callout. |
| T23 — NOTES retrospective | Mixed — AI drafted; I curated the "AI got wrong" section to reflect what actually happened in this build, not a generic list. |
