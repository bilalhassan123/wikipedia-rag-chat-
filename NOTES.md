# Notes

The honest retrospective the brief asks for. Not a victory lap.

## What I'd change with another two days

In rough priority order — what I'd actually pick up, not a wish list.

1. **Cross-encoder reranker on the chat path.** Retrieval is currently single-stage cosine similarity with `nomic-embed-text` and top-k=5. On topics with many near-duplicate sections (sports careers, dynasty histories, list articles), the top-5 can be five paraphrases of the same chunk. A small cross-encoder reranker (`bge-reranker-base` or `ms-marco-MiniLM-L-6-v2`) over the top-20 would meaningfully improve precision. I left it out because it's a second model and a second network hop on the chat path — fine for v2.

2. **Streaming responses.** The chat box disables the input while the LLM is thinking. Streaming via Server-Sent Events would let tokens appear as they're generated, which makes a CPU-bound `phi3:mini` feel much less sluggish on the first call. Ollama's streaming API is one parameter flip; the FastAPI side needs a small refactor to a generator response. Out of scope for the brief but the next thing I'd ship.

3. **Sharper error UI on the frontend.** Every error currently renders identically as a red box with the backend's message. The user can't tell whether to retry (`WikipediaUnreachableError`), pick a different URL (`DisambiguationError`), or wait (`LLMUnavailableError`). Distinct styling and a per-class "what to do" line would close that loop.

4. **Frontend test breadth.** I tested the chat box's state model thoroughly (multi-turn, evidence rendering, error path) but skipped the URL form and the article→chat handoff. The brief is explicit that backend tests carry more weight, and I followed that, but two more frontend tests would be cheap.

5. **Smarter ingestion concurrency.** Ollama's embeddings endpoint doesn't accept batches today, so the embedding adapter is a sequential loop — ~30 seconds on CPU for a 30-chunk article. Issuing parallel embed calls via `asyncio.gather` with a small semaphore would compress this. I prioritised correctness over throughput at this scope.

6. **Citation deep-links from each evidence chunk to the article anchor.** Requires tracking section context through the chunker, which I deliberately didn't do for scope reasons. The obvious UX win.

7. **Soften the thin-article reject.** The 200-character threshold is a hard reject. With more time I'd surface a warning ("this article only has X paragraphs of content; expect short answers") and let the user proceed if they want.

8. **ReAct-style agent on the chat path.** The current chat is a one-shot retrieve-then-generate. A ReAct loop where the model can issue multiple retrievals (e.g. "I need more context on his early career, retrieve again") would help on multi-hop questions. I left it out for two reasons: `phi3:mini` is not strong at structured tool-use, and the multi-turn loop multiplies CPU latency by the number of steps. Worth doing on a stronger model.

9. **Smarter ingestion progress for the map-reduce summary.** The summariser now does map-reduce over up to 4 segments + 1 combiner call, but the route emits a single "Generating summary" event rather than streaming per-segment progress. With more time I'd thread a progress callback through `summarise_article` so the UI shows "Summarising part 2 of 4" while it's working.

## What the AI got wrong that I had to correct

The brief asks for this; I'd rather be specific than flattering.

- **Documentary noise.** First drafts of REQUIREMENTS.md and DESIGN.md had headers like *"Status: DRAFT — drafted with Claude as a pair"* and footers crediting authorship inline. I told it to strip this; the brief asks for AI-vs-human transparency in TASKS.md and NOTES.md, not as a header on every deliverable. The agent over-indexed on visible attribution rather than treating each doc as a candidate's submission.

- **Naming the hiring company in deliverables.** Early drafts referenced the company by name in REQUIREMENTS.md. I told it to stop; the project should stand on its own merits, not as a memo addressed to a specific reader. One round-trip to remove.

- **Python version drift.** The agent pinned `requires-python = ">=3.12"` because `python --version` reported 3.12 on the host — without checking the conda env I actually use, which is 3.11.9. One round-trip to fix; small but the kind of assumption I expect to push back on.

- **Initial model choice — `phi3:mini` over `qwen2.5:3b`.** The first cut of DESIGN.md picked `phi3:mini` for its instruction-following reputation. In practice `phi3:mini`'s 4 K-token window forced aggressive summary truncation — most Wikipedia articles overflow it 5×. We migrated to `qwen2.5:3b` (32 K context) once this surfaced during real testing. The agent should have caught the context-window math during DESIGN; I caught it during execution. NOTES is the honest record of that.

- **`git init` reflex.** The agent kept reaching for `git init` and a commit on every task. I told it to stop touching git; I'm managing the commit history myself so the rhythm and messages reflect a human pacing rather than 25 atomic agent-flush commits. The agent respected this once told but the impulse was clearly trained in.

- **`venv` over conda.** First attempt at running pytest tried to spin up a `.venv` inside `backend/`. I have a conda env named `ai` already configured and redirected. Reasonable default for a generic project but wrong for my setup.

- **`ScraperError` base name.** When defining the typed exception hierarchy in T05, the agent named the base class `ScraperError`. That became misleading once T06 added LLM and vector-store exceptions inheriting from it. Renamed to `AppError` in T06. Minor but the kind of thing that compounds if not fixed early — I caught it at T06 rather than letting it ship.

- **Working-directory drift between Bash invocations.** Pytest occasionally ran from the repo root rather than `backend/`, silently dropping the `asyncio_mode = "auto"` config and the testpaths and failing every async test. Took two confused iterations to realise the cwd was the issue. Fix: always `cd backend` before pytest. A `conftest.py` at the project root that adjusts the rootdir, or running tests via `python -m pytest` from a known dir, would be more robust against this.

## Where I made unilateral calls in REQUIREMENTS

These are calls in [REQUIREMENTS.md §7](REQUIREMENTS.md) that I closed without bouncing them off anyone. Listed for the reviewer who wants to know which way I'd flex:

- **English Wikipedia only.** Multi-language multiplies the scrape, prompt, and tokenisation surface for no demo value. Listed as a v2 extension.
- **Disambiguation → reject.** Indexing a list of links produces meaningless RAG; rejecting is simplest.
- **Thin-article threshold = 200 chars.** Judgment call. Stub articles below this are useless to chat with.
- **One article at a time, replace on new URL.** Brief explicitly forbids history; this makes the wipe-on-replace behaviour explicit rather than letting old vectors linger.
- **Backend returns chunk evidence in the chat response.** Borderline scope expansion — I added it because grounding has to be verifiable for FR-14 to mean anything. The frontend renders it minimally (a disclosure list).
- **Refusal mechanism is prompt-driven, not similarity-threshold-gated.** Cosine thresholds for `nomic-embed-text` vary by topic and would need empirical tuning. The model itself refusing per the system prompt is more robust and tests as a behavioural property.
- **No streaming for v1.** Default to single fully-formed responses. See "with another two days" above.

## On the rubric-critical refusal contract

The FR-14 behaviour ("model says it cannot find the answer in the article when retrieval doesn't support the question") is pinned in three layers:

1. **Prompt level** — `CHAT_SYSTEM_PROMPT` in `app/core/rag.py` forbids prior knowledge and instructs the model to reply with the verbatim phrase `REFUSAL_PHRASE`.
2. **Unit test** — `tests/unit/test_rag.py` asserts the prompt carries the refusal contract and that the pipeline surfaces the model's refusal verbatim when fed unrelated chunks (with a `_FakeLLM` that returns the phrase).
3. **Integration test** — `tests/integration/test_e2e.py` (opt-in via `pytest -m integration`, needs the live stack) asks an off-article question and asserts the answer contains "cannot find" and "article".

`phi3:mini` occasionally paraphrases the refusal rather than emitting it verbatim. The integration test allows for that with a substring match; the unit test pins the verbatim phrase. With more time and a larger model the verbatim contract would be tighter.
