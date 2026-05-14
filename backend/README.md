# Backend

FastAPI service for the Wikipedia RAG chat app. The full architecture lives in [DESIGN.md](../DESIGN.md).

## Local development

Requires Python 3.11+.

```bash
# with conda
conda activate ai          # or any env with Python 3.11+
pip install -e ".[dev]"
pytest

# with venv
python -m venv .venv
. .venv/Scripts/activate    # macOS/Linux: source .venv/bin/activate
pip install -e ".[dev]"
pytest
```

`pytest` runs the unit suite with coverage. The integration test that hits the live compose stack is opt-in:

```bash
pytest -m integration
```

## Layout

```
src/app/
  api/         # FastAPI routes (thin)
  core/        # Domain logic: scraper, chunker, summariser, RAG pipeline
  adapters/    # Concrete LLM / embedding / vector-store clients
  main.py      # ASGI entrypoint  (coverage-excluded)
  schemas.py   # Pydantic request/response models  (coverage-excluded)
  deps.py      # Dependency-injection providers  (coverage-excluded)
tests/
  unit/        # Mock LLM + vector store
  integration/ # @pytest.mark.integration — needs docker compose
```
