# Backend image. Single-stage; the slim base + uvicorn[standard] keeps
# the final image around 250 MB.

FROM python:3.11-slim

WORKDIR /app

# Copy project metadata and source, then install in one layer.
# Hatchling resolves the package from src/ via pyproject's
# [tool.hatch.build.targets.wheel] config.
COPY pyproject.toml ./
COPY src ./src

RUN pip install --no-cache-dir .

EXPOSE 8000

# The /health route is a lightweight liveness probe (DESIGN §11).
HEALTHCHECK --interval=10s --timeout=5s --start-period=20s --retries=5 \
    CMD python -c "import httpx; httpx.get('http://localhost:8000/health').raise_for_status()"

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
