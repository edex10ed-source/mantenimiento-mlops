FROM python:3.13-slim

COPY --from=ghcr.io/astral-sh/uv:0.12.19 /uv /uvx /bin/

WORKDIR /app

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never

COPY pyproject.toml uv.lock ./
RUN uv sync --locked --no-dev --no-install-project


COPY . .

RUN uv sync --locked --no-dev

ENV PATH="/app/.venv/bin:$PATH"

RUN uv run preparar-datos && uv run entrenar

RUN useradd --create-home appuser
USER appuser

EXPOSE 8000

CMD ["sh", "-c", "exec fastapi run src/mantenimiento_mlops/api.py --port ${PORT:-8000}"]