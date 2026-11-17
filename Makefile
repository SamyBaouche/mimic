.PHONY: install up down db api sandbox invoices test lint

install:
	uv sync
	uv sync --extra agent
	uv run playwright install chromium

up:
	docker compose up -d

down:
	docker compose down

db:
	uv run python -m mimic.db

api:
	uv run uvicorn mimic.api.main:app --reload --port 8000

sandbox:
	uv run uvicorn sandbox.app:app --reload --port 8001

invoices:
	uv run python -m sandbox.generate_invoices

test:
	uv run pytest -q

lint:
	uv run ruff check . && uv run ruff format --check .
