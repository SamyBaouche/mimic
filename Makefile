.PHONY: install up down db api sandbox invoices record test lint

NAME ?= enter_invoice

record:
	uv run python -m mimic.recorder --name $(NAME) --url http://localhost:8001

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
