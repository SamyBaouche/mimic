# Mimic

**Show it once. It does it forever.**

Mimic is a local computer-use agent that learns enterprise data-entry workflows from a single
screen demonstration, then replays them autonomously — and self-heals when the UI changes.

## Architecture

```
Demonstration (screen + clicks)
        │
  [Recorder] ──► [Planner LLM (Ollama)] ──► Skill (steps) ──► pgvector
                                                   │
Task batch ──► [LangGraph agent]  observe → plan → act → verify → repair
                    │         │                          │
            Vision (Ollama +  │                  XGBoost failure predictor
            fine-tuned HF     │                          │
            UI detector)   Playwright                step_logs (telemetry)
                    │
             FastAPI + WebSocket ──► live dashboard
```

## Stack
Python · LangGraph · Ollama · HuggingFace · XGBoost · pgvector · FastAPI · Playwright

## Quickstart (macOS, Apple Silicon)

```bash
brew install uv ollama
ollama serve &                      # native Ollama uses the Mac GPU (Metal)
ollama pull qwen2.5vl:7b
ollama pull qwen2.5:7b

cp .env.example .env
make install                        # deps + Playwright Chromium
make up                             # Postgres + pgvector (Docker)
make db                             # create tables
make invoices                       # 50 synthetic invoices

make sandbox                        # LedgerLite on http://localhost:8001
make api                            # Mimic API on http://localhost:8000/health
make test
```

## Record a demonstration

```bash
uv run python -m mimic.recorder --name enter_invoice --url http://localhost:8001
```
A browser opens: do the task once, then press Enter in the terminal.
Each step (field, value, label, position, screenshot) is saved to
`recordings/enter_invoice/recording.json`.

## Learn a skill from the demonstration

```bash
ollama pull nomic-embed-text   # once, for skill search
uv run python -m mimic.learn learn recordings/enter_invoice/recording.json
uv run python -m mimic.learn search "record supplier bills"
```
The planner turns typed values into parameters (`Vendor <- {vendor}`), writes a description,
and stores the skill with its embedding in pgvector. If Ollama is unavailable, a deterministic
draft is kept instead.

## Sandbox: LedgerLite
A fake accounts-payable app used as the training and evaluation target.
- `/?layout=v1` — original UI
- `/?layout=v2` — redesigned UI (fields reordered, button moved, ids renamed) to test self-healing
- `/api/entries` — ground truth to score the agent automatically

## Roadmap
See [docs/PLAN.md](docs/PLAN.md).
