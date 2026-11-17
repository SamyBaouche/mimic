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

## Sandbox: LedgerLite
A fake accounts-payable app used as the training and evaluation target.
- `/?layout=v1` — original UI
- `/?layout=v2` — redesigned UI (fields reordered, button moved, ids renamed) to test self-healing
- `/api/entries` — ground truth to score the agent automatically

## Roadmap
See [docs/PLAN.md](docs/PLAN.md).
