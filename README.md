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

## Replay a skill on a batch and measure it

```bash
uv run python -m mimic.run --skill enter_invoice                 # 50 invoices, original UI
uv run python -m mimic.run --skill enter_invoice --layout v2     # redesigned UI
uv run python -m mimic.run --skill enter_invoice --show --limit 5
```
A LangGraph loop (locate → act → verify → retry) executes each step with fallback locator
strategies (selector → label → name → text). Success is graded against the app's own records,
and every step is logged to `step_logs` for the failure predictor.

## You stay in control

Mimic is safe by default: before each task it shows its plan and waits for you.

```
Task 3/50  — Bureau Express
  Vendor             <- Bureau Express
  Total amount       <- 249.99
  Category           <- Software
  then click   'Save invoice'
[A]pprove  [E]dit  [S]kip  [T] approve all  [Q]uit >
```

- `--confirm-submit --show`: fills the form, then asks again before pressing submit
- `--yes`: no review (for trusted, already-validated skills)

## Self-healing

When every recorded locator fails, the healer tries, cheapest first:
1. **memory** — a fix found earlier for this step (pgvector + in-run cache)
2. **semantic** — compares the demo element with every candidate by meaning (embeddings)
3. **vision** — Qwen2.5-VL sees the demo screenshot (element boxed in red) and the current
   screen; *vision proposes, DOM verifies* before acting

A heal is only remembered after the action is verified (a submit must navigate), so a wrong
guess never poisons the next tasks.

```bash
uv run python -m mimic.run --skill enter_invoice --layout v2 --no-heal   # baseline
uv run python -m mimic.run --skill enter_invoice --layout v2             # with healing
```

## Failure risk model (XGBoost)

The web is flaky: cookie banners cover buttons, buttons stay disabled while a page loads,
servers fail. Before every action Mimic probes the element (covered? enabled? on screen?
how was it found, with what confidence?) and an XGBoost model predicts the probability that
the action fails. Fixing the page first (dismiss overlays, wait for it to settle) costs time,
so the agent only pays for it when the predicted risk is high.

```bash
uv run python -m mimic.riskml collect --runs 6     # chaos runs in the sandbox, telemetry in Postgres
uv run python -m mimic.riskml train                # XGBoost vs logistic regression vs rules
uv run python -m mimic.riskml evaluate --tasks 50  # same seeded chaos under 3 policies
uv run python -m mimic.run --skill enter_invoice --chaos 0.4 --risk-policy predicted
```

- Train/test split **by run**, so no step of a test run is seen in training
- Decision threshold chosen on training data with **F2** (a missed failure costs a task;
  an unnecessary mitigation costs ~0.4 s)
- Compared against logistic regression and two hand-written rules
- Evaluation replays **identical seeded perturbations** for every policy (fair A/B test)

## Sandbox: LedgerLite
A fake accounts-payable app used as the training and evaluation target.
- `/?layout=v1` — original UI
- `/?layout=v2` — redesigned UI (fields reordered, button moved, ids renamed) to test self-healing
- `/api/entries` — ground truth to score the agent automatically
- `?chaos=0.4&seed=1` — reproducible flakiness: cookie banner, slow buttons, relabels, HTTP 500

## Roadmap
See [docs/PLAN.md](docs/PLAN.md).
