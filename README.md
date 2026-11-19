<div align="center">

# Mimic

### Show it once. It does it forever.

**A local AI agent that learns a repetitive web task from a single demonstration,<br/>
repairs itself when the interface changes, and asks you before it acts.**

[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?logo=python&logoColor=white)](#quickstart)
[![LangGraph](https://img.shields.io/badge/Agent-LangGraph-1C3C3C)](#how-it-works)
[![Ollama](https://img.shields.io/badge/Models-100%25%20local-000000?logo=ollama&logoColor=white)](#quickstart)
[![pgvector](https://img.shields.io/badge/Memory-pgvector-4169E1?logo=postgresql&logoColor=white)](#how-it-works)
[![XGBoost](https://img.shields.io/badge/ML-XGBoost-EB6E00)](#predicting-failures)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

<!-- Record docs/demo.gif and it will appear here -->
<img src="docs/demo.gif" alt="Mimic demo: replays a task, survives a redesign" width="820"/>

[Results](#results) · [How it works](#how-it-works) · [Quickstart](#quickstart) · [Roadmap](#roadmap)

</div>

---

## What it does

<table>
<tr>
<td width="25%" align="center"><b>Learns from one demo</b><br/><sub>Do the task once. Mimic records every field, value and click.</sub></td>
<td width="25%" align="center"><b>Replays at scale</b><br/><sub>Runs the same task on 50, 500, 5 000 records.</sub></td>
<td width="25%" align="center"><b>Heals itself</b><br/><sub>Button renamed or moved? It finds it again and remembers.</sub></td>
<td width="25%" align="center"><b>Asks first</b><br/><sub>Shows its plan. Nothing happens without your approval.</sub></td>
</tr>
</table>

> **Everything runs on your machine.** Screens and form data never leave it.

## Results

<sub>LedgerLite sandbox · 50 invoices per run · local models on a consumer PC · graded against what the app really recorded</sub>

| | Without Mimic's safeguards | With Mimic |
|:--|:--:|:--:|
| Replay of a learned task | — | **100%** success (50/50) |
| After a UI redesign | **0%** | **100%** |
| Cost of a repair | — | **8.4 s** first time → **0.3 s** from memory |
| Flaky pages (cookie banners, slow buttons, errors) | 29 failed actions | **7** failed actions (**−76%**) |
| ⏱ Speed vs. always fixing the page | 3.1 s / task | **1.3 s** / task (**59% faster**) |

## How it works

```mermaid
flowchart LR
    A([Your demo]) --> B[Recorder]
    B --> C[Planner LLM<br/>values → parameters]
    C --> D[(Skill library<br/>pgvector)]
    E([Records]) --> F{You approve}
    D --> F
    F --> G[LangGraph agent<br/>locate → act → verify]
    G -. element lost .-> H[Self-healing]
    G -. before acting .-> I[Risk model]
```

| Step | What happens |
|:--|:--|
| **1. Record** | You do the task once; each field, value, label, position and screenshot is captured. |
| **2. Learn** | A local LLM turns typed values into parameters (`Vendor ← {vendor}`) and stores the skill in pgvector. |
| **3. Review** | You see exactly what will be typed and clicked: approve, edit, skip or approve all. |
| **4. Replay** | A LangGraph loop finds each element, acts, and checks the result really happened. |
| **5. Heal** | If an element is lost: reuse a past fix → match by meaning → ask a vision model. |
| **6. Predict** | Before risky actions, an XGBoost model decides if the page needs fixing first. |

### Self-healing, step by step

When the redesign renames **Save invoice** to **Record** and moves it next to two decoy buttons:

1. **Memory**: was this step already repaired on this page? Reuse the fix instantly.
2. **Meaning**: compare the old button with every candidate using embeddings.
3. **Vision**: Qwen2.5-VL sees the demo with the old button **boxed in red**, and the new
   screen with **numbered boxes**. It answers a number, never a guessed coordinate.

> **Vision proposes, the DOM verifies.** A repair is remembered only after the action worked:
> a submit must navigate, an error page counts as a failure. A wrong guess never spreads.

### Predicting failures

Real pages are flaky. Before each action Mimic checks the element (covered? disabled? on screen?
found how, with what confidence?) and an **XGBoost** model predicts the chance of failure.
Fixing the page costs time, so Mimic only does it when the risk is high: **9% of steps**.

<details>
<summary><b>How the model was evaluated (click to expand)</b></summary>
<br/>

- **No leakage:** train/test split by run, so steps of one run never appear on both sides.
- **Cost-aware threshold:** tuned on training data with F2. A missed failure costs a whole
  task; an unnecessary fix costs ~0.4 s.
- **Fair A/B test:** the three policies (never / always / predicted) replay the exact same
  seeded perturbations.
- **Honest baselines:** compared with logistic regression and hand-written rules. On today's
  data they are close (logistic regression is slightly ahead). XGBoost is kept because it
  learns feature interactions without hand-coding them; the comparison reruns on every training.

| Model (test set) | ROC AUC | PR AUC | F1 |
|:--|:--:|:--:|:--:|
| XGBoost | 0.98 | 0.84 | 0.72 |
| Logistic regression | 0.99 | 0.88 | 0.79 |
| Rule: click on covered/disabled | 0.91 | 0.68 | 0.82 |
| Always predict failure | 0.50 | 0.07 | 0.13 |

</details>

<details>
<summary><b>Design decisions (click to expand)</b></summary>
<br/>

- **Numbered boxes instead of coordinates.** A 7B vision model was unreliable at pixel
  coordinates but robust at picking a number, and a number always maps to a real element.
- **Forced choice + confidence gating.** On a submit button a wrong pick is caught by the
  navigation check, so a best guess is safe. On a text field it would go unnoticed, so the
  model must be at least 60% confident.
- **Give up after 3 failed repairs** on the same step, so a broken page does not keep
  paying for model calls.
- **Mapping by meaning.** The data says `total`, the skill says `total_amount`, the screen
  says `Total amount`: all three are understood, by you and by the agent.

</details>

## Quickstart

**You need:** Python 3.11+, [uv](https://docs.astral.sh/uv/), [Ollama](https://ollama.com), Docker Desktop.

```bash
# 1. Models (run locally, use your GPU / Apple Silicon)
ollama pull qwen2.5vl:7b && ollama pull qwen2.5:7b && ollama pull nomic-embed-text

# 2. Project
git clone https://github.com/SamyBaouche/mimic.git && cd mimic
cp .env.example .env                  # Windows: Copy-Item .env.example .env
uv sync --extra agent && uv run playwright install chromium

# 3. Database
docker compose up -d && uv run python -m mimic.db
uv run python -m sandbox.generate_invoices
```

In a second terminal, start the demo app:

```bash
uv run uvicorn sandbox.app:app --port 8001
```

Then **teach → learn → run**:

```bash
uv run python -m mimic.recorder --name enter_invoice --url http://localhost:8001   # do it once
uv run python -m mimic.learn learn recordings/enter_invoice/recording.json         # learn it
uv run python -m mimic.run --skill enter_invoice --show --limit 5                  # watch it work
uv run python -m mimic.run --skill enter_invoice --layout v2 --debug               # survive a redesign
```

<details>
<summary><b>All commands (click to expand)</b></summary>
<br/>

| Command | What it does |
|:--|:--|
| `mimic.recorder --name N --url U` | Record one demonstration |
| `mimic.learn learn PATH` | Turn a recording into a reusable skill |
| `mimic.learn search "record supplier bills"` | Find a skill by meaning |
| `mimic.run --skill N` | Replay on a batch, with plan review |
| `--yes` | Skip the review (trusted skills only) |
| `--confirm-submit --show` | Fill the form, then ask before pressing submit |
| `--layout v2` · `--no-heal` · `--debug` | Redesigned UI · baseline · show healing decisions |
| `--chaos 0.4 --risk-policy predicted` | Flaky pages, fix only the risky steps |
| `mimic.riskml collect` · `train` · `evaluate` | Telemetry → XGBoost → A/B test |

</details>

<details>
<summary><b>The sandbox: LedgerLite (click to expand)</b></summary>
<br/>

A small accounts-payable app used to train and grade the agent.

| URL | What you get |
|:--|:--|
| `/?layout=v1` | Original interface |
| `/?layout=v2` | Redesign: fields reordered, button renamed and moved, decoy buttons, new ids |
| `/?chaos=0.4&seed=1` | Reproducible flakiness: cookie banner, slow buttons, renamed labels, HTTP 500 |
| `/api/entries` | What was really recorded, used to grade the agent |

</details>

## Built with

| Layer | Tools |
|:--|:--|
| Agent | LangGraph · Playwright |
| Models (local) | Ollama · Qwen2.5 · Qwen2.5-VL · nomic-embed-text |
| Memory & data | PostgreSQL · pgvector |
| Machine learning | XGBoost · scikit-learn |
| API & quality | FastAPI · pytest (74 tests) · Ruff · GitHub Actions |

## Roadmap

- [x] Learn from one demonstration · replay · plan review
- [x] Self-healing with memory, embeddings and vision
- [x] XGBoost failure risk model with fair A/B evaluation
- [ ] Read inputs from documents (invoice PDF, business card photo)
- [ ] `pip install mimic-agent`, a `mimic` CLI, `mimic doctor`
- [ ] Checkboxes, radio buttons, keyboard, uploads, multi-page flows
- [ ] WorkSuite sandbox: mail, tickets, expenses, approvals
- [ ] Learn from a screen-recording video
- [ ] Browser extension to record in one click

## Limitations

Mimic works on **web apps** for now, learns from **live demonstrations** (not videos yet), and
was measured on a sandbox built to be hard in known ways. Real sites will bring new surprises:
that is exactly where [contributions](CONTRIBUTING.md) help most.

---

<div align="center">

**Found a task Mimic should handle?** [Open an issue](../../issues/new/choose) · [Contribute](CONTRIBUTING.md)

MIT © 2026 [Samy Baouche](https://www.linkedin.com/in/samybaouche)

</div>
