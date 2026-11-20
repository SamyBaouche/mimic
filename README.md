<div align="center">

# Mimic

### Show it once. It does it forever.

**A local AI agent that learns a repetitive web task from a single demonstration,<br/>
repairs itself when the interface changes, and asks you before it acts.**

[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?logo=python&logoColor=white)](#quickstart)
[![LangGraph](https://img.shields.io/badge/Agent-LangGraph-1C3C3C)](#how-it-works)
[![Ollama](https://img.shields.io/badge/Models-100%25%20local-000000?logo=ollama&logoColor=white)](#quickstart)
[![Memory](https://img.shields.io/badge/Memory-SQLite%20%7C%20pgvector-4169E1?logo=postgresql&logoColor=white)](#how-it-works)
[![XGBoost](https://img.shields.io/badge/ML-XGBoost-EB6E00)](#predicting-failures)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

<!-- Demo GIF goes here once recorded:
<img src="docs/demo.gif" alt="Mimic demo: replays a task, survives a redesign" width="820"/> -->

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

<sub>Local models on a consumer PC · every result checked against what the app really saved</sub>

| | Without Mimic's safeguards | With Mimic |
|:--|:--:|:--:|
| Replay of a learned task (sandbox, 50 invoices) | — | **100%** success (50/50) |
| **Real app it was not built for** (stock Django admin, 30 rows from Excel) | — | **30/30** saved correctly, **0.44 s** per row |
| After a UI redesign | **0%** | **100%** |
| Cost of a repair | — | **8.4 s** with vision → **7.7 ms** with the fine-tuned matcher |
| Reading invoice PDFs (3 layouts) | 76% of fields right | **96%** right, **100%** of errors flagged |
| Flaky pages (cookie banners, slow buttons, errors) | 29 failed actions | **7** failed actions (**−76%**) |
| Speed vs. always fixing the page | 3.1 s / task | **1.3 s** / task (**59% faster**) |

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
2. **Matcher**: a small Hugging Face cross-encoder (MiniLM, 22M parameters) fine-tuned on
   synthetic redesigns scores every candidate in milliseconds on a CPU.
3. **Meaning**: compare the old button with every candidate using embeddings.
4. **Vision**: only if the matcher is unsure, Qwen2.5-VL sees the demo with the old button **boxed in red**, and the new
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

### Reading documents

Point Mimic at a folder of invoices (PDF or photos) instead of a spreadsheet. The skill knows
which values it needs, so Qwen2.5-VL is asked for exactly those, with a confidence per field.
Values are normalised (`22/02/2026` → `2026-02-22`, `3 389,96 $` → `3389.96`, `software` →
`Software`), and anything uncertain is flagged in the plan for you to check.

```bash
mimic sandbox                                       # also writes 50 invoice PDFs, 3 layouts
mimic extract eval                                  # field-level accuracy vs ground truth
mimic run enter_invoice --docs sandbox/data/documents --limit 5
```

### Real apps, not just the sandbox

The sandbox was built to be hard in known ways; a real app is hard in unknown ones. So Mimic is
also tested on a **stock Django admin**, the kind of internal tool many companies run, with
markup Mimic was not designed around (labels without `for`, `<fieldset>` legends, buttons
written as `<input value="Save">`, dropdowns whose label differs from their value).

- **Every common action:** text, dates, numbers, dropdowns, radio buttons, checkboxes, file
  uploads, Enter in a search box, and flows over several pages.
- **Knows when it worked:** at the end of the demo Mimic remembers the page and the
  confirmation message ("The expense “Taxi” was added successfully."). On every replay it
  checks for the same kind of message, with the ids and values of that task.
- **Skips what the app skips:** signed in after the first task? The sign-in steps are skipped.
- **Your data as it is:** an Excel or CSV file with your own column names (`Date`, `Montant`);
  each row comes back with `done` / `failed` and the reason, and `--only-failed` retries the rest.

```bash
uv run python scripts/realapp_benchmark.py --rows 30   # starts the admin, records, replays, checks the DB
```

### Distilling vision into a small model

The 7B vision model repairs well but needs seconds and a GPU. The **element matcher** is a
MiniLM cross-encoder fine-tuned on thousands of synthetic redesigns ("Save invoice" vs
"Submit bill" vs decoys like "Export CSV"), built from banks of real-world synonyms.

- Every label used by the LedgerLite v2 redesign is **held out of training**, so the sandbox
  test measures generalisation, not memorisation.
- Compared with string similarity and with the same model before fine-tuning.
- If its best candidate is not clearly ahead, the vision model still decides.

```bash
uv sync --extra matcher
mimic matcher train     # downloads MiniLM once, fine-tunes, evaluates
```

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

**You need:** Python 3.11+ and [Ollama](https://ollama.com). No database to install.

```bash
pip install mimic-agent
playwright install chromium
ollama pull qwen2.5vl:7b && ollama pull qwen2.5:7b && ollama pull nomic-embed-text

mimic doctor        # checks the browser, the models and storage, and says how to fix what is missing
```

Try it on the demo app (a small accounts-payable tool) in a second terminal:

```bash
mimic sandbox       # http://localhost:8001, plus 50 demo invoices and PDFs in ./sandbox/data
```

Then **teach, learn, run**:

```bash
mimic record enter_invoice --url http://localhost:8001   # do the task once in the browser
mimic learn recordings/enter_invoice/recording.json     # turn it into a skill
mimic run enter_invoice --limit 5 --show                 # watch it repeat the task
mimic run enter_invoice --layout v2                      # survive a redesign
mimic serve                                              # or drive it from the live dashboard
```

**Passwords are never recorded.** If a demo types into a password-like field, the value stays in
the browser and the skill gets a secret parameter. Give it for the session:
`MIMIC_SECRET_PASSWORD=...` (PowerShell: `$env:MIMIC_SECRET_PASSWORD="..."`).

<details>
<summary><b>Install from source, optional extras (click to expand)</b></summary>
<br/>

```bash
git clone https://github.com/SamyBaouche/mimic.git && cd mimic
uv sync && uv run playwright install chromium
uv run mimic doctor
uv run pytest
```

| Extra | Adds |
|:--|:--|
| `mimic-agent[ml]` | XGBoost failure-risk model (`mimic risk ...`, `--risk-policy predicted`) |
| `mimic-agent[matcher]` | Fine-tuned Hugging Face element matcher (`mimic matcher train`) |
| `mimic-agent[postgres]` | Postgres + pgvector storage (`DATABASE_URL=postgresql://...`, `docker compose up -d`) |
| `mimic-agent[all]` | Everything above |

</details>

<details>
<summary><b>All commands (click to expand)</b></summary>
<br/>

| Command | What it does |
|:--|:--|
| `mimic doctor` | Check browser, local models, storage; print the fix for anything missing |
| `mimic sandbox [--port 8001]` | Start the demo app and write demo invoices and PDFs |
| `mimic record NAME --url URL` | Record one demonstration |
| `mimic learn PATH` | Turn a recording into a reusable skill |
| `mimic skills` · `show N` · `search "TEXT"` · `delete N` | Manage the skill library |
| `mimic run N --data FILE.xlsx` | Replay on every row of an Excel / CSV / JSON file, with plan review |
| `--only-failed` · `--results out.xlsx` | Retry only the rows not done · choose where results go |
| `--yes` | Skip the review (trusted skills only) |
| `--confirm-submit --show` | Fill the form, then ask before pressing submit |
| `--layout v2` · `--no-heal` · `--debug` | Redesigned UI · baseline · show healing decisions |
| `--chaos 0.4 --risk-policy predicted` | Flaky pages, fix only the risky steps |
| `--docs FOLDER` | Read the inputs from PDFs / photos instead of a data file |
| `mimic serve [--port 8765]` | Live dashboard: watch the agent, approve from the browser |
| `mimic read FILE` · `mimic extract eval` | Extract one document · measure extraction accuracy |
| `mimic risk collect` · `train` · `evaluate` | Telemetry, XGBoost, A/B test |
| `mimic matcher train` · `evaluate` | Fine-tune the Hugging Face element matcher |

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
| `/login` | Sign-in page (user `demo`, password `ledger-demo-2026`) to try secret fields |
| `/api/entries` | What was really recorded, used to grade the agent |

</details>

## Built with

| Layer | Tools |
|:--|:--|
| Agent | LangGraph · Playwright |
| Models (local) | Ollama · Qwen2.5 · Qwen2.5-VL · nomic-embed-text |
| Fine-tuning | Hugging Face Transformers · PyTorch · MiniLM cross-encoder |
| Memory & data | SQLite (default) · PostgreSQL + pgvector |
| Machine learning | XGBoost · scikit-learn |
| Dashboard & CLI | FastAPI · WebSocket · Chrome DevTools Protocol screencast · `mimic` CLI |
| Quality | pytest (149 tests) · Ruff · GitHub Actions |

## Roadmap

- [x] Learn from one demonstration · replay · plan review
- [x] Self-healing with memory, embeddings and vision
- [x] XGBoost failure risk model with fair A/B evaluation
- [x] Fine-tuned Hugging Face element matcher (vision only when unsure)
- [x] Read inputs from documents (PDF, photos) with per-field confidence
- [x] Live dashboard: streamed agent browser, approve / edit / skip / pause from the page
- [x] `pip install mimic-agent`, a `mimic` CLI, `mimic doctor`, SQLite by default
- [x] Passwords never recorded: secret fields become run-time parameters
- [x] Checkboxes, radio buttons, Enter, uploads, multi-page flows; tested on a real Django admin
- [x] Success learned from the demo's confirmation; Excel / CSV in, results file out
- [ ] WorkSuite sandbox: mail, tickets, expenses, approvals
- [ ] Learn from a screen-recording video
- [ ] Browser extension to record in one click

## Limitations

- **Web apps only** for now, learned from **live demonstrations** (not videos yet).
- A skill **repeats one path**. Decisions ("if the amount is over 5 000 $, send for approval")
  are not learned yet.
- A field you **leave at its default** during the demo is not learned: change it once if
  other tasks need another value.
- **Captchas and two-factor codes** are not bypassed, by design: Mimic stops and asks you.
- Measured on a sandbox and on a stock Django admin. Every new app will bring new surprises:
  that is exactly where [contributions](CONTRIBUTING.md) help most.

---

<div align="center">

**Found a task Mimic should handle?** [Open an issue](../../issues/new/choose) · [Contribute](CONTRIBUTING.md)

MIT © 2026 [Samy Baouche](https://www.linkedin.com/in/samybaouche)

</div>
