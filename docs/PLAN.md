# Mimic — 8 sprints (Nov 2026 → New Year)

| Sprint | Dates | Goal | Definition of Done |
|---|---|---|---|
| S1 | Nov 9–15 | Setup + sandbox | `make up/db/api/sandbox` work, CI green, LedgerLite v1/v2 |
| S2 | Nov 16–22 | Recorder | Playwright captures clicks, typed values, selectors and screenshots of one demo |
| S3 | Nov 23–29 | Demo → skill | Ollama planner turns a recording into parametrised steps, stored in pgvector |
| S4 | Nov 30–Dec 6 | Replay agent | LangGraph loop (observe → act → verify) replays a skill on 50 invoices on v1 |
| S5 | Dec 7–13 | Vision + self-healing | Ollama VLM + visual embeddings relocate elements on v2 |
| S6 | Dec 14–20 | Fine-tuned detector | HuggingFace model fine-tuned on sandbox screenshots, accuracy vs. VLM baseline |
| S7 | Dec 21–27 | Failure predictor | XGBoost on step_logs, retries/human review, live dashboard via WebSocket |
| S8 | Dec 28–Jan 3 | Evals + polish | Success-rate eval in CI, README GIF, real CV metrics |

Cut order if late: dashboard polish → fine-tuning (keep VLM only) → XGBoost.
Never cut: recorder, replay agent, self-healing, evals.
