# Changelog

All notable changes to Mimic. Versions follow [Semantic Versioning](https://semver.org).

## [Unreleased]

### Added
- `mimic skills edit`: fix a value, remove a step, turn a step into an input, rename inputs,
  change the description, interactively or with flags; every edit is checked, and the previous
  version is backed up. `mimic skills export` / `import` to share a skill or undo an edit.
- Model profiles picked from the hardware: 7B models with a GPU or 16 GB+ of RAM, 3B models
  ("light") otherwise; `MIMIC_PROFILE=auto|standard|light`.
- `mimic doctor` shows the hardware and the profile it chose, warns when models are too large
  for the machine, and `mimic doctor --pull` downloads the missing models.
- Document reading: labels printed before identifiers ("No.", "Ref #", "Facture n°") are
  removed; an identifier misread by a digit is repaired from the PDF's own text when exactly one
  printed token matches. Measured on the 50 demo invoices: 7B models 99.2% of fields right
  (96% of documents), 3B models 98.4% (92%, up from 6%), every error flagged in both cases.

### Fixed
- An identifier read with a missing digit passed the PDF text check because it was part of a
  longer printed number; the check now needs the whole word.
- `mimic doctor` says when VISION_MODEL / PLANNER_MODEL in .env override the chosen profile.
- `mimic extract eval` warns clearly when dropdown choices cannot be read from the app.

## [0.3.0] - 2026-11-20

### Added
- WorkSuite sandbox: a mailbox (supplier emails with invoice PDFs, replies) and an approvals app.
- Workflows (`mimic workflow check | run`): chain skills across apps for every email or row;
  read attachments; `if:` conditions in plain words, translated into checked rules; one plan
  per email; every step verified with its own confirmation.
- Messages with email addresses or links are recognised as varying in success signals.
- New actions: checkboxes, radio buttons, Enter in a field, file uploads, multi-page flows.
- Success signal learned from the end of the demo (final page + confirmation message); runs are
  graded with it on any site.
- Excel / CSV input with your own column names; a results file per run; `--only-failed` retries.
- Steps on pages the app skips (already signed in) are skipped.
- Real-app test and benchmark on a stock Django admin: 30/30 rows saved correctly.
- The fine-tuned element matcher is published on the Hugging Face Hub and downloaded on first
  use; `mimic matcher push` publishes your own, with a model card.
- CI on Linux, Windows and macOS; releases published to PyPI by GitHub Actions.

### Fixed
- Labels without `for`, fields named by a `<fieldset>` legend, buttons written as
  `<input value="Save">`, dropdowns whose label differs from their value.

## [0.2.0] - 2026-11-19

### Added
- `pip install mimic-agent` and a single `mimic` command, with `mimic doctor`.
- SQLite storage by default (no Docker); Postgres + pgvector as an option.
- Passwords and other secret fields are never recorded.
- Live dashboard (`mimic serve`): streamed agent browser, approve / edit / skip / pause.

## [0.1.0] - 2026-11-17

### Added
- Learn a web task from one demonstration, replay it on a batch, review each plan.
- Self-healing with memory, embeddings, a fine-tuned matcher and a vision model.
- XGBoost failure-risk model; reading inputs from invoice PDFs.
