# Contributing to Mimic

Thanks for helping! Mimic is young, so real-world feedback is the most valuable contribution.

## Good first contributions

- **A site where Mimic fails**: open an issue with the steps, what you expected, and the
  `--debug` output (remove any personal data first).
- **New sandbox perturbations** in `sandbox/app.py` (a new kind of flakiness the agent should
  survive), with a test.
- **New actions** (checkbox, radio, keyboard, upload) in `src/mimic/agent.py`.

## Development setup

```bash
uv sync --extra agent
uv run playwright install chromium
docker compose up -d && uv run python -m mimic.db
uv run pytest -q
```

Tests use a separate `mimic_test` database and never touch your learned skills.
Tests that need a browser or Postgres skip themselves cleanly when unavailable.

## Before opening a pull request

```bash
uv run ruff check . && uv run ruff format --check .
uv run pytest -q
```

- One feature or fix per pull request, with a test that fails without your change.
- Keep inference local: no new dependency on a hosted model API.
- If you change behaviour the agent depends on (locators, healing, risk features),
  include before/after numbers from `mimic.run` or `mimic.riskml evaluate`.

## Ground rules

Be kind, be specific, and never include real personal or financial data in issues,
recordings or test fixtures.
