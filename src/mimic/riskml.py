"""Failure risk model: collect telemetry, train XGBoost, and A/B test mitigation policies.

    uv run python -m mimic.riskml collect --runs 6          # chaos runs, no mitigation
    uv run python -m mimic.riskml train                     # XGBoost vs baselines
    uv run python -m mimic.riskml evaluate --tasks 50       # never / always / predicted

Training uses only steps executed WITHOUT mitigation: their outcome is the natural failure
the model must predict. Evaluation replays the same seeded chaos under every policy.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from mimic.risk import FEATURE_NAMES, vectorize

MODEL_PATH = Path("models/risk_model.json")


# ---------------------------------------------------------------- data


def load_from_db() -> list[dict]:
    from mimic.storage import get_storage

    return get_storage().step_samples()


def load_from_reports(folder: Path = Path("reports")) -> list[dict]:
    samples = []
    for path in sorted(folder.glob("*.json")):
        report = json.loads(path.read_text())
        if report.get("risk_policy", "never") != "never":
            continue
        for step in report.get("steps", []):
            if step.get("features") and not step.get("mitigations"):
                samples.append(
                    {
                        "run_id": report["run_id"],
                        "features": step["features"],
                        "failed": not step["success"],
                    }
                )
    return samples


def load_samples(source: str = "auto") -> list[dict]:
    if source in ("db", "auto"):
        try:
            samples = load_from_db()
            if samples or source == "db":
                return samples
        except Exception as exc:  # noqa: BLE001
            if source == "db":
                raise
            print(f"  (database unavailable: {type(exc).__name__}, reading reports/)")
    return load_from_reports()


# ---------------------------------------------------------------- training


def _rule(f: dict) -> float:
    """Hand-written baseline: an action on a covered or disabled element will fail."""
    return float(f.get("covered", False) or not f.get("enabled", True))


def _smart_rule(f: dict) -> float:
    """Stronger baseline: only clicks are blocked by overlays and disabled buttons."""
    return float(f.get("action") == "click" and _rule(f))


def _metrics(y: np.ndarray, scores: np.ndarray, threshold: float) -> dict:
    from sklearn.metrics import (
        average_precision_score,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
    )

    pred = (scores >= threshold).astype(int)
    both = len(set(y)) > 1
    return {
        "roc_auc": round(float(roc_auc_score(y, scores)), 4) if both else None,
        "pr_auc": round(float(average_precision_score(y, scores)), 4) if both else None,
        "precision": round(float(precision_score(y, pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y, pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y, pred, zero_division=0)), 4),
    }


def _best_threshold(y: np.ndarray, scores: np.ndarray, beta: float = 2.0) -> float:
    """Threshold maximising F-beta on the TRAINING data (never tuned on the test set).

    beta=2 weights recall twice as much as precision: a missed failure costs a whole task,
    an unnecessary mitigation only costs a few hundred milliseconds.
    """
    from sklearn.metrics import fbeta_score

    candidates = np.unique(np.round(scores, 3))
    if len(candidates) == 0:
        return 0.5

    def score(t):
        return fbeta_score(y, (scores >= t).astype(int), beta=beta, zero_division=0)

    return float(max(candidates, key=score))


def train(samples: list[dict], out: Path = MODEL_PATH, seed: int = 42) -> dict:
    import xgboost as xgb
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupShuffleSplit
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    if len(samples) < 50:
        raise SystemExit(f"Only {len(samples)} samples: collect more data first.")
    X = np.array([vectorize(s["features"]) for s in samples])
    y = np.array([int(s["failed"]) for s in samples])
    groups = np.array([s["run_id"] for s in samples])
    if y.sum() < 5:
        raise SystemExit(f"Only {y.sum()} failures: collect with a higher --chaos.")

    # Split by run: steps of the same run never appear in both train and test.
    if len(set(groups)) >= 3:
        splitter = GroupShuffleSplit(n_splits=1, test_size=0.3, random_state=seed)
        train_idx, test_idx = next(splitter.split(X, y, groups))
    else:
        rng = np.random.default_rng(seed)
        order = rng.permutation(len(y))
        cut = int(len(y) * 0.7)
        train_idx, test_idx = order[:cut], order[cut:]
    Xtr, Xte, ytr, yte = X[train_idx], X[test_idx], y[train_idx], y[test_idx]

    pos = max(int(ytr.sum()), 1)
    model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=3,
        learning_rate=0.1,
        subsample=0.9,
        colsample_bytree=0.9,
        scale_pos_weight=(len(ytr) - pos) / pos,  # failures are the minority class
        eval_metric="logloss",
        random_state=seed,
    )
    model.fit(Xtr, ytr)
    threshold = _best_threshold(ytr, model.predict_proba(Xtr)[:, 1])

    logreg = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    logreg.fit(Xtr, ytr)
    lr_threshold = _best_threshold(ytr, logreg.predict_proba(Xtr)[:, 1])

    rule_test = np.array([_rule(samples[i]["features"]) for i in test_idx])
    smart_test = np.array([_smart_rule(samples[i]["features"]) for i in test_idx])
    results = {
        "xgboost": _metrics(yte, model.predict_proba(Xte)[:, 1], threshold),
        "logistic_regression": _metrics(yte, logreg.predict_proba(Xte)[:, 1], lr_threshold),
        "rule_covered_or_disabled": _metrics(yte, rule_test, 0.5),
        "rule_click_and_blocked": _metrics(yte, smart_test, 0.5),
        "always_fail_baseline": _metrics(yte, np.ones(len(yte)), 0.5),
    }

    booster = model.get_booster()
    booster.feature_names = FEATURE_NAMES
    gain = booster.get_score(importance_type="gain")
    importance = dict(sorted(gain.items(), key=lambda kv: kv[1], reverse=True)[:6])

    out.parent.mkdir(exist_ok=True)
    booster.save_model(str(out))
    meta = {
        "features": FEATURE_NAMES,
        "threshold": threshold,
        "samples": {"train": len(ytr), "test": len(yte), "runs": len(set(groups))},
        "failure_rate": round(float(y.mean()), 4),
        "test_metrics": results,
        "top_features_by_gain": {k: round(v, 2) for k, v in importance.items()},
        "trained_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    out.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2))
    return meta


def track_training(meta: dict, source: str, path: Path = MODEL_PATH) -> dict | None:
    """Log the run in MLflow and register the model as a new version (if MLflow is on)."""
    import shutil
    import tempfile

    from mimic import tracking

    with tempfile.TemporaryDirectory() as tmp:
        for f in (path, path.with_suffix(".meta.json")):
            shutil.copy(f, Path(tmp) / f.name)
        return tracking.log_run(
            "mimic-risk",
            "xgboost",
            {
                "source": source,
                "n_estimators": 200,
                "max_depth": 3,
                "learning_rate": 0.1,
                "threshold": meta["threshold"],
                "features": len(meta["features"]),
                **{f"samples_{k}": v for k, v in meta["samples"].items()},
            },
            {**meta["test_metrics"], "failure_rate": meta["failure_rate"]},
            model=("risk", Path(tmp)),
        )


def print_training(meta: dict) -> None:
    s = meta["samples"]
    print(f"\nSamples: {s['train']} train / {s['test']} test from {s['runs']} runs")
    print(f"Failure rate: {meta['failure_rate']:.1%}   decision threshold: {meta['threshold']:.3f}")
    print(f"\n{'model':<26}{'ROC AUC':>9}{'PR AUC':>9}{'prec':>7}{'recall':>8}{'F1':>7}")
    for name, m in meta["test_metrics"].items():
        roc = f"{m['roc_auc']:.3f}" if m["roc_auc"] is not None else "  -"
        pr = f"{m['pr_auc']:.3f}" if m["pr_auc"] is not None else "  -"
        print(f"{name:<26}{roc:>9}{pr:>9}{m['precision']:>7.2f}{m['recall']:>8.2f}{m['f1']:>7.2f}")
    print(f"\nTop features: {', '.join(meta['top_features_by_gain'])}")
    print(f"Model saved to {MODEL_PATH}")


# ---------------------------------------------------------------- collect / evaluate


def _setup(skill_name: str):
    from mimic.config import get_settings
    from mimic.store import SkillStore

    skill = SkillStore().get(skill_name)
    if skill is None:
        raise SystemExit(f"Unknown skill '{skill_name}'.")
    return skill, skill.start_url or get_settings().sandbox_url


def collect(skill_name: str, runs: int, tasks: int, chaos_levels: list[float]) -> None:
    from mimic.run import print_report, run_batch
    from mimic.sandbox.generate_invoices import generate

    skill, url = _setup(skill_name)
    for r in range(runs):
        chaos = chaos_levels[r % len(chaos_levels)]
        records = generate(tasks, seed=100 + r)
        print(f"\n[collect {r + 1}/{runs}] chaos={chaos}")
        report = run_batch(
            skill, records, url, "v1", chaos=chaos, seed_base=10_000 * (r + 1), risk_policy="never"
        )
        print_report(report)


def evaluate(skill_name: str, tasks: int, chaos: float, seed: int) -> dict:
    from mimic.risk import RiskModel
    from mimic.run import run_batch
    from mimic.sandbox.generate_invoices import generate

    skill, url = _setup(skill_name)
    model = RiskModel.load(MODEL_PATH)  # the model just trained, before you promote it
    records = generate(tasks, seed=seed)
    rows = {}
    for policy in ["never", "always", "predicted"]:
        print(f"\n[evaluate] policy={policy}")
        report = run_batch(
            skill,
            records,
            url,
            "v1",
            log_to_db=False,
            chaos=chaos,
            seed_base=seed,  # same perturbations for every policy: a fair A/B test
            risk_policy=policy,
            risk_model=model if policy == "predicted" else None,
        )
        rows[policy] = {
            "task_success": report["task_success_rate"],
            "failed_actions": report["failed_actions"],
            "mitigated_steps": report["mitigated_steps"],
            "avg_task_ms": report["avg_task_ms"],
        }
    Path("reports").mkdir(exist_ok=True)
    Path("reports/risk_evaluation.json").write_text(json.dumps(rows, indent=2))
    from mimic import tracking

    tracking.log_run(
        "mimic-risk",
        "a/b test",
        {
            "skill": skill_name,
            "tasks": tasks,
            "chaos": chaos,
            "seed": seed,
            "model": str(MODEL_PATH),
        },  # fmt: skip
        rows,
        files=[Path("reports/risk_evaluation.json")],
    )
    return rows


def print_evaluation(rows: dict) -> None:
    print(
        f"\n{'policy':<11}{'task success':>13}{'failed actions':>16}{'mitigated':>11}{'ms/task':>9}"
    )
    for name, r in rows.items():
        print(
            f"{name:<11}{r['task_success']:>12.0%}{r['failed_actions']:>16}"
            f"{r['mitigated_steps']:>11}{r['avg_task_ms']:>9}"
        )
    n, a, p = rows["never"], rows["always"], rows["predicted"]
    if n["failed_actions"]:
        cut = 1 - p["failed_actions"] / n["failed_actions"]
        print(f"\nPredicted policy: {cut:.0%} fewer failed actions than no mitigation")
    if a["mitigated_steps"]:
        share = p["mitigated_steps"] / a["mitigated_steps"]
        print(f"Mitigated only {share:.0%} of the steps that always-on mitigation touches")
    if a["avg_task_ms"]:
        print(
            f"Time per task: {p['avg_task_ms']} ms vs {a['avg_task_ms']} ms always-on "
            f"({1 - p['avg_task_ms'] / a['avg_task_ms']:.0%} faster) "
            f"and {n['avg_task_ms']} ms with no mitigation"
        )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Failure risk model")
    sub = parser.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("collect", help="Run chaos batches to gather telemetry")
    c.add_argument("--skill", default="enter_invoice")
    c.add_argument("--runs", type=int, default=6)
    c.add_argument("--tasks", type=int, default=50)
    c.add_argument("--chaos", default="0.2,0.4,0.6", help="Comma-separated levels to cycle")
    t = sub.add_parser("train", help="Train XGBoost and compare with baselines")
    t.add_argument("--source", choices=["auto", "db", "reports"], default="auto")
    e = sub.add_parser("evaluate", help="A/B test never / always / predicted mitigation")
    e.add_argument("--skill", default="enter_invoice")
    e.add_argument("--tasks", type=int, default=50)
    e.add_argument("--chaos", type=float, default=0.4)
    e.add_argument("--seed", type=int, default=777_000)
    args = parser.parse_args(argv)

    if args.cmd == "collect":
        levels = [float(x) for x in args.chaos.split(",")]
        collect(args.skill, args.runs, args.tasks, levels)
    elif args.cmd == "train":
        samples = load_samples(args.source)
        print(f"Loaded {len(samples)} steps ({sum(s['failed'] for s in samples)} failures)")
        meta = train(samples)
        print_training(meta)
        track_training(meta, args.source)
    else:
        print_evaluation(evaluate(args.skill, args.tasks, args.chaos, args.seed))


if __name__ == "__main__":
    main()
