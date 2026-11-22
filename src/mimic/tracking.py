"""Experiment tracking and model registry with MLflow (optional: pip install "mimic-agent[mlflow]").

Every training run (failure-risk model, element matcher) and every extraction benchmark is
logged with its parameters, metrics and files. Trained models are registered as new versions;
the version you promote to the `production` alias is the one Mimic loads:

    mimic models                      # versions, their metrics, which one is in production
    mimic models promote risk 3       # use version 3 of the risk model from now on
    mimic models ui                   # browse runs and models in the MLflow UI

Runs live next to your skills (~/.mimic/mlflow.db and ~/.mimic/mlruns), or on any MLflow
server with MLFLOW_TRACKING_URI. Without MLflow installed, or with MIMIC_TRACKING=off,
nothing is logged and Mimic loads the models in ./models as before.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

# kind -> registered model name
MODELS = {"risk": "mimic-risk-model", "matcher": "mimic-element-matcher"}
ALIAS = "production"


def available() -> bool:
    from mimic.config import get_settings

    if not get_settings().mimic_tracking:
        return False
    try:
        import mlflow  # noqa: F401
    except ImportError:
        return False
    return True


def tracking_uri() -> str:
    from mimic.config import get_settings

    settings = get_settings()
    db = (Path(settings.mimic_home) / "mlflow.db").resolve().as_posix()  # C:/... on Windows
    return settings.mlflow_tracking_uri or f"sqlite:///{db}"


def _client():
    import logging

    import mlflow
    from mlflow.tracking import MlflowClient

    for name in ("alembic", "mlflow"):  # migrations and info lines on first use
        logging.getLogger(name).setLevel(logging.WARNING)
    Path(_home()).mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(tracking_uri())
    return mlflow, MlflowClient()


def _home() -> Path:
    from mimic.config import get_settings

    return Path(get_settings().mimic_home)


def _experiment(mlflow, client, name: str) -> str:
    found = client.get_experiment_by_name(name)
    if found is not None:
        return found.experiment_id
    location = (_home() / "mlruns" / name).resolve().as_uri()
    if not tracking_uri().startswith("sqlite:"):
        location = None  # a remote server keeps its own artifact store
    return client.create_experiment(name, artifact_location=location)


def flatten(metrics: dict, prefix: str = "") -> dict[str, float]:
    """{"xgboost": {"roc_auc": 0.9}} -> {"xgboost.roc_auc": 0.9}; non-numbers dropped."""
    out = {}
    for key, value in metrics.items():
        name = f"{prefix}{key}"
        if isinstance(value, dict):
            out |= flatten(value, f"{name}.")
        elif isinstance(value, bool):
            out[name] = float(value)
        elif isinstance(value, int | float):
            out[name] = float(value)
    return out


def log_run(
    experiment: str,
    run_name: str,
    params: dict,
    metrics: dict,
    files: list[Path] = (),
    model: tuple[str, Path] | None = None,
    log=print,
) -> dict | None:
    """Log one run. `model=(kind, folder)` also registers the folder as a new model version.
    Never breaks the caller: tracking problems are reported in one line."""
    if not available():
        return None
    try:
        mlflow, client = _client()
        exp_id = _experiment(mlflow, client, experiment)
        with mlflow.start_run(experiment_id=exp_id, run_name=run_name) as run:
            mlflow.log_params({k: str(v)[:500] for k, v in params.items()})
            mlflow.log_metrics(flatten(metrics))
            for path in files:
                if Path(path).exists():
                    mlflow.log_artifact(str(path))
            version = None
            if model is not None:
                kind, folder = model
                mlflow.log_artifacts(str(folder), artifact_path="model")
                name = MODELS[kind]
                if client.search_registered_models(f"name='{name}'") == []:
                    client.create_registered_model(name)
                version = str(
                    client.create_model_version(
                        name, source=f"{run.info.artifact_uri}/model", run_id=run.info.run_id
                    ).version
                )
        out = {"run_id": run.info.run_id, "experiment": experiment, "version": version}
        where = f"MLflow run {run.info.run_id[:8]} in '{experiment}'"
        if version is not None:
            where += f", registered as {MODELS[model[0]]} v{version}"
            if production_version(model[0]) is None:
                promote(model[0], version)
                where += f" (now {ALIAS}: the first version)"
            else:
                where += f" (promote it: mimic models promote {model[0]} {version})"
        log(f"Tracked: {where}")
        return out
    except Exception as exc:  # noqa: BLE001 - tracking must never lose a training run
        log(f"  ! not tracked in MLflow ({type(exc).__name__}: {exc})")
        return None


# ---------------------------------------------------------------- registry


def production_version(kind: str) -> str | None:
    if not available():
        return None
    try:
        _, client = _client()
        return str(client.get_model_version_by_alias(MODELS[kind], ALIAS).version)
    except Exception:  # noqa: BLE001 - no model, no alias yet
        return None


def promote(kind: str, version: int | str) -> None:
    if kind not in MODELS:
        raise SystemExit(f"Unknown model '{kind}': {', '.join(MODELS)}")
    _, client = _client()
    client.set_registered_model_alias(MODELS[kind], ALIAS, str(version))


def production_dir(kind: str) -> Path | None:
    """The files of the version in production, downloaded once into ~/.mimic/models/registry.
    None when MLflow is off or nothing is promoted: callers fall back to ./models."""
    version = production_version(kind)
    if version is None:
        return None
    target = _home() / "models" / "registry" / MODELS[kind] / f"v{version}"
    if target.exists() and any(target.iterdir()):
        return target
    try:
        mlflow, _ = _client()
        tmp = target.with_name(target.name + ".part")
        shutil.rmtree(tmp, ignore_errors=True)
        tmp.mkdir(parents=True)
        mlflow.artifacts.download_artifacts(f"models:/{MODELS[kind]}@{ALIAS}", dst_path=str(tmp))
        tmp.rename(target)
        return target
    except Exception as exc:  # noqa: BLE001
        print(f"  ! {MODELS[kind]} v{version} not downloaded ({type(exc).__name__})")
        return None


def versions(kind: str, keys: list[str]) -> list[dict]:
    """Every registered version with a few of its run's metrics, newest first."""
    if not available():
        return []
    _, client = _client()
    name = MODELS[kind]
    try:
        found = client.search_model_versions(f"name='{name}'")
    except Exception:  # noqa: BLE001
        return []
    prod = production_version(kind)
    rows = []
    for mv in sorted(found, key=lambda v: int(v.version), reverse=True):
        metrics = client.get_run(mv.run_id).data.metrics if mv.run_id else {}
        rows.append(
            {
                "version": str(mv.version),
                "production": str(mv.version) == prod,
                "created": mv.creation_timestamp,
                "run_id": mv.run_id,
                "metrics": {k: metrics.get(k) for k in keys},
            }
        )
    return rows


# ---------------------------------------------------------------- CLI

SUMMARY = {
    "risk": ["xgboost.pr_auc", "xgboost.recall", "xgboost.f1"],
    "matcher": ["fine_tuned.synthetic_top1", "fine_tuned.sandbox_v2_top1"],
}


def main(argv: list[str] | None = None) -> int:
    import argparse
    import subprocess
    import sys
    import time

    parser = argparse.ArgumentParser(prog="mimic models", description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="cmd")
    sub.add_parser("list", help="Registered versions and which one is in production")
    p = sub.add_parser("promote", help="Make a version the one Mimic loads")
    p.add_argument("kind", choices=list(MODELS))
    p.add_argument("version")
    u = sub.add_parser("ui", help="Open the MLflow UI on these runs")
    u.add_argument("--port", type=int, default=5000)
    args = parser.parse_args(argv)

    if not available():
        print('MLflow tracking is off. Turn it on: pip install "mimic-agent[mlflow]"')
        print("(and remove MIMIC_TRACKING=off from .env if it is there)")
        return 1
    if args.cmd == "promote":
        try:
            promote(args.kind, args.version)
        except SystemExit:
            raise
        except Exception as exc:  # noqa: BLE001
            print(f"Cannot promote {MODELS[args.kind]} v{args.version}: {exc}")
            return 1
        print(f"{MODELS[args.kind]} v{args.version} is now {ALIAS}: Mimic loads it from now on.")
        return 0
    if args.cmd == "ui":
        print(f"MLflow UI on http://127.0.0.1:{args.port}  (Ctrl+C to stop)")
        cmd = [sys.executable, "-m", "mlflow", "ui", "--backend-store-uri", tracking_uri(),
               "--port", str(args.port)]  # fmt: skip
        return subprocess.call(cmd)

    print(f"Tracking: {tracking_uri()}")
    for kind, keys in SUMMARY.items():
        rows = versions(kind, keys)
        print(f"\n{MODELS[kind]}")
        if not rows:
            print(f"  no version yet (train one: mimic {kind} train)")
            continue
        print(f"  {'ver':<5}{'trained':<18}" + "".join(f"{k.split('.', 1)[-1]:>18}" for k in keys))
        for r in rows:
            when = time.strftime("%Y-%m-%d %H:%M", time.localtime(r["created"] / 1000))
            vals = "".join(
                f"{v:>18.3f}" if v is not None else f"{'-':>18}" for v in r["metrics"].values()
            )
            mark = f"  <- {ALIAS}" if r["production"] else ""
            print(f"  v{r['version']:<4}{when:<18}{vals}{mark}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
