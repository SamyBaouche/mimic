"""MLflow tracking and model registry: runs logged, versions registered, production loaded."""

import json
from pathlib import Path

import pytest

from mimic import tracking
from mimic.config import get_settings

pytest.importorskip("mlflow")


@pytest.fixture
def mlflow_home(tmp_path, monkeypatch):
    """A fresh MLflow store per test, inside a throwaway MIMIC_HOME."""
    settings = get_settings()
    monkeypatch.setattr(settings, "mimic_home", tmp_path)
    monkeypatch.setattr(settings, "mlflow_tracking_uri", "")
    monkeypatch.setattr(settings, "mimic_tracking", True)
    return tmp_path


def _model(folder: Path, marker: str) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "risk_model.json").write_text(json.dumps({"marker": marker}))
    return folder


def test_flatten_keeps_numbers_and_nests_names():
    flat = tracking.flatten(
        {"xgboost": {"roc_auc": 0.91, "note": "x", "per": {"a": 1}}, "ok": True, "n": 3}
    )
    assert flat == {"xgboost.roc_auc": 0.91, "xgboost.per.a": 1.0, "ok": 1.0, "n": 3.0}


def test_runs_are_logged_and_the_first_version_goes_to_production(mlflow_home, tmp_path):
    lines = []
    out = tracking.log_run(
        "mimic-risk",
        "xgboost",
        {"max_depth": 3},
        {"xgboost": {"pr_auc": 0.8, "recall": 0.9, "f1": 0.7}},
        model=("risk", _model(tmp_path / "v1", "one")),
        log=lines.append,
    )
    assert out["version"] == "1"
    assert "registered as mimic-risk-model v1" in lines[0] and "now production" in lines[0]
    assert (mlflow_home / "mlflow.db").exists()
    assert tracking.production_version("risk") == "1"

    out = tracking.log_run(
        "mimic-risk", "xgboost", {}, {"xgboost": {"pr_auc": 0.85}},
        model=("risk", _model(tmp_path / "v2", "two")), log=lines.append,
    )  # fmt: skip
    assert out["version"] == "2"
    assert "mimic models promote risk 2" in lines[-1]  # a new version is not used until promoted
    assert tracking.production_version("risk") == "1"

    rows = tracking.versions("risk", ["xgboost.pr_auc"])
    assert [(r["version"], r["production"]) for r in rows] == [("2", False), ("1", True)]
    assert rows[1]["metrics"]["xgboost.pr_auc"] == 0.8


def test_the_production_version_is_the_one_loaded(mlflow_home, tmp_path):
    for v in ("one", "two"):
        tracking.log_run("mimic-risk", "x", {}, {}, model=("risk", _model(tmp_path / v, v)),
                         log=lambda *_: None)  # fmt: skip
    first = tracking.production_dir("risk")
    assert json.loads((first / "risk_model.json").read_text()) == {"marker": "one"}
    tracking.promote("risk", 2)
    second = tracking.production_dir("risk")
    assert second != first and json.loads((second / "risk_model.json").read_text()) == {
        "marker": "two"
    }


def test_off_means_nothing_is_logged_and_local_models_are_used(mlflow_home, tmp_path, monkeypatch):
    monkeypatch.setattr(get_settings(), "mimic_tracking", False)
    assert tracking.log_run("e", "r", {}, {}, model=("risk", _model(tmp_path / "m", "x"))) is None
    assert tracking.production_dir("risk") is None
    assert not (mlflow_home / "mlflow.db").exists()


def test_a_tracking_failure_never_breaks_training(mlflow_home, monkeypatch):
    monkeypatch.setattr(get_settings(), "mlflow_tracking_uri", "http://127.0.0.1:9")
    monkeypatch.setenv("MLFLOW_HTTP_REQUEST_MAX_RETRIES", "0")
    monkeypatch.setenv("MLFLOW_HTTP_REQUEST_TIMEOUT", "2")
    lines = []
    assert tracking.log_run("e", "r", {"a": 1}, {"m": 1.0}, log=lines.append) is None
    assert lines and "not tracked in MLflow" in lines[0]


def test_risk_training_is_tracked_and_loaded_from_the_registry(mlflow_home, tmp_path):
    xgb = pytest.importorskip("xgboost")
    import numpy as np

    from mimic.risk import FEATURE_NAMES, RiskModel
    from mimic.riskml import track_training

    rng = np.random.default_rng(0)
    X, y = rng.random((60, len(FEATURE_NAMES))), rng.integers(0, 2, 60)
    booster = xgb.train({"max_depth": 2}, xgb.DMatrix(X, label=y), num_boost_round=3)
    path = tmp_path / "risk_model.json"
    booster.save_model(str(path))
    meta = {"features": FEATURE_NAMES, "threshold": 0.37, "failure_rate": 0.5,
            "samples": {"train": 42, "test": 18, "runs": 3},
            "test_metrics": {"xgboost": {"pr_auc": 0.6, "recall": 0.7, "f1": 0.5}}}  # fmt: skip
    path.with_suffix(".meta.json").write_text(json.dumps(meta))

    out = track_training(meta, "db", path)
    assert out["version"] == "1"
    model = RiskModel.load()  # no path: the production version from MLflow
    assert model.threshold == 0.37
    assert 0.0 <= model.score({}) <= 1.0


def test_models_command_lists_versions(mlflow_home, tmp_path, capsys):
    tracking.log_run("mimic-risk", "x", {}, {"xgboost": {"pr_auc": 0.75}},
                     model=("risk", _model(tmp_path / "a", "a")), log=lambda *_: None)  # fmt: skip
    assert tracking.main(["list"]) == 0
    out = capsys.readouterr().out
    assert "mimic-risk-model" in out and "v1" in out and "<- production" in out
    assert "0.750" in out
    assert "mimic-element-matcher" in out and "no version yet" in out
    assert tracking.main(["promote", "risk", "7"]) == 1  # no such version
