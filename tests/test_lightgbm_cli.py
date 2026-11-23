"""Public five-model choices preserve historical and RF-only contracts."""
from __future__ import annotations

from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from eeg_ma.classifier_config import classifier_policy
from eeg_ma.config import load_config
from eeg_ma.experiment_runs import apply_classifier_arguments


ROOT = Path(__file__).resolve().parents[1]
ROSTER = ["LDA", "RBF-SVM", "KFDA", "RandomForest", "LightGBM"]


def test_explicit_profile_has_bounded_five_model_search_and_keeps_old_profiles():
    cfg = load_config(ROOT / "configs/lightgbm.json")
    policy = classifier_policy(cfg)
    assert policy["schema_version"] == "eeg-classifier-policy-2"
    assert policy["classifiers"] == ROSTER
    assert len(policy["lightgbm"]["candidates"]) == 4
    assert {(p["num_leaves"], p["lambda_l2"]) for p in policy["lightgbm"]["candidates"]} == {
        (3, 1.0), (3, 5.0), (7, 1.0), (7, 5.0)}
    assert all(p["early_stopping_rounds"] is None for p in policy["lightgbm"]["candidates"])
    forest = classifier_policy(load_config(ROOT / "configs/random_forest.json"))
    historical = classifier_policy(load_config(ROOT / "configs/lab_replication.json"))
    assert forest["schema_version"] == historical["schema_version"] == "eeg-classifier-policy-1"
    assert "lightgbm" not in forest and "lightgbm" not in historical
    assert forest["classifiers"] == ROSTER[:-1]
    assert historical["classifiers"] == ROSTER[:3]
    assert policy["random_forest"] == forest["random_forest"]


@pytest.mark.parametrize("names, retained", [
    (["LightGBM"], {"lightgbm"}),
    (["RandomForest"], {"random_forest"}),
    (["LightGBM", "RandomForest"], {"random_forest", "lightgbm"}),
    (["LDA"], set()),
])
def test_overrides_keep_only_requested_model_configurations_without_mutation(names, retained):
    original = load_config(ROOT / "configs/lightgbm.json")
    before = deepcopy(original)
    effective = apply_classifier_arguments(original, names)
    assert effective["classifiers"] == names
    assert {name for name in ("lightgbm", "random_forest") if name in effective} == retained
    for name in retained:
        assert effective[name] == before[name]
        effective[name]["parameters"]["n_estimators"] = 2
    assert original == before


@pytest.mark.parametrize("roster", [["LightGBM", "LightGBM"], ["lightgbm"], ["LGBM"]])
def test_new_classifier_spelling_and_uniqueness_are_explicit(roster):
    with pytest.raises(ValueError, match="classifiers"):
        apply_classifier_arguments(load_config(ROOT / "configs/lab_replication.json"), roster)


@pytest.mark.parametrize("script", ["03_intra_subject.py", "04_inter_subject.py"])
@pytest.mark.parametrize("parameters, diagnostic", [
    ({"num_leaves": 1}, "num_leaves"),
    ({"early_stopping_rounds": 2}, "early stopping"),
])
def test_invalid_model_fails_before_feature_loading_or_output(tmp_path, script, parameters, diagnostic):
    cfg = load_config(ROOT / "configs/lightgbm.json")
    cfg["lightgbm"]["parameters"].update(parameters)
    config = tmp_path / "invalid.json"
    config.write_text(json.dumps(cfg), encoding="utf-8")
    destination = tmp_path / "must-not-exist"
    result = subprocess.run([sys.executable, str(ROOT / "scripts" / script),
        "--features", str(tmp_path / "missing.csv"), "--config", str(config),
        "--out-dir", str(destination)], cwd=ROOT, text=True, capture_output=True, timeout=30)
    assert result.returncode != 0
    assert diagnostic in result.stderr
    assert "FileNotFoundError" not in result.stderr
    assert not destination.exists()


def test_run_all_forwards_five_classifier_roster_only_to_evaluations(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location("lgb_test_run_all", ROOT / "scripts/run_all.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    calls = []
    monkeypatch.setattr(module, "run", lambda *args: calls.append(args))
    monkeypatch.setattr(sys, "argv", [str(ROOT / "scripts/run_all.py"),
        "--config", str(ROOT / "configs/lightgbm.json"), "--classifiers", *ROSTER,
        "--out-root", str(tmp_path / "pipeline")])
    assert module.main() == 0
    assert len(calls) == 5
    assert all("--classifiers" not in args for args in calls[:3])
    assert all(args[-6:] == ("--classifiers", *ROSTER) for args in calls[3:])


def test_generic_figures_display_five_declared_numerical_series(tmp_path, monkeypatch):
    from eeg_ma import visualization
    means, deviations = [0.55, 0.61, 0.63, 0.67, 0.69], [0.02, 0.03, 0.04, 0.05, 0.06]
    intra = pd.DataFrame({"comparison": ["rest1_vs_type1"] * 5, "classifier": ROSTER,
        "participant_accuracy_mean": means, "participant_accuracy_std": deviations})
    inter = intra.rename(columns={"participant_accuracy_mean": "lopo_accuracy_mean",
        "participant_accuracy_std": "lopo_accuracy_std"})
    observed = []
    original = visualization.plt.Axes.bar
    def capture(self, x, height, *args, **kwargs):
        observed.append((kwargs["label"], np.asarray(height), np.asarray(kwargs["yerr"])))
        return original(self, x, height, *args, **kwargs)
    monkeypatch.setattr(visualization.plt.Axes, "bar", capture)
    def close(fig, base):
        visualization.plt.close(fig)
        return []
    monkeypatch.setattr(visualization, "_save", close)
    visualization.plot_all_feature_classification(intra, inter, tmp_path)
    assert [row[0] for row in observed] == ROSTER * 2
    for (_, height, error), expected, sd in zip(observed, means * 2, deviations * 2):
        np.testing.assert_array_equal(height, [expected])
        np.testing.assert_array_equal(error, [sd])
