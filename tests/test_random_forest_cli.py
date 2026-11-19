"""Public configuration/CLI choices, without changing the frozen experiment."""
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


def test_forest_example_has_four_explicit_classifiers_and_four_candidates():
    cfg = load_config(ROOT / "configs/random_forest.json")
    policy = classifier_policy(cfg)
    assert policy["classifiers"] == ["LDA", "RBF-SVM", "KFDA", "RandomForest"]
    assert len(policy["random_forest"]["candidates"]) == 4
    assert {x["n_estimators"] for x in policy["random_forest"]["candidates"]} == {128}
    old = classifier_policy(load_config(ROOT / "configs/lab_replication.json"))
    assert old["classifiers"] == ["LDA", "RBF-SVM", "KFDA"]
    assert old["random_forest"] is None


def test_explicit_cli_override_is_detached_and_preserves_forest_settings():
    original = load_config(ROOT / "configs/random_forest.json")
    before = deepcopy(original)
    only_forest = apply_classifier_arguments(original, ["RandomForest"])
    assert only_forest["classifiers"] == ["RandomForest"]
    assert only_forest["random_forest"] == before["random_forest"]
    only_forest["random_forest"]["parameters"]["n_estimators"] = 2
    only_lda = apply_classifier_arguments(original, ["LDA"])
    assert only_lda["classifiers"] == ["LDA"]
    assert "random_forest" not in only_lda
    assert original == before


@pytest.mark.parametrize("roster", [["RandomForest", "RandomForest"], ["randomforest"], []])
def test_invalid_explicit_rosters_are_not_silently_accepted(roster):
    with pytest.raises(ValueError, match="classifiers"):
        apply_classifier_arguments(load_config(ROOT / "configs/lab_replication.json"), roster)


@pytest.mark.parametrize("script", ["03_intra_subject.py", "04_inter_subject.py"])
def test_bad_forest_config_fails_before_missing_feature_file_is_opened(tmp_path, script):
    cfg = load_config(ROOT / "configs/random_forest.json")
    cfg["random_forest"]["parameters"]["n_estimators"] = 0
    config = tmp_path / "invalid.json"
    config.write_text(json.dumps(cfg))
    destination = tmp_path / "must-not-exist"
    result = subprocess.run([sys.executable, str(ROOT / "scripts" / script),
        "--features", str(tmp_path / "unavailable-features.csv"), "--config", str(config),
        "--out-dir", str(destination)], cwd=ROOT, text=True, capture_output=True, timeout=30)
    assert result.returncode != 0
    assert "n_estimators" in result.stderr
    assert "FileNotFoundError" not in result.stderr
    assert not destination.exists()


def test_run_all_forwards_classifier_choice_only_to_evaluation(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location("rf_test_run_all", ROOT / "scripts/run_all.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    calls = []
    monkeypatch.setattr(module, "run", lambda *args: calls.append(args))
    monkeypatch.setattr(sys, "argv", [str(ROOT / "scripts/run_all.py"),
        "--config", str(ROOT / "configs/random_forest.json"), "--classifiers", "RandomForest",
        "--out-root", str(tmp_path / "pipeline")])
    assert module.main() == 0
    assert len(calls) == 5
    assert all("--classifiers" not in args for args in calls[:3])
    assert all(args[-2:] == ("--classifiers", "RandomForest") for args in calls[3:])


def test_generic_classification_figures_include_forest_numerical_values(tmp_path, monkeypatch):
    from eeg_ma import visualization
    names = ["LDA", "RBF-SVM", "KFDA", "RandomForest"]
    means, deviations = [0.55, 0.61, 0.63, 0.67], [0.02, 0.03, 0.04, 0.05]
    intra = pd.DataFrame({"comparison": ["rest1_vs_type1"] * 4, "classifier": names,
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
    assert [row[0] for row in observed] == names * 2
    for (name, height, error), expected, sd in zip(observed, means * 2, deviations * 2):
        np.testing.assert_array_equal(height, [expected])
        np.testing.assert_array_equal(error, [sd])
