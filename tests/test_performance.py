"""CPU budget planning and benchmark science guardrails; no human EEG inputs."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from eeg_ma.performance import (
    CALIBRATION_VERSION, CostLimitExceeded, CostPlanError, machine_identity,
    plan_evaluation_cost, tree_work_units,
)
from eeg_ma.classifier_config import classifier_policy
from exploratory_fixture import create_exploratory_fixture


@pytest.fixture(scope="module")
def fixture(tmp_path_factory):
    root = tmp_path_factory.mktemp("performance-cost") / "source"
    features, config, _, _, frame = create_exploratory_fixture(root)
    return features, config, frame


def test_structural_cost_and_grouped_inner_fit_counts(fixture):
    _, config, frame = fixture
    cfg = json.loads(Path(config).read_text())
    plan = plan_evaluation_cost(frame, cfg, scope="inter")
    assert plan["status"] == "preflight_only"
    assert plan["outer_folds"] == 3
    assert plan["full_tree_model_fits"] == 30
    assert plan["dataset"]["participants"] == 3
    assert plan["tree_work_units"] > 0
    assert plan["projected_nodes_upper"] > plan["peak_model_nodes_upper"]
    assert plan["retained_outer_model_nodes_upper"] > 0
    assert set(plan["other_classifier_model_fits"]) == {"LDA", "RBF-SVM", "KFDA"}
    assert plan["sfs_inner_model_fits_upper"] > 0
    assert plan["estimated_seconds"] is None
    assert "not_hard" in plan["memory_scope"]


def test_full_feature_mode_avoids_fake_sfs_work(fixture):
    _, config, frame = fixture
    cfg = json.loads(Path(config).read_text())
    cfg["classifiers"] = ["RandomForest", "LightGBM"]
    cfg["feature_selection_policy"] = "all_declared_features"
    cfg["sfs"]["max_features"] = None
    plan = plan_evaluation_cost(frame, cfg, scope="inter")
    assert plan["sfs_inner_model_fits_upper"] == 0
    assert plan["dataset"]["max_model_features"] == 3


@pytest.mark.parametrize("limit", [0, -1, True, float("nan"), float("inf")])
def test_unbounded_and_bogus_limits_fail_closed(fixture, limit):
    _, config, frame = fixture
    cfg = json.loads(Path(config).read_text())
    with pytest.raises(CostPlanError):
        plan_evaluation_cost(frame, cfg, scope="inter", max_estimated_rss_mib=limit)


def test_runtime_budget_requires_real_measured_same_host_calibration(fixture, tmp_path):
    _, config, frame = fixture
    cfg = json.loads(Path(config).read_text())
    with pytest.raises(CostLimitExceeded, match="same-host calibration"):
        plan_evaluation_cost(frame, cfg, scope="inter", max_estimated_seconds=5)
    fake = tmp_path / "bad.json"
    fake.write_text(json.dumps({"schema_version": CALIBRATION_VERSION, "host": {"cpu_model": "other"},
                                "seconds_per_unit": .0001, "measurements": 3,
                                "safety_factor": 1.5}))
    with pytest.raises(CostPlanError, match="calibration"):
        plan_evaluation_cost(frame, cfg, scope="inter", calibration=fake)


def test_estimated_memory_budget_rejects_before_training(fixture):
    _, config, frame = fixture
    cfg = json.loads(Path(config).read_text())
    with pytest.raises(CostLimitExceeded, match="planned memory"):
        plan_evaluation_cost(frame, cfg, scope="inter", max_estimated_rss_mib=1)


def test_cli_dry_run_and_budget_failure_publish_no_results(fixture, tmp_path):
    features, config, frame = fixture
    script = Path(__file__).resolve().parents[1] / "scripts" / "04_inter_subject.py"
    output = tmp_path / "never_complete"
    preflight = subprocess.run([sys.executable, str(script),
                               "--features", str(features), "--config", str(config),
                               "--out-dir", str(output), "--dry-run"],
                               capture_output=True, text=True)
    assert preflight.returncode == 0, preflight.stderr
    result = json.loads(preflight.stdout)
    assert result["status"] == "preflight_only"
    assert not output.exists()
    rejected_dir = tmp_path / "blocked-plan"
    rejected = subprocess.run([sys.executable, str(script),
                               "--features", str(features), "--config", str(config),
                               "--out-dir", str(output),
                               "--cost-max-estimated-seconds", "1",
                               "--cost-plan-out", str(rejected_dir)],
                               capture_output=True, text=True)
    assert rejected.returncode == 2, rejected.stderr
    assert json.loads(rejected.stdout)["status"] == "blocked"
    assert not output.exists()
    evidence = json.loads((rejected_dir / "cost_plan.json").read_text())
    assert evidence["complete_run_published"] is False


def test_deterministic_synthetic_benchmark_generation(tmp_path):
    from importlib.util import spec_from_file_location, module_from_spec
    src = Path(__file__).resolve().parents[1] / "benchmarks" / "run.py"
    spec = spec_from_file_location("bench_engineering", src)
    m = module_from_spec(spec)
    spec.loader.exec_module(m)
    a, y = m.fixture(300, 126, "sparse", 270)
    b, z = m.fixture(300, 126, "sparse", 270)
    np.testing.assert_array_equal(a, b)
    np.testing.assert_array_equal(y, z)
    assert not np.isnan(a).any()
    assert set(np.unique(y)) == {0, 1}
    assert len(m.PRESETS["full"]) == 16
    assert {case[2] for case in m.PRESETS["full"]} == {
        "sparse", "dense", "constant", "correlated"
    }


def test_work_units_monotone_across_dimensions():
    rf = {"n_estimators": 8, "max_features": "sqrt"}
    boost = {"n_estimators": 8, "num_leaves": 7, "feature_fraction": 1}
    for name, cfg in (("RandomForest", rf), ("LightGBM", boost)):
        assert (tree_work_units(name, 300, 21, cfg)
                < tree_work_units(name, 600, 21, cfg)
                < tree_work_units(name, 600, 126, cfg))


def test_calibration_is_recomputed_from_observed_rows_and_exact_model_sources(fixture, tmp_path):
    from eeg_ma import lightgbm, random_forest
    _, config, frame = fixture
    cfg = json.loads(Path(config).read_text())
    cfg["classifiers"] = ["RandomForest", "LightGBM"]
    cfg["feature_selection_policy"] = "all_declared_features"
    cfg["sfs"]["max_features"] = None
    host = machine_identity()
    rows = [
        {"model": model, "units": 100, "fit_wall_s": .1,
         "host": host, "source_module_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        for model, path in (
            ("RandomForest", Path(random_forest.__file__)),
            ("LightGBM", Path(lightgbm.__file__)),
            ("RandomForest", Path(random_forest.__file__)),
        )
    ]
    observations = tmp_path / "measurements.jsonl"
    observations.write_text("".join(json.dumps(x, sort_keys=True) + "\n" for x in rows))
    calib = {
        "schema_version": CALIBRATION_VERSION, "host": host,
        "seconds_per_unit": .001, "measurements": 3, "safety_factor": 2,
        "measurements_sha256": hashlib.sha256(observations.read_bytes()).hexdigest(),
        "learner_sha256": {
            "RandomForest": hashlib.sha256(Path(random_forest.__file__).read_bytes()).hexdigest(),
            "LightGBM": hashlib.sha256(Path(lightgbm.__file__).read_bytes()).hexdigest(),
        },
    }
    calibration = tmp_path / "calibration.json"
    calibration.write_text(json.dumps(calib))
    plan = plan_evaluation_cost(frame, cfg, scope="inter", calibration=calibration,
                                max_estimated_seconds=10**10)
    assert plan["estimated_seconds"] > 0
    # Even an observation with a plausible positive time must fail if its
    # original source-bound bytes no longer match the archived calibration.
    observations.write_text(observations.read_text().replace('"fit_wall_s": 0.1', '"fit_wall_s": 0.2', 1))
    with pytest.raises(CostPlanError, match="observation"):
        plan_evaluation_cost(frame, cfg, scope="inter", calibration=calibration)
    observations.write_text("".join(json.dumps(x, sort_keys=True) + "\n" for x in rows))
    calib["learner_sha256"]["RandomForest"] = "0" * 64
    calibration.write_text(json.dumps(calib))
    with pytest.raises(CostPlanError, match="model implementation"):
        plan_evaluation_cost(frame, cfg, scope="inter", calibration=calibration)


def test_rf_unit_weight_fast_path_matches_original_two_class_scan_bitwise():
    from eeg_ma.random_forest import _feature_gains, _impurity
    rng = np.random.default_rng(456)
    for size in (8, 17, 64):
        X = rng.integers(0, 8, size=(size, 4)).astype(float)
        y = rng.integers(0, 2, size=size)
        y[0], y[-1] = 0, 1
        weights = np.ones(size)
        indices = np.arange(size)
        f = 0
        scanned = _feature_gains(X, y, weights, indices, f, 1,
                                 _impurity(np.bincount(y, minlength=2)))
        assert scanned is not None
        order, sorted_values, valid, gains = scanned
        slow_order = np.argsort(X[:, f], kind="stable")
        assert np.array_equal(order, slow_order)
        sorted_y, sorted_w = y[slow_order], weights[slow_order]
        left_mass = np.cumsum(sorted_w)[:-1]
        right_mass = np.cumsum(sorted_w[::-1])[::-1][1:]
        left_gini, right_gini = np.ones(size - 1), np.ones(size - 1)
        for label in np.unique(sorted_y):
            cw = np.where(sorted_y == label, sorted_w, 0.0)
            left_gini -= (np.cumsum(cw)[:-1] / left_mass) ** 2
            right_gini -= (np.cumsum(cw[::-1])[::-1][1:] / right_mass) ** 2
        np.clip(left_gini, 0, 1, out=left_gini)
        np.clip(right_gini, 0, 1, out=right_gini)
        total = left_mass + right_mass
        original = _impurity(np.bincount(y, minlength=2)) - (
            (left_mass / total) * left_gini + (right_mass / total) * right_gini)
        np.testing.assert_array_equal(gains, original)


def test_benchmark_cannot_write_to_canonical_research_trees():
    from importlib.util import spec_from_file_location, module_from_spec
    from eeg_ma.provenance import ProvenanceError
    root = Path(__file__).resolve().parents[1]
    spec = spec_from_file_location("safe_bench", root / "benchmarks" / "run.py")
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    for destination in (root / "data/features/perf", root / "data/raw/perf",
                        root / "results/perf"):
        with pytest.raises(ProvenanceError):
            module.safe_benchmark_path(destination)



def test_benchmark_refuses_profiler_contaminated_comparison(tmp_path):
    from argparse import Namespace
    from importlib.util import spec_from_file_location, module_from_spec
    root = Path(__file__).resolve().parents[1]
    spec = spec_from_file_location("comparison_proof", root / "benchmarks" / "run.py")
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    before, after = tmp_path / "before", tmp_path / "after"
    before.mkdir()
    after.mkdir()
    report = {"schema_version": module.SCHEMA,
              "host": module.machine(), "preset": "ci",
              "profiling_excluded_from_measured_repeats": False,
              "cases": []}
    for folder in (before, after):
        (folder / "summary.json").write_text(json.dumps(report))
    with pytest.raises(ValueError, match="Profiler-contaminated"):
        module.compare(Namespace(baseline=str(before), candidate=str(after),
                                 output=str(tmp_path / "comparison.json")))
    assert not (tmp_path / "comparison.json").exists()
