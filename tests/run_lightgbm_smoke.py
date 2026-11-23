"""Exercise the real five-classifier CLIs and handwritten boosting workloads.

Synthetic engineering evidence only. No model, environment probe, code
identity, clock, or subprocess is mocked. Existing human results are not read
for model selection and neither frozen A/B protocol is changed.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

import eeg_ma
from eeg_ma.environment import verify_reference_environment
from eeg_ma.extract import extract_all
from eeg_ma.lightgbm import ALGORITHM_VERSION, HandcraftedLightGBMClassifier
from eeg_ma.numerical_validation import fixture_definition, write_synthetic_raw
from eeg_ma.provenance import code_identity, protect_output, read_csv_literal, sha256_file, validate_complete_run
from eeg_ma.random_forest import HandcraftedRandomForestClassifier
from eeg_ma.schema import META_COLUMNS
from eeg_ma.visualization import plot_all_feature_classification


ROOT = Path(__file__).resolve().parents[1]
ROSTER = ["LDA", "RBF-SVM", "KFDA", "RandomForest", "LightGBM"]


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, sort_keys=True,
        indent=2, allow_nan=False) + "\n", encoding="utf-8")


def file_hashes(root):
    return {p.relative_to(root).as_posix(): sha256_file(p)
            for p in sorted(root.rglob("*")) if p.is_file()}


def run(out_dir):
    require(Path(eeg_ma.__file__).resolve().is_relative_to(ROOT / "src/eeg_ma"),
        "Install this checkout in the reference interpreter before its smoke")
    environment = verify_reference_environment()
    before_code = code_identity()
    protocols = {name: sha256_file(ROOT / "configs" / name)
                 for name in ("confirmatory_protocol.json", "phase_control_plan.json")}
    target = protect_output(out_dir)
    target.mkdir(parents=True, exist_ok=False)
    (target / "inputs").mkdir()
    (target / "logs").mkdir()
    started = time.monotonic()
    report = {
        "schema_version": "eeg-lightgbm-smoke-1", "status": "running",
        "scope": "SYNTHETIC engineering evidence only; no new human experiment or EEG accuracy claim",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "algorithm_version": ALGORITHM_VERSION, "commands": [],
        "code": {k: before_code[k] for k in ("commit", "dirty", "working_source_sha256")},
        "validation_script_sha256": sha256_file(__file__),
        "reference_environment_id": environment["environment_id"],
        "reference_lock_id": environment["lock_id"],
        "runtime_or_clock_overrides": False, "n_human_participants": 0,
    }
    write_json(target / "reference_environment.json", environment)

    def cli(name, script, arguments, expected_returncode=0):
        command = [sys.executable, str(ROOT / "scripts" / script), *map(str, arguments)]
        result = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, timeout=240)
        stdout, stderr = target / "logs" / f"{name}.stdout.txt", target / "logs" / f"{name}.stderr.txt"
        stdout.write_text(result.stdout, encoding="utf-8")
        stderr.write_text(result.stderr, encoding="utf-8")
        report["commands"].append({"name": name, "argv": command, "returncode": result.returncode,
            "expected_returncode": expected_returncode, "stdout": str(stdout.relative_to(target)),
            "stderr": str(stderr.relative_to(target))})
        require(result.returncode == expected_returncode,
            f"{name} returned {result.returncode}; inspect {stdout} and {stderr}")
        return result

    try:
        definition = fixture_definition()
        raw = write_synthetic_raw(target / "raw", definition)
        frame, qa = extract_all(raw, definition["analysis"], target / "source_features")
        require(qa.qa_pass.all(), "Synthetic raw inputs must pass the shared QA")
        names = ["BP__FP1__alpha", "BP__FP2__alpha", "COH__FP1__FP2__alpha"]
        frame = frame[[*META_COLUMNS, *names]].copy()
        feature_path = target / "inputs/features.csv"
        frame.to_csv(feature_path, index=False, encoding="utf-8")
        cfg = deepcopy(definition["analysis"])
        cfg["classifiers"] = ROSTER
        cfg["random_forest"] = {"parameters": {"n_estimators": 8, "max_depth": 4,
            "random_state": 20261119, "oob_score": True, "n_jobs": 1},
            "grid": {"min_samples_leaf": [1, 2]}}
        cfg["lightgbm"] = {"parameters": {"n_estimators": 8, "num_leaves": 3,
            "max_depth": 3, "max_bin": 7, "min_data_in_leaf": 2, "learning_rate": 0.1,
            "random_state": 20261119, "n_jobs": 1}, "grid": {"lambda_l2": [1.0, 3.0]}}
        config_path = target / "inputs/config.json"
        write_json(config_path, cfg)
        report["inputs"] = {p.name: sha256_file(p) for p in (feature_path, config_path)}
        require(frame.subject_id.nunique() == 3 and len(frame) == 48
                and len(frame[["subject_id", "session_id"]].drop_duplicates()) == 6,
                "Expected three synthetic people, two sessions each, and 48 epochs")

        runs = {}
        for scope, script, n_folds in (("intra", "03_intra_subject.py", 6),
                                      ("inter", "04_inter_subject.py", 3)):
            destination = target / "evaluation" / scope
            args = ("--features", feature_path, "--config", config_path, "--out-dir", destination,
                    "--classifiers", *ROSTER)
            cli(scope, script, args)
            manifest = validate_complete_run(destination)
            require(manifest["identity"]["identity_version"] == "evaluation-3",
                    "Explicit classifier runs must retain their versioned roster contract")
            predictions = read_csv_literal(destination / f"{scope}_predictions.csv")
            folds = read_csv_literal(destination / f"{scope}_folds.csv")
            require(set(predictions.classifier) == set(ROSTER) and len(predictions) == 48 * 5,
                    f"{scope} did not predict every epoch with all five classifiers")
            require(len(folds) == n_folds * 5, f"{scope} outer fold coverage differs")
            for learner, cls in (("lightgbm", HandcraftedLightGBMClassifier),
                                 ("random_forest", HandcraftedRandomForestClassifier)):
                audit = json.loads((destination / f"{learner}_audit.json").read_text())
                require(len(audit["folds"]) == n_folds, f"{scope} {learner} evidence incomplete")
                for record in audit["folds"]:
                    restored = cls.from_dict(record["model"])
                    require(restored.to_dict() == record["model"],
                            f"{learner} JSON roundtrip changed fitted evidence")
                    require(len(record["candidates"]) == 2, "Bounded inner search was not retained")
            previous = file_hashes(destination)
            cli(f"{scope}-resume", script, (*args, "--resume"))
            require(file_hashes(destination) == previous, "Validated resume changed existing artifacts")
            refused = cli(f"{scope}-refuse-overwrite", script, args, expected_returncode=1)
            require("Run already exists" in refused.stderr,
                    "Second run was not refused for the expected immutable-output reason")
            require(file_hashes(destination) == previous, "Refused overwrite changed existing artifacts")
            runs[scope] = {"run_id": manifest["run_id"], "n_predictions": len(predictions),
                           "n_folds": len(folds), "n_lightgbm_models": n_folds,
                           "n_random_forest_models": n_folds}

        figures = plot_all_feature_classification(
            read_csv_literal(target / "evaluation/intra/intra_overall_summary.csv"),
            read_csv_literal(target / "evaluation/inter/inter_summary.csv"), target / "figures")
        require(len(figures) == 4 and all(p.is_file() and p.stat().st_size > 0 for p in figures),
                "Two five-classifier figures (PNG/SVG) were not generated")

        # Dense EEG-sized dimensional workload, with serial/thread equivalence.
        rng = np.random.default_rng(7319)
        X = rng.normal(size=(300, 126))
        y = (X[:, 0] + 0.6 * X[:, 1] - 0.3 * X[:, 2] > 0).astype(int)
        np.savez(target / "inputs/dense_workload.npz", X=X, y=y)
        params = {"n_estimators": 16, "num_leaves": 7, "max_depth": 5,
            "min_data_in_leaf": 5, "max_bin": 31, "feature_fraction": 0.75,
            "bagging_fraction": 0.8, "random_state": 7319}
        tick = time.monotonic()
        dense = HandcraftedLightGBMClassifier(**params, n_jobs=2).fit(X, y)
        elapsed = time.monotonic() - tick
        serial = HandcraftedLightGBMClassifier(**params, n_jobs=1).fit(X, y)
        np.testing.assert_array_equal(dense.decision_function(X), serial.decision_function(X))
        snapshot = dense.to_dict()
        write_json(target / "dense_model.json", snapshot)
        restored = HandcraftedLightGBMClassifier.from_dict(snapshot)
        np.testing.assert_array_equal(restored.decision_function(X), dense.decision_function(X))
        np.testing.assert_array_equal(restored.predict_proba(X), dense.predict_proba(X))
        require(restored.to_dict() == snapshot, "Dense model JSON roundtrip changed evidence")
        require(np.isfinite(dense.predict_proba(X)).all(), "Nonfinite boosting probabilities")

        # Mutually exclusive sparse columns allow real EFB compression. GOSS
        # acts on the gradients of each actual boosting prefix, without warmup.
        sparse = np.zeros((240, 12), dtype=np.float64)
        group_a, group_b = rng.integers(0, 6, 240), rng.integers(6, 12, 240)
        sparse[np.arange(240), group_a] = rng.uniform(-3, 3, 240)
        sparse[np.arange(240), group_b] = rng.uniform(1, 3, 240)
        labels = ((group_a < 2) | (group_b == 7)).astype(int)
        heldout = sparse[:24].copy()
        heldout[:, :4] = rng.normal(size=(24, 4))  # unseen within-bundle overlap
        np.savez(target / "inputs/goss_efb_workload.npz", X=sparse, y=labels, heldout=heldout)
        sparse_params = {"n_estimators": 12, "num_leaves": 5, "max_depth": 4,
            "min_data_in_leaf": 2, "max_bin": 15, "data_sample_strategy": "goss",
            "top_rate": 0.2, "other_rate": 0.25, "random_state": 991}
        bundled = HandcraftedLightGBMClassifier(**sparse_params, enable_bundle=True).fit(sparse, labels)
        plain = HandcraftedLightGBMClassifier(**sparse_params, enable_bundle=False).fit(sparse, labels)
        require(bundled.bundle_layout_.n_bundles_ < sparse.shape[1]
                and plain.bundle_layout_.n_bundles_ == sparse.shape[1],
                "Sparse EFB workload did not exercise genuine bundle compression")
        heldout_bins = bundled.bin_mapper_.transform(heldout)
        zero_bins = bundled.bin_mapper_.zero_bins_
        require(any(np.any(np.count_nonzero(heldout_bins[:, group] != zero_bins[list(group)], axis=1) > 1)
                    for group in bundled.bundle_layout_.bundles_),
                "Held-out EFB workload must contain actual overlapping nondefault bins")
        require(bundled.n_estimators_ > 0 and dense.n_estimators_ > 0,
                "The workloads must exercise actual split-bearing boosting rounds")
        np.testing.assert_allclose(bundled.decision_function(heldout), plain.decision_function(heldout),
                                   rtol=1e-12, atol=1e-12)
        np.testing.assert_array_equal(bundled.predict(heldout), plain.predict(heldout))
        write_json(target / "goss_efb_model.json", bundled.to_dict())
        write_json(target / "goss_unbundled_model.json", plain.to_dict())
        replay = HandcraftedLightGBMClassifier.from_dict(bundled.to_dict())
        np.testing.assert_array_equal(replay.predict_proba(heldout), bundled.predict_proba(heldout))
        require(all(sha256_file(ROOT / "configs" / name) == digest for name, digest in protocols.items()),
                "The LightGBM workflow changed a frozen A/B protocol")
        require(code_identity() == before_code, "Source changed during the actual reference smoke")
        report.update(status="passed", reference_environment_status="matched",
            n_synthetic_participants=3, n_sessions=6, n_epochs=48, classifiers=ROSTER, runs=runs,
            manual_core_dependencies=["Python standard library", "NumPy"],
            selection="shared_training_only_LDA_SFS_then_manual_RF_and_LightGBM_inner_search",
            immutable_resume_and_no_overwrite="verified", safe_json_replay="verified",
            original_protocols_unchanged=True,
            workload={"dense": {"n_samples": 300, "n_features": 126, "requested_rounds": 16,
                "fitted_rounds": dense.n_estimators_, "n_jobs": 2, "serial_thread_equality": True,
                "fit_seconds_observed_not_a_performance_guarantee": elapsed},
                "sparse": {"n_samples": 240, "n_features": 12, "requested_rounds": 12,
                    "fitted_rounds": bundled.n_estimators_, "n_bundles": bundled.bundle_layout_.n_bundles_,
                    "sampling": "goss", "bundled_plain_heldout_equivalence": True,
                    "unseen_heldout_overlap_supported": True}},
            figures={p.relative_to(target).as_posix(): sha256_file(p) for p in figures})
    except Exception as exc:
        report.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        report["duration_seconds"] = time.monotonic() - started
        report["finished_at"] = datetime.now(timezone.utc).isoformat()
        write_json(target / "validation_report.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", required=True, type=Path)
    args = parser.parse_args()
    try:
        report = run(args.out_dir)
    except (OSError, ValueError, RuntimeError, AssertionError, subprocess.SubprocessError) as exc:
        print(json.dumps({"status": "failed", "error": str(exc)}, ensure_ascii=False))
        return 1
    print(json.dumps({key: report[key] for key in ("status", "scope", "reference_environment_status",
        "n_synthetic_participants", "n_sessions", "n_epochs", "classifiers", "runs", "workload")},
        ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
