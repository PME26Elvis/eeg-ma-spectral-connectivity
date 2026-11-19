"""Run the actual four-classifier CLIs on synthetic multisession EEG inputs.

No estimator, environment probe, code identity, clock, or subprocess is mocked.
The fixture is engineering evidence, not a new human experiment or accuracy
claim. Run this script with the repository's verified reference interpreter.
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
from eeg_ma.numerical_validation import fixture_definition, write_synthetic_raw
from eeg_ma.provenance import code_identity, protect_output, read_csv_literal, sha256_file, validate_complete_run
from eeg_ma.random_forest import ALGORITHM_VERSION, HandcraftedRandomForestClassifier
from eeg_ma.schema import META_COLUMNS
from eeg_ma.visualization import plot_all_feature_classification


ROOT = Path(__file__).resolve().parents[1]
ROSTER = ["LDA", "RBF-SVM", "KFDA", "RandomForest"]


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
        "schema_version": "eeg-random-forest-smoke-1", "status": "running",
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
        result = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, timeout=180)
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
        require(qa.qa_pass.all(), "Synthetic raw input must pass actual shared QA")
        names = ["BP__FP1__alpha", "BP__FP2__alpha", "COH__FP1__FP2__alpha"]
        frame = frame[[*META_COLUMNS, *names]].copy()
        feature_path = target / "inputs/features.csv"
        frame.to_csv(feature_path, index=False, encoding="utf-8")
        cfg = deepcopy(definition["analysis"])
        cfg["classifiers"] = ROSTER
        cfg["random_forest"] = {"parameters": {"n_estimators": 8, "max_depth": 4,
            "random_state": 20261119, "oob_score": True, "n_jobs": 1},
            "grid": {"min_samples_leaf": [1, 2]}}
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
            require(set(predictions.classifier) == set(ROSTER) and len(predictions) == 48 * 4,
                    f"{scope} did not predict every epoch with all four classifiers")
            require(len(folds) == n_folds * 4, f"{scope} outer fold coverage differs")
            audit = json.loads((destination / "random_forest_audit.json").read_text())
            require(len(audit["folds"]) == n_folds, f"{scope} forest evidence incomplete")
            for record in audit["folds"]:
                restored = HandcraftedRandomForestClassifier.from_dict(record["model"])
                require(restored.to_dict() == record["model"], "Forest JSON roundtrip changed fitted evidence")
                require(len(record["candidates"]) == 2, "The bounded inner search was not retained")
            # Resume must retain all bytes, rather than silently fitting again.
            previous = file_hashes(destination)
            cli(f"{scope}-resume", script, (*args, "--resume"))
            require(file_hashes(destination) == previous, "Validated resume changed existing artifacts")
            refused = cli(f"{scope}-refuse-overwrite", script, args, expected_returncode=1)
            require("Run already exists" in refused.stderr,
                    "Second run was not refused for the expected immutable-output reason")
            require(file_hashes(destination) == previous, "Refused overwrite changed existing artifacts")
            runs[scope] = {"run_id": manifest["run_id"], "n_predictions": len(predictions),
                           "n_folds": len(folds), "n_forest_models": n_folds}

        intra = read_csv_literal(target / "evaluation/intra/intra_overall_summary.csv")
        inter = read_csv_literal(target / "evaluation/inter/inter_summary.csv")
        figures = plot_all_feature_classification(intra, inter, target / "figures")
        require(len(figures) == 4 and all(p.is_file() and p.stat().st_size > 0 for p in figures),
                "Two four-classifier figures (PNG/SVG) were not generated")

        # Exercise a nontrivial feature dimension and threaded forest directly;
        # this is a bounded engineering workload, without an accuracy threshold.
        rng = np.random.default_rng(8123)
        X = rng.normal(size=(300, 126))
        y = ((X[:, 0] > 0) ^ (X[:, 1] > 0)).astype(int)
        np.savez(target / "inputs/forest_workload.npz", X=X, y=y)
        tick = time.monotonic()
        forest = HandcraftedRandomForestClassifier(n_estimators=32, max_depth=10,
            min_samples_leaf=2, random_state=8123, oob_score=True, n_jobs=2).fit(X, y)
        elapsed = time.monotonic() - tick
        snapshot = forest.to_dict()
        write_json(target / "forest_model.json", snapshot)
        restored = HandcraftedRandomForestClassifier.from_dict(snapshot)
        np.testing.assert_array_equal(restored.predict_proba(X), forest.predict_proba(X))
        np.testing.assert_array_equal(restored.predict(X), forest.predict(X))
        require(np.isfinite(forest.predict_proba(X)).all(), "Nonfinite forest probabilities")
        require(all(sha256_file(ROOT / "configs" / name) == digest for name, digest in protocols.items()),
                "The forest workflow changed a frozen A/B protocol")
        require(code_identity() == before_code, "Source changed during the actual reference smoke")
        report.update(status="passed", reference_environment_status="matched",
            n_synthetic_participants=3, n_sessions=6, n_epochs=48, classifiers=ROSTER, runs=runs,
            manual_core_dependencies=["Python standard library", "NumPy"],
            selection="shared_training_only_LDA_SFS_then_manual_RF_inner_search",
            immutable_resume_and_no_overwrite="verified", safe_json_replay="verified",
            original_protocols_unchanged=True,
            workload={"n_samples": 300, "n_features": 126, "n_estimators": 32,
                      "n_jobs": 2, "fit_seconds_observed_not_a_performance_guarantee": elapsed},
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
