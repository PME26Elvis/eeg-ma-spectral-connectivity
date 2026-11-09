"""Exercise actual fixed-model CLIs from a clean, verified reference checkout.

Only EEG values and study attestations are synthetic. This utility never
replaces code identities, the environment verifier, the clock, or subprocess
environment variables. Complete prospective scoring needs real elapsed
acquisition time; the full scoring lifecycle is covered separately by the
deterministic synthetic tests in test_external_workflow.py.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import subprocess
import sys
import time

import eeg_ma
from eeg_ma.environment import verify_reference_environment
from eeg_ma.provenance import code_identity, protect_output, read_csv_literal, sha256_file

from fixed_model_fixtures import make_feature_table, write_development_fixture, write_json


ROOT = Path(__file__).resolve().parents[1]
FIXED_CLI = ROOT / "scripts/21_fixed_external.py"
PROTOCOL_CLI = ROOT / "scripts/16_protocol.py"
SCOPE = "SYNTHETIC engineering evidence only; no human study or acquisition approval"


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def run(out_dir):
    package = Path(eeg_ma.__file__).resolve()
    require(package.is_relative_to(ROOT / "src/eeg_ma"),
            "Install this checkout in the reference interpreter; another editable checkout is active")
    status = subprocess.run(["git", "-C", str(ROOT), "status", "--porcelain=v1", "--untracked-files=all"],
                            capture_output=True, text=True, check=True)
    require(not status.stdout.strip(), "The reference subprocess smoke test requires a fully clean checkout")
    code = code_identity()
    require(code["dirty"] is False and re.fullmatch(r"[0-9a-f]{40}", code["commit"]),
            "A real clean Git commit is required")
    reference = verify_reference_environment()
    target = protect_output(out_dir)
    target.mkdir(parents=True, exist_ok=False)
    (target / "logs").mkdir()
    started = time.monotonic()
    report = {"schema_version": "eeg-external-reference-smoke-1", "status": "running", "scope": SCOPE,
              "started_at": datetime.now(timezone.utc).isoformat(), "commands": [],
              "code": {key: code[key] for key in ("commit", "dirty", "working_source_sha256")},
              "reference_environment_id": reference["environment_id"], "reference_lock_id": reference["lock_id"],
              "runtime_or_clock_overrides": False}
    write_json(target / "reference_environment.json", reference)

    def cli(name, script, *arguments, expected_returncode=0):
        command = [sys.executable, str(script), *map(str, arguments)]
        # Deliberately inherit the operator's verified environment unchanged.
        result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, timeout=300)
        stdout = target / "logs" / f"{name}.stdout.json"
        stderr = target / "logs" / f"{name}.stderr.txt"
        stdout.write_text(result.stdout, encoding="utf-8")
        stderr.write_text(result.stderr, encoding="utf-8")
        report["commands"].append({"name": name, "argv": command, "returncode": result.returncode,
            "expected_returncode": expected_returncode, "stdout": str(stdout.relative_to(target)),
            "stderr": str(stderr.relative_to(target))})
        require(result.returncode == expected_returncode,
                f"{name} returned {result.returncode}; inspect {stdout} and {stderr}")
        return json.loads(result.stdout)

    try:
        source, spec, development = write_development_fixture(target / "development")
        # Timestamp fixture generation truthfully, before fitting and locking.
        generated_at = datetime.now(timezone.utc).isoformat()
        spec["approval"]["approved_at"] = generated_at
        spec["study"]["cohort_id"] = "SYNTHETIC-reference-external"
        evidence = spec["freeze"]["artifacts"]["acquisition_evidence"]
        evidence_path = source.parent / evidence["path"]
        acquisition = read_json(evidence_path)
        acquisition["verified_at"] = generated_at
        write_json(evidence_path, acquisition)
        evidence["sha256"] = sha256_file(evidence_path)
        write_json(source, spec)

        fitted = cli("fit-development", FIXED_CLI, "fit-development", "--protocol", source,
                     "--out-dir", target / "models")
        bundle = read_json(target / "models/bundle_manifest.json")
        require(fitted["bundle_id"] == bundle["bundle_id"], "Fit CLI and exported bundle differ")
        require(bundle["execution_scope"] == "reference_development_fit", "Fit did not use strict reference mode")
        require(bundle["code"] == code, "Model fit used different source code")
        for role in ("environment", "primary_model", "reference_model", "primary_training_manifest", "reference_training_manifest"):
            entry = bundle["artifacts"][role]
            path = target / "models" / entry["path"]
            require(sha256_file(path) == entry["sha256"], f"Exported artifact hash differs: {role}")
            spec["freeze"]["artifacts"][role] = {"path": str(path), "sha256": entry["sha256"]}
        write_json(source, spec)

        locked = cli("lock", PROTOCOL_CLI, "lock", "--protocol", source, "--out-dir", target / "lock")
        require(locked["code"] == code, "Protocol lock used different source code")
        people = [f"SYNTHETIC-external-{number:03d}" for number in range(94)]
        cohort = {"schema_version": "eeg-confirmatory-cohort-1", "cohort_id": spec["study"]["cohort_id"],
                  "protocol_id": spec["protocols"]["A"]["protocol_id"], "outcomes_accessed": False,
                  "subject_ids": people}
        write_json(target / "cohort.json", cohort)
        external = make_feature_table(people[:3], cohort["cohort_id"], sessions=1, epochs_per_class=30,
            seed=2311, protocol_id=cohort["protocol_id"], protocol_version=spec["protocols"]["A"]["protocol_version"])
        external.to_csv(target / "external_features.csv", index=False)
        prepared = cli("prepare-inputs", FIXED_CLI, "prepare-inputs", "--features", target / "external_features.csv",
            "--out-dir", target / "prepared", "--lock-dir", target / "lock", "--cohort", target / "cohort.json")
        predicted = cli("predict-external", FIXED_CLI, "predict-external", "--inputs", target / "prepared/inputs",
            "--out-dir", target / "predictions", "--lock-dir", target / "lock", "--cohort", target / "cohort.json")
        verified = cli("verify-predictions", FIXED_CLI, "verify-predictions", "--predictions", target / "predictions",
            "--lock-dir", target / "lock")
        require(verified["prediction_id"] == predicted["prediction_id"], "Prediction archive identity changed")
        manifest = read_json(target / "predictions/prediction_manifest.json")
        require(manifest["code"] == code, "Prediction used different source code")
        require(manifest["environment"]["lock_id"] == reference["lock_id"], "Inference used a different reference lock")
        inputs = read_csv_literal(target / "prepared/inputs/features.csv")
        predictions = read_csv_literal(target / "predictions/predictions.csv")
        require(not {"true_label", "class_label", "class_name", "stage"} & set(inputs.columns),
                "Inference inputs contain labels")
        require(prepared["n_epochs"] == len(external) == len(inputs), "Prepared epoch coverage changed")
        require(len(predictions) == 2 * len(inputs) and not predictions.duplicated(["model_role", "epoch_id"]).any(),
                "Both models must cover every external epoch exactly once")
        require(set(predictions.model_role) == {"primary", "reference"} and predictions.predicted_label.isin([0, 1]).all(),
                "Prediction model roles or binary outputs differ")
        require(set(predictions.subject_id) == set(people[:3])
                and not set(predictions.subject_id) & set(development.subject_id), "Subject independence changed")

        # Do not invent completed sessions in the few seconds since this lock.
        # A truthful pending roster must block scoring before any labels open.
        recorded_at = datetime.now(timezone.utc).isoformat()
        pending = {"schema_version": "eeg-enrollment-1", "cohort_id": cohort["cohort_id"], "outcomes_accessed": False,
            "participants": [{"subject_id": subject, "recruited_at": recorded_at, "population_eligible": True,
                "disposition": "pending" if index < 3 else "lost_to_followup",
                "closed_at": None if index < 3 else recorded_at, "sessions": []}
                for index, subject in enumerate(people)]}
        write_json(target / "enrollment_pending.json", pending)
        blocked = cli("score-pending-blocked", FIXED_CLI, "score-external", "--predictions", target / "predictions",
            "--labels", target / "intentionally-absent-labels", "--out-dir", target / "must-not-score",
            "--lock-dir", target / "lock", "--cohort", target / "cohort.json", "--enrollment", target / "enrollment_pending.json",
            expected_returncode=2)
        require(blocked.get("status") == "blocked" and "fully closed cumulative enrollment roster" in blocked.get("error", ""),
                "Scoring did not block at the enrollment boundary before attempting label access")
        require(not (target / "must-not-score").exists(), "Blocked scoring published an output")
        verified_lock = cli("verify-lock", PROTOCOL_CLI, "verify", "--lock-dir", target / "lock")
        require(verified_lock["lock_id"] == locked["lock_id"] and verified_lock["outcome_access_events"] == 0,
                "Pending enrollment must not record outcome access")
        require(code_identity() == code, "Source code changed during reference verification")
        minimum_seconds = (2 * spec["eligibility"]["planned_rest_seconds_per_block"]
            + 2 * spec["eligibility"]["max_epochs_per_class"] * spec["protocols"]["A"]["calculation_seconds"])
        report.update(status="passed", bundle_id=bundle["bundle_id"], protocol_lock_id=locked["lock_id"],
            locked_at=locked["locked_at"], prediction_id=manifest["prediction_id"], input_id=prepared["input_id"],
            model_ids=manifest["model_ids"], model_sha256=manifest["model_sha256"], n_development_subjects=3,
            n_cohort_subjects=len(people), n_feature_subjects=3, n_epochs=len(inputs), n_predictions=len(predictions),
            feature_candidates={"primary": 21, "reference": 168}, pending_score_gate="blocked_before_label_access_without_event",
            scoring_evidence={"completed_in_this_real_clock_run": False,
                "reason": "Post-lock prospective acquisition has not elapsed; no session times were fabricated",
                "minimum_two_quota_task_session_seconds": minimum_seconds,
                "separate_full_lifecycle_test": "tests/test_external_workflow.py::test_cli_fit_prepare_predict_score_and_archive_verification",
                "separate_test_scope": "real fitted numerical models with explicitly synthetic study/code/time attestations"})
    except Exception as exc:
        report.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        report["finished_at"] = datetime.now(timezone.utc).isoformat()
        report["duration_seconds"] = time.monotonic() - started
        write_json(target / "validation_report.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", required=True)
    arguments = parser.parse_args()
    try:
        report = run(arguments.out_dir)
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as exc:
        print(json.dumps({"status": "failed", "error": str(exc)}, ensure_ascii=False))
        return 1
    print(json.dumps({key: report[key] for key in ("status", "scope", "bundle_id", "protocol_lock_id", "prediction_id",
        "n_epochs", "n_predictions", "pending_score_gate")}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
