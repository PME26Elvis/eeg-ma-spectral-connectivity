"""Bounded synthetic raw-to-report regression, never a private EEG reproduction.

The fixture retains the historical 8 raw/7 analysis channels and six bands,
then evaluates a declared tiny feature subset and search grid. It exercises
the real QA/extraction/grouped-CV/publication code;
its accuracies are software checks, not research or external-model estimates.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import re
import shutil

import numpy as np
import pandas as pd

from .artifacts import new_output_directory
from .experiment_runs import prepare_matrix
from .extensions import extract_all_extension_features, merge_baseline_and_extensions
from .extract import extract_all, feature_columns
from .provenance import (code_identity, fingerprint, read_csv_literal,
                         resolve_report_root, sha256_file, validate_complete_run)


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REFERENCE = ROOT / "tests/fixtures/numerical/reference.json"
FIXTURE_VERSION = "synthetic-raw-grouped-cv-1"
REPORT_VERSION = "eeg-numerical-validation-1"
ROW_KEYS = ["subject_id", "session_id", "comparison", "stage", "epoch_index", "attempt_index"]


class NumericalValidationError(ValueError):
    """Numerical evidence is incomplete, inconsistent, or outside tolerance."""


def fixture_definition() -> dict:
    """Versioned, outcome-independent inputs; deliberately not the study protocol."""
    return {
        "fixture_version": FIXTURE_VERSION, "purpose": "synthetic_grouped_cv_regression_only",
        "seed": 1092026, "subjects": ["001", "1", "NA"], "sessions_per_subject": 2,
        "epochs_per_class_per_session": 4, "recording_seconds": 44,
        "evaluation_features": ["BP__FP1__alpha", "BP__FP2__alpha", "COH__FP1__FP2__alpha", "PLV__FP1__FP2__alpha"],
        "baseline_sentinels": ["BP__FP1__alpha", "BP__FP2__alpha", "BP__Fz__theta", "COH__FP1__FP2__alpha", "COH__F3__F4__beta_low", "COH__F7__F8__gamma"],
        "extension_sentinels": ["ASYM_LOGRATIO__FP1__FP2__alpha", "ASYM_LOGRATIO__F3__F4__theta", "PLV__FP1__FP2__alpha", "PLV__F3__F4__beta_low", "PLV__F7__F8__gamma"],
        "analysis": {
            "sampling_rate_hz": 500, "epoch_seconds": 5.0,
            "raw_channels": ["FP1", "FP2", "F7", "F3", "Fz", "F4", "F8", "Cz"],
            "analysis_channels": ["FP1", "FP2", "F7", "F3", "Fz", "F4", "F8"],
            "bandpass": {"low_hz": 2.0, "high_hz": 50.0, "order": 4}, "notch_hz": None,
            "bands_hz": {"delta": [1.0, 4.0], "theta": [4.0, 8.0], "alpha": [8.0, 13.0],
                         "beta_low": [13.0, 20.0], "beta_high": [20.0, 30.0], "gamma": [30.0, 45.0]},
            "band_power": {"mode": "lab_dft_squared_sum"},
            "coherence": {"mode": "lab_segmented_fft_msc", "segment_seconds": 1.0, "band_reduce": "mean"},
            "comparisons": [{"name": "rest1_vs_type1", "rest_marker": 20, "task_marker": 11, "task_type": "Type1"}],
            "expected_epochs_per_class": 4, "feature_set": "all",
            "sfs": {"selector": "LDA", "max_features": 1, "min_improvement": 0.0},
            "cv": {"intra_outer_folds": 2, "intra_inner_folds": 2, "random_state": 1092026},
            "rbf_grid": {"C": [1.0], "gamma_values": [0.5]}, "n_jobs": 1,
            "ingestion": {"marker_policy": "hardware_required", "clock_policy": "session_start",
                          "timestamp_policy": "require_utc", "sample_period_tolerance_s": 2e-6,
                          "clock_tolerance_s": 0.005, "flat_std_epsilon": 1e-15},
        },
        "extensions": {"asymmetry": {"mode": "log_ratio", "epsilon": 1e-12, "pairs": [["FP1", "FP2"], ["F3", "F4"], ["F7", "F8"]]},
                       "plv": {"filter_order": 4}},
        "experiments": [{"name": "baseline", "feature_set": "all"}, {"name": "phase", "feature_set": "plv"}],
    }


def _write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _parse_json(value: str) -> dict:
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise NumericalValidationError(f"Duplicate JSON field: {key}")
            result[key] = value
        return result
    try:
        result = json.loads(value, object_pairs_hook=pairs,
                            parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))
    except (TypeError, ValueError) as exc:
        raise NumericalValidationError("Invalid numerical JSON evidence") from exc
    if not isinstance(result, dict):
        raise NumericalValidationError("Numerical evidence must be a JSON object")
    return result


def _read_json(path: Path) -> dict:
    return _parse_json(path.read_text(encoding="utf-8"))


def _raw_inventory(raw: Path) -> dict:
    if not raw.is_dir() or raw.is_symlink():
        raise NumericalValidationError("Complete ordinary baseline raw directory is required")
    files = {}
    for path in sorted(raw.rglob("*")):
        if path.is_symlink():
            raise NumericalValidationError("Synthetic raw evidence cannot contain symlinks")
        if path.is_file():
            files[path.relative_to(raw).as_posix()] = sha256_file(path)
    if not files:
        raise NumericalValidationError("Synthetic raw evidence is empty")
    return files


def write_synthetic_raw(destination: str | Path, definition: dict | None = None) -> Path:
    """Generate six explicit acquisitions, with repeated sessions per fake subject."""
    definition = definition or fixture_definition()
    destination = Path(destination)
    with new_output_directory(destination) as stage:
        cfg = definition["analysis"]
        fs = cfg["sampling_rate_hz"]
        t = np.arange(fs * definition["recording_seconds"], dtype=float) / fs
        for subject_index, subject in enumerate(definition["subjects"]):
            for session_index in range(definition["sessions_per_subject"]):
                session = f"session-{session_index + 1}"
                folder = stage / f"person-{subject_index + 1}-{session}"
                folder.mkdir()
                rng = np.random.default_rng(definition["seed"] + 100 * subject_index + session_index)
                transition = 0.5 * (1 + np.tanh((t - 21) * 2))
                modulation = 1 + .08 * np.sin(2 * np.pi * .07 * t + subject_index + session_index)
                alpha_phase = 2 * np.pi * (10 + .02 * subject_index) * t + .03 * np.sin(t)
                lag = .3 + transition * (1.9 * np.sin(2 * np.pi * .63 * t + session_index))
                scale = 1 + .09 * subject_index + .04 * session_index
                first = scale * modulation * (35 * (1 + .55 * transition) * np.sin(alpha_phase)
                    + 13 * np.sin(2 * np.pi * 17 * t + .2)) + rng.normal(0, 1.2, len(t))
                second = scale * (31 * np.sin(alpha_phase + lag)
                    + (12 + 5 * transition) * np.sin(2 * np.pi * 17.3 * t + .5)) + rng.normal(0, 1.1, len(t))
                raw = pd.DataFrame({"sample_index": np.arange(len(t)), "elapsed_s": t, "FP1": first, "FP2": second})
                for channel_index, channel in enumerate(cfg["raw_channels"][2:], start=2):
                    raw[channel] = (scale * (22 * np.sin(alpha_phase + channel_index * .27 + transition * .15 * channel_index * np.sin(t))
                        + (9 + channel_index) * np.sin(2 * np.pi * (16 + .31 * channel_index) * t))
                        + rng.normal(0, 1.0 + .05 * channel_index, len(t)))
                raw.to_csv(folder / "eeg_raw.csv", index=False, float_format="%.12e", lineterminator="\n")
                origin = datetime(2026, 1, 1, tzinfo=timezone.utc) + timedelta(days=subject_index, hours=session_index)
                def utc(seconds):
                    return (origin + timedelta(seconds=float(seconds))).isoformat()
                events = [{"session_id": session, "subject_id": subject, "stage": "Session", "phase": "SessionStart",
                    "task_type": "", "attempt_index": 0, "utc_timestamp": utc(0), "elapsed_ms": 0,
                    "marker_code": 1, "marker_sent": True, "marker_message": "synthetic"}]
                rests, trials = [], []
                for index in range(definition["epochs_per_class_per_session"]):
                    for task, onset, code in ((False, 5 * index, 20), (True, 22 + 5 * index, 11)):
                        events.append({"session_id": session, "subject_id": subject, "stage": "Type1" if task else "Rest1",
                            "phase": "Calculation" if task else "Rest", "task_type": "Type1" if task else "",
                            "attempt_index": index + 1 if task else 0, "utc_timestamp": utc(onset), "elapsed_ms": onset * 1000,
                            "marker_code": code, "marker_sent": True, "marker_message": "synthetic"})
                        if task:
                            trials.append({"session_id": session, "subject_id": subject, "task_type": "Type1", "attempt_index": index + 1,
                                "response_correct": True, "analysis_eligible": True, "calculation_onset": utc(onset)})
                        else:
                            rests.append({"session_id": session, "subject_id": subject, "rest_stage": "Rest1", "rest_epoch_index": index + 1,
                                "onset_utc": utc(onset), "epoch_seconds": 5, "marker_code": 20, "marker_sent": True, "marker_message": "synthetic"})
                for filename, rows in (("events.csv", sorted(events, key=lambda row: row["elapsed_ms"])),
                                       ("trials.csv", trials), ("rest_epochs.csv", rests)):
                    pd.DataFrame(rows).to_csv(folder / filename, index=False, lineterminator="\n")
                (folder / "session_info.txt").write_text(f"Subject: {subject}\nSampling rate: {fs} Hz\n", encoding="utf-8")
                _write_json(folder / "session_metadata.json", {"schema_version": "2.0", "cohort_id": "synthetic-numerical-v1",
                    "subject_id": subject, "session_id": session, "units": "uV", "reference": "synthetic-common-reference",
                    "protocol_id": "synthetic-bounded-grouped-regression", "protocol_version": "1", "channel_order": cfg["raw_channels"]})
    return destination


def _keys(frame: pd.DataFrame, columns: list[str]) -> list[list]:
    return json.loads(frame[columns].to_json(orient="values"))


def _observations(baseline, extension, inventory, report_root) -> tuple[dict, dict]:
    baseline = baseline.sort_values(ROW_KEYS).reset_index(drop=True)
    extension = extension.sort_values(ROW_KEYS).reset_index(drop=True)
    baseline_columns = feature_columns(baseline, "all")
    extension_columns = [c for c in extension if c.startswith(("ASYM_", "PLV__"))]
    definition = fixture_definition()
    fold_frames, prediction_frames = [], []
    for experiment, root in sorted(inventory.items()):
        for frames, name in ((fold_frames, "folds"), (prediction_frames, "predictions")):
            table = read_csv_literal(root / "inter" / f"inter_{name}.csv")
            table.insert(0, "experiment", experiment)
            frames.append(table)
    folds = pd.concat(fold_frames).sort_values(["experiment", "held_out_subject", "classifier"]).reset_index(drop=True)
    predictions = pd.concat(prediction_frames).sort_values(["experiment", *ROW_KEYS, "classifier"]).reset_index(drop=True)
    summary = read_csv_literal(report_root / "numerical_summary.csv").sort_values(["experiment", "classifier"]).reset_index(drop=True)
    structure = {
        "n_subjects": int(baseline.subject_id.nunique()), "n_sessions": int(baseline[["subject_id", "session_id"]].drop_duplicates().shape[0]),
        "n_epochs": len(baseline), "row_keys": _keys(baseline, ROW_KEYS),
        "baseline_feature_order": baseline_columns, "extension_feature_order": extension_columns,
        "baseline_sentinel_columns": definition["baseline_sentinels"], "extension_sentinel_columns": definition["extension_sentinels"],
        "fold_keys": _keys(folds, ["experiment", "held_out_subject", "comparison", "classifier"]),
        "selected_features": folds.selected_features.tolist(),
        "prediction_keys": _keys(predictions, ["experiment", *ROW_KEYS, "classifier"]),
        "true_labels": predictions.true_label.astype(int).tolist(),
        "summary_keys": _keys(summary, ["experiment", "comparison", "classifier"]),
    }
    def moments(frame, columns):
        values = frame[columns].to_numpy(float)
        return np.column_stack([values.mean(axis=0), values.std(axis=0), values.min(axis=0), values.max(axis=0)]).tolist()
    numbers = {"baseline_features": moments(baseline, baseline_columns),
        "extension_features": moments(extension, extension_columns),
        "baseline_epoch_features": baseline[definition["baseline_sentinels"]].to_numpy(float).tolist(),
        "extension_epoch_features": extension[definition["extension_sentinels"]].to_numpy(float).tolist(),
        "fold_accuracy": folds.accuracy.to_numpy(float).tolist(),
        "predicted_labels": predictions.predicted_label.astype(int).tolist(),
        "summary_values": summary[["lopo_accuracy_mean", "lopo_accuracy_std", "n_held_out_subjects", "pooled_epoch_accuracy"]].to_numpy(float).tolist()}
    return structure, numbers


def evaluation_table(baseline, extension, definition=None) -> pd.DataFrame:
    """Preserve v2 metadata and use an explicitly prespecified bounded subset."""
    from .schema import META_COLUMNS
    definition = definition or fixture_definition()
    merged = merge_baseline_and_extensions(baseline, extension)
    missing = set(definition["evaluation_features"]) - set(merged.columns)
    if missing:
        raise NumericalValidationError(f"Prespecified evaluation features are missing: {sorted(missing)}")
    return merged[[*META_COLUMNS, *definition["evaluation_features"]]].copy()


def compare_observations(expected: dict, actual: dict, tolerances: dict) -> dict:
    """Keep numerical tolerances separate from exact identity/coverage contracts."""
    if not isinstance(expected, dict) or not isinstance(actual, dict) or not isinstance(tolerances, dict):
        raise NumericalValidationError("Numerical observations and tolerances must be objects")
    differences = []
    if expected.get("structure") != actual.get("structure"):
        differences.append({"field": "structure", "reason": "identity, ordering, selected features or coverage differs"})
    expected_numbers, actual_numbers = expected.get("numbers"), actual.get("numbers")
    if not isinstance(expected_numbers, dict) or not isinstance(actual_numbers, dict) or set(expected_numbers) != set(actual_numbers) or set(tolerances) != set(expected_numbers):
        raise NumericalValidationError("Numerical fields/tolerances are missing, additional or inconsistent")
    statistics = {}
    for name, values in expected_numbers.items():
        wanted, observed = np.asarray(values, dtype=float), np.asarray(actual_numbers[name], dtype=float)
        tolerance = tolerances[name]
        if not isinstance(tolerance, dict) or set(tolerance) != {"rtol", "atol"} or any(type(x) not in {int, float} or not np.isfinite(x) or x < 0 for x in tolerance.values()):
            raise NumericalValidationError("A tolerance must explicitly declare finite nonnegative rtol/atol")
        if wanted.shape != observed.shape or not np.isfinite(wanted).all() or not np.isfinite(observed).all():
            differences.append({"field": name, "reason": "shape or finite-value contract differs"})
            continue
        delta = np.abs(observed - wanted)
        statistics[name] = {"max_absolute_difference": float(np.max(delta)) if delta.size else 0.0,
                            "rtol": tolerance["rtol"], "atol": tolerance["atol"]}
        if not np.allclose(observed, wanted, **tolerance):
            differences.append({"field": name, "reason": "outside declared tolerance", **statistics[name]})
    return {"passed": not differences, "differences": differences, "metrics": statistics}


def _validate_reference(reference: dict, definition: dict) -> None:
    from .environment import validate_environment_evidence
    try:
        if reference.get("schema_version") != "eeg-numerical-reference-1" or reference.get("fixture_definition_sha256") != fingerprint(definition):
            raise NumericalValidationError("Reference does not describe this exact synthetic fixture")
        if not compare_observations(reference["observations"], reference["observations"], reference["tolerances"])["passed"]:
            raise NumericalValidationError("Reference observations violate their own contract")
        validated = validate_environment_evidence(reference["generation_environment"], require_reference=True)
        if reference["generation_environment_validation"] != validated:
            raise NumericalValidationError("Reference generation environment evidence is contradictory")
    except NumericalValidationError:
        raise
    except (KeyError, TypeError, ValueError, AttributeError) as exc:
        raise NumericalValidationError(f"Incomplete or invalid numerical reference: {exc}") from exc


def validate_numerical_report(report: dict, *, _depth=0) -> None:
    """Validate portable evidence, including recomputed reference comparisons.

    A fresh outer JSON hash cannot make contradictory values/status valid.
    Source evaluation manifests additionally validate prediction coverage and
    report arithmetic when the complete evidence directory is produced.
    """
    from .environment import validate_environment_evidence
    try:
        if _depth > 4:
            raise NumericalValidationError("Dependency evidence nesting exceeds four baseline comparisons")
        identity = dict(report)
        digest = identity.pop("report_sha256", None)
        if report.get("schema_version") != REPORT_VERSION or digest != fingerprint(identity):
            raise NumericalValidationError("Numerical report identity/hash is invalid")
        if report.get("fixture_definition_sha256") != fingerprint(report["fixture_definition"]):
            raise NumericalValidationError("Numerical fixture definition digest differs")
        for kind in ("source_algorithm", "input_data"):
            files = report[f"{kind}_files"]
            if not isinstance(files, dict) or not files or any(
                not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value) for value in files.values()
            ) or report[f"{kind}_sha256"] != fingerprint(files):
                raise NumericalValidationError(f"Numerical {kind} identity differs")
        if type(report["reference_environment_required"]) is not bool:
            raise NumericalValidationError("Reference environment requirement must be explicit")
        validate_environment_evidence(report["environment"], require_reference=report["reference_environment_required"])
        text = report["reference_text"]
        if hashlib.sha256(text.encode("utf-8")).hexdigest() != report["reference_sha256"]:
            raise NumericalValidationError("Embedded numerical reference digest differs")
        reference = _parse_json(text)
        _validate_reference(reference, report["fixture_definition"])
        comparisons = report["comparisons"]
        if not isinstance(comparisons, dict) or not {"committed_reference"}.issubset(comparisons) or not set(comparisons).issubset({"committed_reference", "dependency_comparison"}):
            raise NumericalValidationError("Numerical comparison inventory is incomplete or unknown")
        recomputed = compare_observations(reference["observations"], report["observations"], reference["tolerances"])
        if comparisons["committed_reference"] != recomputed:
            raise NumericalValidationError("Numerical comparison contradicts reference/observations")
        for comparison in comparisons.values():
            if type(comparison.get("passed")) is not bool or comparison["passed"] != (comparison.get("differences") == []):
                raise NumericalValidationError("Numerical comparison status contradicts its differences")
        if "dependency_comparison" in comparisons:
            baseline = report["dependency_baseline"]
            validate_numerical_report(baseline, _depth=_depth + 1)
            if comparisons["dependency_comparison"] != _compare_validated_reports(baseline, report, reference["tolerances"]):
                raise NumericalValidationError("Dependency comparison contradicts its baseline/observations")
        elif "dependency_baseline" in report:
            raise NumericalValidationError("Dependency baseline has no comparison evidence")
        expected_status = "passed" if all(c["passed"] for c in comparisons.values()) else "numerical_drift"
        if report.get("status") != expected_status:
            raise NumericalValidationError("Numerical status contradicts its comparison evidence")
    except NumericalValidationError:
        raise
    except (KeyError, TypeError, ValueError, AttributeError) as exc:
        raise NumericalValidationError(f"Incomplete or invalid numerical report: {exc}") from exc


def compare_numerical_reports(baseline: dict, candidate: dict, tolerances: dict) -> dict:
    """Compare dependency environments only when input/method identities agree."""
    validate_numerical_report(baseline)
    validate_numerical_report(candidate)
    return _compare_validated_reports(baseline, candidate, tolerances)


def _compare_validated_reports(baseline: dict, candidate: dict, tolerances: dict) -> dict:
    if baseline["status"] != "passed":
        raise NumericalValidationError("Controlled comparison requires a passing baseline")
    for field in ("fixture_definition_sha256", "source_algorithm_sha256", "input_data_sha256", "reference_sha256"):
        if baseline[field] != candidate[field]:
            raise NumericalValidationError(f"Controlled environment comparison requires identical {field}")
    if tolerances != _parse_json(baseline["reference_text"])["tolerances"]:
        raise NumericalValidationError("Controlled comparison cannot change reviewed tolerances")
    return {**compare_observations(baseline["observations"], candidate["observations"], tolerances),
            "baseline_report_sha256": baseline["report_sha256"],
            "baseline_environment_sha256": fingerprint(baseline["environment"]),
            "candidate_environment_sha256": fingerprint(candidate["environment"])}


def run_numerical_validation(out_dir: str | Path, *, reference_path: str | Path = DEFAULT_REFERENCE,
                             require_reference_environment=False, environment_spec=None, compare_to=None) -> dict:
    """Run real stages and atomically publish complete synthetic evidence.

    Numerical drift is a completed diagnostic report with ``status`` set to
    ``numerical_drift``; malformed inputs and structural failures raise and never
    publish a misleading successful directory.
    """
    from .environment import capture_environment, verify_reference_environment
    definition = fixture_definition()
    reference_path = Path(reference_path)
    reference_text = reference_path.read_bytes().decode("utf-8")
    reference = _parse_json(reference_text)
    _validate_reference(reference, definition)
    environment = capture_environment(reference_path=environment_spec)
    if require_reference_environment:
        verify_reference_environment(environment, reference_path=environment_spec)
    # Environment specifications intentionally vary in a controlled dependency
    # comparison; actual analysis source and synthetic inputs must not vary.
    files = {f"src/eeg_ma/{name}.py": sha256_file(ROOT / f"src/eeg_ma/{name}.py") for name in
             ("numerical_validation", "extract", "extensions", "preprocess", "features", "evaluation",
              "models", "sfs", "kfda", "schema", "qa", "io", "constants", "config", "provenance", "experiment_runs")}
    code = code_identity()
    baseline_report = None
    if compare_to is not None:
        baseline_report = _read_json(Path(compare_to))
        validate_numerical_report(baseline_report)
        if baseline_report["status"] != "passed":
            raise NumericalValidationError("Controlled comparison requires a passing baseline")
        expected = {"fixture_definition_sha256": fingerprint(definition),
                    "source_algorithm_sha256": fingerprint(files),
                    "reference_sha256": hashlib.sha256(reference_text.encode("utf-8")).hexdigest()}
        for field, value in expected.items():
            if baseline_report[field] != value:
                raise NumericalValidationError(f"Controlled environment comparison requires identical {field}")
        baseline_raw = Path(compare_to).parent / "raw"
        if _raw_inventory(baseline_raw) != baseline_report["input_data_files"]:
            raise NumericalValidationError("Baseline raw bytes differ from their recorded input identity")
    with new_output_directory(out_dir) as stage:
        if baseline_report is None:
            raw = write_synthetic_raw(stage / "raw", definition)
        else:
            # Keep source bytes fixed while changing numerical dependencies.
            raw = Path(shutil.copytree(baseline_raw, stage / "raw"))
        inputs = _raw_inventory(raw)
        if baseline_report is not None and inputs != baseline_report["input_data_files"]:
            raise NumericalValidationError("Baseline raw bytes changed during evidence copy")
        baseline, qa = extract_all(raw, definition["analysis"], stage / "features/baseline")
        baseline_path = stage / "features/baseline/features_all.csv"
        extension = extract_all_extension_features(raw, baseline_path, definition["analysis"], definition["extensions"], stage / "features/extensions")
        merged = evaluation_table(baseline, extension, definition)
        if not qa.qa_pass.all():
            raise NumericalValidationError("Synthetic raw acquisition unexpectedly failed shared QA")
        matrix, evaluator = prepare_matrix(merged, definition["analysis"], experiments=definition["experiments"],
            selected=[x["name"] for x in definition["experiments"]], scopes=["inter"], out_root=stage / "evaluation",
            input_paths={"baseline_features": baseline_path, "extension_features": stage / "features/extensions/features_extensions.csv"},
            reference_paths={}, matrix_config={"fixture_definition_sha256": fingerprint(definition)}, code=code, environment=environment)
        def report(inventory):
            tables = []
            for experiment, root in sorted(inventory.items()):
                table = read_csv_literal(root / "inter/inter_summary.csv")
                table.insert(0, "experiment", experiment)
                tables.append(table)
            return {"numerical_summary.csv": pd.concat(tables, ignore_index=True)}
        inventory, report_root = matrix.run(evaluator, report, report_metadata={"purpose": "synthetic_grouped_cv_regression_only"})
        if resolve_report_root(stage / "evaluation") != report_root:
            raise NumericalValidationError("Validated report pointer does not match the published report")
        for path in inventory.values():
            validate_complete_run(path)
        validate_complete_run(report_root)
        structure, numbers = _observations(baseline, extension, inventory, report_root)
        observations = {"structure": structure, "numbers": numbers}
        comparison = compare_observations(reference["observations"], observations, reference["tolerances"])
        result = {"schema_version": REPORT_VERSION, "purpose": "synthetic_grouped_cv_regression_only",
            "fixture_definition": definition, "fixture_definition_sha256": fingerprint(definition),
            "source_algorithm_files": files, "source_algorithm_sha256": fingerprint(files),
            "input_data_files": inputs, "input_data_sha256": fingerprint(inputs),
            "input_origin": "generated" if baseline_report is None else "baseline_raw_reused",
            "reference_text": reference_text, "reference_sha256": hashlib.sha256(reference_text.encode("utf-8")).hexdigest(),
            "environment": environment,
            "reference_environment_required": bool(require_reference_environment), "code": code,
            "observations": observations, "comparisons": {"committed_reference": comparison},
            "evaluation_report": str(report_root.relative_to(stage)),
            "status": "passed" if comparison["passed"] else "numerical_drift"}
        result["report_sha256"] = fingerprint(result)
        if baseline_report is not None:
            candidate_comparison = compare_numerical_reports(baseline_report, result, reference["tolerances"])
            result["comparisons"]["dependency_comparison"] = candidate_comparison
            result["dependency_baseline"] = baseline_report
            result["status"] = "passed" if all(c["passed"] for c in result["comparisons"].values()) else "numerical_drift"
            result.pop("report_sha256")
            result["report_sha256"] = fingerprint(result)
        validate_numerical_report(result)
        _write_json(stage / "numerical_report.json", result)
    return result
