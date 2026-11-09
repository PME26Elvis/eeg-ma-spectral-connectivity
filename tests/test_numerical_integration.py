"""Risk-based checks on the actual public synthetic raw-to-report workflow."""
from __future__ import annotations

from collections import Counter
import copy
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from eeg_ma import evaluation
from eeg_ma.extract import extract_all, feature_columns
from eeg_ma.kfda import KernelFisherDiscriminant
from eeg_ma.numerical_validation import (
    DEFAULT_REFERENCE, NumericalValidationError, compare_numerical_reports,
    compare_observations, evaluation_table, fixture_definition,
    run_numerical_validation, validate_numerical_report,
)
from eeg_ma.provenance import (
    ProvenanceError, RunTransaction, _artifacts, fingerprint, read_csv_literal,
    resolve_report_root, validate_complete_run,
)
from eeg_ma.qa import QAError
from eeg_ma.schema import load_feature_table


@pytest.fixture(scope="module")
def complete(tmp_path_factory):
    root = tmp_path_factory.mktemp("numerical-integration") / "evidence"
    report = run_numerical_validation(root)
    baseline = load_feature_table(root / "features/baseline/features_all.csv")
    extension = load_feature_table(root / "features/extensions/features_extensions.csv")
    return root, report, baseline, extension


def _rehash_report(report):
    report.pop("report_sha256", None)
    report["report_sha256"] = fingerprint(report)
    return report


def _rehash_artifacts(root):
    """Simulate a writer that blesses bad CSV bytes, to exercise semantic guards."""
    path = root / "run_manifest.json"
    manifest = json.loads(path.read_text())
    manifest["artifacts"] = _artifacts(root)
    path.write_text(json.dumps(manifest))


def test_actual_raw_to_validated_report_has_complete_multisession_coverage(complete):
    root, report, baseline, extension = complete
    assert report["status"] == "passed"
    validate_numerical_report(report)
    structure = report["observations"]["structure"]
    assert (structure["n_subjects"], structure["n_sessions"], structure["n_epochs"]) == (3, 6, 48)
    assert set(baseline.subject_id) == {"001", "1", "NA"}
    assert len(structure["baseline_feature_order"]) == 168
    assert len(structure["extension_feature_order"]) == 144
    assert len(structure["fold_keys"]) == 18
    assert len(structure["prediction_keys"]) == 288
    assert set(baseline.epoch_id) == set(extension.epoch_id)
    assert read_csv_literal(root / "features/baseline/raw_validation.csv").qa_pass.all()
    assert report["environment"]["schema_version"] == "eeg-environment-1"

    for name in ("baseline", "phase"):
        manifest = validate_complete_run(root / "evaluation" / name)
        assert manifest["identity"]["environment"] == report["environment"]
        for split in manifest["identity"]["splits"]["inter"]:
            held_out = split["held_out_subject"]
            held_out_rows = baseline.loc[baseline.subject_id == held_out]
            assert held_out_rows.session_id.nunique() == 2
            assert set(split["test_epoch_ids"]) == set(held_out_rows.epoch_id)
            assert held_out not in split["train_subject_ids"]
            for inner in split["inner_splits"]:
                assert not set(held_out_rows.epoch_id) & (set(inner["train_epoch_ids"]) | set(inner["test_epoch_ids"]))
    report_root = resolve_report_root(root / "evaluation")
    assert report_root == root / report["evaluation_report"]
    validate_complete_run(report_root)


@pytest.mark.parametrize("fault,code", [
    ("missing_event", "TASK_EVENT_TRIAL_SET"),
    ("duplicate_event", "DUPLICATE_TASK_EVENT"),
    ("sample_gap", "SAMPLE_INDEX"),
])
def test_real_ingestion_blocks_raw_faults_before_feature_publication(complete, tmp_path, fault, code):
    source, _, _, _ = complete
    raw = Path(shutil.copytree(source / "raw", tmp_path / "raw"))
    session = raw / "person-1-session-1"
    if fault == "sample_gap":
        path = session / "eeg_raw.csv"
        rows = path.read_bytes().splitlines(keepends=True)
        path.write_bytes(b"".join(rows[:1001] + rows[1002:]))
    else:
        path = session / "events.csv"
        events = read_csv_literal(path)
        index = events.index[events.phase == "Calculation"][0]
        if fault == "missing_event":
            events = events.drop(index)
        else:
            events = pd.concat([events, events.loc[[index]]]).sort_values("elapsed_ms")
        events.to_csv(path, index=False)
    output = tmp_path / "features"
    with pytest.raises(QAError, match=code):
        extract_all(raw, fixture_definition()["analysis"], output)
    assert not output.exists()


def test_every_scaler_selector_and_classifier_fit_excludes_heldout_subject_sessions(complete, tmp_path, monkeypatch):
    """Spy on real training calls, including SFS and grid inner fits/refits.

    Raw feature values identify their source epochs uniquely. Estimator fits
    must receive precisely the scaled training rows from the preceding scaler,
    so inspecting only a split manifest cannot make this test pass.
    """
    _, _, baseline, extension = complete
    frame = evaluation_table(baseline, extension)
    cfg = fixture_definition()["analysis"]
    columns = feature_columns(frame, "all")
    X, y = frame[columns].to_numpy(float), frame.class_label.to_numpy(int)
    subjects = frame.subject_id.to_numpy()
    ids = frame.epoch_id.to_numpy()
    positions = {value: index for index, value in enumerate(ids)}
    splits = evaluation.make_evaluation_splits(frame, cfg, "inter")
    value_rows = [{value: row for row, value in enumerate(X[:, column])} for column in range(X.shape[1])]
    assert all(len(mapping) == len(frame) for mapping in value_rows)
    context = {"outer": -1, "scaler": None}
    calls = Counter()
    training_sets = []
    original_selector, original_scaler = evaluation.forward_select_lda, StandardScaler.fit

    def selector(train_x, train_y, feature_names, inner_splits, **kwargs):
        context["outer"] += 1
        split = splits[context["outer"]]
        train_rows = np.array([positions[value] for value in split["train_epoch_ids"]])
        np.testing.assert_array_equal(train_x, X[train_rows])
        np.testing.assert_array_equal(train_y, y[train_rows])
        assert list(feature_names) == columns
        assert len(inner_splits) == 2
        for (train, valid), recorded in zip(inner_splits, split["inner_splits"]):
            assert ids[train_rows[train]].tolist() == recorded["train_epoch_ids"]
            assert ids[train_rows[valid]].tolist() == recorded["test_epoch_ids"]
            assert set(subjects[train_rows[train]]).isdisjoint(subjects[train_rows[valid]])
        calls["selector"] += 1
        return original_selector(train_x, train_y, feature_names, inner_splits, **kwargs)

    def scaler(self, fit_x, fit_y=None, **kwargs):
        assert context["outer"] >= 0, "A scaler fit occurred before training-only SFS"
        split = splits[context["outer"]]
        assert fit_x.shape[1] == 1  # The fixture fixes the SFS cap at one.
        matches = [[mapping[value] for value in fit_x[:, 0]] for mapping in value_rows
                   if all(value in mapping for value in fit_x[:, 0])]
        assert len(matches) == 1, "Fit rows do not correspond to original feature epochs"
        rows = np.array(matches[0])
        expected_sets = [set(split["train_epoch_ids"]), *[set(inner["train_epoch_ids"]) for inner in split["inner_splits"]]]
        fitted_ids = set(ids[rows])
        assert fitted_ids in expected_sets and len(fitted_ids) == len(rows)
        assert split["held_out_subject"] not in set(subjects[rows])
        assert fitted_ids.isdisjoint(split["test_epoch_ids"])
        np.testing.assert_array_equal(fit_y, y[rows])
        result = original_scaler(self, fit_x, fit_y, **kwargs)
        assert context["scaler"] is None, "Previous scaler had no matching estimator fit"
        context["scaler"] = (self.transform(fit_x), y[rows])
        training_sets.append((split["held_out_subject"], len(rows)))
        calls["scaler"] += 1
        return result

    def spy_estimator(cls):
        original = cls.fit
        def fit(self, fit_x, fit_y, **kwargs):
            assert context["scaler"] is not None, "Estimator bypassed its training-only scaler"
            scaled_x, expected_y = context["scaler"]
            np.testing.assert_array_equal(fit_x, scaled_x)
            np.testing.assert_array_equal(fit_y, expected_y)
            context["scaler"] = None
            calls[cls.__name__] += 1
            return original(self, fit_x, fit_y, **kwargs)
        monkeypatch.setattr(cls, "fit", fit)

    monkeypatch.setattr(evaluation, "forward_select_lda", selector)
    monkeypatch.setattr(StandardScaler, "fit", scaler)
    for cls in (LinearDiscriminantAnalysis, SVC, KernelFisherDiscriminant):
        spy_estimator(cls)
    evaluation.run_inter_subject(frame, cfg, tmp_path / "spied")
    assert calls == {"selector": 3, "scaler": 39, "LinearDiscriminantAnalysis": 21,
                     "SVC": 9, "KernelFisherDiscriminant": 9}
    assert Counter(training_sets) == Counter({(subject, size): count for subject in ("001", "1", "NA")
                                             for size, count in ((16, 10), (32, 3))})
    assert context["scaler"] is None


@pytest.mark.parametrize("fault,match", [
    ("partial_predictions", "row count"),
    ("summary_arithmetic", "inter summary"),
    ("feature_order", "feature order"),
])
def test_rehashed_bad_outputs_still_fail_actual_semantic_validator(complete, tmp_path, fault, match):
    source, _, _, _ = complete
    run = Path(shutil.copytree(source / "evaluation/baseline", tmp_path / "run"))
    if fault == "partial_predictions":
        path = run / "inter/inter_predictions.csv"
        read_csv_literal(path).iloc[:-1].to_csv(path, index=False)
    elif fault == "summary_arithmetic":
        path = run / "inter/inter_summary.csv"
        table = read_csv_literal(path)
        table.loc[0, "lopo_accuracy_mean"] = 0.123
        table.to_csv(path, index=False)
    else:
        path = run / "inter/inter_sfs_path.csv"
        table = read_csv_literal(path)
        table.loc[0, "candidate_index"] = (int(table.loc[0, "candidate_index"]) + 1) % 3
        table.to_csv(path, index=False)
    _rehash_artifacts(run)
    with pytest.raises(ProvenanceError, match=match):
        validate_complete_run(run)


def test_schema_alignment_and_prespecified_features_cannot_silently_shrink(complete):
    _, _, baseline, extension = complete
    bad = extension.copy()
    bad.loc[0, "session_id"] = "foreign-session"
    with pytest.raises(ValueError):
        evaluation_table(baseline, bad)
    with pytest.raises(NumericalValidationError, match="Prespecified"):
        evaluation_table(baseline.drop(columns=["BP__FP1__alpha"]), extension)


def test_feature_order_mismatch_cannot_reuse_a_completed_run(complete):
    root, _, _, _ = complete
    run = root / "evaluation/baseline"
    identity = copy.deepcopy(validate_complete_run(run)["identity"])
    identity["feature_order"].reverse()
    before = (run / "run_manifest.json").read_bytes()
    with pytest.raises(ProvenanceError, match="identity mismatch"):
        RunTransaction(run, identity, resume=True).execute(lambda _: pytest.fail("must not fit"))
    assert (run / "run_manifest.json").read_bytes() == before


def test_numeric_tolerances_allow_roundoff_but_not_drift_or_prediction_change(complete):
    _, report, _, _ = complete
    reference = json.loads(DEFAULT_REFERENCE.read_text())
    expected = report["observations"]
    tiny = copy.deepcopy(expected)
    tiny["numbers"]["baseline_features"][0][0] += 1e-12
    assert compare_observations(expected, tiny, reference["tolerances"])["passed"]
    for field, value in (("baseline_features", 0.5), ("predicted_labels", 1)):
        changed = copy.deepcopy(expected)
        if field == "predicted_labels":
            changed["numbers"][field][0] = 1 - changed["numbers"][field][0]
        else:
            changed["numbers"][field][0][0] += value
        assert not compare_observations(expected, changed, reference["tolerances"])["passed"]
    reordered = copy.deepcopy(expected)
    reordered["structure"]["baseline_feature_order"].reverse()
    assert not compare_observations(expected, reordered, reference["tolerances"])["passed"]


@pytest.mark.parametrize("fault", ["empty_comparisons", "numeric_values", "contradictory_status", "missing_environment"])
def test_rehash_cannot_bless_partial_or_contradictory_numerical_evidence(complete, fault):
    _, original, _, _ = complete
    report = copy.deepcopy(original)
    if fault == "empty_comparisons":
        report["comparisons"] = {}
    elif fault == "numeric_values":
        report["observations"]["numbers"]["baseline_features"][0][0] += 10
    elif fault == "contradictory_status":
        report["status"] = "numerical_drift"
    else:
        report["environment"] = {}
    _rehash_report(report)
    with pytest.raises(NumericalValidationError):
        validate_numerical_report(report)


@pytest.mark.parametrize("field", ["source_algorithm", "input_data", "fixture_definition"])
def test_environment_comparison_requires_identical_analysis_and_inputs(complete, field):
    _, original, _, _ = complete
    candidate = copy.deepcopy(original)
    if field == "fixture_definition":
        candidate["fixture_definition"]["seed"] += 1
        candidate["fixture_definition_sha256"] = fingerprint(candidate["fixture_definition"])
        # The embedded reviewed reference no longer binds this definition.
    else:
        files = candidate[f"{field}_files"]
        files[next(iter(files))] = "a" * 64
        candidate[f"{field}_sha256"] = fingerprint(files)
    _rehash_report(candidate)
    reference = json.loads(DEFAULT_REFERENCE.read_text())
    with pytest.raises(NumericalValidationError, match="identical|exact synthetic fixture"):
        compare_numerical_reports(original, candidate, reference["tolerances"])


def test_environment_identity_can_differ_without_requiring_feature_byte_identity(complete):
    _, original, _, _ = complete
    candidate = copy.deepcopy(original)
    candidate["environment"]["platform"] += "-different-observed-host"
    env = candidate["environment"]
    env.pop("environment_id")
    env["environment_id"] = fingerprint(env)
    _rehash_report(candidate)
    comparison = compare_numerical_reports(original, candidate, json.loads(DEFAULT_REFERENCE.read_text())["tolerances"])
    assert comparison["passed"]
    assert comparison["baseline_environment_sha256"] != comparison["candidate_environment_sha256"]


def test_comparison_reuses_verified_baseline_raw_and_rejects_changed_source(complete, tmp_path, monkeypatch):
    source, original, _, _ = complete
    # Compare on a frozen source checkout; the generator must never be rerun.
    import eeg_ma.numerical_validation as numerics
    monkeypatch.setattr(numerics, "write_synthetic_raw", lambda *a, **k: pytest.fail("must reuse raw bytes"))
    candidate = run_numerical_validation(tmp_path / "candidate", compare_to=source / "numerical_report.json")
    assert candidate["status"] == "passed"
    assert candidate["input_origin"] == "baseline_raw_reused"
    assert candidate["input_data_sha256"] == original["input_data_sha256"]
    assert candidate["comparisons"]["dependency_comparison"]["passed"]
    for fault in ("metric", "status", "baseline_hash", "environment_hash", "missing_baseline"):
        broken = copy.deepcopy(candidate)
        comparison = broken["comparisons"]["dependency_comparison"]
        if fault == "metric":
            comparison["metrics"]["baseline_features"]["max_absolute_difference"] = 1.0
        elif fault == "status":
            comparison.update(passed=False, differences=[{"reason": "invented drift"}])
            broken["status"] = "numerical_drift"
        elif fault == "baseline_hash":
            comparison["baseline_report_sha256"] = "0" * 64
        elif fault == "environment_hash":
            comparison["candidate_environment_sha256"] = "0" * 64
        else:
            broken.pop("dependency_baseline")
        _rehash_report(broken)
        with pytest.raises(NumericalValidationError):
            validate_numerical_report(broken)
    tampered = tmp_path / "tampered"
    tampered.mkdir()
    shutil.copy2(source / "numerical_report.json", tampered / "numerical_report.json")
    shutil.copytree(source / "raw", tampered / "raw")
    (tampered / "raw/person-1-session-1/events.csv").unlink()
    output = tmp_path / "blocked"
    with pytest.raises(NumericalValidationError, match="Baseline raw bytes"):
        run_numerical_validation(output, compare_to=tampered / "numerical_report.json")
    assert not output.exists()


def test_numerical_output_is_immutable(complete):
    root, _, _, _ = complete
    before = (root / "numerical_report.json").read_bytes()
    with pytest.raises(FileExistsError, match="already exists"):
        run_numerical_validation(root)
    assert (root / "numerical_report.json").read_bytes() == before


def test_cli_missing_reference_fails_closed(tmp_path):
    script = Path(__file__).resolve().parents[1] / "scripts/19_validate_numerics.py"
    output = tmp_path / "never-published"
    process = subprocess.run([sys.executable, str(script), "--out-dir", str(output), "--reference", str(tmp_path / "missing.json")],
                             capture_output=True, text=True, check=False)
    assert process.returncode == 2
    assert json.loads(process.stdout)["status"] == "blocked"
    assert not output.exists()


def test_cli_numerical_drift_is_nonzero_and_keeps_complete_diagnostic_evidence(tmp_path):
    reference = json.loads(DEFAULT_REFERENCE.read_text())
    reference["observations"]["numbers"]["baseline_features"][0][0] += 10
    reviewed = tmp_path / "deliberately-incompatible-reference.json"
    # Preserve the actual reviewed bytes, including CRLF; text-mode newline
    # normalization must not silently change the advertised reference digest.
    reference_bytes = (json.dumps(reference, indent=2) + "\n").replace("\n", "\r\n").encode("utf-8")
    reviewed.write_bytes(reference_bytes)
    script = Path(__file__).resolve().parents[1] / "scripts/19_validate_numerics.py"
    output = tmp_path / "drift"
    process = subprocess.run([sys.executable, str(script), "--out-dir", str(output), "--reference", str(reviewed)],
                             capture_output=True, text=True, check=False)
    assert process.returncode == 2
    assert json.loads(process.stdout)["status"] == "numerical_drift"
    report = json.loads((output / "numerical_report.json").read_text())
    validate_numerical_report(report)
    assert report["reference_text"] == reference_bytes.decode("utf-8")
    assert report["reference_sha256"] == hashlib.sha256(reference_bytes).hexdigest()
    assert not report["comparisons"]["committed_reference"]["passed"]
    validate_complete_run(resolve_report_root(output / "evaluation"))
    assert json.loads(reviewed.read_text()) == reference, "A drift must never rewrite its reference"
    assert reviewed.read_bytes() == reference_bytes
