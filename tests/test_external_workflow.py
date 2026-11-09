"""Real fixed-model external lifecycle with explicit synthetic study evidence.

The fixture fits the complete 21/168 candidate feature schemas once. Only
human/acquisition attestations and, in compatibility CI, the live reference
probe are synthetic. Predict and score still use the production protocol gates,
immutable files, exported numerical models and recorded outcome boundary.
"""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import runpy
import shutil
import sys

import numpy as np
import pandas as pd
import pytest
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.preprocessing import StandardScaler

from eeg_ma import environment, fixed_model, protocol, sfs
from eeg_ma.provenance import fingerprint, read_csv_literal, sha256_file
from fixed_model_fixtures import make_feature_table, write_development_fixture, write_json
from test_environment import synthetic_reference_environment
from test_protocol_gate import LOCK_TIME, cohort_for, enrollment_for, synthetic_code


def _rehash(value, key):
    value.pop(key, None)
    value[key] = fingerprint(value)
    return value


def _table_descriptor(path):
    frame = read_csv_literal(path)
    return {"sha256": sha256_file(path), "bytes": path.stat().st_size,
            "rows": len(frame), "columns": list(frame.columns)}


def _rewrite_labels(root, transform):
    path = root / "labels.csv"
    frame = transform(read_csv_literal(path))
    frame.to_csv(path, index=False)
    manifest_path = root / "labels_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["labels"] = _table_descriptor(path)
    write_json(manifest_path, _rehash(manifest, "labels_id"))


def _rewrite_inputs(root, transform):
    path = root / "features.csv"
    transform(read_csv_literal(path)).to_csv(path, index=False)
    manifest_path = root / "input_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["features"] = _table_descriptor(path)
    write_json(manifest_path, _rehash(manifest, "input_id"))


def _forbid_fit(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("External observations must never enter fitting or feature selection")
    monkeypatch.setattr(StandardScaler, "fit", forbidden)
    monkeypatch.setattr(StandardScaler, "partial_fit", forbidden)
    monkeypatch.setattr(LinearDiscriminantAnalysis, "fit", forbidden)
    monkeypatch.setattr(sfs, "forward_select_lda", forbidden)
    monkeypatch.setattr(fixed_model, "forward_select_lda", forbidden)


@pytest.fixture(scope="module")
def external_base(tmp_path_factory):
    from eeg_ma import external
    root = tmp_path_factory.mktemp("fixed-external-study")
    patches = pytest.MonkeyPatch()
    observed = environment.capture_environment()
    if observed["reference_validation"]["status"] != "matched":
        # Compatibility CI exercises the same production logic without claiming
        # its broad dependency stack is the research reference.
        proof = synthetic_reference_environment()
        probe = lambda *args, **kwargs: deepcopy(proof)
        patches.setattr(fixed_model, "verify_reference_environment", probe)
        patches.setattr(external, "verify_reference_environment", probe)
    patches.setattr(fixed_model, "code_identity", synthetic_code)
    patches.setattr(protocol, "code_identity", synthetic_code)
    patches.setattr(external, "code_identity", synthetic_code)
    try:
        source, spec, development = write_development_fixture(root / "development")
        bundle_root = root / "models"
        bundle = fixed_model.fit_development(source, bundle_root)
        for role in ("environment", "primary_model", "reference_model", "primary_training_manifest", "reference_training_manifest"):
            entry = bundle["artifacts"][role]
            spec["freeze"]["artifacts"][role] = {"path": str(bundle_root / entry["path"]), "sha256": entry["sha256"]}
        write_json(source, spec)
        now = protocol._now
        patches.setattr(protocol, "_now", lambda: LOCK_TIME)
        lock = root / "lock"
        protocol.lock_protocol(source, lock)
        patches.setattr(protocol, "_now", now)

        people = [f"external-{number:03d}" for number in range(94)]
        cohort = cohort_for(spec, people)
        enrollment = enrollment_for(spec, cohort)
        for index, item in enumerate(enrollment["participants"]):
            if index < 3:
                item["sessions"][0]["session_id"] = "session-1"
            else:
                item.update(disposition="lost_to_followup", sessions=[])
        write_json(root / "cohort.json", cohort)
        write_json(root / "enrollment.json", enrollment)
        frame = make_feature_table(people[:3], cohort["cohort_id"], sessions=1, epochs_per_class=30, seed=2311,
            protocol_id=spec["protocols"]["A"]["protocol_id"], protocol_version=spec["protocols"]["A"]["protocol_version"])
        frame.to_csv(root / "external_features.csv", index=False)
        external.prepare_external_inputs(root / "external_features.csv", root / "prepared",
            lock_dir=lock, cohort_manifest_path=root / "cohort.json")
        external.predict_external(root / "prepared/inputs", root / "predictions",
            lock_dir=lock, cohort_manifest_path=root / "cohort.json")
        yield {"root": root, "spec": spec, "development": development, "external": frame}
    finally:
        patches.undo()


@pytest.fixture
def external_case(external_base, tmp_path):
    root = Path(shutil.copytree(external_base["root"], tmp_path / "case"))
    return {**external_base, "root": root}


def _score(case, out_name="score"):
    from eeg_ma.external import score_external
    root = case["root"]
    return score_external(root / "predictions", root / "prepared/labels", root / out_name,
        lock_dir=root / "lock", cohort_manifest_path=root / "cohort.json",
        enrollment_manifest_path=root / "enrollment.json")


def test_development_export_load_external_prediction_and_score_are_complete(external_case, monkeypatch):
    from eeg_ma.external import validate_predictions, validate_external_score
    root = external_case["root"]
    _forbid_fit(monkeypatch)
    checked = validate_predictions(root / "predictions", lock_dir=root / "lock")
    manifest, predictions = checked["manifest"], checked["predictions"]
    assert set(predictions.model_role) == {"primary", "reference"}
    assert set(predictions.subject_id) == set(external_case["external"].subject_id)
    assert not set(predictions.subject_id) & set(external_case["development"].subject_id)
    assert len(predictions) == len(external_case["external"]) * 2
    assert not predictions.duplicated(["epoch_id", "model_role"]).any()
    assert set(predictions.protocol_lock_id) == {manifest["protocol_lock_id"]}
    assert predictions[["cohort_id", "subject_id", "session_id", "recording_id", "epoch_id", "model_id"]].notna().all().all()
    before = {p.name: p.read_bytes() for p in (root / "models").iterdir() if p.is_file()}
    score = _score(external_case)
    assert score["status"] == "complete" and score["prediction_id"] == manifest["prediction_id"]
    rows = read_csv_literal(root / "score/participant_scores.csv")
    assert len(rows) == 94 * 4
    evaluable = rows.loc[rows.evaluable]
    assert len(evaluable) == 12
    assert set(evaluable.n_rest) == {30} and set(evaluable.n_task) == {30}
    assert evaluable.balanced_accuracy.between(0, 1).all()
    assert set(rows.loc[~rows.evaluable, "n_rest"]) == {0}
    assert set(rows.loc[~rows.evaluable, "n_task"]) == {0}
    assert all(item["n_evaluable"] == 3 for item in score["summary"])
    assert score["enrollment_audit"]["n_recruited"] == 94
    assert score["enrollment_audit"]["n_evaluable"] == 3
    assert {p.name: p.read_bytes() for p in (root / "models").iterdir() if p.is_file()} == before
    validate_external_score(root / "score", prediction_dir=root / "predictions", lock_dir=root / "lock")


def test_prediction_never_reads_labels_and_never_fits(external_case, monkeypatch):
    from eeg_ma.external import predict_external, validate_predictions
    root = external_case["root"]
    _forbid_fit(monkeypatch)
    shutil.rmtree(root / "prepared/labels")
    predict_external(root / "prepared/inputs", root / "without-labels",
        lock_dir=root / "lock", cohort_manifest_path=root / "cohort.json")
    validate_predictions(root / "without-labels", lock_dir=root / "lock")
    assert (root / "without-labels/predictions.csv").read_bytes() == (root / "predictions/predictions.csv").read_bytes()


def test_heldout_label_changes_affect_only_scores_and_reuse_the_committed_outcome_boundary(external_case, monkeypatch):
    root = external_case["root"]
    _forbid_fit(monkeypatch)
    model_hashes = {p.name: sha256_file(p) for p in (root / "models").iterdir() if p.is_file()}
    prediction_bytes = (root / "predictions/predictions.csv").read_bytes()
    first = _score(external_case, "first-score")
    _rewrite_labels(root / "prepared/labels", lambda frame: frame.assign(true_label=1 - frame.true_label))
    second = _score(external_case, "changed-label-score")
    a = read_csv_literal(root / "first-score/participant_scores.csv").sort_values(["subject_id", "comparison", "model_role"])
    b = read_csv_literal(root / "changed-label-score/participant_scores.csv").sort_values(["subject_id", "comparison", "model_role"])
    np.testing.assert_allclose(a.loc[a.evaluable, "balanced_accuracy"].to_numpy()
        + b.loc[b.evaluable, "balanced_accuracy"].to_numpy(), 1.0, rtol=0, atol=1e-14)
    assert first["score_id"] != second["score_id"] and first["prediction_id"] == second["prediction_id"]
    assert (root / "predictions/predictions.csv").read_bytes() == prediction_bytes
    assert {p.name: sha256_file(p) for p in (root / "models").iterdir() if p.is_file()} == model_hashes
    assert len(protocol.validate_lock(root / "lock")["events"]) == 1


def test_label_row_order_does_not_change_subject_metrics(external_case, monkeypatch):
    root = external_case["root"]
    _forbid_fit(monkeypatch)
    _score(external_case, "first-score")
    _rewrite_labels(root / "prepared/labels", lambda frame: frame.sample(frac=1, random_state=31))
    _score(external_case, "reordered-score")
    assert (root / "first-score/participant_scores.csv").read_bytes() == (root / "reordered-score/participant_scores.csv").read_bytes()


def test_changed_external_features_cannot_change_any_training_state(external_case, monkeypatch):
    from eeg_ma.external import prepare_external_inputs, predict_external, validate_predictions
    root = external_case["root"]
    _forbid_fit(monkeypatch)
    before = {p.name: sha256_file(p) for p in (root / "models").iterdir() if p.is_file()}
    frame = external_case["external"].copy()
    for name in frame:
        if name.startswith("PLV__"):
            frame[name] = 1.0 - frame[name]
    frame.to_csv(root / "changed_features.csv", index=False)
    prepare_external_inputs(root / "changed_features.csv", root / "changed-prepared",
        lock_dir=root / "lock", cohort_manifest_path=root / "cohort.json")
    predict_external(root / "changed-prepared/inputs", root / "changed-predictions",
        lock_dir=root / "lock", cohort_manifest_path=root / "cohort.json")
    changed = validate_predictions(root / "changed-predictions", lock_dir=root / "lock")
    original = validate_predictions(root / "predictions", lock_dir=root / "lock")
    assert changed["manifest"]["prediction_id"] != original["manifest"]["prediction_id"]
    assert set(changed["predictions"].model_id) == set(original["predictions"].model_id)
    assert {p.name: sha256_file(p) for p in (root / "models").iterdir() if p.is_file()} == before


@pytest.mark.parametrize("fault", ["feature_order", "label_column", "duplicate_epoch", "training_subject", "acquisition_version", "nonfinite_feature"])
def test_rehashed_external_input_contract_faults_fail_before_prediction_publication(external_case, fault):
    from eeg_ma.external import predict_external
    root = external_case["root"]
    input_root = root / "prepared/inputs"
    manifest = json.loads((input_root / "input_manifest.json").read_text())
    def transform(frame):
        if fault == "feature_order":
            features = manifest["feature_order"]
            return frame[[c for c in frame if c not in features] + list(reversed(features))]
        if fault == "label_column":
            return frame.assign(true_label=0)
        if fault == "duplicate_epoch":
            return pd.concat([frame, frame.iloc[[0]]], ignore_index=True)
        if fault == "acquisition_version":
            return frame.assign(protocol_version="unlocked-version")
        if fault == "nonfinite_feature":
            return frame.assign(**{manifest["feature_order"][0]: np.inf})
        return frame.assign(subject_id="001")
    _rewrite_inputs(input_root, transform)
    with pytest.raises(ValueError):
        predict_external(input_root, root / "rejected", lock_dir=root / "lock", cohort_manifest_path=root / "cohort.json")
    assert not (root / "rejected").exists()


@pytest.mark.parametrize("stage", ["live_prediction", "prediction_archive"])
@pytest.mark.parametrize("fault", ["outcome_metadata", "source_hash", "extraction_policy", "feature_descriptor"])
def test_rehashed_input_manifest_cannot_carry_outcomes_or_weaken_evidence(external_case, stage, fault):
    from eeg_ma import external
    root = external_case["root"]
    bundle = root / "prepared/inputs" if stage == "live_prediction" else root / "predictions"
    path = bundle / "input_manifest.json"
    manifest = json.loads(path.read_text())
    if fault == "outcome_metadata":
        manifest["true_labels"] = [0, 1]
    elif fault == "source_hash":
        manifest["source_feature_sha256"] = "unknown"
    elif fault == "extraction_policy":
        manifest["extraction_policy"] = "outcome_selected_epochs_after_target_fit"
    else:
        manifest["features"]["true_labels"] = [0, 1]
    write_json(path, _rehash(manifest, "input_id"))
    if stage == "prediction_archive":
        outer_path = bundle / "prediction_manifest.json"
        outer = json.loads(outer_path.read_text())
        outer["input_id"] = manifest["input_id"]
        outer["artifacts"]["input_manifest.json"] = {"sha256": sha256_file(path), "bytes": path.stat().st_size}
        write_json(outer_path, _rehash(outer, "prediction_id"))
        with pytest.raises(external.ExternalError):
            external.validate_predictions(bundle, lock_dir=root / "lock")
    else:
        with pytest.raises(external.ExternalError):
            external.predict_external(bundle, root / "rejected",
                lock_dir=root / "lock", cohort_manifest_path=root / "cohort.json")
        assert not (root / "rejected").exists()
    assert protocol.validate_lock(root / "lock")["events"] == []


@pytest.mark.parametrize("fault", ["missing_label", "duplicate_label", "foreign_epoch", "nonbinary_label", "ledger_counts"])
def test_labels_require_exact_prediction_coverage_without_selective_masks(external_case, fault):
    root = external_case["root"]
    def transform(frame):
        if fault == "missing_label":
            return frame.iloc[:-1]
        if fault == "duplicate_label":
            return pd.concat([frame, frame.iloc[[0]]], ignore_index=True)
        frame = frame.copy()
        if fault == "foreign_epoch":
            frame.loc[0, "epoch_id"] = "epoch-" + "f" * 64
        elif fault == "nonbinary_label":
            frame.loc[0, "true_label"] = 2
        else:
            frame.loc[0, "true_label"] = 1 - frame.loc[0, "true_label"]
        return frame
    _rewrite_labels(root / "prepared/labels", transform)
    with pytest.raises(ValueError):
        _score(external_case, "rejected")
    assert not (root / "rejected").exists()
    # Opening invalid outcomes is still an exposure event, never a silent retry
    # opportunity to fit or select a different model after looking at labels.
    assert len(protocol.validate_lock(root / "lock")["events"]) == 1


@pytest.mark.parametrize("fault", ["wrong_label", "subject_case", "epoch_case", "nonbinary_float"])
def test_rehashed_wrong_predictions_fail_semantic_model_recomputation(external_case, monkeypatch, fault):
    from eeg_ma import external
    root = external_case["root"] / "predictions"
    path = root / "predictions.csv"
    frame = read_csv_literal(path)
    if fault == "wrong_label":
        frame.loc[0, "predicted_label"] = 1 - int(frame.loc[0, "predicted_label"])
    elif fault == "nonbinary_float":
        frame["predicted_label"] = frame.predicted_label.astype(float)
        frame.loc[0, "predicted_label"] += 1e-13
    else:
        column = "subject_id" if fault == "subject_case" else "epoch_id"
        frame.loc[0, column] = frame.loc[0, column].upper()
    frame.to_csv(path, index=False)
    manifest_path = root / "prediction_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["artifacts"]["predictions.csv"] = {"sha256": sha256_file(path), "bytes": path.stat().st_size}
    write_json(manifest_path, _rehash(manifest, "prediction_id"))
    with pytest.raises(ValueError):
        external.validate_predictions(root)
    def no_labels(*args, **kwargs):
        raise AssertionError("Invalid predictions must be rejected before outcome access")
    monkeypatch.setattr(external, "_load_labels", no_labels)
    with pytest.raises(ValueError):
        _score(external_case, "rejected")
    assert protocol.validate_lock(external_case["root"] / "lock")["events"] == []


def test_immutable_predictions_and_scores_refuse_existing_destinations(external_case):
    from eeg_ma.external import predict_external
    root = external_case["root"]
    old = (root / "predictions/prediction_manifest.json").read_bytes()
    with pytest.raises((ValueError, FileExistsError)):
        predict_external(root / "prepared/inputs", root / "predictions", lock_dir=root / "lock", cohort_manifest_path=root / "cohort.json")
    assert (root / "predictions/prediction_manifest.json").read_bytes() == old
    _score(external_case)
    old = (root / "score/score_manifest.json").read_bytes()
    with pytest.raises((ValueError, FileExistsError)):
        _score(external_case)
    assert (root / "score/score_manifest.json").read_bytes() == old


def test_unapproved_second_valid_session_is_not_selected_after_seeing_performance(external_case, monkeypatch):
    from eeg_ma import external
    root = external_case["root"]
    path = root / "enrollment.json"
    enrollment = json.loads(path.read_text())
    second = deepcopy(enrollment["participants"][0]["sessions"][0])
    second.update(session_id="session-2", attempt_number=2, started_at="2020-01-02T09:00:00Z")
    enrollment["participants"][0]["sessions"].append(second)
    write_json(path, enrollment)
    def no_labels(*args, **kwargs):
        raise AssertionError("Invalid enrollment must be rejected before outcome access")
    monkeypatch.setattr(external, "_load_labels", no_labels)
    with pytest.raises(ValueError):
        _score(external_case, "rejected")
    assert not (root / "rejected").exists()
    assert protocol.validate_lock(root / "lock")["events"] == []


@pytest.mark.parametrize("fault", ["dirty", "changed_source", "missing_commit"])
def test_live_prediction_requires_the_locked_clean_source(external_case, monkeypatch, fault):
    from eeg_ma import external
    root = external_case["root"]
    code = synthetic_code()
    if fault == "dirty":
        code["dirty"] = True
    elif fault == "missing_commit":
        code["commit"] = None
    else:
        code["source_files"] = {"changed.py": "b" * 64}
        code["working_source_sha256"] = fingerprint(code["source_files"])
    monkeypatch.setattr(external, "code_identity", lambda: code)
    with pytest.raises(ValueError):
        external.predict_external(root / "prepared/inputs", root / "rejected",
            lock_dir=root / "lock", cohort_manifest_path=root / "cohort.json")
    assert not (root / "rejected").exists()


def test_outcome_exposure_blocks_new_prediction_and_projection(external_case):
    from eeg_ma.external import prepare_external_inputs, predict_external
    root = external_case["root"]
    _score(external_case)
    with pytest.raises(ValueError):
        predict_external(root / "prepared/inputs", root / "new-predictions",
            lock_dir=root / "lock", cohort_manifest_path=root / "cohort.json")
    with pytest.raises(ValueError):
        prepare_external_inputs(root / "external_features.csv", root / "new-inputs",
            lock_dir=root / "lock", cohort_manifest_path=root / "cohort.json")
    assert not (root / "new-predictions").exists()
    assert not (root / "new-inputs").exists()


@pytest.mark.parametrize("fault", ["balanced_accuracy", "literal_subject", "non_numeric_missing_score"])
def test_rehashed_score_is_recomputed_from_frozen_labels_and_enrollment(external_case, fault):
    from eeg_ma.external import validate_external_score
    root = external_case["root"]
    _score(external_case)
    path = root / "score/participant_scores.csv"
    rows = read_csv_literal(path)
    if fault == "balanced_accuracy":
        rows.loc[0, "balanced_accuracy"] = 1.0 - rows.loc[0, "balanced_accuracy"]
    elif fault == "literal_subject":
        rows.loc[0, "subject_id"] = rows.loc[0, "subject_id"].upper()
    else:
        rows["balanced_accuracy"] = rows.balanced_accuracy.astype(object)
        rows.loc[rows.index[~rows.evaluable][0], "balanced_accuracy"] = "not-a-score"
    rows.to_csv(path, index=False)
    manifest_path = root / "score/score_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["artifacts"][path.name] = {"sha256": sha256_file(path), "bytes": path.stat().st_size}
    write_json(manifest_path, _rehash(manifest, "score_id"))
    with pytest.raises(ValueError):
        validate_external_score(root / "score", prediction_dir=root / "predictions", lock_dir=root / "lock")


@pytest.mark.parametrize("timestamp", ["2019-12-31T23:59:59+00:00", "2199-01-01T00:00:00+00:00", "2025-01-01T00:00:00"])
def test_prediction_commit_requires_a_real_utc_time_after_lock(external_case, timestamp):
    from eeg_ma.external import validate_predictions
    root = external_case["root"]
    path = root / "predictions/prediction_manifest.json"
    manifest = json.loads(path.read_text())
    manifest["created_at"] = timestamp
    write_json(path, _rehash(manifest, "prediction_id"))
    with pytest.raises(ValueError):
        validate_predictions(root / "predictions", lock_dir=root / "lock")


@pytest.mark.parametrize("fault", ["before_exposure", "future", "inference_claim"])
def test_score_record_cannot_precede_exposure_or_claim_unimplemented_inference(external_case, fault):
    from eeg_ma.external import validate_external_score
    root = external_case["root"]
    _score(external_case)
    path = root / "score/score_manifest.json"
    manifest = json.loads(path.read_text())
    if fault == "before_exposure":
        manifest["created_at"] = LOCK_TIME
    elif fault == "future":
        manifest["created_at"] = "2199-01-01T00:00:00+00:00"
    else:
        manifest["inference_status"] = "confirmatory_superiority_established"
    write_json(path, _rehash(manifest, "score_id"))
    with pytest.raises(ValueError):
        validate_external_score(root / "score", prediction_dir=root / "predictions", lock_dir=root / "lock")


def test_participant_macro_balanced_accuracy_uses_individual_class_rates(external_case, monkeypatch):
    from eeg_ma.external import prepare_external_inputs, predict_external, score_external
    root = external_case["root"]
    _forbid_fit(monkeypatch)
    source = external_case["external"].copy()
    subjects = sorted(source.subject_id.unique())
    # Unequal, prospectively eligible task counts make pooled epoch accuracy
    # differ from the prespecified equal-weight participant estimand.
    for subject, count in zip(subjects, (20, 25, 30)):
        source = source.loc[~((source.subject_id == subject) & (source.class_label == 1)
                             & (source.epoch_index > count))]
    names = [column for column in source if column.startswith("PLV__")]
    source.loc[source.subject_id == subjects[0], names] = 1.0 - source.loc[source.subject_id == subjects[0], names]
    source.to_csv(root / "unequal_features.csv", index=False)
    enrollment_path = root / "enrollment.json"
    enrollment = json.loads(enrollment_path.read_text())
    for item, count in zip(enrollment["participants"][:3], (20, 25, 30)):
        session = item["sessions"][0]
        for task in ("Type1", "Type2"):
            session["correct"][task] = session["eligible_correct"][task] = count
            if count < 30:
                session["attempts"][task] = 45
                session["task_end"][task] = "attempt_limit"
    write_json(enrollment_path, enrollment)
    prepare_external_inputs(root / "unequal_features.csv", root / "unequal-inputs",
        lock_dir=root / "lock", cohort_manifest_path=root / "cohort.json")
    predict_external(root / "unequal-inputs/inputs", root / "unequal-predictions",
        lock_dir=root / "lock", cohort_manifest_path=root / "cohort.json")
    result = score_external(root / "unequal-predictions", root / "unequal-inputs/labels", root / "unequal-score",
        lock_dir=root / "lock", cohort_manifest_path=root / "cohort.json", enrollment_manifest_path=enrollment_path)
    predictions = read_csv_literal(root / "unequal-predictions/predictions.csv")
    labels = read_csv_literal(root / "unequal-inputs/labels/labels.csv")
    combined = predictions.merge(labels, on="epoch_id", validate="many_to_one")
    for summary in result["summary"]:
        group = combined.loc[(combined.comparison == summary["comparison"]) & (combined.model_role == summary["model_role"])]
        ba = [balanced_accuracy_score(person.true_label, person.predicted_label) for _, person in group.groupby("subject_id")]
        accuracy = [accuracy_score(person.true_label, person.predicted_label) for _, person in group.groupby("subject_id")]
        assert summary["participant_macro_balanced_accuracy"] == pytest.approx(np.mean(ba))
        assert summary["participant_macro_accuracy"] == pytest.approx(np.mean(accuracy))
        if summary["model_role"] == "primary":
            pooled = balanced_accuracy_score(group.true_label, group.predicted_label)
            assert abs(summary["participant_macro_balanced_accuracy"] - pooled) > 0.01


def test_cli_fit_prepare_predict_score_and_archive_verification(external_case, monkeypatch, capsys):
    root = external_case["root"]
    script = Path(__file__).resolve().parents[1] / "scripts/21_fixed_external.py"
    main = runpy.run_path(str(script))["main"]
    def command(*args):
        monkeypatch.setattr(sys, "argv", [str(script), *map(str, args)])
        assert main() == 0
        return json.loads(capsys.readouterr().out)
    fitted = command("fit-development", "--protocol", root / "development/protocol.json", "--out-dir", root / "cli-models")
    assert fitted["bundle_id"]
    _forbid_fit(monkeypatch)
    prepared = command("prepare-inputs", "--features", root / "external_features.csv", "--out-dir", root / "cli-inputs",
        "--lock-dir", root / "lock", "--cohort", root / "cohort.json")
    assert prepared["n_epochs"] == len(external_case["external"])
    predicted = command("predict-external", "--inputs", root / "cli-inputs/inputs", "--out-dir", root / "cli-predictions",
        "--lock-dir", root / "lock", "--cohort", root / "cohort.json")
    verified = command("verify-predictions", "--predictions", root / "cli-predictions", "--lock-dir", root / "lock")
    assert predicted["prediction_id"] == verified["prediction_id"]
    scored = command("score-external", "--predictions", root / "cli-predictions", "--labels", root / "cli-inputs/labels",
        "--enrollment", root / "enrollment.json", "--out-dir", root / "cli-scores", "--lock-dir", root / "lock", "--cohort", root / "cohort.json")
    verified_score = command("verify-score", "--scores", root / "cli-scores", "--predictions", root / "cli-predictions", "--lock-dir", root / "lock")
    assert scored["score_id"] == verified_score["score_id"]
    monkeypatch.setattr(sys, "argv", [str(script), "predict-external", "--inputs", str(root / "cli-inputs/inputs"),
        "--out-dir", str(root / "rejected-cli"), "--lock-dir", str(root / "lock"), "--cohort", str(root / "cohort.json")])
    assert main() == 2
    assert json.loads(capsys.readouterr().out)["status"] == "blocked"
    assert not (root / "rejected-cli").exists()
