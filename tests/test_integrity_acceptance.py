"""Independent adversarial acceptance tests for ingestion and run integrity.

These fixtures use explicit acquisition metadata and event clocks instead of
assuming that the historical experiment files provide verified provenance.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

from eeg_ma.config import load_config
from eeg_ma.extract import extract_all
from eeg_ma.evaluation import make_evaluation_splits, run_inter_subject, run_intra_subject
from eeg_ma.extensions import extract_all_extension_features, merge_baseline_and_extensions
from eeg_ma.final_validation import extract_all_iplv_features
from eeg_ma.io import load_session
from eeg_ma.qa import validate_session
from eeg_ma.schema import IDENTITY_COLUMNS, adapt_legacy_features, load_feature_table, validate_feature_table
from eeg_ma.provenance import (
    MatrixRun, ProvenanceError, RunTransaction, build_run_identity, fingerprint,
    legacy_inventory, resolve_report_root, table_contract, validate_complete_run,
)


CHANNELS = ["FP1", "FP2", "F7", "F3", "Fz", "F4", "F8", "Cz"]
REPO = Path(__file__).resolve().parents[1]


def _config() -> dict:
    cfg = load_config(REPO / "configs/lab_replication.json")
    cfg["expected_epochs_per_class"] = 2
    cfg["ingestion"] = {
        "clock_policy": "session_start",
        "marker_policy": "hardware_required",
        "timestamp_policy": "require_utc",
    }
    return cfg


def _write_session(
    root: Path,
    *,
    subject: str = "person",
    session: str = "session",
    cohort: str = "cohort",
) -> Path:
    root.mkdir(parents=True)
    fs = 500
    t = np.arange(fs * 45, dtype=float) / fs
    raw = pd.DataFrame({"sample_index": np.arange(len(t)), "elapsed_s": t})
    for i, channel in enumerate(CHANNELS):
        raw[channel] = 20 * np.sin(2 * np.pi * (8 + 0.23 * i) * t) + 2 * np.sin(2 * np.pi * 18 * t)
    raw.to_csv(root / "eeg_raw.csv", index=False)

    origin = datetime(2026, 11, 9, tzinfo=timezone.utc)

    def utc(seconds: float) -> str:
        return (origin + timedelta(seconds=seconds)).isoformat()

    def event(seconds: float, stage: str, phase: str, task: str, attempt: int, code: int) -> dict:
        return {
            "session_id": session, "subject_id": subject, "stage": stage,
            "phase": phase, "task_type": task, "attempt_index": attempt,
            "utc_timestamp": utc(seconds), "elapsed_ms": (seconds + 1) * 1000,
            "marker_code": code, "marker_sent": True, "marker_message": "synthetic",
        }

    events = [event(0, "Session", "SessionStart", "", 0, 1)]
    trials = []
    rests = []
    for stage, task, base, code in [
        ("Rest1", "", 0, 20), ("Type1", "Type1", 10, 11),
        ("Rest2", "", 20, 24), ("Type2", "Type2", 30, 12),
    ]:
        for index in (1, 2):
            seconds = base + 5 * (index - 1)
            events.append(event(seconds, stage, "Calculation" if task else "Rest", task, index if task else 0, code))
            if task:
                trials.append({
                    "session_id": session, "subject_id": subject, "task_type": task,
                    "attempt_index": index, "analysis_eligible": True, "response_correct": True,
                    "calculation_onset": utc(seconds),
                })
            else:
                rests.append({
                    "session_id": session, "subject_id": subject, "rest_stage": stage,
                    "rest_epoch_index": index, "onset_utc": utc(seconds), "epoch_seconds": 5,
                    "marker_code": code, "marker_sent": True, "marker_message": "synthetic",
                })
    pd.DataFrame(events).to_csv(root / "events.csv", index=False)
    pd.DataFrame(trials).to_csv(root / "trials.csv", index=False)
    pd.DataFrame(rests).to_csv(root / "rest_epochs.csv", index=False)
    (root / "session_info.txt").write_text(f"Subject: {subject}\nSampling rate: 500 Hz\n", encoding="utf-8")
    (root / "session_metadata.json").write_text(json.dumps({
        "schema_version": "2.0", "cohort_id": cohort, "subject_id": subject,
        "session_id": session, "protocol_id": "synthetic-hnc", "protocol_version": "1",
        "units": "uV", "reference": "synthetic-reference", "channel_order": CHANNELS,
    }), encoding="utf-8")
    return root


def _edit_csv(root: Path, name: str, mutate) -> None:
    frame = pd.read_csv(root / name, keep_default_na=False)
    mutate(frame)
    frame.to_csv(root / name, index=False)


@pytest.mark.parametrize("subject", ["001", "1", "NA", "null"])
def test_literal_identifiers_survive_ingestion(tmp_path, subject):
    path = _write_session(tmp_path / "raw", subject=subject, session="001")
    session = load_session(path, configured_fs=500)
    assert session.subject_id == subject
    assert set(session.events["subject_id"].astype(str)) == {subject}
    assert set(session.events["session_id"].astype(str)) == {"001"}
    report = validate_session(session, _config())
    assert report.passed


@pytest.mark.parametrize("fault", ["sample_gap", "noninteger_sample", "clock_reverse", "clock_duplicate", "clock_gap", "nonfinite", "flat"])
def test_single_sample_clock_and_signal_faults_fail_the_gate(tmp_path, fault):
    path = _write_session(tmp_path / "raw")

    def mutate(frame):
        if fault == "sample_gap":
            frame.loc[100:, "sample_index"] += 1
        elif fault == "noninteger_sample":
            frame["sample_index"] = frame["sample_index"].astype(float) + 0.5
        elif fault == "clock_reverse":
            frame.loc[100, "elapsed_s"] = frame.loc[99, "elapsed_s"] - 0.001
        elif fault == "clock_duplicate":
            frame.loc[100, "elapsed_s"] = frame.loc[99, "elapsed_s"]
        elif fault == "clock_gap":
            frame.loc[100:, "elapsed_s"] += 0.1
        elif fault == "nonfinite":
            frame.loc[100, "FP1"] = np.inf
        else:
            frame["FP1"] = 0.0

    _edit_csv(path, "eeg_raw.csv", mutate)
    with pytest.raises((ValueError, RuntimeError)):
        validate_session(load_session(path, configured_fs=500), _config()).raise_for_errors()


def test_duplicate_event_cannot_replace_a_missing_trial_even_when_counts_match(tmp_path):
    path = _write_session(tmp_path / "raw")

    def duplicate(frame):
        indices = frame.index[frame["marker_code"] == 11]
        frame.loc[indices[1]] = frame.loc[indices[0]]

    _edit_csv(path, "events.csv", duplicate)
    with pytest.raises((ValueError, RuntimeError)):
        validate_session(load_session(path, configured_fs=500), _config()).raise_for_errors()


@pytest.mark.parametrize("fault", ["unknown_calculation_marker", "unknown_rest_manifest_marker", "conflicting_calculation_stage"])
def test_unexpected_events_and_rest_manifest_rows_are_not_silently_filtered_out(tmp_path, fault):
    path = _write_session(tmp_path / "raw")
    if fault == "unknown_rest_manifest_marker":
        rows = pd.read_csv(path / "rest_epochs.csv", keep_default_na=False)
        extra = rows.iloc[[0]].copy()
        extra["marker_code"] = 999
        extra["rest_stage"] = "Extra"
        pd.concat([rows, extra], ignore_index=True).to_csv(path / "rest_epochs.csv", index=False)
    else:
        rows = pd.read_csv(path / "events.csv", keep_default_na=False)
        if fault == "unknown_calculation_marker":
            extra = rows.loc[rows.marker_code == 11].iloc[[0]].copy()
            extra["marker_code"] = 999
            extra["attempt_index"] = 99
            extra["elapsed_ms"] = 41000
            extra["utc_timestamp"] = "2026-11-09T00:00:40+00:00"
            rows = pd.concat([rows, extra], ignore_index=True)
        else:
            rows.loc[rows.marker_code == 11, "stage"] = "Type2"
        rows.to_csv(path / "events.csv", index=False)
    with pytest.raises((ValueError, RuntimeError)):
        validate_session(load_session(path, configured_fs=500), _config()).raise_for_errors()


@pytest.mark.parametrize("fault", ["extra_start_phase", "wrong_start_phase"])
def test_session_start_phase_and_marker_cannot_disagree(tmp_path, fault):
    path = _write_session(tmp_path / "raw")
    rows = pd.read_csv(path / "events.csv", keep_default_na=False)
    if fault == "extra_start_phase":
        extra = rows.iloc[[0]].copy()
        extra["marker_code"] = 999
        rows = pd.concat([extra, rows], ignore_index=True)
    else:
        rows.loc[rows.marker_code == 1, "phase"] = "Unrelated"
    rows.to_csv(path / "events.csv", index=False)
    with pytest.raises((ValueError, RuntimeError)):
        validate_session(load_session(path, configured_fs=500), _config()).raise_for_errors()


def test_two_recordings_keep_one_subject_and_distinct_epoch_identity(tmp_path):
    raw = tmp_path / "raw"
    _write_session(raw / "one", subject="001", session="session/a")
    _write_session(raw / "two", subject="001", session="session?a")
    output = tmp_path / "features"
    frame, _ = extract_all(raw, _config(), output)
    assert len(frame) == 16
    assert set(frame["subject_id"].astype(str)) == {"001"}
    assert set(frame["session_id"].astype(str)) == {"session/a", "session?a"}
    assert frame["epoch_id"].nunique() == 16
    assert frame["recording_id"].nunique() == 2
    assert set(frame.groupby("session_id").size()) == {8}


def test_preflight_prevents_first_session_output_when_second_session_is_invalid(tmp_path):
    raw = tmp_path / "raw"
    _write_session(raw / "a", session="a")
    broken = _write_session(raw / "b", session="b")
    _edit_csv(broken, "eeg_raw.csv", lambda frame: frame.__setitem__("FP1", 0.0))
    output = tmp_path / "features"
    with pytest.raises((ValueError, RuntimeError)):
        extract_all(raw, _config(), output)
    assert not list(output.rglob("*.csv"))


def test_all_feature_families_share_the_same_preflight_failure_and_no_partial_output(tmp_path):
    raw = tmp_path / "raw"
    _write_session(raw / "a", session="a")
    broken = _write_session(raw / "b", session="b")
    cfg = _config()
    baseline_output = tmp_path / "baseline"
    extract_all(raw, cfg, baseline_output)
    baseline_csv = baseline_output / "features_all.csv"
    before = _file_hashes(baseline_output)
    _edit_csv(broken, "eeg_raw.csv", lambda frame: frame.__setitem__("FP1", 0.0))
    extension_cfg = json.loads((REPO / "configs/extensions_e1_e2.json").read_text())
    calls = {
        "baseline": lambda out: extract_all(raw, cfg, out),
        "extensions": lambda out: extract_all_extension_features(raw, baseline_csv, cfg, extension_cfg, out),
        "iplv": lambda out: extract_all_iplv_features(raw, baseline_csv, cfg, out),
    }
    for family, call in calls.items():
        output = tmp_path / f"rejected-{family}"
        with pytest.raises(ValueError, match="FLAT_CHANNEL"):
            call(output)
        assert not output.exists()
    assert _file_hashes(baseline_output) == before


def _load_cli(filename: str):
    spec = importlib.util.spec_from_file_location("adversarial_" + filename.replace(".", "_"), REPO / "scripts" / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_validation_cli_reports_failed_gate_and_returns_nonzero(tmp_path, monkeypatch):
    raw = tmp_path / "raw"
    path = _write_session(raw / "session", subject="001")
    _edit_csv(path, "eeg_raw.csv", lambda frame: frame.loc.__setitem__((100, "elapsed_s"), frame.loc[99, "elapsed_s"]))
    cfg_path = tmp_path / "config.json"
    cfg_path.write_text(json.dumps(_config()))
    output = tmp_path / "qa.csv"
    module = _load_cli("01_validate_raw.py")
    monkeypatch.setattr(sys, "argv", ["validate", "--raw-root", str(raw), "--config", str(cfg_path), "--out", str(output)])
    assert module.main() == 1
    table = pd.read_csv(output, keep_default_na=False, dtype={"subject_id": str})
    assert table.subject_id.tolist() == ["001"]
    assert table.qc_status.tolist() == ["failed"]
    assert "SAMPLE_CLOCK" in table.error_codes.iloc[0]


@pytest.mark.parametrize("fault", ["missing_start", "duplicate_start", "events_subject", "trials_session", "rest_subject", "info_subject", "raw_channel_order", "unknown_boolean"])
def test_cross_file_identity_and_alignment_cannot_be_inferred_past_conflicts(tmp_path, fault):
    path = _write_session(tmp_path / "raw")
    if fault in {"missing_start", "duplicate_start"}:
        events = pd.read_csv(path / "events.csv", keep_default_na=False)
        if fault == "missing_start":
            events = events.loc[events["marker_code"] != 1]
        else:
            events = pd.concat([events.iloc[[0]], events], ignore_index=True)
        events.to_csv(path / "events.csv", index=False)
    elif fault == "events_subject":
        _edit_csv(path, "events.csv", lambda frame: frame.loc.__setitem__((1, "subject_id"), "different"))
    elif fault == "trials_session":
        _edit_csv(path, "trials.csv", lambda frame: frame.loc.__setitem__((0, "session_id"), "different"))
    elif fault == "rest_subject":
        _edit_csv(path, "rest_epochs.csv", lambda frame: frame.loc.__setitem__((0, "subject_id"), "different"))
    elif fault == "info_subject":
        (path / "session_info.txt").write_text("Subject: different\nSampling rate: 500 Hz\n")
    elif fault == "raw_channel_order":
        frame = pd.read_csv(path / "eeg_raw.csv")
        frame[["sample_index", "elapsed_s", "FP2", "FP1", *CHANNELS[2:]]].to_csv(path / "eeg_raw.csv", index=False)
    else:
        def unknown_boolean(frame):
            frame["marker_sent"] = frame["marker_sent"].astype(object)
            frame.loc[1, "marker_sent"] = "maybe"
        _edit_csv(path, "events.csv", unknown_boolean)
    with pytest.raises((ValueError, RuntimeError)):
        validate_session(load_session(path, configured_fs=500), _config()).raise_for_errors()


def test_software_marker_permission_is_explicit_and_does_not_hide_clock_errors(tmp_path):
    path = _write_session(tmp_path / "raw")
    _edit_csv(path, "events.csv", lambda frame: frame.loc.__setitem__((3, "marker_sent"), False))
    cfg = _config()
    assert not validate_session(load_session(path, configured_fs=500), cfg).passed
    cfg["ingestion"]["marker_policy"] = "software_timing"
    assert validate_session(load_session(path, configured_fs=500), cfg).passed
    _edit_csv(path, "eeg_raw.csv", lambda frame: frame.loc.__setitem__((100, "elapsed_s"), frame.loc[99, "elapsed_s"]))
    assert not validate_session(load_session(path, configured_fs=500), cfg).passed


def test_stable_epoch_join_preserves_rows_but_rejects_nonmatching_sets_and_metadata(tmp_path):
    raw = tmp_path / "raw"
    _write_session(raw / "session")
    baseline, _ = extract_all(raw, _config(), tmp_path / "features")
    extension = baseline.drop(columns=[c for c in baseline if c.startswith(("BP__", "COH__"))]).copy()
    extension["PLV__FP1__FP2__alpha"] = np.arange(len(extension)) / len(extension)
    expected = dict(zip(extension["epoch_id"], extension["PLV__FP1__FP2__alpha"]))
    merged = merge_baseline_and_extensions(baseline, extension.sample(frac=1, random_state=7))
    assert list(merged["epoch_id"]) == list(baseline["epoch_id"])
    assert list(merged["PLV__FP1__FP2__alpha"]) == [expected[e] for e in baseline["epoch_id"]]
    corrupted = [extension.iloc[:-1], pd.concat([extension, extension.iloc[[0]]], ignore_index=True)]
    metadata_conflict = extension.copy()
    metadata_conflict.loc[0, "reference"] = "different-reference"
    corrupted.append(metadata_conflict)
    extra = extension.iloc[[0]].copy()
    extra["epoch_id"] = "extra-unexpected-epoch"
    corrupted.append(pd.concat([extension, extra], ignore_index=True))
    for bad in corrupted:
        with pytest.raises((ValueError, RuntimeError)):
            merge_baseline_and_extensions(baseline, bad)


def test_legacy_feature_adaptation_is_explicit_and_never_invents_raw_evidence(tmp_path):
    raw = tmp_path / "raw"
    _write_session(raw / "session", subject="001")
    baseline, _ = extract_all(raw, _config(), tmp_path / "features")
    legacy = baseline.drop(columns=IDENTITY_COLUMNS)
    legacy_path = tmp_path / "legacy.csv"
    legacy.to_csv(legacy_path, index=False)
    original_bytes = legacy_path.read_bytes()
    with pytest.raises(ValueError, match="legacy"):
        load_feature_table(legacy_path)
    mapping = {
        "schema_version": "legacy-mapping-1",
        "subjects": {"001": {
            "cohort_id": "historical", "session_id": "mapped-session", "recording_id": "mapped-recording",
            "source_sha256": "unknown", "units": "unknown", "reference": "unknown",
            "protocol_id": "historical-protocol", "protocol_version": "unknown", "channel_order": "unknown",
        }},
    }
    adapted = load_feature_table(legacy_path, legacy_manifest=mapping)
    assert len(adapted) == len(legacy)
    assert set(adapted["subject_id"]) == {"001"}
    assert set(adapted["source_sha256"]) == {"unknown"}
    assert set(adapted["identity_origin"]) == {"legacy_manifest"}
    assert set(adapted["qc_status"]) == {"not_revalidated"}
    assert legacy_path.read_bytes() == original_bytes
    parsed_legacy = pd.read_csv(legacy_path, keep_default_na=False, dtype={"subject_id": str})
    for feature in [c for c in legacy if c.startswith(("BP__", "COH__"))]:
        np.testing.assert_array_equal(adapted[feature], parsed_legacy[feature])
    mapping["subjects"]["001"]["source_sha256"] = "a" * 64
    with pytest.raises(ValueError, match="raw|unknown"):
        load_feature_table(legacy_path, legacy_manifest=mapping)


def test_raw_source_hash_cannot_change_without_recording_identity(tmp_path):
    raw = tmp_path / "raw"
    _write_session(raw / "session")
    frame, _ = extract_all(raw, _config(), tmp_path / "features")
    altered = frame.copy()
    altered["source_sha256"] = "a" * 64
    with pytest.raises(ValueError, match="recording|source"):
        validate_feature_table(altered)


def _evaluation_fixture() -> tuple[pd.DataFrame, dict]:
    rng = np.random.default_rng(123)
    rows = []
    subjects = ["001", "1", "NA"]
    for subject in subjects:
        for label in (0, 1):
            for index in range(12):
                start = (label * 12 + index) * 2500
                rows.append({
                    "subject_id": subject, "comparison": "rest1_vs_type1", "class_label": label,
                    "class_name": "MentalArithmetic" if label else "Rest", "stage": "Type1" if label else "Rest1",
                    "epoch_index": index + 1, "attempt_index": index + 1 if label else 0,
                    "onset_elapsed_s": start / 500, "start_sample": start, "end_sample": start + 2500,
                    "BP__FP1__alpha": 2 * label + rng.normal(), "COH__FP1__FP2__alpha": rng.uniform(),
                })
    mapping = {"schema_version": "legacy-mapping-1", "subjects": {subject: {
        "cohort_id": "synthetic", "session_id": "same-session-label",
        "recording_id": f"explicit-synthetic-{subject}", "source_sha256": "unknown",
        "units": "unknown", "reference": "unknown", "protocol_id": "synthetic", "protocol_version": "1",
        "channel_order": "unknown",
    } for subject in subjects}}
    frame = adapt_legacy_features(pd.DataFrame(rows), mapping)
    cfg = _config()
    cfg["sfs"]["max_features"] = 1
    cfg["cv"]["intra_outer_folds"] = 2
    cfg["cv"]["intra_inner_folds"] = 2
    cfg["rbf_grid"] = {"C": [1.0], "gamma_values": [0.5]}
    cfg["n_jobs"] = 1
    return frame, cfg


@pytest.mark.parametrize("scope", ["intra", "inter"])
def test_recorded_split_memberships_match_prediction_identity_after_row_reordering(tmp_path, scope):
    """Protect historical splitter semantics and the new manifest/CSV mapping.

    Row reordering may change intra splits; every actual prediction must retain
    the identity of its source row and match the recorded split for this run.
    """
    from sklearn.model_selection import LeaveOneGroupOut, StratifiedKFold

    frame, cfg = _evaluation_fixture()
    frame = frame.sample(frac=1, random_state=17).reset_index(drop=True)
    manifest = make_evaluation_splits(frame, cfg, scope)
    runner = run_intra_subject if scope == "intra" else run_inter_subject
    _, predictions = runner(frame, cfg, tmp_path / scope)
    assert set(predictions["subject_id"]) == {"001", "1", "NA"}
    assert predictions.groupby("epoch_id").size().eq(3).all()
    indexed = frame.set_index("epoch_id")
    for column in ["subject_id", "session_id", "cohort_id", "recording_id", "protocol_id", "source_sha256"]:
        assert predictions[column].tolist() == indexed.loc[predictions["epoch_id"], column].tolist()
    assert predictions["true_label"].tolist() == indexed.loc[predictions["epoch_id"], "class_label"].tolist()

    # These are the historical sklearn split definitions, independent of the
    # new manifest helper. No training or feature-selection implementation is copied.
    groups = [frame] if scope == "inter" else [frame.loc[frame.subject_id == s].reset_index(drop=True) for s in sorted(frame.subject_id.unique())]
    expected = []
    for group in groups:
        y = group.class_label.to_numpy()
        if scope == "intra":
            splitter = StratifiedKFold(2, shuffle=True, random_state=cfg["cv"]["random_state"])
            outer = splitter.split(np.zeros((len(group), 1)), y)
        else:
            outer = LeaveOneGroupOut().split(np.zeros((len(group), 1)), y, group.subject_id.to_numpy())
        for fold, (train, test) in enumerate(outer, start=1):
            expected.append((fold, group.iloc[train].epoch_id.tolist(), group.iloc[test].epoch_id.tolist()))
    assert [(row["outer_fold"], row["train_epoch_ids"], row["test_epoch_ids"]) for row in manifest] == expected
    for split in manifest:
        rows = predictions.loc[predictions.outer_fold == split["outer_fold"]]
        if scope == "intra":
            rows = rows.loc[rows.subject_id == split["subject_id"]]
        else:
            assert not set(split["train_subject_ids"]) & set(split["test_subject_ids"])
        for _, model_rows in rows.groupby("classifier"):
            assert set(model_rows.epoch_id) == set(split["test_epoch_ids"])
        for inner in split["inner_splits"]:
            assert not set(inner["train_epoch_ids"]) & set(inner["test_epoch_ids"])
            assert set(inner["train_epoch_ids"]) | set(inner["test_epoch_ids"]) == set(split["train_epoch_ids"])


def _file_hashes(root: Path) -> dict[str, str]:
    return {p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in root.rglob("*") if p.is_file()}


def _report_identity(value: int = 1) -> tuple[dict, pd.DataFrame]:
    table = pd.DataFrame({"experiment": ["synthetic"], "value": [value]})
    return {"kind": "report", "fixture_version": value, "table_contracts": {"summary.csv": table_contract(table)}}, table


def test_run_resume_is_read_only_and_changed_identity_rejected_before_writes(tmp_path):
    identity, table = _report_identity()
    destination = tmp_path / "run"
    RunTransaction(destination, identity).execute(lambda stage: table.to_csv(stage / "summary.csv", index=False))
    before = _file_hashes(destination)

    def never_write(_):
        raise AssertionError("resume must not invoke the writer")

    RunTransaction(destination, identity, resume=True).execute(never_write)
    assert _file_hashes(destination) == before
    changed = copy.deepcopy(identity)
    changed["fixture_version"] = 2
    with pytest.raises(ProvenanceError, match="identity"):
        RunTransaction(destination, changed, resume=True).execute(never_write)
    assert _file_hashes(destination) == before


def test_run_identity_changes_for_data_order_input_bytes_config_and_environment(tmp_path):
    frame, cfg = _evaluation_fixture()
    source = tmp_path / "input.csv"
    frame.to_csv(source, index=False)
    options = {"scopes": ["inter"], "input_paths": {"features": source}, "code": {"commit": "fixture"}, "environment": {"lock": "fixture"}}
    original = build_run_identity(frame, cfg, **options)
    originals = fingerprint(original)
    changed_cfg = copy.deepcopy(cfg)
    changed_cfg["sfs"]["max_features"] = 2
    assert fingerprint(build_run_identity(frame, changed_cfg, **options)) != originals
    changed_values = frame.copy()
    changed_values.loc[0, "BP__FP1__alpha"] += 0.25
    assert fingerprint(build_run_identity(changed_values, cfg, **options)) != originals
    assert fingerprint(build_run_identity(frame.iloc[::-1].reset_index(drop=True), cfg, **options)) != originals
    assert fingerprint(build_run_identity(frame[frame.columns[::-1]], cfg, **options)) != originals
    assert fingerprint(build_run_identity(frame, cfg, **{**options, "environment": {"lock": "different"}})) != originals
    source.write_bytes(source.read_bytes() + b"\n")
    assert fingerprint(build_run_identity(frame, cfg, **options)) != originals


@pytest.mark.parametrize("failure", [RuntimeError, KeyboardInterrupt])
def test_interrupted_staging_is_not_complete_and_recovery_recomputes(tmp_path, failure):
    identity, table = _report_identity()
    destination = tmp_path / "run"

    def interrupted(stage):
        table.to_csv(stage / "summary.csv", index=False)
        raise failure("injected interruption")

    with pytest.raises(failure):
        RunTransaction(destination, identity).execute(interrupted)
    assert not destination.exists()
    staged = list(tmp_path.glob(".staging-run-*"))
    assert len(staged) == 1
    state = json.loads((staged[0] / "transaction_state.json").read_text())
    assert state["status"] == ("interrupted" if failure is KeyboardInterrupt else "failed")
    assert not (staged[0] / "run_manifest.json").exists()
    with pytest.raises(ProvenanceError):
        validate_complete_run(staged[0])
    RunTransaction(destination, identity, resume=True).execute(lambda stage: table.to_csv(stage / "summary.csv", index=False))
    assert validate_complete_run(destination)["status"] == "complete"


def test_failed_atomic_publish_cannot_leave_a_complete_run(tmp_path, monkeypatch):
    import eeg_ma.provenance as provenance

    identity, table = _report_identity()
    destination = tmp_path / "run"

    def fail_publish(*_):
        raise OSError("injected directory rename failure")

    with monkeypatch.context() as patch:
        patch.setattr(provenance.os, "rename", fail_publish)
        with pytest.raises(OSError, match="injected"):
            RunTransaction(destination, identity).execute(lambda stage: table.to_csv(stage / "summary.csv", index=False))
    assert not destination.exists()
    assert not list(tmp_path.glob(".staging-*/run_manifest.json"))
    RunTransaction(destination, identity, resume=True).execute(lambda stage: table.to_csv(stage / "summary.csv", index=False))
    validate_complete_run(destination)


def test_partial_matrix_preserves_validated_inventory_and_complete_report(tmp_path):
    first, first_table = _report_identity(1)
    second, second_table = _report_identity(2)
    identities = {"first": first, "second": second}
    tables = {"first": first_table, "second": second_table}
    destination = tmp_path / "matrix"

    def evaluate(name, stage):
        tables[name].to_csv(stage / "summary.csv", index=False)

    def report(inventory):
        return {"matrix.csv": pd.DataFrame({"experiment": sorted(inventory)})}

    _, first_report = MatrixRun(destination, {"matrix": "synthetic"}, identities, ["first"]).run(evaluate, report)
    manifest = validate_complete_run(first_report)
    assert manifest["identity"]["coverage"]["missing"] == ["second"]
    assert not manifest["identity"]["coverage"]["study_matrix_complete"]
    _, full_report = MatrixRun(destination, {"matrix": "synthetic"}, identities, ["second"]).run(evaluate, report)
    assert validate_complete_run(full_report)["identity"]["coverage"]["study_matrix_complete"]
    before = _file_hashes(destination)
    _, resumed_report = MatrixRun(destination, {"matrix": "synthetic"}, identities, ["first"], resume=True).run(evaluate, report)
    assert resumed_report == full_report
    assert resolve_report_root(destination) == full_report
    assert _file_hashes(destination) == before
    assert set(pd.read_csv(full_report / "matrix.csv")["experiment"]) == {"first", "second"}


def test_legacy_inventory_does_not_authorize_resume_or_verified_reporting(tmp_path):
    legacy = tmp_path / "legacy"
    legacy.mkdir()
    pd.DataFrame({"accuracy": [0.7]}).to_csv(legacy / "inter_folds.csv", index=False)
    inventory = legacy_inventory(legacy)
    assert inventory["status"] == "legacy_unverified"
    assert inventory["historical_environment"] == "unknown"
    assert inventory["raw_hashes"] == "unknown"
    identity, _ = _report_identity()
    with pytest.raises(ProvenanceError):
        RunTransaction(legacy, identity, resume=True).preflight()
    with pytest.raises(ProvenanceError):
        resolve_report_root(legacy)
    assert resolve_report_root(legacy, allow_legacy=True) == legacy


def test_canonical_protection_applies_to_children_and_symlink_aliases(tmp_path):
    project = tmp_path / "project"
    (project / "results").mkdir(parents=True)
    (project / "data" / "features").mkdir(parents=True)
    (project / "data" / "raw").mkdir(parents=True)
    identity, _ = _report_identity()
    for target in [project, project / "results" / "new-run", project / "data" / "features" / "new-run", project / "data" / "raw" / "new-run"]:
        with pytest.raises(ProvenanceError, match="Protected"):
            RunTransaction(target, identity, project_root=project)
    alias = tmp_path / "alias"
    try:
        alias.symlink_to(project / "results", target_is_directory=True)
    except OSError:
        pytest.skip("Symlink creation unavailable on this platform")
    with pytest.raises(ProvenanceError):
        RunTransaction(alias / "new-run", identity, project_root=project)
    assert not list((project / "results").iterdir())


@pytest.mark.parametrize("alias", [False, True])
def test_direct_evaluation_api_cannot_bypass_canonical_output_protection(tmp_path, monkeypatch, alias):
    import eeg_ma.provenance as provenance

    # A temporary project makes a failing guard safe to test without ever
    # sending an adversarial write to the actual historical results tree.
    project = tmp_path / "project"
    canonical = project / "results"
    canonical.mkdir(parents=True)
    monkeypatch.setattr(provenance, "PROJECT_ROOT", project)
    parent = canonical
    if alias:
        parent = tmp_path / "alias"
        parent.symlink_to(canonical, target_is_directory=True)
    frame, cfg = _evaluation_fixture()
    with pytest.raises(ProvenanceError):
        run_inter_subject(frame, cfg, parent / "unexpected")
    assert not list(canonical.iterdir())


def test_baseline_figure_legacy_fallback_records_every_consumed_summary(tmp_path, monkeypatch):
    root = tmp_path / "legacy-results"
    expected_sources = []
    for number, feature_set in enumerate(["all", "bp", "coh"]):
        base = root if feature_set == "all" else root / "feature_sets" / feature_set
        for scope in ["intra", "inter"]:
            folder = base / scope
            folder.mkdir(parents=True)
            if scope == "intra":
                name = "intra_overall_summary.csv"
                metrics = {"participant_accuracy_mean": [0.60 + number / 10], "participant_accuracy_std": [0.1], "n_subjects": [3]}
            else:
                name = "inter_summary.csv"
                metrics = {"lopo_accuracy_mean": [0.55 + number / 10], "lopo_accuracy_std": [0.1], "n_held_out_subjects": [3]}
            pd.DataFrame({"comparison": ["rest1_vs_type1"], "classifier": ["LDA"], **metrics}).to_csv(folder / name, index=False)
            expected_sources.append(folder / name)
    for scope in ["intra", "inter"]:
        pd.DataFrame({"feature": ["BP__FP1__alpha"], "selection_count": [1]}).to_csv(root / scope / f"{scope}_feature_frequency.csv", index=False)
    pd.DataFrame({"subject_id": ["001"], "accuracy": [0.8]}).to_csv(root / "behavioral_summary.csv", index=False)
    module = _load_cli("07_make_figures.py")
    for name in ["plot_all_feature_classification", "plot_sfs_top_features", "plot_connectivity_networks", "plot_behavioral", "plot_feature_set_comparison"]:
        monkeypatch.setattr(module, name, lambda *args, **kwargs: [])
    output = tmp_path / "figures"
    monkeypatch.setattr(sys, "argv", ["figures", "--results-root", str(root), "--out-dir", str(output), "--allow-legacy-results"])
    assert module.main() == 0
    inventory = json.loads((output / "figure_sources.json").read_text())["inputs"]
    recorded = {entry["sha256"] for entry in inventory.values()}
    assert {hashlib.sha256(path.read_bytes()).hexdigest() for path in expected_sources} <= recorded


def test_managed_final_report_cannot_use_unverified_alpha_frequency_even_with_legacy_flag(tmp_path, monkeypatch):
    root = tmp_path / "results"
    table = pd.DataFrame({"comparison": ["rest1_vs_type1"], "value": [1]})
    for family, filenames in {
        "extensions": ["extension_summary.csv"],
        "plv_decomposition": ["plv_decomposition_summary.csv"],
        "final_validation": ["final_validation_summary.csv", "delta_vs_poster_by_unit.csv"],
    }.items():
        experiment = "plv_alpha_only" if family == "final_validation" else "synthetic"
        identity = {"kind": "report", "table_contracts": {"inter/inter_feature_frequency.csv": table_contract(table)}}

        def evaluate(name, stage):
            (stage / "inter").mkdir()
            table.to_csv(stage / "inter/inter_feature_frequency.csv", index=False)

        MatrixRun(root / family, {"family": family}, {experiment: identity}, [experiment]).run(
            evaluate, lambda inventory, names=filenames: {filename: table for filename in names})
    # The immutable report remains readable, but its alpha experiment can no
    # longer prove that the separate frequency table belongs to that report.
    (root / "final_validation/plv_alpha_only/run_manifest.json").unlink()
    module = _load_cli("13_make_final_figures.py")
    for name in ["plot_analysis_story", "plot_band_ranking", "plot_alpha_subjects", "plot_phase_robustness", "plot_alpha_network"]:
        monkeypatch.setattr(module, name, lambda *args, **kwargs: [])
    output = tmp_path / "figures"
    monkeypatch.setattr(sys, "argv", ["figures", "--results-root", str(root), "--out-dir", str(output), "--allow-legacy-results"])
    with pytest.raises(ValueError, match="[Aa]lpha|[Mm]anifest|[Uu]nreadable"):
        module.main()
    assert not output.exists()
