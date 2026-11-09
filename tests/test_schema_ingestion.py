"""Extraction-family and historical-adapter contracts, without research reruns."""
from pathlib import Path
import json

import numpy as np
import pandas as pd
import pytest

from eeg_ma.extract import extract_all, feature_columns
from eeg_ma.extensions import extract_all_extension_features, merge_baseline_and_extensions
from eeg_ma.final_validation import extract_all_iplv_features
from eeg_ma.io import load_session
from eeg_ma.qa import validate_session
from eeg_ma.schema import assert_same_epochs, hash_file, load_feature_table, validate_feature_table
from test_extract_end_to_end import _write_session


REPO = Path(__file__).resolve().parents[1]


def _cfg():
    cfg = json.loads((REPO / "configs/lab_replication.json").read_text())
    cfg["expected_epochs_per_class"] = 2
    return cfg


def test_all_three_extraction_families_share_stable_session_epoch_identity(tmp_path):
    raw = tmp_path / "raw"
    first = raw / "a"
    second = raw / "b"
    first.mkdir(parents=True)
    second.mkdir()
    _write_session(first)
    _write_session(second)
    # Same person, same attempt numbering, separate acquisition sessions.
    for file in ["events.csv", "trials.csv", "rest_epochs.csv"]:
        frame = pd.read_csv(second / file, keep_default_na=False, dtype={"subject_id": str, "session_id": str})
        frame["session_id"] = "second"
        frame.to_csv(second / file, index=False)
    meta = json.loads((second / "session_metadata.json").read_text())
    meta["session_id"] = "second"
    (second / "session_metadata.json").write_text(json.dumps(meta))
    baseline, _ = extract_all(raw, _cfg(), tmp_path / "baseline")
    baseline_path = tmp_path / "baseline/features_all.csv"
    ext_cfg = json.loads((REPO / "configs/extensions_e1_e2.json").read_text())
    extension = extract_all_extension_features(raw, baseline_path, _cfg(), ext_cfg, tmp_path / "extensions")
    iplv = extract_all_iplv_features(raw, baseline_path, _cfg(), tmp_path / "iplv")
    assert len(baseline) == len(extension) == len(iplv) == 16
    assert baseline.subject_id.nunique() == 1
    assert baseline.session_id.nunique() == 2
    assert baseline.epoch_id.nunique() == 16
    assert_same_epochs(baseline, extension)
    assert_same_epochs(baseline, iplv)
    merged = merge_baseline_and_extensions(merge_baseline_and_extensions(baseline, extension), iplv)
    assert len(merged) == 16
    assert len(feature_columns(merged, "all")) == 168
    assert len(feature_columns(merged, "plv")) == 126
    assert len(feature_columns(merged, "iplv")) == 126
    assert len(list((tmp_path / "baseline").glob("subj__*_features.csv"))) == 2
    for name in ("baseline", "extensions", "iplv"):
        manifest = json.loads((tmp_path / name / "feature_manifest.json").read_text())
        assert len(manifest["sessions"]) == 2
        for file, digest in manifest["output_sha256"].items():
            assert hash_file(tmp_path / name / file) == digest


def test_canonical_legacy_adapter_is_explicit_pinned_and_lossless():
    mapping_path = REPO / "configs/legacy_discovery_mapping.json"
    manifest = json.loads(mapping_path.read_text())
    adapted = []
    before = {}
    for relative, expected_hash in manifest["feature_inputs"].items():
        path = REPO / relative
        before[relative] = hash_file(path)
        assert before[relative] == expected_hash
        with pytest.raises(ValueError, match="legacy-manifest"):
            load_feature_table(path)
        original = pd.read_csv(path, encoding="utf-8-sig", keep_default_na=False, dtype={"subject_id": str})
        transformed = load_feature_table(path, legacy_manifest=mapping_path)
        assert len(transformed) == 600
        assert transformed.subject_id.nunique() == 5
        assert transformed.epoch_id.nunique() == 600
        pd.testing.assert_frame_equal(transformed[original.columns], original)
        assert transformed.source_sha256.eq("unknown").all()
        assert transformed.qc_status.eq("not_revalidated").all()
        assert transformed.identity_origin.eq("legacy_manifest").all()
        assert transformed.attrs["source_feature_sha256"] == expected_hash
        adapted.append(transformed)
    for right in adapted[1:]:
        assert_same_epochs(adapted[0], right)
    merged = adapted[0]
    for right in adapted[1:]:
        merged = merge_baseline_and_extensions(merged, right)
    assert len(merged) == 600
    for relative, digest in before.items():
        assert hash_file(REPO / relative) == digest


def test_pinned_legacy_manifest_rejects_changed_input_bytes(tmp_path):
    source = REPO / "data/features/features_all.csv"
    changed = tmp_path / "features.csv"
    changed.write_bytes(source.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="hash is not approved"):
        load_feature_table(changed, legacy_manifest=REPO / "configs/legacy_discovery_mapping.json")


def test_source_sha_is_of_exact_parsed_raw_and_declared_conflicts_fail(tmp_path):
    _write_session(tmp_path)
    session = load_session(tmp_path, 500)
    assert session.metadata["source_sha256"] == hash_file(tmp_path / "eeg_raw.csv")
    assert session.source_hashes["events.csv"] == hash_file(tmp_path / "events.csv")
    meta_path = tmp_path / "session_metadata.json"
    meta = json.loads(meta_path.read_text())
    meta["source_sha256"] = "0" * 64
    meta_path.write_text(json.dumps(meta))
    with pytest.raises(ValueError, match="SOURCE_HASH_MISMATCH"):
        load_session(tmp_path, 500)


def test_duplicate_csv_header_is_rejected_before_pandas_mangles_it(tmp_path):
    _write_session(tmp_path)
    path = tmp_path / "events.csv"
    text = path.read_text(encoding="utf-8-sig")
    text = text.replace("marker_message", "subject_id", 1)
    path.write_text(text)
    with pytest.raises(ValueError, match="INVALID_CSV_HEADER"):
        load_session(tmp_path, 500)


def test_recording_metadata_cannot_change_between_epochs(tmp_path):
    _write_session(tmp_path)
    frame, _ = extract_all(tmp_path, _cfg(), tmp_path / "out")
    changed = frame.copy()
    changed.loc[1, "units"] = "uV"
    with pytest.raises(ValueError, match="inconsistent"):
        validate_feature_table(changed)


def test_acquisition_unknowns_are_explicit_and_do_not_rescale_values(tmp_path):
    _write_session(tmp_path)
    session = load_session(tmp_path, 500)
    report = validate_session(session, _cfg())
    assert report.passed
    assert {"UNITS_UNKNOWN", "REFERENCE_UNKNOWN"}.issubset(report.reason_codes)
    assert session.metadata["units"] == "unknown"
    assert session.metadata["reference"] == "unknown"
    original = pd.read_csv(tmp_path / "eeg_raw.csv")
    np.testing.assert_array_equal(session.raw["FP1"].to_numpy(), original["FP1"].to_numpy())


def test_correct_only_selection_has_disjoint_auditable_exclusion_counts(tmp_path):
    _write_session(tmp_path)
    trials = pd.read_csv(tmp_path / "trials.csv", keep_default_na=False)
    first = (trials.task_type == "Type1") & (trials.attempt_index == 2)
    second = (trials.task_type == "Type2") & (trials.attempt_index == 2)
    trials.loc[first, ["analysis_eligible", "response_correct"]] = False
    trials.loc[second, "response_correct"] = False
    trials.to_csv(tmp_path / "trials.csv", index=False)
    cfg = _cfg()
    cfg["expected_epochs_per_class"] = None
    frame, qa = extract_all(tmp_path, cfg, tmp_path / "out")
    exclusions = pd.read_csv(tmp_path / "out/excluded_trials.csv", keep_default_na=False)
    assert len(frame) == 6
    assert len(exclusions) == 2
    assert exclusions.trial_key.nunique() == 2
    assert set(exclusions.reason_code) == {"TRIAL_INELIGIBLE", "INCORRECT_RESPONSE"}
    assert exclusions.trial_source_sha256.eq(hash_file(tmp_path / "trials.csv")).all()
    for comparison in ("rest1_vs_type1", "rest2_vs_type2"):
        row = qa.iloc[0]
        assert row[comparison + "_total_attempts"] == (
            row[comparison + "_selected_attempts"] + row[comparison + "_excluded_ineligible"] + row[comparison + "_excluded_incorrect"]
        )
