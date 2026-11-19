from copy import deepcopy
from dataclasses import FrozenInstanceError
import json

import numpy as np
import pytest

from eeg_ma.phase_controls import (
    ANALYSIS_ID, CHANNELS, EPOCH_SCHEMA, POLICY_SCHEMA, SPLIT_SCHEMA,
    PhaseControlError, canonical_sha256, export_matched_phase_bundle,
    extract_matched_alpha_features, validate_matched_phase_manifests, verify_matched_phase_bundle,
)


def phase_fixture():
    """Numerical test inputs only; no synthetic performance or device claims."""
    rng = np.random.default_rng(831)
    subjects = ["fixture_001", "fixture_002", "fixture_003"]
    epochs = {"schema_version": EPOCH_SCHEMA, "epochs": []}
    for subject in subjects:
        for label in (0, 1):
            for repeat in range(2):
                index = 2 * label + repeat
                epochs["epochs"].append({
                    "epoch_id": f"{subject}:epoch:{index}", "cohort_id": "synthetic_phase_fixture",
                    "subject_id": subject, "session_id": "session_001", "recording_id": "recording_001",
                    "recording_sha256": canonical_sha256({"synthetic_subject": subject}),
                    "start_sample": 64 * index, "end_sample": 64 * (index + 1),
                    "comparison": "rest1_vs_type1", "class_label": label,
                    "qc_passed": True, "qc_reason_codes": [],
                })
    splits = {"schema_version": SPLIT_SCHEMA, "splits": []}
    for subject in subjects:
        training = [other for other in subjects if other != subject]
        splits["splits"].append({"split_id": "outer_" + subject, "train_subject_ids": training,
            "test_subject_ids": [subject], "inner_splits": [
                {"split_id": "inner_" + held, "train_subject_ids": [other for other in training if other != held],
                 "validation_subject_ids": [held]} for held in training]})
    policy = {"schema_version": POLICY_SCHEMA, "analysis_id": ANALYSIS_ID,
        "evidence_kind": "synthetic_fixture", "protocol_lock_id": None,
        "code_sha256": canonical_sha256("synthetic code identity"),
        "environment_sha256": canonical_sha256("synthetic environment identity"),
        "preprocessing_sha256": canonical_sha256("preprocessed synthetic phase fixture"),
        "selection_sha256": canonical_sha256("no model fitting in fixture"),
        "epoch_manifest_sha256": canonical_sha256(epochs), "split_manifest_sha256": canonical_sha256(splits),
        "channels": list(CHANNELS), "band_hz": [8.0, 13.0], "ciplv_min_denominator": 1e-6,
        "undefined_ciplv_policy": "exclude_epoch_from_all_representations",
        "min_epochs_per_class": 1, "max_epochs_per_class": 30}
    phases = rng.uniform(-np.pi, np.pi, (len(epochs["epochs"]), 64, 7))
    return phases, epochs, splits, policy


def extract(data):
    phases, epochs, splits, policy = data
    return extract_matched_alpha_features(phases, epochs, splits, input_policy=policy)


def refresh_policy(data):
    _, epochs, splits, policy = data
    policy["epoch_manifest_sha256"] = canonical_sha256(epochs)
    policy["split_manifest_sha256"] = canonical_sha256(splits)


def test_exact_matched_representation_order_identity_and_bounds():
    data = phase_fixture()
    bundle = extract(data)
    assert bundle.epoch_ids == tuple(row["epoch_id"] for row in data[1]["epochs"])
    assert bundle.evaluable
    bundle.require_evaluable()
    manifest = bundle.manifest
    assert manifest["input_policy"]["evidence_kind"] == "synthetic_fixture"
    assert manifest["contains_model_performance_results"] is False
    assert len(manifest["edge_audit"]) == 12 * 21
    assert manifest["numerical_contract"]["denominator_epsilon_added"] is False
    hashes = []
    for name in ("plv", "iplv", "ciplv"):
        matrix = bundle.matrix(name)
        assert matrix.shape == (12, 21)
        assert np.isfinite(matrix).all()
        assert np.all((0 <= matrix) & (matrix <= 1))
        info = manifest["representations"][name]
        assert info["columns"][0] == f"{name.upper()}__FP1__FP2__alpha"
        assert info["columns"][-1] == f"{name.upper()}__F4__F8__alpha"
        hashes.append(info["shared_provenance_sha256"])
    assert hashes == [manifest["shared_provenance_sha256"]] * 3
    for outer in manifest["applied_splits"]:
        assert not set(outer["train"]["subject_ids"]) & set(outer["test"]["subject_ids"])
        assert not set(outer["train"]["epoch_ids"]) & set(outer["test"]["epoch_ids"])


def test_source_qc_and_undefined_ciplv_share_one_exclusion_mask():
    data = phase_fixture()
    phases, epochs, _, _ = data
    phases[0, :, 1] = phases[0, :, 0]
    epochs["epochs"][1]["qc_passed"] = False
    epochs["epochs"][1]["qc_reason_codes"] = ["SYNTHETIC_QC_FAILURE"]
    refresh_policy(data)
    bundle = extract(data)
    assert len(bundle.epoch_ids) == 10
    assert bundle.plv.shape == bundle.iplv.shape == bundle.ciplv.shape == (10, 21)
    manifest = bundle.manifest
    assert manifest["exclusions"][0]["reason_codes"] == ["UNDEFINED_CIPLV_COMMON_EXCLUSION"]
    assert manifest["exclusions"][0]["undefined_pairs"][0]["channels"] == ["FP1", "FP2"]
    assert manifest["exclusions"][1]["source_qc_reason_codes"] == ["SYNTHETIC_QC_FAILURE"]
    first_edge = manifest["edge_audit"][0]["metrics"]
    assert first_edge["plv"] == 1
    assert first_edge["iplv"] == 0
    assert first_edge["ciplv"] is None
    assert first_edge["ciplv_status"] == "undefined"
    assert manifest["split_manifest"] == data[2]


def test_post_exclusion_missing_class_is_audited_without_replanning():
    data = phase_fixture()
    for row in data[1]["epochs"][:2]:
        row["qc_passed"] = False
        row["qc_reason_codes"] = ["SYNTHETIC_QC_FAILURE"]
    refresh_policy(data)
    bundle = extract(data)
    assert not bundle.evaluable
    with pytest.raises(PhaseControlError, match="unevaluable"):
        bundle.require_evaluable()
    manifest = bundle.manifest
    assert manifest["split_manifest"] == data[2]
    assert any(finding["missing_class_labels"] == [0] for finding in manifest["evaluation_findings"])
    assert len(manifest["applied_splits"]) == 3


def test_public_manifest_validation_rejects_leakage_before_phase_loading():
    _, epochs, splits, policy = phase_fixture()
    audit = validate_matched_phase_manifests(epochs, splits, input_policy=policy, n_samples=64)
    assert audit["n_epochs"] == 12
    assert audit["source_class_counts"][0]["class_counts"] == {"0": 2, "1": 2}
    splits["splits"][0]["train_subject_ids"].append("fixture_001")
    policy["split_manifest_sha256"] = canonical_sha256(splits)
    with pytest.raises(PhaseControlError, match="leakage"):
        validate_matched_phase_manifests(epochs, splits, input_policy=policy, n_samples=64)


def test_prospective_post_qc_minimum_cannot_be_lowered_or_silently_drop_people():
    data = phase_fixture()
    data[3].update(evidence_kind="prospective_data", protocol_lock_id="synthetic-only-test-lock", min_epochs_per_class=20)
    bundle = extract(data)
    assert not bundle.evaluable
    assert bundle.epoch_ids == tuple(row["epoch_id"] for row in data[1]["epochs"])
    assert bundle.manifest["split_manifest"] == data[2]
    findings = bundle.manifest["evaluation_findings"]
    assert len(findings) == 3
    assert all(item["code"] == "UNEVALUABLE_PARTICIPANT_EPOCH_COUNT" for item in findings)
    assert all(item["class_counts"] == {"0": 2, "1": 2} for item in findings)
    with pytest.raises(PhaseControlError, match="unevaluable"):
        bundle.require_evaluable()
    data[3]["min_epochs_per_class"] = 1
    with pytest.raises(PhaseControlError, match="20–30"):
        extract(data)


def test_source_class_count_cannot_exceed_fixed_acquisition_ceiling():
    data = list(phase_fixture())
    original = data[1]["epochs"][0]
    for index in range(4, 34):
        row = {**original, "epoch_id": f"extra_epoch_{index}", "start_sample": 64 * index,
               "end_sample": 64 * (index + 1)}
        data[1]["epochs"].append(row)
    data[0] = np.zeros((len(data[1]["epochs"]), 64, 7))
    refresh_policy(data)
    with pytest.raises(PhaseControlError, match="acquisition maximum"):
        validate_matched_phase_manifests(data[1], data[2], input_policy=data[3], n_samples=64)


def test_empty_common_retention_remains_an_explicit_nonevaluable_artifact(tmp_path):
    data = phase_fixture()
    data[0][:] = 0
    bundle = extract(data)
    assert bundle.epoch_ids == ()
    assert not bundle.evaluable
    assert bundle.plv.shape == bundle.iplv.shape == bundle.ciplv.shape == (0, 21)
    assert len(bundle.manifest["exclusions"]) == 12
    target = export_matched_phase_bundle(bundle, tmp_path / "empty_common_features")
    assert not verify_matched_phase_bundle(target).evaluable


def test_bundle_does_not_expose_mutable_arrays_or_metadata():
    data = phase_fixture()
    bundle = extract(data)
    original = bundle.plv.copy()
    with pytest.raises(ValueError):
        bundle.plv[0, 0] = 0
    with pytest.raises(ValueError):
        bundle.plv.setflags(write=True)
    with pytest.raises(FrozenInstanceError):
        bundle._n_rows = 1
    view = bundle.plv
    view.shape = (252,)
    assert bundle.plv.shape == (12, 21)
    metadata = bundle.manifest
    metadata["accepted_epoch_ids"].clear()
    data[0][:] = 0
    data[1]["epochs"][0]["subject_id"] = "changed"
    data[3]["channels"].reverse()
    assert len(bundle.epoch_ids) == 12
    np.testing.assert_array_equal(bundle.plv, original)


@pytest.mark.parametrize("mutation", ["duplicate_epoch", "bad_span", "duplicate_source", "bad_recording_hash",
    "bad_subject_literal", "bad_class", "qc_failure_without_reason", "qc_pass_with_reason", "two_cohorts"])
def test_invalid_epoch_identity_or_qc_rejected(mutation):
    data = phase_fixture()
    rows = data[1]["epochs"]
    if mutation == "duplicate_epoch":
        rows[1]["epoch_id"] = rows[0]["epoch_id"]
    elif mutation == "bad_span":
        rows[0]["end_sample"] -= 1
    elif mutation == "duplicate_source":
        rows[1]["start_sample"], rows[1]["end_sample"] = rows[0]["start_sample"], rows[0]["end_sample"]
    elif mutation == "bad_recording_hash":
        rows[1]["recording_sha256"] = "f" * 64
    elif mutation == "bad_subject_literal":
        rows[0]["subject_id"] = " fixture_001"
    elif mutation == "bad_class":
        rows[0]["class_label"] = True
    elif mutation == "qc_failure_without_reason":
        rows[0]["qc_passed"] = False
    elif mutation == "qc_pass_with_reason":
        rows[0]["qc_reason_codes"] = ["bad"]
    elif mutation == "two_cohorts":
        rows[0]["cohort_id"] = "different_cohort"
    refresh_policy(data)
    with pytest.raises(PhaseControlError):
        extract(data)


@pytest.mark.parametrize("mutation", ["outer_overlap", "unknown_subject", "incomplete_outer", "repeat_test",
    "inner_leaks_outer_test", "inner_overlap", "incomplete_inner", "duplicate_outer_id"])
def test_participant_partition_membership_leakage_and_coverage_rejected(mutation):
    data = phase_fixture()
    outer = data[2]["splits"]
    if mutation == "outer_overlap":
        outer[0]["train_subject_ids"].append(outer[0]["test_subject_ids"][0])
    elif mutation == "unknown_subject":
        outer[0]["train_subject_ids"][0] = "unknown_subject"
    elif mutation == "incomplete_outer":
        outer.pop()
    elif mutation == "repeat_test":
        outer[1] = deepcopy(outer[0])
        outer[1]["split_id"] = "different_name_same_test"
    elif mutation == "inner_leaks_outer_test":
        outer[0]["inner_splits"][0]["train_subject_ids"] = outer[0]["test_subject_ids"]
    elif mutation == "inner_overlap":
        inner = outer[0]["inner_splits"][0]
        inner["train_subject_ids"] = inner["validation_subject_ids"]
    elif mutation == "incomplete_inner":
        outer[0]["inner_splits"].pop()
    elif mutation == "duplicate_outer_id":
        outer[1]["split_id"] = outer[0]["split_id"]
    refresh_policy(data)
    with pytest.raises(PhaseControlError):
        extract(data)


@pytest.mark.parametrize("field,value", [
    ("channels", list(reversed(CHANNELS))), ("band_hz", [8.0, 12.0]),
    ("ciplv_min_denominator", 1e-8), ("undefined_ciplv_policy", "replace_with_zero"),
    ("selection_sha256", "missing"), ("evidence_kind", "legacy_exploration"),
    ("epoch_manifest_sha256", "0" * 64), ("split_manifest_sha256", "0" * 64),
])
def test_scientific_scope_and_provenance_policy_cannot_silently_drift(field, value):
    data = phase_fixture()
    data[3][field] = value
    with pytest.raises(PhaseControlError):
        extract(data)


def test_prospective_input_requires_a_named_lock_but_does_not_claim_to_authenticate_it():
    data = phase_fixture()
    data[3]["evidence_kind"] = "prospective_data"
    with pytest.raises(PhaseControlError, match="protocol_lock_id"):
        extract(data)


@pytest.mark.parametrize("replacement", [
    np.zeros((12, 64, 6)), np.zeros((12, 0, 7)), np.zeros((0, 64, 7)),
    np.zeros((12, 64)), np.zeros((12, 64, 7), dtype=complex),
    np.full((12, 64, 7), "0"), np.full((12, 64, 7), np.nan),
    np.full((12, 64, 7), np.inf),
])
def test_phase_shape_real_type_and_finiteness_enforced(replacement):
    data = list(phase_fixture())
    data[0] = replacement
    with pytest.raises(PhaseControlError):
        extract(data)


def test_phase_tensor_or_frozen_selection_change_provenance():
    data = phase_fixture()
    first = extract(data)
    data[0][0, 0, 0] += 0.1
    changed_phases = extract(data)
    assert first.manifest["source_phases"]["sha256"] != changed_phases.manifest["source_phases"]["sha256"]
    data[3]["selection_sha256"] = canonical_sha256("another explicitly frozen selection contract")
    changed_selection = extract(data)
    assert changed_phases.manifest["shared_provenance_sha256"] != changed_selection.manifest["shared_provenance_sha256"]


def test_atomic_export_roundtrip_and_no_overwrite(tmp_path):
    bundle = extract(phase_fixture())
    path = export_matched_phase_bundle(bundle, tmp_path / "matched_features")
    restored = verify_matched_phase_bundle(path)
    assert restored.manifest == bundle.manifest
    assert restored.epoch_ids == bundle.epoch_ids
    for name in ("plv", "iplv", "ciplv"):
        np.testing.assert_array_equal(restored.matrix(name), bundle.matrix(name))
    with pytest.raises(FileExistsError):
        export_matched_phase_bundle(bundle, path)
    assert verify_matched_phase_bundle(path).manifest == bundle.manifest


def test_export_corrupted_matrix_and_extra_inventory_rejected(tmp_path):
    bundle = extract(phase_fixture())
    path = export_matched_phase_bundle(bundle, tmp_path / "corrupt")
    with (path / "plv.npy").open("ab") as stream:
        stream.write(b"unexpected bytes")
    with pytest.raises(PhaseControlError, match="Corrupted"):
        verify_matched_phase_bundle(path)
    path = export_matched_phase_bundle(bundle, tmp_path / "extra")
    (path / "untracked.txt").write_text("not in the numerical artifact")
    with pytest.raises(PhaseControlError, match="inventory"):
        verify_matched_phase_bundle(path)


def test_rehashed_but_inconsistent_numerical_evidence_rejected(tmp_path):
    bundle = extract(phase_fixture())
    path = export_matched_phase_bundle(bundle, tmp_path / "inconsistent_audit")
    exported = json.loads((path / "manifest.json").read_text())
    exported["bundle"]["edge_audit"][0]["metrics"]["iplv"] = 0.0
    exported["bundle"].pop("bundle_sha256")
    exported["bundle"]["bundle_sha256"] = canonical_sha256(exported["bundle"])
    exported.pop("export_sha256")
    exported["export_sha256"] = canonical_sha256(exported)
    (path / "manifest.json").write_text(json.dumps(exported))
    with pytest.raises(PhaseControlError, match="formulas"):
        verify_matched_phase_bundle(path)
