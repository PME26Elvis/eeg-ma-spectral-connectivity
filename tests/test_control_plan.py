"""Prospective B planning guards; every completed plan here is SYNTHETIC."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

import eeg_ma.control_plan as control
from eeg_ma.provenance import fingerprint, sha256_file
from test_environment import synthetic_reference_environment
from test_protocol_gate import synthetic_code

ROOT = Path(__file__).resolve().parents[1]
LOCKED_AT = "2020-01-01T00:00:00+00:00"


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def ready_plan(tmp_path, *, sensitivities=False):
    spec = control.read_plan(ROOT / "configs/phase_control_plan.json")
    spec["status"] = "ready"
    spec["data"].update(availability="prospective_arranged_no_acquisition", cohort_id="synthetic-B")
    spec["acquisition"].update(device="SYNTHETIC device", units="uV", reference="SYNTHETIC reference", ground="SYNTHETIC ground",
                               artifact_annotations_available=True, effective_rank=7, verified=True)
    spec["procedure"]["response_seconds"] = 3.0
    spec["recruitment"].update(target_unique_participants=4, cap_unique_participants=6, resource_and_sample_size_rationale="SYNTHETIC descriptive fixture, not a real power calculation")
    spec["approval"].update(registered_at="2019-12-01T00:00:00Z", registration_reference="SYNTHETIC registration",
        approved_at="2019-12-01T00:00:00Z", data_use_terms="SYNTHETIC test data only", resources_confirmed=True)
    spec["matrix"]["reference_sensitivity"].update(decision="enabled" if sensitivities else "disabled", reason="SYNTHETIC feasibility decision")
    spec["matrix"]["artifact_sensitivity"].update(decision="enabled" if sensitivities else "disabled", reason="SYNTHETIC feasibility decision", annotation_ontology="SYNTHETIC blink/vigilance ontology")
    refresh_evidence(tmp_path, spec)
    return tmp_path / "plan.json", spec


def refresh_evidence(tmp_path, spec):
    payloads = {
        "acquisition_feasibility": {"schema_version": "eeg-control-feasibility-1", "verified": True,
            "acquisition_sha256": fingerprint(spec["acquisition"]), "procedure_sha256": fingerprint(spec["procedure"]),
            "sensitivity_sha256": fingerprint(spec["matrix"]), "outcomes_used": False},
        "data_use_approval": {"schema_version": "eeg-control-approval-1", "study_id": spec["study_id"],
            "approval_sha256": fingerprint(spec["approval"]), "approved": True},
        "participant_registry": {"schema_version": "eeg-control-registry-1", "cohort_id": spec["data"]["cohort_id"],
            "excluded_subject_ids": spec["data"]["excluded_subject_ids"], "cross_cohort_identity_checked": True},
    }
    for name, value in payloads.items():
        value["evidence_reference"] = "SYNTHETIC fixture only"
        path = tmp_path / f"{name}.json"
        write(path, value)
        spec["evidence"][name] = {"path": path.name, "sha256": sha256_file(path)}
    write(tmp_path / "plan.json", spec)


@pytest.fixture
def frozen(tmp_path, monkeypatch):
    path, spec = ready_plan(tmp_path / "source")
    monkeypatch.setattr(control, "_now", lambda: LOCKED_AT)
    monkeypatch.setattr(control, "code_identity", synthetic_code)
    monkeypatch.setattr(control, "environment_identity", synthetic_reference_environment)
    destination = tmp_path / "frozen"
    control.freeze_control_plan(path, destination)
    monkeypatch.setattr(control, "_now", lambda: "2020-01-04T00:00:00Z")
    return destination, path, spec


def history(spec, *, paired=False):
    people = []
    sequence = control.allocation_schedule(spec)["slots"]
    for i, subject in enumerate(("001", "1", "NA", "lost")):
        sessions = []
        if subject != "lost":
            if paired:
                sessions.append({"session_id": subject + "-A", "protocol": "A", "started_at": "2020-01-02T10:00:00Z",
                                 "qa_passed": True, "qc_reason_codes": [], "sequence_index": None, "eligible_epochs_per_class": {"0": 30, "1": 30},
                                 "recording_id": subject + "-recording-A", "recording_sha256": "a" * 64, "qa_evidence_sha256": "b" * 64, "eligibility_manifest_sha256": "c" * 64})
            sessions.append({"session_id": subject + "-B", "protocol": "B", "started_at": "2020-01-03T10:00:00Z",
                             "qa_passed": True, "qc_reason_codes": [], "sequence_index": sequence[i]["sequence_index"], "eligible_epochs_per_class": {"0": 30, "1": 30},
                             "recording_id": subject + "-recording-B", "recording_sha256": "a" * 64, "qa_evidence_sha256": "b" * 64, "eligibility_manifest_sha256": "c" * 64})
        people.append({"subject_id": subject, "recruited_at": f"2020-01-02T09:0{i}:00Z", "disposition": "lost" if subject == "lost" else "completed", "sessions": sessions})
    return {"schema_version": "eeg-control-session-history-1", "cohort_id": spec["data"]["cohort_id"], "closed": True, "participants": people,
            "stop_reason": "administrative_shortfall", "stop_reason_detail": "SYNTHETIC shortfall fixture; not a completed real cohort"}


def result_rows(spec):
    return [{"cell_id": cell["cell_id"], "analysis_id": spec["analysis_id"], "protocol_lock_id": "synthetic-lock",
             "run_id": "synthetic-run-" + cell["representation"], "model_id": "synthetic-model-" + cell["representation"],
             "participant_ids": ["001", "1", "NA"], "status": "evaluable", "reason": None,
             "eligible_epoch_manifest_sha256": "1" * 64, "split_manifest_sha256": "2" * 64, "preprocessing_sha256": "3" * 64,
             "selection_sha256": fingerprint(spec["selection"]), "source_predictions_sha256": "4" * 64} for cell in control.expected_cells(spec)]


def test_real_draft_remains_blocked_and_does_not_activate_A(tmp_path):
    original = (ROOT / "configs/confirmatory_protocol.json").read_bytes()
    spec = control.read_plan(ROOT / "configs/phase_control_plan.json")
    report = control.check_control_plan(spec, base_dir=ROOT / "configs")
    assert not report["ready_to_lock"]
    assert {"data.availability", "acquisition.reference", "acquisition.ground", "acquisition.effective_rank", "recruitment", "matrix.reference_sensitivity", "matrix.artifact_sensitivity"} <= {x["path"] for x in report["findings"]}
    with pytest.raises(control.ControlPlanError):
        control.freeze_control_plan(ROOT / "configs/phase_control_plan.json", tmp_path / "must-not-exist")
    assert not (tmp_path / "must-not-exist").exists()
    assert (ROOT / "configs/confirmatory_protocol.json").read_bytes() == original
    assert not json.loads(original)["protocols"]["B"]["enabled"]


@pytest.mark.parametrize("path,value", [
    ("matrix.representations", ["plv", "iplv", "ciplv", "wpli"]),
    ("matrix.classifiers", ["LDA", "RBF-SVM"]),
    ("preprocessing.phase_band_hz", [7.0, 14.0]),
    ("preprocessing.ciplv_min_denominator", 0.0),
    ("selection.target_fit", True),
    ("reporting.report_all_signs", False),
    ("reporting.stop_after_matrix", False),
    ("data.outcomes_accessed", True),
    ("data.collection_started_at", "2019-12-25T00:00:00Z"),
    ("procedure.sequences", [["Rest1", "Type1", "Rest2", "Type2"]]),
    ("procedure.independent_N_increment_for_repeat_sessions", 1),
])
def test_unplanned_scientific_choices_and_posthoc_activation_are_rejected(tmp_path, path, value):
    _, spec = ready_plan(tmp_path)
    section, key = path.split(".")
    spec[section][key] = value
    report = control.check_control_plan(spec)
    assert path in {x["path"] for x in report["findings"]}


def test_new_cohort_cannot_remove_old_five_or_exposed_people(tmp_path):
    _, spec = ready_plan(tmp_path)
    spec["data"]["excluded_subject_ids"].remove("lin")
    spec["data"]["prior_exposed_subject_ids"] = ["new-person-already-seen"]
    assert "data.excluded_subject_ids" in {x["path"] for x in control.check_control_plan(spec)["findings"]}


def test_matrix_is_bounded_to_nine_without_factorial_crossing(tmp_path):
    _, spec = ready_plan(tmp_path, sensitivities=True)
    assert control.check_control_plan(spec, base_dir=tmp_path)["ready_to_lock"]
    cells = control.expected_cells(spec)
    assert len(cells) == 9
    assert {x["arm"] for x in cells} == {"recorded_reference", "seven_frontal_average", "annotation_clean"}
    assert all(x["features"] == 21 and x["band_hz"] == [8, 13] for x in cells)
    spec["acquisition"]["artifact_annotations_available"] = False
    assert "matrix.artifact_sensitivity" in {x["path"] for x in control.check_control_plan(spec)["findings"]}


def test_wrong_acquisition_or_decision_evidence_is_not_just_a_matching_file_hash(tmp_path):
    _, spec = ready_plan(tmp_path)
    spec["acquisition"]["reference"] = "Different electrode"
    assert "evidence.acquisition_feasibility" in {x["path"] for x in control.check_control_plan(spec, base_dir=tmp_path)["findings"]}


@pytest.mark.parametrize("cap", [6, 7, 21])
def test_allocation_balanced_at_every_early_stop_and_seeded(tmp_path, cap):
    _, spec = ready_plan(tmp_path)
    spec["recruitment"]["cap_unique_participants"] = cap
    values = [x["sequence_index"] for x in control.allocation_schedule(spec)["slots"]]
    assert len(values) == cap
    assert values == [x["sequence_index"] for x in control.allocation_schedule(spec)["slots"]]
    for n in range(1, cap + 1):
        assert abs(values[:n].count(0) - values[:n].count(1)) <= 1


def test_locked_plan_self_contained_and_immutable(frozen):
    destination, path, _ = frozen
    original = (destination / "control_plan_lock.json").read_bytes()
    with pytest.raises(FileExistsError):
        control.freeze_control_plan(path, destination)
    shutil.rmtree(path.parent)
    verified = control.validate_control_plan_lock(destination)
    assert verified["manifest"]["control_plan_lock_id"]
    assert (destination / "control_plan_lock.json").read_bytes() == original


def test_rehashed_matrix_change_rejected_by_semantic_contract(frozen):
    destination, _, _ = frozen
    matrix = control.read_plan(destination / "matrix.json")
    matrix["cells"].pop()
    write(destination / "matrix.json", matrix)
    manifest = control.read_plan(destination / "control_plan_lock.json")
    manifest["artifacts"]["matrix.json"] = {"sha256": sha256_file(destination / "matrix.json"), "bytes": (destination / "matrix.json").stat().st_size}
    manifest["control_plan_lock_id"] = fingerprint({k: v for k, v in manifest.items() if k != "control_plan_lock_id"})
    write(destination / "control_plan_lock.json", manifest)
    with pytest.raises(control.ControlPlanError, match="matrix or balanced allocation"):
        control.validate_control_plan_lock(destination)


def test_freeze_rejects_dirty_implementation_before_publish(tmp_path, monkeypatch):
    path, _ = ready_plan(tmp_path / "source")
    monkeypatch.setattr(control, "code_identity", lambda: {**synthetic_code(), "dirty": True})
    with pytest.raises(control.ControlPlanError, match="clean committed"):
        control.freeze_control_plan(path, tmp_path / "frozen")
    assert not (tmp_path / "frozen").exists()


def test_freeze_rejects_future_registration(tmp_path, monkeypatch):
    _, spec = ready_plan(tmp_path / "source")
    spec["approval"]["registered_at"] = "2030-01-01T00:00:00Z"
    refresh_evidence(tmp_path / "source", spec)
    monkeypatch.setattr(control, "_now", lambda: LOCKED_AT)
    with pytest.raises(control.ControlPlanError, match="precede"):
        control.freeze_control_plan(tmp_path / "source/plan.json", tmp_path / "frozen")


def test_paired_AB_sessions_do_not_increase_N_and_losses_remain_visible(tmp_path):
    _, spec = ready_plan(tmp_path)
    spec["procedure"]["same_participant_AB"] = "paired_transport_descriptive"
    audit = control.validate_session_history(spec, history(spec, paired=True), locked_at=LOCKED_AT)
    assert audit["n_recruited"] == audit["n_independent_participants"] == 4
    assert audit["n_B_QA_evaluable"] == 3
    assert audit["N_increment_from_repeat_sessions"] == 0
    assert sum(row["n_sessions"] for row in audit["flow"]) == 6
    assert audit["flow"][-1]["disposition"] == "lost"
    assert set(audit["selected_B_sessions"]) == {"001", "1", "NA"}


@pytest.mark.parametrize("attack,match", [
    ("legacy", "Excluded"), ("duplicate_person", "unique literal"), ("prelock", "after plan freeze"),
    ("wrong_sequence", "balanced allocation"), ("repeat", "Additional sessions"),
    ("missing_QA_reason", "exclusion reasons"), ("reverse_order", "A-before-B"),
    ("unplanned_A", "Unplanned protocol"), ("open", "closed control"),
])
def test_session_history_prevents_unplanned_reuse_and_outcome_selection(tmp_path, attack, match):
    _, spec = ready_plan(tmp_path)
    spec["procedure"]["same_participant_AB"] = "paired_transport_descriptive"
    data = history(spec, paired=True)
    first = data["participants"][0]
    if attack == "legacy": first["subject_id"] = "irene"
    elif attack == "duplicate_person": data["participants"][1]["subject_id"] = first["subject_id"]
    elif attack == "prelock": first["recruited_at"] = LOCKED_AT
    elif attack == "wrong_sequence": first["sessions"][1]["sequence_index"] ^= 1
    elif attack == "repeat": first["sessions"].append({**first["sessions"][1], "session_id": "extra-B"})
    elif attack == "missing_QA_reason": first["sessions"][1]["qa_passed"] = False
    elif attack == "reverse_order": first["sessions"][0]["started_at"] = "2020-01-04T10:00:00Z"
    elif attack == "unplanned_A": spec["procedure"]["same_participant_AB"] = "disabled"
    elif attack == "open": data["closed"] = False
    with pytest.raises(control.ControlPlanError, match=match):
        control.validate_session_history(spec, data, locked_at=LOCKED_AT)


def test_complete_control_matrix_reports_negative_and_unevaluable_cells(tmp_path):
    _, spec = ready_plan(tmp_path)
    rows = result_rows(spec)
    rows[-1].update(status="unevaluable", reason="Missing task class after common QC; retained in flow")
    report = control.validate_control_result_matrix(spec, rows, protocol_lock_id="synthetic-lock", participant_ids=["001", "1", "NA"])
    assert report == {"n_planned_cells": 3, "n_reported_cells": 3, "n_participants": 3, "complete": True, "all_signs_required": True, "stop_after_matrix": True, "cross_arm_paired_deltas_authorized": False}


@pytest.mark.parametrize("attack,match", [("missing", "Complete frozen"), ("duplicate", "Complete frozen"), ("cohort", "exact same participant"), ("split", "share split_manifest"), ("QC", "share eligible_epoch"), ("selection", "selection differs")])
def test_result_matrix_cannot_publish_a_selected_subset(tmp_path, attack, match):
    _, spec = ready_plan(tmp_path)
    rows = result_rows(spec)
    if attack == "missing": rows.pop()
    elif attack == "duplicate": rows.append(deepcopy(rows[0]))
    elif attack == "cohort": rows[-1]["participant_ids"] = ["001", "1", "other"]
    elif attack == "split": rows[-1]["split_manifest_sha256"] = "9" * 64
    elif attack == "QC": rows[-1]["eligible_epoch_manifest_sha256"] = "9" * 64
    elif attack == "selection":
        for row in rows: row["selection_sha256"] = "9" * 64
    with pytest.raises(control.ControlPlanError, match=match):
        control.validate_control_result_matrix(spec, rows, protocol_lock_id="synthetic-lock", participant_ids=["001", "1", "NA"])


def test_duplicate_json_keys_are_rejected_before_validation(tmp_path):
    path = tmp_path / "duplicate.json"
    path.write_text('{"status":"ready","status":"draft"}')
    with pytest.raises(control.ControlPlanError, match="Duplicate JSON"):
        control.read_plan(path)


def test_check_CLI_reports_blocked_draft_without_creating_a_lock(tmp_path):
    destination = tmp_path / "draft-check"
    run = subprocess.run([sys.executable, str(ROOT / "scripts/23_phase_controls.py"), "check", "--plan", str(ROOT / "configs/phase_control_plan.json"), "--out-dir", str(destination)], capture_output=True, text=True)
    assert run.returncode == 2, run.stderr
    report = json.loads((destination / "readiness.json").read_text())
    assert report["plan_sha256"] == sha256_file(ROOT / "configs/phase_control_plan.json")
    assert not report["ready_to_lock"]
    assert not (destination / "control_plan_lock.json").exists()


def prospective_manifests(spec):
    from eeg_ma.phase_controls import EPOCH_SCHEMA, SPLIT_SCHEMA
    data = history(spec)
    epochs = {"schema_version": EPOCH_SCHEMA, "epochs": []}
    subjects = [person["subject_id"] for person in data["participants"] if person["disposition"] == "completed"]
    for person in data["participants"]:
        if not person["sessions"]:
            continue
        session = person["sessions"][0]
        rows = []
        for label in (0, 1):
            for number in range(30):
                index = 30 * label + number
                rows.append({"epoch_id": f"{person['subject_id']}:e:{index}", "cohort_id": spec["data"]["cohort_id"],
                             "subject_id": person["subject_id"], "session_id": session["session_id"],
                             "recording_id": session["recording_id"], "recording_sha256": session["recording_sha256"],
                             "start_sample": 2500 * index, "end_sample": 2500 * (index + 1),
                             "comparison": "copy_control_vs_type1", "class_label": label, "qc_passed": True, "qc_reason_codes": []})
        session["eligibility_manifest_sha256"] = control.session_epoch_identity(rows)
        epochs["epochs"].extend(rows)
    splits = {"schema_version": SPLIT_SCHEMA, "splits": []}
    for subject in subjects:
        train = [s for s in subjects if s != subject]
        splits["splits"].append({"split_id": "outer:" + subject, "train_subject_ids": train, "test_subject_ids": [subject],
            "inner_splits": [{"split_id": "inner:" + held, "train_subject_ids": [s for s in train if s != held], "validation_subject_ids": [held]} for held in train]})
    return epochs, splits, data


def test_authorization_binds_actual_B_arm_history_class_mapping_and_full_manifest(frozen):
    destination, _, spec = frozen
    epochs, splits, data = prospective_manifests(spec)
    authorization = control.authorize_phase_inputs(destination, epochs, splits, data)
    policy = authorization["input_policy"]
    assert policy["min_epochs_per_class"] == 20 and policy["max_epochs_per_class"] == 30
    assert authorization["arm_contract"]["acquisition"]["reference"] == "SYNTHETIC reference"
    assert policy["preprocessing_sha256"] == fingerprint(authorization["arm_contract"])
    assert authorization["recruitment_audit"]["stop_reason"] == "administrative_shortfall"
    assert authorization["recruitment_audit"]["n_recruited"] == 4
    assert spec["reporting"]["class_mapping"] == {"0": "CopyControl", "1": "Type1"}


@pytest.mark.parametrize("attack,match", [("leaked_split", "participant leakage"), ("missing_epoch", "eligibility identity"),
    ("wrong_recording", "recording identity"), ("duplicate_epoch", "eligibility identity"), ("wrong_stop", "Stopping evidence"),
    ("short_epoch", "5 seconds"), ("wrong_cohort", "Epoch cohort"), ("unplanned_arm", "absent"), ("unplanned_QC", "unplanned selective")])
def test_prospective_authorization_rejects_incomplete_or_incompatible_inputs(frozen, attack, match):
    destination, _, spec = frozen
    epochs, splits, data = prospective_manifests(spec)
    options = {}
    if attack == "leaked_split": splits["splits"][0]["train_subject_ids"].append(splits["splits"][0]["test_subject_ids"][0])
    elif attack == "missing_epoch": epochs["epochs"].pop()
    elif attack == "wrong_recording": epochs["epochs"][0]["recording_sha256"] = "d" * 64
    elif attack == "duplicate_epoch": epochs["epochs"].append(deepcopy(epochs["epochs"][0]))
    elif attack == "wrong_stop": data["stop_reason"] = "target_reached"
    elif attack == "short_epoch": epochs["epochs"][0]["end_sample"] -= 1
    elif attack == "wrong_cohort": epochs["epochs"][0]["cohort_id"] = "other-cohort"
    elif attack == "unplanned_arm": options["arm"] = "seven_frontal_average"
    elif attack == "unplanned_QC": epochs["epochs"][0].update(qc_passed=False, qc_reason_codes=["POSTHOC_THRESHOLD"])
    with pytest.raises(ValueError, match=match):
        control.authorize_phase_inputs(destination, epochs, splits, data, **options)


def test_result_manifest_cannot_substitute_a_uniformly_wrong_lock_or_drop_the_same_person_everywhere(tmp_path):
    _, spec = ready_plan(tmp_path)
    rows = result_rows(spec)
    with pytest.raises(control.ControlPlanError, match="plan lock"):
        control.validate_control_result_matrix(spec, rows, protocol_lock_id="different-lock", participant_ids=["001", "1", "NA"])
    with pytest.raises(control.ControlPlanError, match="participant roster"):
        control.validate_control_result_matrix(spec, rows, protocol_lock_id="synthetic-lock", participant_ids=["001", "1", "NA", "lost"])


def test_session_history_rejects_future_acquisition_even_with_realistic_metadata(tmp_path):
    _, spec = ready_plan(tmp_path)
    data = history(spec)
    data["participants"][0]["sessions"][0]["started_at"] = "2030-01-01T00:00:00Z"
    with pytest.raises(control.ControlPlanError, match="current audit time"):
        control.validate_session_history(spec, data, locked_at=LOCKED_AT, as_of="2020-01-04T00:00:00Z")


def test_checked_in_readiness_is_reproducible_and_bound_to_current_decisions():
    plan_path = ROOT / "configs/phase_control_plan.json"
    expected = control.check_control_plan(control.read_plan(plan_path), base_dir=plan_path.parent)
    expected.update(plan_sha256=sha256_file(plan_path), validator_sha256=sha256_file(ROOT / "src/eeg_ma/control_plan.py"))
    assert control.read_plan(ROOT / "docs/phase_control_readiness.json") == expected
    assert expected["matrix_status"] == "provisional_pending_readiness"
    assert expected["maximum_planned_cells"] == 9
