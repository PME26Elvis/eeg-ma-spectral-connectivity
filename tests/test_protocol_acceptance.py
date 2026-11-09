"""Independent acceptance tests for prospective study boundaries.

These tests deliberately remove or contradict decisions, and preserve the
ordinary report hash after a mutation so that semantic gates are exercised.
No historical participant performance is used to choose a protocol or N.
"""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from eeg_ma.precision import participant_bootstrap, validate_precision_report
from eeg_ma.protocol import ARTIFACT_ROLES, validate_protocol
from eeg_ma.cohort_policy import assess_enrollment, interpret_endpoints
import eeg_ma.protocol as protocol_gate

from test_cohort_policy import person, roster
from test_protocol_gate import ACCESS_TIME, LOCK_TIME, cohort_for, complete_spec, enrollment_for, synthetic_code


ROOT = Path(__file__).resolve().parents[1]


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _resign_report(report: dict) -> None:
    payload = {key: value for key, value in report.items() if key != "report_sha256"}
    report["report_sha256"] = hashlib.sha256(json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode()).hexdigest()


def _decision_spec() -> dict:
    """Complete decision fields only; fake hashes cannot create a real lock."""
    spec = _read_json(ROOT / "configs/confirmatory_protocol.json")
    spec["acquisition"].update(
        units="uV", reference="synthetic verified reference", ground="synthetic verified ground",
        device="synthetic acquisition device", evidence="synthetic bench verification record",
        verified=True, historical_alignment_verified=True,
    )
    spec["recruitment"].update(target_evaluable=60, recruitment_cap=94)
    spec["approval"].update(
        responsible_person="synthetic verifier", approved_at="2020-01-01T00:00:00+00:00",
        ethics_status="approved", ethics_reference="synthetic approval",
        resource_commitment_confirmed=True,
    )
    spec["freeze"]["artifacts"] = {
        role: {"path": f"synthetic-{role}.bin", "sha256": "a" * 64}
        for role in ARTIFACT_ROLES
    }
    return spec


def _path_parent(spec: dict, path: str):
    parts = path.split(".")
    for key in parts[:-1]:
        spec = spec[key]
    return spec, parts[-1]


@pytest.mark.parametrize("path", [
    "acquisition.device", "acquisition.event_clock", "acquisition.raw_channels",
    "acquisition.analysis_channels", "acquisition.sampling_rate_hz", "acquisition.epoch_seconds",
    "acquisition.historical_alignment_verified", "eligibility.qc", "eligibility.max_epochs_per_class",
    "eligibility.technical_failure_reasons", "eligibility.replacement_policy", "inference.seed",
    "protocols.A.order", "protocols.A.session_selection", "endpoints.primary", "endpoints.key_secondary",
    "endpoints.secondary_policy", "endpoints.multiple_comparisons", "freeze.model_training_scope",
    "recruitment.stop_rule", "recruitment.score_based_extension_allowed", "approval.ethics_status",
    "approval.ethics_reference", "approval.resource_commitment_confirmed",
])
def test_deleting_a_required_scientific_decision_blocks_a_draft(path):
    spec = _decision_spec()
    initial = validate_protocol(spec)
    assert initial.ready, initial.as_dict()
    parent, key = _path_parent(spec, path)
    del parent[key]
    assert not validate_protocol(spec).ready, f"Removing {path} must block readiness"


@pytest.mark.parametrize("path,value", [
    ("acquisition.historical_alignment_verified", False),
    ("approval.resource_commitment_confirmed", False),
    ("protocols.A.enabled", False),
    ("protocols.B.enabled", True),
    ("protocols.B.role", "secondary"),
    ("eligibility.min_epochs_per_class", True),
    ("eligibility.max_attempts_per_task", 45.0),
    ("eligibility.min_epochs_per_class", 31),
    ("eligibility.max_replacement_sessions", -1),
    ("recruitment.recruitment_cap", 59),
    ("recruitment.score_based_extension_allowed", True),
    ("freeze.model_training_scope", "development_and_new_cohort"),
    ("inference.seed", True),
])
def test_inconsistent_decisions_and_unsupported_protocol_activation_block_a_draft(path, value):
    spec = _decision_spec()
    initial = validate_protocol(spec)
    assert initial.ready, initial.as_dict()
    parent, key = _path_parent(spec, path)
    parent[key] = value
    assert not validate_protocol(spec).ready, f"Invalid {path}={value!r} must block readiness"


@pytest.mark.parametrize("mutation", ["omit_scenario", "duplicate_scenario", "wrong_cap", "unplanned_selected_n"])
def test_precision_evidence_cannot_hide_a_scenario_or_change_recruitment(mutation):
    report = _read_json(ROOT / "docs/precision/precision_report.json")
    validate_precision_report(report)
    if mutation == "omit_scenario":
        report["validation"].pop()
    elif mutation == "duplicate_scenario":
        report["validation"].append(copy.deepcopy(report["validation"][0]))
    elif mutation == "wrong_cap":
        report["recruitment_cap"] -= 1
    else:
        report["selected_target_n"] = max(report["config"]["candidate_n"]) + 1
    _resign_report(report)
    with pytest.raises(ValueError):
        validate_precision_report(report)


def test_paired_delta_degeneracy_does_not_hide_ordinary_primary_uncertainty():
    primary = [.35, .55, .75, .85]
    reference = [.25, .45, .65, .75]
    paired = participant_bootstrap(primary, reference, seed=17, n_resamples=1000)
    alone = participant_bootstrap(primary, seed=17, n_resamples=1000)
    assert paired["primary"] == alone["primary"]
    assert paired["primary"]["confirmatory_eligible"]
    assert paired["primary"]["ci_upper"] > paired["primary"]["ci_lower"]
    assert paired["paired_delta"]["degenerate"]
    assert not paired["paired_delta"]["confirmatory_eligible"]
    assert paired["n_participants"] == len(primary)


def test_time_capped_primary_can_remain_evaluable_without_starting_secondary():
    spec = _decision_spec()
    p = person(spec)
    s = p["sessions"][0]
    s["duration_seconds"] = spec["eligibility"]["max_session_seconds"]
    s["attempts"] = {"Type1": 35, "Type2": 0}
    s["correct"] = s["eligible_correct"] = {"Type1": 20, "Type2": 0}
    s["rest_acquired"]["Rest2"] = s["rest_eligible"]["Rest2"] = s["rest_seconds"]["Rest2"] = 0
    s["task_end"] = {"Type1": "session_time_limit", "Type2": "not_started"}
    report = assess_enrollment(roster(spec, [p]), spec)
    assert report["valid"], report["errors"]
    assert report["n_evaluable"] == 1
    assert report["n_secondary_evaluable"] == 0
    s["duration_seconds"] -= 1
    assert not assess_enrollment(roster(spec, [p]), spec)["valid"]


def test_rest2_cannot_be_acquired_after_type1_consumes_the_session_time_budget():
    spec = _decision_spec()
    p = person(spec)
    s = p["sessions"][0]
    s["duration_seconds"] = spec["eligibility"]["max_session_seconds"]
    s["attempts"] = {"Type1": 35, "Type2": 0}
    s["correct"] = s["eligible_correct"] = {"Type1": 20, "Type2": 0}
    s["task_end"] = {"Type1": "session_time_limit", "Type2": "not_started"}
    report = assess_enrollment(roster(spec, [p]), spec)
    assert not report["valid"], "Rest2 cannot follow a Type1 stop at the session deadline"


def test_reaching_target_does_not_drop_already_recruited_pending_participant():
    spec = _decision_spec()
    spec["recruitment"].update(target_evaluable=2, recruitment_cap=4)
    people = [person(spec, f"person-{i}") for i in range(3)]
    people[-1].update(disposition="pending", closed_at=None, sessions=[])
    report = assess_enrollment(roster(spec, people), spec)
    assert report["valid"] and report["stop_recruitment"]
    assert report["n_evaluable"] == 2 and report["n_recruited"] == 3
    assert report["pending_participants"] == 1 and not report["outcome_ready"]
    people[-1].update(disposition="lost_to_followup", closed_at="2027-01-10T00:00:00Z")
    final = assess_enrollment(roster(spec, people), spec)
    assert final["valid"] and final["outcome_ready"]
    assert final["n_evaluable"] == 2 and final["n_recruited"] == 3


def test_literal_participant_ids_survive_the_cumulative_roster():
    spec = _decision_spec()
    spec["recruitment"].update(target_evaluable=4, recruitment_cap=6)
    identities = ["001", "1", "NA", "null"]
    report = assess_enrollment(roster(spec, [person(spec, subject) for subject in identities]), spec)
    assert report["valid"] and report["n_evaluable"] == 4
    assert {p["subject_id"] for p in report["participants"]} == set(identities)


def test_interpretation_cannot_promote_a_numerically_degenerate_bootstrap():
    spec = _decision_spec()
    spec["recruitment"].update(target_evaluable=3, recruitment_cap=4)
    boot = participant_bootstrap([.8, .8 + 4e-13, .8 + 8e-13], seed=17, n_resamples=1000)
    primary = boot["primary"]
    assert primary["degenerate"] and not primary["confirmatory_eligible"]
    conclusion = interpret_endpoints([primary["ci_lower"], primary["ci_upper"]], None,
                                    n_evaluable=3, spec=spec)
    assert conclusion["above_chance"] == "uncertain_degenerate"
    assert not conclusion["precision_target_met"]


@pytest.mark.parametrize("mode,n,scope", [
    ("pilot_estimation", 60, "descriptive_pilot_only"),
    ("confirmatory", 59, "descriptive_target_shortfall"),
])
def test_narrow_positive_interval_cannot_override_pilot_mode_or_target_shortfall(mode, n, scope):
    spec = _decision_spec()
    spec["study"]["mode"] = mode
    conclusion = interpret_endpoints([.75, .85], [.10, .15], n_evaluable=n, spec=spec)
    assert conclusion["above_chance_interval_position"] == "supported"
    assert conclusion["above_chance"] == "descriptive_only"
    assert conclusion["inference_scope"] == scope
    assert conclusion["paired_delta_inference"] == "descriptive_nominal_interval_no_confirmatory_claim"
    if n < spec["recruitment"]["target_evaluable"]:
        assert not conclusion["precision_target_met"]


@pytest.fixture
def acceptance_lock(tmp_path, monkeypatch):
    path, spec = complete_spec(tmp_path / "source")
    monkeypatch.setattr(protocol_gate, "code_identity", synthetic_code)
    monkeypatch.setattr(protocol_gate, "_now", lambda: LOCK_TIME)
    out = tmp_path / "locked"
    protocol_gate.lock_protocol(path, out)
    monkeypatch.setattr(protocol_gate, "_now", lambda: ACCESS_TIME)
    cohort = cohort_for(spec)
    cp, ep = tmp_path / "cohort.json", tmp_path / "enrollment.json"
    _write_json(cp, cohort)
    _write_json(ep, enrollment_for(spec, cohort))
    return out, path, spec, cp, ep


@pytest.mark.parametrize("fault", ["target_fitted", "different_training_people", "different_reference_learner"])
def test_matching_file_hashes_cannot_authorize_inconsistent_training_provenance(tmp_path, monkeypatch, fault):
    path, spec = complete_spec(tmp_path / "source")
    role = "reference_training_manifest" if fault == "different_reference_learner" else "primary_training_manifest"
    training_path = path.parent / spec["freeze"]["artifacts"][role]["path"]
    training = _read_json(training_path)
    if fault == "target_fitted":
        training["target_fit_performed"] = True
    elif fault == "different_training_people":
        training["training_subject_ids"].append("new-confirmatory-person")
    else:
        training["learner_id"] = "bp_coh_sfs_svm"
    _write_json(training_path, training)
    spec["freeze"]["artifacts"][role]["sha256"] = hashlib.sha256(training_path.read_bytes()).hexdigest()
    _write_json(path, spec)
    monkeypatch.setattr(protocol_gate, "code_identity", synthetic_code)
    monkeypatch.setattr(protocol_gate, "_now", lambda: LOCK_TIME)
    out = tmp_path / "rejected"
    with pytest.raises(protocol_gate.ProtocolError, match="training|development"):
        protocol_gate.lock_protocol(path, out)
    assert not out.exists()


@pytest.mark.parametrize("field,value", [
    ("target_half_width", .07), ("completion_assurance", .97), ("minimum_retention_for_planning", .70),
])
def test_precision_evidence_must_match_the_locked_recruitment_assumptions(tmp_path, monkeypatch, field, value):
    path, spec = complete_spec(tmp_path / "source")
    spec["recruitment"][field] = value
    _write_json(path, spec)
    monkeypatch.setattr(protocol_gate, "code_identity", synthetic_code)
    monkeypatch.setattr(protocol_gate, "_now", lambda: LOCK_TIME)
    out = tmp_path / "rejected"
    with pytest.raises(protocol_gate.ProtocolError, match="Precision report"):
        protocol_gate.lock_protocol(path, out)
    assert not out.exists()


def test_outcome_event_must_include_recruited_people_who_are_not_evaluable(acceptance_lock):
    out, _, spec, cp, ep = acceptance_lock
    enrollment = _read_json(ep)
    failed = copy.deepcopy(enrollment["participants"][0])
    failed.update(subject_id="not-evaluable", disposition="lost_to_followup", sessions=[])
    enrollment["participants"].append(failed)
    _write_json(ep, enrollment)
    assert assess_enrollment(enrollment, spec)["outcome_ready"]
    with pytest.raises(protocol_gate.ProtocolError, match="complete cumulative"):
        protocol_gate.record_outcome_access(out, cp, enrollment_manifest_path=ep)
    assert not list((out / "events").iterdir())
    cohort = _read_json(cp)
    cohort["subject_ids"].append("not-evaluable")
    _write_json(cp, cohort)
    event = protocol_gate.record_outcome_access(out, cp, enrollment_manifest_path=ep)
    assert event["n_recruited"] == spec["recruitment"]["target_evaluable"] + 1
    assert event["n_evaluable"] == spec["recruitment"]["target_evaluable"]
    assert "not-evaluable" in protocol_gate.validate_lock(out)["exposed_subject_ids"]


@pytest.mark.parametrize("fault", ["reset_state_pointer", "remove_committed_event"])
def test_recorded_exposure_cannot_disappear_from_a_locally_damaged_ledger(acceptance_lock, fault):
    out, _, _, cp, ep = acceptance_lock
    event = protocol_gate.record_outcome_access(out, cp, enrollment_manifest_path=ep)
    if fault == "reset_state_pointer":
        state_path = out / "STATE.json"
        state = _read_json(state_path)
        state.update(sequence=0, head=None)
        _write_json(state_path, state)
    else:
        (out / "events" / (event["event_id"] + ".json")).unlink()
    with pytest.raises(protocol_gate.ProtocolError):
        protocol_gate.validate_lock(out)
    with pytest.raises(protocol_gate.ProtocolError):
        protocol_gate.check_launch(out, cp)


def test_development_identity_exclusions_survive_two_expansions(acceptance_lock, tmp_path):
    first, _, original, _, _ = acceptance_lock
    parent = first
    forbidden = {"001", "010", "dev-old"}
    for generation, ids in ((2, ["001", "new-development"]), (3, ["latest-development"])):
        path, spec = complete_spec(tmp_path / f"source-{generation}", subject_ids=ids)
        spec["protocol_version"] = f"1.{generation}"
        spec["study"]["cohort_id"] = f"fresh-cohort-{generation}"
        _write_json(path, spec)
        cp = tmp_path / f"holdout-{generation}.json"
        _write_json(cp, cohort_for(spec, [f"unseen-{generation}"]))
        out = tmp_path / f"lock-{generation}"
        protocol_gate.amend_protocol(parent, path, out, reason="SYNTHETIC expanded development",
                                    holdout_manifest_path=cp)
        forbidden.update(ids)
        assert set(protocol_gate.validate_lock(out)["development_subject_ids"]) == forbidden
        for subject in forbidden:
            with pytest.raises(protocol_gate.ProtocolError, match="overlap"):
                protocol_gate.check_launch(out, cohort_for(spec, [subject]))
        parent = out
    assert protocol_gate.validate_lock(first)["protocol"]["study"]["cohort_id"] == original["study"]["cohort_id"]
