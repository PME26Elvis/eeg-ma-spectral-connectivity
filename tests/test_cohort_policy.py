"""Policy edge cases use synthetic metadata, never historical model outcomes."""
from copy import deepcopy
import json
from pathlib import Path

import pytest

from eeg_ma.cohort_policy import EnrollmentError, assess_enrollment, interpret_endpoints


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def cohort_spec():
    spec = json.loads((ROOT / "configs/confirmatory_protocol.json").read_text())
    spec["recruitment"].update(target_evaluable=2, recruitment_cap=3)
    return spec


def session(spec, *, session_id="s1", attempt=1, start="2027-01-01T10:00:00Z"):
    return {"session_id": session_id, "protocol_id": spec["protocols"]["A"]["protocol_id"],
        "attempt_number": attempt, "started_at": start, "duration_seconds": 1000,
        "qa": {"passed": True, "reason_codes": [], "evidence_sha256": "a" * 64},
        "attempts": {"Type1": 30, "Type2": 30}, "correct": {"Type1": 30, "Type2": 30},
        "eligible_correct": {"Type1": 30, "Type2": 30},
        "rest_acquired": {"Rest1": 30, "Rest2": 30}, "rest_eligible": {"Rest1": 30, "Rest2": 30},
        "rest_seconds": {"Rest1": 150, "Rest2": 150},
        "task_end": {"Type1": "quota_reached", "Type2": "quota_reached"}}


def person(spec, subject="new-001"):
    return {"subject_id": subject, "recruited_at": "2027-01-01T09:00:00Z", "population_eligible": True,
        "disposition": "completed", "closed_at": "2027-01-02T11:00:00Z", "sessions": [session(spec)]}


def roster(spec, people=None):
    return {"schema_version": "eeg-enrollment-1", "cohort_id": spec["study"]["cohort_id"],
        "outcomes_accessed": False, "participants": people if people is not None else [person(spec)]}


def test_unique_people_target_and_input_order_do_not_change_results(cohort_spec):
    people = [person(cohort_spec, "new-002"), person(cohort_spec)]
    actual = assess_enrollment(roster(cohort_spec, people), cohort_spec)
    assert actual["valid"] and actual["outcome_ready"] and actual["n_evaluable"] == 2
    assert actual == assess_enrollment(roster(cohort_spec, people[::-1]), cohort_spec)


def test_valid_technical_replacement_selects_first_valid_once(cohort_spec):
    p = person(cohort_spec)
    p["sessions"][0]["qa"].update(passed=False, reason_codes=["DEVICE_DISCONNECT"])
    p["sessions"].append(session(cohort_spec, session_id="s2", attempt=2, start="2027-01-02T10:00:00Z"))
    report = assess_enrollment(roster(cohort_spec, [p]), cohort_spec)
    assert report["valid"] and report["n_evaluable"] == 1
    assert report["participants"][0]["selected_session_id"] == "s2"
    assert report["participants"][0]["technical_failures"] == 1


@pytest.mark.parametrize("failure", ["correctness", "too_late", "third", "nontechnical", "out_of_order"])
def test_invalid_replacement_rejected(cohort_spec, failure):
    p = person(cohort_spec)
    p["closed_at"] = "2027-01-20T00:00:00Z"
    first = p["sessions"][0]
    first["qa"].update(passed=False, reason_codes=["DEVICE_DISCONNECT"])
    second = session(cohort_spec, session_id="s2", attempt=2, start="2027-01-02T10:00:00Z")
    p["sessions"].append(second)
    if failure == "correctness":
        first["qa"].update(passed=True, reason_codes=[])
        first["attempts"]["Type1"] = 45
        first["correct"]["Type1"] = first["eligible_correct"]["Type1"] = 19
        first["task_end"]["Type1"] = "attempt_limit"
    elif failure == "too_late":
        second["started_at"] = "2027-01-09T10:00:00Z"
    elif failure == "third":
        p["sessions"].append(session(cohort_spec, session_id="s3", attempt=3))
    elif failure == "nontechnical":
        first["qa"]["reason_codes"] = ["LOW_MODEL_ACCURACY"]
    else:
        second["started_at"] = "2027-01-01T10:01:00Z"
    assert not assess_enrollment(roster(cohort_spec, [p]), cohort_spec)["valid"]


@pytest.mark.parametrize("count,evaluable", [(19, False), (20, True), (29, True)])
def test_partial_task_only_at_prespecified_attempt_cap(cohort_spec, count, evaluable):
    p = person(cohort_spec)
    s = p["sessions"][0]
    s["attempts"]["Type1"] = 45
    s["correct"]["Type1"] = s["eligible_correct"]["Type1"] = count
    s["task_end"]["Type1"] = "attempt_limit"
    report = assess_enrollment(roster(cohort_spec, [p]), cohort_spec)
    assert report["valid"] and bool(report["n_evaluable"]) == evaluable
    s["attempts"]["Type1"] = 44
    assert not assess_enrollment(roster(cohort_spec, [p]), cohort_spec)["valid"]


def test_primary_retained_when_secondary_correctness_insufficient(cohort_spec):
    p = person(cohort_spec)
    s = p["sessions"][0]
    s["attempts"]["Type2"] = 45
    s["correct"]["Type2"] = s["eligible_correct"]["Type2"] = 18
    s["task_end"]["Type2"] = "attempt_limit"
    report = assess_enrollment(roster(cohort_spec, [p]), cohort_spec)
    assert report["valid"] and report["n_evaluable"] == 1 and report["n_secondary_evaluable"] == 0
    s["qa"].update(passed=False, reason_codes=["SAMPLE_CLOCK"])
    report = assess_enrollment(roster(cohort_spec, [p]), cohort_spec)
    assert report["valid"] and report["n_evaluable"] == 0


def test_minimum_epochs_does_not_shorten_planned_rest(cohort_spec):
    p = person(cohort_spec)
    s = p["sessions"][0]
    s["rest_acquired"]["Rest1"] = s["rest_eligible"]["Rest1"] = 20
    s["rest_seconds"]["Rest1"] = 100
    report = assess_enrollment(roster(cohort_spec, [p]), cohort_spec)
    assert report["valid"] and report["n_evaluable"] == 0


def test_no_selective_correct_or_rest_mask(cohort_spec):
    p = person(cohort_spec)
    p["sessions"][0]["eligible_correct"]["Type1"] = 29
    assert not assess_enrollment(roster(cohort_spec, [p]), cohort_spec)["valid"]
    p = person(cohort_spec)
    p["sessions"][0]["rest_eligible"]["Rest1"] = 29
    assert not assess_enrollment(roster(cohort_spec, [p]), cohort_spec)["valid"]


def test_cap_stops_enrollment_but_pending_subject_blocks_outcomes(cohort_spec):
    people = [person(cohort_spec, f"new-{i}") for i in range(3)]
    for p in people:
        p.update(disposition="pending", closed_at=None, sessions=[])
    report = assess_enrollment(roster(cohort_spec, people), cohort_spec)
    assert report["valid"] and report["stop_recruitment"] and not report["outcome_ready"]
    for p in people:
        p.update(disposition="lost_to_followup", closed_at="2027-01-10T00:00:00Z")
    report = assess_enrollment(roster(cohort_spec, people), cohort_spec)
    assert report["outcome_ready"] and report["n_evaluable"] == 0


def test_already_recruited_overrun_included_but_late_enrollment_rejected(cohort_spec):
    people = [person(cohort_spec, f"new-{i}") for i in range(3)]
    report = assess_enrollment(roster(cohort_spec, people), cohort_spec)
    assert report["valid"] and report["n_evaluable"] == 3
    people[2]["recruited_at"] = "2027-01-03T09:00:00Z"
    people[2]["sessions"][0]["started_at"] = "2027-01-03T10:00:00Z"
    people[2]["closed_at"] = "2027-01-03T11:00:00Z"
    report = assess_enrollment(roster(cohort_spec, people), cohort_spec)
    assert not report["valid"] and "stopping time" in report["errors"][0]


@pytest.mark.parametrize("change", ["duplicate", "overcap", "score", "timezone", "duration", "boolean", "B"])
def test_malformed_or_adaptive_rosters_fail_closed(cohort_spec, change):
    m = roster(cohort_spec)
    p, s = m["participants"][0], m["participants"][0]["sessions"][0]
    if change == "duplicate":
        m["participants"].append(deepcopy(p))
    elif change == "overcap":
        m["participants"] = [person(cohort_spec, f"new-{i}") for i in range(4)]
    elif change == "score":
        s["balanced_accuracy"] = .99
    elif change == "timezone":
        p["recruited_at"] = "2027-01-01T09:00:00"
    elif change == "duration":
        s["duration_seconds"] = 1801
    elif change == "boolean":
        s["attempts"]["Type1"] = True
    else:
        s["protocol_id"] = cohort_spec["protocols"]["B"]["protocol_id"]
    assert not assess_enrollment(m, cohort_spec)["valid"]


def test_withdrawn_subject_retained_in_flow_not_denominator(cohort_spec):
    p = person(cohort_spec)
    p["disposition"] = "withdrawn"
    report = assess_enrollment(roster(cohort_spec, [p]), cohort_spec)
    assert report["valid"] and report["n_recruited"] == 1 and report["n_evaluable"] == 0


@pytest.mark.parametrize("locked_at,as_of", [
    ("2027-01-02T00:00:00Z", "2027-01-10T00:00:00Z"),
    ("2026-12-01T00:00:00Z", "2026-12-30T00:00:00Z"),
    ("2026-12-01T00:00:00Z", "2027-01-02T10:00:00Z"),
])
def test_lock_and_audit_chronology_do_not_report_ready_for_invalid_dates(cohort_spec, locked_at, as_of):
    report = assess_enrollment(roster(cohort_spec), cohort_spec, locked_at=locked_at, as_of=as_of)
    assert not report["valid"] and not report["outcome_ready"]


def test_interpretation_keeps_significance_value_precision_separate(cohort_spec):
    result = interpret_endpoints([.55, .63], [-.03, .08], n_evaluable=2, spec=cohort_spec)
    assert result["above_chance"] == "supported"
    assert result["absolute_research_target"] == "not_met"
    assert result["paired_delta_research_target"] == "uncertain"
    assert result["precision_target_met"]
    boundary = interpret_endpoints([.65, .75], [.05, .15], n_evaluable=2, spec=cohort_spec)
    assert boundary["absolute_research_target"] == boundary["paired_delta_research_target"] == "uncertain"


def test_degenerate_ci_or_missing_class_not_treated_as_success(cohort_spec):
    result = interpret_endpoints([.8, .8], None, n_evaluable=2, spec=cohort_spec)
    assert result["above_chance"] == "uncertain_degenerate" and not result["precision_target_met"]
    assert result["paired_delta_research_target"] == "not_estimable"
    with pytest.raises(EnrollmentError):
        interpret_endpoints([float("nan"), .9], [.1, .2], n_evaluable=2, spec=cohort_spec)
