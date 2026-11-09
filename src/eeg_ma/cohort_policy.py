"""Score-blind, participant-level enrollment policy and endpoint interpretation.

Enrollment records are operator attestations about recruitment and shared raw QA.
This module audits their internal consistency; it does not authenticate people,
replace ``qa.preflight_sessions``, or fit, predict, or score either model.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
import math
import re


class EnrollmentError(ValueError):
    """An enrollment record cannot establish the prespecified policy."""


def _text(value, label):
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise EnrollmentError(f"{label}: expected a nonempty literal string")
    return value


def _integer(value, label, lower=0, upper=None):
    if type(value) is not int or value < lower or (upper is not None and value > upper):
        raise EnrollmentError(f"{label}: integer outside the prespecified range")
    return value


def _number(value, label, lower=0, upper=None):
    if (type(value) not in {int, float} or not math.isfinite(value) or value < lower
            or (upper is not None and value > upper)):
        raise EnrollmentError(f"{label}: finite number outside the prespecified range")
    return value


def _utc(value, label):
    _text(value, label)
    try:
        result = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise EnrollmentError(f"{label}: invalid timestamp") from exc
    if result.tzinfo is None or result.utcoffset() != timedelta(0):
        raise EnrollmentError(f"{label}: an explicit UTC timestamp is required")
    return result.astimezone(timezone.utc)


def _keys(value, keys, label):
    if not isinstance(value, dict) or set(value) != set(keys):
        raise EnrollmentError(f"{label}: expected exactly {', '.join(sorted(keys))}; scores and extra fields are prohibited")


def _session(session, spec, recruited_at, disposition):
    _keys(session, {"session_id", "protocol_id", "attempt_number", "started_at", "duration_seconds",
                    "qa", "attempts", "correct", "eligible_correct", "rest_acquired", "rest_eligible",
                    "rest_seconds", "task_end"}, "session")
    _text(session["session_id"], "session_id")
    if session["protocol_id"] != spec["protocols"]["A"]["protocol_id"]:
        raise EnrollmentError("Only the locked primary A protocol belongs in this enrollment ledger")
    rules = spec["eligibility"]
    attempt = _integer(session["attempt_number"], "attempt_number", 1, 1 + rules["max_replacement_sessions"])
    start = _utc(session["started_at"], "started_at")
    if start < recruited_at:
        raise EnrollmentError("Session precedes recruitment")
    duration = _number(session["duration_seconds"], "duration_seconds", 0, rules["max_session_seconds"])
    if duration == 0:
        raise EnrollmentError("A recorded session must have positive duration")
    end = start + timedelta(seconds=duration)
    qa = session["qa"]
    _keys(qa, {"passed", "reason_codes", "evidence_sha256"}, "qa")
    if type(qa["passed"]) is not bool or not isinstance(qa["reason_codes"], list):
        raise EnrollmentError("QA pass must be boolean and reason_codes a list")
    codes = qa["reason_codes"]
    if any(not isinstance(code, str) or not code or code != code.strip() for code in codes) or len(set(codes)) != len(codes):
        raise EnrollmentError("QA reason codes must be unique literal strings")
    if qa["passed"] == bool(codes):
        raise EnrollmentError("QA pass requires no error codes; failure requires explicit error codes")
    if not isinstance(qa["evidence_sha256"], str) or not re.fullmatch("[0-9a-f]{64}", qa["evidence_sha256"]):
        raise EnrollmentError("QA evidence requires the exact shared preflight report SHA-256")
    for name in ("attempts", "correct", "eligible_correct", "task_end"):
        _keys(session[name], {"Type1", "Type2"}, name)
    for name in ("rest_acquired", "rest_eligible", "rest_seconds"):
        _keys(session[name], {"Rest1", "Rest2"}, name)
    for rest in ("Rest1", "Rest2"):
        acquired = _integer(session["rest_acquired"][rest], rest + " acquired", 0, rules["planned_rest_epochs_per_block"])
        eligible = _integer(session["rest_eligible"][rest], rest + " eligible", 0, acquired)
        seconds = _number(session["rest_seconds"][rest], rest + " seconds", 0, rules["planned_rest_seconds_per_block"])
        if acquired * spec["acquisition"]["epoch_seconds"] > seconds + 1e-9:
            raise EnrollmentError("Rest epoch count exceeds recorded rest duration")
        if qa["passed"] and eligible != acquired:
            raise EnrollmentError("Structural-only QA permits no unplanned rest-epoch rejection mask")
    for task in ("Type1", "Type2"):
        attempts = _integer(session["attempts"][task], task + " attempts", 0, rules["max_attempts_per_task"])
        formal_correct = _integer(session["correct"][task], task + " correct", 0, rules["max_epochs_per_class"])
        correct = _integer(session["eligible_correct"][task], task + " eligible_correct", 0, formal_correct)
        if formal_correct > attempts:
            raise EnrollmentError("Eligible correct trials cannot exceed attempts")
        if qa["passed"] and correct != formal_correct:
            raise EnrollmentError("Every correct formal trial must be eligible after whole-recording QA; selective masks are prohibited")
        reason = session["task_end"][task]
        valid_end = {
            "quota_reached": correct == rules["max_epochs_per_class"],
            "attempt_limit": attempts == rules["max_attempts_per_task"] and correct < rules["max_epochs_per_class"],
            "session_time_limit": duration == rules["max_session_seconds"] and correct < rules["max_epochs_per_class"],
            "not_started": attempts == correct == 0,
            "technical_failure": not qa["passed"],
            "withdrawal": disposition == "withdrawn",
        }
        if not isinstance(reason, str) or reason not in valid_end or not valid_end[reason]:
            raise EnrollmentError(f"{task}: stop reason is inconsistent with the prespecified quota, budget or failure")
    # Rest2/Type2 cannot precede completion or a permitted stop of Type1.
    if ((session["attempts"]["Type2"] or session["rest_acquired"]["Rest2"] or session["rest_seconds"]["Rest2"])
            and session["task_end"]["Type1"] not in {"quota_reached", "attempt_limit"}):
        raise EnrollmentError("Rest2/Type2 cannot start after an unstarted, timed-out, withdrawn or failed Type1")
    if (qa["passed"] and disposition != "withdrawn" and
            any(reason == "not_started" for reason in session["task_end"].values()) and
            duration != rules["max_session_seconds"]):
        raise EnrollmentError("An unstarted task requires the fixed session time limit, technical failure or withdrawal")
    if sum(session["rest_seconds"].values()) + sum(session["attempts"].values()) * spec["protocols"]["A"]["calculation_seconds"] > duration + 1e-9:
        raise EnrollmentError("Recorded blocks and calculation periods exceed session duration")
    results = {}
    for endpoint, rest, task in (("primary", "Rest1", "Type1"), ("secondary", "Rest2", "Type2")):
        full_rest = (session["rest_acquired"][rest] == rules["planned_rest_epochs_per_block"]
                     and session["rest_eligible"][rest] == rules["planned_rest_epochs_per_block"]
                     and session["rest_seconds"][rest] == rules["planned_rest_seconds_per_block"])
        results[endpoint] = (qa["passed"] and full_rest
                             and session["eligible_correct"][task] >= rules["min_epochs_per_class"])
    return {"attempt_number": attempt, "started_at": start, "ended_at": end, "qa_passed": qa["passed"],
            "reason_codes": codes, "session_id": session["session_id"],
            "withdrawal_recorded": "withdrawal" in session["task_end"].values(), **results}


def assess_enrollment(manifest: dict, spec: dict, *, locked_at=None, as_of=None) -> dict:
    """Audit a *complete cumulative roster*, without inspecting model outcomes.

    Returns ``valid=False`` with errors for malformed or inconsistent records.
    Reaching the cap stops recruitment; outcome access also requires every
    already-recruited participant to have a final, score-blind disposition.
    """
    result = {"schema_version": "eeg-enrollment-audit-1", "valid": False, "errors": [],
              "n_recruited": 0, "n_evaluable": 0, "n_secondary_evaluable": 0,
              "pending_participants": 0, "stop_recruitment": False, "outcome_ready": False,
              "stop_reason": "invalid", "participants": [], "evidence_scope": "operator_attested_metadata_shared_raw_QA_still_required"}
    try:
        _keys(manifest, {"schema_version", "cohort_id", "outcomes_accessed", "participants"}, "enrollment")
        if manifest["schema_version"] != "eeg-enrollment-1":
            raise EnrollmentError("Unknown enrollment schema")
        if manifest["cohort_id"] != spec["study"]["cohort_id"] or manifest["outcomes_accessed"] is not False:
            raise EnrollmentError("Enrollment must match the cohort and precede model outcome access")
        target = _integer(spec["recruitment"]["target_evaluable"], "target_evaluable", 2)
        cap = _integer(spec["recruitment"]["recruitment_cap"], "recruitment_cap", target)
        lock_time = _utc(locked_at, "locked_at") if locked_at is not None else None
        audit_time = _utc(as_of, "as_of") if as_of is not None else None
        if not isinstance(manifest["participants"], list):
            raise EnrollmentError("participants must be the complete cumulative roster")
        ids, recruitments, completions = set(), [], []
        for participant in manifest["participants"]:
            _keys(participant, {"subject_id", "recruited_at", "population_eligible", "disposition", "closed_at", "sessions"}, "participant")
            subject = _text(participant["subject_id"], "subject_id")
            if subject in ids:
                raise EnrollmentError("Duplicate subject: repeated sessions never increase participant N")
            ids.add(subject)
            recruited_at = _utc(participant["recruited_at"], "recruited_at")
            if lock_time is not None and recruited_at < lock_time:
                raise EnrollmentError("Recruitment precedes the protocol lock")
            if audit_time is not None and recruited_at > audit_time:
                raise EnrollmentError("Recruitment is in the future relative to this audit")
            recruitments.append(recruited_at)
            if participant["population_eligible"] is not True:
                raise EnrollmentError("Enrollment requires attested population eligibility and consent")
            disposition = participant["disposition"]
            if disposition not in {"pending", "completed", "withdrawn", "lost_to_followup"}:
                raise EnrollmentError("Unknown participant disposition")
            if disposition == "pending":
                if participant["closed_at"] is not None:
                    raise EnrollmentError("A pending participant cannot have closed_at")
                closed_at = None
                result["pending_participants"] += 1
            else:
                closed_at = _utc(participant["closed_at"], "closed_at")
                if closed_at < recruited_at:
                    raise EnrollmentError("Disposition precedes recruitment")
                if audit_time is not None and closed_at > audit_time:
                    raise EnrollmentError("Disposition is in the future relative to this audit")
            if not isinstance(participant["sessions"], list):
                raise EnrollmentError("sessions must be a list")
            sessions = [_session(s, spec, recruited_at, disposition) for s in participant["sessions"]]
            sessions.sort(key=lambda session: session["attempt_number"])
            if [s["attempt_number"] for s in sessions] != list(range(1, len(sessions) + 1)):
                raise EnrollmentError("Session attempt numbers must be contiguous, unique and start at one")
            if len({s["session_id"] for s in sessions}) != len(sessions):
                raise EnrollmentError("Duplicate session ID within a participant")
            if closed_at and any(s["ended_at"] > closed_at for s in sessions):
                raise EnrollmentError("A session ends after the recorded final disposition")
            if audit_time is not None and any(s["ended_at"] > audit_time for s in sessions):
                raise EnrollmentError("A recorded session ends after this audit")
            for previous, replacement in zip(sessions, sessions[1:]):
                if previous["withdrawal_recorded"] or previous["qa_passed"] or not set(previous["reason_codes"]).issubset(spec["eligibility"]["technical_failure_reasons"]):
                    raise EnrollmentError("Replacement is allowed only after an enumerated technical QA failure")
                if not previous["ended_at"] <= replacement["started_at"] <= previous["started_at"] + timedelta(days=spec["eligibility"]["replacement_window_days"]):
                    raise EnrollmentError("Replacement is out of order or outside the fixed replacement window")
            selected = next((s for s in sessions if s["qa_passed"]), None)
            if disposition == "completed" and not sessions:
                raise EnrollmentError("A completed participant needs a recorded session")
            primary = bool(selected and selected["primary"] and disposition == "completed")
            secondary = bool(selected and selected["secondary"] and disposition == "completed")
            if primary:
                # QA and disposition have been finalized at this timestamp.
                completions.append(closed_at)
            result["n_evaluable"] += int(primary)
            result["n_secondary_evaluable"] += int(secondary)
            result["participants"].append({"subject_id": subject, "disposition": disposition,
                "selected_session_id": selected["session_id"] if selected else None,
                "primary_evaluable": primary, "secondary_evaluable": secondary,
                "technical_failures": sum(not s["qa_passed"] for s in sessions)})
        result["n_recruited"] = len(ids)
        if len(ids) > cap:
            raise EnrollmentError("Cumulative unique-participant recruitment exceeds the frozen cap")
        completions.sort()
        if len(completions) >= target and any(t >= completions[target - 1] for t in recruitments):
            raise EnrollmentError("A participant was newly recruited at or after the evaluable-N stopping time")
        result["valid"] = True
        result["stop_recruitment"] = len(completions) >= target or len(ids) == cap
        result["stop_reason"] = "target_evaluable" if len(completions) >= target else "recruitment_cap" if len(ids) == cap else "continue"
        result["outcome_ready"] = result["stop_recruitment"] and result["pending_participants"] == 0
        result["participants"].sort(key=lambda participant: participant["subject_id"])
    except (EnrollmentError, KeyError, TypeError) as exc:
        result["errors"].append(str(exc))
    return result


def interpret_endpoints(primary_ci, delta_ci, *, n_evaluable: int, spec: dict) -> dict:
    """Keep statistical, research-target and precision conclusions separate.

    Inputs are already computed participant bootstrap intervals. This function
    does not compute predictions/scores and does not implement the #11 pipeline.
    """
    def interval(values, label, minimum):
        if not isinstance(values, (list, tuple)) or len(values) != 2:
            raise EnrollmentError(f"{label} must contain exactly two interval bounds")
        low, high = (_number(v, label, minimum, 1) for v in values)
        if low > high:
            raise EnrollmentError(f"{label} bounds are reversed")
        return low, high
    primary = interval(primary_ci, "primary_ci", 0)
    delta = interval(delta_ci, "delta_ci", -1) if delta_ci is not None else None
    _integer(n_evaluable, "n_evaluable", 0)
    def compare(bounds, threshold):
        if bounds is None:
            return "not_estimable"
        if bounds[1] - bounds[0] <= 1e-12 or n_evaluable < 2:
            return "uncertain_degenerate"
        return "supported" if bounds[0] > threshold else "not_met" if bounds[1] < threshold else "uncertain"
    target = spec["recruitment"]["target_evaluable"]
    half_width = (primary[1] - primary[0]) / 2
    scope = ("descriptive_pilot_only" if spec["study"]["mode"] == "pilot_estimation" else
             "descriptive_target_shortfall" if n_evaluable < target else "confirmatory_primary")
    return {
        "above_chance": compare(primary, spec["endpoints"]["chance_level"]) if scope == "confirmatory_primary" else "descriptive_only",
        "above_chance_interval_position": compare(primary, spec["endpoints"]["chance_level"]),
        "inference_scope": scope,
        "absolute_research_target": compare(primary, spec["endpoints"]["absolute_threshold"]),
        "paired_delta_research_target": compare(delta, spec["endpoints"]["delta_min"]),
        "paired_delta_inference": "descriptive_nominal_interval_no_confirmatory_claim",
        "primary_half_width": half_width,
        "precision_target_met": (n_evaluable >= target and primary[1] - primary[0] > 1e-12
                                 and half_width <= spec["recruitment"]["target_half_width"]),
        "evaluable_target_met": n_evaluable >= target,
        "scope": "QA_evaluable_correct_only_protocol_A_not_causal_MA_or_online_BCI_usability",
    }
