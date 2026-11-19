"""Bounded prospective B study plans, kept separate from the A primary lock.

This is a planning and input-authorization contract, not a new acquisition
program, a claim of registration, or an implementation of B model fitting.
Human evidence remains an attestation; hashes do not authenticate its truth.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re

import numpy as np

from .artifacts import new_output_directory
from .environment import validate_environment_evidence
from .provenance import code_identity, environment_identity, fingerprint, sha256_file

PLAN_SCHEMA = "eeg-phase-control-plan-1"
LOCK_SCHEMA = "eeg-phase-control-plan-lock-1"
CHANNELS = ["FP1", "FP2", "F7", "F3", "Fz", "F4", "F8"]
LEGACY_SUBJECTS = {"irene", "jh", "lin", "pftest", "yen"}
ROLES = ("acquisition_feasibility", "data_use_approval", "participant_registry")
SHA = re.compile(r"[0-9a-f]{64}\Z")

SECTIONS = {
    "data": "availability cohort_id collection_started_at outcomes_accessed excluded_subject_ids prior_exposed_subject_ids use original_five_stop_rule",
    "acquisition": "sampling_rate_hz epoch_seconds channels device units reference ground eog_channels artifact_annotations_available effective_rank verified",
    "procedure": "protocol_id comparison blocks epochs_per_block calculation_seconds response_seconds washout_seconds sequences allocation allocation_seed display response control_task task scored_window session_selection maximum_B_sessions transport same_participant_AB AB_order_if_enabled independent_N_increment_for_repeat_sessions",
    "preprocessing": "continuous_filter_hz filter_order phase_band_hz phase_filter_order phase_method target_fit_ICA_ASR_alignment channel_interpolation structural_QA task_eligibility min_epochs_per_class max_epochs_per_class undefined_ciplv_policy ciplv_min_denominator",
    "matrix": "representations pair_count classifiers reference_sensitivity artifact_sensitivity maximum_cells crossed_reference_artifact_arm paired_comparison_policy shared_within_arm",
    "selection": "outer inner selector inner_metric candidate_features max_features min_improvement tie_break stopping scaler classifier lda_tol lda_priors shared_splits target_fit",
    "recruitment": "target_unique_participants cap_unique_participants resource_and_sample_size_rationale stop_rule target_definition early_termination_policy",
    "reporting": "unit weighting metrics class_mapping uncertainty complete_matrix_required report_all_signs missing_class AB_pairing interpretation stop_after_matrix new_hypothesis deferred",
    "approval": "registered_at registration_reference approved_at data_use_terms resources_confirmed",
    "evidence": " ".join(ROLES),
}

# These are the bounded design, not tunable command-line search parameters.
FIXED = {
    "schema_version": PLAN_SCHEMA,
    "purpose": "prospective_control_sensitivity_not_neural_source_identification",
    "analysis_id": "prospective_control_grouped_lopo",
    "data.use": "new_control_cohort_only",
    "data.original_five_stop_rule": "no_new_features_models_or_threshold_search",
    "data.outcomes_accessed": False,
    "data.collection_started_at": None,
    "acquisition.sampling_rate_hz": 500,
    "acquisition.epoch_seconds": 5.0,
    "acquisition.channels": CHANNELS,
    "procedure.protocol_id": "eeg-ma-B-control-v1",
    "procedure.comparison": "copy_control_vs_type1",
    "procedure.blocks": 6,
    "procedure.epochs_per_block": 10,
    "procedure.calculation_seconds": 5.0,
    "procedure.washout_seconds": 30,
    "procedure.sequences": [["CopyControl", "Type1"] * 3, ["Type1", "CopyControl"] * 3],
    "procedure.allocation": "balanced_random_slots_before_acquisition",
    "procedure.allocation_seed": 20261119,
    "procedure.display": "identical_layout_luminance_and_5s_duration",
    "procedure.response": "same_numeric_interface_and_response_window_after_each_epoch",
    "procedure.control_task": "copy_displayed_number_without_arithmetic",
    "procedure.task": "three_digit_subtraction_two_borrows",
    "procedure.scored_window": "calculation_only_excludes_response",
    "procedure.session_selection": "first_QA_valid_B_session_no_accuracy_selection",
    "procedure.maximum_B_sessions": 1,
    "procedure.transport": "B_discrimination_separate_from_A_fixed_model_primary",
    "procedure.AB_order_if_enabled": "A_before_B",
    "procedure.independent_N_increment_for_repeat_sessions": 0,
    "preprocessing.continuous_filter_hz": [2.0, 50.0],
    "preprocessing.filter_order": 4,
    "preprocessing.phase_band_hz": [8.0, 13.0],
    "preprocessing.phase_filter_order": 4,
    "preprocessing.phase_method": "continuous_bandpass_then_hilbert_before_epoching",
    "preprocessing.target_fit_ICA_ASR_alignment": "disabled",
    "preprocessing.channel_interpolation": "disabled",
    "preprocessing.structural_QA": "whole_recording_reject",
    "preprocessing.task_eligibility": "correct_trials_only_both_conditions",
    "preprocessing.min_epochs_per_class": 20,
    "preprocessing.max_epochs_per_class": 30,
    "preprocessing.undefined_ciplv_policy": "exclude_epoch_from_all_representations",
    "preprocessing.ciplv_min_denominator": 1e-6,
    "matrix.representations": ["plv", "iplv", "ciplv"],
    "matrix.pair_count": 21,
    "matrix.classifiers": ["LDA"],
    "matrix.maximum_cells": 9,
    "matrix.crossed_reference_artifact_arm": "disabled",
    "matrix.paired_comparison_policy": "within_arm_exact_epochs_cross_arm_delta_requires_separate_exact_pair_validation",
    "matrix.shared_within_arm": ["samples", "channels", "pairs", "band", "QC", "outer_splits", "inner_splits", "selection_rule", "classifier"],
    "selection.outer": "leave_one_participant_out_all_sessions_together",
    "selection.inner": "leave_one_training_participant_out",
    "selection.selector": "LDA_forward_selection_training_only",
    "selection.inner_metric": "participant_macro_balanced_accuracy",
    "selection.candidate_features": 21,
    "selection.max_features": 21,
    "selection.min_improvement": 0.0,
    "selection.tie_break": "ordered_channel_pairs_first",
    "selection.stopping": "retain_first_nonempty_best_subset_then_require_strict_improvement",
    "selection.scaler": "training_only_StandardScaler",
    "selection.classifier": "LDA_svd_no_hyperparameter_search",
    "selection.lda_tol": 1e-4,
    "selection.lda_priors": "training_class_frequencies",
    "selection.shared_splits": True,
    "selection.target_fit": False,
    "recruitment.stop_rule": "first_of_prespecified_target_or_cap_without_score_based_extension",
    "recruitment.target_definition": "QA_valid_B_participants_with_minimum_correct_epochs_before_common_phase_QC",
    "recruitment.early_termination_policy": "retain_all_and_report_administrative_shortfall_without_score_based_extension",
    "reporting.unit": "participant",
    "reporting.weighting": "equal_participant_weights",
    "reporting.metrics": ["balanced_accuracy", "accuracy", "recall_control", "recall_task", "confusion_counts"],
    "reporting.class_mapping": {"0": "CopyControl", "1": "Type1"},
    "reporting.uncertainty": "descriptive_SD_only_no_IID_CV_confidence_interval",
    "reporting.complete_matrix_required": True,
    "reporting.report_all_signs": True,
    "reporting.missing_class": "unevaluable_retain_recruitment_and_QC_denominators",
    "reporting.AB_pairing": "exact_same_people_no_inner_join_no_double_N",
    "reporting.interpretation": "sensitivity_does_not_identify_artifact_or_quantify_zero_lag_source",
    "reporting.stop_after_matrix": True,
    "reporting.new_hypothesis": "new_study_and_new_unexposed_holdout",
    "reporting.deferred": ["source_space", "high_density", "wPLI", "new_feature_families", "calibration_EA", "streaming"],
}


class ControlPlanError(ValueError):
    pass


def _json(data: bytes):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ControlPlanError(f"Duplicate JSON key: {key}")
            result[key] = value
        return result
    try:
        value = json.loads(data.decode("utf-8-sig"), object_pairs_hook=pairs,
                           parse_constant=lambda x: (_ for _ in ()).throw(ControlPlanError(f"Nonfinite JSON: {x}")))
    except (ValueError, UnicodeError) as exc:
        raise ControlPlanError(f"Invalid control evidence: {exc}") from exc
    if not isinstance(value, dict):
        raise ControlPlanError("Control evidence must be a JSON object")
    return value


def read_plan(path):
    return _json(Path(path).read_bytes())


def _write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _text(value):
    return isinstance(value, str) and value == value.strip() and bool(value) and value.lower() not in {"unknown", "pending", "tbd", "todo"}


def _ids(value, label, *, allow_empty=False):
    if not isinstance(value, list) or (not value and not allow_empty) or any(not _text(x) for x in value) or len(set(value)) != len(value):
        raise ControlPlanError(f"{label}: need unique literal IDs, without blanks or coercion")
    return value


def _at(spec, path):
    for key in path.split("."):
        if not isinstance(spec, dict):
            return None
        spec = spec.get(key)
    return spec


def _utc(value, label):
    try:
        result = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if result.tzinfo is None or result.utcoffset() is None:
            raise ValueError("missing offset")
        return result.astimezone(timezone.utc)
    except (ValueError, AttributeError) as exc:
        raise ControlPlanError(f"{label}: explicit timezone-aware timestamp required") from exc


def _now():
    return datetime.now(timezone.utc).isoformat()


def expected_cells(spec):
    """Enumerate the entire allowed matrix; sensitivities never form a factorial."""
    arms = ["recorded_reference"]
    for key, arm in (("reference_sensitivity", "seven_frontal_average"), ("artifact_sensitivity", "annotation_clean")):
        if _at(spec, f"matrix.{key}.decision") == "enabled":
            arms.append(arm)
    return [{"cell_id": f"B:{arm}:{metric}:LDA", "arm": arm, "representation": metric,
             "classifier": "LDA", "comparison": "copy_control_vs_type1", "features": 21,
             "band_hz": [8.0, 13.0]} for arm in arms for metric in ("plv", "iplv", "ciplv")]


def allocation_schedule(spec):
    cap = _at(spec, "recruitment.cap_unique_participants")
    if type(cap) is not int or not 4 <= cap <= 10_000:
        raise ControlPlanError("Recruitment cap must be an integer in [4, 10000] before allocating slots")
    # Permuted blocks of two retain balance at every completed block, not merely
    # at the cap. An early target stop can differ by at most one participant.
    rng = np.random.default_rng(spec["procedure"]["allocation_seed"])
    assignments = [int(v) for _ in range((cap + 1) // 2) for v in rng.permutation(2)][:cap]
    return {"schema_version": "eeg-control-allocation-1", "seed": spec["procedure"]["allocation_seed"],
            "slots": [{"slot": i + 1, "sequence_index": value} for i, value in enumerate(assignments)]}


def _evidence(spec, base, supplied=None):
    output = {}
    for role in ROLES:
        item = spec["evidence"][role]
        if not isinstance(item, dict) or set(item) != {"path", "sha256"} or not _text(item["path"]) or not isinstance(item["sha256"], str) or not SHA.fullmatch(item["sha256"]):
            raise ControlPlanError(f"evidence.{role}: actual file path and SHA-256 required")
        if supplied is not None:
            data = supplied[role]
        else:
            path = Path(item["path"])
            path = path if path.is_absolute() else Path(base) / path
            if any(part.is_symlink() for part in [path, *path.parents]):
                raise ControlPlanError(f"evidence.{role}: symlink is not evidence")
            data = path.read_bytes()
        if hashlib.sha256(data).hexdigest() != item["sha256"]:
            raise ControlPlanError(f"evidence.{role}: SHA-256 mismatch")
        output[role] = data
    return output


def check_control_plan(spec, *, base_dir=None, evidence_bytes=None):
    """Return all unmet decisions. The repository draft deliberately stays blocked."""
    findings = []
    def fail(path, message, code="CONTROL_DECISION"):
        findings.append({"code": code, "path": path, "message": message})
    top = {"schema_version", "study_id", "version", "status", "purpose", "analysis_id", *SECTIONS}
    if not isinstance(spec, dict) or set(spec) != top:
        return {"ready_to_lock": False, "status": "blocked", "findings": [{"code": "CONTROL_SCHEMA", "path": "$", "message": "Exact control-plan fields required"}], "cells": []}
    for name, fields in SECTIONS.items():
        if not isinstance(spec[name], dict) or set(spec[name]) != set(fields.split()):
            fail(name, "Exact section fields required", "CONTROL_SCHEMA")
    for name, fields in (("reference_sensitivity", {"decision", "method", "reason"}), ("artifact_sensitivity", {"decision", "method", "annotation_ontology", "reason"})):
        value = _at(spec, f"matrix.{name}")
        if not isinstance(value, dict) or set(value) != fields:
            fail(f"matrix.{name}", "Exact sensitivity fields required", "CONTROL_SCHEMA")
    if findings:
        return {"ready_to_lock": False, "status": "blocked", "findings": findings, "cells": []}
    for path, wanted in FIXED.items():
        actual = _at(spec, path)
        if actual != wanted or (type(wanted) is bool and type(actual) is not bool) or (isinstance(actual, bool) and type(wanted) is not bool):
            fail(path, f"Frozen bounded design requires {wanted!r}; a changed hypothesis needs a new study")
    for path in ("study_id", "version", "data.cohort_id", "acquisition.device", "acquisition.units", "acquisition.reference", "acquisition.ground",
                 "recruitment.resource_and_sample_size_rationale", "approval.registration_reference", "approval.data_use_terms"):
        if not _text(_at(spec, path)):
            fail(path, "Explicit verified value required; do not infer unavailable facts")
    if spec["status"] != "ready":
        fail("status", "Draft must be reviewed and marked ready before plan freeze")
    if spec["data"]["availability"] != "prospective_arranged_no_acquisition":
        fail("data.availability", "Arrange new data access before acquisition; existing discovery data cannot activate B")
    for path in ("acquisition.verified", "approval.resources_confirmed"):
        if _at(spec, path) is not True:
            fail(path, "Documented verification required")
    acq = spec["acquisition"]
    if type(acq["artifact_annotations_available"]) is not bool:
        fail("acquisition.artifact_annotations_available", "Record whether blinded artifact annotation is available")
    if type(acq["effective_rank"]) is not int or not 1 <= acq["effective_rank"] <= 7:
        fail("acquisition.effective_rank", "Measure and record input rank from feasibility data in [1,7]")
    try:
        _ids(acq["eog_channels"], "acquisition.eog_channels", allow_empty=True)
        if set(acq["eog_channels"]) & set(CHANNELS):
            fail("acquisition.eog_channels", "EOG availability must not relabel an analysis EEG channel")
        excluded = set(_ids(spec["data"]["excluded_subject_ids"], "data.excluded_subject_ids"))
        prior = set(_ids(spec["data"]["prior_exposed_subject_ids"], "data.prior_exposed_subject_ids", allow_empty=True))
        if not LEGACY_SUBJECTS | prior <= excluded:
            fail("data.excluded_subject_ids", "All discovery and previously exposed identities must remain excluded")
    except ControlPlanError as exc:
        fail("data", str(exc))
    response = spec["procedure"]["response_seconds"]
    if type(response) not in (int, float) or not np.isfinite(response) or not 0 < response <= 60:
        fail("procedure.response_seconds", "Fix one finite response window for both conditions before acquisition")
    if not isinstance(spec["procedure"]["same_participant_AB"], str) or spec["procedure"]["same_participant_AB"] not in {"disabled", "paired_transport_descriptive"}:
        fail("procedure.same_participant_AB", "Choose no A/B reuse or exact paired descriptive transport before acquisition")
    target, cap = (spec["recruitment"][x] for x in ("target_unique_participants", "cap_unique_participants"))
    if type(target) is not int or type(cap) is not int or not 4 <= target <= cap <= 10_000:
        fail("recruitment", "Prespecify integer unique-person target and cap: 4 <= target <= cap <= 10000")
    for name, method in (("reference_sensitivity", "seven_frontal_average"), ("artifact_sensitivity", "blinded_annotation_reject_any_overlap")):
        item = spec["matrix"][name]
        if not isinstance(item["decision"], str) or item["decision"] not in {"enabled", "disabled"} or item["method"] != method or not _text(item["reason"]):
            fail(f"matrix.{name}", "Resolve feasibility and record enabled/disabled plus reason before seeing outcomes")
        if name == "artifact_sensitivity" and item["decision"] == "enabled":
            if acq["artifact_annotations_available"] is not True or not _text(item["annotation_ontology"]):
                fail(f"matrix.{name}", "Enabled annotation sensitivity requires available blinded annotations and a frozen ontology")
    for name in ("registered_at", "approved_at"):
        try:
            _utc(spec["approval"][name], f"approval.{name}")
        except ControlPlanError as exc:
            fail(f"approval.{name}", str(exc))
    for role in ROLES:
        item = spec["evidence"][role]
        if not isinstance(item, dict) or set(item) != {"path", "sha256"} or not _text(item.get("path")) or not isinstance(item.get("sha256"), str) or not SHA.fullmatch(item["sha256"]):
            fail(f"evidence.{role}", "Actual hashed feasibility/permission/identity evidence required", "CONTROL_EVIDENCE")
    if base_dir is None and evidence_bytes is None:
        fail("evidence", "Provide an evidence base directory or archived bytes; declared hashes alone cannot establish readiness", "CONTROL_EVIDENCE")
    else:
        try:
            blobs = _evidence(spec, base_dir, evidence_bytes)
            feasibility, approval, registry = (_json(blobs[role]) for role in ROLES)
            expected = {
                "acquisition_feasibility": {"schema_version": "eeg-control-feasibility-1", "verified": True,
                    "acquisition_sha256": fingerprint(acq), "procedure_sha256": fingerprint(spec["procedure"]),
                    "sensitivity_sha256": fingerprint(spec["matrix"]), "outcomes_used": False},
                "data_use_approval": {"schema_version": "eeg-control-approval-1", "study_id": spec["study_id"],
                    "approval_sha256": fingerprint(spec["approval"]), "approved": True},
                "participant_registry": {"schema_version": "eeg-control-registry-1", "cohort_id": spec["data"]["cohort_id"],
                    "excluded_subject_ids": spec["data"]["excluded_subject_ids"], "cross_cohort_identity_checked": True},
            }
            for role, value in zip(ROLES, (feasibility, approval, registry)):
                if set(value) != {*expected[role], "evidence_reference"} or any(value.get(k) != v or (type(v) is bool and type(value.get(k)) is not bool) for k, v in expected[role].items()) or not _text(value.get("evidence_reference")):
                    fail(f"evidence.{role}", "Evidence does not bind the actual prospective decisions", "CONTROL_EVIDENCE")
        except (OSError, KeyError, ControlPlanError) as exc:
            fail("evidence", str(exc), "CONTROL_EVIDENCE")
    return {"ready_to_lock": not findings, "status": "ready" if not findings else "blocked", "findings": findings,
            "cells": expected_cells(spec), "scope": "prospective_B_sensitivity_plan_A_primary_unchanged",
            "matrix_status": "ready_to_freeze" if not findings else "provisional_pending_readiness",
            "maximum_planned_cells": 9}


def _require_ready(report):
    if not report["ready_to_lock"]:
        raise ControlPlanError("; ".join(f"{x['path']}: {x['message']}" for x in report["findings"]))


def _validate_code(code):
    if not isinstance(code, dict) or not isinstance(code.get("commit"), str) or not re.fullmatch(r"[0-9a-f]{40}", code["commit"]) or code.get("dirty") is not False or not isinstance(code.get("source_files"), dict) or not code["source_files"] or code.get("working_source_sha256") != fingerprint(code["source_files"]):
        raise ControlPlanError("Control plan requires a clean committed implementation and its source hashes")
    if any(not isinstance(k, str) or not isinstance(v, str) or not SHA.fullmatch(v) for k, v in code["source_files"].items()):
        raise ControlPlanError("Invalid implementation source hash inventory")


def freeze_control_plan(plan_path, out_dir):
    """Freeze actual reviewed planning evidence; never activates protocol A's B flag."""
    path = Path(plan_path)
    plan_bytes = path.read_bytes()
    spec = _json(plan_bytes)
    blobs = _evidence(spec, path.parent)
    _require_ready(check_control_plan(spec, evidence_bytes=blobs))
    created = _now()
    for name in ("registered_at", "approved_at"):
        if _utc(spec["approval"][name], name) > _utc(created, "created_at"):
            raise ControlPlanError(f"approval.{name} must precede the plan freeze")
    code = code_identity()
    _validate_code(code)
    environment = environment_identity()
    validate_environment_evidence(environment, require_reference=True)
    with new_output_directory(out_dir) as stage:
        (stage / "plan.json").write_bytes(plan_bytes)
        for role, data in blobs.items():
            (stage / f"{role}.json").write_bytes(data)
        _write(stage / "matrix.json", {"cells": expected_cells(spec)})
        _write(stage / "allocation.json", allocation_schedule(spec))
        _write(stage / "environment.json", environment)
        manifest = {"schema_version": LOCK_SCHEMA, "study_id": spec["study_id"], "created_at": created,
                    "code": code, "scope": "B_plan_and_input_authorization_only_not_A_primary_or_B_model_fit",
                    "artifacts": {p.name: {"sha256": sha256_file(p), "bytes": p.stat().st_size} for p in sorted(stage.iterdir())}}
        manifest["control_plan_lock_id"] = fingerprint(manifest)
        _write(stage / "control_plan_lock.json", manifest)
        if path.read_bytes() != plan_bytes or _evidence(spec, path.parent) != blobs or code_identity() != code:
            raise ControlPlanError("Planning inputs or implementation changed while freezing")
    return manifest


def validate_control_plan_lock(lock_dir):
    root = Path(lock_dir)
    if any(p.is_symlink() for p in [root, *root.parents]):
        raise ControlPlanError("Control plan archive cannot contain symlink paths")
    manifest = read_plan(root / "control_plan_lock.json")
    body = {k: v for k, v in manifest.items() if k != "control_plan_lock_id"}
    if set(manifest) != {"schema_version", "study_id", "created_at", "code", "scope", "artifacts", "control_plan_lock_id"} or manifest["schema_version"] != LOCK_SCHEMA or manifest["control_plan_lock_id"] != fingerprint(body):
        raise ControlPlanError("Control plan lock identity mismatch")
    names = {"plan.json", "matrix.json", "allocation.json", "environment.json", *(f"{role}.json" for role in ROLES)}
    if set(manifest["artifacts"]) != names or {p.name for p in root.iterdir()} != names | {"control_plan_lock.json"}:
        raise ControlPlanError("Control plan archive inventory is incomplete or contains unexpected files")
    for name, descriptor in manifest["artifacts"].items():
        path = root / name
        if path.is_symlink() or not path.is_file() or descriptor != {"sha256": sha256_file(path), "bytes": path.stat().st_size}:
            raise ControlPlanError(f"Control plan archive artifact mismatch: {name}")
    spec = read_plan(root / "plan.json")
    _require_ready(check_control_plan(spec, evidence_bytes={role: (root / f"{role}.json").read_bytes() for role in ROLES}))
    if manifest["study_id"] != spec["study_id"] or manifest["scope"] != "B_plan_and_input_authorization_only_not_A_primary_or_B_model_fit":
        raise ControlPlanError("Control plan scope or study identity mismatch")
    created = _utc(manifest["created_at"], "created_at")
    if any(_utc(spec["approval"][name], name) > created for name in ("registered_at", "approved_at")):
        raise ControlPlanError("Approval and registration must precede plan freeze")
    if read_plan(root / "matrix.json") != {"cells": expected_cells(spec)} or read_plan(root / "allocation.json") != allocation_schedule(spec):
        raise ControlPlanError("Frozen matrix or balanced allocation differs from the plan")
    _validate_code(manifest["code"])
    validate_environment_evidence(read_plan(root / "environment.json"), require_reference=True)
    return {"manifest": manifest, "plan": spec}


def validate_session_history(spec, history, *, locked_at, as_of=None):
    """Audit all recruited people, including no session or failed B acquisition."""
    if not isinstance(history, dict) or set(history) != {"schema_version", "cohort_id", "closed", "participants", "stop_reason", "stop_reason_detail"} or history["schema_version"] != "eeg-control-session-history-1" or history["cohort_id"] != spec["data"]["cohort_id"] or history["closed"] is not True:
        raise ControlPlanError("Complete closed control session history for the locked cohort is required")
    observed_at = _utc(as_of or _now(), "as_of")
    if _utc(locked_at, "locked_at") > observed_at:
        raise ControlPlanError("Control lock cannot be dated after the current audit time")
    people = history["participants"]
    if not isinstance(people, list) or not people:
        raise ControlPlanError("Full recruited participant roster is required")
    ids = _ids([p.get("subject_id") if isinstance(p, dict) else None for p in people], "session history participants")
    overlap = set(ids) & set(spec["data"]["excluded_subject_ids"])
    if overlap:
        raise ControlPlanError(f"Excluded discovery/development/exposed subjects: {sorted(overlap)}")
    if len(ids) > spec["recruitment"]["cap_unique_participants"]:
        raise ControlPlanError("Recruitment cap counts unique people across every session")
    allocations = allocation_schedule(spec)["slots"]
    selected, flow, seen_sessions = {}, [], set()
    recruitment_times = []
    for slot, person in enumerate(people, 1):
        if set(person) != {"subject_id", "recruited_at", "disposition", "sessions"} or person["disposition"] not in {"completed", "technical_failure", "withdrawn", "lost", "task_shortfall"} or not isinstance(person["sessions"], list):
            raise ControlPlanError("Each roster member needs recruitment, disposition and full session history")
        recruited = _utc(person["recruited_at"], "recruited_at")
        if recruited <= _utc(locked_at, "locked_at"):
            raise ControlPlanError("New control participants must be recruited after plan freeze")
        if recruited > observed_at:
            raise ControlPlanError("Recruitment is dated after the current audit time")
        recruitment_times.append(recruited)
        a_sessions, b_sessions = [], []
        for session in person["sessions"]:
            required = {"session_id", "protocol", "started_at", "qa_passed", "qc_reason_codes", "sequence_index", "eligible_epochs_per_class",
                        "recording_id", "recording_sha256", "qa_evidence_sha256", "eligibility_manifest_sha256"}
            if not isinstance(session, dict) or set(session) != required or not _text(session["session_id"]) or type(session["qa_passed"]) is not bool:
                raise ControlPlanError("Invalid session identity or QA evidence")
            if session["session_id"] in seen_sessions:
                raise ControlPlanError("Duplicate session ID across the complete history")
            seen_sessions.add(session["session_id"])
            if not _text(session["recording_id"]) or any(not isinstance(session[key], str) or not SHA.fullmatch(session[key]) for key in ("recording_sha256", "qa_evidence_sha256", "eligibility_manifest_sha256")):
                raise ControlPlanError("Each session requires literal recording identity and recording/QA/eligibility source hashes")
            _ids(session["qc_reason_codes"], "qc_reason_codes", allow_empty=True)
            if session["qa_passed"] == bool(session["qc_reason_codes"]):
                raise ControlPlanError("Session QA status and exclusion reasons conflict")
            counts = session["eligible_epochs_per_class"]
            if not isinstance(counts, dict) or set(counts) != {"0", "1"} or any(type(value) is not int or not 0 <= value <= spec["preprocessing"]["max_epochs_per_class"] for value in counts.values()):
                raise ControlPlanError("Session eligible class counts must be explicit integers within the locked budget")
            if not session["qa_passed"] and any(counts.values()):
                raise ControlPlanError("Whole-recording QA rejection leaves zero eligible epochs")
            started = _utc(session["started_at"], "started_at")
            if started < recruited:
                raise ControlPlanError("Session acquisition cannot precede prospective recruitment")
            if started > observed_at:
                raise ControlPlanError("Session acquisition is dated after the current audit time")
            if session["protocol"] == "B":
                if type(session["sequence_index"]) is not int or session["sequence_index"] != allocations[slot - 1]["sequence_index"]:
                    raise ControlPlanError("Condition-order assignment differs from the locked balanced allocation")
                b_sessions.append(session)
            elif session["protocol"] == "A" and spec["procedure"]["same_participant_AB"] == "paired_transport_descriptive":
                if session["sequence_index"] is not None:
                    raise ControlPlanError("A session must not claim a B sequence")
                a_sessions.append(session)
            else:
                raise ControlPlanError("Unplanned protocol session in control history")
        if len(b_sessions) > 1 or len(a_sessions) > 1:
            raise ControlPlanError("Additional sessions cannot be used to select the best score")
        if a_sessions and b_sessions and _utc(a_sessions[0]["started_at"], "A") >= _utc(b_sessions[0]["started_at"], "B"):
            raise ControlPlanError("Paired A/B transport requires prospectively fixed A-before-B order")
        accepted = b_sessions[0] if b_sessions and b_sessions[0]["qa_passed"] else None
        eligible = bool(accepted and all(n >= spec["preprocessing"]["min_epochs_per_class"] for n in accepted["eligible_epochs_per_class"].values()))
        if person["disposition"] == "completed" and not eligible:
            raise ControlPlanError("Completed participant lacks a QA-valid B session with minimum class counts")
        if eligible and person["disposition"] != "completed":
            raise ControlPlanError("QA-valid B session with sufficient epochs conflicts with participant disposition")
        if accepted is not None and not eligible and person["disposition"] != "task_shortfall":
            raise ControlPlanError("Insufficient correct epochs require an explicit task_shortfall disposition")
        if accepted is not None:
            selected[person["subject_id"]] = accepted["session_id"]
        flow.append({"subject_id": person["subject_id"], "disposition": person["disposition"], "selected_B_session": accepted["session_id"] if accepted else None,
                     "n_sessions": len(person["sessions"]), "pre_phase_QC_evaluable": eligible,
                     "eligible_epochs_per_class": accepted["eligible_epochs_per_class"] if accepted else {"0": 0, "1": 0},
                     "AB_pair_available": bool(eligible and a_sessions and a_sessions[0]["qa_passed"]),
                     "qc_reason_codes": sorted({reason for s in person["sessions"] for reason in s["qc_reason_codes"]})})
    if recruitment_times != sorted(recruitment_times):
        raise ControlPlanError("Roster order must follow recruitment order for balanced slot allocation")
    n_complete = sum(item["pre_phase_QC_evaluable"] for item in flow)
    target = spec["recruitment"]["target_unique_participants"]
    cap = spec["recruitment"]["cap_unique_participants"]
    reason = "target_reached" if n_complete >= target else "cap_reached" if len(ids) == cap else "administrative_shortfall"
    if history["stop_reason"] != reason or (reason == "administrative_shortfall" and not _text(history["stop_reason_detail"])) or (reason != "administrative_shortfall" and history["stop_reason_detail"] is not None):
        raise ControlPlanError("Stopping evidence contradicts target/cap; an administrative shortfall needs an explicit reason")
    return {"n_recruited": len(ids), "n_B_QA_evaluable": len(selected), "selected_B_sessions": selected, "flow": flow,
            "n_pre_phase_QC_evaluable": n_complete, "target_reached": n_complete >= target, "stop_reason": reason,
            "stop_reason_detail": history["stop_reason_detail"], "planned_recruitment_complete": reason != "administrative_shortfall",
            "n_independent_participants": len(ids), "N_increment_from_repeat_sessions": 0}


def phase_arm_contract(spec, arm, annotation_manifest, session_history):
    """The scientific preprocessing identity includes its full acquisition context."""
    return {"arm": arm, "acquisition": spec["acquisition"], "preprocessing": spec["preprocessing"],
                    "reference_policy": spec["matrix"]["reference_sensitivity"] if arm == "seven_frontal_average" else "recorded_reference",
                    "artifact_policy": spec["matrix"]["artifact_sensitivity"] if arm == "annotation_clean" else "structural_QA_only",
                    "session_history_sha256": fingerprint(session_history),
                    "annotation_manifest_sha256": fingerprint(annotation_manifest) if annotation_manifest is not None else None,
                    "seven_frontal_average_is_whole_scalp": False, "rank_upper_bound": min(spec["acquisition"]["effective_rank"], 6 if arm == "seven_frontal_average" else 7)}


def _phase_input_policy(spec, manifest, epoch_manifest, split_manifest, arm_contract, *, code_sha256, environment_sha256):
    return {"schema_version": "eeg-matched-phase-policy-1", "analysis_id": spec["analysis_id"], "evidence_kind": "prospective_data",
              "protocol_lock_id": manifest["control_plan_lock_id"], "code_sha256": code_sha256,
              "environment_sha256": environment_sha256, "preprocessing_sha256": fingerprint(arm_contract),
              "selection_sha256": fingerprint(spec["selection"]), "epoch_manifest_sha256": fingerprint(epoch_manifest),
              "split_manifest_sha256": fingerprint(split_manifest), "channels": CHANNELS, "band_hz": [8.0, 13.0],
              "ciplv_min_denominator": 1e-6, "undefined_ciplv_policy": "exclude_epoch_from_all_representations",
              "min_epochs_per_class": 20, "max_epochs_per_class": 30}


def validate_control_phase_context(locked, epoch_manifest, split_manifest, session_history, *,
                                   arm="recorded_reference", annotation_manifest=None, as_of=None):
    """Verify archived scientific/session context without consulting live code or packages."""
    spec, manifest = locked["plan"], locked["manifest"]
    if arm not in {cell["arm"] for cell in expected_cells(spec)}:
        raise ControlPlanError("Requested sensitivity arm is absent from the prospectively frozen matrix")
    if arm == "annotation_clean":
        if not isinstance(annotation_manifest, dict) or set(annotation_manifest) != {"schema_version", "ontology", "blinded_to_outcomes", "excluded_epoch_ids", "evidence_sha256"} or annotation_manifest["schema_version"] != "eeg-control-annotation-1" or annotation_manifest["ontology"] != spec["matrix"]["artifact_sensitivity"]["annotation_ontology"] or annotation_manifest["blinded_to_outcomes"] is not True:
            raise ControlPlanError("Annotation arm requires its frozen ontology and outcome-blinded annotation evidence")
        _ids(annotation_manifest["excluded_epoch_ids"], "annotation excluded epochs", allow_empty=True)
        if not isinstance(annotation_manifest["evidence_sha256"], str) or not SHA.fullmatch(annotation_manifest["evidence_sha256"]):
            raise ControlPlanError("Annotation evidence requires an actual source SHA-256")
    elif annotation_manifest is not None:
        raise ControlPlanError("Annotation evidence is only valid for the prespecified annotation arm")
    audit = validate_session_history(spec, session_history, locked_at=manifest["created_at"], as_of=as_of)
    rows = epoch_manifest.get("epochs", []) if isinstance(epoch_manifest, dict) else []
    if not isinstance(rows, list) or not rows or any(not isinstance(row, dict) for row in rows) or {row.get("subject_id") for row in rows} != set(audit["selected_B_sessions"]):
        raise ControlPlanError("Epochs must cover every selected B participant; no missing-person inner join")
    for row in rows:
        if row.get("cohort_id") != spec["data"]["cohort_id"] or row.get("session_id") != audit["selected_B_sessions"].get(row.get("subject_id")) or row.get("comparison") != spec["procedure"]["comparison"]:
            raise ControlPlanError("Epoch cohort, comparison or session differs from locked B history")
        if type(row.get("start_sample")) is not int or type(row.get("end_sample")) is not int or row["end_sample"] - row["start_sample"] != 2500:
            raise ControlPlanError("Prospective B epochs must contain the locked 5 seconds at 500 Hz")
    for person in session_history["participants"]:
        selected_id = audit["selected_B_sessions"].get(person["subject_id"])
        if selected_id is not None:
            session = next(s for s in person["sessions"] if s["session_id"] == selected_id)
            person_rows = [row for row in rows if row.get("subject_id") == person["subject_id"]]
            if any(row.get("recording_id") != session["recording_id"] or row.get("recording_sha256") != session["recording_sha256"] for row in person_rows):
                raise ControlPlanError("Epoch recording identity differs from the selected session evidence")
            if session["eligibility_manifest_sha256"] != session_epoch_identity(person_rows):
                raise ControlPlanError("Epoch eligibility identity differs from the acquisition history manifest")
    for flow in audit["flow"]:
        if flow["selected_B_session"] is not None:
            observed = {str(label): sum(row.get("class_label") == label for row in rows if row.get("subject_id") == flow["subject_id"]) for label in (0, 1)}
            if observed != flow["eligible_epochs_per_class"]:
                raise ControlPlanError("Epoch class coverage differs from the full acquisition/session history")
    if annotation_manifest is not None:
        excluded = set(annotation_manifest["excluded_epoch_ids"])
        if not excluded <= {row.get("epoch_id") for row in rows}:
            raise ControlPlanError("Annotation manifest contains unknown epochs")
        for row in rows:
            expected_passed = row.get("epoch_id") not in excluded
            expected_reasons = [] if expected_passed else ["BLINDED_ARTIFACT_ANNOTATION"]
            if row.get("qc_passed") is not expected_passed or row.get("qc_reason_codes") != expected_reasons:
                raise ControlPlanError("Annotated epoch exclusions must be explicit and identical for all representations")
    elif any(row.get("qc_passed") is not True or row.get("qc_reason_codes") != [] for row in rows):
        raise ControlPlanError("QA-valid recording arms cannot introduce an unplanned selective epoch mask")
    arm_contract = phase_arm_contract(spec, arm, annotation_manifest, session_history)
    # Only the common structural validator consumes this transient policy. The
    # actual extraction policy separately binds its captured runtime evidence.
    policy = _phase_input_policy(spec, manifest, epoch_manifest, split_manifest, arm_contract,
        code_sha256=manifest["code"]["working_source_sha256"],
        environment_sha256=manifest["artifacts"]["environment.json"]["sha256"])
    from .phase_controls import validate_matched_phase_manifests
    validate_matched_phase_manifests(epoch_manifest, split_manifest, input_policy=policy, n_samples=2500)
    return {"arm_contract": arm_contract, "session_history_audit": audit}


def authorize_phase_inputs(lock_dir, epoch_manifest, split_manifest, session_history, *, arm="recorded_reference", annotation_manifest=None):
    """Bind a reviewed B plan and attested acquisition history to phase extraction.

    This checks identities, counts and scientific decisions. It does not verify
    electrode placement, correct responses or block timing from phase values.
    The referenced raw recording and QA evidence must remain available.
    """
    locked = validate_control_plan_lock(lock_dir)
    spec, manifest = locked["plan"], locked["manifest"]
    context = validate_control_phase_context(locked, epoch_manifest, split_manifest, session_history,
        arm=arm, annotation_manifest=annotation_manifest)
    current = code_identity()
    _validate_code(current)
    if current["working_source_sha256"] != manifest["code"]["working_source_sha256"]:
        raise ControlPlanError("Phase extraction implementation differs from the frozen B plan")
    environment = environment_identity()
    validate_environment_evidence(environment, require_reference=True)
    archived = read_plan(Path(lock_dir) / "environment.json")
    if environment["lock_id"] != archived["lock_id"]:
        raise ControlPlanError("Phase extraction environment differs from the frozen reference lock")
    policy = _phase_input_policy(spec, manifest, epoch_manifest, split_manifest, context["arm_contract"],
        code_sha256=current["working_source_sha256"], environment_sha256=fingerprint(environment))
    return {"input_policy": policy, "arm_contract": context["arm_contract"],
            "recruitment_audit": context["session_history_audit"], "session_history_sha256": fingerprint(session_history),
            "environment": environment, "implementation": current}


def session_epoch_identity(rows):
    """Bind original eligibility identities separately from sensitivity QC flags.

    The caller must retain the referenced recording and QA evidence. This hash
    cannot establish correct-trial selection, measured reference, or event
    timing from phase values alone.
    """
    fields = ("epoch_id", "cohort_id", "subject_id", "session_id", "recording_id", "recording_sha256",
              "start_sample", "end_sample", "comparison", "class_label")
    try:
        values = [{key: row[key] for key in fields} for row in rows]
        return fingerprint(sorted(values, key=lambda row: row["epoch_id"]))
    except (KeyError, TypeError) as exc:
        raise ControlPlanError("Incomplete epoch eligibility identity") from exc


def validate_control_result_matrix(spec, rows, *, protocol_lock_id, participant_ids):
    """Require every planned result, including unevaluable/negative outcomes.

    This manifest checks completeness and shared inputs; numerical participant
    metrics must additionally pass result_contract before publication.
    """
    if not _text(protocol_lock_id):
        raise ControlPlanError("An independently verified control-plan lock ID is required")
    roster = set(_ids(participant_ids, "complete recruited participant roster"))
    wanted = {x["cell_id"]: x for x in expected_cells(spec)}
    required = {"cell_id", "analysis_id", "protocol_lock_id", "run_id", "model_id", "participant_ids", "status", "reason",
                "eligible_epoch_manifest_sha256", "split_manifest_sha256", "preprocessing_sha256", "selection_sha256", "source_predictions_sha256"}
    if not isinstance(rows, list) or any(not isinstance(row, dict) or set(row) != required for row in rows):
        raise ControlPlanError("Control result cells require exact identity, manifest, status and source fields")
    ids = [r["cell_id"] for r in rows]
    if len(set(ids)) != len(ids) or set(ids) != set(wanted):
        raise ControlPlanError(f"Complete frozen matrix required: missing={sorted(set(wanted) - set(ids))}; unexpected={sorted(set(ids) - set(wanted))}; duplicates={len(ids) != len(set(ids))}")
    for row in rows:
        _ids(row["participant_ids"], "result participant_ids")
        if row["analysis_id"] != spec["analysis_id"] or any(not _text(row[k]) for k in ("protocol_lock_id", "run_id", "model_id")):
            raise ControlPlanError("Control results require compatible analysis/run/model/plan identities")
        if row["status"] not in {"evaluable", "unevaluable"} or (row["status"] == "unevaluable" and not _text(row["reason"])) or (row["status"] == "evaluable" and row["reason"] is not None):
            raise ControlPlanError("Every planned cell needs an evaluability status with an explicit reason when unavailable")
        for key in ("eligible_epoch_manifest_sha256", "split_manifest_sha256", "preprocessing_sha256", "selection_sha256", "source_predictions_sha256"):
            if not isinstance(row[key], str) or not SHA.fullmatch(row[key]):
                raise ControlPlanError(f"Invalid control result hash: {key}")
    if {r["protocol_lock_id"] for r in rows} != {protocol_lock_id} or any(set(r["participant_ids"]) != roster for r in rows):
        raise ControlPlanError("Control result cells require the exact same participant roster and plan lock")
    for arm in {x["arm"] for x in wanted.values()}:
        selected = [r for r in rows if wanted[r["cell_id"]]["arm"] == arm]
        for key in ("eligible_epoch_manifest_sha256", "split_manifest_sha256", "preprocessing_sha256", "selection_sha256"):
            if len({r[key] for r in selected}) != 1:
                raise ControlPlanError(f"The three {arm} representations must share {key}")
    if any(r["selection_sha256"] != fingerprint(spec["selection"]) for r in rows):
        raise ControlPlanError("Control selection differs from the frozen rule")
    return {"n_planned_cells": len(wanted), "n_reported_cells": len(rows), "n_participants": len(rows[0]["participant_ids"]),
            "complete": True, "all_signs_required": True, "stop_after_matrix": True,
            "cross_arm_paired_deltas_authorized": False}
