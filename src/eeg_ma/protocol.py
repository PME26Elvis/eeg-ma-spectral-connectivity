"""Prospective protocol locks, immutable evidence, and auditable outcome access.

The ledger enforces a local audit contract.  It cannot establish that a person
has never seen an unrecorded result or authenticate an operator's attestations.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any, Callable

from .artifacts import new_output_directory
from .provenance import (
    _atomic_json, _matrix_lock, canonical_json, code_identity, fingerprint,
    protect_output, sha256_file,
)


PROTOCOL_SCHEMA = "eeg-confirmatory-protocol-1"
LOCK_SCHEMA = "eeg-protocol-lock-1"
EVENT_SCHEMA = "eeg-protocol-event-1"
SHA = re.compile(r"[0-9a-f]{64}\Z")
COMMIT = re.compile(r"[0-9a-f]{40}\Z")
SENTINELS = {"", "pending", "unknown", "tbd", "todo", "to be determined", "not_verified"}
ARTIFACT_ROLES = (
    "development_manifest", "development_data_manifest", "analysis_config", "environment",
    "primary_model", "reference_model", "primary_training_manifest",
    "reference_training_manifest", "feature_schema", "precision_report", "acquisition_evidence",
)
SENSITIVE_SECTIONS = ("acquisition", "protocols", "eligibility", "endpoints", "inference", "models", "recruitment")


class ProtocolError(ValueError):
    """A required prospective decision or artifact is missing or inconsistent."""


@dataclass(frozen=True)
class ProtocolFinding:
    code: str
    path: str
    message: str


@dataclass
class ProtocolCheck:
    findings: list[ProtocolFinding] = field(default_factory=list)

    @property
    def ready(self) -> bool:
        return not self.findings

    @property
    def passed(self) -> bool:
        return self.ready

    def as_dict(self) -> dict:
        return {"status": "ready" if self.ready else "blocked", "ready_to_lock": self.ready,
                "findings": [vars(item) for item in self.findings]}

    def raise_for_errors(self) -> None:
        if self.findings:
            raise ProtocolError("; ".join(f"{f.code} {f.path}: {f.message}" for f in self.findings))


def _object_pairs(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise ProtocolError(f"Duplicate JSON key: {key}")
        value[key] = item
    return value


def _json_bytes(data: bytes, label: str) -> dict:
    try:
        value = json.loads(data.decode("utf-8-sig"), object_pairs_hook=_object_pairs,
                           parse_constant=lambda value: (_ for _ in ()).throw(ProtocolError(f"Nonfinite JSON: {value}")))
    except (ValueError, UnicodeError) as exc:
        raise ProtocolError(f"Invalid JSON evidence {label}: {exc}") from exc
    if not isinstance(value, dict):
        raise ProtocolError(f"JSON evidence must be an object: {label}")
    return value


def _read(path: str | Path) -> dict:
    try:
        return _json_bytes(Path(path).read_bytes(), str(path))
    except OSError as exc:
        raise ProtocolError(f"Missing evidence: {path}") from exc


def _write(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _utc(value: Any, label: str) -> datetime:
    if not isinstance(value, str):
        raise ProtocolError(f"{label} must be an explicit timezone-aware timestamp")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ProtocolError(f"Invalid timestamp: {label}") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ProtocolError(f"Timestamp lacks a UTC offset: {label}")
    return parsed.astimezone(timezone.utc)


def _at(value: dict, path: str, default=None):
    for key in path.split("."):
        if not isinstance(value, dict) or key not in value:
            return default
        value = value[key]
    return value


def _text(value: Any) -> bool:
    return (isinstance(value, str) and value == value.strip() and value.lower() not in SENTINELS
            and re.match(r"^(pending|unknown|tbd|todo)(?:\b|[_:-])", value, flags=re.IGNORECASE) is None)


def _ids(value: Any, label: str, *, allow_empty=False) -> list[str]:
    if not isinstance(value, list) or (not value and not allow_empty):
        raise ProtocolError(f"{label} requires a list of literal subject IDs")
    if any(not _text(v) for v in value) or len(set(value)) != len(value):
        raise ProtocolError(f"{label} contains blank, coerced, or duplicate subject IDs")
    return value


def validate_protocol(spec: dict, *, stage="draft") -> ProtocolCheck:
    """List every blocking decision; a draft is never silently promoted."""
    if stage not in {"draft", "lock"}:
        raise ProtocolError("stage must be draft or lock")
    report = ProtocolCheck()
    def fail(code, path, message):
        report.findings.append(ProtocolFinding(code, path, message))
    if not isinstance(spec, dict):
        fail("SCHEMA", "", "Protocol must be an object")
        return report
    if spec.get("schema_version") != PROTOCOL_SCHEMA:
        fail("SCHEMA", "schema_version", f"Expected {PROTOCOL_SCHEMA}")
    if spec.get("status") not in {"draft", "locked"}:
        fail("STATE", "status", "Only draft and locked protocol snapshots are supported")
    for path in ("protocol_id", "protocol_version", "study.cohort_id", "study.mode", "study.population",
                 "study.primary_estimand", "study.expanded_lopo_estimand", "acquisition.units", "acquisition.device",
                 "acquisition.reference", "acquisition.ground", "acquisition.event_clock", "acquisition.evidence",
                 "approval.responsible_person", "approval.approved_at", "approval.ethics_status", "approval.ethics_reference",
                 "eligibility.missing_policy", "eligibility.replacement_policy", "endpoints.secondary_policy",
                 "endpoints.multiple_comparisons", "endpoints.weighting", "recruitment.stop_rule",
                 "protocols.B.visual_response_control", "inference.sampling_assumption"):
        if not _text(_at(spec, path)):
            fail("PENDING", path, "An explicit verified decision is required")
    if _at(spec, "study.mode") not in {"confirmatory", "pilot_estimation"}:
        fail("STUDY_MODE", "study.mode", "Choose confirmatory or pilot_estimation prospectively")
    if _at(spec, "study.outcomes_accessed") is not False:
        fail("OUTCOME_ACCESS", "study.outcomes_accessed", "A prospective snapshot must precede outcome access")
    if _at(spec, "acquisition.verified") is not True:
        fail("ACQUISITION_UNVERIFIED", "acquisition.verified", "Actual acquisition facts need operator verification")
    for path, minimum in (("recruitment.target_evaluable", 2), ("recruitment.recruitment_cap", 2),
                          ("eligibility.min_epochs_per_class", 1), ("eligibility.max_epochs_per_class", 1), ("eligibility.max_attempts_per_task", 1),
                          ("eligibility.max_session_seconds", 1), ("eligibility.max_replacement_sessions", 0),
                          ("eligibility.replacement_window_days", 1), ("inference.seed", 0),
                          ("study.minimum_age_years", 18), ("protocols.A.rest_seconds", 1),
                          ("protocols.A.target_correct_trials_per_task", 1), ("protocols.A.mini_chain_length", 1),
                          ("eligibility.planned_rest_epochs_per_block", 1), ("eligibility.planned_rest_seconds_per_block", 1),
                          ("protocols.B.proposed_blocks", 1), ("protocols.B.epochs_per_block", 1), ("protocols.B.washout_seconds", 1),
                          ("inference.resamples", 1000)):
        value = _at(spec, path)
        if type(value) is not int or value < minimum:
            fail("NUMERIC_DECISION", path, f"Requires integer >= {minimum}; booleans and coercions are forbidden")
    n, cap = _at(spec, "recruitment.target_evaluable"), _at(spec, "recruitment.recruitment_cap")
    if type(n) is int and type(cap) is int and cap < n:
        fail("RECRUITMENT_CAP", "recruitment", "Recruitment cap must be at least target evaluable N")
    count_paths = ("eligibility.min_epochs_per_class", "eligibility.max_epochs_per_class",
                   "protocols.A.target_correct_trials_per_task", "eligibility.max_attempts_per_task")
    counts = [_at(spec, path) for path in count_paths]
    if all(type(value) is int for value in counts) and not (counts[0] <= counts[1] == counts[2] <= counts[3]):
        fail("EPOCH_COUNTS", "eligibility", "Require minimum <= maximum = task target <= attempt budget")
    rest = _at(spec, "eligibility.planned_rest_epochs_per_block")
    if type(rest) is int and type(counts[1]) is int and rest != counts[1]:
        fail("EPOCH_COUNTS", "eligibility.planned_rest_epochs_per_block", "Rest target must equal the maximum per-class epoch count")
    maximum_age = _at(spec, "study.maximum_age_years")
    if maximum_age is not None and (type(maximum_age) is not int or type(_at(spec, "study.minimum_age_years")) is not int or maximum_age < _at(spec, "study.minimum_age_years")):
        fail("ELIGIBILITY", "study.maximum_age_years", "Null means no upper age limit; otherwise provide an integer at least the minimum age")
    for path in ("endpoints.delta_min", "endpoints.absolute_threshold", "inference.confidence"):
        value = _at(spec, path)
        if type(value) not in {float, int} or not 0 < value < 1:
            fail("NUMERIC_DECISION", path, "Requires a finite number strictly between 0 and 1")
    if _at(spec, "eligibility.correct_only") is not True:
        fail("ELIGIBILITY", "eligibility.correct_only", "This protocol fixes correct-only offline eligibility")
    if _at(spec, "inference.unit") != "participant" or _at(spec, "inference.method") != "participant_percentile_bootstrap":
        fail("INFERENCE", "inference", "Bootstrap unit must be participant and method percentile")
    a, b = _at(spec, "protocols.A", {}), _at(spec, "protocols.B", {})
    if not isinstance(a, dict) or not isinstance(b, dict):
        fail("PROTOCOL_IDENTITIES", "protocols", "A and B must be explicit objects")
        a, b = {}, {}
    if not _text(a.get("protocol_id")) or not _text(b.get("protocol_id")) or a.get("protocol_id") == b.get("protocol_id"):
        fail("PROTOCOL_IDENTITIES", "protocols", "A and B need distinct explicit protocol IDs")
    if a.get("role") != "primary" or b.get("role") != "next_study":
        fail("PROTOCOL_ROLES", "protocols", "This v1 lock supports A primary and B as a separate inactive next study")
    if b.get("role") == "next_study" and b.get("enabled") is not False:
        fail("PROTOCOL_ACTIVATION", "protocols.B.enabled", "A next-study design is inactive")
    for path in ("acquisition.historical_alignment_verified", "approval.resource_commitment_confirmed",
                 "eligibility.analysis_eligible_required", "eligibility.primary_independent_of_secondary_completion"):
        if _at(spec, path) is not True:
            fail("VERIFICATION_PENDING", path, "The stated prospective condition must be verified")
    required_values = {
        "acquisition.sampling_rate_hz": 500, "acquisition.epoch_seconds": 5.0,
        "acquisition.raw_channels": ["FP1", "FP2", "F7", "F3", "Fz", "F4", "F8", "Cz"],
        "acquisition.analysis_channels": ["FP1", "FP2", "F7", "F3", "Fz", "F4", "F8"],
        "acquisition.event_clock": "session_start_elapsed_ms", "acquisition.marker_policy": "hardware_required",
        "study.sampling_unit": "independent_participant", "study.site_policy": "one_verified_acquisition_setup",
        "protocols.A.enabled": True, "protocols.A.order": ["Rest1", "Type1", "Rest2", "Type2"],
        "protocols.A.session_selection": "first_QA_valid_A_session", "protocols.A.participant_weight": 1.0,
        "protocols.A.primary_requires_type2_completion": False,
        "protocols.A.calculation_seconds": 5.0, "protocols.A.mini_chain_length": 4,
        "protocols.A.task_types": {"Type1": "three_digit_subtraction_two_borrows", "Type2": "two_digit_subtraction_one_borrow"},
        "protocols.A.primary_comparison": "rest1_vs_type1", "protocols.A.secondary_comparison": "rest2_vs_type2",
        "protocols.A.interpretation": "historical_task_aligned_capped_protocol_discrimination",
        "protocols.A.practice_policy": "instructions_and_response_interface_only_no_scored_task_practice",
        "protocols.A.timing_scope": "formal_session_start_to_session_end_excludes_equipment_setup",
        "protocols.B.primary_weight": 0.0, "protocols.B.independent_N_increment_for_repeat_subjects": 0,
        "protocols.B.session_selection": "separate_protocol_lock_required_before_collection",
        "protocols.B.session_order": "A_before_B_if_the_same_participants_are_reused",
        "protocols.B.allocation": "balanced_random_assignment_before_acquisition",
        "protocols.B.activation_rule": "new_study_lock_with_acquisition_feasibility_evidence; never_promote_B_after_A_results",
        "protocols.B.balanced_sequences": [["RestControl", "Type1"] * 3, ["Type1", "RestControl"] * 3],
        "protocols.B.proposed_blocks": 6,
        "endpoints.primary": "participant_macro_balanced_accuracy_A_rest1_vs_type1",
        "endpoints.primary_model": "alpha_plv_sfs_lda", "endpoints.reference_model": "bp_coh_sfs_lda",
        "endpoints.key_secondary": "participant_paired_delta_vs_bp_coh_sfs_lda", "endpoints.chance_level": 0.5,
        "endpoints.secondary_policy": "descriptive_nominal_CIs_no_confirmatory_secondary_claims",
        "endpoints.secondary_family": ["paired_delta", "Type2", "expanded_nested_LOPO"],
        "endpoints.multiple_comparisons": "one_confirmatory_primary_secondary_descriptive_only",
        "endpoints.weighting": "equal_participant_weights_no_epoch_or_session_pseudoreplication",
        "endpoints.threshold_interpretation": "prospective_research_targets_not_demonstrated_BCI_usability",
        "recruitment.stop_rule": "first_of_target_evaluable_or_unique_participant_cap",
        "recruitment.score_based_extension_allowed": False,
        "recruitment.cap_shortfall_policy": "stop_and_report_precision_shortfall_without_posthoc_extension",
        "recruitment.inflight_policy": "finish_already_recruited_before_outcomes_include_all_eligible",
        "eligibility.amplitude_rejection": "disabled", "eligibility.target_fit_ICA_ASR": "disabled",
        "eligibility.channel_interpolation": "disabled", "eligibility.qc.max_sample_gaps": 0,
        "eligibility.qc.max_nonfinite_values": 0, "eligibility.qc.require_all_raw_channels": True,
        "eligibility.qc.structural_QA_failure": "whole_recording_rejected",
        "eligibility.replacement_policy": "first_session_fails_prespecified_technical_QA_only; never_for_accuracy_or_correctness_shortfall",
        "eligibility.missing_policy": "retain_in_flow_and_exclude_only_unevaluable_endpoint_no_chance_imputation",
        "eligibility.correct_formal_trial_policy": "all_correct_formal_trials_eligible_after_whole_recording_QA",
        "inference.paired_resampling": "identical_participant_indices_for_both_models", "inference.tail": "two_sided",
        "inference.degenerate_policy": "report_degenerate_CI_and_no_precision_adequacy_claim",
        "freeze.model_training_scope": "development_only",
        "freeze.expanded_development_rule": "freeze_new_models_and_reserve_new_unexposed_holdout",
        "freeze.source_code_policy": "capture_commit_and_working_source_fingerprint_at_lock",
        "freeze.environment_policy": "verified_reference_environment",
        "amendment_policy.before_outcomes": "new_version_linked_to_prior_lock_and_no_silent_overwrite",
        "amendment_policy.after_outcomes_sensitive_change": "old_test_subjects_exposed_new_untouched_holdout_required",
        "amendment_policy.pilot_downgrade": "only_before_outcomes_with_new_mode_and_new_lock; preserve_original_plan",
    }
    for path, expected in required_values.items():
        actual = _at(spec, path)
        if actual != expected or (isinstance(expected, bool) and type(actual) is not bool) or (type(expected) in {int, float} and type(actual) not in {int, float}):
            fail("FIXED_CONTRACT", path, "Missing or inconsistent prospective protocol contract")
    for path in ("eligibility.qc.flat_std_epsilon", "eligibility.qc.sample_period_tolerance_s",
                 "eligibility.qc.clock_tolerance_s", "eligibility.qc.sampling_rate_tolerance_hz",
                 "recruitment.target_half_width", "recruitment.completion_assurance", "recruitment.minimum_retention_for_planning",
                 "recruitment.precision_coverage_floor", "recruitment.precision_width_probability_floor"):
        value = _at(spec, path)
        if type(value) not in {int, float} or not 0 < value < 1:
            fail("NUMERIC_DECISION", path, "A positive finite prospective threshold below 1 is required")
    for path in ("recruitment.precision_coverage_floor", "recruitment.precision_width_probability_floor"):
        if type(_at(spec, path)) in {int, float} and _at(spec, path) < .90:
            fail("PRECISION_FLOOR", path, "Coverage and width probability floors cannot be weakened below 0.90")
    if (not isinstance(_at(spec, "eligibility.technical_failure_reasons"), list) or not _at(spec, "eligibility.technical_failure_reasons")
            or any(not _text(x) for x in _at(spec, "eligibility.technical_failure_reasons", []))):
        fail("REPLACEMENT_POLICY", "eligibility.technical_failure_reasons", "Technical replacement reasons must be enumerated")
    if type(_at(spec, "eligibility.max_replacement_sessions")) is int and _at(spec, "eligibility.max_replacement_sessions") > 1:
        fail("REPLACEMENT_POLICY", "eligibility.max_replacement_sessions", "This protocol permits at most one technical replacement")
    if type(rest) is int:
        for path in ("eligibility.planned_rest_seconds_per_block", "protocols.A.rest_seconds"):
            if _at(spec, path) != rest * 5:
                fail("EPOCH_COUNTS", path, "Rest duration must equal its fixed 5-second epoch budget")
    if not isinstance(_at(spec, "freeze.artifacts"), dict):
        fail("ARTIFACTS", "freeze.artifacts", "Frozen evidence map is required")
    elif set(spec["freeze"]["artifacts"]) != set(ARTIFACT_ROLES):
        fail("ARTIFACTS", "freeze.artifacts", "Exactly the declared artifact roles must be frozen")
    for role in ARTIFACT_ROLES:
        entry = _at(spec, "freeze.artifacts." + role)
        if not isinstance(entry, dict) or not _text(entry.get("path")) or not isinstance(entry.get("sha256"), str) or not SHA.fullmatch(entry["sha256"]):
            fail("ARTIFACT_PENDING", f"freeze.artifacts.{role}", "Real file path and exact SHA-256 are required")
    registration = _at(spec, "study.registration", {})
    if not isinstance(registration, dict):
        registration = {}
    if registration.get("status") == "registered":
        url = registration.get("url")
        if not isinstance(url, str) or not url.startswith("https://"):
            fail("REGISTRATION", "study.registration.url", "Provide the actual registration URL")
        try:
            if _utc(registration.get("timestamp"), "registration.timestamp") > datetime.now(timezone.utc):
                fail("REGISTRATION", "study.registration.timestamp", "Registration cannot be in the future")
        except ProtocolError as exc:
            fail("REGISTRATION", "study.registration.timestamp", str(exc))
    elif registration.get("status") == "not_registered":
        if registration.get("url") is not None or registration.get("timestamp") is not None:
            fail("REGISTRATION", "study.registration", "Unregistered protocols cannot claim a URL or registration time")
    else:
        fail("REGISTRATION", "study.registration.status", "Registration status must be explicit")
    try:
        approved = _utc(_at(spec, "approval.approved_at"), "approval.approved_at")
        if approved > datetime.now(timezone.utc):
            fail("APPROVAL_TIME", "approval.approved_at", "Approval cannot be in the future")
    except ProtocolError as exc:
        fail("APPROVAL_PENDING", "approval.approved_at", str(exc))
    allowed_null = {"study.registration.url", "study.registration.timestamp", "study.maximum_age_years"}
    def pending(value, path=""):
        if isinstance(value, dict):
            for key, child in value.items():
                pending(child, f"{path}.{key}".lstrip("."))
        elif isinstance(value, list):
            for index, child in enumerate(value):
                pending(child, f"{path}[{index}]")
        elif value is None:
            if path not in allowed_null:
                fail("PENDING", path, "Null is not a finalized decision")
        elif isinstance(value, str) and not _text(value):
            fail("PENDING", path, "Placeholder values cannot enter a lock")
    pending(spec)
    return report


def _dataset_entries(data: bytes) -> dict:
    manifest = _json_bytes(data, "development_data_manifest")
    if manifest.get("schema_version") != "eeg-development-data-1" or not isinstance(manifest.get("sources"), list) or not manifest["sources"]:
        raise ProtocolError("Development dataset inventory is missing its frozen sources")
    entries = {}
    for item in manifest["sources"]:
        if not isinstance(item, dict):
            raise ProtocolError("Invalid development source entry")
        name = item.get("name")
        if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z][A-Za-z0-9_-]*", name) or name in entries:
            raise ProtocolError("Unsafe or duplicate development source name")
        entries[name] = item
    if "legacy_mapping" in manifest:
        if "legacy_mapping" in entries:
            raise ProtocolError("Duplicate legacy mapping source")
        entries["legacy_mapping"] = manifest["legacy_mapping"]
    for entry in entries.values():
        if not isinstance(entry, dict) or not _text(entry.get("path")) or not isinstance(entry.get("sha256"), str) or not SHA.fullmatch(entry["sha256"]):
            raise ProtocolError("Every development source requires its path and actual bytes hash")
    return entries


def _source_artifacts(spec: dict, base: Path) -> tuple[dict, dict, dict, dict]:
    contents, sources = {}, {}
    for role in ARTIFACT_ROLES:
        entry = spec["freeze"]["artifacts"][role]
        path = Path(entry["path"])
        path = path if path.is_absolute() else base / path
        try:
            data = path.read_bytes()
        except OSError as exc:
            raise ProtocolError(f"Missing frozen artifact {role}: {path}") from exc
        digest = hashlib.sha256(data).hexdigest()
        if digest != entry["sha256"] or not data:
            raise ProtocolError(f"Artifact hash mismatch or empty artifact: {role}")
        contents[role] = data
        sources[role] = path
    dataset_contents, dataset_sources = {}, {}
    dataset_base = sources["development_data_manifest"].parent
    for name, entry in _dataset_entries(contents["development_data_manifest"]).items():
        path = Path(entry["path"])
        path = path if path.is_absolute() else dataset_base / path
        try:
            data = path.read_bytes()
        except OSError as exc:
            raise ProtocolError(f"Missing development data source: {name}") from exc
        if not data or hashlib.sha256(data).hexdigest() != entry["sha256"]:
            raise ProtocolError(f"Development source hash mismatch: {name}")
        dataset_contents[name], dataset_sources[name] = data, path
    return contents, sources, dataset_contents, dataset_sources


def _artifact_semantics(spec: dict, contents: dict[str, bytes], dataset_contents: dict[str, bytes]) -> dict:
    dev = _json_bytes(contents["development_manifest"], "development_manifest")
    if dev.get("schema_version") != "eeg-development-manifest-1":
        raise ProtocolError("development_manifest has an unsupported schema")
    ids = _ids(dev.get("subject_ids"), "development subject_ids")
    data_sha = hashlib.sha256(contents["development_data_manifest"]).hexdigest()
    if dev.get("data_sha256") != data_sha:
        raise ProtocolError("Development data bytes do not match the development manifest")
    from io import BytesIO
    import pandas as pd
    entries = _dataset_entries(contents["development_data_manifest"])
    if set(dataset_contents) != set(entries):
        raise ProtocolError("Incomplete copied development dataset")
    for name, entry in entries.items():
        if hashlib.sha256(dataset_contents[name]).hexdigest() != entry["sha256"]:
            raise ProtocolError("Copied development data differ from the frozen dataset manifest")
        if name == "legacy_mapping":
            _json_bytes(dataset_contents[name], "legacy mapping")
            continue
        data = pd.read_csv(BytesIO(dataset_contents[name]), dtype={"subject_id": str}, keep_default_na=False)
        if "subject_id" not in data or set(data.subject_id) != set(ids):
            raise ProtocolError("Development data subject IDs do not match their manifest")
    schema_sha = hashlib.sha256(contents["feature_schema"]).hexdigest()
    if dev.get("feature_schema_sha256") != schema_sha:
        raise ProtocolError("Development manifest references a different feature schema")
    environment = _json_bytes(contents["environment"], "environment")
    from .environment import validate_environment_evidence
    try:
        validate_environment_evidence(environment, require_reference=True)
    except (ValueError, TypeError, KeyError) as exc:
        raise ProtocolError(f"Frozen environment is not verified reference evidence: {exc}") from exc
    for role, learner in (("primary", "alpha_plv_sfs_lda"), ("reference", "bp_coh_sfs_lda")):
        training = _json_bytes(contents[role + "_training_manifest"], role + "_training_manifest")
        expected = {
            "schema_version": "eeg-training-manifest-1", "model_role": role,
            "learner_id": learner, "model_sha256": hashlib.sha256(contents[role + "_model"]).hexdigest(),
            "development_manifest_sha256": hashlib.sha256(contents["development_manifest"]).hexdigest(),
            "analysis_config_sha256": hashlib.sha256(contents["analysis_config"]).hexdigest(),
            "environment_sha256": hashlib.sha256(contents["environment"]).hexdigest(),
            "environment_id": environment["environment_id"],
            "environment_lock_id": environment["lock_id"],
            "feature_schema_sha256": hashlib.sha256(contents["feature_schema"]).hexdigest(),
            "training_data_sha256": data_sha, "eligibility_rule_sha256": fingerprint(spec["eligibility"]),
            "fitted_on_development_only": True, "target_fit_performed": False,
        }
        if any(training.get(key) != value or (isinstance(value, bool) and type(training.get(key)) is not bool) for key, value in expected.items()):
            raise ProtocolError(f"Frozen {role} training provenance is inconsistent")
        if set(_ids(training.get("training_subject_ids"), role + " training_subject_ids")) != set(ids):
            raise ProtocolError(f"Frozen {role} model used a different development subject set")
    evidence = _json_bytes(contents["acquisition_evidence"], "acquisition_evidence")
    if evidence.get("schema_version") != "eeg-acquisition-verification-1" or evidence.get("verified") is not True:
        raise ProtocolError("Acquisition evidence must contain a verified operator record")
    if evidence.get("acquisition_sha256") != fingerprint(spec["acquisition"]):
        raise ProtocolError("Acquisition evidence does not match declared acquisition facts")
    if not _text(evidence.get("verified_by")):
        raise ProtocolError("Acquisition evidence needs its actual verifier")
    if _utc(evidence.get("verified_at"), "acquisition verified_at") > datetime.now(timezone.utc):
        raise ProtocolError("Acquisition verification cannot be in the future")
    analysis = _json_bytes(contents["analysis_config"], "analysis_config")
    for key in ("sampling_rate_hz", "epoch_seconds", "raw_channels", "analysis_channels"):
        if analysis.get(key) != spec["acquisition"][key]:
            raise ProtocolError(f"Analysis configuration differs from acquisition: {key}")
    if analysis.get("expected_epochs_per_class", "missing") is not None:
        raise ProtocolError("Prospective bounded epoch counts require a null exact-count preflight setting")
    for key in ("sample_period_tolerance_s", "clock_tolerance_s", "flat_std_epsilon"):
        if _at(analysis, "ingestion." + key) != spec["eligibility"]["qc"][key]:
            raise ProtocolError(f"Analysis configuration differs from frozen QA: {key}")
    if _at(analysis, "ingestion.marker_policy") != spec["acquisition"]["marker_policy"]:
        raise ProtocolError("Analysis configuration differs from the frozen marker policy")
    schema = _json_bytes(contents["feature_schema"], "feature_schema")
    expected_schema = {"schema_version": "eeg-confirmatory-feature-schema-1", "feature_table_schema_version": "2.0",
                       "class_mapping": {"0": "Rest", "1": "MentalArithmetic"},
                       "analysis_channels": spec["acquisition"]["analysis_channels"],
                       "sampling_rate_hz": spec["acquisition"]["sampling_rate_hz"], "epoch_seconds": spec["acquisition"]["epoch_seconds"],
                       "identity_policy": "stable_subject_across_all_cohorts_and_sessions", "feature_order_policy": "exact_frozen_order"}
    if any(schema.get(key) != value for key, value in expected_schema.items()):
        raise ProtocolError("Feature schema identity, labels or acquisition contract is inconsistent")
    from itertools import combinations
    channels = spec["acquisition"]["analysis_channels"]
    bands = ["delta", "theta", "alpha", "beta_low", "beta_high", "gamma"]
    expected_features = {
        "primary_features": [f"PLV__{a}__{b}__alpha" for a, b in combinations(channels, 2)],
        "reference_features": [f"BP__{c}__{b}" for c in channels for b in bands] +
                              [f"COH__{a}__{b}__{band}" for a, b in combinations(channels, 2) for band in bands],
    }
    for key, expected_names in expected_features.items():
        if schema.get(key) != expected_names:
            raise ProtocolError(f"Feature schema has invalid ordered {key}")
    if schema.get("primary_learner_id") != spec["endpoints"]["primary_model"] or schema.get("reference_learner_id") != spec["endpoints"]["reference_model"]:
        raise ProtocolError("Feature schema learner identities disagree with the locked models")
    from .precision import validate_precision_report
    precision_document = _json_bytes(contents["precision_report"], "precision_report")
    try:
        precision = validate_precision_report(precision_document)
    except (KeyError, TypeError, ValueError) as exc:
        raise ProtocolError(f"Invalid precision evidence: {exc}") from exc
    if precision["coverage_floor"] < 0.90 or precision["width_probability_floor"] < 0.90:
        raise ProtocolError("Precision validation must retain coverage and half-width probability floors of at least 0.90")
    for actual, expected, label in ((precision["selected_target_n"], spec["recruitment"]["target_evaluable"], "N"),
                                    (precision["recruitment_cap"], spec["recruitment"]["recruitment_cap"], "recruitment cap"),
                                    (precision["confidence_level"], spec["inference"]["confidence"], "confidence"),
                                    (precision["bootstrap_resamples"], spec["inference"]["resamples"], "bootstrap resamples"),
                                    (precision["seed"], spec["inference"]["seed"], "inference seed"),
                                    (precision["ci_method"], "percentile", "CI method"),
                                    (precision["half_width_target"], spec["recruitment"]["target_half_width"], "target half-width"),
                                    (precision["coverage_floor"], spec["recruitment"]["precision_coverage_floor"], "coverage floor"),
                                    (precision["width_probability_floor"], spec["recruitment"]["precision_width_probability_floor"], "width-probability floor"),
                                    (precision["completion_assurance"], spec["recruitment"]["completion_assurance"], "completion assurance"),
                                    (precision["design_retention_floor"], spec["recruitment"]["minimum_retention_for_planning"], "retention assumption")):
        if not (math.isclose(actual, expected, rel_tol=0, abs_tol=1e-12) if type(actual) is float and type(expected) in {int, float} else actual == expected):
            raise ProtocolError(f"Precision report does not match the protocol's {label}")
    epoch_counts = [n for scenario in precision_document["config"]["scenarios"] for n in scenario["epochs_per_class"]]
    if min(epoch_counts) != spec["eligibility"]["min_epochs_per_class"] or max(epoch_counts) != spec["eligibility"]["max_epochs_per_class"]:
        raise ProtocolError("Precision scenarios do not span the locked eligible epoch count range")
    return {"development_subject_ids": sorted(ids), "precision": precision}


def _event_chain(root: Path, lock_id: str, *, allow_orphans=False) -> tuple[list[dict], list[Path]]:
    state = _read(root / "STATE.json")
    if set(state) != {"schema_version", "lock_id", "sequence", "head"} or state.get("schema_version") != "eeg-protocol-state-1" or state.get("lock_id") != lock_id:
        raise ProtocolError("Invalid protocol state pointer")
    sequence, head = state["sequence"], state["head"]
    if type(sequence) is not int or sequence < 0 or (sequence == 0) != (head is None):
        raise ProtocolError("Invalid event ledger length/head")
    directory = root / "events"
    if directory.is_symlink() or not directory.is_dir():
        raise ProtocolError("Protocol event directory is missing or a symlink")
    files = {}
    for path in directory.iterdir():
        if path.is_symlink() or not path.is_file() or not re.fullmatch(r"[0-9a-f]{64}\.json", path.name):
            raise ProtocolError("Unexpected event ledger entry")
        files[path.stem] = path
    events, visited = [], set()
    while head is not None:
        if head in visited or head not in files:
            raise ProtocolError("Event ledger is cyclic or missing a committed event")
        visited.add(head)
        event = _read(files[head])
        expected = dict(event)
        event_id = expected.pop("event_id", None)
        if event_id != head or fingerprint(expected) != head or event.get("lock_id") != lock_id or event.get("schema_version") != EVENT_SCHEMA:
            raise ProtocolError("Event ledger digest/lock mismatch")
        if event.get("sequence") != sequence or event.get("event_type") != "outcome_access":
            raise ProtocolError("Event ledger ordering/type mismatch")
        _validate_event(event)
        events.append(event)
        sequence -= 1
        head = event.get("previous_event_id")
    if sequence != 0:
        raise ProtocolError("Event ledger sequence is incomplete")
    orphaned = [path for key, path in files.items() if key not in visited]
    if orphaned and not allow_orphans:
        raise ProtocolError("Orphan event detected: recover the append before any further launch or amendment")
    events.reverse()
    if any(_utc(a["recorded_at"], "event time") > _utc(b["recorded_at"], "event time") for a, b in zip(events, events[1:])):
        raise ProtocolError("Event times are nonmonotonic")
    return events, orphaned


def _validate_event(event: dict) -> None:
    _ids(event.get("subject_ids"), "exposed subject_ids")
    _utc(event.get("recorded_at"), "event recorded_at")
    text = event.get("cohort_manifest_text")
    if not isinstance(text, str) or hashlib.sha256(text.encode("utf-8")).hexdigest() != event.get("cohort_manifest_sha256"):
        raise ProtocolError("Outcome event's cohort evidence hash is invalid")
    cohort = _json_bytes(text.encode("utf-8"), "event cohort evidence")
    if sorted(_cohort_ids(cohort)) != sorted(event["subject_ids"]):
        raise ProtocolError("Outcome event subject IDs disagree with its cohort evidence")
    enrollment_text = event.get("enrollment_manifest_text")
    if not isinstance(enrollment_text, str) or hashlib.sha256(enrollment_text.encode("utf-8")).hexdigest() != event.get("enrollment_manifest_sha256"):
        raise ProtocolError("Outcome event enrollment evidence hash is invalid")
    _json_bytes(enrollment_text.encode("utf-8"), "event enrollment evidence")


def _cohort_ids(cohort: dict) -> list[str]:
    if cohort.get("schema_version") != "eeg-confirmatory-cohort-1":
        raise ProtocolError("Cohort manifest requires schema_version='eeg-confirmatory-cohort-1'")
    if not _text(cohort.get("cohort_id")):
        raise ProtocolError("Cohort ID must be explicit")
    return _ids(cohort.get("subject_ids"), "cohort subject_ids")


def validate_lock(lock_dir: str | Path, *, allow_orphan_events=False) -> dict:
    """Read a self-contained lock and verify frozen evidence and ledger links."""
    try:
        return _validate_lock(Path(lock_dir), allow_orphan_events=allow_orphan_events)
    except ProtocolError:
        raise
    except (OSError, KeyError, TypeError, ValueError, AttributeError) as exc:
        raise ProtocolError(f"Invalid protocol lock evidence: {exc}") from exc


def _validate_lock(root: Path, *, allow_orphan_events=False) -> dict:
    """Read a self-contained lock and verify all frozen evidence and ledger links."""
    if root.is_symlink() or not root.is_dir():
        raise ProtocolError("Lock root must be a real directory")
    allowed = {"protocol_lock.json", "protocol.json", "artifacts", "data_sources", "events", "STATE.json", ".matrix.lock"}
    if set(p.name for p in root.iterdir()) - allowed or any(p.is_symlink() for p in root.iterdir()):
        raise ProtocolError("Unexpected file or symlink in protocol lock")
    manifest = _read(root / "protocol_lock.json")
    identity = dict(manifest)
    lock_id = identity.pop("lock_id", None)
    if manifest.get("schema_version") != LOCK_SCHEMA or manifest.get("status") != "locked" or fingerprint(identity) != lock_id:
        raise ProtocolError("Protocol lock completion/identity digest is invalid")
    if sha256_file(root / "protocol.json") != manifest.get("protocol_sha256"):
        raise ProtocolError("Locked protocol snapshot was modified")
    spec = _read(root / "protocol.json")
    if spec.get("status") != "locked":
        raise ProtocolError("Protocol snapshot is not locked")
    validate_protocol(spec, stage="lock").raise_for_errors()
    locked_at = _utc(manifest.get("locked_at"), "lock timestamp")
    if locked_at > datetime.now(timezone.utc) or _utc(spec["approval"]["approved_at"], "approval timestamp") > locked_at:
        raise ProtocolError("Lock/approval chronology is invalid")
    registration = spec["study"]["registration"]
    if registration["status"] == "registered" and _utc(registration["timestamp"], "registration timestamp") > locked_at:
        raise ProtocolError("Registration timestamp follows this protocol lock")
    code = manifest.get("code", {})
    if not isinstance(code.get("commit"), str) or not COMMIT.fullmatch(code["commit"]) or code.get("dirty") is not False:
        raise ProtocolError("Lock lacks a clean, identifiable source-code freeze")
    if not isinstance(code.get("source_files"), dict) or fingerprint(code["source_files"]) != code.get("working_source_sha256"):
        raise ProtocolError("Lock source-code fingerprint is inconsistent")
    inventory = manifest.get("artifacts", {})
    if set(inventory) != set(ARTIFACT_ROLES):
        raise ProtocolError("Frozen artifact inventory is incomplete")
    contents, actual_names = {}, set()
    for role, item in inventory.items():
        relative = item.get("file")
        path = Path(relative) if isinstance(relative, str) else Path("invalid")
        if len(path.parts) != 2 or path.parts[0] != "artifacts" or path.name != role + ".bin":
            raise ProtocolError("Unsafe artifact inventory path")
        artifact = root / path
        if artifact.is_symlink() or not artifact.is_file():
            raise ProtocolError("Missing artifact or artifact symlink")
        data = artifact.read_bytes()
        if hashlib.sha256(data).hexdigest() != item.get("sha256") or len(data) != item.get("bytes"):
            raise ProtocolError(f"Frozen artifact was modified: {role}")
        if spec["freeze"]["artifacts"][role] != {"path": relative, "sha256": item["sha256"]}:
            raise ProtocolError("Protocol and artifact inventory disagree")
        contents[role] = data
        actual_names.add(path.name)
    if set(p.name for p in (root / "artifacts").iterdir()) != actual_names:
        raise ProtocolError("Unexpected frozen artifact")
    dataset_inventory = manifest.get("data_sources", {})
    if set(dataset_inventory) != set(_dataset_entries(contents["development_data_manifest"])):
        raise ProtocolError("Frozen development source inventory is incomplete")
    dataset_contents = {}
    directory = root / "data_sources"
    if directory.is_symlink() or not directory.is_dir():
        raise ProtocolError("Missing development source directory")
    for name, item in dataset_inventory.items():
        relative = f"data_sources/{name}.bin"
        if item.get("file") != relative:
            raise ProtocolError("Unsafe development source path")
        source = root / relative
        if source.is_symlink() or not source.is_file():
            raise ProtocolError("Missing development source or source symlink")
        data = source.read_bytes()
        if len(data) != item.get("bytes") or hashlib.sha256(data).hexdigest() != item.get("sha256"):
            raise ProtocolError("Frozen development source was modified")
        dataset_contents[name] = data
    if set(p.name for p in directory.iterdir()) != {name + ".bin" for name in dataset_inventory}:
        raise ProtocolError("Unexpected development source file")
    evidence = _artifact_semantics(spec, contents, dataset_contents)
    acquisition_evidence = _json_bytes(contents["acquisition_evidence"], "acquisition evidence")
    if _utc(acquisition_evidence["verified_at"], "acquisition verification") > locked_at:
        raise ProtocolError("Acquisition verification follows the protocol lock")
    if evidence["development_subject_ids"] != manifest.get("development_subject_ids"):
        raise ProtocolError("Lock development boundary differs from evidence")
    events, orphaned = _event_chain(root, lock_id, allow_orphans=allow_orphan_events)
    lineage = manifest.get("lineage", {})
    prior_exposed = _ids(lineage.get("exposed_subject_ids", []), "lineage exposed IDs", allow_empty=True)
    prior_development = _ids(lineage.get("historical_development_subject_ids", []), "lineage development IDs", allow_empty=True)
    if lineage.get("parent_lock_id") is not None:
        for key in ("parent_lock_id", "parent_manifest_sha256", "parent_state_sha256"):
            if not isinstance(lineage.get(key), str) or not SHA.fullmatch(lineage[key]):
                raise ProtocolError("Amendment lineage digest is missing")
        amendment = lineage.get("amendment", {})
        if not _text(amendment.get("reason")) or amendment.get("phase") not in {"pre_outcome", "post_outcome"}:
            raise ProtocolError("Amendment audit record is incomplete")
        if _utc(amendment.get("recorded_at"), "amendment time") > locked_at:
            raise ProtocolError("Amendment chronology is inconsistent")
        if amendment.get("new_holdout") is not None and fingerprint(amendment["new_holdout"]) != amendment.get("new_holdout_sha256"):
            raise ProtocolError("Amendment holdout evidence hash is inconsistent")
    elif lineage.get("amendment") is not None or prior_exposed or prior_development:
        raise ProtocolError("An initial lock cannot invent prior outcome/development history")
    all_development = sorted(set(prior_development) | set(evidence["development_subject_ids"]))
    for event in events:
        if _utc(event["recorded_at"], "event time") < locked_at:
            raise ProtocolError("Outcome access predates the protocol lock")
        cohort = _json_bytes(event["cohort_manifest_text"].encode("utf-8"), "outcome cohort")
        _check_cohort(spec, cohort, all_development, prior_exposed)
        enrollment = _json_bytes(event["enrollment_manifest_text"].encode("utf-8"), "outcome enrollment")
        audit = _assess_outcome_evidence(enrollment, spec, cohort, locked_at, event["recorded_at"])
        if any(type(event.get(key)) is not int or event[key] != audit[key] for key in ("n_evaluable", "n_recruited")):
            raise ProtocolError("Outcome event enrollment counts disagree with the frozen evidence")
    if len(events) > 1:
        raise ProtocolError("Outcome access closes a lock; additional cohort enrollment cannot follow")
    return {"manifest": manifest, "protocol": spec, "events": events,
            "orphan_events": [str(p) for p in orphaned],
            "exposed_subject_ids": sorted(set(prior_exposed).union(*(set(e["subject_ids"]) for e in events))),
            "development_subject_ids": all_development}


def lock_protocol(spec_path: str | Path, out_dir: str | Path, *, lineage: dict | None = None,
                  _before_publish: Callable[[], None] | None = None) -> dict:
    path = Path(spec_path)
    source_bytes = path.read_bytes()
    spec = _json_bytes(source_bytes, str(path))
    if spec.get("status") != "draft":
        raise ProtocolError("Only a draft can create a new lock; validate existing locks using their bundle")
    validate_protocol(spec, stage="lock").raise_for_errors()
    contents, sources, dataset_contents, dataset_sources = _source_artifacts(spec, path.parent)
    evidence = _artifact_semantics(spec, contents, dataset_contents)
    code = code_identity()
    if not isinstance(code.get("commit"), str) or not COMMIT.fullmatch(code["commit"]) or code.get("dirty") is not False:
        raise ProtocolError("Code freeze requires a clean checked-out commit")
    lineage = lineage or {"parent_lock_id": None, "exposed_subject_ids": [], "historical_development_subject_ids": [], "amendment": None}
    prior_exposed = _ids(lineage.get("exposed_subject_ids"), "lineage exposed IDs", allow_empty=True)
    manifest = None
    with new_output_directory(out_dir) as stage:
        (stage / "artifacts").mkdir()
        (stage / "data_sources").mkdir()
        (stage / "events").mkdir()
        frozen = json.loads(canonical_json(spec))
        frozen["status"] = "locked"
        inventory = {}
        for role, data in contents.items():
            relative = f"artifacts/{role}.bin"
            (stage / relative).write_bytes(data)
            digest = hashlib.sha256(data).hexdigest()
            frozen["freeze"]["artifacts"][role] = {"path": relative, "sha256": digest}
            inventory[role] = {"file": relative, "sha256": digest, "bytes": len(data)}
        dataset_inventory = {}
        for name, data in dataset_contents.items():
            relative = f"data_sources/{name}.bin"
            (stage / relative).write_bytes(data)
            dataset_inventory[name] = {"file": relative, "sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}
        _write(stage / "protocol.json", frozen)
        manifest = {"schema_version": LOCK_SCHEMA, "status": "locked", "locked_at": _now(),
                    "source_protocol_sha256": hashlib.sha256(source_bytes).hexdigest(),
                    "protocol_sha256": sha256_file(stage / "protocol.json"),
                    "protocol_id": frozen["protocol_id"], "protocol_version": frozen["protocol_version"],
                    "code": code, "artifacts": inventory, "data_sources": dataset_inventory,
                    "development_subject_ids": evidence["development_subject_ids"],
                    "lineage": {**lineage, "exposed_subject_ids": sorted(prior_exposed)}}
        manifest["lock_id"] = fingerprint(manifest)
        _write(stage / "protocol_lock.json", manifest)
        _write(stage / "STATE.json", {"schema_version": "eeg-protocol-state-1", "lock_id": manifest["lock_id"], "sequence": 0, "head": None})
        validate_lock(stage)
        if path.read_bytes() != source_bytes or any(sources[role].read_bytes() != data for role, data in contents.items()):
            raise ProtocolError("Source evidence changed during protocol freeze")
        if any(dataset_sources[name].read_bytes() != data for name, data in dataset_contents.items()):
            raise ProtocolError("Development source data changed during protocol freeze")
        if code_identity() != code:
            raise ProtocolError("Code changed during protocol freeze")
        if _before_publish is not None:
            _before_publish()
    return manifest


def check_launch(lock_dir: str | Path, cohort_manifest: str | Path | dict) -> dict:
    """Check the complete cumulative roster, before any access to model outcomes."""
    locked = validate_lock(lock_dir)
    if locked["events"]:
        raise ProtocolError("Outcome access has closed enrollment for this lock; reserve a new untouched holdout")
    cohort = _read(cohort_manifest) if not isinstance(cohort_manifest, dict) else cohort_manifest
    _check_cohort(locked["protocol"], cohort, locked["development_subject_ids"], locked["exposed_subject_ids"])
    ids = _cohort_ids(cohort)
    return {"status": "ready", "lock_id": locked["manifest"]["lock_id"],
            "cohort_id": cohort["cohort_id"], "n_independent_participants": len(ids),
            "subject_ids": sorted(ids), "cohort_sha256": fingerprint(cohort),
            "scope": "complete_cumulative_roster_attestation; outcome_access_also_requires_closed_enrollment"}


def _check_cohort(spec: dict, cohort: dict, development_ids: list, exposed_ids: list) -> None:
    if set(cohort) != {"schema_version", "cohort_id", "protocol_id", "subject_ids", "outcomes_accessed"}:
        raise ProtocolError("Cohort manifest must contain only schema, cohort/protocol IDs, subject_ids and outcomes_accessed; outcome scores are forbidden")
    ids = _cohort_ids(cohort)
    if cohort["cohort_id"] != spec["study"]["cohort_id"]:
        raise ProtocolError("Cohort definition differs from the locked primary study")
    if cohort.get("protocol_id") != spec["protocols"]["A"]["protocol_id"]:
        raise ProtocolError("Primary launch requires the locked A protocol ID")
    if cohort.get("outcomes_accessed") is not False:
        raise ProtocolError("A new holdout must be attested as not previously outcome-accessed")
    forbidden = set(development_ids) | set(exposed_ids)
    overlap = sorted(set(ids) & forbidden)
    if overlap:
        raise ProtocolError(f"Development/exposed subject overlap: {overlap}")
    if len(ids) > spec["recruitment"]["recruitment_cap"]:
        raise ProtocolError("Planned recruitment exceeds the locked recruitment cap")


def _assess_outcome_evidence(enrollment: dict, spec: dict, cohort: dict,
                             locked_at: datetime, recorded_at: str) -> dict:
    from .cohort_policy import assess_enrollment
    report = assess_enrollment(enrollment, spec)
    if not report.get("valid") or not report.get("stop_recruitment") or not report.get("outcome_ready"):
        raise ProtocolError(f"Outcome access requires a valid stopped, fully closed cumulative enrollment roster: {report.get('errors', [])}")
    ids = [participant["subject_id"] for participant in enrollment["participants"]]
    if sorted(_ids(ids, "enrollment subject_ids")) != sorted(_cohort_ids(cohort)):
        raise ProtocolError("Outcome cohort must equal the complete cumulative enrollment roster")
    access_time = _utc(recorded_at, "outcome access time")
    if access_time > datetime.now(timezone.utc):
        raise ProtocolError("Outcome access cannot be recorded in the future")
    for participant in enrollment["participants"]:
        if _utc(participant["recruited_at"], "recruitment time") < locked_at:
            raise ProtocolError("Recruitment must follow the prospective protocol lock")
        if _utc(participant["closed_at"], "disposition time") > access_time:
            raise ProtocolError("Outcome access must follow closure of all enrolled participants")
    return report


def record_outcome_access(lock_dir: str | Path, cohort_manifest: str | Path, *,
                          enrollment_manifest_path: str | Path) -> dict:
    root = protect_output(lock_dir)
    source = Path(cohort_manifest)
    data = source.read_bytes()
    cohort = _json_bytes(data, str(source))
    enrollment_source = Path(enrollment_manifest_path)
    enrollment_data = enrollment_source.read_bytes()
    enrollment = _json_bytes(enrollment_data, str(enrollment_source))
    with _matrix_lock(root):
        check_launch(root, cohort)
        locked = validate_lock(root)
        recorded_at = _now()
        report = _assess_outcome_evidence(enrollment, locked["protocol"], cohort,
                                          _utc(locked["manifest"]["locked_at"], "lock time"), recorded_at)
        state = _read(root / "STATE.json")
        event = {"schema_version": EVENT_SCHEMA, "event_type": "outcome_access",
                 "lock_id": locked["manifest"]["lock_id"], "sequence": state["sequence"] + 1,
                 "previous_event_id": state["head"], "recorded_at": recorded_at,
                 "subject_ids": sorted(_cohort_ids(cohort)), "cohort_manifest_text": data.decode("utf-8"),
                 "cohort_manifest_sha256": hashlib.sha256(data).hexdigest(),
                 "enrollment_manifest_text": enrollment_data.decode("utf-8"),
                 "enrollment_manifest_sha256": hashlib.sha256(enrollment_data).hexdigest(),
                 "n_evaluable": report["n_evaluable"], "n_recruited": report["n_recruited"]}
        event["event_id"] = fingerprint(event)
        event_path = root / "events" / (event["event_id"] + ".json")
        if event_path.exists():
            raise ProtocolError("Outcome event already exists")
        _atomic_json(event_path, event)
        if source.read_bytes() != data or enrollment_source.read_bytes() != enrollment_data:
            raise ProtocolError("Cohort/enrollment evidence changed while recording outcome access; ledger recovery required")
        _atomic_json(root / "STATE.json", {**state, "sequence": event["sequence"], "head": event["event_id"]})
        validate_lock(root)
        return event


def recover_event_append(lock_dir: str | Path) -> dict:
    """Conservatively commit one valid orphan append; never discard exposure."""
    root = protect_output(lock_dir)
    with _matrix_lock(root):
        locked = validate_lock(root, allow_orphan_events=True)
        orphans = locked["orphan_events"]
        if len(orphans) != 1:
            raise ProtocolError("Recovery requires exactly one valid pending event")
        event = _read(orphans[0])
        identity = dict(event)
        event_id = identity.pop("event_id", None)
        state = _read(root / "STATE.json")
        if (fingerprint(identity) != event_id or Path(orphans[0]).stem != event_id or
                event.get("lock_id") != state["lock_id"] or event.get("previous_event_id") != state["head"] or
                event.get("sequence") != state["sequence"] + 1 or event.get("event_type") != "outcome_access"):
            raise ProtocolError("Orphan event does not extend the committed ledger")
        _validate_event(event)
        cohort = _json_bytes(event["cohort_manifest_text"].encode("utf-8"), "orphan cohort")
        _check_cohort(locked["protocol"], cohort, locked["development_subject_ids"], locked["exposed_subject_ids"])
        enrollment = _json_bytes(event["enrollment_manifest_text"].encode("utf-8"), "orphan enrollment")
        _assess_outcome_evidence(enrollment, locked["protocol"], cohort,
                                 _utc(locked["manifest"]["locked_at"], "lock time"), event["recorded_at"])
        if state["sequence"]:
            raise ProtocolError("Outcome access has already closed this lock")
        _atomic_json(root / "STATE.json", {**state, "sequence": event["sequence"], "head": event_id})
        validate_lock(root)
        return event


def amend_protocol(lock_dir: str | Path, new_spec_path: str | Path, out_dir: str | Path, *,
                   reason: str, holdout_manifest_path: str | Path | None = None) -> dict:
    if not _text(reason):
        raise ProtocolError("An amendment requires a specific reason")
    root = protect_output(lock_dir)
    with _matrix_lock(root):
        parent = validate_lock(root)
        before = {"manifest": sha256_file(root / "protocol_lock.json"), "state": sha256_file(root / "STATE.json")}
        new_spec = _read(new_spec_path)
        old_spec = parent["protocol"]
        if new_spec.get("protocol_id") != old_spec["protocol_id"] or new_spec.get("protocol_version") == old_spec["protocol_version"]:
            raise ProtocolError("Amendment must retain study protocol_id and advance protocol_version")
        changed = [key for key in SENSITIVE_SECTIONS if new_spec.get(key) != old_spec.get(key)]
        old_artifacts = old_spec["freeze"]["artifacts"]
        new_artifacts = _at(new_spec, "freeze.artifacts", {})
        artifact_changes = [key for key in ARTIFACT_ROLES if _at(new_artifacts, key + ".sha256") != old_artifacts[key]["sha256"]]
        changed += ["artifact:" + key for key in artifact_changes]
        exposed = parent["exposed_subject_ids"]
        if exposed and new_spec.get("study", {}).get("mode") != old_spec["study"]["mode"]:
            raise ProtocolError("A pilot/estimation downgrade cannot be used after outcome access")
        holdout = None
        expanded_development = any(key in artifact_changes for key in ("development_manifest", "development_data_manifest"))
        if exposed or expanded_development:
            if holdout_manifest_path is None:
                raise ProtocolError("Post-outcome or expanded-development amendment requires a new untouched holdout manifest")
            holdout = _read(holdout_manifest_path)
            ids = _cohort_ids(holdout)
            new_development = _read(Path(new_spec_path).parent / new_spec["freeze"]["artifacts"]["development_manifest"]["path"])
            development_ids = _ids(new_development.get("subject_ids"), "new development subject IDs")
            forbidden = set(parent["development_subject_ids"]) | set(exposed) | set(development_ids)
            _check_cohort(new_spec, holdout, sorted(forbidden), [])
            if holdout["cohort_id"] == old_spec["study"]["cohort_id"] or holdout["cohort_id"] != new_spec["study"]["cohort_id"]:
                raise ProtocolError("Post-outcome amendment requires a distinct new cohort definition")
            if holdout.get("protocol_id") != new_spec["protocols"]["A"]["protocol_id"]:
                raise ProtocolError("New holdout protocol ID differs from amended A")
        lineage = {"parent_lock_id": parent["manifest"]["lock_id"], "parent_manifest_sha256": before["manifest"],
                   "parent_state_sha256": before["state"], "exposed_subject_ids": exposed,
                   "historical_development_subject_ids": parent["development_subject_ids"],
                   "amendment": {"reason": reason, "changed_sections": changed, "recorded_at": _now(),
                                 "phase": "post_outcome" if exposed else "pre_outcome",
                                 "new_holdout_sha256": fingerprint(holdout) if holdout else None,
                                 "new_holdout": holdout}}
        def verify_parent():
            if sha256_file(root / "protocol_lock.json") != before["manifest"] or sha256_file(root / "STATE.json") != before["state"]:
                raise ProtocolError("Parent state changed during amendment")
            validate_lock(root)
            if holdout_manifest_path is not None and holdout is not None and _read(holdout_manifest_path) != holdout:
                raise ProtocolError("New holdout evidence changed during amendment")
        result = lock_protocol(new_spec_path, out_dir, lineage=lineage, _before_publish=verify_parent)
        return result
