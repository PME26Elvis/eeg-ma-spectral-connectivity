"""Participant inference and immutable reports for already scored fixed models.

The inputs are the complete validated #11 archives, not arbitrary score arrays
or a newly fitted estimator.  Reports carry their source archives so validation
does not depend on a mutable working directory, a live protocol ledger, or the
availability of the original source paths.
"""
from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd

from .artifacts import new_output_directory
from .cohort_policy import interpret_endpoints
from .external import TITLE, validate_external_score, validate_predictions
from .precision import participant_bootstrap
from .protocol import validate_lock
from .provenance import canonical_json, code_identity, fingerprint, read_csv_literal
from .result_contract import (PREDICTION_COLUMNS, RESULT_GROUP_COLUMNS,
                              paired_participant_deltas, participant_results,
                              validate_predictions as validate_result_predictions)


SCHEMA = "eeg-external-report-1"
MANIFEST = "report_manifest.json"
METRICS = ("balanced_accuracy", "accuracy", "recall_0", "recall_1")
ROLES = ("primary", "reference")
SOURCE_FOLDERS = {"score": "sources/score", "predictions": "sources/predictions", "protocol": "sources/protocol"}
TABLE_FILES = ("predictions.csv", "eligible_epochs.csv", "participant_results.csv",
               "participant_flow.csv", "session_flow.csv", "figure_data.csv")
REPORT_FILES = {*TABLE_FILES, "summary.json", "figure_metadata.json", "report.md"}
SHA = re.compile(r"[0-9a-f]{64}\Z")
INFERENCE_SCOPE = "sampling_of_independent_QA_evaluable_participants_conditional_on_fixed_development_models"
SCOPE_TEXT = (
    "The confidence intervals describe sampling of independent QA-evaluable participants, "
    "conditional on the already fixed development models. They do not include model fitting, "
    "feature selection, retraining, or learning-procedure uncertainty. This is correct-only, "
    "offline Protocol A discrimination; it does not establish a causal mental-arithmetic "
    "effect, general workload measurement, online BCI usability, or transfer to Protocol B."
)


class ExternalReportError(ValueError):
    """An external report lacks complete, numerically consistent evidence."""


def _bytes(path):
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise ExternalReportError(f"Ordinary report evidence file required: {path.name}")
    return path.read_bytes()


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def _json(path):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ExternalReportError(f"Duplicate report JSON field: {key}")
            result[key] = value
        return result
    try:
        value = json.loads(_bytes(path).decode("utf-8"), object_pairs_hook=pairs,
                           parse_constant=lambda v: (_ for _ in ()).throw(ValueError(v)))
    except (UnicodeError, ValueError) as exc:
        raise ExternalReportError(f"Invalid report JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise ExternalReportError("Report evidence must be a JSON object")
    return value


def _write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _inventory(root, *, ignore=()):
    root = Path(root)
    if root.is_symlink() or not root.is_dir():
        raise ExternalReportError("Report/source root must be an ordinary directory")
    result = {}
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ExternalReportError("Reports and their source archives cannot contain symlinks")
        if path.is_file():
            name = path.relative_to(root).as_posix()
            if name not in ignore:
                data = _bytes(path)
                result[name] = {"sha256": _sha(data), "bytes": len(data)}
        elif not path.is_dir():
            raise ExternalReportError("Report archives permit only ordinary files and directories")
    return result


def _source_inventories(roots):
    return {key: _inventory(root, ignore=(".matrix.lock",) if key == "protocol" else ())
            for key, root in roots.items()}


def _copy_sources(roots, stage, inventories):
    for key, source in roots.items():
        destination = stage / SOURCE_FOLDERS[key]
        destination.mkdir(parents=True)
        for name, descriptor in inventories[key].items():
            data = _bytes(Path(source) / name)
            if {"sha256": _sha(data), "bytes": len(data)} != descriptor:
                raise ExternalReportError("Source evidence changed during report publication")
            target = destination / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
        # Protocol validation expects the events/data directories even when empty.
        if key == "protocol":
            for name in ("events", "data_sources", "artifacts"):
                (destination / name).mkdir(exist_ok=True)


def _load_evidence(score_dir, prediction_dir, lock_dir):
    score = validate_external_score(score_dir, prediction_dir=prediction_dir, lock_dir=lock_dir)
    prediction = validate_predictions(prediction_dir, lock_dir=lock_dir)
    locked = validate_lock(lock_dir)
    spec = locked["protocol"]
    inference = spec["inference"]
    expected = {"method": "participant_percentile_bootstrap", "confidence": .95,
                "resamples": 10_000, "seed": 20261109, "unit": "participant",
                "paired_resampling": "identical_participant_indices_for_both_models", "tail": "two_sided"}
    if any(inference.get(key) != value for key, value in expected.items()):
        raise ExternalReportError("The report requires the locked 10,000-resample, seed-20261109, two-sided 95% participant percentile bootstrap")
    enrollment = _json(Path(score_dir) / "enrollment.json")
    cohort = _json(Path(score_dir) / "cohort.json")
    labels = read_csv_literal(Path(score_dir) / "labels.csv")
    ids = [person["subject_id"] for person in enrollment["participants"]]
    if sorted(ids) != sorted(cohort["subject_ids"]) or sorted(ids) != prediction["manifest"]["cohort_subject_ids"]:
        raise ExternalReportError("Report requires the complete, exact declared enrollment roster")
    if not score["enrollment_audit"]["outcome_ready"]:
        raise ExternalReportError("External reporting requires the prospectively closed recruitment flow")
    return {"score": score, "prediction": prediction, "locked": locked,
            "enrollment": enrollment, "cohort": cohort, "labels": labels}


def _comparisons(spec):
    a = spec["protocols"]["A"]
    return ((a["primary_comparison"], "primary", "Rest1", "Type1"),
            (a["secondary_comparison"], "secondary", "Rest2", "Type2"))


def _exclusion_reasons(person, audit, spec, endpoint, rest, task):
    if audit[endpoint + "_evaluable"]:
        return []
    reasons = []
    if person["disposition"] != "completed":
        reasons.append("DISPOSITION_" + person["disposition"].upper())
    selected = next((s for s in person["sessions"] if s["session_id"] == audit["selected_session_id"]), None)
    if not person["sessions"]:
        reasons.append("NO_RECORDED_SESSION")
    elif selected is None:
        reasons.append("NO_QA_VALID_SESSION")
    if selected is not None:
        rules = spec["eligibility"]
        if (selected["rest_eligible"][rest] < rules["planned_rest_epochs_per_block"]
                or selected["rest_seconds"][rest] < rules["planned_rest_seconds_per_block"]):
            reasons.append("INCOMPLETE_REST_BLOCK")
        if selected["eligible_correct"][task] < rules["min_epochs_per_class"]:
            reasons.append("INSUFFICIENT_CORRECT_TASK_EPOCHS")
        if selected["rest_eligible"][rest] == 0:
            reasons.append("MISSING_CLASS_0")
        if selected["eligible_correct"][task] == 0:
            reasons.append("MISSING_CLASS_1")
    if not reasons:
        raise ExternalReportError("An unevaluable participant has no prespecified exclusion reason")
    return reasons


def _flow(evidence):
    spec, score = evidence["locked"]["protocol"], evidence["score"]
    audits = {p["subject_id"]: p for p in score["enrollment_audit"]["participants"]}
    participant_rows, session_rows = [], []
    for person in sorted(evidence["enrollment"]["participants"], key=lambda p: p["subject_id"]):
        audit = audits[person["subject_id"]]
        row = {"cohort_id": evidence["cohort"]["cohort_id"], "protocol_id": spec["protocols"]["A"]["protocol_id"],
               "subject_id": person["subject_id"], "recruited_at": person["recruited_at"],
               "population_eligible": person["population_eligible"], "disposition": person["disposition"],
               "closed_at": person["closed_at"], "n_sessions": len(person["sessions"]),
               "selected_session_id": audit["selected_session_id"] or "", "technical_failures": audit["technical_failures"],
               "primary_evaluable": audit["primary_evaluable"], "secondary_evaluable": audit["secondary_evaluable"]}
        for _, endpoint, rest, task in _comparisons(spec):
            row[endpoint + "_exclusion_reasons"] = canonical_json(_exclusion_reasons(person, audit, spec, endpoint, rest, task))
        participant_rows.append(row)
        for session in sorted(person["sessions"], key=lambda s: s["attempt_number"]):
            item = {"cohort_id": row["cohort_id"], "protocol_id": session["protocol_id"], "subject_id": person["subject_id"],
                    "session_id": session["session_id"], "attempt_number": session["attempt_number"],
                    "selected_for_participant": session["session_id"] == audit["selected_session_id"],
                    "started_at": session["started_at"], "duration_seconds": session["duration_seconds"],
                    "qa_passed": session["qa"]["passed"], "qa_reason_codes": canonical_json(session["qa"]["reason_codes"]),
                    "qa_evidence_sha256": session["qa"]["evidence_sha256"]}
            for category in ("attempts", "correct", "eligible_correct", "rest_acquired", "rest_eligible", "rest_seconds", "task_end"):
                item.update({category + "_" + key: value for key, value in session[category].items()})
            session_rows.append(item)
    participants = pd.DataFrame(participant_rows)
    sessions = pd.DataFrame(session_rows)
    # At least one predicted recording exists in a valid external archive.
    if sessions.empty:
        raise ExternalReportError("Scored external predictions have no recorded source sessions")
    audit = score["enrollment_audit"]
    counts = {"n_recruited": len(participants), "n_unique_participants": participants.subject_id.nunique(),
              "n_recorded_sessions": len(sessions), "n_replacement_sessions": int((sessions.attempt_number > 1).sum()),
              "n_qa_passed_sessions": int(sessions.qa_passed.sum()), "n_qa_failed_sessions": int((~sessions.qa_passed).sum()),
              "n_without_recorded_session": int((participants.n_sessions == 0).sum()),
              "n_without_QA_valid_session": int((participants.selected_session_id == "").sum()),
              "n_evaluable": int(participants.primary_evaluable.sum()),
              "n_secondary_evaluable": int(participants.secondary_evaluable.sum()),
              "n_primary_unevaluable": int((~participants.primary_evaluable).sum()),
              "n_secondary_unevaluable": int((~participants.secondary_evaluable).sum()),
              "target_evaluable": spec["recruitment"]["target_evaluable"], "recruitment_cap": spec["recruitment"]["recruitment_cap"],
              "stop_reason": audit["stop_reason"], "outcome_ready": audit["outcome_ready"],
              "pending_participants": audit["pending_participants"],
              "dispositions": dict(sorted(Counter(participants.disposition).items())),
              "exclusion_reason_counts_are_overlapping": True,
              "evidence_scope": audit["evidence_scope"],
              "repeated_session_independent_N_increment": 0}
    for endpoint in ("primary", "secondary"):
        counts[endpoint + "_exclusion_reason_counts"] = dict(sorted(Counter(
            reason for reasons in participants[endpoint + "_exclusion_reasons"] for reason in json.loads(reasons)).items()))
    return participants, sessions, counts


def _canonical_predictions(evidence, flow):
    prediction, spec = evidence["prediction"], evidence["locked"]["protocol"]
    selected = flow.set_index("subject_id").selected_session_id.to_dict()
    source = prediction["predictions"]
    source = source.loc[[selected[subject] == session for subject, session in zip(source.subject_id, source.session_id)]].copy()
    labels = evidence["labels"].set_index("epoch_id").true_label
    source["y_true"] = labels.reindex(source.epoch_id).to_numpy()
    if source.y_true.isna().any():
        raise ExternalReportError("Predicted epochs have missing independent labels")
    eligible_columns = ["cohort_id", "protocol_id", "comparison", "subject_id", "session_id", "epoch_id", "y_true"]
    eligible = source.loc[source.model_role == "primary", eligible_columns].sort_values(eligible_columns[:-1]).reset_index(drop=True)
    eligible_id = fingerprint({"selection": spec["protocols"]["A"]["session_selection"],
                               "protocol_lock_id": prediction["manifest"]["protocol_lock_id"],
                               "enrollment": evidence["score"]["artifacts"]["enrollment.json"]["sha256"],
                               "epochs": eligible.to_dict("records")})
    eligible["eligible_epoch_manifest_id"] = eligible_id
    source["eligible_epoch_manifest_id"] = eligible_id
    source["run_id"] = prediction["manifest"]["prediction_id"]
    source["experiment"] = source.model_role.map({role: spec["endpoints"][role + "_model"] for role in ROLES})
    source["classifier"] = "LDA"
    source["y_pred"] = source.predicted_label
    canonical = validate_result_predictions(source[list(PREDICTION_COLUMNS)], eligible_epochs=eligible)
    canonical = canonical.sort_values([*RESULT_GROUP_COLUMNS, "subject_id", "session_id", "epoch_id"]).reset_index(drop=True)
    # Pair validation includes labels, sessions, epochs and eligible-manifest identity;
    # never intersect partially overlapping participant sets.
    if not canonical.empty:
        paired_participant_deltas(canonical.loc[canonical.experiment == spec["endpoints"]["reference_model"]],
                                  canonical.loc[canonical.experiment == spec["endpoints"]["primary_model"]])
    return canonical, eligible, eligible_id


def _participant_table(evidence, flow, canonical, eligible_id):
    spec, pred = evidence["locked"]["protocol"], evidence["prediction"]["manifest"]
    observed = participant_results(canonical) if not canonical.empty else pd.DataFrame()
    lookup = {(r["subject_id"], r["comparison"], r["experiment"]): r for r in observed.to_dict("records")}
    rows = []
    for person in flow.to_dict("records"):
        for comparison, endpoint, _, _ in _comparisons(spec):
            for role in ROLES:
                experiment = spec["endpoints"][role + "_model"]
                original = lookup.get((person["subject_id"], comparison, experiment))
                evaluable = person[endpoint + "_evaluable"]
                row = {"analysis_id": pred["analysis_id"], "run_id": pred["prediction_id"],
                       "cohort_id": person["cohort_id"], "protocol_id": person["protocol_id"],
                       "experiment": experiment, "classifier": "LDA", "comparison": comparison,
                       "model_id": pred["model_ids"][role], "model_role": role, "subject_id": person["subject_id"],
                       "session_id": person["selected_session_id"], "eligible_epoch_manifest_id": eligible_id,
                       "endpoint": endpoint, "evaluable": evaluable,
                       "exclusion_reasons": person[endpoint + "_exclusion_reasons"],
                       "n_recorded_sessions": person["n_sessions"], "n_selected_sessions": int(bool(person["selected_session_id"]))}
                for column in ("n_epochs", "n_class_0", "n_class_1", "tn", "fp", "fn", "tp"):
                    row[column] = int(original[column]) if original else 0
                row["recall_0"] = row["tn"] / row["n_class_0"] if row["n_class_0"] else np.nan
                row["recall_1"] = row["tp"] / row["n_class_1"] if row["n_class_1"] else np.nan
                if evaluable and (not original or original["ba_status"] != "eligible"):
                    raise ExternalReportError("A declared evaluable participant lacks both classes")
                row["observed_accuracy"] = (row["tn"] + row["tp"]) / row["n_epochs"] if row["n_epochs"] else np.nan
                row["accuracy"] = row["observed_accuracy"] if evaluable else np.nan
                row["balanced_accuracy"] = .5 * (row["recall_0"] + row["recall_1"]) if evaluable else np.nan
                row["ba_status"] = ("eligible" if evaluable else "missing_both_classes" if not row["n_epochs"] else
                                    "missing_class_0" if not row["n_class_0"] else "missing_class_1" if not row["n_class_1"] else "endpoint_ineligible")
                rows.append(row)
    table = pd.DataFrame(rows).sort_values(["comparison", "model_role", "subject_id"]).reset_index(drop=True)
    if table.duplicated(["comparison", "model_role", "subject_id"]).any() or len(table) != len(flow) * 4:
        raise ExternalReportError("Participant report does not preserve the complete endpoint/model/roster matrix")
    return table


def _no_interval(values, *, status):
    return {"estimate": float(np.mean(values)) if len(values) else None, "ci_lower": None, "ci_upper": None,
            "half_width": None, "degenerate": bool(len(values) == 1) if len(values) else None,
            "confirmatory_eligible": False, "status": status}


def _bootstrap_metrics(primary, reference, *, spec, subject_ids):
    """Use identical participant ordering and RNG draws for every paired metric."""
    ids = list(subject_ids)
    if len(ids) != len(set(ids)) or len(primary) != len(ids) or len(reference) != len(ids):
        raise ExternalReportError("Bootstrap requires exact one-to-one participant pairing")
    kwargs = {"seed": spec["inference"]["seed"], "n_resamples": spec["inference"]["resamples"],
              "confidence_level": spec["inference"]["confidence"], "subject_ids": ids}
    endpoints = {role: {} for role in ROLES}
    paired = {}
    for metric in METRICS:
        a, b = primary[metric].to_numpy(float), reference[metric].to_numpy(float)
        if not np.isfinite(a).all() or not np.isfinite(b).all():
            raise ExternalReportError("Evaluable participant scores must be finite; missing values cannot be dropped")
        if np.any((a < 0) | (a > 1)) or np.any((b < 0) | (b > 1)):
            raise ExternalReportError("Participant rates must lie in [0,1]")
        if len(ids) < 2:
            status = "insufficient_participants" if ids else "not_estimable"
            endpoints["primary"][metric] = _no_interval(a, status=status)
            endpoints["reference"][metric] = _no_interval(b, status=status)
            if metric in ("balanced_accuracy", "accuracy"):
                paired[metric] = _no_interval(a - b, status=status)
            continue
        boot = participant_bootstrap(a, b, **kwargs)
        other = participant_bootstrap(b, **kwargs)
        for role, result in (("primary", boot["primary"]), ("reference", other["primary"])):
            degenerate = result["degenerate"] or result["ci_upper"] - result["ci_lower"] <= 1e-12
            endpoints[role][metric] = {**result, "degenerate": bool(degenerate), "confirmatory_eligible": False,
                                       "status": "degenerate" if degenerate else "estimated"}
        if metric in ("balanced_accuracy", "accuracy"):
            result = boot["paired_delta"]
            degenerate = result["degenerate"] or result["ci_upper"] - result["ci_lower"] <= 1e-12
            paired[metric] = {**result, "degenerate": bool(degenerate), "confirmatory_eligible": False,
                              "status": "degenerate" if degenerate else "estimated"}
    indices = (np.random.default_rng(kwargs["seed"]).integers(0, len(ids),
               size=(1, kwargs["n_resamples"], len(ids)), dtype=np.int32)[0] if len(ids) >= 2 else None)
    resampling = {"participant_ids_in_order": ids, "n_participants": len(ids),
                  "indices_shape": list(indices.shape) if indices is not None else None,
                  "indices_dtype": "little_endian_int32" if indices is not None else None,
                  "indices_sha256": _sha(indices.astype("<i4", copy=False).tobytes()) if indices is not None else None,
                  "generator": "numpy.random.default_rng_PCG64_integers_int32",
                  "shared_across_models_and_metrics": True}
    return endpoints, paired, resampling


def _scope(spec, n):
    return ("descriptive_pilot_only" if spec["study"]["mode"] == "pilot_estimation" else
            "descriptive_target_shortfall" if n < spec["recruitment"]["target_evaluable"] else "confirmatory_primary")


def _interpret(primary, delta, *, n, spec):
    if primary["ci_lower"] is not None:
        result = interpret_endpoints([primary["ci_lower"], primary["ci_upper"]],
            None if delta["ci_lower"] is None else [delta["ci_lower"], delta["ci_upper"]], n_evaluable=n, spec=spec)
    else:
        result = {"above_chance": "descriptive_only", "above_chance_interval_position": "not_estimable",
                  "inference_scope": _scope(spec, n), "absolute_research_target": "not_estimable",
                  "paired_delta_research_target": "not_estimable",
                  "paired_delta_inference": "descriptive_nominal_interval_no_confirmatory_claim",
                  "primary_half_width": None, "precision_target_met": False,
                  "evaluable_target_met": n >= spec["recruitment"]["target_evaluable"],
                  "scope": "QA_evaluable_correct_only_protocol_A_not_causal_MA_or_online_BCI_usability"}
    eligible = result["inference_scope"] == "confirmatory_primary" and primary["status"] == "estimated"
    result.update(formal_primary_claim_eligible=eligible, primary_ci_status=primary["status"],
                  uncertainty_scope=INFERENCE_SCOPE,
                  model_fitting_uncertainty_included=False,
                  result_dependent_reporting_or_recruitment_extension_allowed=False)
    return result


def _inference_summary(evidence, participants, flow_counts):
    spec, pred = evidence["locked"]["protocol"], evidence["prediction"]["manifest"]
    endpoints, deltas, pooled, resampling = [], [], [], {}
    for comparison, endpoint, _, _ in _comparisons(spec):
        by_role = {role: participants.loc[(participants.comparison == comparison) &
            (participants.model_role == role) & participants.evaluable].sort_values("subject_id").reset_index(drop=True) for role in ROLES}
        primary, reference = by_role["primary"], by_role["reference"]
        if primary.subject_id.tolist() != reference.subject_id.tolist():
            raise ExternalReportError("Paired analysis requires the exact same evaluable participant set")
        if primary.session_id.tolist() != reference.session_id.tolist():
            raise ExternalReportError("Paired analysis requires the exact same selected participant sessions")
        metrics, paired, indices = _bootstrap_metrics(primary, reference, spec=spec, subject_ids=primary.subject_id.tolist())
        resampling[comparison] = indices
        scope = _scope(spec, len(primary)) if endpoint == "primary" else "descriptive_secondary_nominal_intervals"
        for role in ROLES:
            if endpoint == "primary" and role == "primary" and scope == "confirmatory_primary":
                metrics[role]["balanced_accuracy"]["confirmatory_eligible"] = metrics[role]["balanced_accuracy"]["status"] == "estimated"
            endpoints.append({"comparison": comparison, "endpoint": endpoint, "model_role": role,
                "model_id": pred["model_ids"][role], "experiment": spec["endpoints"][role + "_model"],
                "n_recruited": flow_counts["n_recruited"], "n_evaluable": len(by_role[role]),
                "n_unevaluable": flow_counts["n_recruited"] - len(by_role[role]),
                "participant_weighting": "equal_one_per_independent_participant",
                "inference_scope": scope if role == "primary" else "descriptive_reference_nominal_intervals",
                "metrics": metrics[role]})
            table = by_role[role]
            counts = {key: int(table[key].sum()) for key in ("tn", "fp", "fn", "tp", "n_epochs", "n_class_0", "n_class_1")}
            recall0 = counts["tn"] / counts["n_class_0"] if counts["n_class_0"] else None
            recall1 = counts["tp"] / counts["n_class_1"] if counts["n_class_1"] else None
            pooled.append({"comparison": comparison, "model_role": role, "n_participants": len(table), **counts,
                "accuracy": (counts["tn"] + counts["tp"]) / counts["n_epochs"] if counts["n_epochs"] else None,
                "recall_0": recall0, "recall_1": recall1,
                "balanced_accuracy": .5 * (recall0 + recall1) if recall0 is not None and recall1 is not None else None,
                "scope": "descriptive_epoch_pooled_among_endpoint_evaluable_participants",
                "is_primary_estimand": False, "confidence_interval": None})
        deltas.append({"comparison": comparison, "n_pairs": len(primary), "direction": "primary_minus_reference",
                       "balanced_accuracy": paired["balanced_accuracy"], "accuracy": paired["accuracy"],
                       "inference": "descriptive_nominal_interval_no_confirmatory_claim"})
    main = next(row for row in endpoints if row["endpoint"] == "primary" and row["model_role"] == "primary")
    main_delta = next(row for row in deltas if row["comparison"] == main["comparison"])
    interpretation = _interpret(main["metrics"]["balanced_accuracy"], main_delta["balanced_accuracy"], n=main["n_evaluable"], spec=spec)
    return {"schema_version": "eeg-external-report-summary-1", "report_title": TITLE,
            "analysis_id": pred["analysis_id"], "run_id": pred["prediction_id"], "score_id": evidence["score"]["score_id"],
            "cohort_id": evidence["cohort"]["cohort_id"], "protocol_id": spec["protocols"]["A"]["protocol_id"],
            "protocol_lock_id": pred["protocol_lock_id"], "protocol_sha256": pred["protocol_sha256"],
            "model_ids": pred["model_ids"], "model_sha256": pred["model_sha256"],
            "cohort_flow": flow_counts, "endpoints": endpoints, "paired_deltas": deltas,
            "pooled_epoch_metrics": pooled, "bootstrap_resampling": resampling,
            "interpretation": interpretation, "inference": {**spec["inference"],
                "quantile_method": "linear", "uncertainty_scope": INFERENCE_SCOPE,
                "model_fitting_uncertainty_included": False},
            "scope_text": SCOPE_TEXT,
            "publication_policy": "complete_prespecified_fixed_model_A_endpoint_model_and_roster_matrix_regardless_of_sign_or_threshold",
            "separate_analysis_families": {
                "expanded_nested_LOPO": "requires_separate_descriptive_retrained_learning_procedure_report",
                "Protocol_B": "inactive_next_study_not_part_of_this_A_protocol_report"},
            "missing_endpoint_policy": "retain_in_flow_no_chance_imputation_no_silent_drop",
            "registration": spec["study"]["registration"]}


def _figure_data(summary):
    rows = []
    common = {key: summary[key] for key in ("analysis_id", "run_id", "score_id", "cohort_id", "protocol_id", "protocol_lock_id", "protocol_sha256")}
    for endpoint in summary["endpoints"]:
        for metric, result in endpoint["metrics"].items():
            rows.append({**common, "comparison": endpoint["comparison"], "model_role": endpoint["model_role"],
                "model_id": endpoint["model_id"], "metric": metric, "n_participants": endpoint["n_evaluable"],
                "estimate": result["estimate"], "ci_lower": result["ci_lower"], "ci_upper": result["ci_upper"],
                "ci_method": "participant_percentile_bootstrap", "confidence": summary["inference"]["confidence"],
                "resamples": summary["inference"]["resamples"], "seed": summary["inference"]["seed"],
                "ci_status": result["status"], "inference_scope": endpoint["inference_scope"]})
    return pd.DataFrame(rows)


def _number(value):
    return "not estimable" if value is None else f"{value:.4f}"


def _interval(metric):
    if metric["ci_lower"] is None:
        return metric["status"]
    return f"[{metric['ci_lower']:.4f}, {metric['ci_upper']:.4f}]" + (" (degenerate)" if metric["degenerate"] else "")


def _markdown(summary):
    flow, interpretation = summary["cohort_flow"], summary["interpretation"]
    lines = [f"# {TITLE}", "", f"**Inference scope: {interpretation['inference_scope']}.**", "", SCOPE_TEXT, "",
             "This report covers the complete fixed-model Protocol A endpoint family. Expanded nested LOPO describes a "
             "separately retrained learning procedure and requires its own descriptive report. Protocol B remains an inactive next study.", "",
             "## Complete recruitment and QA flow", "",
             f"{flow['n_recruited']} unique participants were recruited; {flow['n_evaluable']} are primary-endpoint evaluable "
             f"and {flow['n_primary_unevaluable']} are unevaluable. The secondary endpoint has {flow['n_secondary_evaluable']} evaluable participants. "
             f"The target was {flow['target_evaluable']} evaluable participants and the unique-participant cap was {flow['recruitment_cap']}. "
             f"Recruitment stopped for `{flow['stop_reason']}`; pending participants: {flow['pending_participants']}.", "",
             f"There were {flow['n_recorded_sessions']} recorded sessions, including {flow['n_replacement_sessions']} technical replacements, "
             f"{flow['n_qa_failed_sessions']} QA-failed sessions, and {flow['n_without_recorded_session']} participants without a recorded session. "
             "Only the prospectively selected first QA-valid A session contributes to each participant. Repeated sessions add zero independent participants.", "",
             "The full roster, dispositions, endpoint exclusion reasons, and session QA evidence hashes are retained in "
             "`participant_flow.csv` and `session_flow.csv`. Exclusion reasons can overlap. "
             f"QA evidence scope: `{flow['evidence_scope']}`.", "",
             "## Equal-weight participant results", "",
             "Each participant contributes one score. Balanced accuracy is half the sum of that participant's rest and task recalls. "
             "A missing class makes balanced accuracy unevaluable; no chance score is imputed. "
             "Endpoint-ineligible participants remain in the participant table and do not enter endpoint means or resampling.", "",
             "| Comparison | Model | Evaluable N | Mean BA | 95% participant CI | Mean accuracy | 95% participant CI |", "|---|---|---:|---:|---|---:|---|"]
    for row in summary["endpoints"]:
        ba, acc = row["metrics"]["balanced_accuracy"], row["metrics"]["accuracy"]
        lines.append(f"| {row['comparison']} | {row['model_role']} | {row['n_evaluable']} | {_number(ba['estimate'])} | {_interval(ba)} | {_number(acc['estimate'])} | {_interval(acc)} |")
    lines += ["", "Participant-level rest/task recalls and confusion counts are in `participant_results.csv`; "
              "their equal-weight means and intervals are in `summary.json`. Available-class recalls and observed accuracy "
              "for endpoint-ineligible participants are retained descriptively and never silently included in endpoint inference.", "",
              "## Paired model differences", "", "Differences are primary minus reference on exactly the same participants, sessions, labels, and eligible epochs. "
              "Every paired interval is descriptive and nominal; none is a confirmatory secondary claim.", "",
              "| Comparison | Paired N | Mean BA difference | 95% paired participant CI | Mean accuracy difference | 95% paired participant CI |",
              "|---|---:|---:|---|---:|---|"]
    for row in summary["paired_deltas"]:
        lines.append(f"| {row['comparison']} | {row['n_pairs']} | {_number(row['balanced_accuracy']['estimate'])} | {_interval(row['balanced_accuracy'])} | {_number(row['accuracy']['estimate'])} | {_interval(row['accuracy'])} |")
    lines += ["", "## Prespecified interpretation", "",
              f"Primary above-chance conclusion: `{interpretation['above_chance']}`. "
              f"Interval position relative to chance: `{interpretation['above_chance_interval_position']}`. "
              f"Absolute research target: `{interpretation['absolute_research_target']}`. "
              f"Paired-delta research target: `{interpretation['paired_delta_research_target']}`.", "",
              f"Evaluable-N target met: `{str(interpretation['evaluable_target_met']).lower()}`. "
              f"Precision target met: `{str(interpretation['precision_target_met']).lower()}`. "
              f"Formal primary inference eligible: `{str(interpretation['formal_primary_claim_eligible']).lower()}`. "
              "Pilot mode, target shortfall, insufficient participants, and degenerate intervals cannot be promoted by a favorable point estimate. "
              "Degenerate intervals are reported explicitly and do not establish precision adequacy. Recruitment cannot be extended after seeing scores.", "",
              "## Separately pooled epoch descriptions", "",
              "These describe all epochs among endpoint-evaluable participants, so participants with more epochs receive more weight. "
              "They are separate descriptive quantities and are not the primary participant estimand. No epoch-i.i.d. confidence interval is calculated.", "",
              "| Comparison | Model | Participants | Epochs | Pooled BA | Pooled accuracy | TN | FP | FN | TP |",
              "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for row in summary["pooled_epoch_metrics"]:
        lines.append(f"| {row['comparison']} | {row['model_role']} | {row['n_participants']} | {row['n_epochs']} | {_number(row['balanced_accuracy'])} | {_number(row['accuracy'])} | {row['tn']} | {row['fp']} | {row['fn']} | {row['tp']} |")
    inference = summary["inference"]
    lines += ["", "## Reproducibility and source binding", "",
              f"Intervals use {inference['resamples']:,} participant resamples, seed {inference['seed']}, "
              f"two-sided {100 * inference['confidence']:.0f}% percentile limits, and linear quantiles. "
              "Identical participant index draws are shared across both models and metrics within each comparison. "
              "The ordered participant IDs and reproducible resampling-index hashes are recorded in `summary.json`.", "",
              "The bundle includes the original validated score, prediction, model, and protocol-lock archives, including the exact "
              "authorized outcome-access event. `report_manifest.json` inventories every file. Validation independently recomputes "
              "predictions, score arithmetic, participant tables, bootstrap limits, and this narrative from those sources.", "",
              "`figure_data.csv` and `figure_metadata.json` provide numerically verified plotting data with metric, participant N, "
              "CI method, cohort, protocol, source IDs, and source-file hashes. All prespecified fixed-model Protocol A endpoints are reported regardless of signs or threshold results.", "",
              f"- Analysis ID: `{summary['analysis_id']}`", f"- Prediction/run ID: `{summary['run_id']}`",
              f"- Score ID: `{summary['score_id']}`", f"- Cohort ID: `{summary['cohort_id']}`",
              f"- Protocol ID: `{summary['protocol_id']}`", f"- Protocol lock ID: `{summary['protocol_lock_id']}`",
              f"- Protocol SHA-256: `{summary['protocol_sha256']}`",
              f"- Primary model ID: `{summary['model_ids']['primary']}`", f"- Reference model ID: `{summary['model_ids']['reference']}`", ""]
    return "\n".join(lines)


def _assemble(evidence):
    flow, sessions, flow_counts = _flow(evidence)
    canonical, eligible, eligible_id = _canonical_predictions(evidence, flow)
    participants = _participant_table(evidence, flow, canonical, eligible_id)
    summary = _inference_summary(evidence, participants, flow_counts)
    tables = {"predictions.csv": canonical, "eligible_epochs.csv": eligible,
              "participant_results.csv": participants, "participant_flow.csv": flow,
              "session_flow.csv": sessions, "figure_data.csv": _figure_data(summary)}
    return tables, summary


def _binding(evidence, summary):
    pred = evidence["prediction"]["manifest"]
    return {"analysis_id": pred["analysis_id"], "report_title": TITLE, "score_id": evidence["score"]["score_id"],
            "prediction_id": pred["prediction_id"], "run_id": pred["prediction_id"],
            "protocol_lock_id": pred["protocol_lock_id"], "protocol_sha256": pred["protocol_sha256"],
            "cohort_id": evidence["cohort"]["cohort_id"], "model_ids": pred["model_ids"],
            "model_sha256": pred["model_sha256"], "outcome_event_id": evidence["score"]["outcome_event_id"],
            "inference_scope": summary["interpretation"]["inference_scope"], "inference": summary["inference"],
            "publication_policy": summary["publication_policy"]}


def _figure_metadata(root, summary):
    return {"schema_version": "eeg-external-figure-data-1", "title": TITLE,
            "data_file": "figure_data.csv", "data_sha256": _sha(_bytes(root / "figure_data.csv")),
            "source_participant_file": "participant_results.csv",
            "source_participant_sha256": _sha(_bytes(root / "participant_results.csv")),
            "metric_names": list(METRICS), "sample_size_unit": "independent_participant",
            "inference": summary["inference"], "cohort_id": summary["cohort_id"],
            "protocol_id": summary["protocol_id"], "protocol_sha256": summary["protocol_sha256"],
            "prediction_id": summary["run_id"], "score_id": summary["score_id"],
            "caption": "Equal-weight participant means and two-sided 95% participant percentile intervals. "
                       "N counts independent endpoint-evaluable participants; all reference and secondary intervals are descriptive. "
                       "A degenerate interval does not establish precision adequacy."}


def _check_table(path, expected):
    # Parsing every value literally preserves IDs such as 001, NA and null. The
    # recomputed schema alone decides which columns are numerical or boolean.
    data = _bytes(path).decode("utf-8-sig")
    actual = pd.read_csv(io.StringIO(data), dtype=str, keep_default_na=False)
    if list(actual.columns) != list(expected.columns) or len(actual) != len(expected):
        raise ExternalReportError(f"Report table coverage/schema differs: {Path(path).name}")
    for column in expected:
        if pd.api.types.is_bool_dtype(expected[column]):
            equal = actual[column].tolist() == expected[column].astype(str).tolist()
        elif pd.api.types.is_numeric_dtype(expected[column]):
            try:
                values = pd.to_numeric(actual[column].replace("", np.nan), errors="raise").to_numpy(float)
            except (TypeError, ValueError) as exc:
                raise ExternalReportError(f"Invalid report numeric column: {column}") from exc
            right = expected[column].to_numpy(float)
            equal = np.allclose(values, right, rtol=1e-12, atol=1e-14, equal_nan=True)
        else:
            equal = actual[column].tolist() == expected[column].fillna("").astype(str).tolist()
        if not equal:
            raise ExternalReportError(f"Report table values differ from sources: {Path(path).name}/{column}")


def build_external_report(score_dir, out_dir, *, prediction_dir, lock_dir):
    """Publish the entire report regardless of effect signs; never fit a model."""
    roots = {"score": Path(score_dir), "predictions": Path(prediction_dir), "protocol": Path(lock_dir)}
    try:
        inventories = _source_inventories(roots)
        _load_evidence(score_dir, prediction_dir, lock_dir)
        with new_output_directory(out_dir) as stage:
            _copy_sources(roots, stage, inventories)
            evidence = _load_evidence(stage / SOURCE_FOLDERS["score"], stage / SOURCE_FOLDERS["predictions"], stage / SOURCE_FOLDERS["protocol"])
            tables, summary = _assemble(evidence)
            for filename, table in tables.items():
                table.to_csv(stage / filename, index=False)
            _write_json(stage / "summary.json", summary)
            _write_json(stage / "figure_metadata.json", _figure_metadata(stage, summary))
            (stage / "report.md").write_text(_markdown(summary), encoding="utf-8")
            manifest = {"schema_version": SCHEMA, "status": "complete", "created_at": datetime.now(timezone.utc).isoformat(),
                        **_binding(evidence, summary), "reporting_code": code_identity(),
                        "source_inventory": inventories, "source_folders": SOURCE_FOLDERS,
                        "artifacts": _inventory(stage)}
            manifest["report_id"] = fingerprint(manifest)
            _write_json(stage / MANIFEST, manifest)
            validate_external_report(stage)
            if _source_inventories(roots) != inventories:
                raise ExternalReportError("Source archives changed during report publication")
        return manifest
    except ExternalReportError:
        raise
    except (ValueError, OSError, KeyError, TypeError, AttributeError) as exc:
        raise ExternalReportError(f"Invalid external report evidence: {exc}") from exc


def validate_external_report(report_dir):
    """Verify inventory and recompute every result from the bundled source bytes."""
    try:
        root = Path(report_dir)
        manifest = _json(root / MANIFEST)
        identity = dict(manifest)
        report_id = identity.pop("report_id", None)
        if manifest.get("schema_version") != SCHEMA or manifest.get("status") != "complete" or report_id != fingerprint(identity):
            raise ExternalReportError("External report completion/identity digest is invalid")
        if manifest.get("source_folders") != SOURCE_FOLDERS or manifest.get("artifacts") != _inventory(root, ignore=(MANIFEST,)):
            raise ExternalReportError("External report artifact inventory/hash mismatch")
        roots = {key: root / folder for key, folder in SOURCE_FOLDERS.items()}
        source_inventory = _source_inventories(roots)
        expected_files = REPORT_FILES | {f"{SOURCE_FOLDERS[key]}/{name}" for key, inventory in source_inventory.items() for name in inventory}
        if set(manifest["artifacts"]) != expected_files or manifest.get("source_inventory") != source_inventory:
            raise ExternalReportError("External report source inventory or complete report coverage differs")
        evidence = _load_evidence(roots["score"], roots["predictions"], roots["protocol"])
        tables, summary = _assemble(evidence)
        binding = _binding(evidence, summary)
        expected_keys = {"schema_version", "status", "created_at", "reporting_code", "source_inventory", "source_folders", "artifacts", "report_id", *binding}
        if set(manifest) != expected_keys or any(manifest.get(key) != value for key, value in binding.items()):
            raise ExternalReportError("Report source/model/protocol/inference binding differs")
        created = datetime.fromisoformat(manifest["created_at"].replace("Z", "+00:00"))
        scored = datetime.fromisoformat(evidence["score"]["created_at"].replace("Z", "+00:00"))
        if created.tzinfo is None or created.utcoffset().total_seconds() != 0 or created < scored or created > datetime.now(timezone.utc):
            raise ExternalReportError("Report creation must follow scoring and use a real UTC timestamp")
        code = manifest["reporting_code"]
        if (not isinstance(code, dict) or not isinstance(code.get("source_files"), dict) or not code["source_files"]
                or any(not isinstance(name, str) or not isinstance(digest, str) or not SHA.fullmatch(digest) for name, digest in code["source_files"].items())
                or code.get("working_source_sha256") != fingerprint(code["source_files"])):
            raise ExternalReportError("Report generation source-code identity is inconsistent")
        for name, table in tables.items():
            _check_table(root / name, table)
        if _json(root / "summary.json") != summary:
            raise ExternalReportError("Report summary/uncertainty/interpretation differs from recomputed sources")
        if _json(root / "figure_metadata.json") != _figure_metadata(root, summary):
            raise ExternalReportError("Figure metadata does not match numerical source data")
        if _bytes(root / "report.md").decode("utf-8") != _markdown(summary):
            raise ExternalReportError("Report narrative differs from recomputed results")
        return manifest
    except ExternalReportError:
        raise
    except (ValueError, OSError, KeyError, TypeError, AttributeError) as exc:
        raise ExternalReportError(f"Invalid external report evidence: {exc}") from exc
