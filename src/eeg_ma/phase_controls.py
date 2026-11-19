"""Matched alpha-only phase representations from explicit, preprocessed inputs.

The public extractor is a pure numerical building block, not a scientific-run
authorization gate.  ``prospective_data`` callers must first satisfy the
independent B protocol lock and access gate.  Synthetic fixtures are identified
as such in every exported bundle.  No legacy data discovery, filtering, feature
search, classifier fitting, or performance inference occurs in this module.

One phase tensor, epoch manifest, QC decision, nested participant split plan,
and model-selection policy identity are shared by all three representations.
An undefined ciPLV pair excludes its entire epoch from all three.  Frozen
subject partitions are never regenerated after exclusions; unevaluable folds
are retained with reasons and ``require_evaluable`` prevents their execution.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import re
import sys

import numpy as np

from .artifacts import new_output_directory
from .constants import channel_pairs
from .phase_metrics import (
    DEFAULT_CIPLV_MIN_DENOMINATOR, DENOMINATOR_EVALUATION,
    PHASE_METRICS_VERSION, phase_metrics_from_phases,
    validate_denominator_minimum,
)


POLICY_SCHEMA = "eeg-matched-phase-policy-1"
EPOCH_SCHEMA = "eeg-matched-phase-epochs-1"
SPLIT_SCHEMA = "eeg-matched-phase-splits-1"
BUNDLE_SCHEMA = "eeg-matched-phase-bundle-1"
EXPORT_SCHEMA = "eeg-matched-phase-export-1"
ANALYSIS_ID = "prospective_control_grouped_lopo"
CHANNELS = ("FP1", "FP2", "F7", "F3", "Fz", "F4", "F8")
BAND_HZ = (8.0, 13.0)
REPRESENTATIONS = ("plv", "iplv", "ciplv")
PAIR_ORDER = tuple(channel_pairs(CHANNELS))
_DIGEST_KEYS = ("code_sha256", "environment_sha256", "preprocessing_sha256",
                "selection_sha256", "epoch_manifest_sha256", "split_manifest_sha256")


class PhaseControlError(ValueError):
    """The explicitly matched input or artifact contract was violated."""


def _canonical_json(value) -> str:
    try:
        return json.dumps(value, ensure_ascii=False, sort_keys=True,
                          separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise PhaseControlError("Manifests must contain finite, literal JSON values") from exc


def canonical_sha256(value) -> str:
    """Hash literal JSON with the repository's canonical JSON convention."""
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def observed_phase_runtime() -> dict:
    """Capture the numerical runtime; archived readers retain the original observation."""
    return {"python": sys.version.split()[0], "numpy": np.__version__}


def _keys(value, expected, label: str) -> None:
    if not isinstance(value, dict) or set(value) != set(expected):
        raise PhaseControlError(f"{label} must contain exactly {', '.join(sorted(expected))}")


def _literal(value, label: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.:-]*", value):
        raise PhaseControlError(f"{label} must be a nonempty literal identifier without whitespace")
    return value


def _digest(value, label: str) -> None:
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
        raise PhaseControlError(f"{label} requires a lowercase SHA-256 digest")


def _ids(value, label: str) -> list[str]:
    if not isinstance(value, list) or not value:
        raise PhaseControlError(f"{label} requires a nonempty list of literal identifiers")
    for item in value:
        _literal(item, label)
    if len(set(value)) != len(value):
        raise PhaseControlError(f"{label} contains duplicate identifiers")
    return value


def _validate_policy(policy: dict, epochs: dict, splits: dict) -> None:
    _keys(policy, {"schema_version", "analysis_id", "evidence_kind", "protocol_lock_id",
                   *_DIGEST_KEYS, "channels", "band_hz", "ciplv_min_denominator",
                   "undefined_ciplv_policy", "min_epochs_per_class", "max_epochs_per_class"}, "input_policy")
    if policy["schema_version"] != POLICY_SCHEMA or policy["analysis_id"] != ANALYSIS_ID:
        raise PhaseControlError("Unknown matched phase policy or analysis identifier")
    if policy["evidence_kind"] not in ("synthetic_fixture", "prospective_data"):
        raise PhaseControlError("Explicit synthetic_fixture or prospective_data evidence_kind is required")
    if policy["evidence_kind"] == "prospective_data" or policy["protocol_lock_id"] is not None:
        _literal(policy["protocol_lock_id"], "protocol_lock_id")
    for key in _DIGEST_KEYS:
        _digest(policy[key], key)
    if policy["epoch_manifest_sha256"] != canonical_sha256(epochs):
        raise PhaseControlError("epoch_manifest_sha256 does not bind the supplied ordered epoch manifest")
    if policy["split_manifest_sha256"] != canonical_sha256(splits):
        raise PhaseControlError("split_manifest_sha256 does not bind the supplied frozen split manifest")
    if policy["channels"] != list(CHANNELS) or policy["band_hz"] != list(BAND_HZ):
        raise PhaseControlError("Matched controls require the seven frontal channels in order and alpha 8–13 Hz")
    threshold = validate_denominator_minimum(policy["ciplv_min_denominator"])
    if threshold != DEFAULT_CIPLV_MIN_DENOMINATOR:
        raise PhaseControlError("This bounded study fixes ciplv_min_denominator at 1e-6")
    if policy["undefined_ciplv_policy"] != "exclude_epoch_from_all_representations":
        raise PhaseControlError("Undefined ciPLV must exclude the epoch from all three representations")
    expected_minimum = 1 if policy["evidence_kind"] == "synthetic_fixture" else 20
    if (type(policy["min_epochs_per_class"]) is not int or type(policy["max_epochs_per_class"]) is not int
            or policy["min_epochs_per_class"] != expected_minimum or policy["max_epochs_per_class"] != 30):
        raise PhaseControlError(f"This bounded {policy['evidence_kind']} policy requires {expected_minimum}–30 epochs per class")


def _validate_epochs(manifest: dict, shape: tuple[int, ...]) -> tuple[list[dict], list[str]]:
    _keys(manifest, {"schema_version", "epochs"}, "epoch_manifest")
    if manifest["schema_version"] != EPOCH_SCHEMA:
        raise PhaseControlError("Unknown epoch manifest schema")
    rows = manifest["epochs"]
    if not isinstance(rows, list) or not rows or len(rows) != shape[0]:
        raise PhaseControlError("The ordered epoch manifest must match every row of the phase tensor")
    epoch_ids, spans, cohorts = set(), set(), set()
    subjects: list[str] = []
    fields = {"epoch_id", "cohort_id", "subject_id", "session_id", "recording_id",
              "recording_sha256", "start_sample", "end_sample", "comparison",
              "class_label", "qc_passed", "qc_reason_codes"}
    recordings = {}
    for row in rows:
        _keys(row, fields, "epoch record")
        for key in ("epoch_id", "cohort_id", "subject_id", "session_id", "recording_id", "comparison"):
            _literal(row[key], key)
        _digest(row["recording_sha256"], "recording_sha256")
        if row["epoch_id"] in epoch_ids:
            raise PhaseControlError("Duplicate epoch_id in the phase inputs")
        epoch_ids.add(row["epoch_id"])
        cohorts.add(row["cohort_id"])
        if row["subject_id"] not in subjects:
            subjects.append(row["subject_id"])
        start, end = row["start_sample"], row["end_sample"]
        if type(start) is not int or type(end) is not int or start < 0 or end - start != shape[1]:
            raise PhaseControlError("Integer source sample spans must exactly match the tensor sample axis")
        recording_key = tuple(row[key] for key in ("cohort_id", "subject_id", "session_id", "recording_id"))
        if recording_key in recordings and recordings[recording_key] != row["recording_sha256"]:
            raise PhaseControlError("One recording identity cannot refer to multiple source digests")
        recordings[recording_key] = row["recording_sha256"]
        span = (*recording_key, start, end)
        if span in spans:
            raise PhaseControlError("A source sample span cannot be duplicated under a new epoch identifier")
        spans.add(span)
        if type(row["class_label"]) is not int or row["class_label"] not in {0, 1}:
            raise PhaseControlError("class_label must be the literal integer 0 or 1")
        codes = row["qc_reason_codes"]
        if type(row["qc_passed"]) is not bool or not isinstance(codes, list):
            raise PhaseControlError("Shared QC requires a boolean decision and explicit reason-code list")
        for code in codes:
            _literal(code, "qc_reason_codes")
        if len(set(codes)) != len(codes) or row["qc_passed"] == bool(codes):
            raise PhaseControlError("Passed QC requires no reason codes; failed QC requires unique reason codes")
    if len(cohorts) != 1:
        raise PhaseControlError("A matched bundle must contain exactly one explicitly identified cohort")
    return rows, subjects


def _partition(train, held_out, expected: set[str], label: str) -> None:
    _ids(train, label + " training subjects")
    _ids(held_out, label + " held-out subjects")
    if len(held_out) != 1:
        raise PhaseControlError(f"{label} must leave out exactly one participant")
    if set(train) & set(held_out):
        raise PhaseControlError(f"{label} has participant leakage between partitions")
    if set(train) | set(held_out) != expected:
        raise PhaseControlError(f"{label} does not exactly cover the intended participant set")


def _validate_splits(manifest: dict, subjects: list[str]) -> None:
    _keys(manifest, {"schema_version", "splits"}, "split_manifest")
    if manifest["schema_version"] != SPLIT_SCHEMA or not isinstance(manifest["splits"], list):
        raise PhaseControlError("Unknown split manifest schema")
    if len(subjects) < 3:
        raise PhaseControlError("Nested participant LOPO requires at least three participants")
    outer_ids, outer_tests = set(), []
    for outer in manifest["splits"]:
        _keys(outer, {"split_id", "train_subject_ids", "test_subject_ids", "inner_splits"}, "outer split")
        _literal(outer["split_id"], "outer split_id")
        if outer["split_id"] in outer_ids:
            raise PhaseControlError("Duplicate outer split_id")
        outer_ids.add(outer["split_id"])
        _partition(outer["train_subject_ids"], outer["test_subject_ids"], set(subjects), "outer split")
        outer_tests.extend(outer["test_subject_ids"])
        if not isinstance(outer["inner_splits"], list):
            raise PhaseControlError("Every outer split requires an explicit nested inner split plan")
        inner_ids, inner_tests = set(), []
        for inner in outer["inner_splits"]:
            _keys(inner, {"split_id", "train_subject_ids", "validation_subject_ids"}, "inner split")
            _literal(inner["split_id"], "inner split_id")
            if inner["split_id"] in inner_ids:
                raise PhaseControlError("Duplicate inner split_id within an outer split")
            inner_ids.add(inner["split_id"])
            _partition(inner["train_subject_ids"], inner["validation_subject_ids"],
                       set(outer["train_subject_ids"]), "inner split")
            inner_tests.extend(inner["validation_subject_ids"])
        if sorted(inner_tests) != sorted(outer["train_subject_ids"]):
            raise PhaseControlError("Inner held-out folds must cover each outer training participant exactly once")
    if sorted(outer_tests) != sorted(subjects):
        raise PhaseControlError("Outer held-out folds must cover each participant exactly once")


def _validated_manifests(epochs: dict, splits: dict, policy: dict, shape: tuple[int, ...]):
    _validate_policy(policy, epochs, splits)
    rows, subjects = _validate_epochs(epochs, shape)
    _validate_splits(splits, subjects)
    comparisons = list(dict.fromkeys(row["comparison"] for row in rows))
    counts = [{"subject_id": subject, "comparison": comparison,
               "class_counts": {str(label): sum(row["subject_id"] == subject and row["comparison"] == comparison
                                                  and row["class_label"] == label for row in rows)
                                for label in (0, 1)}}
              for comparison in comparisons for subject in subjects]
    if any(count > policy["max_epochs_per_class"] for entry in counts for count in entry["class_counts"].values()):
        raise PhaseControlError("Supplied epochs exceed the frozen per-participant, per-class acquisition maximum")
    return rows, subjects, comparisons, counts


def validate_matched_phase_manifests(
    epoch_manifest: dict,
    split_manifest: dict,
    *,
    input_policy: dict,
    n_samples: int,
) -> dict:
    """Validate identities, class-count ceilings and every nested LOPO partition.

    This read-only contract check is available to the prospective authorization
    gate before it loads phase values.  It does not claim that the supplied
    metadata or prior preprocessing is independently authenticated.  Participant
    minimum counts are assessed again after common numerical/QC exclusion in
    the extracted bundle, without deleting people from the frozen split plan.
    """
    epochs = json.loads(_canonical_json(epoch_manifest))
    splits = json.loads(_canonical_json(split_manifest))
    policy = json.loads(_canonical_json(input_policy))
    if type(n_samples) is not int or n_samples < 1:
        raise PhaseControlError("n_samples must be a positive literal integer")
    if not isinstance(epochs, dict) or not isinstance(epochs.get("epochs"), list):
        raise PhaseControlError("An ordered epoch manifest is required")
    rows, subjects, comparisons, counts = _validated_manifests(
        epochs, splits, policy, (len(epochs["epochs"]), n_samples, len(CHANNELS)))
    return {"schema_version": "eeg-matched-phase-manifest-audit-1", "n_epochs": len(rows),
            "n_samples_per_epoch": n_samples, "subject_ids": subjects, "comparison_ids": comparisons,
            "source_class_counts": counts, "epoch_manifest_sha256": canonical_sha256(epochs),
            "split_manifest_sha256": canonical_sha256(splits)}


def _array_sha256(array: np.ndarray) -> str:
    values = np.ascontiguousarray(array, dtype="<f8")
    header = _canonical_json({"dtype": "<f8", "shape": list(values.shape)}).encode("utf-8")
    return hashlib.sha256(header + b"\0" + values.tobytes()).hexdigest()


def _resolved_splits(rows: list[dict], accepted_ids: list[str], manifest: dict, *,
                     min_epochs_per_class: int, max_epochs_per_class: int) -> tuple[list[dict], list[dict]]:
    accepted_set = set(accepted_ids)
    accepted_rows = [row for row in rows if row["epoch_id"] in accepted_set]
    comparisons = list(dict.fromkeys(row["comparison"] for row in rows))
    findings: list[dict] = []
    subjects = list(dict.fromkeys(row["subject_id"] for row in rows))
    for comparison in comparisons:
        for subject in subjects:
            counts = {str(label): sum(row["subject_id"] == subject and row["comparison"] == comparison
                                     and row["class_label"] == label for row in accepted_rows) for label in (0, 1)}
            insufficient = [label for label in (0, 1) if counts[str(label)] < min_epochs_per_class]
            excessive = [label for label in (0, 1) if counts[str(label)] > max_epochs_per_class]
            if insufficient or excessive:
                findings.append({"code": "UNEVALUABLE_PARTICIPANT_EPOCH_COUNT", "comparison": comparison,
                    "subject_id": subject, "class_counts": counts,
                    "min_epochs_per_class": min_epochs_per_class, "max_epochs_per_class": max_epochs_per_class,
                    "missing_class_labels": [label for label in (0, 1) if counts[str(label)] == 0],
                    "insufficient_class_labels": insufficient, "excessive_class_labels": excessive})

    def role(subject_ids, comparison, outer_id, inner_id, role_name):
        records = [row for row in accepted_rows
                   if row["subject_id"] in subject_ids and row["comparison"] == comparison]
        counts = {str(label): sum(row["class_label"] == label for row in records) for label in (0, 1)}
        missing_subjects = [subject for subject in subject_ids
                            if not any(row["subject_id"] == subject for row in records)]
        missing_classes = [label for label in (0, 1) if counts[str(label)] == 0]
        if missing_classes or missing_subjects:
            findings.append({"code": "UNEVALUABLE_FROZEN_PARTITION", "comparison": comparison,
                             "outer_split_id": outer_id, "inner_split_id": inner_id,
                             "role": role_name, "missing_class_labels": missing_classes,
                             "missing_subject_ids": missing_subjects})
        return {"subject_ids": list(subject_ids), "epoch_ids": [row["epoch_id"] for row in records],
                "class_counts": counts}

    resolved = []
    for comparison in comparisons:
        for outer in manifest["splits"]:
            oid = outer["split_id"]
            entry = {"comparison": comparison, "split_id": oid,
                     "train": role(outer["train_subject_ids"], comparison, oid, None, "train"),
                     "test": role(outer["test_subject_ids"], comparison, oid, None, "test"),
                     "inner_splits": []}
            for inner in outer["inner_splits"]:
                iid = inner["split_id"]
                entry["inner_splits"].append({"split_id": iid,
                    "train": role(inner["train_subject_ids"], comparison, oid, iid, "inner_train"),
                    "validation": role(inner["validation_subject_ids"], comparison, oid, iid, "validation")})
            resolved.append(entry)
    return resolved, findings


@dataclass(frozen=True)
class MatchedPhaseBundle:
    """Immutable canonical metadata and byte-backed, read-only feature arrays.

    Each ``matrix`` access creates a fresh view over immutable bytes.  Changing
    a returned view's shape or a returned manifest cannot mutate the bundle.
    """

    _manifest_json: str
    _matrix_bytes: tuple[bytes, bytes, bytes]
    _n_rows: int

    @property
    def manifest(self) -> dict:
        return json.loads(self._manifest_json)

    @property
    def epoch_ids(self) -> tuple[str, ...]:
        return tuple(self.manifest["accepted_epoch_ids"])

    @property
    def evaluable(self) -> bool:
        return self.manifest["evaluable"]

    def matrix(self, representation: str) -> np.ndarray:
        if representation not in REPRESENTATIONS:
            raise PhaseControlError("representation must be plv, iplv, or ciplv")
        data = self._matrix_bytes[REPRESENTATIONS.index(representation)]
        return np.frombuffer(data, dtype="<f8").reshape(self._n_rows, len(PAIR_ORDER))

    @property
    def plv(self) -> np.ndarray:
        return self.matrix("plv")

    @property
    def iplv(self) -> np.ndarray:
        return self.matrix("iplv")

    @property
    def ciplv(self) -> np.ndarray:
        return self.matrix("ciplv")

    def require_evaluable(self) -> None:
        if not self.evaluable:
            raise PhaseControlError("Frozen matched splits are unevaluable after common exclusions; inspect evaluation_findings")


def extract_matched_alpha_features(
    phases: np.ndarray,
    epoch_manifest: dict,
    split_manifest: dict,
    *,
    input_policy: dict,
) -> MatchedPhaseBundle:
    """Build the fixed three-by-21 representation set without fitting models.

    ``phases`` must be finite real radians, with shape ``[epochs, samples, 7]``
    in the locked channel order.  The caller supplies already preprocessed
    alpha phases and binds their processing contract by SHA-256.  This function
    cannot authenticate those declarations or prove band-pass provenance.
    The prospective-data gate must be checked before calling it on real B data.
    """
    # Snapshot every caller-owned object before computation and hashing.
    epoch_manifest = json.loads(_canonical_json(epoch_manifest))
    split_manifest = json.loads(_canonical_json(split_manifest))
    input_policy = json.loads(_canonical_json(input_policy))
    original = np.asarray(phases)
    if original.dtype.kind not in "fiu" or original.ndim != 3 or min(original.shape[:2]) < 1 or original.shape[2] != len(CHANNELS):
        raise PhaseControlError("phases must have nonempty real numeric shape [epochs, samples, 7]")
    with np.errstate(over="ignore", invalid="ignore"):
        values = np.array(original, dtype="<f8", order="C", copy=True)
    if not np.isfinite(values).all():
        raise PhaseControlError("Every supplied phase must be finite; sample deletion and imputation are prohibited")
    rows, subjects, _, _ = _validated_manifests(epoch_manifest, split_manifest, input_policy, values.shape)

    accepted_ids, exclusions, edge_audit = [], [], []
    outputs: dict[str, list[list[float]]] = {name: [] for name in REPRESENTATIONS}
    for index, row in enumerate(rows):
        if not row["qc_passed"]:
            exclusions.append({"epoch_id": row["epoch_id"], "subject_id": row["subject_id"],
                "reason_codes": ["SOURCE_QC_FAILED"], "source_qc_reason_codes": row["qc_reason_codes"],
                "undefined_pairs": []})
            continue
        metrics = []
        for ia in range(len(CHANNELS)):
            for ib in range(ia + 1, len(CHANNELS)):
                result = phase_metrics_from_phases(values[index, :, ia], values[index, :, ib],
                    ciplv_min_denominator=input_policy["ciplv_min_denominator"])
                metrics.append(result)
                edge_audit.append({"epoch_id": row["epoch_id"], "pair_index": len(metrics) - 1,
                                   "channels": [CHANNELS[ia], CHANNELS[ib]], "metrics": result.as_dict()})
        undefined = [i for i, result in enumerate(metrics) if result.ciplv is None]
        if undefined:
            exclusions.append({"epoch_id": row["epoch_id"], "subject_id": row["subject_id"],
                "reason_codes": ["UNDEFINED_CIPLV_COMMON_EXCLUSION"], "source_qc_reason_codes": [],
                "undefined_pairs": [{"pair_index": i, "channels": list(PAIR_ORDER[i]),
                                     "reason": metrics[i].ciplv_reason} for i in undefined]})
            continue
        accepted_ids.append(row["epoch_id"])
        for name in REPRESENTATIONS:
            outputs[name].append([getattr(result, name) for result in metrics])

    matrices = {name: np.asarray(outputs[name], dtype="<f8").reshape(len(accepted_ids), len(PAIR_ORDER))
                for name in REPRESENTATIONS}
    resolved, findings = _resolved_splits(rows, accepted_ids, split_manifest,
        min_epochs_per_class=input_policy["min_epochs_per_class"], max_epochs_per_class=input_policy["max_epochs_per_class"])
    accepted_subjects = [subject for subject in subjects
                         if any(row["subject_id"] == subject and row["epoch_id"] in accepted_ids for row in rows)]
    qc_identity = {"source_qc": [{"epoch_id": row["epoch_id"], "qc_passed": row["qc_passed"],
                                  "qc_reason_codes": row["qc_reason_codes"]} for row in rows],
                   "accepted_epoch_ids": accepted_ids, "exclusions": exclusions}
    phases_identity = {"shape": list(values.shape), "dtype": "<f8", "sha256": _array_sha256(values),
                       "hash_scope": "shape_and_canonical_little_endian_float64_phase_values"}
    shared = {"input_policy": input_policy, "source_phases": phases_identity,
              "accepted_epoch_ids": accepted_ids, "accepted_subject_ids": accepted_subjects,
              "shared_qc_sha256": canonical_sha256(qc_identity),
              "applied_splits_sha256": canonical_sha256(resolved)}
    shared_sha = canonical_sha256(shared)
    manifest = {"schema_version": BUNDLE_SCHEMA, **shared,
        "contains_model_performance_results": False,
        "provenance_scope": "caller_supplied_phase_inputs_and_attested_source_contracts",
        "epoch_manifest": epoch_manifest, "split_manifest": split_manifest,
        "exclusions": exclusions, "edge_audit": edge_audit, "applied_splits": resolved,
        "evaluable": not findings, "evaluation_findings": findings,
        "shared_provenance_sha256": shared_sha,
        "numerical_contract": {"schema_version": PHASE_METRICS_VERSION,
            "plv": "abs(mean(exp(1j*delta)))", "iplv": "abs(imag(mean(exp(1j*delta))))",
            "ciplv": "abs(imag(mean(exp(1j*delta))))/sqrt(1-real(mean(exp(1j*delta)))**2)",
            "denominator_evaluation": DENOMINATOR_EVALUATION,
            "undefined_value": None, "denominator_epsilon_added": False,
            "citation_doi": "10.1088/1741-2552/aacfe4"},
        "observed_runtime": observed_phase_runtime(),
        "implementation_sha256": {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                                  for name in ("phase_controls.py", "phase_metrics.py")},
        "representations": {name: {"columns": [f"{name.upper()}__{a}__{b}__alpha" for a, b in PAIR_ORDER],
            "shape": list(matrices[name].shape), "matrix_sha256": _array_sha256(matrices[name]),
            "shared_provenance_sha256": shared_sha} for name in REPRESENTATIONS}}
    manifest["bundle_sha256"] = canonical_sha256(manifest)
    return MatchedPhaseBundle(_canonical_json(manifest), tuple(matrices[name].tobytes() for name in REPRESENTATIONS),
                              len(accepted_ids))


def export_matched_phase_bundle(bundle: MatchedPhaseBundle, out_dir: str | Path) -> Path:
    """Atomically publish a new, protected, self-contained numerical artifact."""
    target = Path(out_dir).absolute()
    with new_output_directory(target) as stage:
        files = {}
        for name in REPRESENTATIONS:
            path = stage / f"{name}.npy"
            np.save(path, bundle.matrix(name), allow_pickle=False)
            files[path.name] = {"sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        exported = {"schema_version": EXPORT_SCHEMA, "bundle": bundle.manifest, "files": files}
        exported["export_sha256"] = canonical_sha256(exported)
        (stage / "manifest.json").write_text(_canonical_json(exported) + "\n", encoding="utf-8")
        verify_matched_phase_bundle(stage)
    return target


def _read_json(path: Path) -> dict:
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise PhaseControlError(f"Duplicate JSON key: {key}")
            result[key] = value
        return result
    try:
        return json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=pairs,
                          parse_constant=lambda value: (_ for _ in ()).throw(PhaseControlError("Nonfinite JSON")))
    except (OSError, ValueError) as exc:
        raise PhaseControlError(f"Unreadable phase bundle manifest: {path}") from exc


def _verify_edge_audit(manifest: dict, rows: list[dict], n_samples: int) -> dict[str, np.ndarray]:
    """Reconcile every recorded numerical decision with the common matrices."""
    audit = manifest["edge_audit"]
    if not isinstance(audit, list):
        raise PhaseControlError("Missing per-pair numerical evidence")
    position, accepted, excluded = 0, [], []
    values = {name: [] for name in REPRESENTATIONS}
    tolerance = 64 * np.finfo(np.float64).eps
    threshold = manifest["input_policy"]["ciplv_min_denominator"]
    metric_keys = {"plv", "iplv", "ciplv", "ciplv_status", "ciplv_reason", "mean_real", "mean_imag",
                   "denominator", "ciplv_min_denominator", "n_samples", "roundoff_adjustments",
                   "schema_version", "denominator_evaluation"}
    for row in rows:
        if not row["qc_passed"]:
            excluded.append({"epoch_id": row["epoch_id"], "subject_id": row["subject_id"],
                "reason_codes": ["SOURCE_QC_FAILED"], "source_qc_reason_codes": row["qc_reason_codes"],
                "undefined_pairs": []})
            continue
        metrics, undefined = [], []
        for pair_index, pair in enumerate(PAIR_ORDER):
            if position >= len(audit):
                raise PhaseControlError("Missing per-pair numerical evidence")
            entry = audit[position]
            position += 1
            _keys(entry, {"epoch_id", "pair_index", "channels", "metrics"}, "edge audit")
            if (entry["epoch_id"] != row["epoch_id"] or type(entry["pair_index"]) is not int
                    or entry["pair_index"] != pair_index or entry["channels"] != list(pair)):
                raise PhaseControlError("Per-pair numerical evidence changed epoch or pair order")
            metric = entry["metrics"]
            _keys(metric, metric_keys, "phase metric evidence")
            if (metric["schema_version"] != PHASE_METRICS_VERSION
                    or metric["denominator_evaluation"] != DENOMINATOR_EVALUATION
                    or type(metric["n_samples"]) is not int or metric["n_samples"] != n_samples
                    or metric["ciplv_min_denominator"] != threshold):
                raise PhaseControlError("Per-pair numerical contract mismatch")
            for key in ("plv", "iplv", "mean_real", "mean_imag", "denominator"):
                if type(metric[key]) not in (int, float) or not math.isfinite(metric[key]):
                    raise PhaseControlError("Nonfinite or nonnumeric phase metric evidence")
            real, imag, denominator = metric["mean_real"], metric["mean_imag"], metric["denominator"]
            if (abs(real) > 1 or abs(imag) > 1 or not 0 <= denominator <= 1
                    or not 0 <= metric["plv"] <= 1 or not 0 <= metric["iplv"] <= 1
                    or abs(metric["plv"] - math.hypot(real, imag)) > tolerance
                    or abs(metric["iplv"] - abs(imag)) > tolerance
                    or abs(denominator * denominator - (1 - real * real)) > 2 * tolerance):
                raise PhaseControlError("Phase metric evidence violates the declared formulas or bounds")
            adjustments = metric["roundoff_adjustments"]
            allowed_adjustments = {name + "_roundoff_clipped_to_unit_interval"
                                   for name in (*REPRESENTATIONS, "denominator")}
            if (not isinstance(adjustments, list) or any(item not in allowed_adjustments for item in adjustments)
                    or len(set(adjustments)) != len(adjustments)):
                raise PhaseControlError("Invalid roundoff audit")
            if denominator <= threshold:
                reason = ("real_unit_mean_zero_or_pi_at_float64_precision"
                          if abs(real) == 1 and denominator <= tolerance
                          else "denominator_at_or_below_prespecified_minimum")
                if metric["ciplv_status"] != "undefined" or metric["ciplv"] is not None or metric["ciplv_reason"] != reason:
                    raise PhaseControlError("Undefined ciPLV must remain null with its numerical reason")
                undefined.append({"pair_index": pair_index, "channels": list(pair), "reason": reason})
            else:
                ci = metric["ciplv"]
                if (metric["ciplv_status"] != "defined" or metric["ciplv_reason"] is not None
                        or type(ci) not in (int, float) or not math.isfinite(ci) or not 0 <= ci <= 1
                        or abs(ci - abs(imag) / denominator) > tolerance):
                    raise PhaseControlError("Defined ciPLV evidence violates the declared formula")
            metrics.append(metric)
        if undefined:
            excluded.append({"epoch_id": row["epoch_id"], "subject_id": row["subject_id"],
                "reason_codes": ["UNDEFINED_CIPLV_COMMON_EXCLUSION"], "source_qc_reason_codes": [],
                "undefined_pairs": undefined})
        else:
            accepted.append(row["epoch_id"])
            for name in REPRESENTATIONS:
                values[name].append([metric[name] for metric in metrics])
    if (position != len(audit) or accepted != manifest["accepted_epoch_ids"]
            or excluded != manifest["exclusions"]):
        raise PhaseControlError("Per-pair numerical status and common exclusion decisions disagree")
    return {name: np.asarray(values[name], dtype="<f8").reshape(len(accepted), len(PAIR_ORDER))
            for name in REPRESENTATIONS}


def verify_matched_phase_bundle(path: str | Path) -> MatchedPhaseBundle:
    """Verify artifact hashes, matrix bounds/order, and shared split/QC identity.

    These checks establish internal consistency, not independent acquisition
    provenance or prospective permission.  The latter belongs to the B gate.
    """
    root = Path(path)
    if root.is_symlink() or not root.is_dir():
        raise PhaseControlError("Phase bundle must be a real directory")
    expected_files = {"manifest.json", *(f"{name}.npy" for name in REPRESENTATIONS)}
    if {entry.name for entry in root.iterdir()} != expected_files or any(entry.is_symlink() for entry in root.iterdir()):
        raise PhaseControlError("Phase bundle inventory differs or contains a symlink")
    exported = _read_json(root / "manifest.json")
    _keys(exported, {"schema_version", "bundle", "files", "export_sha256"}, "phase export")
    digest = exported.pop("export_sha256")
    if exported["schema_version"] != EXPORT_SCHEMA or digest != canonical_sha256(exported):
        raise PhaseControlError("Phase export identity mismatch")
    manifest = exported["bundle"]
    if not isinstance(manifest, dict):
        raise PhaseControlError("Missing phase bundle metadata")
    bundle_digest = manifest.pop("bundle_sha256", None)
    if manifest.get("schema_version") != BUNDLE_SCHEMA or bundle_digest != canonical_sha256(manifest):
        raise PhaseControlError("Phase bundle identity mismatch")
    manifest["bundle_sha256"] = bundle_digest
    try:
        shape = tuple(manifest["source_phases"]["shape"])
        if len(shape) != 3 or any(type(size) is not int or size < 1 for size in shape) or shape[2] != len(CHANNELS):
            raise PhaseControlError("Invalid original phase dimensions")
        policy, epochs, splits = manifest["input_policy"], manifest["epoch_manifest"], manifest["split_manifest"]
        rows, subjects, _, _ = _validated_manifests(epochs, splits, policy, shape)
        audit_matrices = _verify_edge_audit(manifest, rows, shape[1])
        accepted_ids = manifest["accepted_epoch_ids"]
        if not isinstance(accepted_ids, list) or len(set(accepted_ids)) != len(accepted_ids):
            raise PhaseControlError("Invalid accepted epoch identity")
        accepted_set = set(accepted_ids)
        if accepted_ids != [row["epoch_id"] for row in rows if row["epoch_id"] in accepted_set]:
            raise PhaseControlError("Accepted epoch identity differs from the original row order")
        excluded_ids = [entry["epoch_id"] for entry in manifest["exclusions"]]
        if excluded_ids != [row["epoch_id"] for row in rows if row["epoch_id"] not in accepted_set]:
            raise PhaseControlError("Every removed epoch requires one ordered common exclusion record")
        expected_subjects = [subject for subject in subjects
            if any(row["subject_id"] == subject and row["epoch_id"] in accepted_set for row in rows)]
        if manifest["accepted_subject_ids"] != expected_subjects:
            raise PhaseControlError("Accepted participant identity mismatch")
        qc_identity = {"source_qc": [{"epoch_id": row["epoch_id"], "qc_passed": row["qc_passed"],
            "qc_reason_codes": row["qc_reason_codes"]} for row in rows],
            "accepted_epoch_ids": accepted_ids, "exclusions": manifest["exclusions"]}
        resolved, findings = _resolved_splits(rows, accepted_ids, splits,
            min_epochs_per_class=policy["min_epochs_per_class"], max_epochs_per_class=policy["max_epochs_per_class"])
        if (manifest["shared_qc_sha256"] != canonical_sha256(qc_identity)
                or manifest["applied_splits"] != resolved
                or manifest["applied_splits_sha256"] != canonical_sha256(resolved)
                or manifest["evaluation_findings"] != findings or manifest["evaluable"] is not (not findings)):
            raise PhaseControlError("Shared QC, frozen split resolution, or evaluability identity mismatch")
        shared = {key: manifest[key] for key in ("input_policy", "source_phases", "accepted_epoch_ids",
            "accepted_subject_ids", "shared_qc_sha256", "applied_splits_sha256")}
        if manifest["shared_provenance_sha256"] != canonical_sha256(shared):
            raise PhaseControlError("Shared representation provenance mismatch")
        _keys(exported["files"], {f"{name}.npy" for name in REPRESENTATIONS}, "matrix file inventory")
        _keys(manifest["representations"], REPRESENTATIONS, "representations")
        matrices = []
        for name in REPRESENTATIONS:
            source = root / f"{name}.npy"
            if hashlib.sha256(source.read_bytes()).hexdigest() != exported["files"][source.name]["sha256"]:
                raise PhaseControlError(f"Corrupted {name} matrix file")
            values = np.load(source, allow_pickle=False)
            expected_shape = (len(accepted_ids), len(PAIR_ORDER))
            info = manifest["representations"][name]
            if (values.dtype != np.dtype("<f8") or values.shape != expected_shape
                    or not np.isfinite(values).all() or np.any(values < 0) or np.any(values > 1)
                    or info["shape"] != list(expected_shape) or info["matrix_sha256"] != _array_sha256(values)
                    or not np.array_equal(values, audit_matrices[name])
                    or info["columns"] != [f"{name.upper()}__{a}__{b}__alpha" for a, b in PAIR_ORDER]
                    or info["shared_provenance_sha256"] != manifest["shared_provenance_sha256"]):
                raise PhaseControlError(f"Invalid or unmatched {name} representation")
            matrices.append(np.ascontiguousarray(values, dtype="<f8").tobytes())
    except (KeyError, TypeError, OSError) as exc:
        raise PhaseControlError("Incomplete or malformed phase bundle metadata") from exc
    return MatchedPhaseBundle(_canonical_json(manifest), tuple(matrices), len(accepted_ids))
