"""Archive prospective B phase extraction with its full authorization evidence.

The numerical phase bundle alone intentionally cannot authenticate a protocol
lock or participant history. This outer archive binds those inputs, the actual
phase tensor, and the captured implementation/environment to one immutable
publication. It neither fits B models nor reports model performance.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path

import numpy as np

from . import control_plan
from .artifacts import new_output_directory
from .environment import validate_environment_evidence
from .phase_controls import (REPRESENTATIONS, extract_matched_alpha_features,
                              export_matched_phase_bundle, verify_matched_phase_bundle)
from .provenance import fingerprint


SCHEMA = "eeg-control-extraction-archive-1"
MANIFEST = "archive_manifest.json"
IMPLEMENTATION_FILES = ("phase_controls.py", "phase_metrics.py")
TOLERANCE = 64 * np.finfo(np.float64).eps
VERIFICATION = {"absolute_tolerance": TOLERANCE, "relative_tolerance": 0.0,
                "scope": "finite_float64_roundoff_only_exact_identities_counts_exclusions_and_partitions"}
SCOPE = "authorized_prospective_B_phase_extraction_no_model_fit_or_performance"


class ControlExtractionError(ValueError):
    """A prospective phase archive is incomplete or contradicts its sources."""


def _bytes(path):
    path = Path(path)
    if any(part.is_symlink() for part in (path, *path.parents)) or not path.is_file():
        raise ControlExtractionError("Extraction evidence requires ordinary files without symlink paths")
    return path.read_bytes()


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def _inventory(root, *, ignore=()):
    root = Path(root)
    if any(part.is_symlink() for part in (root, *root.parents)) or not root.is_dir():
        raise ControlExtractionError("Extraction archives require an ordinary directory")
    found = {}
    for path in sorted(root.rglob("*")):
        if path.is_symlink() or (not path.is_file() and not path.is_dir()):
            raise ControlExtractionError("Extraction archives prohibit symlinks and special files")
        if path.is_file() and path.relative_to(root).as_posix() not in ignore:
            payload = _bytes(path)
            found[path.relative_to(root).as_posix()] = {"sha256": _sha(payload), "bytes": len(payload)}
    return found


def _json(path):
    return control_plan._json(_bytes(path))


def _write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _snapshot(value):
    # Do not coerce numpy scalars, paths, booleans or stringified numbers into
    # stronger literal evidence than the caller actually supplied.
    return json.loads(json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False))


def _phase_bytes(phases):
    if isinstance(phases, (str, Path)):
        data = _bytes(phases)
    else:
        stream = io.BytesIO()
        np.save(stream, np.asarray(phases), allow_pickle=False)
        data = stream.getvalue()
    if not data.startswith(b"\x93NUMPY"):
        raise ControlExtractionError("Phase inputs must be a single safe .npy array")
    return data


def _load_phases(data):
    stream = io.BytesIO(data)
    value = np.load(stream, allow_pickle=False)
    if not isinstance(value, np.ndarray) or stream.tell() != len(data):
        raise ControlExtractionError("Phase input contains extra arrays or trailing bytes")
    return value


def _implementation_sources():
    return {name: _bytes(Path(__file__).with_name(name)) for name in IMPLEMENTATION_FILES}


def _validate_implementation(implementation, parent, archived_sources):
    control_plan._validate_code(implementation)
    if implementation["working_source_sha256"] != parent["code"]["working_source_sha256"]:
        raise ControlExtractionError("Extraction implementation differs from the archived plan code")
    expected = {name: _sha(payload) for name, payload in archived_sources.items()}
    for name, digest in expected.items():
        key = "src/eeg_ma/" + name
        if implementation["source_files"].get(key) != digest:
            raise ControlExtractionError("Numerical implementation snapshot differs from its authorized source inventory")
    return expected


def _authorization_context(root, features, created_at, arm):
    locked = control_plan.validate_control_plan_lock(root / "plan_lock")
    authorization = _json(root / "authorization.json")
    required = {"input_policy", "arm_contract", "recruitment_audit", "session_history_sha256", "environment", "implementation"}
    if set(authorization) != required:
        raise ControlExtractionError("Extraction authorization evidence is incomplete or contains unknown fields")
    history = _json(root / "session_history.json")
    annotation_path = root / "annotation_manifest.json"
    annotations = _json(annotation_path) if annotation_path.exists() else None
    manifest = features.manifest
    epochs, splits = manifest["epoch_manifest"], manifest["split_manifest"]
    context = control_plan.validate_control_phase_context(locked, epochs, splits, history,
        arm=arm, annotation_manifest=annotations, as_of=created_at)
    sources = {name: _bytes(root / "implementation" / name) for name in IMPLEMENTATION_FILES}
    hashes = _validate_implementation(authorization["implementation"], locked["manifest"], sources)
    environment = authorization["environment"]
    validate_environment_evidence(environment, require_reference=True)
    locked_environment = _json(root / "plan_lock/environment.json")
    if environment["lock_id"] != locked_environment["lock_id"]:
        raise ControlExtractionError("Extraction environment differs from the archived reference lock")
    policy = control_plan._phase_input_policy(locked["plan"], locked["manifest"], epochs, splits, context["arm_contract"],
        code_sha256=authorization["implementation"]["working_source_sha256"], environment_sha256=fingerprint(environment))
    if (authorization["input_policy"] != policy or manifest["input_policy"] != policy
            or authorization["arm_contract"] != context["arm_contract"]
            or authorization["recruitment_audit"] != context["session_history_audit"]
            or authorization["session_history_sha256"] != fingerprint(history)):
        raise ControlExtractionError("Authorization policy, history, arm contract or recruitment audit differs from archived evidence")
    if manifest["implementation_sha256"] != hashes:
        raise ControlExtractionError("Phase feature implementation hashes differ from their archived authorized sources")
    runtime = {"python": environment["python"], "numpy": environment["packages"]["numpy"]}
    if manifest["observed_runtime"] != runtime:
        raise ControlExtractionError("Phase feature runtime observations differ from the captured authorization environment")
    return locked, authorization, history, annotations


def _equal_semantics(actual, expected, label="feature manifest"):
    """Allow only declared finite float roundoff; structure and decisions are exact."""
    if isinstance(expected, dict):
        if not isinstance(actual, dict) or set(actual) != set(expected):
            raise ControlExtractionError(f"Recomputed {label} fields differ")
        for key in expected:
            _equal_semantics(actual[key], expected[key], label + "/" + key)
    elif isinstance(expected, list):
        if not isinstance(actual, list) or len(actual) != len(expected):
            raise ControlExtractionError(f"Recomputed {label} coverage differs")
        for index, (left, right) in enumerate(zip(actual, expected)):
            _equal_semantics(left, right, label + "/" + str(index))
    elif type(expected) is float:
        if type(actual) not in (int, float) or not np.isfinite(actual) or not np.isfinite(expected) or abs(actual - expected) > TOLERANCE:
            raise ControlExtractionError(f"Recomputed {label} numerical value differs")
    elif type(actual) is not type(expected) or actual != expected:
        raise ControlExtractionError(f"Recomputed {label} value differs")


def _verify_recomputation(archived, phases):
    old = archived.manifest
    recomputed = extract_matched_alpha_features(phases, old["epoch_manifest"], old["split_manifest"], input_policy=old["input_policy"])
    current = recomputed.manifest
    for name in REPRESENTATIONS:
        left, right = archived.matrix(name), recomputed.matrix(name)
        if left.shape != right.shape or not np.allclose(left, right, rtol=0, atol=TOLERANCE):
            raise ControlExtractionError(f"Stored {name} features differ from the actual archived phase tensor")
        # The stored matrix hash is independently checked by the numerical
        # bundle verifier. A reader's float64 roundoff can change its recomputed
        # byte hash without changing any scientifically allowed value.
        current["representations"][name]["matrix_sha256"] = old["representations"][name]["matrix_sha256"]
    # These fields describe the historical extraction, not the current reader.
    # Their bindings were independently checked against captured authorization
    # and source snapshots above; archived code is never executed.
    current["observed_runtime"] = old["observed_runtime"]
    current["implementation_sha256"] = old["implementation_sha256"]
    current["bundle_sha256"] = old["bundle_sha256"]
    _equal_semantics(old, current)


def _binding(root, features, locked, authorization, history, annotations, arm):
    bundle = features.manifest
    exported = _json(root / "features/manifest.json")
    return {"analysis_id": locked["plan"]["analysis_id"], "study_id": locked["plan"]["study_id"],
            "cohort_id": locked["plan"]["data"]["cohort_id"],
            "control_plan_lock_id": locked["manifest"]["control_plan_lock_id"], "arm": arm,
            "bundle_sha256": bundle["bundle_sha256"], "export_sha256": exported["export_sha256"],
            "authorization_sha256": _sha(_bytes(root / "authorization.json")),
            "session_history_sha256": fingerprint(history),
            "annotation_manifest_sha256": fingerprint(annotations) if annotations is not None else None,
            "source_phases": bundle["source_phases"], "phases_file_sha256": _sha(_bytes(root / "phases.npy")),
            "implementation": authorization["implementation"],
            "environment_id": authorization["environment"]["environment_id"],
            "environment_lock_id": authorization["environment"]["lock_id"],
            "plan_lock_inventory": _inventory(root / "plan_lock"),
            "n_recruited": authorization["recruitment_audit"]["n_recruited"],
            "n_source_participants": len(authorization["recruitment_audit"]["selected_B_sessions"]),
            "n_source_epochs": len(bundle["epoch_manifest"]["epochs"]),
            "n_accepted_epochs": len(bundle["accepted_epoch_ids"]), "n_excluded_epochs": len(bundle["exclusions"]),
            "evaluable": bundle["evaluable"], "contains_model_performance_results": False,
            "scope": SCOPE, "numerical_verification": VERIFICATION,
            "acquisition_evidence_scope": "operator_attested_recording_QA_and_preprocessing_metadata_not_independent_acquisition_authentication"}


def build_control_extraction(phases, epoch_manifest, split_manifest, session_history, out_dir, *,
                             lock_dir, arm="recorded_reference", annotation_manifest=None):
    """Authorize once, extract without fitting, and publish complete evidence atomically.

    ``phases`` may be a real numerical ndarray or a path to a single safe NPY
    array. The tensor remains an explicitly supplied preprocessed phase input;
    this function does not infer its reference or band from numerical values.
    """
    try:
        originals = {"epochs": epoch_manifest, "splits": split_manifest, "history": session_history,
                     "annotations": annotation_manifest}
        snapshots = _snapshot(originals)
        lock_inventory = _inventory(lock_dir)
        authorization = control_plan.authorize_phase_inputs(lock_dir, snapshots["epochs"], snapshots["splits"], snapshots["history"],
            arm=arm, annotation_manifest=snapshots["annotations"])
        created = control_plan._now()
        phase_bytes = _phase_bytes(phases)
        phase_values = _load_phases(phase_bytes)
        source_code = _implementation_sources()
        locked = control_plan.validate_control_plan_lock(lock_dir)
        _validate_implementation(authorization["implementation"], locked["manifest"], source_code)
        bundle = extract_matched_alpha_features(phase_values, snapshots["epochs"], snapshots["splits"], input_policy=authorization["input_policy"])
        with new_output_directory(out_dir) as stage:
            (stage / "plan_lock").mkdir()
            for name, descriptor in lock_inventory.items():
                payload = _bytes(Path(lock_dir) / name)
                if {"sha256": _sha(payload), "bytes": len(payload)} != descriptor:
                    raise ControlExtractionError("Plan lock changed during extraction publication")
                path = stage / "plan_lock" / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(payload)
            (stage / "implementation").mkdir()
            for name, payload in source_code.items():
                (stage / "implementation" / name).write_bytes(payload)
            (stage / "phases.npy").write_bytes(phase_bytes)
            _write_json(stage / "authorization.json", authorization)
            _write_json(stage / "session_history.json", snapshots["history"])
            if snapshots["annotations"] is not None:
                _write_json(stage / "annotation_manifest.json", snapshots["annotations"])
            export_matched_phase_bundle(bundle, stage / "features")
            manifest = {"schema_version": SCHEMA, "status": "complete", "created_at": created,
                **_binding(stage, bundle, locked, authorization, snapshots["history"], snapshots["annotations"], arm),
                "artifacts": _inventory(stage)}
            manifest["archive_id"] = fingerprint(manifest)
            _write_json(stage / MANIFEST, manifest)
            verify_control_extraction(stage)
            if (_snapshot(originals) != snapshots or _phase_bytes(phases) != phase_bytes
                    or _inventory(lock_dir) != lock_inventory or _implementation_sources() != source_code
                    or control_plan.code_identity() != authorization["implementation"]):
                raise ControlExtractionError("Extraction source inputs or implementation changed during publication")
        return manifest
    except (ControlExtractionError, FileExistsError):
        raise
    except (ValueError, TypeError, KeyError, AttributeError, OSError, OverflowError) as exc:
        raise ControlExtractionError(f"Invalid prospective extraction evidence: {exc}") from exc


def verify_control_extraction(out_dir):
    """Revalidate archived authorization and recompute features without live gates."""
    try:
        root = Path(out_dir)
        manifest = _json(root / MANIFEST)
        body = {key: value for key, value in manifest.items() if key != "archive_id"}
        if manifest.get("schema_version") != SCHEMA or manifest.get("status") != "complete" or manifest.get("archive_id") != fingerprint(body):
            raise ControlExtractionError("Extraction archive identity/completion digest is invalid")
        if manifest.get("artifacts") != _inventory(root, ignore=(MANIFEST,)):
            raise ControlExtractionError("Extraction archive inventory/hash mismatch")
        created = control_plan._utc(manifest["created_at"], "archive created_at")
        if created > datetime.now(timezone.utc):
            raise ControlExtractionError("Extraction archive creation cannot be in the future")
        features = verify_matched_phase_bundle(root / "features")
        locked, authorization, history, annotations = _authorization_context(root, features, manifest["created_at"], manifest["arm"])
        if created < control_plan._utc(locked["manifest"]["created_at"], "plan created_at"):
            raise ControlExtractionError("Extraction archive cannot predate its plan lock")
        expected_files = {"authorization.json", "session_history.json", "phases.npy", "features/manifest.json",
            *(f"features/{name}.npy" for name in REPRESENTATIONS),
            *(f"implementation/{name}" for name in IMPLEMENTATION_FILES),
            *(f"plan_lock/{name}" for name in _inventory(root / "plan_lock"))}
        if annotations is not None:
            expected_files.add("annotation_manifest.json")
        if set(manifest["artifacts"]) != expected_files:
            raise ControlExtractionError("Extraction archive has an unexpected or missing artifact")
        binding = _binding(root, features, locked, authorization, history, annotations, manifest["arm"])
        expected_keys = {"schema_version", "status", "created_at", "archive_id", "artifacts", *binding}
        if set(manifest) != expected_keys or any(manifest.get(key) != value for key, value in binding.items()):
            raise ControlExtractionError("Extraction archive source, count, policy or model-performance scope binding differs")
        _verify_recomputation(features, _load_phases(_bytes(root / "phases.npy")))
        return manifest
    except ControlExtractionError:
        raise
    except (ValueError, TypeError, KeyError, AttributeError, OSError, OverflowError) as exc:
        raise ControlExtractionError(f"Invalid prospective extraction archive: {exc}") from exc
