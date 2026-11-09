"""Immutable, label-separated inference for fixed development models.

Preparation is a data-custodian operation. Inference opens only its label-free
inputs directory. Scoring commits outcome access before it opens labels and
uses the prospectively selected sessions from the independent enrollment ledger.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd

from .artifacts import new_output_directory
from .cohort_policy import assess_enrollment
from .environment import validate_environment_evidence, verify_reference_environment
from .fixed_model import ANALYSIS_ID, CLASS_MAPPING, FixedLDAModel, validate_feature_schema
from .protocol import (authorize_external_score, check_external_prediction,
                       guard_external_prediction, protocol_training_contract, validate_lock)
from .provenance import code_identity, fingerprint, read_csv_literal, sha256_file
from .schema import load_feature_table, stable_id


TITLE = "Fixed development models on an independent external cohort"
EXTRACTION_POLICY = "correct_eligible_formal_trials_after_shared_whole_recording_QA"
IDENTITY = ["cohort_id", "subject_id", "session_id", "recording_id", "epoch_id", "comparison",
            "protocol_id", "protocol_version", "source_sha256", "start_sample", "end_sample",
            "units", "reference", "channel_order"]
PREDICTION_COLUMNS = IDENTITY + ["model_role", "model_id", "protocol_lock_id", "analysis_id", "predicted_label"]
COMMIT_FIELDS = ("prediction_id", "protocol_lock_id", "cohort_manifest_sha256", "cohort_subject_ids", "created_at")
SHA = re.compile(r"[0-9a-f]{64}\Z")


class ExternalError(ValueError):
    """External evidence is incomplete or violates the frozen boundary."""


def _now():
    return datetime.now(timezone.utc).isoformat()


def _timestamp(value, label):
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except (AttributeError, TypeError, ValueError) as exc:
        raise ExternalError(f"Invalid {label} timestamp") from exc
    if parsed.tzinfo is None or parsed.utcoffset().total_seconds() != 0:
        raise ExternalError(f"{label} timestamp must specify UTC")
    if parsed > datetime.now(timezone.utc):
        raise ExternalError(f"{label} timestamp cannot be in the future")
    return parsed


def _validate_code(code, *, locked_code=None):
    if (not isinstance(code, dict) or not isinstance(code.get("commit"), str)
            or not re.fullmatch(r"[0-9a-f]{40}", code["commit"]) or code.get("dirty") is not False
            or not isinstance(code.get("source_files"), dict) or not code["source_files"]
            or any(not isinstance(name, str) or not isinstance(digest, str) or not SHA.fullmatch(digest)
                   for name, digest in code["source_files"].items())
            or fingerprint(code["source_files"]) != code.get("working_source_sha256")):
        raise ExternalError("External inference requires a clean, identifiable source-code freeze")
    if locked_code is not None and code["working_source_sha256"] != locked_code["working_source_sha256"]:
        raise ExternalError("Inference source code differs from the frozen protocol code")


def _json(data: bytes):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ExternalError(f"Duplicate JSON field: {key}")
            result[key] = value
        return result
    try:
        value = json.loads(data.decode("utf-8"), object_pairs_hook=pairs,
                           parse_constant=lambda v: (_ for _ in ()).throw(ValueError(v)))
    except (ValueError, TypeError, UnicodeError) as exc:
        raise ExternalError(f"Invalid external evidence JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise ExternalError("External evidence must be an object")
    return value


def _bytes(path):
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise ExternalError(f"Ordinary evidence file required: {path.name}")
    return path.read_bytes()


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def _write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _signed(value, key):
    result = dict(value)
    result[key] = fingerprint(result)
    return result


def _validate_id(value, key, schema):
    payload = dict(value)
    digest = payload.pop(key, None)
    if value.get("schema_version") != schema or digest != fingerprint(payload):
        raise ExternalError(f"Invalid {key} schema or digest")


def _inventory(root, names):
    return {name: {"sha256": _sha(_bytes(Path(root) / name)), "bytes": (Path(root) / name).stat().st_size}
            for name in sorted(names)}


def _check_inventory(root, inventory, expected_names, manifest_name):
    root = Path(root)
    if root.is_symlink() or not root.is_dir() or set(inventory) != set(expected_names):
        raise ExternalError("External artifact inventory is incomplete")
    paths = list(root.rglob("*"))
    if any(p.is_symlink() for p in paths):
        raise ExternalError("External artifacts cannot contain symlinks")
    if {p.relative_to(root).as_posix() for p in paths if p.is_file()} != {*expected_names, manifest_name}:
        raise ExternalError("Unexpected or missing external artifact")
    if inventory != _inventory(root, expected_names):
        raise ExternalError("External artifact bytes/hash mismatch")


def _frame(data):
    try:
        return read_csv_literal(io.StringIO(data.decode("utf-8-sig")))
    except (ValueError, UnicodeError) as exc:
        raise ExternalError(f"Invalid external CSV: {exc}") from exc


def _csv_descriptor(path, frame):
    return {"sha256": sha256_file(path), "bytes": Path(path).stat().st_size,
            "rows": len(frame), "columns": list(frame.columns)}


def _ordered_features(schema):
    validate_feature_schema(schema)
    return schema["reference_features"] + schema["primary_features"]


def _validate_input_manifest(manifest):
    keys = {"schema_version", "analysis_id", "feature_schema_text", "feature_schema_sha256",
            "feature_order", "cohort_manifest_text", "cohort_manifest_sha256", "source_feature_sha256",
            "extraction_policy", "features", "input_id"}
    _validate_id(manifest, "input_id", "eeg-external-input-1")
    if (set(manifest) != keys or manifest["analysis_id"] != ANALYSIS_ID
            or manifest["extraction_policy"] != EXTRACTION_POLICY
            or any(not isinstance(manifest[key], str) or not SHA.fullmatch(manifest[key])
                   for key in ("feature_schema_sha256", "cohort_manifest_sha256", "source_feature_sha256"))
            or any(not isinstance(manifest[key], str) for key in ("feature_schema_text", "cohort_manifest_text"))):
        raise ExternalError("External input manifest must contain only the fixed label-free contract")


def _validate_features(frame, manifest, *, spec=None, development_ids=()):
    schema_text = manifest["feature_schema_text"]
    if _sha(schema_text.encode("utf-8")) != manifest["feature_schema_sha256"]:
        raise ExternalError("External feature schema hash mismatch")
    schema = _json(schema_text.encode("utf-8"))
    columns = _ordered_features(schema)
    if manifest.get("feature_order") != columns or list(frame.columns) != IDENTITY + columns:
        raise ExternalError("External feature order/schema mismatch; labels, stage and extra predictors are prohibited")
    if frame.empty or not np.isfinite(frame[columns].to_numpy(float)).all():
        raise ExternalError("External features must be nonempty and finite")
    for key in set(IDENTITY) - {"start_sample", "end_sample"}:
        if any(not isinstance(v, str) or not v or v != v.strip() for v in frame[key]):
            raise ExternalError(f"External identity must be literal and nonempty: {key}")
    for key in ("start_sample", "end_sample"):
        values = pd.to_numeric(frame[key], errors="raise").to_numpy(float)
        if not np.isfinite(values).all() or np.any(values != np.floor(values)):
            raise ExternalError("External sample intervals must be finite integers")
    if (frame.start_sample < 0).any() or (frame.end_sample <= frame.start_sample).any():
        raise ExternalError("Invalid external sample interval")
    physical = ["cohort_id", "subject_id", "session_id", "recording_id", "start_sample", "end_sample"]
    if frame.epoch_id.duplicated().any() or frame.duplicated(physical).any():
        raise ExternalError("Duplicate external epoch or physical interval")
    cohort_text = manifest["cohort_manifest_text"]
    if _sha(cohort_text.encode("utf-8")) != manifest["cohort_manifest_sha256"]:
        raise ExternalError("External cohort evidence hash mismatch")
    cohort = _json(cohort_text.encode("utf-8"))
    cohort_keys = {"schema_version", "cohort_id", "protocol_id", "subject_ids", "outcomes_accessed"}
    ids = cohort.get("subject_ids")
    if (set(cohort) != cohort_keys or cohort.get("schema_version") != "eeg-confirmatory-cohort-1"
            or cohort.get("outcomes_accessed") is not False or not isinstance(ids, list) or not ids
            or any(not isinstance(value, str) or not value or value != value.strip() for value in ids)
            or len(set(ids)) != len(ids)):
        raise ExternalError("External cohort manifest must preserve the complete label-free roster contract")
    if (set(frame.cohort_id) != {cohort["cohort_id"]}
            or set(frame.protocol_id) != {cohort["protocol_id"]}
            or not set(frame.subject_id).issubset(cohort["subject_ids"])):
        raise ExternalError("External rows differ from the complete declared cohort")
    if set(frame.subject_id) & set(development_ids):
        raise ExternalError("Development/external subject overlap")
    recordings = ["cohort_id", "subject_id", "session_id", "recording_id"]
    immutable = ["source_sha256", "units", "reference", "channel_order", "protocol_id", "protocol_version"]
    if (frame.groupby(recordings)[immutable].nunique() > 1).any().any():
        raise ExternalError("External recording has inconsistent acquisition metadata")
    for row in frame.to_dict("records"):
        if not SHA.fullmatch(row["source_sha256"]) or not re.fullmatch(r"epoch-[0-9a-f]{64}", row["epoch_id"]):
            raise ExternalError("External inference requires raw source and stable epoch identities")
        recording = stable_id("recording", {"cohort_id": row["cohort_id"], "subject_id": row["subject_id"],
            "session_id": row["session_id"], "raw_sha256": row["source_sha256"]})
        if row["recording_id"] != recording:
            raise ExternalError("External recording identity differs from raw/session identity")
    if spec is not None:
        acquisition, a = spec["acquisition"], spec["protocols"]["A"]
        if (set(frame.protocol_version) != {a["protocol_version"]}
                or set(frame.units) != {acquisition["units"]}
                or set(frame.reference) != {acquisition["reference"]}
                or any(json.loads(value) != acquisition["raw_channels"] for value in frame.channel_order)):
            raise ExternalError("External acquisition/protocol metadata differ from the lock")
        if not np.allclose(frame.end_sample - frame.start_sample,
                           acquisition["sampling_rate_hz"] * acquisition["epoch_seconds"], rtol=0, atol=0):
            raise ExternalError("External epoch length differs from the lock")
        comparisons = {a["primary_comparison"], a["secondary_comparison"]}
        if not set(frame.comparison).issubset(comparisons):
            raise ExternalError("External comparison is outside the frozen A protocol")
    return schema, cohort


def _load_inputs(input_dir, *, spec=None, development_ids=()):
    root = Path(input_dir)
    manifest = _json(_bytes(root / "input_manifest.json"))
    _validate_input_manifest(manifest)
    if root.is_symlink() or {p.name for p in root.iterdir()} != {"features.csv", "input_manifest.json"}:
        raise ExternalError("Inference input directory must contain only label-free features and its manifest")
    data = _bytes(root / "features.csv")
    frame = _frame(data)
    if manifest.get("features") != _csv_descriptor(root / "features.csv", frame):
        raise ExternalError("External feature payload hash/schema mismatch")
    schema, cohort = _validate_features(frame, manifest, spec=spec, development_ids=development_ids)
    return manifest, frame, schema, cohort


def prepare_external_inputs(feature_path, out_dir, *, lock_dir, cohort_manifest_path):
    """Project a QA-validated feature table into physically separate inputs/labels.

    The existing extraction contract contains correct eligible formal trials
    after whole-recording QA. Session/participant eligibility is independently
    audited at score time; no operator-selected epoch mask is introduced here.
    """
    context = check_external_prediction(lock_dir, cohort_manifest_path)
    locked, spec = context["locked"], context["locked"]["protocol"]
    source = Path(feature_path)
    frame = load_feature_table(source)
    if set(frame.identity_origin) != {"raw_metadata"}:
        raise ExternalError("New-cohort inference cannot use legacy/unverified acquisition features")
    schema_entry = locked["manifest"]["artifacts"]["feature_schema"]
    schema_bytes = _bytes(Path(lock_dir) / schema_entry["file"])
    schema = _json(schema_bytes)
    columns = _ordered_features(schema)
    if set(columns) - set(frame.columns):
        raise ExternalError("External features lack the complete frozen feature families")
    features = frame[IDENTITY + columns].sort_values("epoch_id").reset_index(drop=True)
    labels = frame[["epoch_id", "class_label"]].rename(columns={"class_label": "true_label"}).sort_values("epoch_id").reset_index(drop=True)
    manifest = {"schema_version": "eeg-external-input-1", "analysis_id": ANALYSIS_ID,
        "feature_schema_text": schema_bytes.decode("utf-8"), "feature_schema_sha256": _sha(schema_bytes),
        "feature_order": columns, "cohort_manifest_text": context["cohort_manifest_text"],
        "cohort_manifest_sha256": context["cohort_manifest_sha256"],
        "source_feature_sha256": frame.attrs["source_feature_sha256"],
        "extraction_policy": EXTRACTION_POLICY}
    _validate_features(features, manifest, spec=spec, development_ids=locked["development_subject_ids"])
    with guard_external_prediction(lock_dir, cohort_manifest_path,
            expected_lock_id=locked["manifest"]["lock_id"], expected_cohort_sha256=context["cohort_manifest_sha256"]), \
            new_output_directory(out_dir) as stage:
        (stage / "inputs").mkdir()
        (stage / "labels").mkdir()
        fp, lp = stage / "inputs/features.csv", stage / "labels/labels.csv"
        features.to_csv(fp, index=False)
        labels.to_csv(lp, index=False)
        manifest["features"] = _csv_descriptor(fp, features)
        manifest = _signed(manifest, "input_id")
        _write_json(stage / "inputs/input_manifest.json", manifest)
        label_manifest = _signed({"schema_version": "eeg-external-labels-1", "class_mapping": CLASS_MAPPING,
            "cohort_manifest_sha256": context["cohort_manifest_sha256"], "input_id": manifest["input_id"],
            "labels": _csv_descriptor(lp, labels)}, "labels_id")
        _write_json(stage / "labels/labels_manifest.json", label_manifest)
        _load_inputs(stage / "inputs", spec=spec, development_ids=locked["development_subject_ids"])
        if _sha(_bytes(source)) != manifest["source_feature_sha256"]:
            raise ExternalError("Source feature table changed during projection")
    return {"input_id": manifest["input_id"], "labels_id": label_manifest["labels_id"], "n_epochs": len(features)}


def _prediction_table(features, documents, lock_id):
    tables = []
    for role in ("primary", "reference"):
        model = FixedLDAModel(documents[role])
        for comparison in sorted(features.comparison.unique()):
            selected = features.loc[features.comparison == comparison].sort_values("epoch_id")
            table = selected[IDENTITY].copy()
            table["model_role"], table["model_id"] = role, model.model_id
            table["protocol_lock_id"], table["analysis_id"] = lock_id, ANALYSIS_ID
            table["predicted_label"] = model.predict(selected[list(model.feature_order)], comparison=comparison)
            tables.append(table)
    return pd.concat(tables, ignore_index=True)[PREDICTION_COLUMNS].sort_values(["model_role", "epoch_id"]).reset_index(drop=True)


def _same_table(actual, expected, label):
    if list(actual.columns) != list(expected.columns) or len(actual) != len(expected):
        raise ExternalError(f"{label} coverage/columns mismatch")
    for column in expected:
        if pd.api.types.is_numeric_dtype(expected[column]) and not pd.api.types.is_bool_dtype(expected[column]):
            try:
                left = pd.to_numeric(actual[column], errors="raise").to_numpy(float)
            except (TypeError, ValueError) as exc:
                raise ExternalError(f"{label} contains invalid numeric values: {column}") from exc
            right = expected[column].to_numpy(float)
            equal = (np.array_equal(left, right) if pd.api.types.is_integer_dtype(expected[column])
                     else np.allclose(left, right, rtol=1e-12, atol=1e-14, equal_nan=True))
            if not equal:
                raise ExternalError(f"{label} values differ: {column}")
        elif actual[column].astype(str).tolist() != expected[column].astype(str).tolist():
            raise ExternalError(f"{label} identity/values differ: {column}")


PREDICTION_ARTIFACTS = {"features.csv", "input_manifest.json", "models/primary.json", "models/reference.json", "predictions.csv"}


def predict_external(input_dir, out_dir, *, lock_dir, cohort_manifest_path):
    """Read label-free inputs, transform with frozen state, publish once; never fit."""
    context = check_external_prediction(lock_dir, cohort_manifest_path)
    locked, spec = context["locked"], context["locked"]["protocol"]
    code = code_identity()
    _validate_code(code, locked_code=locked["manifest"]["code"])
    source_manifest, features, schema, cohort = _load_inputs(input_dir, spec=spec,
        development_ids=locked["development_subject_ids"])
    if source_manifest["cohort_manifest_sha256"] != context["cohort_manifest_sha256"]:
        raise ExternalError("Prediction inputs refer to a different cohort declaration")
    schema_sha = locked["manifest"]["artifacts"]["feature_schema"]["sha256"]
    if source_manifest["feature_schema_sha256"] != schema_sha:
        raise ExternalError("Prediction inputs use a different frozen feature schema")
    environment = verify_reference_environment()
    if any(model["environment_lock_id"] != environment["lock_id"] for model in context["models"].values()):
        raise ExternalError("Inference runtime must match the model's verified reference lock")
    predictions = _prediction_table(features, context["models"], locked["manifest"]["lock_id"])
    with guard_external_prediction(lock_dir, cohort_manifest_path,
            expected_lock_id=locked["manifest"]["lock_id"], expected_cohort_sha256=context["cohort_manifest_sha256"]), \
            new_output_directory(out_dir) as stage:
        (stage / "models").mkdir()
        (stage / "features.csv").write_bytes(_bytes(Path(input_dir) / "features.csv"))
        (stage / "input_manifest.json").write_bytes(_bytes(Path(input_dir) / "input_manifest.json"))
        for role in ("primary", "reference"):
            entry = locked["manifest"]["artifacts"][role + "_model"]
            data = _bytes(Path(lock_dir) / entry["file"])
            if _sha(data) != context["model_sha256"][role]:
                raise ExternalError("Locked model changed during inference")
            (stage / f"models/{role}.json").write_bytes(data)
        predictions.to_csv(stage / "predictions.csv", index=False)
        manifest = _signed({"schema_version": "eeg-external-predictions-1", "status": "complete",
            "analysis_id": ANALYSIS_ID, "report_title": TITLE, "created_at": _now(),
            "protocol_lock_id": locked["manifest"]["lock_id"], "protocol_sha256": locked["manifest"]["protocol_sha256"],
            "study_protocol_id": spec["protocol_id"], "study_protocol_version": spec["protocol_version"],
            "acquisition_protocol_id": spec["protocols"]["A"]["protocol_id"],
            "acquisition_protocol_version": spec["protocols"]["A"]["protocol_version"],
            "cohort_manifest_sha256": context["cohort_manifest_sha256"], "cohort_subject_ids": sorted(cohort["subject_ids"]),
            "input_id": source_manifest["input_id"], "model_sha256": context["model_sha256"],
            "model_ids": {role: model["model_id"] for role, model in context["models"].items()},
            "environment": environment, "code": code, "n_epochs": len(features),
            "n_predictions": len(predictions), "artifacts": _inventory(stage, PREDICTION_ARTIFACTS)}, "prediction_id")
        _write_json(stage / "prediction_manifest.json", manifest)
        validate_predictions(stage, lock_dir=lock_dir)
        # The shared ledger lock stays held until the atomic directory rename,
        # so outcome access cannot race publication of a new prediction set.
        if code_identity() != code:
            raise ExternalError("Source code changed during prediction publication")
    return manifest


def validate_predictions(prediction_dir, *, lock_dir=None):
    """Check complete immutable predictions against actual stored fixed model state."""
    try:
        root = Path(prediction_dir)
        manifest = _json(_bytes(root / "prediction_manifest.json"))
        _validate_id(manifest, "prediction_id", "eeg-external-predictions-1")
        if (manifest.get("status") != "complete" or manifest.get("analysis_id") != ANALYSIS_ID
                or manifest.get("report_title") != TITLE):
            raise ExternalError("External prediction status/analysis contract mismatch")
        _check_inventory(root, manifest["artifacts"], PREDICTION_ARTIFACTS, "prediction_manifest.json")
        inputs = _json(_bytes(root / "input_manifest.json"))
        _validate_input_manifest(inputs)
        features = _frame(_bytes(root / "features.csv"))
        if inputs.get("features") != _csv_descriptor(root / "features.csv", features):
            raise ExternalError("Stored prediction features differ from their input contract")
        schema, cohort = _validate_features(features, inputs)
        if (inputs["input_id"] != manifest["input_id"] or inputs["analysis_id"] != ANALYSIS_ID
                or inputs["cohort_manifest_sha256"] != manifest["cohort_manifest_sha256"]
                or sorted(cohort["subject_ids"]) != manifest["cohort_subject_ids"]):
            raise ExternalError("Prediction cohort/input binding differs")
        validate_environment_evidence(manifest["environment"], require_reference=True)
        _validate_code(manifest["code"])
        created_at = _timestamp(manifest["created_at"], "prediction creation")
        documents = {role: _json(_bytes(root / f"models/{role}.json")) for role in ("primary", "reference")}
        for role, document in documents.items():
            model = FixedLDAModel(document)
            if (document["model_role"] != role or document["execution_scope"] != "reference_development_fit"
                    or document["feature_schema"] != schema or document["feature_schema_sha256"] != inputs["feature_schema_sha256"]
                    or document["environment_lock_id"] != manifest["environment"]["lock_id"]
                    or model.model_id != manifest["model_ids"][role]
                    or sha256_file(root / f"models/{role}.json") != manifest["model_sha256"][role]
                    or set(features.subject_id) & set(document["training_subject_ids"])):
                raise ExternalError("Prediction model/schema/runtime/development boundary differs")
        contract = documents["primary"]["protocol_training_contract"]
        if documents["reference"]["protocol_training_contract"] != contract:
            raise ExternalError("External models were trained for different protocol contracts")
        spec = contract["protocol"]
        _validate_features(features, inputs, spec=spec, development_ids=documents["primary"]["training_subject_ids"])
        if (manifest["study_protocol_id"] != spec["protocol_id"] or manifest["study_protocol_version"] != spec["protocol_version"]
                or manifest["acquisition_protocol_id"] != spec["protocols"]["A"]["protocol_id"]
                or manifest["acquisition_protocol_version"] != spec["protocols"]["A"]["protocol_version"]):
            raise ExternalError("Prediction protocol identities contradict model contract")
        if lock_dir is not None:
            locked = validate_lock(lock_dir)
            _validate_code(manifest["code"], locked_code=locked["manifest"]["code"])
            if created_at < _timestamp(locked["manifest"]["locked_at"], "protocol lock"):
                raise ExternalError("Prediction creation predates its protocol lock")
            if (manifest["protocol_lock_id"] != locked["manifest"]["lock_id"]
                    or manifest["protocol_sha256"] != locked["manifest"]["protocol_sha256"]
                    or contract != protocol_training_contract(locked["protocol"])):
                raise ExternalError("Prediction was produced for a different protocol lock")
            if any(manifest["model_sha256"][role] != locked["manifest"]["artifacts"][role + "_model"]["sha256"]
                   for role in ("primary", "reference")):
                raise ExternalError("Prediction models are not the frozen lock models")
        predictions = _frame(_bytes(root / "predictions.csv"))
        expected = _prediction_table(features, documents, manifest["protocol_lock_id"])
        _same_table(predictions, expected, "Prediction")
        if manifest.get("n_epochs") != len(features) or manifest.get("n_predictions") != len(expected):
            raise ExternalError("Prediction counts contradict epoch coverage")
        return {"manifest": manifest, "features": features, "predictions": predictions,
                "models": documents, "input_manifest": inputs}
    except ExternalError:
        raise
    except (KeyError, TypeError, ValueError, OSError, AttributeError) as exc:
        raise ExternalError(f"Invalid external prediction evidence: {exc}") from exc


def _load_labels(root, prediction):
    root = Path(root)
    if root.is_symlink() or {p.name for p in root.iterdir()} != {"labels.csv", "labels_manifest.json"}:
        raise ExternalError("Independent labels directory has extra or missing files")
    manifest = _json(_bytes(root / "labels_manifest.json"))
    _validate_id(manifest, "labels_id", "eeg-external-labels-1")
    labels = _frame(_bytes(root / "labels.csv"))
    if (list(labels.columns) != ["epoch_id", "true_label"] or labels.epoch_id.duplicated().any()
            or set(labels.epoch_id) != set(prediction["features"].epoch_id)
            or not labels.true_label.isin([0, 1]).all()):
        raise ExternalError("Independent labels require exact unique epoch coverage and binary labels")
    if (manifest.get("class_mapping") != CLASS_MAPPING or manifest.get("input_id") != prediction["manifest"]["input_id"]
            or manifest.get("cohort_manifest_sha256") != prediction["manifest"]["cohort_manifest_sha256"]
            or manifest.get("labels") != _csv_descriptor(root / "labels.csv", labels)):
        raise ExternalError("Independent labels class mapping, source binding or hash mismatch")
    return manifest, labels


def _score_tables(prediction, labels, enrollment, audit, spec):
    features = prediction["features"]
    truth = labels.set_index("epoch_id").true_label
    people = {person["subject_id"]: person for person in enrollment["participants"]}
    eligibility = {person["subject_id"]: person for person in audit["participants"]}
    for (subject, session), rows in features.groupby(["subject_id", "session_id"]):
        if subject not in people:
            raise ExternalError("Predictions contain a subject outside the enrollment ledger")
        sessions = {s["session_id"]: s for s in people[subject]["sessions"]}
        if session not in sessions or not sessions[session]["qa"]["passed"]:
            raise ExternalError("Predicted features belong to an unknown or QA-rejected session")
    a = spec["protocols"]["A"]
    comparisons = [(a["primary_comparison"], "primary", "Rest1", "Type1"),
                   (a["secondary_comparison"], "secondary", "Rest2", "Type2")]
    rows = []
    for subject in sorted(people):
        selected = eligibility[subject]["selected_session_id"]
        session = next((s for s in people[subject]["sessions"] if s["session_id"] == selected), None)
        for comparison, endpoint, rest, task in comparisons:
            frame = features.loc[(features.subject_id == subject) & (features.session_id == selected) & (features.comparison == comparison)]
            actual = truth.reindex(frame.epoch_id).to_numpy(int)
            counts = (int(np.sum(actual == 0)), int(np.sum(actual == 1)))
            required = (session["rest_eligible"][rest], session["eligible_correct"][task]) if session else (0, 0)
            if counts != required:
                raise ExternalError("Labels/predictions do not cover every eligible rest/correct-task epoch in enrollment")
            evaluable = bool(eligibility[subject][endpoint + "_evaluable"])
            if evaluable and (counts[0] != spec["eligibility"]["planned_rest_epochs_per_block"]
                              or not spec["eligibility"]["min_epochs_per_class"] <= counts[1] <= spec["eligibility"]["max_epochs_per_class"]):
                raise ExternalError("Endpoint eligibility contradicts actual epoch counts")
            for role in ("primary", "reference"):
                predictions = prediction["predictions"]
                pred = predictions.loc[(predictions.subject_id == subject) & (predictions.session_id == selected)
                    & (predictions.comparison == comparison) & (predictions.model_role == role)].set_index("epoch_id")
                values = pred.predicted_label.reindex(frame.epoch_id).to_numpy(int)
                tn, fp = int(np.sum((actual == 0) & (values == 0))), int(np.sum((actual == 0) & (values == 1)))
                fn, tp = int(np.sum((actual == 1) & (values == 0))), int(np.sum((actual == 1) & (values == 1)))
                rows.append({"subject_id": subject, "session_id": selected or "", "comparison": comparison,
                    "model_role": role, "evaluable": evaluable, "n_rest": counts[0], "n_task": counts[1],
                    "tn": tn, "fp": fp, "fn": fn, "tp": tp,
                    "balanced_accuracy": .5 * (tn / counts[0] + tp / counts[1]) if evaluable else np.nan,
                    "accuracy": (tn + tp) / len(actual) if evaluable else np.nan})
    participants = pd.DataFrame(rows).sort_values(["subject_id", "comparison", "model_role"]).reset_index(drop=True)
    summaries, deltas = [], []
    for comparison, endpoint, _, _ in comparisons:
        pair = participants.loc[(participants.comparison == comparison) & participants.evaluable]
        for role in ("primary", "reference"):
            table = pair.loc[pair.model_role == role]
            summaries.append({"analysis_id": ANALYSIS_ID, "comparison": comparison, "model_role": role,
                "n_recruited": len(people), "n_evaluable": len(table),
                "participant_macro_balanced_accuracy": float(table.balanced_accuracy.mean()) if len(table) else np.nan,
                "participant_macro_accuracy": float(table.accuracy.mean()) if len(table) else np.nan})
        wide = pair.pivot(index="subject_id", columns="model_role", values="balanced_accuracy")
        deltas.append({"comparison": comparison, "n_pairs": len(wide),
            "mean_primary_minus_reference": float((wide.primary - wide.reference).mean()) if len(wide) else None})
    return participants, pd.DataFrame(summaries).sort_values(["comparison", "model_role"]).reset_index(drop=True), deltas


def _records(frame):
    return json.loads(frame.to_json(orient="records", double_precision=15))


SCORE_ARTIFACTS = {"labels.csv", "labels_manifest.json", "cohort.json", "enrollment.json", "participant_scores.csv", "score_summary.csv"}


def score_external(prediction_dir, labels_dir, out_dir, *, lock_dir, cohort_manifest_path, enrollment_manifest_path):
    """Authorize access to immutable predictions, then read separate labels/eligibility."""
    prediction = validate_predictions(prediction_dir, lock_dir=lock_dir)
    with new_output_directory(out_dir) as stage:
        commit = {key: prediction["manifest"][key] for key in COMMIT_FIELDS}
        context = authorize_external_score(lock_dir, cohort_manifest_path, enrollment_manifest_path, prediction_commit=commit)
        # Everything above this boundary is label blind. Failures below retain
        # the committed exposure event, so retry cannot choose another model.
        label_manifest, labels = _load_labels(labels_dir, prediction)
        participants, summary, deltas = _score_tables(prediction, labels, context["enrollment"], context["audit"], context["locked"]["protocol"])
        for name in ("labels.csv", "labels_manifest.json"):
            (stage / name).write_bytes(_bytes(Path(labels_dir) / name))
        (stage / "cohort.json").write_bytes(context["event"]["cohort_manifest_text"].encode("utf-8"))
        (stage / "enrollment.json").write_bytes(context["event"]["enrollment_manifest_text"].encode("utf-8"))
        participants.to_csv(stage / "participant_scores.csv", index=False)
        summary.to_csv(stage / "score_summary.csv", index=False)
        manifest = _signed({"schema_version": "eeg-external-score-1", "status": "complete", "analysis_id": ANALYSIS_ID,
            "report_title": TITLE, "created_at": _now(), "prediction_id": prediction["manifest"]["prediction_id"],
            "protocol_lock_id": prediction["manifest"]["protocol_lock_id"], "labels_id": label_manifest["labels_id"],
            "outcome_event_id": context["event"]["event_id"], "summary": _records(summary), "paired_deltas": deltas,
            "enrollment_audit": context["audit"], "inference_status": "point_estimates_only_participant_uncertainty_and_publication_in_issue_12",
            "artifacts": _inventory(stage, SCORE_ARTIFACTS)}, "score_id")
        _write_json(stage / "score_manifest.json", manifest)
        validate_external_score(stage, prediction_dir=prediction_dir, lock_dir=lock_dir)
    return manifest


def validate_external_score(score_dir, *, prediction_dir, lock_dir):
    """Recompute score arithmetic from its frozen labels and enrollment evidence."""
    try:
        root = Path(score_dir)
        manifest = _json(_bytes(root / "score_manifest.json"))
        _validate_id(manifest, "score_id", "eeg-external-score-1")
        _check_inventory(root, manifest["artifacts"], SCORE_ARTIFACTS, "score_manifest.json")
        if manifest.get("status") != "complete" or manifest.get("analysis_id") != ANALYSIS_ID or manifest.get("report_title") != TITLE:
            raise ExternalError("External score status/analysis contract mismatch")
        prediction = validate_predictions(prediction_dir, lock_dir=lock_dir)
        locked = validate_lock(lock_dir)
        if manifest["prediction_id"] != prediction["manifest"]["prediction_id"] or manifest["protocol_lock_id"] != locked["manifest"]["lock_id"]:
            raise ExternalError("Score prediction/protocol binding mismatch")
        events = [e for e in locked["events"] if e["event_id"] == manifest["outcome_event_id"]]
        if len(events) != 1:
            raise ExternalError("Score has no matching outcome-access event")
        event = events[0]
        if (_timestamp(manifest["created_at"], "score creation") < _timestamp(event["recorded_at"], "outcome access")
                or manifest.get("inference_status") != "point_estimates_only_participant_uncertainty_and_publication_in_issue_12"):
            raise ExternalError("External score chronology/reporting contract mismatch")
        if event.get("prediction_commit") != {key: prediction["manifest"][key] for key in COMMIT_FIELDS}:
            raise ExternalError("Score event does not authorize these predictions")
        if (event["enrollment_manifest_sha256"] != sha256_file(root / "enrollment.json")
                or event["cohort_manifest_sha256"] != sha256_file(root / "cohort.json")):
            raise ExternalError("Score enrollment/cohort differs from the authorized event")
        labels_meta = _json(_bytes(root / "labels_manifest.json"))
        _validate_id(labels_meta, "labels_id", "eeg-external-labels-1")
        labels = _frame(_bytes(root / "labels.csv"))
        if (labels_meta.get("labels") != _csv_descriptor(root / "labels.csv", labels)
                or labels_meta.get("class_mapping") != CLASS_MAPPING or labels_meta.get("labels_id") != manifest["labels_id"]
                or labels_meta.get("input_id") != prediction["manifest"]["input_id"]
                or labels_meta.get("cohort_manifest_sha256") != prediction["manifest"]["cohort_manifest_sha256"]
                or list(labels.columns) != ["epoch_id", "true_label"] or labels.epoch_id.duplicated().any()
                or set(labels.epoch_id) != set(prediction["features"].epoch_id) or not labels.true_label.isin([0, 1]).all()):
            raise ExternalError("Frozen score labels are inconsistent")
        enrollment = _json(_bytes(root / "enrollment.json"))
        audit = assess_enrollment(enrollment, locked["protocol"])
        if not audit["valid"] or audit != manifest["enrollment_audit"]:
            raise ExternalError("Score eligibility contradicts independent enrollment evidence")
        participants, summary, deltas = _score_tables(prediction, labels, enrollment, audit, locked["protocol"])
        _same_table(_frame(_bytes(root / "participant_scores.csv")), participants, "Participant score")
        _same_table(_frame(_bytes(root / "score_summary.csv")), summary, "Score summary")
        if manifest["summary"] != _records(summary) or manifest["paired_deltas"] != deltas:
            raise ExternalError("Score manifest contradicts recomputed point estimates")
        return manifest
    except ExternalError:
        raise
    except (KeyError, TypeError, ValueError, OSError, AttributeError) as exc:
        raise ExternalError(f"Invalid external scoring evidence: {exc}") from exc
