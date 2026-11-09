"""Development-only SFS/LDA fitting and portable, predict-only JSON artifacts.

The exported binary LDA decision rule is its fitted scaler followed by fitted
coefficients/intercept. Loading never unpickles Python objects and never fits an
estimator. The model records its actual training evidence; study authorization
is a separate protocol-lock decision.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
from itertools import combinations
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd
from sklearn.model_selection import LeaveOneGroupOut

from .artifacts import new_output_directory
from .environment import capture_environment, validate_environment_evidence, verify_reference_environment
from .extensions import merge_baseline_and_extensions
from .models import lda_pipeline
from .protocol import protocol_training_contract, protocol_training_contract_sha256
from .provenance import code_identity, fingerprint
from .schema import META_COLUMNS, load_feature_table, validate_feature_table
from .sfs import forward_select_lda

MODEL_SCHEMA = "eeg-fixed-lda-model-1"
BUNDLE_SCHEMA = "eeg-development-model-bundle-1"
ANALYSIS_ID = "fixed_model_external"
ROLES = {"primary": "alpha_plv_sfs_lda", "reference": "bp_coh_sfs_lda"}
CLASS_MAPPING = {"0": "Rest", "1": "MentalArithmetic"}
SCALER_PARAMS = {"copy": True, "with_mean": True, "with_std": True}
LDA_PARAMS = {"covariance_estimator": None, "n_components": None, "priors": None,
              "shrinkage": None, "solver": "svd", "store_covariance": False, "tol": 0.0001}
SHA = re.compile(r"[0-9a-f]{64}\Z")
BANDS = ["delta", "theta", "alpha", "beta_low", "beta_high", "gamma"]
EPOCH_FIELDS = ["cohort_id", "subject_id", "session_id", "recording_id", "epoch_id", "comparison", "class_label"]


class FixedModelError(ValueError):
    """Model input or stored training evidence violates its declared contract."""


def _json_bytes(value) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n").encode("utf-8")


def _read_json_bytes(payload: bytes, label: str) -> dict:
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise FixedModelError(f"Duplicate {label} JSON field: {key}")
            result[key] = value
        return result
    try:
        value = json.loads(payload.decode("utf-8"), object_pairs_hook=pairs,
            parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))
        if not isinstance(value, dict):
            raise ValueError("not an object")
        return value
    except (UnicodeError, ValueError, TypeError) as exc:
        raise FixedModelError(f"Invalid {label} JSON: {exc}") from exc


def _sha(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _strings(value, label, *, ordered=False):
    if (not isinstance(value, list) or not value or len(set(value)) != len(value)
            or any(not isinstance(x, str) or not x.strip() or x != x.strip() for x in value)):
        raise FixedModelError(f"{label} must contain unique literal string IDs")
    if ordered and value != sorted(value):
        raise FixedModelError(f"{label} must use canonical ordering")
    return value


def _digest(value, label):
    if not isinstance(value, str) or not SHA.fullmatch(value):
        raise FixedModelError(f"{label} must be a SHA-256 identity")
    return value


def _array(value, shape, label):
    array = np.asarray(value, dtype=float)
    if array.shape != shape or not np.isfinite(array).all():
        raise FixedModelError(f"Invalid finite numeric shape for {label}; expected {shape}")
    return array


def validate_feature_schema(schema: dict) -> None:
    channels = ["FP1", "FP2", "F7", "F3", "Fz", "F4", "F8"]
    expected = {
        "schema_version": "eeg-confirmatory-feature-schema-1", "feature_table_schema_version": "2.0",
        "primary_learner_id": ROLES["primary"], "reference_learner_id": ROLES["reference"],
        "class_mapping": CLASS_MAPPING, "analysis_channels": channels,
        "sampling_rate_hz": 500, "epoch_seconds": 5.0,
        "identity_policy": "stable_subject_across_all_cohorts_and_sessions", "feature_order_policy": "exact_frozen_order",
        "primary_features": [f"PLV__{a}__{b}__alpha" for a, b in combinations(channels, 2)],
        "reference_features": [f"BP__{channel}__{band}" for channel in channels for band in BANDS]
            + [f"COH__{a}__{b}__{band}" for a, b in combinations(channels, 2) for band in BANDS],
    }
    if not isinstance(schema, dict) or any(schema.get(key) != value for key, value in expected.items()):
        raise FixedModelError("Feature schema, class mapping or exact feature order is inconsistent")


def _validate_contract(contract, cfg, schema):
    if not isinstance(contract, dict) or contract.get("schema_version") != "eeg-model-training-contract-1":
        raise FixedModelError("Model training protocol contract is missing")
    spec = contract.get("protocol")
    if not isinstance(spec, dict) or protocol_training_contract(spec) != contract:
        raise FixedModelError("Training protocol contract is not its canonical projection")
    if spec.get("endpoints", {}).get("primary_model") != ROLES["primary"] or spec.get("endpoints", {}).get("reference_model") != ROLES["reference"]:
        raise FixedModelError("Only the prespecified primary/reference LDA learners are supported")
    if (spec.get("eligibility", {}).get("correct_only") is not True
            or spec["eligibility"].get("analysis_eligible_required") is not True
            or spec["eligibility"].get("target_fit_ICA_ASR") != "disabled"):
        raise FixedModelError("The fixed model requires the declared correct-only, zero-calibration policy")
    if (spec.get("freeze", {}).get("model_training_scope") != "development_only"
            or spec["freeze"].get("environment_policy") != "verified_reference_environment"):
        raise FixedModelError("Development/reference environment policies cannot be bypassed")
    for key in ("sampling_rate_hz", "epoch_seconds", "analysis_channels"):
        if cfg.get(key) != schema[key] or cfg[key] != spec["acquisition"][key]:
            raise FixedModelError(f"Analysis/schema/protocol acquisition mismatch: {key}")
    if cfg.get("raw_channels") != spec["acquisition"].get("raw_channels"):
        raise FixedModelError("Analysis/protocol raw channel order differs")
    if cfg.get("expected_epochs_per_class", "missing") is not None:
        raise FixedModelError("Prospective analysis must use its declared bounded-count eligibility")
    for key in ("sample_period_tolerance_s", "clock_tolerance_s", "flat_std_epsilon"):
        if cfg.get("ingestion", {}).get(key) != spec["eligibility"]["qc"][key]:
            raise FixedModelError(f"Analysis/protocol QA mismatch: {key}")
    if cfg.get("ingestion", {}).get("marker_policy") != spec["acquisition"]["marker_policy"]:
        raise FixedModelError("Analysis/protocol marker policy differs")
    if cfg.get("sfs", {}).get("selector") != "LDA":
        raise FixedModelError("Only development-only LDA SFS is allowed")
    cap = cfg["sfs"].get("max_features")
    if cap is not None and (type(cap) is not int or cap < 1):
        raise FixedModelError("SFS cap must be null or a positive integer")
    improvement = cfg["sfs"].get("min_improvement")
    if type(improvement) not in (int, float) or not np.isfinite(improvement) or improvement < 0:
        raise FixedModelError("SFS improvement must be a finite nonnegative number")
    comparisons = cfg.get("comparisons")
    if (not isinstance(comparisons, list) or not comparisons or any(not isinstance(c, dict) for c in comparisons)
            or len({c.get("name") for c in comparisons}) != len(comparisons)):
        raise FixedModelError("Declared comparisons must be unique")
    declared = [c.get("name") for c in comparisons]
    allowed = [spec["protocols"]["A"][key] for key in ("primary_comparison", "secondary_comparison")]
    if declared != allowed or declared != ["rest1_vs_type1", "rest2_vs_type2"]:
        raise FixedModelError("Development must fit every declared primary/secondary comparison in protocol order")
    return spec, declared


def _validate_comparison(name, state, model):
    if not isinstance(state, dict):
        raise FixedModelError("Comparison fitted state is missing")
    metadata = state.get("training_epochs")
    if not isinstance(metadata, list) or not metadata:
        raise FixedModelError("Training epoch membership is missing")
    ids = _strings([r.get("epoch_id") for r in metadata], "training epoch IDs", ordered=True)
    epoch_rows = {}
    for row in metadata:
        if set(row) != set(EPOCH_FIELDS) or row["comparison"] != name or type(row["class_label"]) is not int or row["class_label"] not in (0, 1):
            raise FixedModelError("Training epoch metadata/class mapping is invalid")
        for field in EPOCH_FIELDS[:-1]:
            if not isinstance(row[field], str) or not row[field].strip() or row[field] != row[field].strip():
                raise FixedModelError("Training epoch identity must preserve literal strings")
        epoch_rows[row["epoch_id"]] = row
    subjects = sorted({r["subject_id"] for r in metadata})
    if subjects != model["training_subject_ids"] or state.get("training_subject_ids") != subjects:
        raise FixedModelError("Comparison training subjects differ from the development set")
    for subject in subjects:
        labels = [r["class_label"] for r in metadata if r["subject_id"] == subject]
        if min(labels.count(0), labels.count(1)) < 2:
            raise FixedModelError("Each development participant needs both classes with at least two epochs")
    inner = state.get("inner_splits")
    if not isinstance(inner, list) or len(inner) != len(subjects):
        raise FixedModelError("Development SFS requires one validation fold per training subject")
    for number, (split, subject) in enumerate(zip(inner, subjects), start=1):
        valid = [i for i in ids if epoch_rows[i]["subject_id"] == subject]
        train = [i for i in ids if epoch_rows[i]["subject_id"] != subject]
        expected = {"fold": number, "train_epoch_ids": train, "validation_epoch_ids": valid,
                    "train_subject_ids": [s for s in subjects if s != subject], "validation_subject_ids": [subject]}
        if split != expected:
            raise FixedModelError("Inner split membership does not match grouped development epochs")
    indices, names = state.get("selected_indices"), state.get("selected_names")
    if (not isinstance(indices, list) or not indices or len(set(indices)) != len(indices)
            or any(type(i) is not int or not 0 <= i < len(model["feature_order"]) for i in indices)
            or names != [model["feature_order"][i] for i in indices]):
        raise FixedModelError("Selected feature indices/names disagree with the exact model schema")
    cap = model["analysis_config"]["sfs"].get("max_features")
    if cap is not None and len(indices) > cap:
        raise FixedModelError("Selection exceeds its declared SFS cap")
    diagnostic = state.get("selection_diagnostics", {})
    if diagnostic.get("interpretation") != "selection_diagnostic_not_selection_independent_performance":
        raise FixedModelError("Selection accuracy must be marked as a selection diagnostic")
    score = diagnostic.get("sfs_inner_accuracy")
    if type(score) not in (int, float) or not np.isfinite(score) or not 0 <= score <= 1:
        raise FixedModelError("Invalid SFS selection diagnostic")
    path = diagnostic.get("path")
    if (not isinstance(path, list) or not path or [r.get("step") for r in path] != list(range(1, len(path) + 1))
            or [r.get("candidate_index") for r in path if r.get("accepted") is True] != indices):
        raise FixedModelError("SFS path does not match selected features")
    for row in path:
        index = row.get("candidate_index")
        if (type(index) is not int or not 0 <= index < len(model["feature_order"])
                or row.get("candidate_name") != model["feature_order"][index]
                or type(row.get("accepted")) is not bool
                or type(row.get("cv_accuracy")) not in (int, float) or not 0 <= row["cv_accuracy"] <= 1):
            raise FixedModelError("Invalid SFS diagnostic path row")
    if (not path[0]["accepted"] or any(not row["accepted"] for row in path[:-1])
            or len({row["candidate_index"] for row in path}) != len(path)
            or len(path) not in {len(indices), len(indices) + 1}):
        raise FixedModelError("SFS path must be a unique accepted prefix with at most one final rejection")
    accepted_scores = [row["cv_accuracy"] for row in path if row["accepted"]]
    if score != accepted_scores[-1] or path[0].get("improvement") is not None:
        raise FixedModelError("SFS final score/initial improvement contradicts its path")
    previous = path[0]["cv_accuracy"]
    threshold = float(model["analysis_config"]["sfs"]["min_improvement"])
    for row in path[1:]:
        improvement = row.get("improvement")
        expected_improvement = row["cv_accuracy"] - previous
        if (type(improvement) not in (int, float) or not np.isfinite(improvement)
                or not np.isclose(improvement, expected_improvement, rtol=0, atol=1e-14)
                or row["accepted"] != (row["cv_accuracy"] > previous + threshold + 1e-15)):
            raise FixedModelError("SFS acceptance/improvement contradicts the stopping policy")
        if row["accepted"]:
            previous = row["cv_accuracy"]
    if state.get("scaler_params") != SCALER_PARAMS or state.get("classifier_params") != LDA_PARAMS:
        raise FixedModelError("Stored classifier/scaler parameters differ from the supported fitted learner")
    count = len(indices)
    scaler = state.get("scaler", {})
    if type(scaler.get("n_features_in")) is not int or scaler["n_features_in"] != count or type(scaler.get("n_samples_seen")) is not int or scaler["n_samples_seen"] != len(ids):
        raise FixedModelError("Scaler fitted dimensions/sample count differ from training membership")
    mean = _array(scaler["mean"], (count,), "scaler mean")
    variance = _array(scaler["var"], (count,), "scaler variance")
    scale = _array(scaler["scale"], (count,), "scaler scale")
    if np.any(variance < 0) or np.any(scale <= 0):
        raise FixedModelError("Scaler variance and scale must be nonnegative/positive")
    # StandardScaler's float64 constant-feature bound; generated scale is 1 for
    # an effectively constant column, otherwise the square root of variance.
    eps = np.finfo(np.float64).eps
    samples = scaler["n_samples_seen"]
    constant = variance <= samples * eps * variance + (samples * mean * eps) ** 2
    expected_scale = np.where(constant, 1.0, np.sqrt(variance))
    if not np.allclose(scale, expected_scale, rtol=1e-12, atol=1e-14):
        raise FixedModelError("Scaler scale contradicts its fitted variance")
    classifier = state.get("classifier", {})
    if classifier.get("classes") != [0, 1] or type(classifier.get("n_features_in")) is not int or classifier["n_features_in"] != count:
        raise FixedModelError("Classifier class mapping or fitted dimensions are invalid")
    _array(classifier["coef"], (1, count), "LDA coefficients")
    _array(classifier["intercept"], (1,), "LDA intercept")
    means = _array(classifier["means"], (2, count), "LDA class means")
    priors = _array(classifier["priors"], (2,), "LDA priors")
    expected_priors = np.array([sum(r["class_label"] == c for r in metadata) / len(ids) for c in (0, 1)])
    if not np.allclose(priors, expected_priors, atol=1e-14, rtol=0):
        raise FixedModelError("LDA priors disagree with development labels")
    scalings = np.asarray(classifier["scalings"], dtype=float)
    if scalings.shape not in {(count, 0), (count, 1)} or not np.isfinite(scalings).all():
        raise FixedModelError("LDA discriminant scaling shape is invalid")
    xbar = _array(classifier["xbar"], (count,), "LDA global mean")
    if not np.allclose(xbar, priors @ means, rtol=1e-12, atol=1e-14):
        raise FixedModelError("LDA global mean contradicts class means and priors")
    explained = classifier["explained_variance_ratio"]
    if scalings.shape[1] == 0:
        if explained != [None]:
            raise FixedModelError("A rank-zero discriminant must mark explained variance undefined")
    else:
        ratio = _array(explained, (1,), "LDA explained variance")
        if np.any(ratio < 0) or np.any(ratio > 1 + 1e-12):
            raise FixedModelError("LDA explained variance ratio is outside its valid range")
    # These are the recorded SVD-LDA equations, evaluated without fitting.
    centers = (means - xbar) @ scalings
    class_coef = centers @ scalings.T
    class_intercept = -.5 * np.sum(centers ** 2, axis=1) + np.log(priors) - xbar @ class_coef.T
    if (not np.allclose(classifier["coef"], np.diff(class_coef, axis=0), rtol=1e-12, atol=1e-14)
            or not np.allclose(classifier["intercept"], np.diff(class_intercept), rtol=1e-12, atol=1e-14)):
        raise FixedModelError("LDA decision rule contradicts its fitted discriminant state")


def validate_model(model: dict, *, expected_role=None, expected_feature_schema=None,
                   expected_training_contract=None, expected_training_subject_ids=None) -> dict:
    """Validate archived executable state without inspecting the current runtime."""
    try:
        if not isinstance(model, dict) or model.get("schema_version") != MODEL_SCHEMA:
            raise FixedModelError("Unsupported executable fixed-model schema")
        payload = dict(model)
        model_id = payload.pop("model_id", None)
        if model_id != fingerprint(payload):
            raise FixedModelError("Fixed model identity/hash mismatch")
        role = model.get("model_role")
        if role not in ROLES or model.get("learner_id") != ROLES[role] or model.get("analysis_id") != ANALYSIS_ID:
            raise FixedModelError("Fixed model role/learner/analysis identity is invalid")
        if model.get("execution_scope") not in {"reference_development_fit", "software_validation"}:
            raise FixedModelError("Model execution scope must be explicit")
        schema, cfg = model["feature_schema"], model["analysis_config"]
        for label, expected_document in (("analysis_config", cfg), ("feature_schema", schema)):
            raw = model[label + "_text"].encode("utf-8")
            if _sha(raw) != model[label + "_sha256"] or _read_json_bytes(raw, label) != expected_document:
                raise FixedModelError(f"Embedded {label} bytes/hash/content are inconsistent")
        source_protocol = model["source_protocol_text"].encode("utf-8")
        if _sha(source_protocol) != model["source_protocol_sha256"]:
            raise FixedModelError("Source protocol bytes/hash are inconsistent")
        if protocol_training_contract(_read_json_bytes(source_protocol, "source protocol")) != model["protocol_training_contract"]:
            raise FixedModelError("Source protocol differs from the frozen training contract")
        validate_feature_schema(schema)
        contract = model["protocol_training_contract"]
        spec, comparisons = _validate_contract(contract, cfg, schema)
        if model.get("protocol_training_contract_sha256") != fingerprint(contract):
            raise FixedModelError("Model protocol training contract hash mismatch")
        if any(model.get(key) != spec[key] for key in ("protocol_id", "protocol_version")):
            raise FixedModelError("Model protocol identity differs from its training contract")
        if model.get("class_mapping") != CLASS_MAPPING or model.get("feature_order") != schema[role + "_features"]:
            raise FixedModelError("Model class mapping/feature order differs from the frozen schema")
        _strings(model["training_subject_ids"], "training subjects", ordered=True)
        if len(model["training_subject_ids"]) < 3:
            raise FixedModelError("Development SFS requires at least three independent subjects")
        for field in ("source_protocol_sha256", "analysis_config_sha256", "feature_schema_sha256", "environment_id", "environment_lock_id"):
            _digest(model[field], field)
        provenance = model["provenance"]
        for field in ("development_manifest_sha256", "training_data_sha256", "eligibility_rule_sha256", "environment_sha256"):
            _digest(provenance[field], field)
        if provenance["eligibility_rule_sha256"] != fingerprint(spec["eligibility"]):
            raise FixedModelError("Model eligibility identity differs from the training contract")
        for field in ("analysis_config_sha256", "feature_schema_sha256", "environment_id", "environment_lock_id"):
            if provenance.get(field) != model[field]:
                raise FixedModelError("Model provenance contains inconsistent identities")
        if provenance.get("fitted_on_development_only") is not True or provenance.get("target_fit_performed") is not False:
            raise FixedModelError("Fixed model cannot claim target fitting")
        code = model["code"]
        if (not isinstance(code, dict) or not isinstance(code.get("source_files"), dict) or not code["source_files"]
                or code.get("working_source_sha256") != fingerprint(code["source_files"])):
            raise FixedModelError("Model source code inventory is inconsistent")
        if not isinstance(model.get("comparisons"), dict) or set(model["comparisons"]) != set(comparisons):
            raise FixedModelError("Fitted model comparison coverage differs from the configured protocol")
        for name, state in model["comparisons"].items():
            _validate_comparison(name, state, model)
        if expected_role is not None and role != expected_role:
            raise FixedModelError("Model role differs from the required role")
        if expected_feature_schema is not None and schema != expected_feature_schema:
            raise FixedModelError("Model feature schema differs from the expected schema")
        if expected_training_contract is not None and contract != expected_training_contract:
            raise FixedModelError("Model was fitted for a different protocol training contract")
        if expected_training_subject_ids is not None and model["training_subject_ids"] != sorted(expected_training_subject_ids):
            raise FixedModelError("Model training subjects differ from the expected development set")
        return deepcopy(model)
    except FixedModelError:
        raise
    except (KeyError, TypeError, ValueError, AttributeError) as exc:
        raise FixedModelError(f"Invalid or incomplete fixed model: {exc}") from exc


class FixedLDAModel:
    """Predict-only fitted numerical state. No fit/partial_fit API exists."""
    def __init__(self, document):
        self._document = validate_model(document)

    @property
    def document(self):
        return deepcopy(self._document)

    @property
    def model_id(self):
        return self._document["model_id"]

    @property
    def feature_order(self):
        return tuple(self._document["feature_order"])

    def decision_function(self, features: pd.DataFrame, *, comparison: str) -> np.ndarray:
        if not isinstance(features, pd.DataFrame) or list(features.columns) != list(self.feature_order):
            raise FixedModelError("Inference requires only the exact frozen feature columns in order")
        if comparison not in self._document["comparisons"]:
            raise FixedModelError("Unfitted comparison")
        values = features.to_numpy(dtype=float, copy=True)
        if values.shape[0] == 0 or not np.isfinite(values).all():
            raise FixedModelError("Inference requires nonempty finite feature values")
        state = self._document["comparisons"][comparison]
        selected = values[:, state["selected_indices"]]
        selected -= np.asarray(state["scaler"]["mean"])
        selected /= np.asarray(state["scaler"]["scale"])
        score = np.dot(selected, np.asarray(state["classifier"]["coef"]).T) + np.asarray(state["classifier"]["intercept"])
        if not np.isfinite(score).all():
            raise FixedModelError("Inference produced nonfinite decision values")
        return score.ravel()

    def predict(self, features: pd.DataFrame, *, comparison: str) -> np.ndarray:
        # sklearn binary LinearClassifierMixin uses strictly > 0; a tie is class0.
        return (self.decision_function(features, comparison=comparison) > 0).astype(int)


def load_fixed_model(path: str | Path, *, expected_sha256=None, **expected) -> FixedLDAModel:
    path = Path(path)
    if path.is_symlink():
        raise FixedModelError("Fixed model cannot be a symlink")
    payload = path.read_bytes()
    if expected_sha256 is not None and _sha(payload) != expected_sha256:
        raise FixedModelError("Fixed model artifact byte hash mismatch")
    document = validate_model(_read_json_bytes(payload, "fixed model"), **expected)
    return FixedLDAModel(document)


def _load_development(protocol_path):
    path = Path(protocol_path)
    if path.is_symlink():
        raise FixedModelError("Development protocol cannot be a symlink")
    source_protocol = path.read_bytes()
    source_snapshot = {path: _sha(source_protocol)}
    spec = _read_json_bytes(source_protocol, "protocol")
    contents, paths, documents = {}, {}, {}
    for role in ("development_manifest", "development_data_manifest", "analysis_config", "feature_schema"):
        entry = spec["freeze"]["artifacts"][role]
        source = Path(entry["path"])
        source = source if source.is_absolute() else path.parent / source
        if source.is_symlink():
            raise FixedModelError("Development evidence cannot be a symlink")
        payload = source.read_bytes()
        if not payload or _sha(payload) != entry["sha256"]:
            raise FixedModelError(f"Frozen development artifact hash mismatch: {role}")
        contents[role], paths[role], documents[role] = payload, source, _read_json_bytes(payload, role)
        source_snapshot[source] = _sha(payload)
    cfg, schema = documents["analysis_config"], documents["feature_schema"]
    validate_feature_schema(schema)
    contract = protocol_training_contract(spec)
    _validate_contract(contract, cfg, schema)
    dev = documents["development_manifest"]
    subjects = sorted(_strings(dev.get("subject_ids"), "development subject IDs"))
    if (dev.get("schema_version") != "eeg-development-manifest-1" or dev.get("use") != "development_only"
            or dev.get("data_sha256") != _sha(contents["development_data_manifest"])
            or dev.get("feature_schema_sha256") != _sha(contents["feature_schema"])):
        raise FixedModelError("Development manifest does not bind the actual data and schema")
    if len(subjects) < 3:
        raise FixedModelError("At least three independent development subjects are required")
    from .protocol import _dataset_entries
    entries = _dataset_entries(contents["development_data_manifest"])
    if set(entries) not in ({"baseline", "extensions"}, {"baseline", "extensions", "legacy_mapping"}):
        raise FixedModelError("Fixed model fitting needs exactly baseline/extensions and optional explicit legacy_mapping")
    sources, source_payloads = {}, {}
    for name, entry in entries.items():
        source = Path(entry["path"])
        source = source if source.is_absolute() else paths["development_data_manifest"].parent / source
        if source.is_symlink():
            raise FixedModelError(f"Development source cannot be a symlink: {name}")
        payload = source.read_bytes()
        if _sha(payload) != entry["sha256"]:
            raise FixedModelError(f"Development source hash mismatch: {name}")
        sources[name], source_payloads[name] = source, payload
        source_snapshot[source] = _sha(payload)
    mapping = _read_json_bytes(source_payloads["legacy_mapping"], "legacy mapping") if "legacy_mapping" in sources else None
    frames = {}
    for name in ("baseline", "extensions"):
        frame = load_feature_table(sources[name], legacy_manifest=mapping)
        if frame.attrs.get("source_feature_sha256") != entries[name]["sha256"]:
            raise FixedModelError("Development source changed while loading")
        if sorted(frame.subject_id.unique()) != subjects or set(frame.cohort_id) != {dev["cohort_id"]}:
            raise FixedModelError("Feature subjects/cohort differ from the declared development set")
        frames[name] = frame
    frame = merge_baseline_and_extensions(frames["baseline"], frames["extensions"])
    validate_feature_table(frame)
    raw = frame.loc[frame.identity_origin == "raw_metadata"]
    if not raw.empty:
        acquisition = spec["acquisition"]
        if (set(raw.units) != {acquisition["units"]}
                or set(raw.reference) != {acquisition["reference"]}
                or any(json.loads(value) != acquisition["raw_channels"] for value in raw.channel_order)
                or not ((raw.end_sample - raw.start_sample) ==
                        acquisition["sampling_rate_hz"] * acquisition["epoch_seconds"]).all()):
            raise FixedModelError("Raw development acquisition metadata differ from the frozen contract")
    # Legacy adaptation deliberately retains unknown raw acquisition facts.
    # Its explicit mapping and feature bytes are bound above, without upgrading
    # that historical evidence into a raw QA/alignment assertion.
    for role in ROLES:
        names = schema[role + "_features"]
        if [column for column in frame if column in names] != names:
            raise FixedModelError(f"Development feature order/schema mismatch for {role}")
    configured = [c["name"] for c in cfg["comparisons"]]
    if set(frame.comparison) != set(configured):
        raise FixedModelError("Development data must contain every and only configured comparison")
    for comparison, stages in (("rest1_vs_type1", ["Rest1", "Type1"]), ("rest2_vs_type2", ["Rest2", "Type2"])):
        for label in (0, 1):
            subset = frame.loc[(frame.comparison == comparison) & (frame.class_label == label)]
            if set(subset.class_name) != {CLASS_MAPPING[str(label)]} or set(subset.stage) != {stages[label]}:
                raise FixedModelError("Development class mapping/stage disagrees with the declared comparison")
    frame = frame.sort_values("epoch_id", kind="stable").reset_index(drop=True)
    contents["source_protocol"] = source_protocol
    return spec, contents, documents, frame, _sha(source_protocol), source_snapshot


def _verify_development_sources(source_snapshot):
    for source, expected in source_snapshot.items():
        try:
            if source.is_symlink() or _sha(source.read_bytes()) != expected:
                raise FixedModelError("Development source changed during fitting")
        except OSError as exc:
            raise FixedModelError("Development source disappeared during fitting") from exc


def _fit_comparison(frame, names, cfg):
    frame = frame.sort_values("epoch_id", kind="stable").reset_index(drop=True)
    X, y = frame[names].to_numpy(float), frame.class_label.to_numpy(int)
    groups = frame.subject_id.to_numpy()
    subjects = sorted(set(groups))
    for subject in subjects:
        labels = y[groups == subject]
        if min(np.count_nonzero(labels == 0), np.count_nonzero(labels == 1)) < 2:
            raise FixedModelError("Every development subject/comparison needs both classes with at least two epochs")
    splits = list(LeaveOneGroupOut().split(X, y, groups))
    selected = forward_select_lda(X, y, names, splits, max_features=cfg["sfs"].get("max_features"),
        min_improvement=float(cfg["sfs"].get("min_improvement", 0.0)))
    fitted = lda_pipeline().fit(X[:, selected.selected_indices], y)
    scaler, classifier = fitted.named_steps["scaler"], fitted.named_steps["clf"]
    path = []
    for row in selected.path.to_dict("records"):
        path.append({"step": int(row["step"]), "candidate_index": int(row["candidate_index"]),
            "candidate_name": str(row["candidate_name"]), "cv_accuracy": float(row["cv_accuracy"]),
            "improvement": None if pd.isna(row["improvement"]) else float(row["improvement"]),
            "accepted": int(row["candidate_index"]) in selected.selected_indices})
    ids = frame.epoch_id.to_numpy()
    state = {"training_subject_ids": subjects, "training_epochs": json.loads(frame[EPOCH_FIELDS].to_json(orient="records")),
        "inner_splits": [{"fold": number, "train_epoch_ids": ids[train].tolist(), "validation_epoch_ids": ids[valid].tolist(),
            "train_subject_ids": sorted(set(groups[train])), "validation_subject_ids": sorted(set(groups[valid]))}
            for number, (train, valid) in enumerate(splits, start=1)],
        "selected_indices": selected.selected_indices, "selected_names": selected.selected_names,
        "selection_diagnostics": {"interpretation": "selection_diagnostic_not_selection_independent_performance",
            "sfs_inner_accuracy": selected.best_score, "path": path},
        "scaler_params": scaler.get_params(), "classifier_params": classifier.get_params(),
        "scaler": {"mean": scaler.mean_.tolist(), "var": scaler.var_.tolist(), "scale": scaler.scale_.tolist(),
            "n_features_in": int(scaler.n_features_in_), "n_samples_seen": int(scaler.n_samples_seen_)},
        "classifier": {"classes": classifier.classes_.astype(int).tolist(), "coef": classifier.coef_.tolist(),
            "intercept": classifier.intercept_.tolist(), "means": classifier.means_.tolist(), "priors": classifier.priors_.tolist(),
            "scalings": classifier.scalings_.tolist(), "xbar": classifier.xbar_.tolist(),
            "explained_variance_ratio": [float(value) if np.isfinite(value) else None for value in classifier.explained_variance_ratio_],
            "n_features_in": int(classifier.n_features_in_)}}
    return state, fitted


def fit_development(protocol_path: str | Path, out_dir: str | Path, *, require_reference_environment=True) -> dict:
    """Fit/export both frozen learners from only the declared development files.

    The nonreference mode is explicitly tagged software_validation and cannot
    authorize a production protocol lock or external study prediction.
    """
    try:
        code = code_identity()
        spec, contents, documents, frame, source_protocol_sha, source_snapshot = _load_development(protocol_path)
        cfg, schema = documents["analysis_config"], documents["feature_schema"]
        environment = verify_reference_environment() if require_reference_environment else capture_environment()
        validate_environment_evidence(environment, require_reference=require_reference_environment)
        environment_bytes = _json_bytes(environment)
        execution_scope = "reference_development_fit" if require_reference_environment else "software_validation"
        provenance = {
            "development_manifest_sha256": _sha(contents["development_manifest"]),
            "training_data_sha256": _sha(contents["development_data_manifest"]),
            "analysis_config_sha256": _sha(contents["analysis_config"]),
            "feature_schema_sha256": _sha(contents["feature_schema"]),
            "environment_sha256": _sha(environment_bytes), "environment_id": environment["environment_id"],
            "environment_lock_id": environment["lock_id"], "eligibility_rule_sha256": fingerprint(spec["eligibility"]),
            "fitted_on_development_only": True, "target_fit_performed": False,
        }
        result = {"schema_version": BUNDLE_SCHEMA, "analysis_id": ANALYSIS_ID, "execution_scope": execution_scope,
                  "protocol_id": spec["protocol_id"], "protocol_version": spec["protocol_version"],
                  "source_protocol_sha256": source_protocol_sha,
                  "protocol_training_contract_sha256": protocol_training_contract_sha256(spec),
                  "training_subject_ids": sorted(frame.subject_id.unique()), "code": code,
                  "environment_id": environment["environment_id"], "environment_lock_id": environment["lock_id"], "artifacts": {}, "input_evidence": {}}
        with new_output_directory(out_dir) as stage:
            outputs = {"environment.json": environment_bytes, **{role + ".json": payload for role, payload in contents.items()}}
            for role, learner in ROLES.items():
                names = schema[role + "_features"]
                states, fitted = {}, {}
                for comparison in [c["name"] for c in cfg["comparisons"]]:
                    subset = frame.loc[frame.comparison == comparison]
                    states[comparison], fitted[comparison] = _fit_comparison(subset, names, cfg)
                model = {"schema_version": MODEL_SCHEMA, "analysis_id": ANALYSIS_ID, "execution_scope": execution_scope,
                    "model_role": role, "learner_id": learner, "protocol_id": spec["protocol_id"], "protocol_version": spec["protocol_version"],
                    "source_protocol_sha256": source_protocol_sha, "source_protocol_text": contents["source_protocol"].decode("utf-8"),
                    "protocol_training_contract": protocol_training_contract(spec),
                    "protocol_training_contract_sha256": protocol_training_contract_sha256(spec),
                    "feature_schema": schema, "feature_schema_text": contents["feature_schema"].decode("utf-8"),
                    "feature_schema_sha256": provenance["feature_schema_sha256"],
                    "analysis_config": cfg, "analysis_config_text": contents["analysis_config"].decode("utf-8"),
                    "analysis_config_sha256": provenance["analysis_config_sha256"],
                    "class_mapping": CLASS_MAPPING, "feature_order": names, "training_subject_ids": result["training_subject_ids"],
                    "environment_id": environment["environment_id"], "environment_lock_id": environment["lock_id"],
                    "code": code, "provenance": provenance, "comparisons": states}
                model["model_id"] = fingerprint(model)
                validate_model(model)
                portable = FixedLDAModel(model)
                for comparison, estimator in fitted.items():
                    subset = frame.loc[frame.comparison == comparison].sort_values("epoch_id", kind="stable")
                    selected = states[comparison]["selected_indices"]
                    if not np.array_equal(portable.predict(subset[names], comparison=comparison), estimator.predict(subset[names].to_numpy(float)[:, selected])):
                        raise FixedModelError("Portable LDA export changed fitted sklearn predictions")
                model_bytes = _json_bytes(model)
                training = {"schema_version": "eeg-training-manifest-1", "model_artifact_schema": MODEL_SCHEMA,
                    "model_role": role, "learner_id": learner, "model_sha256": _sha(model_bytes), "model_id": model["model_id"],
                    "execution_scope": execution_scope, "protocol_training_contract_sha256": model["protocol_training_contract_sha256"],
                    "source_protocol_sha256": source_protocol_sha, "training_subject_ids": model["training_subject_ids"],
                    "code": code, **provenance}
                outputs[role + "_model.json"] = model_bytes
                outputs[role + "_training_manifest.json"] = _json_bytes(training)
            for name, payload in outputs.items():
                (stage / name).write_bytes(payload)
                role = name.removesuffix(".json")
                inventory = result["input_evidence"] if role in contents else result["artifacts"]
                inventory[role] = {"path": name, "sha256": _sha(payload), "bytes": len(payload)}
            result["bundle_id"] = fingerprint(result)
            (stage / "bundle_manifest.json").write_bytes(_json_bytes(result))
            _verify_development_sources(source_snapshot)
            if code_identity() != code:
                raise FixedModelError("Source code changed during development fitting")
        return result
    except FixedModelError:
        raise
    except (KeyError, TypeError, ValueError, AttributeError) as exc:
        raise FixedModelError(f"Development fitting blocked: {exc}") from exc
