"""Versioned epoch identities and explicit adaptation of historical feature CSVs.

Identity never depends on a feature-family CSV hash.  The same physical epoch
therefore retains its ID in BP/COH, PLV and iPLV tables.
"""
from __future__ import annotations

import hashlib
import csv
import io
import json
import os
from pathlib import Path
import re
import shutil
import tempfile
from typing import Any, Mapping

import numpy as np
import pandas as pd

SCHEMA_VERSION = "2.0"
LEGACY_META_COLUMNS = [
    "subject_id", "comparison", "class_label", "class_name", "stage",
    "epoch_index", "attempt_index", "onset_elapsed_s", "start_sample", "end_sample",
]
IDENTITY_COLUMNS = [
    "schema_version", "cohort_id", "session_id", "recording_id", "block_id", "epoch_id",
    "source_sha256", "units", "reference", "identity_origin", "protocol_id",
    "protocol_version", "channel_order", "event_key", "qc_status", "qc_reason_codes",
]
META_COLUMNS = IDENTITY_COLUMNS + LEGACY_META_COLUMNS
FEATURE_PREFIXES = ("BP__", "COH__", "ASYM_", "PLV__", "IPLV__")
MAPPING_FIELDS = (
    "cohort_id", "session_id", "recording_id", "source_sha256", "units", "reference",
    "protocol_id", "protocol_version", "channel_order",
)


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def stable_id(prefix: str, value: Any) -> str:
    return prefix + "-" + hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def hash_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise ValueError(f"{name} must be a nonempty string without surrounding whitespace")
    return value


def epoch_identity(metadata: Mapping[str, Any], row: Mapping[str, Any]) -> dict[str, str]:
    """Build stable IDs from source/session and original event/interval identity."""
    scope = {k: str(metadata[k]) for k in ("cohort_id", "subject_id", "session_id", "recording_id")}
    event = {
        **scope,
        **{k: str(row[k]) for k in ("comparison", "stage", "class_name")},
        **{k: int(row[k]) for k in ("class_label", "epoch_index", "attempt_index", "start_sample", "end_sample")},
    }
    event_key = stable_id("event", event)
    block_id = stable_id("block", {**scope, "stage": str(row["stage"])})
    epoch_id = stable_id("epoch", {
        **scope, "event_key": event_key,
        "start_sample": int(row["start_sample"]), "end_sample": int(row["end_sample"]),
    })
    return {"event_key": event_key, "block_id": block_id, "epoch_id": epoch_id}


def make_epoch_metadata(metadata: Mapping[str, Any], row: Mapping[str, Any], *,
                        qc_status: str, qc_reason_codes: list[str]) -> dict[str, Any]:
    out = {k: metadata[k] for k in (
        "cohort_id", "session_id", "recording_id", "source_sha256", "units", "reference",
        "identity_origin", "protocol_id", "protocol_version",
    )}
    out.update(schema_version=SCHEMA_VERSION, channel_order=canonical_json(metadata["channel_order"]),
               qc_status=qc_status, qc_reason_codes=canonical_json(sorted(set(qc_reason_codes))))
    out.update({k: row[k] for k in LEGACY_META_COLUMNS})
    out.update(epoch_identity({**metadata, "subject_id": row["subject_id"]}, row))
    return {k: out[k] for k in META_COLUMNS}


def validate_feature_table(frame: pd.DataFrame) -> None:
    """Validate a v2 table without guessing provenance for historical tables."""
    if frame.empty:
        raise ValueError("Feature table is empty")
    if frame.columns.duplicated().any():
        raise ValueError("Feature table contains duplicate column names")
    missing = [c for c in META_COLUMNS if c not in frame]
    if missing:
        raise ValueError(f"Feature table requires schema 2.0; missing {missing}. Use an explicit legacy manifest to adapt historical CSVs.")
    for col in IDENTITY_COLUMNS + ["subject_id", "comparison", "class_name", "stage"]:
        for value in frame[col]:
            _text(value, col)
    if set(frame["schema_version"]) != {SCHEMA_VERSION}:
        raise ValueError("Unsupported feature schema_version")
    for col in ("class_label", "epoch_index", "attempt_index", "start_sample", "end_sample"):
        values = pd.to_numeric(frame[col], errors="coerce").to_numpy(dtype=float)
        if not np.isfinite(values).all() or np.any(values != np.floor(values)):
            raise ValueError(f"{col} must contain finite integers")
    if not frame["class_label"].isin([0, 1]).all():
        raise ValueError("class_label must be 0 or 1")
    if (frame["epoch_index"] < 1).any() or (frame["attempt_index"] < 0).any():
        raise ValueError("Invalid epoch_index or attempt_index")
    if (frame["start_sample"] < 0).any() or (frame["end_sample"] <= frame["start_sample"]).any():
        raise ValueError("Invalid epoch sample interval")
    onset = pd.to_numeric(frame["onset_elapsed_s"], errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(onset).all() or np.any(onset < 0):
        raise ValueError("onset_elapsed_s must be finite and nonnegative")
    if frame["epoch_id"].duplicated().any():
        raise ValueError("Duplicate epoch_id")
    physical_keys = ["cohort_id", "subject_id", "session_id", "recording_id", "start_sample", "end_sample"]
    if frame.duplicated(physical_keys).any():
        raise ValueError("Duplicate physical epoch interval")
    recording_keys = ["cohort_id", "subject_id", "session_id", "recording_id"]
    immutable_fields = ["source_sha256", "units", "reference", "protocol_id", "protocol_version", "channel_order", "identity_origin"]
    if (frame.groupby(recording_keys, dropna=False)[immutable_fields].nunique(dropna=False) > 1).any().any():
        raise ValueError("A recording has inconsistent source/acquisition/protocol metadata across epochs")
    for row in frame.to_dict("records"):
        origin = row["identity_origin"]
        if origin not in {"raw_metadata", "legacy_manifest"}:
            raise ValueError("Unknown identity_origin")
        if not re.fullmatch(r"[0-9a-f]{64}", row["source_sha256"]):
            if not (origin == "legacy_manifest" and row["source_sha256"] == "unknown"):
                raise ValueError("source_sha256 must identify raw bytes, or be explicitly unknown for legacy data")
        if origin == "raw_metadata" and row["qc_status"] not in {"passed", "passed_with_warnings"}:
            raise ValueError("Raw feature table does not have a passing QA status")
        if origin == "raw_metadata":
            recording = stable_id("recording", {
                "cohort_id": row["cohort_id"], "subject_id": row["subject_id"],
                "session_id": row["session_id"], "raw_sha256": row["source_sha256"],
            })
            if row["recording_id"] != recording:
                raise ValueError("recording_id disagrees with raw source hash/session identity")
        if origin == "legacy_manifest" and row["qc_status"] != "not_revalidated":
            raise ValueError("Legacy feature adaptation must not claim raw QA was performed")
        try:
            channels = json.loads(row["channel_order"])
            reasons = json.loads(row["qc_reason_codes"])
        except (json.JSONDecodeError, TypeError) as exc:
            raise ValueError("channel_order and qc_reason_codes must be JSON") from exc
        if channels != "unknown" and (not isinstance(channels, list) or not channels or
                                      any(not isinstance(c, str) or not c for c in channels) or len(set(channels)) != len(channels)):
            raise ValueError("Invalid channel_order")
        if channels == "unknown" and origin != "legacy_manifest":
            raise ValueError("Raw metadata must declare channel_order")
        if not isinstance(reasons, list) or any(not isinstance(code, str) for code in reasons):
            raise ValueError("qc_reason_codes must be a JSON string array")
        expected = epoch_identity(row, row)
        if any(row[key] != value for key, value in expected.items()):
            raise ValueError("Epoch identity does not match its event metadata/sample interval")
    fcols = [c for c in frame if c.startswith(FEATURE_PREFIXES)]
    if not fcols:
        raise ValueError("Feature table contains no recognized features")
    if not np.isfinite(frame[fcols].to_numpy(dtype=float)).all():
        raise ValueError("Feature table contains nonfinite feature values")


def _read_mapping(mapping: str | Path | Mapping[str, Any]) -> dict[str, Any]:
    value = json.loads(Path(mapping).read_text(encoding="utf-8")) if isinstance(mapping, (str, Path)) else dict(mapping)
    if value.get("schema_version") != "legacy-mapping-1" or not isinstance(value.get("subjects"), dict):
        raise ValueError("Legacy manifest requires schema_version='legacy-mapping-1' and subjects mapping")
    return value


def adapt_legacy_features(frame: pd.DataFrame, mapping: str | Path | Mapping[str, Any], *,
                          feature_input_sha256: str | None = None) -> pd.DataFrame:
    """Return a new v2 table using explicit, auditable single-session mappings.

    The historical file is never modified.  Raw units, reference and source hash
    may be declared 'unknown'; a feature-file hash is never substituted for raw.
    Multiple historical sessions need a row-level migration, not this adapter.
    """
    if any(c in frame for c in IDENTITY_COLUMNS):
        raise ValueError("Legacy adapter does not accept partial or already-versioned identity columns")
    manifest = _read_mapping(mapping)
    if "feature_inputs" in manifest:
        inventory = manifest["feature_inputs"]
        if (not isinstance(inventory, dict) or not inventory or
                any(not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value) for value in inventory.values())):
            raise ValueError("feature_inputs must map file descriptions to SHA-256 hashes")
        if feature_input_sha256 not in set(inventory.values()):
            raise ValueError("Legacy feature input hash is not approved by the explicit mapping manifest")
    missing = [c for c in LEGACY_META_COLUMNS if c not in frame]
    if missing:
        raise ValueError(f"Legacy table missing metadata: {missing}")
    source = frame.copy()
    for value in source["subject_id"]:
        _text(value, "subject_id (load CSV IDs as strings)")
    rows = []
    for row in source.to_dict("records"):
        subject = row["subject_id"]
        if subject not in manifest["subjects"]:
            raise ValueError(f"Legacy manifest has no explicit mapping for subject_id={subject}")
        entry = dict(manifest["subjects"][subject])
        missing = [key for key in MAPPING_FIELDS if key not in entry]
        if missing:
            raise ValueError(f"Legacy mapping for {subject} missing {missing}")
        for key in MAPPING_FIELDS:
            if key != "channel_order":
                _text(entry[key], key)
        if entry["source_sha256"] != "unknown":
            raise ValueError("This legacy adapter requires source_sha256='unknown'; use a separately verified raw migration to assert a raw hash")
        entry.update(subject_id=subject, identity_origin="legacy_manifest")
        meta = make_epoch_metadata(entry, row, qc_status="not_revalidated", qc_reason_codes=["LEGACY_RAW_NOT_REVALIDATED"])
        meta.update({k: v for k, v in row.items() if k not in LEGACY_META_COLUMNS})
        rows.append(meta)
    out = pd.DataFrame(rows)
    validate_feature_table(out)
    out.attrs["legacy_adapter_manifest"] = manifest
    out.attrs["legacy_mapping_sha256"] = hashlib.sha256(canonical_json(manifest).encode("utf-8")).hexdigest()
    if feature_input_sha256 is not None:
        out.attrs["source_feature_sha256"] = feature_input_sha256
    return out


def load_feature_table(path: str | Path, *, legacy_manifest: str | Path | Mapping[str, Any] | None = None) -> pd.DataFrame:
    # Preserve IDs such as '001', '1', 'NA' and 'null' exactly. Empty numerics fail validation.
    path = Path(path)
    text_columns = IDENTITY_COLUMNS + ["subject_id", "comparison", "class_name", "stage"]
    payload = path.read_bytes()
    with io.TextIOWrapper(io.BytesIO(payload), encoding="utf-8-sig", newline="") as stream:
        header = next(csv.reader(stream), [])
    if not header or len(set(header)) != len(header) or any(not c.strip() for c in header):
        raise ValueError("Feature CSV contains a duplicate or blank header")
    frame = pd.read_csv(io.BytesIO(payload), encoding="utf-8-sig", keep_default_na=False, dtype={c: str for c in text_columns})
    source_hash = hashlib.sha256(payload).hexdigest()
    if "schema_version" not in frame:
        if legacy_manifest is None:
            raise ValueError("Historical feature CSV requires explicit --legacy-manifest; provenance will not be guessed")
        frame = adapt_legacy_features(frame, legacy_manifest, feature_input_sha256=source_hash)
    else:
        validate_feature_table(frame)
    frame.attrs["source_feature_sha256"] = source_hash
    frame.attrs["source_feature_path"] = str(path)
    return frame


def assert_same_epochs(left: pd.DataFrame, right: pd.DataFrame) -> None:
    """Reject missing, excess, duplicate or metadata-disagreeing epoch rows."""
    validate_feature_table(left)
    validate_feature_table(right)
    if set(left["epoch_id"]) != set(right["epoch_id"]):
        raise ValueError("Feature tables have missing or excess epoch IDs")
    a = left.set_index("epoch_id").sort_index()
    b = right.set_index("epoch_id").sort_index()
    for col in META_COLUMNS:
        if col == "epoch_id":
            continue
        if col == "onset_elapsed_s":
            same = np.allclose(a[col].to_numpy(float), b[col].to_numpy(float), rtol=0, atol=1e-9)
        else:
            same = a[col].equals(b[col])
        if not same:
            raise ValueError(f"Feature tables disagree on epoch metadata: {col}")


def session_output_stem(metadata: Mapping[str, Any]) -> str:
    # Hash the complete identity so e.g. a/b and a?b never collide after sanitizing.
    scope = {k: str(metadata[k]) for k in ("cohort_id", "subject_id", "session_id", "recording_id")}
    readable = re.sub(r"[^A-Za-z0-9_-]+", "_", scope["subject_id"]).strip("_")[:40] or "subject"
    return readable + "__" + stable_id("session", scope).split("-", 1)[1]


def assert_output_available(out_dir: str | Path) -> None:
    from .provenance import protect_output
    path = protect_output(out_dir)
    if path.exists() and (not path.is_dir() or any(path.iterdir())):
        raise FileExistsError(f"Refusing to overwrite existing feature output: {path}; choose a new directory")


def write_feature_outputs(out_dir: str | Path, tables: Mapping[str, pd.DataFrame], manifest: Mapping[str, Any]) -> None:
    """Publish complete extraction outputs together, after all sessions pass QA."""
    target = Path(out_dir)
    assert_output_available(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{target.name}.staging-", dir=target.parent))
    try:
        for name, frame in tables.items():
            if Path(name).name != name:
                raise ValueError("Output table filename must not contain a path")
            frame.to_csv(stage / name, index=False, encoding="utf-8-sig")
        payload = dict(manifest)
        payload["output_sha256"] = {name: hash_file(stage / name) for name in tables}
        (stage / "feature_manifest.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        assert_output_available(target)
        os.replace(stage, target)
    finally:
        if stage.exists():
            shutil.rmtree(stage)
