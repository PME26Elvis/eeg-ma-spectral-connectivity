from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import csv
import hashlib
import io
import json
import re
from typing import Any, Dict, Iterable, List, Mapping, Optional

import numpy as np
import pandas as pd

from .schema import stable_id


RAW_REQUIRED = ["sample_index", "elapsed_s"]
EVENT_REQUIRED = [
    "session_id", "subject_id", "stage", "phase", "task_type", "attempt_index",
    "utc_timestamp", "elapsed_ms", "marker_code", "marker_sent", "marker_message"
]
TRIAL_REQUIRED = [
    "session_id", "subject_id", "task_type", "attempt_index", "analysis_eligible",
    "response_correct", "calculation_onset"
]
REST_REQUIRED = [
    "session_id", "subject_id", "rest_stage", "rest_epoch_index", "onset_utc",
    "epoch_seconds", "marker_code", "marker_sent", "marker_message"
]


@dataclass
class SessionData:
    path: Path
    subject_id: str
    sampling_rate_hz: float
    raw: pd.DataFrame
    events: pd.DataFrame
    trials: pd.DataFrame
    rest_epochs: pd.DataFrame
    session_info: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    source_hashes: Dict[str, str] = field(default_factory=dict)
    qa_report: Any = None


def discover_session_dirs(raw_root: str | Path) -> List[Path]:
    raw_root = Path(raw_root)
    dirs = sorted({p.parent.resolve() for p in raw_root.rglob("eeg_raw.csv")})
    return [Path(d) for d in dirs]


def load_session(path: str | Path, configured_fs: Optional[float] = None,
                 metadata: Optional[Mapping[str, Any]] = None) -> SessionData:
    path = Path(path)
    required_files = ["eeg_raw.csv", "events.csv", "trials.csv", "rest_epochs.csv", "session_info.txt"]
    missing = [name for name in required_files if not (path / name).exists()]
    if missing:
        raise FileNotFoundError(f"{path}: 缺少 {missing}")

    source_hashes = {}

    def read_csv_source(name, **kwargs):
        payload = (path / name).read_bytes()
        source_hashes[name] = hashlib.sha256(payload).hexdigest()
        with io.TextIOWrapper(io.BytesIO(payload), encoding="utf-8-sig", newline="") as stream:
            header = next(csv.reader(stream), [])
        if not header or len(set(header)) != len(header) or any(not col.strip() for col in header):
            raise ValueError(f"INVALID_CSV_HEADER: {name} contains duplicate/blank columns")
        return pd.read_csv(io.BytesIO(payload), **kwargs)

    raw = read_csv_source("eeg_raw.csv", encoding="utf-8-sig", keep_default_na=False)
    read_kw = {"encoding": "utf-8-sig", "keep_default_na": False,
               "dtype": {"subject_id": str, "session_id": str}}
    events = read_csv_source("events.csv", **read_kw)
    trials = read_csv_source("trials.csv", **read_kw)
    rest = read_csv_source("rest_epochs.csv", **read_kw)
    info_bytes = (path / "session_info.txt").read_bytes()
    source_hashes["session_info.txt"] = hashlib.sha256(info_bytes).hexdigest()
    info = info_bytes.decode("utf-8-sig")

    _require_columns(raw, RAW_REQUIRED, "eeg_raw.csv")
    _require_columns(events, EVENT_REQUIRED, "events.csv")
    _require_columns(trials, TRIAL_REQUIRED, "trials.csv")
    _require_columns(rest, REST_REQUIRED, "rest_epochs.csv")

    subject = _consistent_identity((events, trials, rest), "subject_id")
    session_id = _consistent_identity((events, trials, rest), "session_id")
    info_subject = _subject_from_info(info)
    if info_subject is not None and info_subject != subject:
        raise ValueError("IDENTITY_MISMATCH: session_info Subject disagrees with CSV subject_id")
    info_session = re.search(r"^Session(?: ID)?:\s*(.+)$", info, flags=re.I | re.M)
    if info_session and info_session.group(1).strip() != session_id:
        raise ValueError("IDENTITY_MISMATCH: session_info Session disagrees with CSV session_id")

    sidecar = path / "session_metadata.json"
    if sidecar.exists():
        sidecar_bytes = sidecar.read_bytes()
        source_hashes[sidecar.name] = hashlib.sha256(sidecar_bytes).hexdigest()
        meta = json.loads(sidecar_bytes.decode("utf-8-sig"))
    else:
        meta = {}
    if not isinstance(meta, dict):
        raise ValueError("session_metadata.json must contain an object")
    if metadata is not None:
        for key, value in metadata.items():
            if key in meta and meta[key] != value:
                raise ValueError(f"METADATA_CONFLICT: override disagrees with sidecar for {key}")
            meta[key] = value
    for key, value in (("subject_id", subject), ("session_id", session_id)):
        if key in meta and meta[key] != value:
            raise ValueError(f"IDENTITY_MISMATCH: sidecar {key} disagrees with source CSVs")
        meta[key] = value
    if "source_sha256" in meta and meta["source_sha256"] != source_hashes["eeg_raw.csv"]:
        raise ValueError("SOURCE_HASH_MISMATCH: declared raw hash disagrees with the bytes parsed")
    if "identity_origin" in meta and meta["identity_origin"] != "raw_metadata":
        raise ValueError("IDENTITY_ORIGIN: raw ingestion cannot claim a legacy feature mapping")
    meta["source_sha256"] = source_hashes["eeg_raw.csv"]
    meta["identity_origin"] = "raw_metadata"
    # A missing cohort remains an error in shared QA; never infer it from paths.
    if meta.get("cohort_id"):
        expected_recording = stable_id("recording", {
            "cohort_id": meta["cohort_id"], "subject_id": subject,
            "session_id": session_id, "raw_sha256": meta["source_sha256"],
        })
        if "recording_id" in meta and meta["recording_id"] != expected_recording:
            raise ValueError("RECORDING_ID_MISMATCH: recording_id does not match source bytes and session identity")
        meta["recording_id"] = expected_recording

    fs_info = _sampling_rate_from_info(info)
    fs = float(configured_fs if configured_fs is not None else (fs_info or _infer_fs(raw)))
    if fs_info is not None and configured_fs is not None and abs(fs_info - configured_fs) > 1e-6:
        raise ValueError(f"{path}: session_info Fs={fs_info} 與 config Fs={configured_fs} 不一致")

    for col in ["marker_sent"]:
        events[col] = _as_bool(events[col])
        rest[col] = _as_bool(rest[col])
    trials["analysis_eligible"] = _as_bool(trials["analysis_eligible"])
    trials["response_correct"] = _as_bool(trials["response_correct"])

    return SessionData(path, subject, fs, raw, events, trials, rest, info, meta, source_hashes)


def validate_session_basic(session: SessionData, analysis_channels: Iterable[str]) -> Dict[str, object]:
    channels = list(analysis_channels)
    missing_channels = [ch for ch in channels if ch not in session.raw.columns]
    if missing_channels:
        raise ValueError(f"{session.path}: eeg_raw.csv 缺少分析通道 {missing_channels}")

    sample_index = session.raw["sample_index"].to_numpy()
    elapsed = session.raw["elapsed_s"].to_numpy(dtype=float)
    x = session.raw[channels].to_numpy(dtype=float)

    index_contiguous = bool(len(sample_index) < 2 or np.all(np.diff(sample_index) == 1))
    dt = np.diff(elapsed)
    median_dt = float(np.median(dt)) if len(dt) else np.nan
    inferred_fs = float(1.0 / median_dt) if median_dt > 0 else np.nan
    finite = bool(np.isfinite(x).all())
    channel_std = np.std(x, axis=0)
    flat_channels = [channels[i] for i, s in enumerate(channel_std) if s <= 1e-15]

    return {
        "subject_id": session.subject_id,
        "session_path": str(session.path),
        "n_samples": int(len(session.raw)),
        "sampling_rate_hz": float(session.sampling_rate_hz),
        "inferred_sampling_rate_hz": inferred_fs,
        "sample_index_contiguous": index_contiguous,
        "all_values_finite": finite,
        "flat_channels": ";".join(flat_channels),
        "all_markers_sent": bool(session.events["marker_sent"].fillna(False).all()),
    }


def _require_columns(df: pd.DataFrame, required: List[str], name: str) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{name} 欄位不符合 v0.15 收案格式，缺少: {missing}; 實際欄位={list(df.columns)}")


def _sampling_rate_from_info(text: str) -> Optional[float]:
    m = re.search(r"Sampling rate:\s*([0-9.]+)\s*Hz", text, flags=re.I)
    return float(m.group(1)) if m else None


def _subject_from_info(text: str) -> Optional[str]:
    m = re.search(r"^Subject:\s*(.+)$", text, flags=re.I | re.M)
    return m.group(1).strip() if m else None


def _infer_fs(raw: pd.DataFrame) -> float:
    elapsed = raw["elapsed_s"].to_numpy(dtype=float)
    if len(elapsed) < 2:
        raise ValueError("無法由 elapsed_s 推估取樣率")
    dt = float(np.median(np.diff(elapsed)))
    if dt <= 0:
        raise ValueError("elapsed_s 非單調遞增")
    return 1.0 / dt


def _as_bool(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series
    mapped = series.astype(str).str.strip().str.lower().map({
        "true": True, "false": False, "1": True, "0": False,
        "yes": True, "no": False,
    })
    if mapped.isna().any():
        raise ValueError(f"INVALID_BOOLEAN: {series.name} contains unknown or blank boolean values")
    return mapped.astype(bool)


def _consistent_identity(frames: Iterable[pd.DataFrame], column: str) -> str:
    values = []
    for frame in frames:
        if frame.empty:
            continue
        entries = frame[column].tolist()
        if any(not isinstance(v, str) or not v.strip() or v != v.strip() for v in entries):
            raise ValueError(f"INVALID_IDENTITY: {column} must be nonempty literal strings")
        values.extend(entries)
    unique = set(values)
    if len(unique) != 1:
        raise ValueError(f"IDENTITY_MISMATCH: expected exactly one {column} across events/trials/rest files")
    return next(iter(unique))
