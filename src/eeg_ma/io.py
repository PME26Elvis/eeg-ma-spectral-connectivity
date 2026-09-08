from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd


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


def discover_session_dirs(raw_root: str | Path) -> List[Path]:
    raw_root = Path(raw_root)
    dirs = sorted({p.parent.resolve() for p in raw_root.rglob("eeg_raw.csv")})
    return [Path(d) for d in dirs]


def load_session(path: str | Path, configured_fs: Optional[float] = None) -> SessionData:
    path = Path(path)
    required_files = ["eeg_raw.csv", "events.csv", "trials.csv", "rest_epochs.csv", "session_info.txt"]
    missing = [name for name in required_files if not (path / name).exists()]
    if missing:
        raise FileNotFoundError(f"{path}: 缺少 {missing}")

    raw = pd.read_csv(path / "eeg_raw.csv", encoding="utf-8-sig")
    events = pd.read_csv(path / "events.csv", encoding="utf-8-sig")
    trials = pd.read_csv(path / "trials.csv", encoding="utf-8-sig")
    rest = pd.read_csv(path / "rest_epochs.csv", encoding="utf-8-sig")
    info = (path / "session_info.txt").read_text(encoding="utf-8-sig")

    _require_columns(raw, RAW_REQUIRED, "eeg_raw.csv")
    _require_columns(events, EVENT_REQUIRED, "events.csv")
    _require_columns(trials, TRIAL_REQUIRED, "trials.csv")
    _require_columns(rest, REST_REQUIRED, "rest_epochs.csv")

    subject_values = [str(x) for x in trials["subject_id"].dropna().unique() if str(x).strip()]
    if not subject_values:
        subject_values = [str(x) for x in events["subject_id"].dropna().unique() if str(x).strip()]
    subject = subject_values[0] if subject_values else _subject_from_info(info) or path.name

    fs_info = _sampling_rate_from_info(info)
    fs = float(configured_fs if configured_fs is not None else (fs_info or _infer_fs(raw)))
    if fs_info is not None and configured_fs is not None and abs(fs_info - configured_fs) > 1e-6:
        raise ValueError(f"{path}: session_info Fs={fs_info} 與 config Fs={configured_fs} 不一致")

    for col in ["marker_sent"]:
        events[col] = _as_bool(events[col])
        rest[col] = _as_bool(rest[col])
    trials["analysis_eligible"] = _as_bool(trials["analysis_eligible"])
    trials["response_correct"] = _as_bool(trials["response_correct"])

    return SessionData(path, subject, fs, raw, events, trials, rest, info)


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
    return series.astype(str).str.strip().str.lower().map({
        "true": True, "false": False, "1": True, "0": False,
        "yes": True, "no": False, "": False, "nan": False
    }).fillna(False).astype(bool)
