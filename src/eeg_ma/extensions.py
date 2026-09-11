from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import signal

from .constants import band_items, channel_pairs
from .extract import META_COLUMNS
from .io import discover_session_dirs, load_session
from .preprocess import EpochRecord, extract_protocol_epochs, filter_continuous


DEFAULT_HEMISPHERIC_PAIRS: Tuple[Tuple[str, str], ...] = (
    ("FP1", "FP2"),
    ("F3", "F4"),
    ("F7", "F8"),
)

# Align tables with integer/categorical epoch identity rather than an exact
# floating-point timestamp equality.  onset_elapsed_s is retained from the
# already-versioned baseline table and still available for auditing.
ALIGN_COLUMNS = [c for c in META_COLUMNS if c != "onset_elapsed_s"]


def _asymmetry_prefix(mode: str) -> str:
    mode = str(mode).lower()
    if mode == "log_ratio":
        return "ASYM_LOGRATIO"
    if mode == "normalized_difference":
        return "ASYM_NORMDIFF"
    raise ValueError(f"未知 asymmetry mode={mode}")


def asymmetry_feature_names(
    bands: Iterable[Tuple[str, float, float]],
    pairs: Sequence[Tuple[str, str]] = DEFAULT_HEMISPHERIC_PAIRS,
    mode: str = "log_ratio",
) -> List[str]:
    """Return hemispheric-asymmetry feature names.

    Each pair is ordered (left, right).  The default log-ratio definition is
    log(BP_right) - log(BP_left); normalized difference is
    (BP_right - BP_left) / (BP_right + BP_left).
    """
    prefix = _asymmetry_prefix(mode)
    band_names = [name for name, _, _ in bands]
    return [
        f"{prefix}__{left}__{right}__{band}"
        for left, right in pairs
        for band in band_names
    ]


def add_asymmetry_features(
    baseline_features: pd.DataFrame,
    cfg: Dict,
    extension_cfg: Dict,
) -> pd.DataFrame:
    """Derive E1 hemispheric asymmetry directly from baseline BP features.

    The computation is class-label agnostic and uses only per-epoch BP values,
    so it does not introduce subject/test-label leakage.
    """
    asym_cfg = extension_cfg.get("asymmetry", {})
    mode = str(asym_cfg.get("mode", "log_ratio"))
    eps = float(asym_cfg.get("epsilon", 1e-12))
    if eps <= 0:
        raise ValueError("asymmetry.epsilon 必須 > 0")

    pairs = [tuple(p) for p in asym_cfg.get("pairs", DEFAULT_HEMISPHERIC_PAIRS)]
    if any(len(p) != 2 for p in pairs):
        raise ValueError("asymmetry.pairs 每組必須是 [left, right]")

    bands = band_items(cfg["bands_hz"])
    out = baseline_features[META_COLUMNS].copy()
    prefix = _asymmetry_prefix(mode)

    for left, right in pairs:
        for band, _, _ in bands:
            left_col = f"BP__{left}__{band}"
            right_col = f"BP__{right}__{band}"
            missing = [c for c in (left_col, right_col) if c not in baseline_features.columns]
            if missing:
                raise ValueError(f"Asymmetry 缺少 BP 欄位: {missing}")

            left_bp = baseline_features[left_col].to_numpy(dtype=float)
            right_bp = baseline_features[right_col].to_numpy(dtype=float)
            if np.any(left_bp < 0) or np.any(right_bp < 0):
                raise ValueError("Band Power 不應為負值，無法計算 asymmetry")

            if mode == "log_ratio":
                values = np.log(np.maximum(right_bp, eps)) - np.log(np.maximum(left_bp, eps))
            elif mode == "normalized_difference":
                values = (right_bp - left_bp) / (right_bp + left_bp + eps)
            else:  # guarded by _asymmetry_prefix, kept explicit for readability
                raise ValueError(mode)

            out[f"{prefix}__{left}__{right}__{band}"] = values

    feature_cols = [c for c in out.columns if c.startswith("ASYM_")]
    expected = len(pairs) * len(bands)
    if len(feature_cols) != expected:
        raise AssertionError(f"Asymmetry features 應為 {expected}，實際 {len(feature_cols)}")
    values = out[feature_cols].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Asymmetry feature 產生 NaN/Inf")
    return out


def plv_from_phases(phase_a: np.ndarray, phase_b: np.ndarray) -> float:
    """Phase Locking Value = |mean(exp(j*(phi_a-phi_b)))|."""
    a = np.asarray(phase_a, dtype=float)
    b = np.asarray(phase_b, dtype=float)
    if a.ndim != 1 or b.ndim != 1 or a.shape != b.shape or len(a) == 0:
        raise ValueError("PLV phase arrays 必須是一維、非空且長度相同")
    value = float(np.abs(np.mean(np.exp(1j * (a - b)))))
    return float(np.clip(value, 0.0, 1.0))


def analytic_phase_continuous(
    x: np.ndarray,
    fs: float,
    low_hz: float,
    high_hz: float,
    order: int = 4,
) -> np.ndarray:
    """Band-pass an entire continuous recording, then return Hilbert phase.

    Filtering is deliberately performed on the continuous 2-50 Hz baseline-
    preprocessed signal before epoch slicing.  This avoids creating a fresh
    zero-phase-filter edge transient at every 5-s epoch boundary.
    """
    values = np.asarray(x, dtype=float)
    if values.ndim != 2:
        raise ValueError("PLV continuous input 必須為 [samples, channels]")
    nyquist = float(fs) / 2.0
    lo = float(low_hz)
    hi = float(high_hz)
    if not (0.0 < lo < hi < nyquist):
        raise ValueError(f"PLV bandpass 不合法: [{lo}, {hi}] Hz, Nyquist={nyquist}")

    sos = signal.butter(
        int(order),
        [lo, hi],
        btype="bandpass",
        fs=float(fs),
        output="sos",
    )
    band_limited = signal.sosfiltfilt(sos, values, axis=0)
    analytic = signal.hilbert(band_limited, axis=0)
    phase = np.angle(analytic)
    if not np.isfinite(phase).all():
        raise ValueError("PLV analytic phase 產生 NaN/Inf")
    return phase


def plv_feature_names(
    channels: Sequence[str],
    bands: Iterable[Tuple[str, float, float]],
) -> List[str]:
    band_names = [name for name, _, _ in bands]
    return [
        f"PLV__{a}__{b}__{band}"
        for a, b in channel_pairs(channels)
        for band in band_names
    ]


def _record_metadata(record: EpochRecord) -> Dict[str, object]:
    return {
        "subject_id": record.subject_id,
        "comparison": record.comparison,
        "class_label": record.class_label,
        "class_name": record.class_name,
        "stage": record.stage,
        "epoch_index": record.epoch_index,
        "attempt_index": record.attempt_index,
        "onset_elapsed_s": record.onset_elapsed_s,
        "start_sample": record.start_sample,
        "end_sample": record.end_sample,
    }


def extract_plv_session(
    session,
    baseline_preprocessed: np.ndarray,
    cfg: Dict,
    extension_cfg: Dict,
) -> pd.DataFrame:
    """Extract 21 channel-pair x 6 band = 126 PLV features for one session."""
    channels = list(cfg["analysis_channels"])
    fs = float(session.sampling_rate_hz)
    records = extract_protocol_epochs(session, baseline_preprocessed, cfg)
    rows: List[Dict[str, object]] = [_record_metadata(rec) for rec in records]

    bands = band_items(cfg["bands_hz"])
    pairs = channel_pairs(channels)
    index = {ch: i for i, ch in enumerate(channels)}
    plv_cfg = extension_cfg.get("plv", {})
    order = int(plv_cfg.get("filter_order", 4))

    # One band at a time keeps memory bounded for ~500k-sample recordings.
    for band_name, lo, hi in bands:
        phase = analytic_phase_continuous(
            baseline_preprocessed,
            fs,
            lo,
            hi,
            order=order,
        )
        for row, rec in zip(rows, records):
            segment = phase[rec.start_sample:rec.end_sample]
            if len(segment) != rec.end_sample - rec.start_sample:
                raise ValueError("PLV epoch slicing 不完整")
            for a, b in pairs:
                row[f"PLV__{a}__{b}__{band_name}"] = plv_from_phases(
                    segment[:, index[a]],
                    segment[:, index[b]],
                )

    out = pd.DataFrame(rows)
    feature_cols = [c for c in out.columns if c.startswith("PLV__")]
    expected = len(pairs) * len(bands)
    if len(feature_cols) != expected:
        raise AssertionError(f"PLV features 應為 {expected}，實際 {len(feature_cols)}")
    values = out[feature_cols].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("PLV feature 產生 NaN/Inf")
    if np.any(values < -1e-12) or np.any(values > 1.0 + 1e-12):
        raise ValueError("PLV feature 超出 [0,1]")
    return out


def merge_baseline_and_extensions(
    baseline_features: pd.DataFrame,
    extension_features: pd.DataFrame,
) -> pd.DataFrame:
    """One-to-one merge that preserves baseline row order and metadata."""
    left = baseline_features.copy()
    right = extension_features.copy()
    left["subject_id"] = left["subject_id"].astype(str)
    right["subject_id"] = right["subject_id"].astype(str)

    if left.duplicated(ALIGN_COLUMNS).any():
        raise ValueError("Baseline feature table 的 epoch alignment key 非唯一")
    if right.duplicated(ALIGN_COLUMNS).any():
        raise ValueError("Extension feature table 的 epoch alignment key 非唯一")

    extension_cols = [c for c in right.columns if c not in META_COLUMNS]
    overlap = sorted(set(extension_cols) & set(left.columns))
    if overlap:
        raise ValueError(f"Extension 欄位與 baseline 重複: {overlap[:5]}")

    left["__row_order"] = np.arange(len(left))
    merged = left.merge(
        right[ALIGN_COLUMNS + extension_cols],
        on=ALIGN_COLUMNS,
        how="left",
        validate="one_to_one",
    ).sort_values("__row_order").drop(columns="__row_order").reset_index(drop=True)

    if len(merged) != len(left):
        raise AssertionError("Baseline/extension merge row count 改變")
    if merged[extension_cols].isna().any().any():
        raise ValueError("有 baseline epoch 找不到對應的 extension features")
    return merged


def extract_all_extension_features(
    raw_root: str | Path,
    baseline_feature_path: str | Path,
    cfg: Dict,
    extension_cfg: Dict,
    out_dir: str | Path,
) -> pd.DataFrame:
    """Build versionable E1/E2 feature tables without modifying baseline CSVs."""
    raw_root = Path(raw_root)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    baseline = pd.read_csv(baseline_feature_path, encoding="utf-8-sig")
    baseline["subject_id"] = baseline["subject_id"].astype(str)
    sessions = discover_session_dirs(raw_root)
    if not sessions:
        raise FileNotFoundError(f"在 {raw_root} 找不到 eeg_raw.csv")

    frames: List[pd.DataFrame] = []
    seen_subjects = set()
    for session_dir in sessions:
        session = load_session(session_dir, configured_fs=float(cfg["sampling_rate_hz"]))
        subject = str(session.subject_id)
        if subject in seen_subjects:
            raise ValueError(f"同一 subject_id 出現多個 raw session: {subject}")
        seen_subjects.add(subject)

        base_subject = baseline.loc[baseline["subject_id"] == subject].copy()
        if base_subject.empty:
            raise ValueError(f"Baseline feature table 找不到 subject_id={subject}")

        asym = add_asymmetry_features(base_subject, cfg, extension_cfg)

        raw_x = session.raw[list(cfg["analysis_channels"])].to_numpy(dtype=float)
        baseline_preprocessed = filter_continuous(raw_x, session.sampling_rate_hz, cfg)
        plv = extract_plv_session(session, baseline_preprocessed, cfg, extension_cfg)

        plv_cols = [c for c in plv.columns if c.startswith("PLV__")]
        asym["__row_order"] = np.arange(len(asym))
        ext = asym.merge(
            plv[ALIGN_COLUMNS + plv_cols],
            on=ALIGN_COLUMNS,
            how="left",
            validate="one_to_one",
        )
        ext = ext.sort_values("__row_order").drop(columns="__row_order").reset_index(drop=True)
        if ext.isna().any().any():
            raise ValueError(f"{subject}: extension features 對齊後出現缺值")
        if len(ext) != len(base_subject):
            raise ValueError(f"{subject}: extension rows={len(ext)}, baseline rows={len(base_subject)}")

        safe_subject = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in subject).strip("_") or "subject"
        ext.to_csv(out_dir / f"{safe_subject}_extensions.csv", index=False, encoding="utf-8-sig")
        frames.append(ext)

    missing_raw = sorted(set(baseline["subject_id"].unique()) - seen_subjects)
    if missing_raw:
        raise ValueError(f"Baseline 中有受試者缺少 raw session: {missing_raw}")

    combined = pd.concat(frames, ignore_index=True)
    combined.to_csv(out_dir / "features_extensions.csv", index=False, encoding="utf-8-sig")

    counts = {
        "n_rows": int(len(combined)),
        "n_subjects": int(combined["subject_id"].nunique()),
        "asymmetry_features": int(sum(c.startswith("ASYM_") for c in combined.columns)),
        "plv_features": int(sum(c.startswith("PLV__") for c in combined.columns)),
    }
    pd.DataFrame([counts]).to_csv(out_dir / "extension_manifest.csv", index=False, encoding="utf-8-sig")
    return combined
