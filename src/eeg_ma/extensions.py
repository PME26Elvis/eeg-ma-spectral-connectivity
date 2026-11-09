from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import signal

from .constants import band_items, channel_pairs
from .extract import META_COLUMNS
from .preprocess import EpochRecord, extract_protocol_epochs, filter_continuous
from .qa import preflight_sessions, exclusion_table
from .schema import (SCHEMA_VERSION, assert_output_available, assert_same_epochs, load_feature_table,
                     session_output_stem, validate_feature_table, write_feature_outputs)


DEFAULT_HEMISPHERIC_PAIRS: Tuple[Tuple[str, str], ...] = (
    ("FP1", "FP2"),
    ("F3", "F4"),
    ("F7", "F8"),
)

# Align tables with integer/categorical epoch identity rather than an exact
# floating-point timestamp equality.  onset_elapsed_s is retained from the
# already-versioned baseline table and still available for auditing.
ALIGN_COLUMNS = ["epoch_id"]


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
    validate_feature_table(baseline_features)
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
    return dict(record.metadata)


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
    assert_same_epochs(left, right)

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
    *,
    legacy_manifest=None,
) -> pd.DataFrame:
    """Build versionable E1/E2 feature tables without modifying baseline CSVs."""
    raw_root = Path(raw_root)
    out_dir = Path(out_dir)
    assert_output_available(out_dir)
    baseline = load_feature_table(baseline_feature_path, legacy_manifest=legacy_manifest)
    sessions = preflight_sessions(raw_root, cfg)

    frames: List[pd.DataFrame] = []
    output_tables = {}
    for session in sessions:
        subject = str(session.subject_id)
        meta = session.metadata
        mask = np.ones(len(baseline), dtype=bool)
        for key in ("cohort_id", "subject_id", "session_id", "recording_id"):
            mask &= baseline[key].eq(meta[key]).to_numpy()
        base_subject = baseline.loc[mask].copy()
        if base_subject.empty:
            raise ValueError(f"Baseline feature table has no exact raw session identity for {subject}; re-extract a v2 baseline before raw extensions")

        asym = add_asymmetry_features(base_subject, cfg, extension_cfg)

        raw_x = session.raw[list(cfg["analysis_channels"])].to_numpy(dtype=float)
        baseline_preprocessed = filter_continuous(raw_x, session.sampling_rate_hz, cfg)
        plv = extract_plv_session(session, baseline_preprocessed, cfg, extension_cfg)

        ext = merge_baseline_and_extensions(asym, plv)
        output_tables[f"{session_output_stem(meta)}_extensions.csv"] = ext
        frames.append(ext)

    combined = pd.concat(frames, ignore_index=True)
    assert_same_epochs(baseline, combined)

    counts = {
        "n_rows": int(len(combined)),
        "n_subjects": int(combined["subject_id"].nunique()),
        "asymmetry_features": int(sum(c.startswith("ASYM_") for c in combined.columns)),
        "plv_features": int(sum(c.startswith("PLV__") for c in combined.columns)),
    }
    output_tables.update({"features_extensions.csv": combined, "extension_manifest.csv": pd.DataFrame([counts]),
                          "raw_validation.csv": pd.DataFrame([s.qa_report.as_dict() for s in sessions]),
                          "excluded_trials.csv": exclusion_table(sessions)})
    write_feature_outputs(out_dir, output_tables, {
        "schema_version": SCHEMA_VERSION, "feature_family": "asymmetry_plv", "config": cfg,
        "extension_config": extension_cfg, "n_rows": len(combined),
        "baseline_feature_sha256": baseline.attrs["source_feature_sha256"],
        "sessions": [{"metadata": s.metadata, "source_hashes": s.source_hashes,
                      "qa": s.qa_report.as_dict()} for s in sessions],
    })
    return combined
