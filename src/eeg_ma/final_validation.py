from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

from .constants import band_items, channel_pairs
from .extract import META_COLUMNS
from .extensions import ALIGN_COLUMNS, analytic_phase_continuous
from .io import discover_session_dirs, load_session
from .preprocess import EpochRecord, extract_protocol_epochs, filter_continuous


def iplv_from_phases(phase_a: np.ndarray, phase_b: np.ndarray) -> float:
    """Imaginary Phase Locking Value.

    iPLV = |Im(mean(exp(j * (phi_a - phi_b))))|.

    Unlike ordinary PLV, exact zero-lag and pi-lag locking contributes zero to
    this measure.  It is therefore used here only as a robustness check against
    the possibility that the observed PLV result is dominated by zero-lag
    synchronization/common-source effects.  It does not eliminate every form
    of volume conduction or reference-related bias.
    """
    a = np.asarray(phase_a, dtype=float)
    b = np.asarray(phase_b, dtype=float)
    if a.ndim != 1 or b.ndim != 1 or a.shape != b.shape or len(a) == 0:
        raise ValueError("iPLV phase arrays 必須是一維、非空且長度相同")
    mean_vector = np.mean(np.exp(1j * (a - b)))
    value = float(abs(np.imag(mean_vector)))
    return float(np.clip(value, 0.0, 1.0))


def iplv_feature_names(
    channels: Sequence[str],
    bands: Sequence[Tuple[str, float, float]],
) -> List[str]:
    band_names = [name for name, _, _ in bands]
    return [
        f"IPLV__{a}__{b}__{band}"
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


def extract_iplv_session(
    session,
    baseline_preprocessed: np.ndarray,
    cfg: Dict,
    filter_order: int = 4,
) -> pd.DataFrame:
    """Extract 21 channel-pair x 6 band = 126 iPLV features for one session."""
    channels = list(cfg["analysis_channels"])
    fs = float(session.sampling_rate_hz)
    records = extract_protocol_epochs(session, baseline_preprocessed, cfg)
    rows: List[Dict[str, object]] = [_record_metadata(rec) for rec in records]

    bands = band_items(cfg["bands_hz"])
    pairs = channel_pairs(channels)
    index = {ch: i for i, ch in enumerate(channels)}

    # Match the existing PLV implementation: filter the continuous 2-50 Hz
    # preprocessed signal band-by-band, then Hilbert transform before epoching.
    for band_name, lo, hi in bands:
        phase = analytic_phase_continuous(
            baseline_preprocessed,
            fs,
            lo,
            hi,
            order=int(filter_order),
        )
        for row, rec in zip(rows, records):
            segment = phase[rec.start_sample:rec.end_sample]
            if len(segment) != rec.end_sample - rec.start_sample:
                raise ValueError("iPLV epoch slicing 不完整")
            for a, b in pairs:
                row[f"IPLV__{a}__{b}__{band_name}"] = iplv_from_phases(
                    segment[:, index[a]],
                    segment[:, index[b]],
                )

    out = pd.DataFrame(rows)
    feature_cols = [c for c in out.columns if c.startswith("IPLV__")]
    expected = len(pairs) * len(bands)
    if len(feature_cols) != expected:
        raise AssertionError(f"iPLV features 應為 {expected}，實際 {len(feature_cols)}")
    values = out[feature_cols].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("iPLV feature 產生 NaN/Inf")
    if np.any(values < -1e-12) or np.any(values > 1.0 + 1e-12):
        raise ValueError("iPLV feature 超出 [0,1]")
    return out


def extract_all_iplv_features(
    raw_root: str | Path,
    baseline_feature_path: str | Path,
    cfg: Dict,
    out_dir: str | Path,
    filter_order: int = 4,
) -> pd.DataFrame:
    """Build a separate versionable iPLV table for the final robustness check."""
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

        raw_x = session.raw[list(cfg["analysis_channels"])].to_numpy(dtype=float)
        baseline_preprocessed = filter_continuous(raw_x, session.sampling_rate_hz, cfg)
        iplv = extract_iplv_session(
            session,
            baseline_preprocessed,
            cfg,
            filter_order=int(filter_order),
        )
        iplv_cols = [c for c in iplv.columns if c.startswith("IPLV__")]

        # Preserve the canonical baseline metadata/timestamps and align on the
        # integer/categorical epoch identity used by the E1/E2 feature merge.
        meta = base_subject[META_COLUMNS].copy()
        meta["__row_order"] = np.arange(len(meta))
        aligned = meta.merge(
            iplv[ALIGN_COLUMNS + iplv_cols],
            on=ALIGN_COLUMNS,
            how="left",
            validate="one_to_one",
        )
        aligned = aligned.sort_values("__row_order").drop(columns="__row_order").reset_index(drop=True)
        if len(aligned) != len(base_subject):
            raise ValueError(f"{subject}: iPLV rows={len(aligned)}, baseline rows={len(base_subject)}")
        if aligned[iplv_cols].isna().any().any():
            raise ValueError(f"{subject}: iPLV 對齊後出現缺值")

        safe_subject = "".join(
            ch if ch.isalnum() or ch in "-_" else "_" for ch in subject
        ).strip("_") or "subject"
        aligned.to_csv(
            out_dir / f"{safe_subject}_iplv.csv",
            index=False,
            encoding="utf-8-sig",
        )
        frames.append(aligned)

    missing_raw = sorted(set(baseline["subject_id"].unique()) - seen_subjects)
    if missing_raw:
        raise ValueError(f"Baseline 中有受試者缺少 raw session: {missing_raw}")

    combined = pd.concat(frames, ignore_index=True)
    combined.to_csv(out_dir / "features_iplv.csv", index=False, encoding="utf-8-sig")

    feature_cols = [c for c in combined.columns if c.startswith("IPLV__")]
    manifest = pd.DataFrame([
        {
            "n_rows": int(len(combined)),
            "n_subjects": int(combined["subject_id"].nunique()),
            "iplv_features": int(len(feature_cols)),
            "filter_order": int(filter_order),
            "definition": "abs(imag(mean(exp(1j*(phase_a-phase_b)))))",
        }
    ])
    manifest.to_csv(out_dir / "iplv_manifest.csv", index=False, encoding="utf-8-sig")
    return combined
