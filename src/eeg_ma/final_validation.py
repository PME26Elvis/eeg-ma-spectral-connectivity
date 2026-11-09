from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

from .constants import band_items, channel_pairs
from .extract import META_COLUMNS
from .extensions import analytic_phase_continuous
from .preprocess import EpochRecord, extract_protocol_epochs, filter_continuous
from .qa import preflight_sessions, exclusion_table
from .schema import (SCHEMA_VERSION, assert_output_available, assert_same_epochs, load_feature_table,
                     session_output_stem, write_feature_outputs)


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
    return dict(record.metadata)


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
    *,
    legacy_manifest=None,
) -> pd.DataFrame:
    """Build a separate versionable iPLV table for the final robustness check."""
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
            raise ValueError(f"Baseline feature table has no exact raw session identity for {subject}; re-extract a v2 baseline before iPLV")

        raw_x = session.raw[list(cfg["analysis_channels"])].to_numpy(dtype=float)
        baseline_preprocessed = filter_continuous(raw_x, session.sampling_rate_hz, cfg)
        iplv = extract_iplv_session(
            session,
            baseline_preprocessed,
            cfg,
            filter_order=int(filter_order),
        )
        assert_same_epochs(base_subject, iplv)
        aligned = iplv.set_index("epoch_id", drop=False).loc[base_subject["epoch_id"]].reset_index(drop=True)
        output_tables[f"{session_output_stem(meta)}_iplv.csv"] = aligned
        frames.append(aligned)

    combined = pd.concat(frames, ignore_index=True)
    assert_same_epochs(baseline, combined)

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
    output_tables.update({"features_iplv.csv": combined, "iplv_manifest.csv": manifest,
                          "raw_validation.csv": pd.DataFrame([s.qa_report.as_dict() for s in sessions]),
                          "excluded_trials.csv": exclusion_table(sessions)})
    write_feature_outputs(out_dir, output_tables, {
        "schema_version": SCHEMA_VERSION, "feature_family": "iplv", "config": cfg,
        "filter_order": int(filter_order), "n_rows": len(combined),
        "baseline_feature_sha256": baseline.attrs["source_feature_sha256"],
        "sessions": [{"metadata": s.metadata, "source_hashes": s.source_hashes,
                      "qa": s.qa_report.as_dict()} for s in sessions],
    })
    return combined
