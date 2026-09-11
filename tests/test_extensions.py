import numpy as np
import pandas as pd

from eeg_ma.constants import DEFAULT_BANDS, DEFAULT_CHANNELS, channel_pairs
from eeg_ma.extract import META_COLUMNS, feature_columns
from eeg_ma.extensions import (
    DEFAULT_HEMISPHERIC_PAIRS,
    add_asymmetry_features,
    analytic_phase_continuous,
    asymmetry_feature_names,
    merge_baseline_and_extensions,
    plv_feature_names,
    plv_from_phases,
)


def _meta_row(subject="s1", epoch=1):
    return {
        "subject_id": subject,
        "comparison": "rest1_vs_type1",
        "class_label": 0,
        "class_name": "Rest",
        "stage": "Rest1",
        "epoch_index": epoch,
        "attempt_index": 0,
        "onset_elapsed_s": float((epoch - 1) * 5),
        "start_sample": (epoch - 1) * 2500,
        "end_sample": epoch * 2500,
    }


def _baseline_bp_frame():
    row = _meta_row()
    # Make every required BP positive; right hemisphere is exactly twice left
    # for homologous pairs so log-ratio has a known value log(2).
    for ch in DEFAULT_CHANNELS:
        for band, _, _ in DEFAULT_BANDS:
            row[f"BP__{ch}__{band}"] = 1.0
    for left, right in DEFAULT_HEMISPHERIC_PAIRS:
        for band, _, _ in DEFAULT_BANDS:
            row[f"BP__{left}__{band}"] = 2.0
            row[f"BP__{right}__{band}"] = 4.0
    return pd.DataFrame([row])


def test_e1_log_ratio_asymmetry_math_and_count():
    cfg = {"bands_hz": {name: [lo, hi] for name, lo, hi in DEFAULT_BANDS}}
    ext_cfg = {
        "asymmetry": {
            "mode": "log_ratio",
            "epsilon": 1e-12,
            "pairs": [list(x) for x in DEFAULT_HEMISPHERIC_PAIRS],
        }
    }
    out = add_asymmetry_features(_baseline_bp_frame(), cfg, ext_cfg)
    cols = [c for c in out.columns if c.startswith("ASYM_")]
    assert len(cols) == 18
    assert np.allclose(out[cols].to_numpy(), np.log(2.0), atol=1e-12)


def test_e1_normalized_difference_is_supported():
    cfg = {"bands_hz": {name: [lo, hi] for name, lo, hi in DEFAULT_BANDS}}
    ext_cfg = {
        "asymmetry": {
            "mode": "normalized_difference",
            "epsilon": 1e-12,
            "pairs": [list(x) for x in DEFAULT_HEMISPHERIC_PAIRS],
        }
    }
    out = add_asymmetry_features(_baseline_bp_frame(), cfg, ext_cfg)
    cols = [c for c in out.columns if c.startswith("ASYM_")]
    assert len(cols) == 18
    assert np.allclose(out[cols].to_numpy(), 1.0 / 3.0, atol=1e-10)


def test_e2_plv_known_phase_lock_is_near_one():
    fs = 500.0
    t = np.arange(0.0, 5.0, 1.0 / fs)
    x = np.sin(2 * np.pi * 10.0 * t)
    y = np.sin(2 * np.pi * 10.0 * t + np.pi / 3.0)
    phase = analytic_phase_continuous(np.column_stack([x, y]), fs, 8.0, 13.0, order=4)
    value = plv_from_phases(phase[:, 0], phase[:, 1])
    assert 0.98 <= value <= 1.0


def test_plv_is_bounded():
    rng = np.random.default_rng(123)
    a = rng.uniform(-np.pi, np.pi, 5000)
    b = rng.uniform(-np.pi, np.pi, 5000)
    value = plv_from_phases(a, b)
    assert 0.0 <= value <= 1.0


def test_extension_feature_counts():
    assert len(asymmetry_feature_names(DEFAULT_BANDS)) == 18
    assert len(plv_feature_names(DEFAULT_CHANNELS, DEFAULT_BANDS)) == 21 * 6 == 126


def test_feature_columns_preserves_baseline_and_adds_extension_sets():
    cols = {}
    for ch in DEFAULT_CHANNELS:
        for band, _, _ in DEFAULT_BANDS:
            cols[f"BP__{ch}__{band}"] = [1.0]
    for a, b in channel_pairs(DEFAULT_CHANNELS):
        for band, _, _ in DEFAULT_BANDS:
            cols[f"COH__{a}__{b}__{band}"] = [0.5]
            cols[f"PLV__{a}__{b}__{band}"] = [0.5]
    for name in asymmetry_feature_names(DEFAULT_BANDS):
        cols[name] = [0.1]
    df = pd.DataFrame(cols)

    assert len(feature_columns(df, "all")) == 168
    assert len(feature_columns(df, "baseline_asym")) == 186
    assert len(feature_columns(df, "baseline_plv")) == 294
    assert len(feature_columns(df, "baseline_asym_plv")) == 312
    assert len(feature_columns(df, "asym")) == 18
    assert len(feature_columns(df, "plv")) == 126


def test_merge_extensions_is_one_to_one_and_preserves_baseline_order():
    b1 = _meta_row("s1", 1)
    b1["BP__FP1__delta"] = 10.0
    b2 = _meta_row("s1", 2)
    b2["BP__FP1__delta"] = 20.0
    baseline = pd.DataFrame([b1, b2])

    e1 = _meta_row("s1", 1)
    e1["ASYM_LOGRATIO__FP1__FP2__delta"] = 0.1
    e2 = _meta_row("s1", 2)
    e2["ASYM_LOGRATIO__FP1__FP2__delta"] = 0.2
    extensions = pd.DataFrame([e2, e1])

    merged = merge_baseline_and_extensions(baseline, extensions)
    assert merged["epoch_index"].tolist() == [1, 2]
    assert merged["ASYM_LOGRATIO__FP1__FP2__delta"].tolist() == [0.1, 0.2]
    assert list(merged[META_COLUMNS].columns) == META_COLUMNS
