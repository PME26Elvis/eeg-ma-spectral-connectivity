import json
from pathlib import Path

import numpy as np
import pandas as pd

from eeg_ma.evaluation import run_inter_subject
from eeg_ma.extract import feature_columns
from eeg_ma.final_validation import iplv_from_phases, iplv_feature_names


def test_iplv_suppresses_zero_lag_and_retains_quadrature():
    phase = np.linspace(-np.pi, np.pi, 1000, endpoint=False)
    assert np.isclose(iplv_from_phases(phase, phase), 0.0, atol=1e-12)

    shifted = phase - np.pi / 2.0
    assert np.isclose(iplv_from_phases(phase, shifted), 1.0, atol=1e-12)


def test_iplv_feature_count_and_range():
    channels = ["FP1", "FP2", "F7", "F3", "Fz", "F4", "F8"]
    bands = [
        ("delta", 1.0, 4.0),
        ("theta", 4.0, 8.0),
        ("alpha", 8.0, 13.0),
        ("beta_low", 13.0, 20.0),
        ("beta_high", 20.0, 30.0),
        ("gamma", 30.0, 45.0),
    ]
    assert len(iplv_feature_names(channels, bands)) == 126

    rng = np.random.default_rng(4)
    a = rng.uniform(-np.pi, np.pi, 500)
    b = rng.uniform(-np.pi, np.pi, 500)
    value = iplv_from_phases(a, b)
    assert 0.0 <= value <= 1.0


def _selector_frame() -> pd.DataFrame:
    row = {}
    bands = ["delta", "theta", "alpha", "beta_low", "beta_high", "gamma"]
    for band in bands:
        for i in range(21):
            row[f"PLV__A{i}__B{i}__{band}"] = 0.5
            row[f"IPLV__A{i}__B{i}__{band}"] = 0.2
    return pd.DataFrame([row])


def test_final_validation_feature_selectors():
    df = _selector_frame()
    assert len(feature_columns(df, "plv")) == 126
    assert len(feature_columns(df, "iplv")) == 126
    for band in ["delta", "theta", "alpha", "beta_low", "beta_high", "gamma"]:
        cols = feature_columns(df, f"plv_band_{band}")
        assert len(cols) == 21
        assert all(c.endswith(f"__{band}") for c in cols)


def _smoke_cfg(feature_set: str):
    cfg = json.loads(Path("configs/smoke_test.json").read_text(encoding="utf-8"))
    cfg["feature_set"] = feature_set
    cfg["sfs"]["max_features"] = 1
    cfg["rbf_grid"] = {"C": [1.0], "gamma_values": [0.5]}
    cfg["n_jobs"] = 1
    return cfg


def _smoke_frame(n_subjects: int = 3) -> pd.DataFrame:
    rng = np.random.default_rng(20260911)
    rows = []
    for si in range(n_subjects):
        subject = f"s{si+1}"
        for y in (0, 1):
            for _ in range(12):
                rows.append(
                    {
                        "subject_id": subject,
                        "comparison": "rest1_vs_type1",
                        "class_label": y,
                        "PLV__FP1__FP2__alpha": np.clip(rng.normal(0.35 + 0.3 * y, 0.08), 0, 1),
                        "PLV__F3__F4__alpha": np.clip(rng.normal(0.4 + 0.2 * y, 0.08), 0, 1),
                        "PLV__FP1__FP2__theta": rng.uniform(0.2, 0.8),
                        "IPLV__FP1__FP2__alpha": np.clip(rng.normal(0.15 + 0.2 * y, 0.06), 0, 1),
                        "IPLV__F3__F4__alpha": np.clip(rng.normal(0.2 + 0.15 * y, 0.06), 0, 1),
                    }
                )
    return pd.DataFrame(rows)


def test_band_plv_and_iplv_use_same_inter_pipeline(tmp_path):
    df = _smoke_frame()

    alpha, _ = run_inter_subject(
        df,
        _smoke_cfg("plv_band_alpha"),
        tmp_path / "alpha",
    )
    iplv, _ = run_inter_subject(
        df,
        _smoke_cfg("iplv"),
        tmp_path / "iplv",
    )

    assert set(alpha["classifier"]) == {"LDA", "RBF-SVM", "KFDA"}
    assert set(iplv["classifier"]) == {"LDA", "RBF-SVM", "KFDA"}
    assert alpha["accuracy"].between(0, 1).all()
    assert iplv["accuracy"].between(0, 1).all()
    assert alpha["selected_features"].str.contains("PLV__").all()
    assert iplv["selected_features"].str.contains("IPLV__").all()
