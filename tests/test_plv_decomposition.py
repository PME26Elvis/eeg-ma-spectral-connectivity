import json
from pathlib import Path

import numpy as np
import pandas as pd

from eeg_ma.evaluation import run_inter_subject, run_intra_subject
from eeg_ma.extract import feature_columns


def _matrix_frame() -> pd.DataFrame:
    row = {}
    for i in range(42):
        row[f"BP__C{i}__b"] = float(i + 1)
    for i in range(126):
        row[f"COH__A{i}__B{i}__b"] = 0.1
    for i in range(18):
        row[f"ASYM_LOGRATIO__L{i}__R{i}__b"] = 0.0
    for i in range(126):
        row[f"PLV__A{i}__B{i}__b"] = 0.5
    return pd.DataFrame([row])


def test_plv_decomposition_feature_counts_and_frozen_baseline():
    df = _matrix_frame()
    assert len(feature_columns(df, "baseline")) == 168
    assert len(feature_columns(df, "all")) == 168
    assert len(feature_columns(df, "plv")) == 126
    assert len(feature_columns(df, "bp_plv")) == 168
    assert len(feature_columns(df, "coh_plv")) == 252
    assert len(feature_columns(df, "asym_plv")) == 144
    assert len(feature_columns(df, "baseline_asym_plv")) == 312


def _smoke_cfg():
    cfg = json.loads(Path("configs/smoke_test.json").read_text(encoding="utf-8"))
    cfg["feature_set"] = "bp_plv"
    cfg["sfs"]["max_features"] = 1
    cfg["cv"]["intra_outer_folds"] = 2
    cfg["cv"]["intra_inner_folds"] = 2
    cfg["rbf_grid"] = {"C": [1.0], "gamma_values": [0.5]}
    cfg["n_jobs"] = 1
    return cfg


def _smoke_frame(n_subjects: int = 3) -> pd.DataFrame:
    rng = np.random.default_rng(20260911)
    rows = []
    for si in range(n_subjects):
        subject = f"s{si+1}"
        subject_offset = rng.normal(scale=0.15)
        for y in (0, 1):
            for _ in range(12):
                rows.append(
                    {
                        "subject_id": subject,
                        "comparison": "rest1_vs_type1",
                        "class_label": y,
                        "BP__FP1__alpha": rng.normal() + subject_offset + 0.8 * y,
                        "COH__FP1__FP2__alpha": rng.uniform(0.2, 0.8),
                        "ASYM_LOGRATIO__FP1__FP2__alpha": rng.normal(scale=0.2),
                        "PLV__FP1__FP2__alpha": np.clip(
                            rng.normal(loc=0.45 + 0.25 * y, scale=0.08), 0.0, 1.0
                        ),
                    }
                )
    return pd.DataFrame(rows)


def test_bp_plv_runs_through_same_intra_and_inter_pipeline(tmp_path):
    cfg = _smoke_cfg()
    df = _smoke_frame()
    intra, _ = run_intra_subject(df, cfg, tmp_path / "intra")
    inter, _ = run_inter_subject(df, cfg, tmp_path / "inter")
    assert set(intra["classifier"]) == {"LDA", "RBF-SVM", "KFDA"}
    assert set(inter["classifier"]) == {"LDA", "RBF-SVM", "KFDA"}
    assert intra["accuracy"].between(0, 1).all()
    assert inter["accuracy"].between(0, 1).all()
    assert intra["selected_features"].str.contains("BP__|PLV__", regex=True).all()
    assert inter["selected_features"].str.contains("BP__|PLV__", regex=True).all()
