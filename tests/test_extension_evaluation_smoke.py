import json
from pathlib import Path

import numpy as np
import pandas as pd

from eeg_ma.evaluation import run_inter_subject, run_intra_subject


def _cfg(feature_set):
    cfg = json.loads(Path("configs/smoke_test.json").read_text(encoding="utf-8"))
    cfg["feature_set"] = feature_set
    cfg["sfs"]["max_features"] = 1
    cfg["cv"]["intra_outer_folds"] = 2
    cfg["cv"]["intra_inner_folds"] = 2
    cfg["rbf_grid"] = {"C": [1.0], "gamma_values": [0.5]}
    cfg["n_jobs"] = 1
    return cfg


def _frame():
    rng = np.random.default_rng(20260911)
    rows = []
    for si in range(3):
        subject = f"s{si+1}"
        subject_offset = rng.normal(scale=0.2)
        for y in (0, 1):
            for i in range(12):
                rows.append(
                    {
                        "subject_id": subject,
                        "comparison": "rest1_vs_type1",
                        "class_label": y,
                        "BP__FP1__alpha": rng.normal() + 1.2 * y + subject_offset,
                        "COH__FP1__FP2__alpha": np.clip(rng.normal(0.5 + 0.1 * y, 0.08), 0, 1),
                        "ASYM_LOGRATIO__FP1__FP2__alpha": rng.normal(0.3 * y, 0.4),
                        "PLV__FP1__FP2__alpha": np.clip(rng.normal(0.45 + 0.12 * y, 0.08), 0, 1),
                    }
                )
    return pd.DataFrame(rows)


def test_e1_and_e2_feature_sets_run_through_intra_and_inter(tmp_path):
    df = _frame()
    for feature_set in ("baseline_asym", "baseline_plv"):
        cfg = _cfg(feature_set)
        root = tmp_path / feature_set
        intra, _ = run_intra_subject(df, cfg, root / "intra")
        inter, _ = run_inter_subject(df, cfg, root / "inter")
        assert set(intra["classifier"]) == {"LDA", "RBF-SVM", "KFDA"}
        assert set(inter["classifier"]) == {"LDA", "RBF-SVM", "KFDA"}
        assert intra["accuracy"].between(0, 1).all()
        assert inter["accuracy"].between(0, 1).all()
