import json
from pathlib import Path

import numpy as np
import pandas as pd

from eeg_ma.evaluation import run_intra_subject, run_inter_subject


def _cfg():
    cfg = json.loads(Path("configs/smoke_test.json").read_text(encoding="utf-8"))
    cfg["sfs"]["max_features"] = 1
    cfg["cv"]["intra_outer_folds"] = 2
    cfg["cv"]["intra_inner_folds"] = 2
    cfg["rbf_grid"] = {"C": [1.0], "gamma_values": [0.5]}
    cfg["n_jobs"] = 1
    return cfg


def _feature_frame(n_subjects=3):
    rng = np.random.default_rng(123)
    rows = []
    for si in range(n_subjects):
        subject = f"s{si+1}"
        offset = rng.normal(scale=0.1, size=4)
        for y in [0, 1]:
            for i in range(12):
                x = rng.normal(size=4) + offset
                x[0] += 2.0 * y
                rows.append({
                    "subject_id": subject,
                    "comparison": "rest1_vs_type1",
                    "class_label": y,
                    "BP__A__delta": x[0],
                    "BP__A__theta": x[1],
                    "COH__A__B__delta": x[2],
                    "COH__A__B__theta": x[3],
                })
    return pd.DataFrame(rows)


def test_intra_and_inter_wrappers_run(tmp_path):
    cfg = _cfg()
    df = _feature_frame(3)
    intra, _ = run_intra_subject(df, cfg, tmp_path / "intra")
    inter, _ = run_inter_subject(df, cfg, tmp_path / "inter")
    assert set(intra["classifier"]) == {"LDA", "RBF-SVM", "KFDA"}
    assert set(inter["classifier"]) == {"LDA", "RBF-SVM", "KFDA"}
    assert intra["accuracy"].between(0, 1).all()
    assert inter["accuracy"].between(0, 1).all()
