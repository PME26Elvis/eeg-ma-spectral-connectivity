"""Real synthetic raw-to-features inputs shared by exploratory tests and smoke."""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

from eeg_ma.exploratory import ANALYSIS_ID, BARE_ELIGIBILITY, DEFAULT_PAIRS, PLAN_VERSION, TRACKS
from eeg_ma.provenance import sha256_file
from eeg_ma.extract import extract_all
from eeg_ma.numerical_validation import fixture_definition, write_synthetic_raw
from eeg_ma.schema import META_COLUMNS


def create_exploratory_fixture(root: Path):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=False)
    definition = fixture_definition()
    raw = write_synthetic_raw(root / "raw", definition)
    frame, qa = extract_all(raw, definition["analysis"], root / "extracted")
    assert qa.qa_pass.all()
    assert len(frame) == 48 and frame.subject_id.nunique() == 3
    assert len(frame[["subject_id", "session_id"]].drop_duplicates()) == 6
    cols = ["BP__FP1__alpha", "BP__FP2__alpha", "COH__FP1__FP2__alpha"]
    frame = frame[[*META_COLUMNS, *cols]].copy()
    features = root / "features.csv"
    frame.to_csv(features, index=False, encoding="utf-8", lineterminator="\n")
    # Independently frozen eligible roster: it must exist before classification.
    eligible = frame[["cohort_id", "protocol_id", "comparison", "subject_id",
                      "session_id", "epoch_id", "class_label"]].rename(
        columns={"class_label": "y_true"})
    eligible = eligible[BARE_ELIGIBILITY].sort_values(BARE_ELIGIBILITY)
    eligibility = root / "eligibility.csv"
    eligible.to_csv(eligibility, index=False, encoding="utf-8", lineterminator="\n")
    cfg = deepcopy(definition["analysis"])
    cfg["classifiers"] = TRACKS["S"]
    cfg["n_jobs"] = 1
    cfg["random_forest"] = {
        "parameters": {"n_estimators": 6, "max_depth": 4,
                       "random_state": 20261119, "oob_score": True, "n_jobs": 1},
        "grid": {"min_samples_leaf": [1, 2]},
    }
    cfg["lightgbm"] = {
        "parameters": {"n_estimators": 7, "num_leaves": 3,
                       "max_depth": 3, "max_bin": 7, "min_data_in_leaf": 2,
                       "learning_rate": 0.1, "random_state": 20261119, "n_jobs": 1},
        "grid": {"lambda_l2": [1.0, 3.0]},
    }
    config = root / "config.json"
    config.write_text(json.dumps(cfg, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    plan = {
        "schema_version": PLAN_VERSION, "analysis_id": ANALYSIS_ID,
        "status": "synthetic_only", "cohort_id": str(frame.cohort_id.iloc[0]),
        "protocol_id": str(frame.protocol_id.iloc[0]),
        "comparisons": [item["name"] for item in cfg["comparisons"]],
        "feature_set": cfg["feature_set"], "ordered_features": cols, "tracks": TRACKS,
        "eligibility_sha256": sha256_file(eligibility),
        "paired_comparisons": DEFAULT_PAIRS, "registration_evidence": None,
        "predeclared_scope": "descriptive_exploration_not_confirmatory",
    }
    plan_path = root / "plan.json"
    plan_path.write_text(json.dumps(plan, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return features, config, plan_path, eligibility, frame
