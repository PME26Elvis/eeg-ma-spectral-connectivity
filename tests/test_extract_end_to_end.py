from pathlib import Path
import json

import numpy as np
import pandas as pd

from eeg_ma.extract import extract_all, feature_columns


def _write_session(root: Path):
    fs = 500
    # 50 seconds is enough for 8 synthetic 5-s epochs placed at 0,5,...35 s.
    n = fs * 50
    t = np.arange(n) / fs
    raw = pd.DataFrame({"sample_index": np.arange(n), "elapsed_s": t})
    for i, ch in enumerate(["FP1","FP2","F7","F3","Fz","F4","F8","Cz"]):
        raw[ch] = 0.001*np.sin(2*np.pi*(8+i*0.2)*t) + 0.0002*np.sin(2*np.pi*18*t)
    raw.to_csv(root / "eeg_raw.csv", index=False, encoding="utf-8-sig")

    events = pd.DataFrame([
        ["s","subj","Rest1","Rest","",0,"",0,20,True,""],
        ["s","subj","Rest1","Rest","",0,"",5000,20,True,""],
        ["s","subj","Type1","Calculation","Type1",1,"",10000,11,True,""],
        ["s","subj","Type1","Calculation","Type1",2,"",15000,11,True,""],
        ["s","subj","Rest2","Rest","",0,"",20000,24,True,""],
        ["s","subj","Rest2","Rest","",0,"",25000,24,True,""],
        ["s","subj","Type2","Calculation","Type2",1,"",30000,12,True,""],
        ["s","subj","Type2","Calculation","Type2",2,"",35000,12,True,""],
    ], columns=["session_id","subject_id","stage","phase","task_type","attempt_index","utc_timestamp","elapsed_ms","marker_code","marker_sent","marker_message"])
    events.to_csv(root / "events.csv", index=False, encoding="utf-8-sig")

    full_trial_cols = [
        "session_id","subject_id","task_type","attempt_index","mini_chain","trial_in_chain",
        "minuend","subtrahend","correct_answer","probe_number","correct_relation","participant_response",
        "response_correct","analysis_eligible","correct_count_after_response","reaction_time_ms",
        "calculation_onset","pre_answer_fixation_onset","answer_onset","response_time","post_answer_fixation_onset"
    ]
    trials = []
    for typ in ["Type1","Type2"]:
        for attempt in [1,2]:
            trials.append(["s","subj",typ,attempt,1,attempt,100,10,90,90,"Equal","Equal",True,True,attempt,1000,"","","","",""])
    pd.DataFrame(trials, columns=full_trial_cols).to_csv(root / "trials.csv", index=False, encoding="utf-8-sig")

    rest_cols = ["session_id","subject_id","rest_stage","rest_epoch_index","onset_utc","epoch_seconds","marker_code","marker_sent","marker_message"]
    pd.DataFrame([
        ["s","subj","Rest1",1,"",5,20,True,""], ["s","subj","Rest1",2,"",5,20,True,""],
        ["s","subj","Rest2",1,"",5,24,True,""], ["s","subj","Rest2",2,"",5,24,True,""]
    ], columns=rest_cols).to_csv(root / "rest_epochs.csv", index=False, encoding="utf-8-sig")
    (root / "session_info.txt").write_text("Subject: subj\nSampling rate: 500 Hz\n", encoding="utf-8")


def test_exact_v015_schema_extracts_168_features(tmp_path):
    session = tmp_path / "subject"
    session.mkdir()
    _write_session(session)
    cfg = json.loads(Path("configs/lab_replication.json").read_text(encoding="utf-8"))
    cfg["expected_epochs_per_class"] = 2
    out = tmp_path / "features"
    df, validation = extract_all(tmp_path, cfg, out)
    assert len(df) == 8
    assert len(feature_columns(df, "all")) == 168
    assert len(feature_columns(df, "bp")) == 42
    assert len(feature_columns(df, "coh")) == 126
    assert np.isfinite(df[feature_columns(df, "all")].to_numpy()).all()
    assert (out / "subj_features.csv").exists()
    assert (out / "features_all.csv").exists()
