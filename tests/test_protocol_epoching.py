from pathlib import Path
import json

import numpy as np
import pandas as pd
import pytest

from eeg_ma.io import load_session
from eeg_ma.preprocess import extract_protocol_epochs
from eeg_ma.qa import QAError
from test_extract_end_to_end import _write_session


def test_event_markers_and_eligibility_drive_epoching(tmp_path):
    _write_session(tmp_path)
    cfg = json.loads(Path("configs/lab_replication.json").read_text(encoding="utf-8"))
    # The retained cohort is allowed to be unbalanced in this focused eligibility test.
    cfg["expected_epochs_per_class"] = None
    trials = pd.read_csv(tmp_path / "trials.csv", keep_default_na=False)
    excluded = (trials.task_type == "Type1") & (trials.attempt_index == 2)
    trials.loc[excluded, ["analysis_eligible", "response_correct"]] = False
    trials.to_csv(tmp_path / "trials.csv", index=False)
    session = load_session(tmp_path, configured_fs=500)
    records = extract_protocol_epochs(session, np.zeros((len(session.raw), 7)), cfg)
    assert len(records) == 7
    assert [r.attempt_index for r in records if r.comparison == "rest1_vs_type1" and r.class_label == 1] == [1]
    assert len({record.metadata["epoch_id"] for record in records}) == 7


def test_missing_session_start_requires_explicit_legacy_policy(tmp_path):
    _write_session(tmp_path)
    cfg = json.loads(Path("configs/lab_replication.json").read_text(encoding="utf-8"))
    cfg["expected_epochs_per_class"] = 2
    events = pd.read_csv(tmp_path / "events.csv", keep_default_na=False)
    events.loc[events.marker_code != 1].to_csv(tmp_path / "events.csv", index=False)
    session = load_session(tmp_path, configured_fs=500)
    filtered = np.zeros((len(session.raw), 7))
    with pytest.raises(QAError, match="SESSION_START"):
        extract_protocol_epochs(session, filtered, cfg)
    cfg["ingestion"] = {"clock_policy": "legacy_zero"}
    records = extract_protocol_epochs(session, filtered, cfg)
    assert len(records) == 8
    assert "POLICY_LEGACY_ZERO" in records[0].metadata["qc_reason_codes"]
