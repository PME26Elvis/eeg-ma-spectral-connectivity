from pathlib import Path
import json

import numpy as np
import pandas as pd

from eeg_ma.io import SessionData
from eeg_ma.preprocess import extract_protocol_epochs


def test_event_markers_and_eligibility_drive_epoching():
    cfg = json.loads(Path('configs/lab_replication.json').read_text(encoding='utf-8'))
    cfg['expected_epochs_per_class'] = 2
    fs = 500
    raw = pd.DataFrame({'sample_index': np.arange(16000), 'elapsed_s': np.arange(16000)/fs})
    for ch in ['FP1','FP2','F7','F3','Fz','F4','F8','Cz']:
        raw[ch] = 0.0

    events = pd.DataFrame([
        ['s','x','Rest1','Rest','',0,'',1000,20,True,''],
        ['s','x','Rest1','Rest','',0,'',7000,20,True,''],
        ['s','x','Type1','Calculation','Type1',1,'',13000,11,True,''],
        ['s','x','Type1','Calculation','Type1',2,'',19000,11,True,''],
        ['s','x','Type1','Calculation','Type1',3,'',25000,11,True,''],
        ['s','x','Rest2','Rest','',0,'',31000,24,True,''],
        ['s','x','Rest2','Rest','',0,'',37000,24,True,''],
        ['s','x','Type2','Calculation','Type2',1,'',43000,12,True,''],
        ['s','x','Type2','Calculation','Type2',2,'',49000,12,True,''],
    ], columns=['session_id','subject_id','stage','phase','task_type','attempt_index','utc_timestamp','elapsed_ms','marker_code','marker_sent','marker_message'])
    trials = pd.DataFrame([
        ['s','x','Type1',1,True,True,''],
        ['s','x','Type1',2,False,False,''],
        ['s','x','Type1',3,True,True,''],
        ['s','x','Type2',1,True,True,''],
        ['s','x','Type2',2,True,True,''],
    ], columns=['session_id','subject_id','task_type','attempt_index','analysis_eligible','response_correct','calculation_onset'])
    rest = pd.DataFrame(columns=['session_id','subject_id','rest_stage','rest_epoch_index','onset_utc','epoch_seconds','marker_code','marker_sent','marker_message'])
    session = SessionData(Path('.'), 'x', fs, raw, events, trials, rest, '')
    filtered = np.zeros((len(raw), 7))

    # Scale event times down to fit synthetic raw while preserving order.
    session.events['elapsed_ms'] = [0, 3000, 6000, 9000, 12000, 15000, 18000, 21000, 24000]
    records = extract_protocol_epochs(session, filtered, cfg)
    assert len(records) == 8  # 2 rest + 2 eligible MA per comparison
    type1_attempts = [r.attempt_index for r in records if r.comparison == 'rest1_vs_type1' and r.class_label == 1]
    assert type1_attempts == [1, 3]
