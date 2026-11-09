from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from .qa import preflight_sessions
from .schema import SCHEMA_VERSION


def summarize_behavior(raw_root: str | Path, cfg: Dict) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for s in preflight_sessions(raw_root, cfg):
        for task_type in ["Type1", "Type2"]:
            d = s.trials.loc[s.trials["task_type"].astype(str) == task_type].copy()
            if d.empty:
                continue
            correct = d["response_correct"].astype(bool)
            rt = pd.to_numeric(d.loc[correct, "reaction_time_ms"], errors="coerce")
            rows.append({
                "schema_version": SCHEMA_VERSION,
                **{key: s.metadata[key] for key in (
                    "cohort_id", "session_id", "recording_id", "source_sha256",
                    "protocol_id", "protocol_version", "identity_origin",
                )},
                "subject_id": s.subject_id,
                "task_type": task_type,
                "attempts": int(len(d)),
                "correct": int(correct.sum()),
                "accuracy": float(correct.mean()),
                "error_rate": float(1.0 - correct.mean()),
                "reaction_time_ms_mean_correct": float(rt.mean()) if len(rt) else np.nan,
                "reaction_time_ms_median_correct": float(rt.median()) if len(rt) else np.nan,
                "attempts_to_30_correct": int(len(d)) if int(correct.sum()) >= 30 else np.nan,
            })
    return pd.DataFrame(rows)
