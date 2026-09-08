from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd

from .io import SessionData, validate_session_basic


def validate_protocol(session: SessionData, cfg: Dict) -> Dict[str, object]:
    row = validate_session_basic(session, cfg["raw_channels"])
    expected = int(cfg.get("expected_epochs_per_class", 30))

    counts = {}
    for comp in cfg["comparisons"]:
        name = str(comp["name"])
        rest_marker = int(comp["rest_marker"])
        task_marker = int(comp["task_marker"])
        task_type = str(comp["task_type"])
        rest_n = int((session.events["marker_code"].astype(int) == rest_marker).sum())
        elig = session.trials.loc[
            (session.trials["task_type"].astype(str) == task_type)
            & session.trials["analysis_eligible"].astype(bool)
            & session.trials["response_correct"].astype(bool)
        ]
        elig_attempts = set(elig["attempt_index"].astype(int))
        task_n = int((
            (session.events["marker_code"].astype(int) == task_marker)
            & session.events["attempt_index"].astype(int).isin(elig_attempts)
        ).sum())
        counts[f"{name}_rest_epochs"] = rest_n
        counts[f"{name}_ma_correct_epochs"] = task_n
        counts[f"{name}_count_pass"] = (rest_n == expected and task_n == expected)

    x = session.raw[cfg["raw_channels"]].to_numpy(dtype=float)
    row.update(counts)
    row["raw_abs_max"] = float(np.max(np.abs(x)))
    row["raw_std_min"] = float(np.min(np.std(x, axis=0)))
    row["protocol_counts_pass"] = all(v for k, v in counts.items() if k.endswith("_count_pass"))
    return row
