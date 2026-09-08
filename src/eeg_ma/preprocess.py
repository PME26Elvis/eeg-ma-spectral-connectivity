from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy import signal

from .io import SessionData


@dataclass
class EpochRecord:
    subject_id: str
    comparison: str
    class_label: int
    class_name: str
    stage: str
    epoch_index: int
    attempt_index: int
    onset_elapsed_s: float
    start_sample: int
    end_sample: int
    data: np.ndarray  # shape: samples x channels


def filter_continuous(raw_values: np.ndarray, fs: float, cfg: Dict) -> np.ndarray:
    """Filter continuous EEG before epoching to avoid per-epoch edge artifacts."""
    x = np.asarray(raw_values, dtype=float)
    if x.ndim != 2:
        raise ValueError("raw_values 必須是 [samples, channels]")

    notch_hz = cfg.get("notch_hz", None)
    y = x.copy()
    if notch_hz is not None:
        q = 30.0
        b, a = signal.iirnotch(float(notch_hz), q, fs=float(fs))
        y = signal.filtfilt(b, a, y, axis=0)

    bp = cfg["bandpass"]
    sos = signal.butter(
        int(bp.get("order", 4)),
        [float(bp["low_hz"]), float(bp["high_hz"])],
        btype="bandpass",
        fs=float(fs),
        output="sos",
    )
    return signal.sosfiltfilt(sos, y, axis=0)


def extract_protocol_epochs(session: SessionData, filtered: np.ndarray, cfg: Dict) -> List[EpochRecord]:
    fs = float(session.sampling_rate_hz)
    epoch_samples = int(round(float(cfg["epoch_seconds"]) * fs))
    events = session.events.copy()
    trials = session.trials.copy()

    # The HNC panel clears EegData_Raw immediately before sending the formal
    # SessionStart marker. ProtocolController's stopwatch starts slightly earlier,
    # so align raw sample 0 to the SessionStart event rather than using absolute
    # elapsed_ms directly. This removes the small setup offset while keeping all
    # epoch onsets on the acquisition clock.
    session_start_rows = events.loc[events["marker_code"].astype(int) == 1]
    timeline_zero_ms = float(session_start_rows["elapsed_ms"].iloc[0]) if len(session_start_rows) else 0.0

    records: List[EpochRecord] = []
    for comp in cfg["comparisons"]:
        name = str(comp["name"])
        rest_marker = int(comp["rest_marker"])
        task_marker = int(comp["task_marker"])
        task_type = str(comp["task_type"])

        rest_rows = events.loc[events["marker_code"].astype(int) == rest_marker].sort_values("elapsed_ms")
        eligible = trials.loc[
            (trials["task_type"].astype(str) == task_type)
            & trials["analysis_eligible"].astype(bool)
            & trials["response_correct"].astype(bool)
        ]
        eligible_attempts = set(eligible["attempt_index"].astype(int).tolist())

        task_rows = events.loc[
            (events["marker_code"].astype(int) == task_marker)
            & (events["task_type"].astype(str) == task_type)
            & (events["attempt_index"].astype(int).isin(eligible_attempts))
        ].sort_values("elapsed_ms")

        _check_expected_count(session, name, "Rest", rest_rows, cfg)
        _check_expected_count(session, name, "MA", task_rows, cfg)

        for i, (_, row) in enumerate(rest_rows.iterrows(), start=1):
            rec = _slice_record(
                session, filtered, fs, epoch_samples, name, 0, "Rest",
                str(row["stage"]), i, 0, (float(row["elapsed_ms"]) - timeline_zero_ms) / 1000.0,
            )
            records.append(rec)

        for i, (_, row) in enumerate(task_rows.iterrows(), start=1):
            rec = _slice_record(
                session, filtered, fs, epoch_samples, name, 1, "MentalArithmetic",
                str(row["stage"]), i, int(row["attempt_index"]), (float(row["elapsed_ms"]) - timeline_zero_ms) / 1000.0,
            )
            records.append(rec)

    return records


def _slice_record(
    session: SessionData,
    filtered: np.ndarray,
    fs: float,
    epoch_samples: int,
    comparison: str,
    class_label: int,
    class_name: str,
    stage: str,
    epoch_index: int,
    attempt_index: int,
    onset_s: float,
) -> EpochRecord:
    start = int(round(onset_s * fs))
    end = start + epoch_samples
    if start < 0 or end > len(filtered):
        raise ValueError(
            f"{session.subject_id} {comparison} {class_name} epoch {epoch_index}: "
            f"範圍 [{start},{end}) 超出 raw EEG 長度 {len(filtered)}"
        )
    data = np.asarray(filtered[start:end], dtype=float)
    if data.shape[0] != epoch_samples:
        raise ValueError("epoch sample 數不完整")
    return EpochRecord(
        subject_id=session.subject_id,
        comparison=comparison,
        class_label=class_label,
        class_name=class_name,
        stage=stage,
        epoch_index=epoch_index,
        attempt_index=attempt_index,
        onset_elapsed_s=onset_s,
        start_sample=start,
        end_sample=end,
        data=data,
    )


def _check_expected_count(session: SessionData, comparison: str, label: str, rows: pd.DataFrame, cfg: Dict) -> None:
    expected = cfg.get("expected_epochs_per_class", None)
    if expected is not None and len(rows) != int(expected):
        raise ValueError(
            f"{session.subject_id} {comparison}: {label} epochs={len(rows)}，預期 {expected}。"
            "請先檢查 events/trials 與 analysis_eligible。"
        )
