"""Shared, fail-closed ingestion checks used by every feature extractor."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Literal

import numpy as np
import pandas as pd

from .io import SessionData, discover_session_dirs, load_session
from .schema import SCHEMA_VERSION, canonical_json, stable_id

EXCLUSION_COLUMNS = ["cohort_id", "subject_id", "session_id", "recording_id", "task_type",
                     "attempt_index", "trial_key", "reason_code", "trial_source_sha256"]


@dataclass(frozen=True)
class QAFinding:
    code: str
    severity: Literal["error", "warning", "policy"]
    message: str


class QAError(ValueError):
    def __init__(self, report: "QAReport"):
        self.report = report
        errors = [f"{f.code}: {f.message}" for f in report.findings if f.severity == "error"]
        super().__init__("Session QA failed: " + "; ".join(errors))


@dataclass
class QAReport:
    findings: list[QAFinding] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)
    exclusions: list[dict[str, Any]] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return not any(f.severity == "error" for f in self.findings)

    @property
    def reason_codes(self) -> list[str]:
        return sorted({f.code for f in self.findings})

    @property
    def status(self) -> str:
        if not self.passed:
            return "failed"
        return "passed_with_warnings" if any(f.severity == "warning" for f in self.findings) else "passed"

    def raise_for_errors(self) -> None:
        if not self.passed:
            raise QAError(self)

    def as_dict(self) -> dict[str, Any]:
        return {**self.metrics, "qa_pass": self.passed, "qc_status": self.status,
                "error_codes": ";".join(f.code for f in self.findings if f.severity == "error"),
                "warning_codes": ";".join(f.code for f in self.findings if f.severity == "warning"),
                "policy_codes": ";".join(f.code for f in self.findings if f.severity == "policy"),
                "excluded_trial_count": len(self.exclusions),
                "excluded_trials_json": canonical_json(self.exclusions)}


def validate_session(session: SessionData, cfg: Dict) -> QAReport:
    report = QAReport()

    def finding(code: str, message: str, severity: Literal["error", "warning", "policy"] = "error") -> None:
        report.findings.append(QAFinding(code, severity, message))

    meta = session.metadata
    report.metrics.update(subject_id=session.subject_id, session_id=meta.get("session_id", "unknown"),
                          cohort_id=meta.get("cohort_id", "unknown"), n_samples=len(session.raw),
                          sampling_rate_hz=session.sampling_rate_hz)
    mandatory = ("schema_version", "cohort_id", "subject_id", "session_id", "recording_id",
                 "source_sha256", "units", "reference", "protocol_id", "protocol_version")
    for key in mandatory:
        value = meta.get(key)
        if not isinstance(value, str) or not value.strip() or value != value.strip():
            finding("METADATA_REQUIRED", f"session_metadata.json must explicitly declare {key}")
    if meta.get("schema_version") != SCHEMA_VERSION:
        finding("SCHEMA_VERSION", "Raw ingestion requires schema_version='2.0'")
    if meta.get("subject_id") != session.subject_id:
        finding("IDENTITY_MISMATCH", "Session object and metadata disagree on subject_id")
    for key in ("units", "reference"):
        if meta.get(key) == "unknown":
            finding(key.upper() + "_UNKNOWN", f"{key} is explicitly unknown; no value was inferred", "warning")
    for table_name, frame in (("events", session.events), ("trials", session.trials), ("rest_epochs", session.rest_epochs)):
        for key in ("subject_id", "session_id"):
            if key not in frame or (len(frame) and not frame[key].eq(meta.get(key)).all()):
                finding("IDENTITY_MISMATCH", f"{table_name}.{key} disagrees with session metadata")

    policy = cfg.get("ingestion", {})
    marker_policy = policy.get("marker_policy", "hardware_required")
    clock_policy = policy.get("clock_policy", "session_start")
    timestamp_policy = policy.get("timestamp_policy", "require_utc")
    if marker_policy not in {"hardware_required", "software_timing"}:
        finding("UNKNOWN_POLICY", "marker_policy must be hardware_required or software_timing")
    if clock_policy not in {"session_start", "legacy_zero"}:
        finding("UNKNOWN_POLICY", "clock_policy must be session_start or legacy_zero")
    if timestamp_policy not in {"require_utc", "elapsed_only"}:
        finding("UNKNOWN_POLICY", "timestamp_policy must be require_utc or elapsed_only")
    if marker_policy == "software_timing":
        finding("POLICY_SOFTWARE_TIMING", "Software event timing is explicitly permitted", "policy")
    if clock_policy == "legacy_zero":
        finding("POLICY_LEGACY_ZERO", "Missing SessionStart explicitly uses legacy elapsed origin zero", "policy")
    if timestamp_policy == "elapsed_only":
        finding("POLICY_ELAPSED_ONLY", "Missing UTC cross-checks are explicitly permitted", "policy")

    fs = float(session.sampling_rate_hz)
    if not np.isfinite(fs) or fs <= 0:
        finding("INVALID_FS", "Sampling rate must be finite and positive")
        return report
    configured_fs = float(cfg.get("sampling_rate_hz", fs))
    if not np.isclose(fs, configured_fs, rtol=0, atol=1e-6):
        finding("FS_MISMATCH", "Session and configured sampling rates differ")
    sample_tolerance = float(policy.get("sample_period_tolerance_s", max(1e-9, 0.001 / fs)))
    clock_tolerance = float(policy.get("clock_tolerance_s", max(0.005, 1.0 / fs)))
    if not np.isfinite(sample_tolerance) or sample_tolerance < 0 or not np.isfinite(clock_tolerance) or clock_tolerance < 0:
        finding("INVALID_TOLERANCE", "Clock tolerances must be finite and nonnegative")
        return report
    raw = session.raw
    channels = list(cfg.get("raw_channels", []))
    analysis = list(cfg.get("analysis_channels", []))
    if len(set(channels)) != len(channels) or not set(analysis).issubset(channels):
        finding("CHANNEL_CONFIG", "Channels must be unique and analysis_channels a subset of raw_channels")
    missing = [col for col in ["sample_index", "elapsed_s", *channels] if col not in raw]
    if missing:
        finding("MISSING_RAW_COLUMNS", f"Missing raw columns: {missing}")
        return report
    declared_order = meta.get("channel_order")
    observed_order = [col for col in raw.columns if col in channels]
    if not isinstance(declared_order, list) or declared_order != observed_order or declared_order != channels:
        finding("CHANNEL_ORDER", "Declared, observed and configured raw channel order must agree")
    if len(raw) < 2:
        finding("RAW_TOO_SHORT", "At least two raw samples are required")
        return report
    index = pd.to_numeric(raw["sample_index"], errors="coerce").to_numpy(float)
    elapsed = pd.to_numeric(raw["elapsed_s"], errors="coerce").to_numpy(float)
    values = raw[channels].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    index_ok = (np.isfinite(index).all() and np.all(index == np.floor(index)) and
                index[0] == 0 and np.all(np.diff(index) == 1))
    report.metrics["sample_index_contiguous"] = bool(index_ok)
    if not index_ok:
        finding("SAMPLE_INDEX", "sample_index must be finite contiguous integers starting at zero")
    dt = np.diff(elapsed)
    time_ok = (np.isfinite(elapsed).all() and abs(elapsed[0]) <= sample_tolerance and
               np.all(dt > 0) and np.all(np.abs(dt - 1.0 / fs) <= sample_tolerance))
    if not time_ok:
        finding("SAMPLE_CLOCK", "Raw elapsed_s must start at zero and every sample period must match Fs")
    median_dt = float(np.median(dt))
    report.metrics["inferred_sampling_rate_hz"] = 1.0 / median_dt if np.isfinite(median_dt) and median_dt > 0 else None
    finite = bool(np.isfinite(values).all())
    report.metrics["all_values_finite"] = finite
    if not finite:
        finding("NONFINITE_SIGNAL", "Raw EEG contains missing or nonfinite samples")
    flat_epsilon = float(policy.get("flat_std_epsilon", 1e-15))
    if not np.isfinite(flat_epsilon) or flat_epsilon < 0:
        finding("INVALID_TOLERANCE", "flat_std_epsilon must be finite and nonnegative")
        flat_epsilon = 1e-15
    channel_std = np.std(values, axis=0) if finite else np.full(len(channels), np.nan)
    flat = [ch for ch, std in zip(channels, channel_std) if std <= flat_epsilon]
    report.metrics["flat_channels"] = ";".join(flat)
    if flat:
        finding("FLAT_CHANNEL", "Flat raw channels: " + ",".join(flat))
    report.metrics["raw_abs_max"] = float(np.max(np.abs(values))) if finite else None
    report.metrics["raw_std_min"] = float(np.min(channel_std)) if finite else None

    events = session.events
    trials = session.trials
    rests = session.rest_epochs
    for name, frame, columns in (("events", events, ["marker_code", "attempt_index", "elapsed_ms"]),
                                ("trials", trials, ["attempt_index"]),
                                ("rest_epochs", rests, ["rest_epoch_index", "marker_code", "epoch_seconds"])):
        for col in columns:
            if col not in frame:
                finding("MISSING_EVENT_COLUMNS", f"{name}.{col} is required")
                return report
            nums = pd.to_numeric(frame[col], errors="coerce").to_numpy(float)
            if not np.isfinite(nums).all() or np.any(nums < 0) or (
                col not in {"elapsed_ms", "epoch_seconds"} and np.any(nums != np.floor(nums))
            ):
                finding("INVALID_EVENT_NUMBER", f"{name}.{col} must contain valid finite nonnegative numbers")
                return report
    for name, frame, columns in (("events", events, ["marker_sent"]), ("trials", trials, ["analysis_eligible", "response_correct"]),
                                ("rest_epochs", rests, ["marker_sent"])):
        for col in columns:
            if col not in frame or not frame[col].map(lambda value: isinstance(value, (bool, np.bool_))).all():
                finding("INVALID_BOOLEAN", f"{name}.{col} requires unambiguous boolean values")
                return report
    all_markers = bool(events["marker_sent"].all() and rests["marker_sent"].all())
    report.metrics["all_markers_sent"] = all_markers
    if not all_markers:
        finding("MARKER_NOT_SENT", "At least one hardware marker was not sent",
                "warning" if marker_policy == "software_timing" else "error")
    event_ms = pd.to_numeric(events["elapsed_ms"]).to_numpy(float)
    if len(events) == 0 or np.any(np.diff(event_ms) < 0):
        finding("EVENT_CLOCK", "Events must be nonempty and ordered by a nondecreasing elapsed clock")
    starts = events.loc[events["marker_code"] == 1]
    start_phases = events.loc[events["phase"] == "SessionStart"]
    if not starts.index.equals(start_phases.index):
        finding("SESSION_START_IDENTITY", "SessionStart phase and marker 1 must identify exactly the same events")
    if len(starts) > 1 or (len(starts) != 1 and clock_policy != "legacy_zero"):
        finding("SESSION_START", "Exactly one SessionStart marker is required")
    zero_ms = float(starts["elapsed_ms"].iloc[0]) if len(starts) == 1 else 0.0
    if len(starts) == 1 and len(events) and event_ms.min() < zero_ms:
        finding("EVENT_BEFORE_START", "Events precede SessionStart")

    def utc_values(frame: pd.DataFrame, column: str, label: str) -> pd.Series:
        if column not in frame:
            finding("TIMESTAMP_REQUIRED", f"{label}.{column} is required")
            return pd.Series(pd.NaT, index=frame.index, dtype="datetime64[ns, UTC]")
        text = frame[column].astype(str)
        supplied = text.str.strip().ne("")
        timezone_explicit = text.str.contains(r"(?:[zZ]|[+-]\d{2}:?\d{2})$", regex=True)
        if (supplied & ~timezone_explicit).any():
            finding("TIMESTAMP_TIMEZONE", f"{label}.{column} must state a UTC offset; naive timestamps are not assumed UTC")
        parsed = pd.to_datetime(text.where(supplied), utc=True, errors="coerce", format="mixed")
        if (supplied & parsed.isna()).any() or (timestamp_policy == "require_utc" and parsed.isna().any()):
            finding("TIMESTAMP_REQUIRED", f"{label}.{column} requires valid UTC timestamps")
        return parsed

    event_utc = utc_values(events, "utc_timestamp", "events")
    trial_utc = utc_values(trials, "calculation_onset", "trials")
    rest_utc = utc_values(rests, "onset_utc", "rest_epochs")
    valid_times = event_utc.notna()
    if valid_times.any():
        first = event_utc.loc[valid_times].index[0]
        clock_delta = (event_utc - event_utc.loc[first]).dt.total_seconds()
        elapsed_delta = (pd.to_numeric(events["elapsed_ms"]) - float(events.loc[first, "elapsed_ms"])) / 1000.0
        if ((clock_delta - elapsed_delta).abs().loc[valid_times] > clock_tolerance).any():
            finding("UTC_CLOCK_MISMATCH", "Event UTC and elapsed clocks disagree")

    epoch_seconds = float(cfg["epoch_seconds"])
    samples_exact = epoch_seconds * fs
    if not np.isfinite(epoch_seconds) or epoch_seconds <= 0 or abs(samples_exact - round(samples_exact)) > 1e-8:
        finding("EPOCH_LENGTH", "epoch_seconds * Fs must be a positive integer")
        return report
    epoch_samples = int(round(samples_exact))
    expected = cfg.get("expected_epochs_per_class")
    intervals: list[tuple[int, int]] = []
    count_passes = []
    known_types = {str(comp["task_type"]) for comp in cfg["comparisons"]}
    task_markers = {str(comp["task_type"]): int(comp["task_marker"]) for comp in cfg["comparisons"]}
    rest_markers = {int(comp["rest_marker"]) for comp in cfg["comparisons"]}
    for _, event in events.loc[events["phase"] == "Calculation"].iterrows():
        typ = str(event["task_type"])
        if typ not in task_markers or int(event["marker_code"]) != task_markers[typ] or str(event["stage"]) != typ:
            finding("EXTRA_CALCULATION_EVENT", "Calculation event must match a configured task type, stage and marker")
    if not set(events.loc[events["phase"] == "Rest", "marker_code"].astype(int)).issubset(rest_markers):
        finding("EXTRA_REST_EVENT", "Rest event marker is absent from configured comparisons")
    if not set(rests["marker_code"].astype(int)).issubset(rest_markers):
        finding("EXTRA_REST_MANIFEST", "Rest manifest contains an unconfigured marker")
    if not set(trials["task_type"].astype(str)).issubset(known_types):
        finding("UNKNOWN_TASK_TYPE", "Trials contain a task type absent from the protocol config")
    if (trials["attempt_index"] <= 0).any():
        finding("TRIAL_ATTEMPT_INDEX", "Task attempt indices must be positive integers")
    if trials.duplicated(["task_type", "attempt_index"]).any():
        finding("DUPLICATE_TRIAL", "A task_type/attempt_index has multiple trial records")
    if rests.duplicated(["rest_stage", "rest_epoch_index"]).any():
        finding("DUPLICATE_REST", "A rest stage/index has multiple manifest records")

    def check_onsets(event_indices: list, other_indices: list, other_times: pd.Series, label: str) -> None:
        for a, b in zip(event_indices, other_indices):
            if pd.notna(event_utc.loc[a]) and pd.notna(other_times.loc[b]):
                delta = abs((event_utc.loc[a] - other_times.loc[b]).total_seconds())
                if delta > clock_tolerance:
                    finding("ONSET_MISMATCH", f"{label} onset disagrees with its event")

    for comp in cfg["comparisons"]:
        name, typ = str(comp["name"]), str(comp["task_type"])
        rest_rows = events.loc[events["marker_code"] == int(comp["rest_marker"])].sort_values("elapsed_ms", kind="stable")
        task_all = events.loc[events["marker_code"] == int(comp["task_marker"])].sort_values("elapsed_ms", kind="stable")
        trial_rows = trials.loc[trials["task_type"] == typ]
        eligible = trial_rows.loc[trial_rows["analysis_eligible"] & trial_rows["response_correct"]]
        # These categories are disjoint: ineligibility takes precedence over an incorrect response.
        ineligible = ~trial_rows["analysis_eligible"]
        incorrect = trial_rows["analysis_eligible"] & ~trial_rows["response_correct"]
        report.metrics.update({f"{name}_total_attempts": len(trial_rows),
                               f"{name}_selected_attempts": len(eligible),
                               f"{name}_excluded_ineligible": int(ineligible.sum()),
                               f"{name}_excluded_incorrect": int(incorrect.sum())})
        for mask, reason in ((ineligible, "TRIAL_INELIGIBLE"), (incorrect, "INCORRECT_RESPONSE")):
            for _, trial in trial_rows.loc[mask].iterrows():
                scope = {key: meta.get(key, "unknown") for key in ("cohort_id", "subject_id", "session_id", "recording_id")}
                identity = {**scope, "task_type": typ, "attempt_index": int(trial["attempt_index"])}
                report.exclusions.append({**identity, "trial_key": stable_id("trial", identity),
                                          "reason_code": reason,
                                          "trial_source_sha256": session.source_hashes.get("trials.csv", "unknown")})
        if (len(task_all) and (not task_all["task_type"].eq(typ).all() or not task_all["phase"].eq("Calculation").all())):
            finding("TASK_EVENT_IDENTITY", f"{name}: task marker has inconsistent task_type or phase")
        if task_all["attempt_index"].duplicated().any():
            finding("DUPLICATE_TASK_EVENT", f"{name}: duplicate calculation event for an attempt")
        event_attempts = set(task_all["attempt_index"].astype(int))
        trial_attempts = set(trial_rows["attempt_index"].astype(int))
        if event_attempts != trial_attempts:
            finding("TASK_EVENT_TRIAL_SET", f"{name}: missing or extra task events/trials")
        if len(task_all) == len(trial_rows) and not task_all["attempt_index"].duplicated().any() and not trial_rows["attempt_index"].duplicated().any():
            lookup = {int(row.attempt_index): index for index, row in trial_rows.iterrows()}
            if event_attempts == trial_attempts:
                check_onsets(task_all.index.tolist(), [lookup[int(a)] for a in task_all["attempt_index"]], trial_utc, name)
        task_rows = task_all.loc[task_all["attempt_index"].isin(set(eligible["attempt_index"]))]
        if len(task_rows) != len(eligible):
            finding("ELIGIBLE_EVENT_COUNT", f"{name}: eligible trials and calculation events are not one-to-one")
        if len(rest_rows) and not rest_rows["phase"].eq("Rest").all():
            finding("REST_EVENT_IDENTITY", f"{name}: rest marker has inconsistent phase")
        rest_manifest = rests.loc[rests["marker_code"] == int(comp["rest_marker"])].sort_values("rest_epoch_index", kind="stable")
        if len(rest_manifest) != len(rest_rows):
            finding("REST_MANIFEST_COUNT", f"{name}: rest manifest and event counts differ")
        elif len(rest_rows):
            if rest_manifest["rest_epoch_index"].tolist() != list(range(1, len(rest_rows) + 1)):
                finding("REST_EPOCH_INDEX", f"{name}: rest indices must enumerate the events exactly")
            if rest_manifest["rest_stage"].astype(str).tolist() != rest_rows["stage"].astype(str).tolist():
                finding("REST_STAGE", f"{name}: rest stage identity differs between manifest and events")
            check_onsets(rest_rows.index.tolist(), rest_manifest.index.tolist(), rest_utc, name)
        if len(rest_manifest) and not np.allclose(rest_manifest["epoch_seconds"].to_numpy(float), epoch_seconds, rtol=0, atol=1e-9):
            finding("REST_DURATION", f"{name}: rest duration differs from configured epoch duration")
        counts_ok = expected is None or (len(rest_rows) == int(expected) and len(task_rows) == int(expected))
        count_passes.append(counts_ok)
        report.metrics.update({f"{name}_rest_epochs": len(rest_rows), f"{name}_ma_correct_epochs": len(task_rows), f"{name}_count_pass": counts_ok})
        if not counts_ok:
            finding("PROTOCOL_COUNTS", f"{name}: expected {expected} rest and eligible task epochs")
        for _, row in pd.concat([rest_rows, task_rows]).iterrows():
            start_exact = (float(row["elapsed_ms"]) - zero_ms) * fs / 1000.0
            start = int(round(start_exact))
            end = start + epoch_samples
            if start < 0 or end > len(raw):
                finding("EPOCH_BOUNDS", f"{name}: selected epoch falls outside raw samples")
            intervals.append((start, end))
    sorted_intervals = sorted(intervals)
    if any(b[0] < a[1] for a, b in zip(sorted_intervals, sorted_intervals[1:])):
        finding("OVERLAPPING_EPOCHS", "Selected raw intervals overlap or repeat")
    report.metrics["protocol_counts_pass"] = all(count_passes)
    session.qa_report = report
    return report


def exclusion_table(sessions: list[SessionData]) -> pd.DataFrame:
    return pd.DataFrame([row for session in sessions for row in session.qa_report.exclusions], columns=EXCLUSION_COLUMNS)


def validate_protocol(session: SessionData, cfg: Dict) -> Dict[str, object]:
    """Compatibility reporting wrapper; callers should use QAReport for gating."""
    return validate_session(session, cfg).as_dict()


def preflight_sessions(raw_root: str | Path, cfg: Dict) -> list[SessionData]:
    paths = discover_session_dirs(raw_root)
    if not paths:
        raise FileNotFoundError(f"No eeg_raw.csv found under {raw_root}")
    sessions = []
    identities = set()
    for path in paths:
        session = load_session(path, configured_fs=float(cfg["sampling_rate_hz"]),
                               metadata=cfg.get("ingestion", {}).get("metadata"))
        report = validate_session(session, cfg)
        report.raise_for_errors()
        identity = tuple(session.metadata[key] for key in ("cohort_id", "subject_id", "session_id"))
        if identity in identities:
            raise ValueError("DUPLICATE_SESSION: multiple raw directories describe the same cohort/subject/session")
        identities.add(identity)
        sessions.append(session)
    return sessions
