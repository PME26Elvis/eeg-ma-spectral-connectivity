"""Explicit synthetic fixed-model fixtures; none is real study evidence."""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from eeg_ma.provenance import fingerprint, sha256_file
from eeg_ma.schema import META_COLUMNS, make_epoch_metadata, stable_id

ROOT = Path(__file__).resolve().parents[1]


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def make_feature_table(subjects, cohort_id, *, sessions=2, epochs_per_class=4, seed=1109,
                       protocol_id="eeg-ma-A-capped-v1", protocol_version="1"):
    """Produce schema2 synthetic metadata and all189 prespecified candidate values."""
    schema = json.loads((ROOT / "configs/confirmatory_feature_schema.json").read_text())
    rng = np.random.default_rng(seed)
    rows = []
    for subject in subjects:
        for session_number in range(1, sessions + 1):
            session = f"session-{session_number}"
            raw_hash = hashlib.sha256(f"SYNTHETIC-{cohort_id}-{subject}-{session}".encode()).hexdigest()
            metadata = {"cohort_id": cohort_id, "subject_id": subject, "session_id": session,
                "source_sha256": raw_hash, "units": "uV", "reference": "SYNTHETIC reference",
                "identity_origin": "raw_metadata", "protocol_id": protocol_id, "protocol_version": protocol_version,
                "channel_order": [*schema["analysis_channels"], "Cz"]}
            metadata["recording_id"] = stable_id("recording", {"cohort_id": cohort_id, "subject_id": subject,
                "session_id": session, "raw_sha256": raw_hash})
            for comparison_number, comparison in enumerate(("rest1_vs_type1", "rest2_vs_type2")):
                for label in (0, 1):
                    for epoch in range(1, epochs_per_class + 1):
                        position = (comparison_number * 2 + label) * epochs_per_class + epoch - 1
                        row = {"subject_id": subject, "comparison": comparison, "class_label": label,
                            "class_name": "Rest" if label == 0 else "MentalArithmetic",
                            "stage": f"Rest{comparison_number + 1}" if label == 0 else f"Type{comparison_number + 1}",
                            "epoch_index": epoch, "attempt_index": 0 if label == 0 else epoch,
                            "onset_elapsed_s": position * 5.0, "start_sample": position * 2500, "end_sample": (position + 1) * 2500}
                        row = make_epoch_metadata(metadata, row, qc_status="passed", qc_reason_codes=[])
                        for column in schema["reference_features"]:
                            noise = rng.normal()
                            row[column] = float(2 + 0.5 * label + 0.3 * noise) if column.startswith("BP__") else float(np.clip(0.3 + .2 * label + .08 * noise, .01, .99))
                        for column in schema["primary_features"]:
                            row[column] = float(np.clip(.25 + .4 * label + .09 * rng.normal(), .01, .99))
                        rows.append(row)
    return pd.DataFrame(rows)[META_COLUMNS + schema["reference_features"] + schema["primary_features"]]


def write_development_fixture(tmp_path, *, frame=None):
    """Write a complete hash-bound synthetic source protocol before fitting.

    Acquisition and approvals are explicitly marked SYNTHETIC. Tests still
    need a clean-code/time override when exercising formal protocol locks.
    """
    root = Path(tmp_path)
    root.mkdir(parents=True, exist_ok=True)
    spec = json.loads((ROOT / "configs/confirmatory_protocol.json").read_text())
    cfg = json.loads((ROOT / "configs/confirmatory_analysis.json").read_text())
    schema = json.loads((ROOT / "configs/confirmatory_feature_schema.json").read_text())
    cfg["sfs"]["max_features"] = 1
    cfg["n_jobs"] = 1
    if frame is None:
        frame = make_feature_table(["001", "1", "NA"], "synthetic-development")
    else:
        frame = frame.copy()
    spec["acquisition"].update(units="uV", reference="SYNTHETIC reference", ground="SYNTHETIC ground",
        device="SYNTHETIC test device", evidence="SYNTHETIC verification record", verified=True, historical_alignment_verified=True)
    spec["approval"].update(responsible_person="SYNTHETIC test operator", approved_at="2019-12-01T00:00:00Z",
        ethics_status="SYNTHETIC test approval", ethics_reference="SYNTHETIC fixture only", resource_commitment_confirmed=True)
    artifacts = spec["freeze"]["artifacts"]
    def artifact(role, value):
        path = root / f"{role}.json"
        write_json(path, value)
        artifacts[role] = {"path": path.name, "sha256": sha256_file(path)}
        return path
    artifact("analysis_config", cfg)
    artifact("feature_schema", schema)
    for role, names in (("baseline", schema["reference_features"]), ("extensions", schema["primary_features"])):
        frame[META_COLUMNS + names].to_csv(root / f"{role}.csv", index=False)
    artifact("development_data_manifest", {"schema_version": "eeg-development-data-1",
        "sources": [{"name": name, "path": f"{name}.csv", "sha256": sha256_file(root / f"{name}.csv")}
            for name in ("baseline", "extensions")], "historical_raw_sha256": "unknown", "scope": "SYNTHETIC test inputs"})
    artifact("development_manifest", {"schema_version": "eeg-development-manifest-1", "cohort_id": str(frame.cohort_id.iloc[0]),
        "subject_ids": sorted(frame.subject_id.unique()), "data_sha256": artifacts["development_data_manifest"]["sha256"],
        "feature_schema_sha256": artifacts["feature_schema"]["sha256"], "use": "development_only", "scope": "SYNTHETIC test input"})
    artifact("acquisition_evidence", {"schema_version": "eeg-acquisition-verification-1", "verified": True,
        "acquisition_sha256": fingerprint(spec["acquisition"]), "verified_by": "SYNTHETIC test operator", "verified_at": "2019-12-01T00:00:00Z"})
    precision = ROOT / "docs/precision/precision_report.json"
    artifacts["precision_report"] = {"path": str(precision), "sha256": sha256_file(precision)}
    path = root / "protocol.json"
    write_json(path, spec)
    return path, spec, frame
