from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from .preprocess import extract_protocol_epochs, filter_continuous
from .features import extract_epoch_features
from .qa import preflight_sessions, exclusion_table
from .schema import (
    META_COLUMNS, SCHEMA_VERSION, assert_output_available, session_output_stem,
    validate_feature_table, write_feature_outputs,
)


def extract_all(raw_root: str | Path, cfg: Dict, out_dir: str | Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    out_dir = Path(out_dir)
    assert_output_available(out_dir)
    sessions = preflight_sessions(raw_root, cfg)

    all_frames: List[pd.DataFrame] = []
    validation_rows: List[Dict[str, object]] = []

    output_tables = {}
    for session in sessions:
        validation_rows.append(session.qa_report.as_dict())

        raw_x = session.raw[cfg["analysis_channels"]].to_numpy(dtype=float)
        filtered = filter_continuous(raw_x, session.sampling_rate_hz, cfg)
        epochs = extract_protocol_epochs(session, filtered, cfg)

        rows = []
        feature_names = None
        for ep in epochs:
            names, values = extract_epoch_features(
                ep.data, session.sampling_rate_hz, cfg["analysis_channels"], cfg
            )
            if feature_names is None:
                feature_names = names
            elif names != feature_names:
                raise AssertionError("Feature name ordering changed within session")
            row = dict(ep.metadata)
            row.update(dict(zip(names, values)))
            rows.append(row)

        frame = pd.DataFrame(rows)
        validate_feature_table(frame)
        output_tables[f"{session_output_stem(session.metadata)}_features.csv"] = frame
        all_frames.append(frame)

    combined = pd.concat(all_frames, ignore_index=True)
    validate_feature_table(combined)
    validation = pd.DataFrame(validation_rows)
    output_tables.update({"features_all.csv": combined, "raw_validation.csv": validation,
                          "excluded_trials.csv": exclusion_table(sessions)})
    manifest = {
        "schema_version": SCHEMA_VERSION, "feature_family": "baseline", "config": cfg,
        "n_rows": len(combined), "feature_columns": feature_columns(combined),
        "sessions": [{"metadata": s.metadata, "source_hashes": s.source_hashes,
                      "qa": s.qa_report.as_dict()} for s in sessions],
    }
    write_feature_outputs(out_dir, output_tables, manifest)
    return combined, validation


def feature_columns(df: pd.DataFrame, feature_set: str = "all") -> List[str]:
    """Resolve feature families without changing the historical baseline semantics.

    `all` intentionally remains the poster/lab baseline BP+COH set.  Every
    post-baseline representation uses an explicit feature-set name so reports
    cannot accidentally relabel an extension as the replication baseline.

    Final validation additionally supports `plv_band_<band>` selectors and
    `iplv`.  These are deliberately narrow hypothesis/robustness checks rather
    than new baseline definitions.
    """
    bp = [c for c in df.columns if c.startswith("BP__")]
    coh = [c for c in df.columns if c.startswith("COH__")]
    asym = [c for c in df.columns if c.startswith("ASYM_")]
    plv = [c for c in df.columns if c.startswith("PLV__")]
    iplv = [c for c in df.columns if c.startswith("IPLV__")]

    sets = {
        # Frozen Poster/Lab baseline semantics.
        "all": bp + coh,
        "baseline": bp + coh,
        # Single-family / original ablations.
        "bp": bp,
        "coh": coh,
        "asym": asym,
        "plv": plv,
        "iplv": iplv,
        # E1 / E2 extensions already evaluated.
        "baseline_asym": bp + coh + asym,
        "baseline_plv": bp + coh + plv,
        "baseline_asym_plv": bp + coh + asym + plv,
        "bp_asym": bp + asym,
        # PLV decomposition experiments.
        "bp_plv": bp + plv,
        "coh_plv": coh + plv,
        "asym_plv": asym + plv,
    }

    if feature_set.startswith("plv_band_"):
        band = feature_set[len("plv_band_"):]
        cols = [c for c in plv if c.endswith(f"__{band}")]
        if not cols:
            raise ValueError(f"feature_set={feature_set} 找不到該 band 的 PLV columns")
        return cols

    if feature_set not in sets:
        raise ValueError(
            "feature_set 必須是 all/baseline/bp/coh/asym/plv/iplv/"
            "baseline_asym/baseline_plv/baseline_asym_plv/bp_asym/"
            "bp_plv/coh_plv/asym_plv，或 plv_band_<band>"
        )
    cols = sets[feature_set]
    if not cols:
        raise ValueError(f"feature_set={feature_set} 找不到 feature columns")
    return cols


def _safe(text: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in str(text)).strip("_") or "subject"
