from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

from eeg_ma.config import load_config
from eeg_ma.io import discover_session_dirs, load_session
from eeg_ma.qa import validate_session
from eeg_ma.provenance import protect_output
from eeg_ma.artifacts import write_new_csv


def main() -> int:
    p = argparse.ArgumentParser(description="驗證 v0.15 HNC raw sessions 的完整性與 protocol counts")
    p.add_argument("--raw-root", default="data/raw")
    p.add_argument("--config", default="configs/lab_replication.json")
    p.add_argument("--out", default="outputs/qa/raw_validation.csv")
    args = p.parse_args()

    cfg = load_config(args.config)
    out = protect_output(args.out)
    if out.exists():
        raise FileExistsError(f"Refusing to overwrite existing QA report: {out}")
    rows = []
    errors = []
    identities = set()
    for d in discover_session_dirs(args.raw_root):
        try:
            s = load_session(d, configured_fs=float(cfg["sampling_rate_hz"]),
                             metadata=cfg.get("ingestion", {}).get("metadata"))
            report = validate_session(s, cfg)
            identity = tuple(s.metadata.get(key) for key in ("cohort_id", "subject_id", "session_id"))
            if identity in identities:
                from eeg_ma.qa import QAFinding
                report.findings.append(QAFinding("DUPLICATE_SESSION", "error", "Multiple raw directories have the same cohort/subject/session identity"))
            identities.add(identity)
            rows.append(report.as_dict())
            if not report.passed:
                errors.append((str(d), "; ".join(f.code + ": " + f.message for f in report.findings if f.severity == "error")))
        except Exception as e:
            errors.append((str(d), str(e)))
            rows.append({"subject_id": "unknown", "session_id": "unknown", "qa_pass": False,
                         "qc_status": "failed", "error_codes": "LOAD_ERROR", "error_message": str(e),
                         "n_samples": None, "inferred_sampling_rate_hz": None})

    if not rows and not errors:
        print(f"找不到 raw sessions: {args.raw_root}", file=sys.stderr)
        return 2

    write_new_csv(pd.DataFrame(rows), out)
    print(f"已寫入 {out}")
    for r in rows:
        print(
            f"[{'PASS' if r['qa_pass'] else 'FAIL'}] "
            f"{r['subject_id']}: samples={r['n_samples']}, Fs={r.get('inferred_sampling_rate_hz')}"
        )
    for path, err in errors:
        print(f"[FAIL] {path}: {err}", file=sys.stderr)
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
