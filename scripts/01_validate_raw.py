from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

from eeg_ma.config import load_config
from eeg_ma.io import discover_session_dirs, load_session
from eeg_ma.qa import validate_protocol


def main() -> int:
    p = argparse.ArgumentParser(description="驗證 v0.15 HNC raw sessions 的完整性與 protocol counts")
    p.add_argument("--raw-root", default="data/raw")
    p.add_argument("--config", default="configs/lab_replication.json")
    p.add_argument("--out", default="results/raw_validation.csv")
    args = p.parse_args()

    cfg = load_config(args.config)
    rows = []
    errors = []
    for d in discover_session_dirs(args.raw_root):
        try:
            s = load_session(d, configured_fs=float(cfg["sampling_rate_hz"]))
            rows.append(validate_protocol(s, cfg))
        except Exception as e:
            errors.append((str(d), str(e)))

    if not rows and not errors:
        print(f"找不到 raw sessions: {args.raw_root}", file=sys.stderr)
        return 2

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False, encoding="utf-8-sig")
    print(f"已寫入 {out}")
    for r in rows:
        print(
            f"[{'PASS' if r['protocol_counts_pass'] and r['all_values_finite'] else 'CHECK'}] "
            f"{r['subject_id']}: samples={r['n_samples']}, Fs={r['inferred_sampling_rate_hz']:.3f}"
        )
    for path, err in errors:
        print(f"[FAIL] {path}: {err}", file=sys.stderr)
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
