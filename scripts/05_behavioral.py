from __future__ import annotations

import argparse
from pathlib import Path

from eeg_ma.behavioral import summarize_behavior
from eeg_ma.config import load_config


def main() -> int:
    p = argparse.ArgumentParser(description="Type1 / Type2 behavioral summary")
    p.add_argument("--raw-root", default="data/raw")
    p.add_argument("--config", default="configs/lab_replication.json")
    p.add_argument("--out", default="results/behavioral_summary.csv")
    args = p.parse_args()

    cfg = load_config(args.config)
    df = summarize_behavior(args.raw_root, cfg)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False, encoding="utf-8-sig")
    print(df.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
