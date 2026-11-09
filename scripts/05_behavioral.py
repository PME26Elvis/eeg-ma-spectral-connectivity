from __future__ import annotations

import argparse
from eeg_ma.artifacts import write_new_csv
from eeg_ma.behavioral import summarize_behavior
from eeg_ma.config import load_config


def main() -> int:
    p = argparse.ArgumentParser(description="Type1 / Type2 behavioral summary")
    p.add_argument("--raw-root", default="data/raw")
    p.add_argument("--config", default="configs/lab_replication.json")
    p.add_argument("--out", default="outputs/results/baseline/behavioral_summary.csv")
    args = p.parse_args()

    cfg = load_config(args.config)
    df = summarize_behavior(args.raw_root, cfg)
    write_new_csv(df, args.out)
    print(df.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
