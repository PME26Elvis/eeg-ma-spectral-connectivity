from __future__ import annotations

import argparse

from eeg_ma.config import load_config
from eeg_ma.extract import extract_all, feature_columns


def main() -> int:
    p = argparse.ArgumentParser(description="2–50 Hz preprocessing + 42 BP + 126 COH feature extraction")
    p.add_argument("--raw-root", default="data/raw")
    p.add_argument("--config", default="configs/lab_replication.json")
    p.add_argument("--out-dir", default="data/features")
    args = p.parse_args()

    cfg = load_config(args.config)
    features, validation = extract_all(args.raw_root, cfg, args.out_dir)
    fcols = feature_columns(features, "all")
    print(f"subjects={features['subject_id'].nunique()}")
    print(f"epochs={len(features)}")
    print(f"features={len(fcols)} (BP={sum(c.startswith('BP__') for c in fcols)}, COH={sum(c.startswith('COH__') for c in fcols)})")
    print(f"combined={args.out_dir}/features_all.csv")
    print(f"validation={args.out_dir}/raw_validation.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
