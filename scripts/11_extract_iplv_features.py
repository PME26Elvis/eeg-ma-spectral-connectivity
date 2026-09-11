from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from eeg_ma.config import load_config
from eeg_ma.final_validation import extract_all_iplv_features


def main() -> int:
    p = argparse.ArgumentParser(
        description="Extract iPLV features for the final zero-lag robustness check"
    )
    p.add_argument("--raw-root", default="data/raw")
    p.add_argument("--baseline-features", default="data/features/features_all.csv")
    p.add_argument("--config", default="configs/lab_replication.json")
    p.add_argument("--validation-config", default="configs/final_validation.json")
    p.add_argument("--out-dir", default="data/features/final_validation")
    args = p.parse_args()

    cfg = load_config(args.config)
    vcfg = json.loads(Path(args.validation_config).read_text(encoding="utf-8"))
    filter_order = int(vcfg.get("phase_robustness", {}).get("filter_order", 4))

    frame = extract_all_iplv_features(
        raw_root=args.raw_root,
        baseline_feature_path=args.baseline_features,
        cfg=cfg,
        out_dir=args.out_dir,
        filter_order=filter_order,
    )

    cols = [c for c in frame.columns if c.startswith("IPLV__")]
    values = frame[cols].to_numpy(dtype=float)
    print("=== FINAL iPLV FEATURE EXTRACTION ===")
    print(f"subjects={frame['subject_id'].nunique()}")
    print(f"epochs={len(frame)}")
    print(f"iplv_features={len(cols)}")
    print(f"NaN={int(np.isnan(values).sum())}")
    print(f"Inf={int(np.isinf(values).sum())}")
    print(f"iPLV range=[{float(np.min(values)):.6f}, {float(np.max(values)):.6f}]")
    print(f"combined={Path(args.out_dir) / 'features_iplv.csv'}")
    print(f"manifest={Path(args.out_dir) / 'iplv_manifest.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
