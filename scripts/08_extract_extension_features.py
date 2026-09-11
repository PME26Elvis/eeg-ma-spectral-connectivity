from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from eeg_ma.config import load_config
from eeg_ma.extensions import extract_all_extension_features


def main() -> int:
    p = argparse.ArgumentParser(
        description="Extract E1 hemispheric asymmetry and E2 PLV features without modifying baseline features"
    )
    p.add_argument("--raw-root", default="data/raw")
    p.add_argument("--baseline-features", default="data/features/features_all.csv")
    p.add_argument("--config", default="configs/lab_replication.json")
    p.add_argument("--extension-config", default="configs/extensions_e1_e2.json")
    p.add_argument("--out-dir", default="data/features/extensions")
    args = p.parse_args()

    cfg = load_config(args.config)
    extension_cfg = json.loads(Path(args.extension_config).read_text(encoding="utf-8"))
    out = extract_all_extension_features(
        args.raw_root,
        args.baseline_features,
        cfg,
        extension_cfg,
        args.out_dir,
    )

    asym = [c for c in out.columns if c.startswith("ASYM_")]
    plv = [c for c in out.columns if c.startswith("PLV__")]
    print("=== E1/E2 FEATURE EXTRACTION ===")
    print(f"subjects={out['subject_id'].nunique()}")
    print(f"epochs={len(out)}")
    print(f"asymmetry_features={len(asym)}")
    print(f"plv_features={len(plv)}")
    print(f"NaN={int(out[asym + plv].isna().sum().sum())}")
    if plv:
        print(f"PLV range=[{out[plv].min().min():.6f}, {out[plv].max().max():.6f}]")
    print(f"combined={Path(args.out_dir) / 'features_extensions.csv'}")
    print(f"manifest={Path(args.out_dir) / 'extension_manifest.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
