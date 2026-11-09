from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys

from eeg_ma.artifacts import new_output_directory


def run(script: str, *args: str) -> None:
    cmd = [sys.executable, str(Path(__file__).parent / script), *args]
    print("\n$", " ".join(cmd))
    subprocess.run(cmd, check=True)


def main() -> int:
    p = argparse.ArgumentParser(description="依序執行 raw validation → features → behavioral → intra → inter")
    p.add_argument("--raw-root", default="data/raw")
    p.add_argument("--config", default="configs/lab_replication.json")
    p.add_argument("--out-root", default="outputs/pipeline", help="New destination; existing output is never replaced")
    args = p.parse_args()

    with new_output_directory(args.out_root) as stage:
        feature_dir = stage / "features" / "baseline"
        feature_csv = feature_dir / "features_all.csv"
        baseline = stage / "results" / "baseline"
        raw_args = ("--raw-root", args.raw_root, "--config", args.config)
        run("01_validate_raw.py", *raw_args, "--out", str(stage / "raw_validation.csv"))
        run("02_extract_features.py", *raw_args, "--out-dir", str(feature_dir))
        run("05_behavioral.py", *raw_args, "--out", str(baseline / "behavioral_summary.csv"))
        for scope, script in (("intra", "03_intra_subject.py"), ("inter", "04_inter_subject.py")):
            run(script, "--config", args.config, "--features", str(feature_csv),
                "--out-dir", str(baseline / scope))
    print(f"Completed pipeline: {Path(args.out_root).resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
