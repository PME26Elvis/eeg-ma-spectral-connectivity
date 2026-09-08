from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys


def run(script: str, *args: str) -> None:
    cmd = [sys.executable, str(Path(__file__).parent / script), *args]
    print("\n$", " ".join(cmd))
    subprocess.run(cmd, check=True)


def main() -> int:
    p = argparse.ArgumentParser(description="依序執行 raw validation → features → behavioral → intra → inter")
    p.add_argument("--raw-root", default="data/raw")
    p.add_argument("--config", default="configs/lab_replication.json")
    args = p.parse_args()

    run("01_validate_raw.py", "--raw-root", args.raw_root, "--config", args.config)
    run("02_extract_features.py", "--raw-root", args.raw_root, "--config", args.config)
    run("05_behavioral.py", "--raw-root", args.raw_root, "--config", args.config)
    run("03_intra_subject.py", "--config", args.config)
    run("04_inter_subject.py", "--config", args.config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
