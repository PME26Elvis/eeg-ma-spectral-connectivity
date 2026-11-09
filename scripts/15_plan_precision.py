"""Run prospective sample-size simulation; never reads research features/results."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from eeg_ma.artifacts import new_output_directory
from eeg_ma.precision import run_precision_plan, validate_precision_report


def main() -> int:
    parser = argparse.ArgumentParser(description="Participant-level precision, coverage and recruitment assurance planning")
    parser.add_argument("--config", default="configs/precision_plan.json")
    parser.add_argument("--out-dir", default="outputs/precision-plan")
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    report = run_precision_plan(config, progress=lambda message: print(message, flush=True))
    if report["validation_pass"]:
        validate_precision_report(report, config)
    with new_output_directory(args.out_dir) as stage:
        (stage / "precision_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        rows = report["screening"] + report["validation"]
        with (stage / "scenarios.csv").open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
    print(json.dumps({key: report[key] for key in ("status", "selected_target_n", "recruitment_cap", "report_sha256")}, indent=2))
    return 0 if report["validation_pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
