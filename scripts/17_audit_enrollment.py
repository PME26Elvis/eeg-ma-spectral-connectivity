"""Audit the complete score-blind recruitment ledger against a real protocol lock."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

from eeg_ma.artifacts import new_output_directory
from eeg_ma.cohort_policy import assess_enrollment
from eeg_ma.protocol import ProtocolError, _read, check_launch, validate_lock
from eeg_ma.provenance import sha256_file


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lock", required=True, help="A validated immutable lock directory")
    parser.add_argument("--enrollment", required=True, help="Complete cumulative metadata-only enrollment JSON")
    parser.add_argument("--out-dir", required=True, help="New audit output directory; existing outputs are never overwritten")
    args = parser.parse_args()
    try:
        locked = validate_lock(args.lock)
        source = Path(args.enrollment)
        digest = sha256_file(source)
        enrollment = _read(source)
        report = assess_enrollment(enrollment, locked["protocol"],
            locked_at=locked["manifest"]["locked_at"], as_of=datetime.now(timezone.utc).isoformat())
        if report["valid"]:
            launch = check_launch(args.lock, {
                "schema_version": "eeg-confirmatory-cohort-1", "cohort_id": enrollment["cohort_id"],
                "protocol_id": locked["protocol"]["protocols"]["A"]["protocol_id"],
                "subject_ids": [p["subject_id"] for p in enrollment["participants"]],
                "outcomes_accessed": False,
            })
            report["launch"] = launch
        report.update(lock_id=locked["manifest"]["lock_id"], enrollment_sha256=digest)
        if sha256_file(source) != digest:
            raise ProtocolError("Enrollment changed during audit")
        with new_output_directory(args.out_dir) as stage:
            (stage / "enrollment_audit.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        print(json.dumps({key: report[key] for key in ("valid", "n_recruited", "n_evaluable", "stop_recruitment", "outcome_ready", "errors")}, indent=2))
        return 0 if report["valid"] else 2
    except (ProtocolError, OSError) as exc:
        parser.exit(2, f"Enrollment audit blocked: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
