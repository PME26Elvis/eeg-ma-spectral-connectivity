"""Inspect, freeze and audit prospective protocol evidence without scoring data."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json

from eeg_ma.cohort_policy import assess_enrollment
from eeg_ma.protocol import (
    ProtocolError, _read, amend_protocol, check_launch, lock_protocol,
    record_outcome_access, recover_event_append, validate_lock, validate_protocol,
)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    validate = commands.add_parser("validate", help="List pending protocol decisions; readiness alone is not a lock")
    validate.add_argument("--protocol", default="configs/confirmatory_protocol.json")
    lock = commands.add_parser("lock", help="Verify real evidence and atomically create a new immutable lock")
    lock.add_argument("--protocol", required=True)
    lock.add_argument("--out-dir", required=True)
    verify = commands.add_parser("verify", help="Verify frozen bytes, model lineage, precision evidence and event chain")
    verify.add_argument("--lock-dir", required=True)
    launch = commands.add_parser("launch", help="Check the complete cumulative, outcome-blind participant roster")
    launch.add_argument("--lock-dir", required=True)
    launch.add_argument("--cohort", required=True)
    enrollment = commands.add_parser("enrollment", help="Audit the cumulative roster and score-blind stop rule")
    enrollment.add_argument("--lock-dir", required=True)
    enrollment.add_argument("--enrollment", required=True)
    event = commands.add_parser("record-outcome-access", help="Record first outcome access after recruitment and all dispositions close")
    event.add_argument("--lock-dir", required=True)
    event.add_argument("--cohort", required=True)
    event.add_argument("--enrollment", required=True)
    recover = commands.add_parser("recover-event", help="Conservatively commit one intact interrupted outcome append")
    recover.add_argument("--lock-dir", required=True)
    amend = commands.add_parser("amend", help="Create a linked new lock; never overwrite an existing protocol")
    amend.add_argument("--lock-dir", required=True)
    amend.add_argument("--protocol", required=True)
    amend.add_argument("--out-dir", required=True)
    amend.add_argument("--reason", required=True)
    amend.add_argument("--holdout")
    args = parser.parse_args(argv)
    try:
        if args.command == "validate":
            report = validate_protocol(_read(args.protocol))
            result = report.as_dict()
            result["scope"] = "decision_readiness_only; lock command verifies all evidence bytes and clean source code"
            code = 0 if report.ready else 2
        elif args.command == "lock":
            result, code = lock_protocol(args.protocol, args.out_dir), 0
        elif args.command == "verify":
            locked = validate_lock(args.lock_dir)
            result = {"status": "valid", "lock_id": locked["manifest"]["lock_id"],
                      "protocol_id": locked["protocol"]["protocol_id"],
                      "outcome_access_events": len(locked["events"]),
                      "n_development": len(locked["development_subject_ids"]),
                      "n_exposed": len(locked["exposed_subject_ids"])}
            code = 0
        elif args.command == "launch":
            result, code = check_launch(args.lock_dir, args.cohort), 0
        elif args.command == "enrollment":
            locked = validate_lock(args.lock_dir)
            manifest = _read(args.enrollment)
            cohort = {"schema_version": "eeg-confirmatory-cohort-1", "cohort_id": manifest.get("cohort_id"),
                      "protocol_id": locked["protocol"]["protocols"]["A"]["protocol_id"],
                      "subject_ids": [person["subject_id"] for person in manifest.get("participants", [])],
                      "outcomes_accessed": manifest.get("outcomes_accessed")}
            check_launch(args.lock_dir, cohort)
            result = assess_enrollment(manifest, locked["protocol"],
                locked_at=locked["manifest"]["locked_at"], as_of=datetime.now(timezone.utc).isoformat())
            code = 0 if result["valid"] else 2
        elif args.command == "record-outcome-access":
            event = record_outcome_access(args.lock_dir, args.cohort, enrollment_manifest_path=args.enrollment)
            result, code = {key: event[key] for key in ("event_id", "lock_id", "sequence", "recorded_at", "n_recruited", "n_evaluable")}, 0
        elif args.command == "recover-event":
            event = recover_event_append(args.lock_dir)
            result, code = {"status": "recovered", "event_id": event["event_id"], "lock_id": event["lock_id"]}, 0
        else:
            result = amend_protocol(args.lock_dir, args.protocol, args.out_dir,
                                    reason=args.reason, holdout_manifest_path=args.holdout)
            code = 0
        print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
        return code
    except (ProtocolError, OSError, ValueError, KeyError, TypeError) as exc:
        print(json.dumps({"status": "blocked", "error": str(exc)}, ensure_ascii=False, indent=2))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
