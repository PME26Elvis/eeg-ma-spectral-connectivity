"""Capture an observed environment or require an exact reference environment."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import tempfile

from eeg_ma.environment import EnvironmentError, capture_environment, validate_environment_evidence, verify_reference_environment
from eeg_ma.provenance import protect_output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["capture", "verify"])
    parser.add_argument("--reference", help="Reference definition JSON; defaults to EEG_MA_REFERENCE_SPEC or environment/reference.json")
    parser.add_argument("--install-report", help="pip --report JSON; defaults to EEG_MA_INSTALL_REPORT")
    parser.add_argument("--project-root", default=None)
    parser.add_argument("--out", help="New environment evidence JSON; existing files are never overwritten")
    args = parser.parse_args()
    try:
        function = verify_reference_environment if args.command == "verify" else capture_environment
        identity = function(project_root=args.project_root, reference_path=args.reference, install_report=args.install_report)
        summary = validate_environment_evidence(identity, require_reference=args.command == "verify")
        if args.out:
            target = protect_output(args.out)
            if target.exists():
                raise EnvironmentError("Environment output already exists; choose a new path")
            target.parent.mkdir(parents=True, exist_ok=True)
            fd, temporary = tempfile.mkstemp(prefix=".environment-", dir=target.parent)
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as stream:
                    stream.write(json.dumps(identity, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n")
                    stream.flush()
                    os.fsync(stream.fileno())
                protect_output(target)
                os.link(temporary, target)
            finally:
                Path(temporary).unlink(missing_ok=True)
        print(json.dumps(summary, indent=2))
        return 0
    except (EnvironmentError, OSError, ValueError) as exc:
        print(json.dumps({"status": "blocked", "error": str(exc)}, ensure_ascii=False))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
