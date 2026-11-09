"""Freeze an explicit, bounded dependency-comparison request before installation.

Only environment definitions are read from the candidate checkout. Both numerical
runs execute the current analysis revision and its synthetic fixture. This
stdlib-only preflight does not run candidate Python source or a formal study.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from eeg_ma.environment import load_reference  # stdlib-only before installing wheels


class ComparisonError(ValueError):
    """The requested environment comparison lacks a concrete, intact identity."""


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _git(root: Path, *args: str) -> str:
    result = subprocess.run(["git", "-C", str(root), *args], text=True, capture_output=True)
    if result.returncode:
        raise ComparisonError("A requested checkout or committed evidence file is unavailable")
    return result.stdout.strip()


def _profile(root: Path) -> dict:
    path = root / "environment/reference.json"
    profile = load_reference(path)
    spec = profile["definition"]
    if (spec.get("implementation") != "CPython" or spec.get("system") != "Linux"
            or spec.get("machine") != "x86_64"):
        raise ComparisonError("This comparison runner supports explicitly locked CPython/Linux/x86_64 profiles")
    python = spec.get("python")
    if not isinstance(python, str) or not re.fullmatch(r"3\.[0-9]+\.[0-9]+", python):
        raise ComparisonError("Every interpreter must be an exact stable Python patch version")
    lock_name = spec["requirements_lock"]["file"]
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", lock_name):
        raise ComparisonError("The requirements lock needs a simple safe filename")
    for relative in ("environment/reference.json", "environment/" + lock_name):
        committed = subprocess.run(["git", "-C", str(root), "show", "HEAD:" + relative], capture_output=True)
        if committed.returncode or committed.stdout != (root / relative).read_bytes():
            raise ComparisonError("Environment evidence differs from its selected commit")
    return {**profile, "python": python, "lock_name": lock_name}


def validate_request_inputs(candidate_commit: str, run_id: str, reason: str) -> dict:
    """Validate user text before it is used to select a candidate Git revision."""
    if not isinstance(candidate_commit, str) or not re.fullmatch(r"[0-9a-f]{40}", candidate_commit):
        raise ComparisonError("candidate_commit must be an exact 40-character lowercase commit SHA")
    if not isinstance(run_id, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}", run_id):
        raise ComparisonError("run_id must contain 1-64 safe identifier characters")
    if not isinstance(reason, str) or not reason.strip() or len(reason) > 2000:
        raise ComparisonError("A concrete prospective reason of at most 2000 characters is required")
    return {"candidate_commit": candidate_commit, "run_id": run_id, "reason": reason}


def prepare_comparison(project_root: Path, candidate_checkout: Path, *, candidate_commit: str,
                       run_id: str, reason: str, out_dir: Path) -> dict:
    """Validate both identities, then publish copied input evidence atomically."""
    validate_request_inputs(candidate_commit, run_id, reason)
    project_root, candidate_checkout = project_root.resolve(), candidate_checkout.resolve()
    analysis_commit = _git(project_root, "rev-parse", "HEAD")
    if not re.fullmatch(r"[0-9a-f]{40}", analysis_commit):
        raise ComparisonError("Analysis code must have an exact commit identity")
    if _git(candidate_checkout, "rev-parse", "HEAD") != candidate_commit:
        raise ComparisonError("Candidate checkout does not match the requested commit")
    tracked_scope = ("src", "scripts", "configs", "environment", "tests/fixtures", "pyproject.toml")
    if _git(project_root, "status", "--porcelain=v1", "--untracked-files=all", "--", *tracked_scope):
        raise ComparisonError("Analysis code, configuration and fixture must be clean at their chosen commit")
    profiles = {"baseline": _profile(project_root), "candidate": _profile(candidate_checkout)}
    if profiles["baseline"]["definition"].get("project") != profiles["candidate"]["definition"].get("project"):
        raise ComparisonError("Both environments must target the same project version and analysis code")
    fixture_path = project_root / "tests/fixtures/numerical/reference.json"
    if fixture_path.is_symlink() or not fixture_path.is_file():
        raise ComparisonError("The analysis revision must contain its reviewed numerical fixture")
    target = out_dir.absolute()
    if not target.resolve().is_relative_to(project_root / "outputs"):
        raise ComparisonError("Comparison evidence belongs in a new project outputs/ directory")
    if any(path.is_symlink() for path in (target, *target.parents)) or target.exists():
        raise ComparisonError("Comparison output must be new and cannot use symlinks")
    details = {}
    for role, profile in profiles.items():
        details[role] = {
            "source_commit": analysis_commit if role == "baseline" else candidate_commit,
            "reference_id": profile["definition"]["reference_id"], "python": profile["python"],
            "definition_sha256": profile["definition_sha256"],
            "requirements_sha256": profile["requirements_sha256"], "lock_id": profile["lock_id"],
            "frozen_definition": f"{role}/reference.json",
            "frozen_requirements": f"{role}/{profile['lock_name']}",
        }
    record = {"schema_version": "eeg-dependency-comparison-1", "scope": "bounded_synthetic_grouped_cv_only",
        "run_id": run_id, "reason": reason, "recorded_at": datetime.now(timezone.utc).isoformat(),
        "analysis_code_commit": analysis_commit, "candidate_environment_commit": candidate_commit,
        "fixture_reference_sha256": _sha(fixture_path.read_bytes()), "environments": details,
        "comparison_policy": "same_analysis_source_and_synthetic_inputs; compare_prespecified_numerical_tolerances"}
    record["request_id"] = _sha(json.dumps(record, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode())
    target.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=".dependency-request-", dir=target.parent))
    try:
        for role, profile in profiles.items():
            (stage / role).mkdir()
            (stage / role / "reference.json").write_text(profile["definition_text"], encoding="utf-8")
            (stage / role / profile["lock_name"]).write_text(profile["requirements_text"], encoding="utf-8")
            if (_sha((stage / role / "reference.json").read_bytes()) != profile["definition_sha256"]
                    or _sha((stage / role / profile["lock_name"]).read_bytes()) != profile["requirements_sha256"]):
                raise ComparisonError("Frozen environment bytes changed during request publication")
        (stage / "comparison_request.json").write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        if target.exists():
            raise ComparisonError("Comparison output appeared during publication")
        os.rename(stage, target)
    finally:
        if stage.exists():
            shutil.rmtree(stage)
    return record


def github_outputs(record: dict, out_dir: Path, project_root: Path) -> str:
    """Only validated versions and safe relative paths enter GITHUB_OUTPUT."""
    values = {}
    for role, entry in record["environments"].items():
        values[role + "_python"] = entry["python"]
        values[role + "_lock"] = str((out_dir / entry["frozen_requirements"]).absolute().relative_to(project_root.resolve()))
    if any("\n" in value or "\r" in value for value in values.values()):
        raise ComparisonError("Multiline action outputs are forbidden")
    return "".join(f"{key}={value}\n" for key, value in values.items())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-checkout", required=True, type=Path)
    parser.add_argument("--candidate-commit", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--reason", required=True)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--github-output", type=Path)
    parser.add_argument("--validate-inputs-only", action="store_true")
    args = parser.parse_args()
    try:
        if args.validate_inputs_only:
            print(json.dumps(validate_request_inputs(args.candidate_commit, args.run_id, args.reason), ensure_ascii=False, indent=2))
            return 0
        record = prepare_comparison(ROOT, args.candidate_checkout, candidate_commit=args.candidate_commit,
            run_id=args.run_id, reason=args.reason, out_dir=args.out_dir)
        if args.github_output is not None:
            with args.github_output.open("a", encoding="utf-8") as stream:
                stream.write(github_outputs(record, args.out_dir, ROOT))
        print(json.dumps(record, ensure_ascii=False, indent=2))
        return 0
    except (ComparisonError, OSError, ValueError, KeyError) as exc:
        print(json.dumps({"status": "blocked", "error": str(exc)}, ensure_ascii=False))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
