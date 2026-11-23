"""Full reference strict replay smoke on already-created S/F synthetic cohort.

This script intentionally does not mock code/environment/learner/clock. It uses
the exact immutable run directories created by the previous reference CI step.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys

from eeg_ma.provenance import protect_output, sha256_file, validate_complete_run


ROOT = Path(__file__).resolve().parents[1]


def inventory(folder):
    return {p.relative_to(folder).as_posix(): sha256_file(p)
            for p in sorted(folder.rglob("*")) if p.is_file()}


def run(source_root, target):
    source_root, target = Path(source_root), protect_output(target)
    target.mkdir(parents=True, exist_ok=False)
    report = {
        "schema_version": "eeg-strict-reference-smoke-1",
        "status": "running",
        "scope": "synthetic training replay only; no human EEG inference",
        "runs": {}, "commands": [],
    }

    def execute(name, args, code=0):
        command = [sys.executable, str(ROOT / "scripts/26_strict_verify.py"), *map(str, args)]
        process = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, timeout=600)
        (target / f"{name}.stdout.txt").write_text(process.stdout, encoding="utf-8")
        (target / f"{name}.stderr.txt").write_text(process.stderr, encoding="utf-8")
        if process.returncode != code:
            raise RuntimeError(f"{name}: unexpected exit {process.returncode} (wanted {code}): "
                               f"{process.stdout[-2500:]} {process.stderr[-2500:]}")
        try:
            record = json.loads(process.stdout)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"{name}: invalid JSON status") from exc
        report["commands"].append({"name": name, "exit": code, "status": record["status"]})
        return record

    original_hashes = {track: inventory(source_root / track) for track in ("S", "F")}
    execute("oracle_canary", ["oracles", "--out-dir", target / "tiny_oracles"])
    for track in ("S", "F"):
        destination = target / f"{track}_strict"
        result = execute(f"{track}_strict", [
            "verify", "--run", source_root / track, "--out-dir", destination,
            "--max-model-fits", "64", "--max-outer-folds", "6",
            "--max-candidates", "4", "--max-projected-nodes", "100000",
            "--max-seconds", "360",
        ])
        if result["status"] != "verified_strict_train_replay":
            raise RuntimeError("Strict mode did not actually fit all models")
        assert result["n_complete_refits"] == 30  # 3 outer × 2 learners × (2 candidates × 2 inner + final)
        assert result["preflight"]["models"] == ["RandomForest", "LightGBM"]
        assert len(result["models"]) == 6
        report["runs"][track] = {
            "run_id": result["run_id"],
            "fit_replays": result["n_complete_refits"],
            "outer_model_entries": len(result["models"]),
            "report_sha256": sha256_file(destination / "strict_report.json"),
        }
        preflight = execute(f"{track}_preflight", ["verify", "--run", source_root / track,
                                                  "--preflight-only"])
        if preflight["status"] != "not_run_preflight_only":
            raise RuntimeError("Preflight was incorrectly claimed as completed verification")
        limit = execute(f"{track}_bounded_refusal", [
            "verify", "--run", source_root / track,
            "--max-model-fits", "1", "--out-dir", target / f"{track}_refused",
        ], code=3)
        if limit["status"] != "not_run_resource_limit" or (target / f"{track}_refused").exists():
            raise RuntimeError("Resource refusal published an apparently successful report")

    # Concrete model-selection evidence can be self-consistent without being
    # produced by the declared training procedure. Modify just one inner RF
    # probability vector so the fast audit's label, accuracy and sum invariants
    # still pass. Rehash the altered artifact. The full refit must reject it.
    attacked = target / "modified_fast_valid_strict_invalid"
    shutil.copytree(source_root / "F", attacked)
    path = attacked / "random_forest_audit.json"
    audit = json.loads(path.read_text(encoding="utf-8"))
    fold = audit["folds"][0]["candidates"][0]["folds"][0]
    first_prediction = int(fold["predictions"][0])
    old_probabilities = fold["probabilities"][0]
    new_probability = .937 if abs(old_probabilities[first_prediction] - .937) > .001 else .811
    fold["probabilities"][0] = ([new_probability, 1 - new_probability]
                                if first_prediction == 0
                                else [1 - new_probability, new_probability])
    path.write_text(json.dumps(audit, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
                    encoding="utf-8")
    manifest_path = attacked / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["artifacts"]["random_forest_audit.json"]["sha256"] = sha256_file(path)
    manifest["artifacts"]["random_forest_audit.json"]["bytes"] = path.stat().st_size
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2)
                             + "\n", encoding="utf-8")
    validate_complete_run(attacked)
    failure = execute("tampered_inner_probability", [
        "verify", "--run", attacked, "--out-dir", target / "no_false_positive",
        "--max-model-fits", "64",
    ], code=2)
    if failure["status"] != "failed_integrity_or_environment" or (target / "no_false_positive").exists():
        raise RuntimeError("Strict verification accepted a changed inner model probability")
    # A second fast-valid attack changes a training-only SFS CV score while
    # preserving selected features, path structure, and all outer predictions.
    attacked_sfs = target / "modified_sfs_fast_valid_strict_invalid"
    shutil.copytree(source_root / "S", attacked_sfs)
    trace = attacked_sfs / "inter_sfs_path.csv"
    from eeg_ma.provenance import read_csv_literal
    trace_table = read_csv_literal(trace)
    original = float(trace_table.loc[0, "cv_accuracy"])
    trace_table.loc[0, "cv_accuracy"] = .222 if abs(original - .222) > .001 else .888
    trace_table.to_csv(trace, index=False)
    manifest_file = attacked_sfs / "run_manifest.json"
    rehashed = json.loads(manifest_file.read_text(encoding="utf-8"))
    rehashed["artifacts"]["inter_sfs_path.csv"]["sha256"] = sha256_file(trace)
    rehashed["artifacts"]["inter_sfs_path.csv"]["bytes"] = trace.stat().st_size
    manifest_file.write_text(json.dumps(rehashed, ensure_ascii=False, sort_keys=True, indent=2)
                             + "\n", encoding="utf-8")
    validate_complete_run(attacked_sfs)
    wrong_sfs = execute("tampered_sfs_cv_accuracy", [
        "verify", "--run", attacked_sfs, "--out-dir", target / "no_false_sfs",
        "--max-model-fits", "64",
    ], code=2)
    if wrong_sfs["status"] != "failed_integrity_or_environment" or (target / "no_false_sfs").exists():
        raise RuntimeError("Strict verification accepted a fabricated LDA-SFS CV score")
    # Reject an existing verification output before repeating expensive fits.
    existing_hashes = inventory(target / "S_strict")
    existing = execute("existing_strict_report_refusal", [
        "verify", "--run", source_root / "S",
        "--out-dir", target / "S_strict", "--max-model-fits", "64"
    ], code=2)
    if existing["status"] != "failed_integrity_or_environment" or inventory(target / "S_strict") != existing_hashes:
        raise RuntimeError("Strict verification overwrote an existing complete report")
    for track in ("S", "F"):
        if inventory(source_root / track) != original_hashes[track]:
            raise RuntimeError("Strict replay mutated a completed source run")
    report["status"] = "passed"
    report["tampered_inner_probabilities_rehashed"] = "fast_passed_strict_rejected"
    report["tampered_sfs_cv_scores_rehashed"] = "fast_passed_strict_rejected"
    report["existing_strict_report_unchanged"] = True
    (target / "smoke_report.json").write_text(
        json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.runs_root, args.out_dir), ensure_ascii=False, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
