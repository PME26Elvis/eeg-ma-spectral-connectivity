"""Run the real synthetic five-model S/F CLI; no human EEG conclusions."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

from exploratory_fixture import create_exploratory_fixture
from eeg_ma.exploratory import TRACKS, validate_exploratory_report
from eeg_ma.provenance import protect_output, read_csv_literal, sha256_file, validate_complete_run


ROOT = Path(__file__).resolve().parents[1]


def run(out_dir):
    target = protect_output(out_dir)
    target.mkdir(parents=True, exist_ok=False)
    source, config, plan, eligibility, frame = create_exploratory_fixture(target / "synthetic")
    output = target / "comparison"
    commands = []

    def execute(name, args, expected=0):
        cmd = [sys.executable, str(ROOT / "scripts/25_compare_classifiers.py"), *map(str, args)]
        proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, timeout=240)
        (target / f"{name}.stdout.txt").write_text(proc.stdout, encoding="utf-8")
        (target / f"{name}.stderr.txt").write_text(proc.stderr, encoding="utf-8")
        commands.append({"name": name, "returncode": proc.returncode, "expected": expected,
                         "command": cmd})
        if proc.returncode != expected:
            raise RuntimeError(f"{name}: expected {expected}, got {proc.returncode}: {proc.stderr}")
        return proc

    execute("draft-refusal", ["check", "--features", source,
             "--config", config, "--plan", plan, "--eligibility", eligibility], 2)
    execute("preflight", ["check", "--features", source, "--config", config,
                           "--plan", plan, "--eligibility", eligibility, "--allow-synthetic"])
    execute("run", ["run", "--features", source, "--config", config,
                    "--plan", plan, "--eligibility", eligibility, "--out-dir", output, "--allow-synthetic"])
    execute("verify", ["verify", "--report", output])
    earlier = {p.relative_to(output).as_posix(): sha256_file(p)
               for p in output.rglob("*") if p.is_file()}
    execute("no-overwrite", ["run", "--features", source, "--config", config,
                             "--plan", plan, "--eligibility", eligibility, "--out-dir", output, "--allow-synthetic"], 2)
    if earlier != {p.relative_to(output).as_posix(): sha256_file(p)
                   for p in output.rglob("*") if p.is_file()}:
        raise RuntimeError("Refusing overwrite changed the published report")
    verified = validate_exploratory_report(output)
    if verified["n_participants"] != 3 or verified["n_groups"] != 7 or verified["n_predictions"] != 336:
        raise RuntimeError(f"Wrong complete cohort/classifier matrix: {verified}")
    for track, count in (("S", 5), ("F", 2)):
        manifest = validate_complete_run(output / track)
        folds = read_csv_literal(output / track / "inter_folds.csv")
        paths = read_csv_literal(output / track / "inter_sfs_path.csv")
        if len(folds) != 3 * count:
            raise RuntimeError(f"{track} fold matrix incomplete")
        if track == "F":
            if not paths.cv_accuracy.isna().all() or set(paths.selection_method) != {"all_declared_features"}:
                raise RuntimeError("Full-feature track misrepresents SFS diagnostics")
            if set(folds.n_selected_features) != {3}:
                raise RuntimeError("Full-feature track did not include all features")
    report = {
        **verified, "schema_version": "eeg-exploratory-smoke-1",
        "n_human_participants": 0, "n_synthetic_sessions": 6, "n_epochs": len(frame),
        "classifiers_by_track": TRACKS,
        "frozen_eligibility_sha256": sha256_file(eligibility),
        "scope": "SYNTHETIC engineering evidence, NOT human EEG research",
        "commands": commands, "output_bytes_sha256": {
            p.relative_to(output).as_posix(): sha256_file(p)
            for p in output.rglob("*") if p.is_file()
        },
    }
    (target / "smoke_report.json").write_text(
        json.dumps(report, sort_keys=True, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.out_dir), ensure_ascii=False, sort_keys=True, indent=2))
