"""Exercise CLI 23 with synthetic phase inputs in the actual reference runtime.

The fixture has 12 epochs, 3 synthetic participant IDs, and no human recordings.
No code identity, environment verifier, clock, approval, or subprocess runtime
is replaced.  A dirty checkout is truthfully recorded and is permitted only
because this script exercises numerical synthetic mode, not prospective locks.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

import eeg_ma
from eeg_ma.environment import verify_reference_environment
from eeg_ma.phase_controls import PAIR_ORDER, canonical_sha256, verify_matched_phase_bundle
from eeg_ma.provenance import code_identity, protect_output, sha256_file

from test_phase_controls import phase_fixture


ROOT = Path(__file__).resolve().parents[1]
CLI = ROOT / "scripts/23_phase_controls.py"
SCOPE = ("SYNTHETIC numerical engineering evidence only; no human study, "
         "acquisition, approval, model fitting, or performance results")


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                    indent=2, allow_nan=False) + "\n", encoding="utf-8")


def run(out_dir):
    require(Path(eeg_ma.__file__).resolve().is_relative_to(ROOT / "src/eeg_ma"),
            "Install this checkout in the reference interpreter before running its CLI smoke")
    reference = verify_reference_environment()
    before_code = code_identity()
    primary_protocol_sha = sha256_file(ROOT / "configs/confirmatory_protocol.json")
    target = protect_output(out_dir)
    target.mkdir(parents=True, exist_ok=False)
    (target / "inputs").mkdir()
    (target / "logs").mkdir()
    started = time.monotonic()
    report = {
        "schema_version": "eeg-phase-controls-smoke-1", "status": "running", "scope": SCOPE,
        "started_at": datetime.now(timezone.utc).isoformat(), "commands": [],
        "code": {key: before_code[key] for key in ("commit", "dirty", "working_source_sha256")},
        "validation_script_sha256": sha256_file(__file__),
        "reference_environment_id": reference["environment_id"], "reference_lock_id": reference["lock_id"],
        "runtime_or_clock_overrides": False, "n_human_participants": 0,
        "fixture_origin": "tests/test_phase_controls.py::phase_fixture; seeded phase angles, not physiological EEG",
    }
    write_json(target / "reference_environment.json", reference)

    def cli(name, *arguments, expected_returncode=0, json_output=True):
        command = [sys.executable, str(CLI), *map(str, arguments)]
        result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, timeout=60)
        stdout = target / "logs" / f"{name}.stdout.txt"
        stderr = target / "logs" / f"{name}.stderr.txt"
        stdout.write_text(result.stdout, encoding="utf-8")
        stderr.write_text(result.stderr, encoding="utf-8")
        report["commands"].append({
            "name": name, "argv": command, "returncode": result.returncode,
            "expected_returncode": expected_returncode, "stdout": str(stdout.relative_to(target)),
            "stderr": str(stderr.relative_to(target)),
        })
        require(result.returncode == expected_returncode,
                f"{name} returned {result.returncode}; inspect {stdout} and {stderr}")
        return json.loads(result.stdout) if json_output else result

    try:
        phases, epochs, splits, _ = phase_fixture()
        # Give the numerical fixture the planned B contrast name.  This is a
        # synthetic identity declaration, never a claim that B was acquired.
        for row in epochs["epochs"]:
            row["comparison"] = "copy_control_vs_type1"
        np.save(target / "inputs/phases.npy", phases, allow_pickle=False)
        write_json(target / "inputs/epochs.json", epochs)
        write_json(target / "inputs/splits.json", splits)
        report["inputs"] = {name: {"sha256": sha256_file(target / "inputs" / name)}
                            for name in ("phases.npy", "epochs.json", "splits.json")}

        draft = cli("draft-check", "check", "--plan", ROOT / "configs/phase_control_plan.json",
                    "--out-dir", target / "draft-readiness", expected_returncode=2)
        require(draft["ready_to_lock"] is False and draft["status"] == "blocked" and draft["findings"],
                "The real repository draft must remain blocked")
        require(not (target / "draft-readiness/control_plan_lock.json").exists(),
                "Readiness inspection must not create a prospective lock")

        extract_args = ("extract", "--mode", "synthetic-fixture", "--phases", target / "inputs/phases.npy",
                        "--epochs", target / "inputs/epochs.json", "--splits", target / "inputs/splits.json",
                        "--out-dir", target / "features")
        extracted = cli("extract-synthetic", *extract_args)
        verified = cli("verify-features", "verify-features", "--bundle", target / "features")
        bundle = verify_matched_phase_bundle(target / "features")
        manifest = bundle.manifest
        require(extracted["bundle_sha256"] == verified["bundle_sha256"] == manifest["bundle_sha256"],
                "Extraction, CLI verification, and numerical artifact identities differ")
        require(verified["valid"] is True and bundle.evaluable,
                "The bounded numerical fixture must be internally valid and evaluable")
        require(manifest["input_policy"]["evidence_kind"] == "synthetic_fixture"
                and manifest["input_policy"]["protocol_lock_id"] is None
                and manifest["contains_model_performance_results"] is False,
                "Synthetic extraction must not acquire prospective or model-performance status")
        require(bundle.epoch_ids == tuple(row["epoch_id"] for row in epochs["epochs"]),
                "Synthetic input row identity changed")
        require(len(bundle.epoch_ids) == 12 and len(manifest["accepted_subject_ids"]) == 3,
                "Expected 12 epochs and 3 synthetic participants")
        require(manifest["input_policy"]["epoch_manifest_sha256"] == canonical_sha256(epochs)
                and manifest["input_policy"]["split_manifest_sha256"] == canonical_sha256(splits),
                "Feature provenance does not bind the supplied manifests")
        require(manifest["split_manifest"] == splits and manifest["exclusions"] == [],
                "Numerical fixture unexpectedly altered the split plan or excluded epochs")

        # Independent direct complex-exponential oracle: this deliberately does
        # not call phase_metrics or use its cancellation-resistant denominator.
        channels = manifest["input_policy"]["channels"]
        means = np.stack([
            np.mean(np.exp(1j * (phases[:, :, channels.index(a)]
                                - phases[:, :, channels.index(b)])), axis=1)
            for a, b in PAIR_ORDER
        ], axis=1)
        direct_denominator = np.sqrt(1.0 - means.real**2)
        require(np.all(direct_denominator > 1e-6),
                "This ordinary-value oracle fixture must stay away from ciPLV degeneracy")
        expected = {"plv": np.abs(means), "iplv": np.abs(means.imag),
                    "ciplv": np.abs(means.imag) / direct_denominator}
        max_errors = {}
        for name, wanted in expected.items():
            actual = bundle.matrix(name)
            require(actual.shape == (12, 21) and np.isfinite(actual).all()
                    and np.all((actual >= 0) & (actual <= 1)),
                    f"{name} shape, finiteness, or mathematical bounds changed")
            np.testing.assert_allclose(actual, wanted, rtol=1e-13, atol=2e-15)
            max_errors[name] = float(np.max(np.abs(actual - wanted)))
            require(manifest["representations"][name]["shared_provenance_sha256"]
                    == manifest["shared_provenance_sha256"],
                    f"{name} does not share the common provenance identity")
            try:
                actual.setflags(write=True)
            except ValueError:
                pass
            else:
                raise RuntimeError(f"{name} exposed mutable feature storage")
        metadata_copy = bundle.manifest
        metadata_copy["accepted_epoch_ids"].clear()
        require(len(bundle.epoch_ids) == 12, "A returned metadata copy mutated the artifact")

        before_files = {path.name: sha256_file(path) for path in (target / "features").iterdir()}
        duplicate = cli("refuse-overwrite", *extract_args, expected_returncode=2)
        require(duplicate.get("status") == "blocked"
                and "Output already exists" in duplicate.get("error", "")
                and str(target / "features") in duplicate.get("error", ""),
                "Duplicate extraction failed for the wrong reason")
        after_files = {path.name: sha256_file(path) for path in (target / "features").iterdir()}
        require(before_files == after_files, "Refused overwrite changed an immutable feature artifact")
        require(sha256_file(ROOT / "configs/confirmatory_protocol.json") == primary_protocol_sha,
                "Phase-control CLI changed the A primary protocol")
        after_code = code_identity()
        require(after_code == before_code, "Source code changed during this reference-runtime smoke run")
        report.update(
            status="passed", bundle_sha256=manifest["bundle_sha256"],
            shared_provenance_sha256=manifest["shared_provenance_sha256"],
            n_synthetic_participants=3, n_epochs=12, features_per_representation=21,
            representations=["plv", "iplv", "ciplv"],
            direct_formula_max_absolute_error=max_errors,
            common_epoch_and_split_identity="verified", immutable_arrays_and_metadata="verified",
            immutable_output_no_overwrite="verified", real_draft_exit_code=2, real_draft_status="blocked",
            acquisition_or_model_execution=False, reference_environment_status="matched",
        )
    except Exception as exc:
        report.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        report["finished_at"] = datetime.now(timezone.utc).isoformat()
        report["duration_seconds"] = time.monotonic() - started
        write_json(target / "validation_report.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", required=True, type=Path)
    arguments = parser.parse_args()
    try:
        report = run(arguments.out_dir)
    except (OSError, ValueError, RuntimeError, AssertionError, subprocess.SubprocessError) as exc:
        print(json.dumps({"status": "failed", "error": str(exc)}, ensure_ascii=False))
        return 1
    print(json.dumps({key: report[key] for key in (
        "status", "scope", "reference_environment_status", "n_synthetic_participants", "n_epochs",
        "features_per_representation", "bundle_sha256", "real_draft_status", "direct_formula_max_absolute_error",
    )}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
