"""Synthetic prospective-context archives; no real B acquisition or performance."""
from __future__ import annotations

from copy import deepcopy
import io
import json
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import pytest

from eeg_ma import control_extraction as archive
from eeg_ma import control_plan, phase_controls
from eeg_ma.provenance import fingerprint, sha256_file
from test_control_plan import LOCKED_AT, prospective_manifests, ready_plan
from test_environment import synthetic_reference_environment


ROOT = Path(__file__).resolve().parents[1]
ARCHIVED_AT = "2020-01-04T00:00:00Z"


def synthetic_implementation():
    """Explicit test attestation with actual numerical source hashes, not a real commit."""
    names = ["phase_controls.py", "phase_metrics.py", "control_plan.py", "control_extraction.py"]
    files = {"src/eeg_ma/" + name: sha256_file(ROOT / "src/eeg_ma" / name) for name in names}
    return {"commit": "1" * 40, "dirty": False, "source_files": files,
            "working_source_sha256": fingerprint(files), "dirty_fingerprint": fingerprint({"fixture": "SYNTHETIC"}),
            "fixture_context": "SYNTHETIC test attestation; no real clean study commit is claimed"}


def _write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n")


def _reseal(root, *, refresh_binding=False):
    manifest = json.loads((root / archive.MANIFEST).read_text())
    if refresh_binding:
        bundle = phase_controls.verify_matched_phase_bundle(root / "features")
        locked = control_plan.validate_control_plan_lock(root / "plan_lock")
        authorization = json.loads((root / "authorization.json").read_text())
        history = json.loads((root / "session_history.json").read_text())
        annotations = json.loads((root / "annotation_manifest.json").read_text()) if (root / "annotation_manifest.json").exists() else None
        manifest.update(archive._binding(root, bundle, locked, authorization, history, annotations, manifest["arm"]))
    manifest["artifacts"] = archive._inventory(root, ignore=(archive.MANIFEST,))
    manifest.pop("archive_id")
    manifest["archive_id"] = fingerprint(manifest)
    _write(root / archive.MANIFEST, manifest)


@pytest.fixture(scope="module")
def extraction_base(tmp_path_factory):
    root = tmp_path_factory.mktemp("SYNTHETIC-control-extraction")
    patches = pytest.MonkeyPatch()
    patches.setattr(control_plan, "code_identity", synthetic_implementation)
    patches.setattr(control_plan, "environment_identity", synthetic_reference_environment)
    # Cross-version contract tests use an explicit synthetic environment proof.
    # Match that fixture's metadata observation only; numerical calculations
    # still execute in the current compatibility runtime. The separate reference
    # CLI smoke verifies actual, unmodified live runtime observations.
    proof = synthetic_reference_environment()
    patches.setattr(phase_controls, "observed_phase_runtime", lambda: {"python": proof["python"], "numpy": proof["packages"]["numpy"]})
    patches.setattr(control_plan, "_now", lambda: LOCKED_AT)
    try:
        path, spec = ready_plan(root / "source", sensitivities=True)
        control_plan.freeze_control_plan(path, root / "lock")
        patches.setattr(control_plan, "_now", lambda: ARCHIVED_AT)
        epochs, splits, history = prospective_manifests(spec)
        phases = np.random.default_rng(7281).uniform(-np.pi, np.pi, (len(epochs["epochs"]), 2500, 7))
        phase_path = root / "input_phases.npy"
        np.save(phase_path, phases, allow_pickle=False)
        original_lock = archive._inventory(root / "lock")
        manifest = archive.build_control_extraction(phase_path, epochs, splits, history, root / "extraction", lock_dir=root / "lock")
        assert archive._inventory(root / "lock") == original_lock
        yield {"root": root, "spec": spec, "epochs": epochs, "splits": splits, "history": history,
               "phases": phases, "manifest": manifest}
    finally:
        patches.undo()


@pytest.fixture
def extraction_case(extraction_base, tmp_path):
    root = Path(shutil.copytree(extraction_base["root"] / "extraction", tmp_path / "extraction"))
    return {**extraction_base, "archive": root}


def test_complete_authorization_phases_and_roster_are_archived(extraction_case):
    root = extraction_case["archive"]
    result = archive.verify_control_extraction(root)
    assert result == extraction_case["manifest"]
    assert result["status"] == "complete"
    assert result["created_at"] == ARCHIVED_AT
    assert result["n_recruited"] == 4 and result["n_source_participants"] == 3
    assert result["n_source_epochs"] == result["n_accepted_epochs"] == 180
    assert result["n_excluded_epochs"] == 0
    assert result["contains_model_performance_results"] is False
    assert result["scope"] == archive.SCOPE
    assert set(np.load(root / "phases.npy", allow_pickle=False).shape) == {180, 2500, 7}
    authorization = json.loads((root / "authorization.json").read_text())
    assert authorization["session_history_sha256"] == fingerprint(extraction_case["history"])
    assert authorization["arm_contract"]["session_history_sha256"] == authorization["session_history_sha256"]
    assert authorization["recruitment_audit"]["N_increment_from_repeat_sessions"] == 0
    assert authorization["recruitment_audit"]["flow"][-1]["subject_id"] == "lost"
    assert authorization["environment"]["platform"].startswith("SYNTHETIC")
    assert "SYNTHETIC" in authorization["implementation"]["fixture_context"]
    pure = phase_controls.verify_matched_phase_bundle(root / "features")
    expected = phase_controls.extract_matched_alpha_features(extraction_case["phases"], extraction_case["epochs"],
        extraction_case["splits"], input_policy=authorization["input_policy"])
    assert pure.manifest == expected.manifest
    for name in phase_controls.REPRESENTATIONS:
        np.testing.assert_array_equal(pure.matrix(name), expected.matrix(name))


def test_archive_validation_needs_neither_original_paths_nor_live_code_environment_or_clock(extraction_case, monkeypatch):
    root = extraction_case["archive"]
    def forbidden(*args, **kwargs):
        raise AssertionError("Archived verification must not consult live authorization, code, environment or audit clock")
    monkeypatch.setattr(control_plan, "authorize_phase_inputs", forbidden)
    monkeypatch.setattr(control_plan, "code_identity", forbidden)
    monkeypatch.setattr(control_plan, "environment_identity", forbidden)
    monkeypatch.setattr(control_plan, "_now", forbidden)
    # These describe this reader, not the historical computation. No numerical
    # operation is changed and no archived code is executed.
    monkeypatch.setattr(phase_controls.sys, "version", "0.0.0 SYNTHETIC different archive reader")
    monkeypatch.setattr(phase_controls.np, "__version__", "0.0.0-SYNTHETIC-reader")
    assert archive.verify_control_extraction(root)["archive_id"] == extraction_case["manifest"]["archive_id"]


@pytest.mark.parametrize("fault", ["phase_bytes", "missing_history", "changed_history", "changed_authorization",
    "forged_environment", "forged_implementation", "source_code_snapshot", "parent_lock", "extra_file", "model_performance", "future_created"])
def test_rehashed_archive_faults_do_not_bypass_source_authorization(extraction_case, fault):
    root = extraction_case["archive"]
    if fault == "phase_bytes":
        phases = np.load(root / "phases.npy", allow_pickle=False)
        phases[0, 0, 0] += .2
        np.save(root / "phases.npy", phases, allow_pickle=False)
    elif fault == "missing_history":
        (root / "session_history.json").unlink()
    elif fault == "changed_history":
        value = json.loads((root / "session_history.json").read_text())
        value["participants"].pop()
        _write(root / "session_history.json", value)
    elif fault == "changed_authorization":
        value = json.loads((root / "authorization.json").read_text())
        value["input_policy"]["selection_sha256"] = "9" * 64
        _write(root / "authorization.json", value)
    elif fault == "forged_environment":
        value = json.loads((root / "authorization.json").read_text())
        environment = value["environment"]
        environment["packages"]["numpy"] = "0.0.0"
        environment.pop("environment_id")
        environment["environment_id"] = fingerprint(environment)
        _write(root / "authorization.json", value)
    elif fault == "forged_implementation":
        value = json.loads((root / "authorization.json").read_text())
        value["implementation"]["source_files"]["src/eeg_ma/phase_controls.py"] = "9" * 64
        value["implementation"]["working_source_sha256"] = fingerprint(value["implementation"]["source_files"])
        _write(root / "authorization.json", value)
    elif fault == "source_code_snapshot":
        path = root / "implementation/phase_metrics.py"
        path.write_bytes(path.read_bytes() + b"\n# changed after extraction\n")
    elif fault == "parent_lock":
        path = root / "plan_lock/plan.json"
        value = json.loads(path.read_text())
        value["selection"]["target_fit"] = True
        _write(path, value)
    elif fault == "extra_file":
        (root / "unplanned_model.pkl").write_bytes(b"forbidden unplanned model artifact")
    elif fault == "model_performance":
        value = json.loads((root / archive.MANIFEST).read_text())
        value["contains_model_performance_results"] = True
        _write(root / archive.MANIFEST, value)
    else:
        value = json.loads((root / archive.MANIFEST).read_text())
        value["created_at"] = "2099-01-01T00:00:00Z"
        _write(root / archive.MANIFEST, value)
    _reseal(root)
    with pytest.raises(archive.ControlExtractionError):
        archive.verify_control_extraction(root)


def test_coherently_rehashed_numerical_bundle_must_match_the_actual_stored_phases(extraction_case):
    root = extraction_case["archive"]
    original = phase_controls.verify_matched_phase_bundle(root / "features")
    changed = np.load(root / "phases.npy", allow_pickle=False)
    changed[0, :, 0] += .7
    wrong = phase_controls.extract_matched_alpha_features(changed, original.manifest["epoch_manifest"],
        original.manifest["split_manifest"], input_policy=original.manifest["input_policy"])
    shutil.rmtree(root / "features")
    phase_controls.export_matched_phase_bundle(wrong, root / "features")
    # The pure numerical artifact is internally valid, but it is about a
    # different tensor. Even a fully rehashed outer manifest must reject it.
    assert phase_controls.verify_matched_phase_bundle(root / "features").manifest == wrong.manifest
    _reseal(root, refresh_binding=True)
    with pytest.raises(archive.ControlExtractionError, match="actual archived phase tensor|Recomputed"):
        archive.verify_control_extraction(root)


def test_undefined_ciplv_keeps_the_whole_source_roster_and_frozen_splits(extraction_base, tmp_path):
    phases = np.zeros_like(extraction_base["phases"])
    out = tmp_path / "all-undefined"
    manifest = archive.build_control_extraction(phases, extraction_base["epochs"], extraction_base["splits"],
        extraction_base["history"], out, lock_dir=extraction_base["root"] / "lock")
    assert manifest["status"] == "complete" and not manifest["evaluable"]
    assert manifest["n_recruited"] == 4 and manifest["n_source_participants"] == 3
    assert manifest["n_source_epochs"] == manifest["n_excluded_epochs"] == 180
    assert manifest["n_accepted_epochs"] == 0
    bundle = phase_controls.verify_matched_phase_bundle(out / "features")
    assert bundle.manifest["split_manifest"] == extraction_base["splits"]
    assert bundle.plv.shape == bundle.iplv.shape == bundle.ciplv.shape == (0, 21)
    assert any(finding["code"] == "UNEVALUABLE_PARTICIPANT_EPOCH_COUNT" for finding in bundle.manifest["evaluation_findings"])
    assert archive.verify_control_extraction(out)["archive_id"] == manifest["archive_id"]


def test_annotation_arm_keeps_full_input_coverage_and_records_common_exclusion(extraction_base, tmp_path):
    epochs = deepcopy(extraction_base["epochs"])
    epochs["epochs"][0].update(qc_passed=False, qc_reason_codes=["BLINDED_ARTIFACT_ANNOTATION"])
    annotations = {"schema_version": "eeg-control-annotation-1",
        "ontology": extraction_base["spec"]["matrix"]["artifact_sensitivity"]["annotation_ontology"],
        "blinded_to_outcomes": True, "excluded_epoch_ids": [epochs["epochs"][0]["epoch_id"]], "evidence_sha256": "d" * 64}
    out = tmp_path / "annotated"
    manifest = archive.build_control_extraction(extraction_base["phases"], epochs, extraction_base["splits"],
        extraction_base["history"], out, lock_dir=extraction_base["root"] / "lock", arm="annotation_clean", annotation_manifest=annotations)
    assert manifest["n_source_epochs"] == 180 and manifest["n_accepted_epochs"] == 179
    assert manifest["annotation_manifest_sha256"] == fingerprint(annotations)
    assert json.loads((out / "annotation_manifest.json").read_text()) == annotations
    features = phase_controls.verify_matched_phase_bundle(out / "features")
    assert features.manifest["exclusions"][0]["reason_codes"] == ["SOURCE_QC_FAILED"]
    assert len(features.epoch_ids) == 179
    assert archive.verify_control_extraction(out)["arm"] == "annotation_clean"


def test_undeclared_annotation_metadata_is_rejected_before_publication(extraction_base, tmp_path):
    out = tmp_path / "must-not-exist"
    with pytest.raises(archive.ControlExtractionError, match="Annotation evidence"):
        archive.build_control_extraction(extraction_base["phases"], extraction_base["epochs"], extraction_base["splits"],
            extraction_base["history"], out, lock_dir=extraction_base["root"] / "lock", annotation_manifest={"unexpected": True})
    assert not out.exists()


def test_existing_destination_and_parent_lock_are_never_modified(extraction_base):
    root = extraction_base["root"]
    before_lock = archive._inventory(root / "lock")
    before_archive = archive._inventory(root / "extraction")
    with pytest.raises(FileExistsError):
        archive.build_control_extraction(extraction_base["phases"], extraction_base["epochs"], extraction_base["splits"],
            extraction_base["history"], root / "extraction", lock_dir=root / "lock")
    assert archive._inventory(root / "lock") == before_lock
    assert archive._inventory(root / "extraction") == before_archive


@pytest.mark.parametrize("fault", ["history", "phases", "plan_lock"])
def test_source_changes_during_extraction_abort_atomic_publication(extraction_base, tmp_path, monkeypatch, fault):
    phases = extraction_base["phases"].copy()
    history = deepcopy(extraction_base["history"])
    lock = Path(shutil.copytree(extraction_base["root"] / "lock", tmp_path / "lock"))
    actual_extract = archive.extract_matched_alpha_features
    changed = False
    def mutate(*args, **kwargs):
        nonlocal changed
        result = actual_extract(*args, **kwargs)
        if not changed:
            changed = True
            if fault == "history":
                history["stop_reason_detail"] = "changed while extraction was running"
            elif fault == "phases":
                phases[0, 0, 0] += .3
            else:
                path = lock / "plan.json"
                path.write_bytes(path.read_bytes() + b"\n")
        return result
    monkeypatch.setattr(archive, "extract_matched_alpha_features", mutate)
    out = tmp_path / "must-not-exist"
    with pytest.raises(archive.ControlExtractionError, match="changed"):
        archive.build_control_extraction(phases, extraction_base["epochs"], extraction_base["splits"], history, out, lock_dir=lock)
    assert not out.exists()
    assert not list(tmp_path.glob(".must-not-exist.staging-*"))


def test_archive_history_is_audited_at_its_creation_time(extraction_case):
    root = extraction_case["archive"]
    value = json.loads((root / archive.MANIFEST).read_text())
    value["created_at"] = "2020-01-02T00:00:00Z"
    _write(root / archive.MANIFEST, value)
    _reseal(root)
    with pytest.raises(archive.ControlExtractionError, match="current audit time"):
        archive.verify_control_extraction(root)


@pytest.mark.parametrize("fault", ["pickle", "trailing_bytes", "nan"])
def test_phase_input_is_safe_exact_finite_tensor(extraction_base, tmp_path, fault):
    path = tmp_path / "phase.npy"
    if fault == "pickle":
        np.save(path, np.asarray([{"not": "numerical phases"}], dtype=object), allow_pickle=True)
    else:
        phases = extraction_base["phases"].copy()
        if fault == "nan":
            phases[0, 0, 0] = np.nan
        np.save(path, phases, allow_pickle=False)
        if fault == "trailing_bytes":
            with path.open("ab") as stream:
                stream.write(b"unexpected second payload")
    out = tmp_path / "must-not-exist"
    with pytest.raises(archive.ControlExtractionError):
        archive.build_control_extraction(path, extraction_base["epochs"], extraction_base["splits"],
            extraction_base["history"], out, lock_dir=extraction_base["root"] / "lock")
    assert not out.exists()


def test_archive_source_symlinks_are_rejected(extraction_case, tmp_path):
    root = extraction_case["archive"]
    path = root / "phases.npy"
    real = tmp_path / "outside.npy"
    path.rename(real)
    path.symlink_to(real)
    with pytest.raises(archive.ControlExtractionError, match="symlink"):
        archive.verify_control_extraction(root)


def test_verify_features_CLI_checks_the_entire_prospective_archive(extraction_case):
    run = subprocess.run([sys.executable, str(ROOT / "scripts/23_phase_controls.py"), "verify-features",
                          "--bundle", str(extraction_case["archive"])], capture_output=True, text=True)
    assert run.returncode == 0, run.stdout + run.stderr
    result = json.loads(run.stdout)
    assert result["valid"] is True
    assert result["archive_id"] == extraction_case["manifest"]["archive_id"]
    assert result["n_recruited"] == 4
    assert result["scope"] == archive.SCOPE


def test_new_extraction_runtime_cannot_disagree_with_its_authorization(extraction_base, tmp_path, monkeypatch):
    monkeypatch.setattr(phase_controls, "observed_phase_runtime", lambda: {"python": "0.0.0", "numpy": "0.0.0"})
    out = tmp_path / "must-not-exist"
    with pytest.raises(archive.ControlExtractionError, match="runtime observations"):
        archive.build_control_extraction(extraction_base["phases"], extraction_base["epochs"], extraction_base["splits"],
            extraction_base["history"], out, lock_dir=extraction_base["root"] / "lock")
    assert not out.exists()
