"""Protocol gate tests use synthetic evidence, never a claimed live study lock."""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

import eeg_ma.protocol as gate
from eeg_ma.provenance import fingerprint, sha256_file


ROOT = Path(__file__).resolve().parents[1]
LOCK_TIME = "2020-01-01T00:00:00+00:00"
ACCESS_TIME = "2020-01-03T00:00:00+00:00"


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def synthetic_code() -> dict:
    files = {"synthetic_test.py": hashlib.sha256(b"synthetic fixture only").hexdigest()}
    return {"commit": "1" * 40, "dirty": False, "dirty_fingerprint": fingerprint({}),
            "source_files": files, "working_source_sha256": fingerprint(files)}


def complete_spec(tmp_path: Path, *, subject_ids=None) -> tuple[Path, dict]:
    """Create actual hashed synthetic artifacts; caller must mock code/time to lock.

    The precision report and feature/config contracts are the real repository
    design artifacts. Model binaries, acquisition/approval and people are marked
    synthetic. This fixture cannot establish real model or study readiness.
    """
    tmp_path.mkdir(parents=True, exist_ok=True)
    inputs = tmp_path / "inputs"
    inputs.mkdir(exist_ok=True)
    spec = json.loads((ROOT / "configs/confirmatory_protocol.json").read_text())
    spec["acquisition"].update(units="uV", reference="SYNTHETIC reference", ground="SYNTHETIC ground",
        device="SYNTHETIC test device", evidence="SYNTHETIC verification record", verified=True,
        historical_alignment_verified=True)
    spec["approval"].update(responsible_person="SYNTHETIC test operator", approved_at="2019-12-01T00:00:00Z",
        ethics_status="SYNTHETIC test approval", ethics_reference="SYNTHETIC fixture only",
        resource_commitment_confirmed=True)
    precision_path = ROOT / "docs/precision/precision_report.json"
    precision = json.loads(precision_path.read_text())
    spec["recruitment"].update(target_evaluable=precision["selected_target_n"], recruitment_cap=precision["recruitment_cap"])
    ids = ["001", "010", "dev-old"] if subject_ids is None else subject_ids

    def artifact(role, value, *, binary=False):
        path = inputs / (role + (".bin" if binary else ".json"))
        if binary:
            path.write_bytes(value)
        else:
            write_json(path, value)
        spec["freeze"]["artifacts"][role] = {"path": str(path.relative_to(tmp_path)), "sha256": sha256_file(path)}
        return path

    for role, path in (("analysis_config", ROOT / "configs/confirmatory_analysis.json"),
                       ("feature_schema", ROOT / "configs/confirmatory_feature_schema.json"),
                       ("precision_report", precision_path)):
        artifact(role, path.read_bytes(), binary=True)
    data = inputs / "synthetic-development.csv"
    data.write_text("subject_id,value\n" + "".join(f"{subject},1.0\n" for subject in ids))
    artifact("development_data_manifest", {"schema_version": "eeg-development-data-1",
        "sources": [{"name": "synthetic", "path": data.name, "sha256": sha256_file(data)}],
        "historical_raw_sha256": "unknown"})
    artifact("development_manifest", {"schema_version": "eeg-development-manifest-1", "cohort_id": "synthetic-development",
        "subject_ids": ids, "data_sha256": spec["freeze"]["artifacts"]["development_data_manifest"]["sha256"],
        "feature_schema_sha256": spec["freeze"]["artifacts"]["feature_schema"]["sha256"], "use": "development_only"})
    from test_environment import synthetic_reference_environment
    synthetic_environment = synthetic_reference_environment()
    artifact("environment", synthetic_environment)
    artifact("acquisition_evidence", {"schema_version": "eeg-acquisition-verification-1", "verified": True,
        "acquisition_sha256": fingerprint(spec["acquisition"]), "verified_by": "SYNTHETIC test operator",
        "verified_at": "2019-12-01T00:00:00Z"})
    for role, learner in (("primary", "alpha_plv_sfs_lda"), ("reference", "bp_coh_sfs_lda")):
        artifact(role + "_model", ("SYNTHETIC opaque model fixture " + role).encode(), binary=True)
        frozen = spec["freeze"]["artifacts"]
        artifact(role + "_training_manifest", {"schema_version": "eeg-training-manifest-1", "model_role": role,
            "learner_id": learner, "model_sha256": frozen[role + "_model"]["sha256"],
            "development_manifest_sha256": frozen["development_manifest"]["sha256"],
            "analysis_config_sha256": frozen["analysis_config"]["sha256"],
            "environment_sha256": frozen["environment"]["sha256"],
            "environment_id": synthetic_environment["environment_id"],
            "environment_lock_id": synthetic_environment["lock_id"],
            "feature_schema_sha256": frozen["feature_schema"]["sha256"],
            "training_data_sha256": frozen["development_data_manifest"]["sha256"],
            "training_subject_ids": ids, "eligibility_rule_sha256": fingerprint(spec["eligibility"]),
            "fitted_on_development_only": True, "target_fit_performed": False})
    path = tmp_path / "protocol.json"
    write_json(path, spec)
    return path, spec


def cohort_for(spec: dict, ids=None) -> dict:
    return {"schema_version": "eeg-confirmatory-cohort-1", "cohort_id": spec["study"]["cohort_id"],
            "protocol_id": spec["protocols"]["A"]["protocol_id"], "outcomes_accessed": False,
            "subject_ids": [f"new-{i:03d}" for i in range(spec["recruitment"]["target_evaluable"])] if ids is None else ids}


def enrollment_for(spec: dict, cohort: dict) -> dict:
    from test_cohort_policy import person
    people = []
    for subject in cohort["subject_ids"]:
        item = person(spec, subject)
        item["recruited_at"] = "2020-01-01T09:00:00Z"
        item["closed_at"] = "2020-01-02T11:00:00Z"
        item["sessions"][0]["started_at"] = "2020-01-01T10:00:00Z"
        people.append(item)
    return {"schema_version": "eeg-enrollment-1", "cohort_id": cohort["cohort_id"],
            "outcomes_accessed": False, "participants": people}


@pytest.fixture
def locked(tmp_path, monkeypatch):
    path, spec = complete_spec(tmp_path / "source")
    monkeypatch.setattr(gate, "code_identity", synthetic_code)
    monkeypatch.setattr(gate, "_now", lambda: LOCK_TIME)
    out = tmp_path / "locked"
    gate.lock_protocol(path, out)
    monkeypatch.setattr(gate, "_now", lambda: ACCESS_TIME)
    cohort = cohort_for(spec)
    enrollment = enrollment_for(spec, cohort)
    cp, ep = tmp_path / "cohort.json", tmp_path / "enrollment.json"
    write_json(cp, cohort)
    write_json(ep, enrollment)
    return out, path, spec, cp, ep


def test_repository_draft_blocks_without_inventing_live_evidence(tmp_path):
    path = ROOT / "configs/confirmatory_protocol.json"
    report = gate.validate_protocol(json.loads(path.read_text()))
    assert not report.ready
    assert {"acquisition.verified", "approval.resource_commitment_confirmed"} <= {finding.path for finding in report.findings}
    with pytest.raises(gate.ProtocolError):
        gate.lock_protocol(path, tmp_path / "must-not-exist")
    assert not (tmp_path / "must-not-exist").exists()
    snapshot = json.loads((ROOT / "docs/protocol_readiness.json").read_text())
    assert snapshot["protocol_sha256"] == sha256_file(path)
    assert snapshot["validator_sha256"] == sha256_file(ROOT / "src/eeg_ma/protocol.py")
    assert snapshot["decision_readiness"] == report.as_dict()


def test_lock_is_self_contained_and_never_overwritten(locked):
    out, path, _, cp, _ = locked
    first = gate.validate_lock(out)
    shutil.rmtree(path.parent)
    assert gate.validate_lock(out)["manifest"] == first["manifest"]
    assert gate.check_launch(out, cp)["status"] == "ready"
    assert first["manifest"]["data_sources"]


def test_dirty_code_prevents_atomic_publication(tmp_path, monkeypatch):
    path, _ = complete_spec(tmp_path / "source")
    monkeypatch.setattr(gate, "code_identity", lambda: {**synthetic_code(), "dirty": True})
    with pytest.raises(gate.ProtocolError, match="clean"):
        gate.lock_protocol(path, tmp_path / "bad")
    assert not (tmp_path / "bad").exists()


@pytest.mark.parametrize("model_role", ["primary", "reference"])
@pytest.mark.parametrize("fault", ["missing", "mismatched"])
def test_rehashed_training_manifest_must_bind_the_frozen_environment(tmp_path, monkeypatch, model_role, fault):
    path, spec = complete_spec(tmp_path / "source")
    role = model_role + "_training_manifest"
    manifest_path = path.parent / spec["freeze"]["artifacts"][role]["path"]
    training = json.loads(manifest_path.read_text())
    if fault == "missing":
        del training["environment_sha256"]
    else:
        training["environment_sha256"] = "f" * 64
    write_json(manifest_path, training)
    spec["freeze"]["artifacts"][role]["sha256"] = sha256_file(manifest_path)
    write_json(path, spec)
    monkeypatch.setattr(gate, "code_identity", synthetic_code)
    monkeypatch.setattr(gate, "_now", lambda: LOCK_TIME)
    out = tmp_path / "rejected"
    with pytest.raises(gate.ProtocolError, match=model_role + " training provenance"):
        gate.lock_protocol(path, out)
    assert not out.exists()


def test_unrelated_rehashed_environment_cannot_replace_the_models_training_environment(tmp_path, monkeypatch):
    path, spec = complete_spec(tmp_path / "source")
    environment_path = path.parent / spec["freeze"]["artifacts"]["environment"]["path"]
    environment = json.loads(environment_path.read_text())
    environment["python"] = "3.13.0"
    write_json(environment_path, environment)
    spec["freeze"]["artifacts"]["environment"]["sha256"] = sha256_file(environment_path)
    write_json(path, spec)
    monkeypatch.setattr(gate, "code_identity", synthetic_code)
    monkeypatch.setattr(gate, "_now", lambda: LOCK_TIME)
    out = tmp_path / "rejected"
    with pytest.raises(gate.ProtocolError, match="environment|training provenance"):
        gate.lock_protocol(path, out)
    assert not out.exists()


@pytest.mark.parametrize("ids", [["001"], ["010"], ["new", "new"], [1], [" new"]])
def test_literal_identity_overlap_and_coercions_fail(locked, ids):
    out, _, spec, _, _ = locked
    with pytest.raises(gate.ProtocolError):
        gate.check_launch(out, cohort_for(spec, ids))
    assert gate.check_launch(out, cohort_for(spec, ["1"]))["subject_ids"] == ["1"]


@pytest.mark.parametrize("role", ["primary_model", "precision_report", "development_manifest"])
def test_modified_frozen_artifact_fails(locked, role):
    out = locked[0]
    artifact = out / "artifacts" / (role + ".bin")
    artifact.write_bytes(artifact.read_bytes() + b" ")
    with pytest.raises(gate.ProtocolError, match="modified"):
        gate.validate_lock(out)


def test_outcome_access_requires_closed_stop_then_closes_enrollment(locked):
    out, _, spec, cp, ep = locked
    enrollment = json.loads(ep.read_text())
    enrollment["participants"].pop()
    cohort = json.loads(cp.read_text())
    cohort["subject_ids"].pop()
    write_json(cp, cohort)
    write_json(ep, enrollment)
    with pytest.raises(gate.ProtocolError, match="stopped"):
        gate.record_outcome_access(out, cp, enrollment_manifest_path=ep)
    cohort = cohort_for(spec)
    write_json(cp, cohort)
    write_json(ep, enrollment_for(spec, cohort))
    event = gate.record_outcome_access(out, cp, enrollment_manifest_path=ep)
    assert event["n_evaluable"] == spec["recruitment"]["target_evaluable"]
    assert gate.validate_lock(out)["exposed_subject_ids"] == sorted(cohort["subject_ids"])
    with pytest.raises(gate.ProtocolError, match="closed"):
        gate.check_launch(out, cohort_for(spec, ["another-untouched"]))


@pytest.mark.parametrize("field,value", [("recruited_at", "2019-12-31T09:00:00Z"), ("closed_at", "2020-01-04T00:00:00Z")])
def test_recruitment_and_unblinding_chronology_fail_closed(locked, field, value):
    out, _, _, cp, ep = locked
    enrollment = json.loads(ep.read_text())
    enrollment["participants"][0][field] = value
    write_json(ep, enrollment)
    with pytest.raises(gate.ProtocolError):
        gate.record_outcome_access(out, cp, enrollment_manifest_path=ep)
    assert not list((out / "events").iterdir())


def test_interrupted_ledger_append_never_silently_erases_exposure(locked, monkeypatch):
    out, _, _, cp, ep = locked
    original = gate._atomic_json
    def fail_pointer(path, value):
        if Path(path).name == "STATE.json":
            raise OSError("simulated pointer publication failure")
        original(path, value)
    monkeypatch.setattr(gate, "_atomic_json", fail_pointer)
    with pytest.raises(OSError, match="simulated"):
        gate.record_outcome_access(out, cp, enrollment_manifest_path=ep)
    with pytest.raises(gate.ProtocolError, match="Orphan"):
        gate.validate_lock(out)
    monkeypatch.setattr(gate, "_atomic_json", original)
    event = gate.recover_event_append(out)
    assert gate.validate_lock(out)["events"] == [event]
    with pytest.raises(gate.ProtocolError, match="closed"):
        gate.check_launch(out, cp)


def test_amendments_preserve_parent_and_require_new_post_outcome_holdout(locked, tmp_path):
    out, path, spec, cp, ep = locked
    gate.record_outcome_access(out, cp, enrollment_manifest_path=ep)
    before = {str(p.relative_to(out)): p.read_bytes() for p in out.rglob("*") if p.is_file()}
    amended = deepcopy(spec)
    amended["protocol_version"] = "2.0"
    amended["endpoints"]["delta_min"] = .07
    amended["study"]["cohort_id"] = "fresh-v2"
    write_json(path, amended)
    with pytest.raises(gate.ProtocolError, match="untouched"):
        gate.amend_protocol(out, path, tmp_path / "bad", reason="SYNTHETIC revised threshold")
    assert not (tmp_path / "bad").exists()
    write_json(cp, cohort_for(amended, ["fresh-001", "fresh-002"]))
    result = gate.amend_protocol(out, path, tmp_path / "amended", reason="SYNTHETIC revised threshold", holdout_manifest_path=cp)
    assert result["lineage"]["parent_lock_id"] == gate.validate_lock(out)["manifest"]["lock_id"]
    assert {str(p.relative_to(out)): p.read_bytes() for p in out.rglob("*") if p.is_file()} == before
    new = gate.validate_lock(tmp_path / "amended")
    assert new["exposed_subject_ids"] == sorted(cohort_for(spec)["subject_ids"])
    assert gate.check_launch(tmp_path / "amended", cp)["status"] == "ready"


def test_pilot_downgrade_is_prospective_only(locked, tmp_path):
    out, path, spec, cp, ep = locked
    pilot = deepcopy(spec)
    pilot["protocol_version"] = "2.0"
    pilot["study"]["mode"] = "pilot_estimation"
    write_json(path, pilot)
    gate.amend_protocol(out, path, tmp_path / "pilot", reason="SYNTHETIC prospective estimation-only decision")
    gate.record_outcome_access(out, cp, enrollment_manifest_path=ep)
    with pytest.raises(gate.ProtocolError, match="downgrade"):
        gate.amend_protocol(out, path, tmp_path / "late", reason="SYNTHETIC retrospective downgrade", holdout_manifest_path=cp)


def test_expanded_development_requires_new_holdout_and_preserves_all_exclusions(locked, tmp_path):
    out, _, _, _, _ = locked
    path, spec = complete_spec(tmp_path / "expanded", subject_ids=["001", "010", "new-development"])
    spec["protocol_version"] = "2.0"
    spec["study"]["cohort_id"] = "fresh-expanded"
    write_json(path, spec)
    with pytest.raises(gate.ProtocolError, match="expanded-development"):
        gate.amend_protocol(out, path, tmp_path / "bad", reason="SYNTHETIC expanded development")
    cp = tmp_path / "fresh.json"
    write_json(cp, cohort_for(spec, ["fresh-001"]))
    gate.amend_protocol(out, path, tmp_path / "expanded-lock", reason="SYNTHETIC expanded development", holdout_manifest_path=cp)
    with pytest.raises(gate.ProtocolError, match="overlap"):
        gate.check_launch(tmp_path / "expanded-lock", cohort_for(spec, ["dev-old"]))


def test_cli_blocked_draft_returns_nonzero_without_output(tmp_path):
    result = subprocess.run([sys.executable, str(ROOT / "scripts/16_protocol.py"), "lock",
        "--protocol", str(ROOT / "configs/confirmatory_protocol.json"), "--out-dir", str(tmp_path / "bad")],
        text=True, capture_output=True, cwd=ROOT)
    assert result.returncode == 2
    assert json.loads(result.stdout)["status"] == "blocked"
    assert not (tmp_path / "bad").exists()


@pytest.mark.parametrize("before_lock", [False, True])
def test_enrollment_cli_binds_lock_chronology_and_preserves_audit(locked, tmp_path, before_lock):
    out, _, _, _, ep = locked
    if before_lock:
        enrollment = json.loads(ep.read_text())
        enrollment["participants"][0]["recruited_at"] = "2019-12-31T09:00:00Z"
        write_json(ep, enrollment)
    destination = tmp_path / "audit"
    command = [sys.executable, str(ROOT / "scripts/17_audit_enrollment.py"),
        "--lock", str(out), "--enrollment", str(ep), "--out-dir", str(destination)]
    result = subprocess.run(command, text=True, capture_output=True, cwd=ROOT)
    assert result.returncode == (2 if before_lock else 0), result.stderr
    report_path = destination / "enrollment_audit.json"
    report = json.loads(report_path.read_text())
    assert report["valid"] is not before_lock
    assert report["outcome_ready"] is not before_lock
    assert report["enrollment_sha256"] == sha256_file(ep)
    digest = sha256_file(report_path)
    repeated = subprocess.run(command, text=True, capture_output=True, cwd=ROOT)
    assert repeated.returncode == 2 and sha256_file(report_path) == digest
