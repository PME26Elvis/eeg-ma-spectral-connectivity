"""Executable-model protocol identities and external outcome-access boundaries."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import pytest

from eeg_ma import artifacts, protocol
from eeg_ma.provenance import ProvenanceError, fingerprint, sha256_file
from test_external_workflow import external_base, external_case
from test_protocol_gate import complete_spec, locked, synthetic_code, write_json, LOCK_TIME


ROOT = Path(__file__).resolve().parents[1]


def _draft():
    return json.loads((ROOT / "configs/confirmatory_protocol.json").read_bytes())


def test_training_contract_survives_artifact_freeze_and_administrative_changes():
    draft = _draft()
    before = deepcopy(draft)
    wanted = protocol.protocol_training_contract_sha256(draft)
    locked = deepcopy(draft)
    locked["status"] = "locked"
    locked["freeze"]["artifacts"] = {"primary_model": {"path": "artifacts/primary_model.bin", "sha256": "a" * 64}}
    locked["approval"] = {"responsible_person": "SYNTHETIC operator", "approved_at": "2020-01-01T00:00:00Z"}
    locked["study"]["registration"] = {"status": "registered", "url": "https://example.invalid/synthetic"}
    locked["study"]["outcomes_accessed"] = True
    assert protocol.protocol_training_contract_sha256(locked) == wanted
    assert draft == before


@pytest.mark.parametrize("path,value", [
    (("protocol_version",), "a-different-version"),
    (("study", "cohort_id"), "a-different-cohort"),
    (("study", "mode"), "pilot_estimation"),
    (("acquisition", "units"), "mV"),
    (("eligibility", "min_epochs_per_class"), 1),
    (("endpoints", "absolute_threshold"), 0.9),
    (("inference", "seed"), 123),
    (("recruitment", "target_evaluable"), 20),
    (("freeze", "environment_policy"), "a-weaker-policy"),
    (("future_scientific_decision",), {"policy": "must-also-be-bound"}),
])
def test_training_contract_binds_scientific_changes_including_future_fields(path, value):
    draft = _draft()
    wanted = protocol.protocol_training_contract_sha256(draft)
    modified = deepcopy(draft)
    parent = modified
    for key in path[:-1]:
        parent = parent[key]
    parent[path[-1]] = value
    assert protocol.protocol_training_contract_sha256(modified) != wanted


@pytest.mark.parametrize("spec", [None, {}, {"schema_version": protocol.PROTOCOL_SCHEMA}])
def test_training_contract_does_not_invent_missing_protocol_identity(spec):
    with pytest.raises(protocol.ProtocolError, match="Training contract"):
        protocol.protocol_training_contract(spec)


@pytest.mark.parametrize("policy", [None, "opaque", False])
def test_explicit_executable_model_policy_cannot_be_weakened(policy):
    spec = _draft()
    spec["freeze"]["model_artifact_policy"] = policy
    assert any(item.path == "freeze.model_artifact_policy" for item in protocol.validate_protocol(spec).findings)


def test_executable_model_policy_requires_acquisition_protocol_version():
    spec = _draft()
    del spec["protocols"]["A"]["protocol_version"]
    assert any(item.path == "protocols.A.protocol_version" for item in protocol.validate_protocol(spec).findings)


def test_current_policy_cannot_freeze_legacy_opaque_models(tmp_path, monkeypatch):
    source, spec = complete_spec(tmp_path / "legacy-source")
    spec["freeze"]["model_artifact_policy"] = "executable_fixed_lda_v1"
    write_json(source, spec)
    monkeypatch.setattr(protocol, "code_identity", synthetic_code)
    monkeypatch.setattr(protocol, "_now", lambda: LOCK_TIME)
    with pytest.raises(protocol.ProtocolError, match="executable model"):
        protocol.lock_protocol(source, tmp_path / "rejected")
    assert not (tmp_path / "rejected").exists()


def test_legacy_lock_is_valid_archive_but_cannot_authorize_external_prediction(locked):
    out, _, _, cohort, _ = locked
    assert protocol.validate_lock(out)["manifest"]["status"] == "locked"
    with pytest.raises(protocol.ProtocolError, match="executable model"):
        protocol.check_external_prediction(out, cohort)


def _commit(locked):
    """Temporal tests isolate the ledger contract; full fitted-model tests are separate."""
    out, _, _, cohort, _ = locked
    return {"prediction_id": "a" * 64,
        "protocol_lock_id": protocol.validate_lock(out)["manifest"]["lock_id"],
        "cohort_manifest_sha256": sha256_file(cohort),
        "cohort_subject_ids": sorted(json.loads(cohort.read_bytes())["subject_ids"]),
        "created_at": "2020-01-02T12:00:00Z"}


def test_score_authorization_records_and_reuses_exact_prediction_commit(locked):
    out, _, _, cohort, enrollment = locked
    commit = _commit(locked)
    first = protocol.authorize_external_score(out, cohort, enrollment, prediction_commit=commit)
    assert first["event"]["prediction_commit"] == commit
    assert first["audit"]["outcome_ready"]
    before = {str(p.relative_to(out)): p.read_bytes() for p in out.rglob("*") if p.is_file()}
    retry = protocol.authorize_external_score(out, cohort, enrollment, prediction_commit=commit)
    assert retry["event"] == first["event"]
    assert {str(p.relative_to(out)): p.read_bytes() for p in out.rglob("*") if p.is_file()} == before
    with pytest.raises(protocol.ProtocolError, match="closed prediction"):
        protocol.check_external_prediction(out, cohort)


@pytest.mark.parametrize("fault", ["different_prediction", "changed_enrollment_bytes", "changed_cohort_bytes"])
def test_score_retry_cannot_rebind_exposed_cohort_to_different_evidence(locked, fault):
    out, _, _, cohort, enrollment = locked
    commit = _commit(locked)
    first = protocol.authorize_external_score(out, cohort, enrollment, prediction_commit=commit)
    if fault == "different_prediction":
        commit["prediction_id"] = "b" * 64
    else:
        path = enrollment if fault == "changed_enrollment_bytes" else cohort
        path.write_text(json.dumps(json.loads(path.read_bytes()), indent=4) + "\n")
    with pytest.raises(protocol.ProtocolError, match="different|exact committed"):
        protocol.authorize_external_score(out, cohort, enrollment, prediction_commit=commit)
    assert protocol.validate_lock(out)["events"] == [first["event"]]


@pytest.mark.parametrize("fault", ["before_lock", "after_access", "future", "wrong_lock", "wrong_cohort", "partial_roster", "extra_field"])
def test_invalid_prediction_commit_cannot_open_outcome_access(locked, fault):
    out, _, _, cohort, enrollment = locked
    commit = _commit(locked)
    if fault == "before_lock":
        commit["created_at"] = "2019-12-31T00:00:00Z"
    elif fault == "after_access":
        commit["created_at"] = "2020-01-04T00:00:00Z"
    elif fault == "future":
        commit["created_at"] = "2999-01-01T00:00:00Z"
    elif fault == "wrong_lock":
        commit["protocol_lock_id"] = "b" * 64
    elif fault == "wrong_cohort":
        commit["cohort_manifest_sha256"] = "b" * 64
    elif fault == "partial_roster":
        commit["cohort_subject_ids"] = commit["cohort_subject_ids"][:-1]
    else:
        commit["true_labels"] = [0, 1]
    with pytest.raises(protocol.ProtocolError):
        protocol.authorize_external_score(out, cohort, enrollment, prediction_commit=commit)
    assert protocol.validate_lock(out)["events"] == []
    assert not list((out / "events").iterdir())


def test_unbound_historical_outcome_event_cannot_authorize_new_scores(locked):
    out, _, _, cohort, enrollment = locked
    commit = _commit(locked)
    event = protocol.record_outcome_access(out, cohort, enrollment_manifest_path=enrollment)
    assert "prediction_commit" not in event
    with pytest.raises(protocol.ProtocolError, match="not bound"):
        protocol.authorize_external_score(out, cohort, enrollment, prediction_commit=commit)
    assert protocol.validate_lock(out)["events"] == [event]


@pytest.mark.parametrize("created_at", ["2019-12-31T00:00:00Z", "2020-01-04T00:00:00Z"])
def test_rehashed_prediction_event_still_checks_chronology(locked, created_at):
    out, _, _, cohort, enrollment = locked
    event = protocol.authorize_external_score(out, cohort, enrollment, prediction_commit=_commit(locked))["event"]
    old_id = event.pop("event_id")
    event["prediction_commit"]["created_at"] = created_at
    event["event_id"] = fingerprint(event)
    (out / "events" / (old_id + ".json")).unlink()
    write_json(out / "events" / (event["event_id"] + ".json"), event)
    state = json.loads((out / "STATE.json").read_bytes())
    state["head"] = event["event_id"]
    write_json(out / "STATE.json", state)
    with pytest.raises(protocol.ProtocolError, match="predates|before outcome"):
        protocol.validate_lock(out)


def _rebind_lock_inventory(root):
    """Rehash outer evidence so attacks reach scientific/semantic validators."""
    spec = json.loads((root / "protocol.json").read_bytes())
    manifest = json.loads((root / "protocol_lock.json").read_bytes())
    for role, entry in manifest["artifacts"].items():
        path = root / entry["file"]
        entry.update(sha256=sha256_file(path), bytes=path.stat().st_size)
        spec["freeze"]["artifacts"][role]["sha256"] = entry["sha256"]
    write_json(root / "protocol.json", spec)
    manifest["protocol_sha256"] = sha256_file(root / "protocol.json")
    manifest.pop("lock_id")
    manifest["lock_id"] = fingerprint(manifest)
    write_json(root / "protocol_lock.json", manifest)
    state = json.loads((root / "STATE.json").read_bytes())
    assert state["head"] is None
    state["lock_id"] = manifest["lock_id"]
    write_json(root / "STATE.json", state)


@pytest.mark.parametrize("fault", [
    "software_validation_scope", "dirty_training_code", "unidentified_training_commit",
    "foreign_data_provenance", "foreign_environment_provenance",
    "different_config_bytes", "different_schema_bytes", "different_training_contract",
])
def test_self_consistent_executable_models_still_require_actual_frozen_evidence(external_case, fault):
    from eeg_ma.fixed_model import validate_model
    root = external_case["root"] / "lock"
    model_path = root / "artifacts/primary_model.bin"
    training_path = root / "artifacts/primary_training_manifest.bin"
    model, training = json.loads(model_path.read_bytes()), json.loads(training_path.read_bytes())
    if fault == "software_validation_scope":
        model["execution_scope"] = training["execution_scope"] = "software_validation"
    elif fault == "dirty_training_code":
        model["code"]["dirty"] = True
        training["code"] = model["code"]
    elif fault == "unidentified_training_commit":
        model["code"]["commit"] = "unknown"
        training["code"] = model["code"]
    elif fault.startswith("foreign_"):
        field = "training_data_sha256" if fault == "foreign_data_provenance" else "environment_sha256"
        model["provenance"][field] = "b" * 64
    elif fault in {"different_config_bytes", "different_schema_bytes"}:
        field = "analysis_config" if fault == "different_config_bytes" else "feature_schema"
        model[field + "_text"] = json.dumps(model[field], sort_keys=True, indent=4) + "\n"
        model[field + "_sha256"] = hashlib.sha256(model[field + "_text"].encode()).hexdigest()
        model["provenance"][field + "_sha256"] = model[field + "_sha256"]
    else:
        source = json.loads(model["source_protocol_text"])
        source["future_scientific_decision"] = "changed after training"
        model["source_protocol_text"] = json.dumps(source, sort_keys=True, indent=2) + "\n"
        model["source_protocol_sha256"] = hashlib.sha256(model["source_protocol_text"].encode()).hexdigest()
        model["protocol_training_contract"] = protocol.protocol_training_contract(source)
        model["protocol_training_contract_sha256"] = protocol.protocol_training_contract_sha256(source)
        training["source_protocol_sha256"] = model["source_protocol_sha256"]
        training["protocol_training_contract_sha256"] = model["protocol_training_contract_sha256"]
    model.pop("model_id")
    model["model_id"] = fingerprint(model)
    # These remain well-formed portable models. Live authorization additionally
    # requires reference scope, clean code and exact independently frozen bytes.
    validate_model(model)
    write_json(model_path, model)
    training["model_id"], training["model_sha256"] = model["model_id"], sha256_file(model_path)
    write_json(training_path, training)
    _rebind_lock_inventory(root)
    with pytest.raises(protocol.ProtocolError, match="executable model|Software-validation|training source-code"):
        protocol.validate_lock(root)


def test_executable_model_cannot_downgrade_to_opaque_archive_by_removing_outer_markers(external_case):
    root = external_case["root"] / "lock"
    spec = json.loads((root / "protocol.json").read_bytes())
    spec["freeze"].pop("model_artifact_policy")
    write_json(root / "protocol.json", spec)
    markers = {"model_artifact_schema", "model_id", "execution_scope",
               "protocol_training_contract_sha256", "source_protocol_sha256"}
    for role in ("primary", "reference"):
        path = root / f"artifacts/{role}_training_manifest.bin"
        training = json.loads(path.read_bytes())
        write_json(path, {key: value for key, value in training.items() if key not in markers})
    _rebind_lock_inventory(root)
    with pytest.raises(protocol.ProtocolError, match="executable model"):
        protocol.validate_lock(root)


def _publication_guard(root, **overrides):
    manifest = protocol.validate_lock(root / "lock")["manifest"]
    expected = {"expected_lock_id": manifest["lock_id"],
                "expected_cohort_sha256": sha256_file(root / "cohort.json")}
    return protocol.guard_external_prediction(root / "lock", root / "cohort.json", **{**expected, **overrides})


def test_publication_guard_excludes_outcome_access_through_atomic_rename(external_case, monkeypatch):
    root = external_case["root"]
    original_rename = artifacts.os.rename
    observed = []
    def rename_while_guarded(source, target):
        with pytest.raises(ProvenanceError, match="Another process"):
            protocol.record_outcome_access(root / "lock", root / "cohort.json",
                enrollment_manifest_path=root / "enrollment.json")
        observed.append(Path(target))
        original_rename(source, target)
    monkeypatch.setattr(artifacts.os, "rename", rename_while_guarded)
    target = root / "guarded-publication"
    with _publication_guard(root), artifacts.new_output_directory(target) as stage:
        (stage / "complete").write_text("SYNTHETIC publication boundary test\n")
    assert observed == [target]
    assert (target / "complete").exists()
    assert protocol.validate_lock(root / "lock")["events"] == []
    event = protocol.record_outcome_access(root / "lock", root / "cohort.json",
        enrollment_manifest_path=root / "enrollment.json")
    assert protocol.validate_lock(root / "lock")["events"] == [event]


@pytest.mark.parametrize("field", ["expected_lock_id", "expected_cohort_sha256"])
def test_publication_guard_revalidates_exact_snapshot_identity(external_case, field):
    root = external_case["root"]
    with pytest.raises(protocol.ProtocolError, match="changed before prediction publication"):
        with _publication_guard(root, **{field: "b" * 64}):
            pytest.fail("Changed protocol/cohort evidence must fail before publication")
    assert protocol.validate_lock(root / "lock")["events"] == []


def test_aborted_publication_releases_guard_without_inventing_exposure(external_case):
    root = external_case["root"]
    target = root / "aborted-publication"
    with pytest.raises(RuntimeError, match="SYNTHETIC abort"):
        with _publication_guard(root), artifacts.new_output_directory(target):
            raise RuntimeError("SYNTHETIC abort")
    assert not target.exists()
    assert protocol.validate_lock(root / "lock")["events"] == []
    protocol.record_outcome_access(root / "lock", root / "cohort.json",
        enrollment_manifest_path=root / "enrollment.json")
    with pytest.raises(protocol.ProtocolError, match="closed prediction"):
        with _publication_guard(root):
            pytest.fail("An intervening outcome event must stop publication")
