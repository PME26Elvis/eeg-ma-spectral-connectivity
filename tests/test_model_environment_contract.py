"""Reference environment is part of model provenance, not a filename label."""
from __future__ import annotations

import json

import pytest

from eeg_ma import protocol
from eeg_ma.provenance import sha256_file
from test_protocol_gate import LOCK_TIME, complete_spec, locked, synthetic_code, write_json


@pytest.mark.parametrize("role", ["primary", "reference"])
@pytest.mark.parametrize("fault", ["missing", "different"])
@pytest.mark.parametrize("field", ["environment_id", "environment_lock_id"])
def test_model_manifest_requires_the_actual_environment_identity(tmp_path, monkeypatch, role, fault, field):
    source, spec = complete_spec(tmp_path / "inputs")
    entry = spec["freeze"]["artifacts"][role + "_training_manifest"]
    path = source.parent / entry["path"]
    training = json.loads(path.read_text())
    if fault == "missing":
        training.pop(field)
    else:
        training[field] = "f" * 64
    write_json(path, training)
    entry["sha256"] = sha256_file(path)
    write_json(source, spec)
    monkeypatch.setattr(protocol, "code_identity", synthetic_code)
    monkeypatch.setattr(protocol, "_now", lambda: LOCK_TIME)
    with pytest.raises(protocol.ProtocolError, match="training provenance"):
        protocol.lock_protocol(source, tmp_path / "rejected")
    assert not (tmp_path / "rejected").exists()


def test_legacy_or_version_only_environment_cannot_become_live_model_evidence(tmp_path, monkeypatch):
    source, spec = complete_spec(tmp_path / "inputs")
    frozen = spec["freeze"]["artifacts"]
    environment = source.parent / frozen["environment"]["path"]
    write_json(environment, {"python": "3.12.14", "packages": {"numpy": "2.3.5"},
                             "lock_id": "unknown", "status": "legacy_unverified"})
    frozen["environment"]["sha256"] = sha256_file(environment)
    for role in ("primary", "reference"):
        path = source.parent / frozen[role + "_training_manifest"]["path"]
        training = json.loads(path.read_text())
        training.update(environment_sha256=sha256_file(environment), environment_lock_id="unknown")
        write_json(path, training)
        frozen[role + "_training_manifest"]["sha256"] = sha256_file(path)
    write_json(source, spec)
    monkeypatch.setattr(protocol, "code_identity", synthetic_code)
    monkeypatch.setattr(protocol, "_now", lambda: LOCK_TIME)
    with pytest.raises(protocol.ProtocolError, match="environment"):
        protocol.lock_protocol(source, tmp_path / "rejected")
    assert not (tmp_path / "rejected").exists()


@pytest.mark.parametrize("value", [None, "compatibility_only"])
def test_reference_environment_requirement_cannot_be_removed(tmp_path, value):
    _, spec = complete_spec(tmp_path / "inputs")
    if value is None:
        del spec["freeze"]["environment_policy"]
    else:
        spec["freeze"]["environment_policy"] = value
    assert any(f.path == "freeze.environment_policy" for f in protocol.validate_protocol(spec).findings)


def test_archived_protocol_environment_does_not_depend_on_verifiers_current_runtime(locked, monkeypatch):
    from eeg_ma import environment
    def forbidden(*args, **kwargs):
        raise AssertionError("An archived lock must verify its evidence, not recapture the current host")
    monkeypatch.setattr(environment, "capture_environment", forbidden)
    out = locked[0]
    assert protocol.validate_lock(out)["protocol"]["freeze"]["environment_policy"] == "verified_reference_environment"
