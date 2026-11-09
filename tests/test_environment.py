"""Reference-environment contract tests; synthetic evidence is not study evidence."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import pytest

import eeg_ma.environment as env


ROOT = Path(__file__).resolve().parents[1]


def _rehash(value):
    value.pop("environment_id", None)
    value["environment_id"] = env._digest(value)
    return value


def synthetic_reference_environment():
    """Self-contained synthetic proof for cross-version protocol contract tests.

    No current runtime is inspected and no real training environment is claimed.
    The actual checked-in reference definition is embedded, while observed
    runtime fields are explicitly synthetic test attestations.
    """
    reference = env.load_reference(ROOT / "environment/reference.json")
    spec = reference["definition"]
    value = {"schema_version": env.SCHEMA, "python": spec["python"], "implementation": spec["implementation"],
        "platform": "SYNTHETIC environment fixture; never used as live study evidence", "system": "Linux", "machine": "x86_64",
        "libc": {"name": "glibc", "version": "2.39"}, "packages": {item["name"]: item["version"] for item in spec["artifacts"]},
        "duplicate_distributions": [], "thread_environment": deepcopy(spec["thread_environment"]),
        "native_libraries": [{"user_api": "blas", "internal_api": "openblas", "prefix": "SYNTHETIC", "version": "SYNTHETIC",
            "num_threads": 1, "threading_layer": "pthreads", "architecture": "SYNTHETIC", "file_name": "synthetic.so", "sha256": "a" * 64}],
        "install_artifacts": {item["name"]: {k: item[k] for k in ("version", "filename", "sha256")} for item in spec["artifacts"]},
        "package_metadata_sha256": {item["name"]: item["metadata_sha256"] for item in spec["artifacts"]},
        "installed_content": {item["name"]: {"status": "verified_bytes", "sha256": item["installed_content_sha256"], "file_count": item["content_file_count"]} for item in spec["artifacts"]},
        "lock_files": {"reference.json": reference["definition_sha256"], spec["requirements_lock"]["file"]: reference["requirements_sha256"]},
        "lock_id": reference["lock_id"], "reference": {key: reference[key] for key in ("definition_text", "requirements_text", "definition_sha256", "requirements_sha256")},
        "reference_validation": {"status": "matched", "findings": []}}
    return _rehash(value)


def test_archived_reference_does_not_depend_on_the_current_runtime(monkeypatch):
    value = synthetic_reference_environment()
    monkeypatch.setattr(env.platform, "python_version", lambda: "0.0.0")
    assert env.validate_environment_evidence(value, require_reference=True)["reference_status"] == "matched"


@pytest.mark.parametrize("fault", ["package", "missing_transitive", "receipt_hash", "receipt_missing", "content", "metadata", "thread", "blas_thread", "python", "extra_package", "lock_id", "declaration"])
def test_rehashed_false_reference_claims_fail_closed(fault):
    value = synthetic_reference_environment()
    if fault == "package":
        value["packages"]["numpy"] = "2.0.0"
    elif fault == "missing_transitive":
        del value["packages"]["threadpoolctl"]
    elif fault == "receipt_hash":
        value["install_artifacts"]["numpy"]["sha256"] = "b" * 64
    elif fault == "receipt_missing":
        del value["install_artifacts"]["numpy"]
    elif fault == "content":
        value["installed_content"]["numpy"]["sha256"] = "b" * 64
    elif fault == "metadata":
        value["package_metadata_sha256"]["numpy"] = "b" * 64
    elif fault == "thread":
        value["thread_environment"]["OPENBLAS_NUM_THREADS"] = "2"
    elif fault == "blas_thread":
        value["native_libraries"][0]["num_threads"] = 2
    elif fault == "python":
        value["python"] = "3.12.13"
    elif fault == "extra_package":
        value["packages"]["untracked-package"] = "1.0"
    elif fault == "lock_id":
        value["lock_id"] = "b" * 64
    else:
        value["reference_validation"]["status"] = "unavailable"
    with pytest.raises(env.EnvironmentError):
        env.validate_environment_evidence(_rehash(value), require_reference=True)


@pytest.mark.parametrize("bad", ["-r arbitrary.txt", "--index-url https://example.invalid", "numpy>=1.0 --hash=sha256:" + "a" * 64,
    "numpy==2.3.5", "numpy==2.3.5 --hash=sha256:" + "a" * 64 + " ; python_version > '3'", "numpy @ https://example.invalid/wheel.whl"])
def test_lock_preflight_rejects_pip_directives_and_unpinned_inputs(tmp_path, bad):
    spec = json.loads((ROOT / "environment/reference.json").read_text())
    lock = tmp_path / spec["requirements_lock"]["file"]
    lock.write_text(bad + "\n")
    spec["requirements_lock"]["sha256"] = env._sha(lock)
    path = tmp_path / "reference.json"
    path.write_text(json.dumps(spec))
    with pytest.raises(env.EnvironmentError):
        env.load_reference(path)


def test_reference_hashes_preserve_actual_committed_crlf_bytes(tmp_path):
    spec = json.loads((ROOT / "environment/reference.json").read_text())
    filename = spec["requirements_lock"]["file"]
    lock_bytes = (ROOT / "environment" / filename).read_bytes().replace(b"\n", b"\r\n")
    spec["requirements_lock"]["sha256"] = hashlib.sha256(lock_bytes).hexdigest()
    definition_bytes = (json.dumps(spec, indent=2) + "\n").replace("\n", "\r\n").encode()
    path = tmp_path / "reference.json"
    path.write_bytes(definition_bytes)
    (tmp_path / filename).write_bytes(lock_bytes)
    profile = env.load_reference(path)
    assert profile["definition_text"].encode() == definition_bytes
    assert profile["requirements_text"].encode() == lock_bytes
    assert profile["definition_sha256"] == hashlib.sha256(definition_bytes).hexdigest()
    assert profile["requirements_sha256"] == hashlib.sha256(lock_bytes).hexdigest()


def test_reference_cannot_bless_normalized_instead_of_committed_lock_bytes(tmp_path):
    spec = json.loads((ROOT / "environment/reference.json").read_text())
    filename = spec["requirements_lock"]["file"]
    # The definition still declares the original LF lock digest.
    (tmp_path / filename).write_bytes((ROOT / "environment" / filename).read_bytes().replace(b"\n", b"\r\n"))
    path = tmp_path / "reference.json"
    path.write_text(json.dumps(spec))
    with pytest.raises(env.EnvironmentError, match="hash mismatch"):
        env.load_reference(path)


def test_same_version_with_changed_installed_content_is_detected(tmp_path):
    import importlib.metadata
    base = tmp_path / "example-1.0.dist-info"
    base.mkdir()
    (base / "METADATA").write_text("Metadata-Version: 2.1\nName: example\nVersion: 1.0\n")
    (tmp_path / "example.py").write_text("VALUE = 1\n")
    (base / "RECORD").write_text("example.py,sha256=ignored,10\nexample-1.0.dist-info/METADATA,sha256=ignored,51\nexample-1.0.dist-info/RECORD,,\n")
    distribution = importlib.metadata.Distribution.at(base)
    first = env._installed_content(distribution)
    (tmp_path / "example.py").write_text("VALUE = 2\n")
    assert distribution.version == "1.0"
    assert env._installed_content(distribution)["sha256"] != first["sha256"]


def test_receipt_normalization_removes_machine_paths_and_credentials(tmp_path):
    path = tmp_path / "receipt.json"
    path.write_text(json.dumps({"version": "1", "install": [{"metadata": {"name": "Example", "version": "1.0"},
        "download_info": {"url": "https://secret:password@example.invalid/private/example-1.0-py3-none-any.whl",
                          "archive_info": {"hashes": {"sha256": "a" * 64}}}}]}))
    result = env._receipt(path)
    assert result == {"example": {"version": "1.0", "filename": "example-1.0-py3-none-any.whl", "sha256": "a" * 64}}


def test_reference_files_are_part_of_run_and_code_identity(tmp_path):
    from eeg_ma.provenance import code_identity
    (tmp_path / "environment").mkdir()
    path = tmp_path / "environment/reference.lock"
    path.write_text("first")
    first = code_identity(tmp_path)
    path.write_text("second")
    second = code_identity(tmp_path)
    assert first["working_source_sha256"] != second["working_source_sha256"]


def test_capture_is_stable_before_and_after_numerical_fitting():
    import numpy as np
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    before = env.capture_environment(ROOT)
    values = np.array([[0., 1.], [.2, 1.2], [.4, .8], [2., 3.], [2.2, 3.2], [2.4, 2.8]])
    LinearDiscriminantAnalysis().fit(values, [0, 0, 0, 1, 1, 1]).predict(values)
    after = env.capture_environment(ROOT)
    assert before["environment_id"] == after["environment_id"]
    assert before["native_libraries"] == after["native_libraries"]


def test_lock_presence_alone_cannot_make_a_runtime_reference(monkeypatch):
    monkeypatch.delenv("EEG_MA_INSTALL_REPORT", raising=False)
    current = env.capture_environment(ROOT)
    assert current["lock_id"] != "unknown"
    assert current["reference_validation"]["status"] == "mismatched"
    with pytest.raises(env.EnvironmentError, match="mismatch"):
        env.validate_environment_evidence(current, require_reference=True)


@pytest.mark.parametrize("fault", ["contradictory_reference", "unknown_schema", "missing_schema"])
def test_rehashed_complete_run_cannot_bless_invalid_environment(tmp_path, fault):
    import pandas as pd
    from eeg_ma.provenance import (ProvenanceError, RunTransaction, fingerprint,
                                  table_contract, validate_complete_run)
    table = pd.DataFrame({"metric": ["synthetic-check"], "value": [1]})
    identity = {"kind": "report", "environment": synthetic_reference_environment(),
                "table_contracts": {"summary.csv": table_contract(table)}}
    destination = tmp_path / "complete"
    RunTransaction(destination, identity).execute(lambda stage: table.to_csv(stage / "summary.csv", index=False))
    path = destination / "run_manifest.json"
    manifest = json.loads(path.read_text())
    evidence = manifest["identity"]["environment"]
    if fault == "contradictory_reference":
        evidence["packages"]["numpy"] = "2.0.0"
    elif fault == "missing_schema":
        evidence.pop("schema_version")
        evidence["packages"]["numpy"] = "2.0.0"
    else:
        evidence["schema_version"] = "unsupported-environment-999"
    _rehash(evidence)
    manifest["run_id"] = fingerprint(manifest["identity"])
    path.write_text(json.dumps(manifest))
    with pytest.raises(ProvenanceError, match="Invalid run environment evidence"):
        validate_complete_run(destination)


def test_archived_legacy_environment_stub_retains_its_original_contract(tmp_path):
    import pandas as pd
    from eeg_ma.provenance import RunTransaction, table_contract, validate_complete_run
    table = pd.DataFrame({"value": [1]})
    identity = {"kind": "report", "environment": {"historical_environment": "unknown"},
                "table_contracts": {"summary.csv": table_contract(table)}}
    destination = tmp_path / "legacy"
    RunTransaction(destination, identity).execute(lambda stage: table.to_csv(stage / "summary.csv", index=False))
    original = (destination / "run_manifest.json").read_bytes()
    assert validate_complete_run(destination)["identity"]["environment"] == {"historical_environment": "unknown"}
    assert (destination / "run_manifest.json").read_bytes() == original
