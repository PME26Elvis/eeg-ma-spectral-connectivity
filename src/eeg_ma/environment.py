"""Reference wheel locks and observed execution-environment evidence.

The lock is a prospective reference, never a claim about historical runs.  The
archived evidence validator checks its own embedded contract without inspecting
the machine that later reads it.  Only capture/verify inspect a live runtime.
"""
from __future__ import annotations

import csv
import hashlib
import importlib.metadata
import io
import json
import os
from pathlib import Path, PurePosixPath
import platform
import re
from urllib.parse import unquote, urlsplit


SCHEMA = "eeg-environment-1"
REFERENCE_SCHEMA = "eeg-reference-environment-1"
ROOTS = {"numpy", "pandas", "scipy", "scikit-learn", "joblib", "matplotlib", "pytest", "pip", "setuptools", "wheel"}
THREADS = {"OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
           "NUMEXPR_NUM_THREADS": "1", "PYTHONHASHSEED": "0", "MPLBACKEND": "Agg"}
SHA = re.compile(r"[0-9a-f]{64}\Z")
PIN = re.compile(r"([a-z0-9]+(?:-[a-z0-9]+)*)==([A-Za-z0-9.!+_-]+) --hash=sha256:([0-9a-f]{64})\Z")
PROJECT_ROOT = Path(__file__).resolve().parents[2]


class EnvironmentError(ValueError):
    """Environment evidence is incomplete, inconsistent, or not the reference."""


def _canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _digest(value):
    return hashlib.sha256(_canonical(value).encode()).hexdigest()


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for part in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(part)
    return digest.hexdigest()


def _name(value):
    return re.sub(r"[-_.]+", "-", str(value)).lower()


def _pairs(pairs):
    output = {}
    for key, value in pairs:
        if key in output:
            raise EnvironmentError(f"Duplicate JSON field: {key}")
        output[key] = value
    return output


def _json(text):
    try:
        value = json.loads(text, object_pairs_hook=_pairs,
            parse_constant=lambda value: (_ for _ in ()).throw(EnvironmentError(f"Nonfinite JSON: {value}")))
    except (ValueError, TypeError) as exc:
        raise EnvironmentError(f"Invalid environment JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise EnvironmentError("Environment JSON must be an object")
    return value


def _parse_lock(text):
    pins = {}
    for line in text.splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        match = PIN.fullmatch(line)
        if match is None:
            raise EnvironmentError("Lock permits only exact name==version --hash=sha256 lines; options, URLs and includes are prohibited")
        name, version, digest = match.groups()
        if name in pins:
            raise EnvironmentError(f"Duplicate lock requirement: {name}")
        pins[name] = {"version": version, "sha256": digest}
    if not pins:
        raise EnvironmentError("Reference lock is empty")
    return pins


def _reference(definition_text, requirements_text):
    """Validate lock syntax before pip; this part deliberately uses stdlib only."""
    spec = _json(definition_text)
    if spec.get("schema_version") != REFERENCE_SCHEMA:
        raise EnvironmentError("Unsupported reference environment schema")
    if not isinstance(spec.get("reference_id"), str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", spec["reference_id"]):
        raise EnvironmentError("A versioned reference_id is required")
    if not isinstance(spec.get("python"), str) or not re.fullmatch(r"3\.\d+\.\d+", spec["python"]):
        raise EnvironmentError("Pin the full Python patch version")
    if spec.get("implementation") != "CPython" or spec.get("system") != "Linux" or spec.get("machine") != "x86_64":
        raise EnvironmentError("This reference contract supports CPython on Linux x86_64")
    if spec.get("thread_environment") != THREADS:
        raise EnvironmentError("Reference thread, hash-seed and plotting settings must be explicit")
    if spec.get("project") != {"name": "eeg-ma-spectral-connectivity", "version": "0.1.0"}:
        raise EnvironmentError("Reference project identity is inconsistent")
    if spec.get("libc", {}).get("name") != "glibc" or not re.fullmatch(r"\d+\.\d+", str(spec.get("libc", {}).get("minimum_version", ""))):
        raise EnvironmentError("Reference glibc compatibility floor is required")
    locator = spec.get("requirements_lock", {})
    file = locator.get("file")
    if not isinstance(file, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*\.lock", file):
        raise EnvironmentError("Reference lock must be a same-directory basename")
    actual = hashlib.sha256(requirements_text.encode()).hexdigest()
    if locator.get("sha256") != actual:
        raise EnvironmentError("Reference definition/requirements hash mismatch")
    pins = _parse_lock(requirements_text)
    artifacts = spec.get("artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        raise EnvironmentError("Reference wheel inventory is missing")
    seen = {}
    for artifact in artifacts:
        if not isinstance(artifact, dict):
            raise EnvironmentError("Invalid wheel artifact")
        name = artifact.get("name")
        if name in seen or name not in pins or name != _name(name):
            raise EnvironmentError("Reference wheel names must uniquely match the lock")
        if {k: artifact.get(k) for k in ("version", "sha256")} != pins[name]:
            raise EnvironmentError("Reference wheel and exact lock pin disagree")
        if not re.fullmatch(r"[A-Za-z0-9_.+-]+\.whl", str(artifact.get("filename", ""))):
            raise EnvironmentError("Reference artifacts must be wheel basenames")
        if not isinstance(artifact.get("installed_content_sha256"), str) or not SHA.fullmatch(artifact["installed_content_sha256"]):
            raise EnvironmentError("Wheel content inventory digest is required")
        if not isinstance(artifact.get("metadata_sha256"), str) or not SHA.fullmatch(artifact["metadata_sha256"]):
            raise EnvironmentError("Wheel metadata digest is required")
        if type(artifact.get("content_file_count")) is not int or artifact["content_file_count"] < 1:
            raise EnvironmentError("Wheel content inventory count is required")
        if not isinstance(artifact.get("requires"), list) or any(not isinstance(x, str) for x in artifact["requires"]):
            raise EnvironmentError("Resolved transitive requirements must be explicit")
        seen[name] = artifact
    roots = spec.get("roots")
    if (not isinstance(roots, list) or any(not isinstance(name, str) for name in roots)
            or set(seen) != set(pins) or set(roots) != ROOTS or len(roots) != len(ROOTS)):
        raise EnvironmentError("Reference root/dependency inventory is incomplete")
    definition_sha = hashlib.sha256(definition_text.encode()).hexdigest()
    return {"definition_text": definition_text, "requirements_text": requirements_text,
            "definition_sha256": definition_sha, "requirements_sha256": actual,
            "lock_id": _digest({"definition_sha256": definition_sha, "requirements_sha256": actual}),
            "definition": spec}


def load_reference(reference_path):
    """Strict stdlib-only lock preflight, suitable before installing packages."""
    path = Path(reference_path)
    if path.is_symlink():
        raise EnvironmentError("Reference definition cannot be a symlink")
    try:
        # Decode without universal-newline conversion: the lock identity binds
        # the exact committed bytes, including line endings.
        text = path.read_bytes().decode("utf-8")
        spec = _json(text)
        file = spec.get("requirements_lock", {}).get("file")
        if not isinstance(file, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*\.lock", file):
            raise EnvironmentError("Reference lock must be a same-directory basename")
        lock = path.parent / file
        if lock.is_symlink():
            raise EnvironmentError("Reference requirements cannot be a symlink")
        return _reference(text, lock.read_bytes().decode("utf-8"))
    except (OSError, KeyError, TypeError, AttributeError) as exc:
        raise EnvironmentError(f"Missing reference evidence: {exc}") from exc


def _dependency_closure(spec):
    from packaging.requirements import Requirement
    from packaging.version import Version
    artifacts = {item["name"]: item for item in spec["artifacts"]}
    visited, todo = set(), list(spec["roots"])
    while todo:
        name = todo.pop()
        if name in visited:
            continue
        if name not in artifacts:
            raise EnvironmentError(f"Missing transitive dependency: {name}")
        visited.add(name)
        for text in artifacts[name]["requires"]:
            req = Requirement(text)
            child = _name(req.name)
            if req.url or req.marker is not None or req.extras or child not in artifacts:
                raise EnvironmentError(f"Unresolved dependency: {name}: {text}")
            if Version(artifacts[child]["version"]) not in req.specifier:
                raise EnvironmentError(f"Incompatible transitive dependency: {name}: {text}")
            todo.append(child)
    if visited != set(artifacts):
        raise EnvironmentError("Lock contains packages outside the declared resolved closure")


def _receipt(path):
    if path is None:
        return {}
    try:
        receipt = _json(Path(path).read_text(encoding="utf-8"))
    except OSError as exc:
        raise EnvironmentError("Configured installation receipt is missing") from exc
    if receipt.get("version") != "1" or not isinstance(receipt.get("install"), list):
        raise EnvironmentError("Unsupported pip installation report")
    result = {}
    for item in receipt["install"]:
        metadata, download = item.get("metadata", {}), item.get("download_info", {})
        name = _name(metadata.get("name", ""))
        filename = Path(unquote(urlsplit(download.get("url", "")).path)).name
        digest = download.get("archive_info", {}).get("hashes", {}).get("sha256")
        if not name or name in result or not filename.endswith(".whl") or not isinstance(digest, str) or not SHA.fullmatch(digest):
            raise EnvironmentError("Install receipt must uniquely identify hashed wheel artifacts")
        result[name] = {"version": metadata.get("version"), "filename": filename, "sha256": digest}
    return dict(sorted(result.items()))


def _installed_content(distribution):
    """Compare actual installed bytes using an inventory independently pinned in the lock."""
    record = distribution.read_text("RECORD")
    if not record:
        return {"status": "unavailable"}
    metadata_dir = str(next((item for item in distribution.files or []
                             if str(item).count("/") == 1 and str(item).endswith(".dist-info/METADATA")), "")).rsplit("/", 1)[0]
    generated = {metadata_dir + "/" + name for name in ("RECORD", "INSTALLER", "REQUESTED", "direct_url.json")}
    inventory = {}
    for name, encoded, _ in csv.reader(io.StringIO(record)):
        path = PurePosixPath(name)
        if (not encoded or path.is_absolute() or ".." in path.parts or name in generated
                or name.endswith(".pyc") or ".data/" in name):
            continue
        if name in inventory:
            return {"status": "invalid_record"}
        actual = Path(distribution.locate_file(name))
        if not actual.is_file() or actual.is_symlink():
            return {"status": "missing_or_symlinked_content"}
        inventory[name] = {"sha256": _sha(actual), "bytes": actual.stat().st_size}
    return {"status": "verified_bytes", "sha256": _digest(inventory), "file_count": len(inventory)}


def _native_libraries():
    # Load the same numerical libraries before every snapshot, regardless of the
    # caller's import/fitting history. No training data is accessed here.
    import numpy as np
    import scipy.linalg.blas
    from sklearn.utils import _openmp_helpers  # noqa: F401
    from threadpoolctl import threadpool_info
    np.dot(np.ones((2, 2)), np.ones((2, 2)))
    scipy.linalg.blas.dgemm(1.0, np.ones((2, 2)), np.ones((2, 2)))
    result = []
    for row in threadpool_info():
        path = Path(row["filepath"])
        result.append({**{key: row.get(key) for key in ("user_api", "internal_api", "prefix", "version", "num_threads", "threading_layer", "architecture")},
                       "file_name": path.name, "sha256": _sha(path)})
    return sorted(result, key=_canonical)


def _reference_findings(identity, reference, *, check_content=True):
    spec = reference["definition"]
    _dependency_closure(spec)
    findings = []
    for field in ("python", "implementation", "system", "machine"):
        if identity.get(field) != spec[field]:
            findings.append(f"{field} does not match reference")
    actual_libc = identity.get("libc", {})
    try:
        if actual_libc.get("name") != spec["libc"]["name"] or tuple(map(int, actual_libc.get("version", "").split("."))) < tuple(map(int, spec["libc"]["minimum_version"].split("."))):
            findings.append("libc does not satisfy reference")
    except ValueError:
        findings.append("libc identity is invalid")
    if identity.get("thread_environment") != spec["thread_environment"]:
        findings.append("thread/hash-seed/plotting configuration does not match reference")
    wanted = {item["name"]: item for item in spec["artifacts"]}
    packages = identity.get("packages", {})
    allowed = set(wanted) | {spec["project"]["name"]}
    if set(packages) - allowed:
        findings.append("unexpected installed third-party packages")
    if identity.get("duplicate_distributions"):
        findings.append("duplicate installed distributions")
    for name, artifact in wanted.items():
        if packages.get(name) != artifact["version"]:
            findings.append(f"installed version mismatch: {name}")
        if identity.get("package_metadata_sha256", {}).get(name) != artifact["metadata_sha256"]:
            findings.append(f"installed metadata mismatch: {name}")
    if spec["project"]["name"] in packages and packages[spec["project"]["name"]] != spec["project"]["version"]:
        findings.append("installed project version mismatch")
    receipts = identity.get("install_artifacts", {})
    if set(receipts) != set(wanted):
        findings.append("installation receipt must cover every and only locked wheel")
    for name, artifact in wanted.items():
        if receipts.get(name) != {k: artifact[k] for k in ("version", "filename", "sha256")}:
            findings.append(f"wheel receipt mismatch: {name}")
        if check_content:
            observed = identity.get("installed_content", {}).get(name, {})
            if observed != {"status": "verified_bytes", "sha256": artifact["installed_content_sha256"], "file_count": artifact["content_file_count"]}:
                findings.append(f"installed wheel content mismatch: {name}")
    native = identity.get("native_libraries", [])
    if not isinstance(native, list) or not native or not any(item.get("user_api") == "blas" for item in native):
        findings.append("observed BLAS identity is missing")
    elif any(type(item.get("num_threads")) is not int or item["num_threads"] != 1 or
             not isinstance(item.get("sha256"), str) or not SHA.fullmatch(item["sha256"]) for item in native):
        findings.append("native library threads/hash do not satisfy reference")
    return findings


def capture_environment(project_root=None, *, reference_path=None, install_report=None):
    """Capture observed packages/runtime, retaining a truthful reference mismatch."""
    project = Path(project_root or PROJECT_ROOT)
    path = Path(reference_path or os.environ.get("EEG_MA_REFERENCE_SPEC", project / "environment/reference.json"))
    explicit_reference = reference_path is not None or "EEG_MA_REFERENCE_SPEC" in os.environ
    reference = load_reference(path) if explicit_reference or path.exists() else None
    distributions, packages, duplicates = {}, {}, []
    for dist in importlib.metadata.distributions():
        name = _name(dist.metadata.get("Name", ""))
        if not name:
            continue
        if name in packages:
            duplicates.append(name)
            continue
        packages[name] = dist.version
        distributions[name] = dist
    libc_name, libc_version = platform.libc_ver()
    identity = {"schema_version": SCHEMA, "python": platform.python_version(), "implementation": platform.python_implementation(),
        "platform": platform.platform(), "system": platform.system(), "machine": platform.machine(),
        "libc": {"name": libc_name, "version": libc_version}, "packages": dict(sorted(packages.items())),
        "duplicate_distributions": sorted(set(duplicates)), "thread_environment": {name: os.environ.get(name) for name in THREADS},
        "native_libraries": _native_libraries(), "install_artifacts": _receipt(install_report or os.environ.get("EEG_MA_INSTALL_REPORT")),
        "package_metadata_sha256": {}, "installed_content": {}, "lock_files": {}, "lock_id": "unknown", "reference": None}
    if reference:
        spec = reference["definition"]
        identity["reference"] = {key: reference[key] for key in ("definition_text", "requirements_text", "definition_sha256", "requirements_sha256")}
        identity["lock_id"] = reference["lock_id"]
        identity["lock_files"] = {"reference.json": reference["definition_sha256"], spec["requirements_lock"]["file"]: reference["requirements_sha256"]}
        for artifact in spec["artifacts"]:
            name = artifact["name"]
            if name in distributions:
                metadata = distributions[name].read_text("METADATA") or ""
                identity["package_metadata_sha256"][name] = hashlib.sha256(metadata.encode()).hexdigest()
        # Nonreference compatibility processes often contain unrelated tools.
        # Skip the expensive content walk when a definitive mismatch exists.
        if not _reference_findings(identity, reference, check_content=False):
            identity["installed_content"] = {name: _installed_content(distributions[name]) for name in sorted(artifact["name"] for artifact in spec["artifacts"])}
        findings = _reference_findings(identity, reference)
        identity["reference_validation"] = {"status": "matched" if not findings else "mismatched", "findings": findings}
    else:
        identity["reference_validation"] = {"status": "unavailable", "findings": ["No reference definition exists for this project"]}
    identity["environment_id"] = _digest(identity)
    return identity


def validate_environment_evidence(identity, *, require_reference=False):
    """Validate archived evidence without comparing it with the current machine."""
    try:
        if not isinstance(identity, dict) or identity.get("schema_version") != SCHEMA:
            raise EnvironmentError("Unsupported environment evidence schema")
        payload = dict(identity)
        environment_id = payload.pop("environment_id", None)
        if _digest(payload) != environment_id:
            raise EnvironmentError("Environment evidence digest mismatch")
        if not isinstance(identity.get("packages"), dict) or any(not isinstance(k, str) or k != _name(k) or not isinstance(v, str) or not v for k, v in identity["packages"].items()):
            raise EnvironmentError("Observed package identities must be explicit")
        data = identity.get("reference")
        if data is None:
            if identity.get("lock_id") != "unknown" or identity.get("reference_validation", {}).get("status") != "unavailable":
                raise EnvironmentError("Unspecified reference cannot claim a lock")
            if require_reference:
                raise EnvironmentError("A verified reference environment is required")
            return {"environment_id": environment_id, "lock_id": "unknown", "reference_status": "unavailable"}
        reference = _reference(data["definition_text"], data["requirements_text"])
        for key in ("definition_sha256", "requirements_sha256"):
            if data.get(key) != reference[key]:
                raise EnvironmentError("Embedded reference digest mismatch")
        spec = reference["definition"]
        if identity.get("lock_id") != reference["lock_id"] or identity.get("lock_files") != {"reference.json": reference["definition_sha256"], spec["requirements_lock"]["file"]: reference["requirements_sha256"]}:
            raise EnvironmentError("Environment lock identity is inconsistent")
        findings = _reference_findings(identity, reference)
        expected = {"status": "matched" if not findings else "mismatched", "findings": findings}
        if identity.get("reference_validation") != expected:
            raise EnvironmentError("Reference match declaration contradicts observed evidence")
        if require_reference and findings:
            raise EnvironmentError("Reference environment mismatch: " + "; ".join(findings))
        return {"environment_id": environment_id, "lock_id": reference["lock_id"], "reference_id": spec["reference_id"],
                "reference_status": expected["status"], "python": identity["python"]}
    except EnvironmentError:
        raise
    except (KeyError, TypeError, ValueError, AttributeError) as exc:
        raise EnvironmentError(f"Invalid environment evidence: {exc}") from exc


def verify_reference_environment(identity=None, project_root=None, *, reference_path=None, install_report=None):
    """Capture and require the actual current runtime to match the reference."""
    current = capture_environment(project_root, reference_path=reference_path, install_report=install_report)
    if identity is not None and identity.get("environment_id") != current["environment_id"]:
        raise EnvironmentError("Supplied evidence does not describe the current runtime")
    validate_environment_evidence(current, require_reference=True)
    return current
