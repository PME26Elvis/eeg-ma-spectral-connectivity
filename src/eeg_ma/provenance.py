"""Immutable analysis runs and atomic, inventory-based reports.

A manifest records evidence available now; a legacy inventory never invents the
historical execution environment.  A complete run is a directory transaction,
not the existence of one CSV.  All identities are captured once per invocation.
"""
from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import csv
from io import StringIO
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
from typing import Callable, Mapping

import numpy as np
import pandas as pd

MANIFEST_VERSION = "eeg-run-1"
MANIFEST_NAME = "run_manifest.json"
CLASSIFIERS = ("KFDA", "LDA", "RBF-SVM")
PROJECT_ROOT = Path(__file__).resolve().parents[2]


class ProvenanceError(ValueError):
    """An output cannot be safely reused or published."""


def _jsonable(value):
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        raise ProvenanceError("Identity/manifest contains a non-finite number")
    return value


def canonical_json(value) -> str:
    return json.dumps(_jsonable(value), ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def fingerprint(value) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _read_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ProvenanceError(f"Unreadable manifest: {path}") from exc


def _write_json(path: Path, value) -> None:
    path.write_text(json.dumps(_jsonable(value), ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _atomic_json(path: Path, value) -> None:
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}-", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(canonical_json(value) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def safe_name(name: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", str(name)):
        raise ProvenanceError(f"Unsafe experiment/run name: {name!r}")
    return str(name)


def _safe_relative(value: str) -> Path:
    path = Path(value)
    if path.is_absolute() or not path.parts or any(p in ("..", ".") for p in path.parts) or "\\" in value:
        raise ProvenanceError(f"Unsafe artifact path: {value!r}")
    return path


def protect_output(path: str | Path, project_root: str | Path | None = None) -> Path:
    """Protect canonical trees even through aliases, symlinks, or ancestors."""
    project = Path(project_root or PROJECT_ROOT).resolve()
    target = Path(path).absolute()
    resolved = target.resolve()
    protected = [project / "results", project / "data" / "features", project / "data" / "raw"]
    for keep in protected:
        keep = keep.resolve()
        if resolved == keep or keep in resolved.parents or resolved in keep.parents:
            raise ProvenanceError(f"Protected canonical/source tree: {target}; use a new outputs/ directory")
    for part in [target, *target.parents]:
        if part.is_symlink():
            raise ProvenanceError(f"Output path contains a symlink: {part}")
    return target


def code_identity(project_root: str | Path | None = None) -> dict:
    project = Path(project_root or PROJECT_ROOT).resolve()
    def git(*args):
        result = subprocess.run(["git", "-C", str(project), *args], capture_output=True, text=True)
        return result.stdout.strip() if result.returncode == 0 else None
    commit = git("rev-parse", "HEAD")
    source_paths = sorted({*project.glob("src/**/*.py"), *project.glob("scripts/**/*.py"), *project.glob("configs/*.json"),
                          *project.glob("environment/*.json"), *project.glob("environment/*.lock"), *project.glob("pyproject.toml")})
    files = {str(p.relative_to(project)): sha256_file(p) for p in source_paths if p.is_file()}
    status = git("status", "--porcelain=v1", "--untracked-files=all", "--", "src", "scripts", "configs", "environment", "pyproject.toml")
    return {"commit": commit or "unknown", "dirty": bool(status) if status is not None else "unknown",
            "dirty_fingerprint": fingerprint({"status": status, "working_source_files": files}),
            "working_source_sha256": fingerprint(files), "source_files": files}


def environment_identity(project_root: str | Path | None = None) -> dict:
    from .environment import capture_environment
    return capture_environment(project_root)


def input_inventory(paths: Mapping[str, str | Path]) -> dict:
    inventory = {}
    for label, value in sorted(paths.items()):
        path = Path(value)
        if not path.is_file():
            raise ProvenanceError(f"Missing input {label}: {path}")
        inventory[label] = {"sha256": sha256_file(path), "bytes": path.stat().st_size}
    return inventory


def build_run_identity(features: pd.DataFrame, cfg: dict, scopes, *, input_paths=None,
                       references=None, experiment=None, context_config=None,
                       code=None, environment=None, project_root=None, scope_layout="nested") -> dict:
    from .schema import validate_feature_table
    validate_feature_table(features)
    from .extract import feature_columns
    from .evaluation import make_evaluation_splits
    if "epoch_id" not in features or features["epoch_id"].isna().any() or features["epoch_id"].astype(str).duplicated().any():
        raise ProvenanceError("Trusted runs require unique v2 epoch_id values; use an explicit legacy mapping")
    if "schema_version" not in features:
        raise ProvenanceError("Trusted runs require a versioned feature schema")
    if features.columns.duplicated().any():
        raise ProvenanceError("Duplicate feature-table columns")
    columns = feature_columns(features, str(cfg.get("feature_set", "all")))
    if not columns or not np.isfinite(features[columns].to_numpy(dtype=float)).all():
        raise ProvenanceError("Selected features must be nonempty and finite")
    meta_columns = [c for c in features if not c.startswith(("BP__", "COH__", "ASYM_", "PLV__", "IPLV__"))]
    records = json.loads(features[meta_columns].to_json(orient="records", double_precision=15))
    # Include values even when a caller builds a table in memory rather than reading CSV.
    table_hash = hashlib.sha256(features.to_csv(index=False, lineterminator="\n").encode("utf-8")).hexdigest()
    splits = {scope: make_evaluation_splits(features, cfg, scope) for scope in scopes}
    return _jsonable({"identity_version": "evaluation-1", "kind": "evaluation",
        "code": code if code is not None else code_identity(project_root),
        "environment": environment if environment is not None else environment_identity(project_root),
        "config": cfg, "context_config": context_config or {}, "experiment": experiment or {},
        "inputs": input_inventory(input_paths or {}), "references": references or {},
        "dataset": {"columns": list(features.columns), "dtypes": [str(x) for x in features.dtypes],
                    "n_rows": len(features), "table_sha256": table_hash, "metadata": records,
                    "metadata_sha256": fingerprint(records)},
        "feature_order": columns, "scopes": list(scopes), "scope_layout": scope_layout, "splits": splits,
        "split_sha256": fingerprint(splits)})


STRING_COLUMNS = {
    "subject_id", "comparison", "class_name", "stage", "schema_version", "cohort_id",
    "session_id", "recording_id", "block_id", "epoch_id", "source_sha256", "units",
    "reference", "identity_origin", "protocol_id", "protocol_version", "channel_order",
    "event_key", "qc_status", "qc_reason_codes", "identity_status", "held_out_subject",
    "classifier", "selected_features", "best_params", "candidate_name", "analysis", "feature",
    "experiment", "feature_set", "role", "scope", "unit_id", "family", "band",
}


def read_csv_literal(path_or_buffer, *, encoding="utf-8-sig") -> pd.DataFrame:
    if hasattr(path_or_buffer, "getvalue"):
        data = path_or_buffer.getvalue()
    else:
        data = Path(path_or_buffer).read_text(encoding="utf-8-sig")
    header = next(csv.reader(StringIO(data)), [])
    if not header or len(set(header)) != len(header):
        raise ProvenanceError("CSV header is missing or contains duplicate columns")
    strings = set(header) & STRING_COLUMNS
    return pd.read_csv(StringIO(data), keep_default_na=False,
        dtype={c: str for c in strings}, na_values={c: [""] for c in header if c not in strings})


def _csv(path: Path, required=(), keys=()) -> pd.DataFrame:
    try:
        frame = read_csv_literal(path)
    except (OSError, ValueError, pd.errors.ParserError) as exc:
        raise ProvenanceError(f"Invalid or missing CSV: {path}") from exc
    missing = set(required) - set(frame.columns)
    if missing or frame.columns.duplicated().any():
        raise ProvenanceError(f"CSV schema mismatch {path}: missing={sorted(missing)}")
    if keys and (frame[list(keys)].isna().any().any() or
                 frame[list(keys)].astype(str).apply(lambda c: c.str.strip().eq("")).any().any() or
                 frame.duplicated(list(keys)).any()):
        raise ProvenanceError(f"CSV missing/duplicate keys: {path}")
    return frame


def _same_table(actual: pd.DataFrame, expected: pd.DataFrame, keys, label: str) -> None:
    if len(actual) != len(expected) or set(expected.columns) - set(actual.columns):
        raise ProvenanceError(f"Wrong row count/schema for {label}: {len(actual)} != {len(expected)}")
    a = actual.sort_values(list(keys)).reset_index(drop=True)
    e = expected.sort_values(list(keys)).reset_index(drop=True)
    for column in e:
        if pd.api.types.is_numeric_dtype(e[column]):
            if not np.allclose(pd.to_numeric(a[column], errors="coerce"), e[column], rtol=1e-10, atol=1e-12, equal_nan=True):
                raise ProvenanceError(f"Incorrect derived values for {label}.{column}")
        elif a[column].astype(str).tolist() != e[column].astype(str).tolist():
            raise ProvenanceError(f"Incorrect keys/values for {label}.{column}")


def validate_evaluation_outputs(root: Path, identity: dict) -> None:
    """Check complete fold/epoch coverage and recompute metrics, not just hashes."""
    allowed = identity["feature_order"]
    epoch_rows = {str(row["epoch_id"]): row for row in identity["dataset"]["metadata"]}
    for scope in identity["scopes"]:
        folder = root if identity.get("scope_layout") == "flat" else root / scope
        unit = "subject_id" if scope == "intra" else "held_out_subject"
        fold_keys = [unit, "comparison", "outer_fold"]
        folds = _csv(folder / f"{scope}_folds.csv", fold_keys + ["classifier", "accuracy", "n_selected_features", "selected_features", "best_params", "sfs_inner_accuracy", "grid_inner_accuracy"], fold_keys + ["classifier"])
        preds = _csv(folder / f"{scope}_predictions.csv", fold_keys + ["classifier", "epoch_id", "true_label", "predicted_label"], ["epoch_id", "classifier"])
        sfs = _csv(folder / f"{scope}_sfs_path.csv", fold_keys + ["analysis", "step", "candidate_index", "candidate_name", "cv_accuracy", "improvement", "accepted"], fold_keys + ["step"])
        expected_keys = []
        expected_preds = []
        for split in identity["splits"][scope]:
            key = {name: split[name] for name in fold_keys}
            for classifier in CLASSIFIERS:
                expected_keys.append({**key, "classifier": classifier})
                for epoch in split["test_epoch_ids"]:
                    expected_preds.append({**epoch_rows[str(epoch)], **key, "classifier": classifier, "epoch_id": epoch, "true_label": epoch_rows[str(epoch)]["class_label"]})
        _same_table(folds, pd.DataFrame(expected_keys), fold_keys + ["classifier"], f"{scope} fold coverage")
        _same_table(preds, pd.DataFrame(expected_preds), ["epoch_id", "classifier"], f"{scope} prediction coverage")
        if not preds["predicted_label"].isin([0, 1]).all() or not preds["true_label"].isin([0, 1]).all():
            raise ProvenanceError("Invalid prediction labels")
        recomputed = preds.assign(accuracy=(preds.true_label == preds.predicted_label).astype(float)).groupby(fold_keys + ["classifier"], dropna=False).accuracy.mean().reset_index()
        _same_table(folds, recomputed, fold_keys + ["classifier"], f"{scope} fold accuracy")
        if not sfs["accepted"].astype(str).str.lower().isin(["true", "false"]).all():
            raise ProvenanceError("SFS accepted must be boolean")
        sfs["accepted"] = sfs["accepted"].astype(str).str.lower() == "true"
        if set(sfs["analysis"]) != {scope}:
            raise ProvenanceError("SFS analysis scope mismatch")
        if not sfs["cv_accuracy"].between(0, 1).all():
            raise ProvenanceError("SFS accuracy outside [0,1]")
        if set(sfs["candidate_name"]) - set(allowed):
            raise ProvenanceError("Unknown SFS candidate")
        for _, row in sfs.iterrows():
            index = int(row.candidate_index)
            if row.candidate_index != index or index < 0 or index >= len(allowed) or allowed[index] != row.candidate_name:
                raise ProvenanceError("SFS candidate index does not match feature order")
        for split in identity["splits"][scope]:
            mask = np.ones(len(folds), dtype=bool)
            smask = np.ones(len(sfs), dtype=bool)
            for key in fold_keys:
                mask &= folds[key].astype(str).to_numpy() == str(split[key])
                smask &= sfs[key].astype(str).to_numpy() == str(split[key])
            path = sfs.loc[smask].sort_values("step")
            if path.empty or path.step.tolist() != list(range(1, len(path) + 1)):
                raise ProvenanceError("Missing/non-contiguous SFS path")
            if path.candidate_name.duplicated().any() or not bool(path.accepted.iloc[0]) or not path.accepted.iloc[:-1].all():
                raise ProvenanceError("Invalid SFS acceptance path")
            selected = path.loc[path.accepted, "candidate_name"].tolist()
            if not selected or len(set(selected)) != len(selected):
                raise ProvenanceError("Invalid selected features")
            cap = identity["config"]["sfs"].get("max_features")
            if cap is not None and len(selected) > int(cap):
                raise ProvenanceError("Selected features exceed configured SFS cap")
            for _, fold in folds.loc[mask].iterrows():
                if str(fold.selected_features).split(";") != selected or int(fold.n_selected_features) != len(selected):
                    raise ProvenanceError("Fold selected features disagree with SFS")
                try:
                    params = json.loads(fold.best_params)
                except (TypeError, ValueError) as exc:
                    raise ProvenanceError("Invalid model parameter JSON") from exc
                if not isinstance(params, dict):
                    raise ProvenanceError("Model parameters must be an object")
        # Reject extra SFS groups, including partial or foreign participants.
        _same_table(sfs[fold_keys].drop_duplicates(), pd.DataFrame(expected_keys)[fold_keys].drop_duplicates(), fold_keys, f"{scope} SFS fold coverage")
        if scope == "inter":
            expected = folds.groupby(["comparison", "classifier"]).accuracy.agg(["mean", "std", "count"]).reset_index().rename(columns={"mean": "lopo_accuracy_mean", "std": "lopo_accuracy_std", "count": "n_held_out_subjects"})
            pooled = preds.assign(pooled_epoch_accuracy=(preds.true_label == preds.predicted_label).astype(float)).groupby(["comparison", "classifier"]).pooled_epoch_accuracy.mean().reset_index()
            expected = expected.merge(pooled, on=["comparison", "classifier"], validate="one_to_one")
            summary = _csv(folder / "inter_summary.csv", expected.columns, ["comparison", "classifier"])
            _same_table(summary, expected, ["comparison", "classifier"], "inter summary")
        else:
            expected = folds.groupby(["subject_id", "comparison", "classifier"]).accuracy.agg(["mean", "std", "count"]).reset_index().rename(columns={"mean": "accuracy_mean", "std": "accuracy_std", "count": "n_outer_folds"})
            summary = _csv(folder / "intra_summary.csv", expected.columns, ["subject_id", "comparison", "classifier"])
            _same_table(summary, expected, ["subject_id", "comparison", "classifier"], "intra summary")
            overall = expected.groupby(["comparison", "classifier"]).accuracy_mean.agg(["mean", "std", "count"]).reset_index().rename(columns={"mean": "participant_accuracy_mean", "std": "participant_accuracy_std", "count": "n_subjects"})
            _same_table(_csv(folder / "intra_overall_summary.csv", overall.columns, ["comparison", "classifier"]), overall, ["comparison", "classifier"], "intra overall")
        freq_keys = (["subject_id"] if scope == "intra" else []) + ["comparison", "feature"]
        expected_freq = sfs.loc[sfs.accepted].rename(columns={"candidate_name": "feature"}).groupby(freq_keys).size().reset_index(name="selection_count")
        _same_table(_csv(folder / f"{scope}_feature_frequency.csv", expected_freq.columns, freq_keys), expected_freq, freq_keys, f"{scope} frequency")
    if _read_json(root / "config_snapshot.json") != identity["config"]:
        raise ProvenanceError("Config snapshot disagrees with run identity")
    if _read_json(root / "split_manifest.json") != identity["splits"]:
        raise ProvenanceError("Split manifest disagrees with run identity")


def _artifacts(root: Path) -> dict:
    out = {}
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ProvenanceError(f"Artifact symlink is not allowed: {path}")
        if not path.is_file() or (path.parent == root and path.name in (MANIFEST_NAME, "transaction_state.json")):
            continue
        rel = path.relative_to(root).as_posix()
        item = {"sha256": sha256_file(path), "bytes": path.stat().st_size}
        if path.suffix == ".csv":
            frame = _csv(path)
            item.update(rows=len(frame), columns=list(frame.columns))
        out[rel] = item
    return out


def _validate_outputs(root: Path, identity: dict) -> None:
    if identity.get("kind") == "evaluation":
        validate_evaluation_outputs(root, identity)
    elif identity.get("kind") == "report":
        for relative, contract in identity["table_contracts"].items():
            frame = _csv(root / _safe_relative(relative), contract["columns"], contract.get("keys", []))
            if len(frame) != contract["rows"] or list(frame.columns) != contract["columns"]:
                raise ProvenanceError(f"Report table mismatch: {relative}")
            if fingerprint(json.loads(frame.to_json(orient="split", index=False, double_precision=15))) != contract["values_sha256"]:
                raise ProvenanceError(f"Report table values mismatch: {relative}")
    else:
        raise ProvenanceError(f"Unknown run kind: {identity.get('kind')}")


def validate_complete_run(root: str | Path, expected_identity: dict | None = None) -> dict:
    root = Path(root)
    if root.is_symlink():
        raise ProvenanceError(f"Run root is a symlink: {root}")
    if (root / "transaction_state.json").exists():
        raise ProvenanceError("Complete run contains contradictory transaction state")
    manifest = _read_json(root / MANIFEST_NAME)
    if manifest.get("schema_version") != MANIFEST_VERSION or manifest.get("status") != "complete":
        raise ProvenanceError(f"Run is not complete: {root}")
    identity = manifest.get("identity", {})
    if manifest.get("run_id") != fingerprint(identity):
        raise ProvenanceError(f"Run identity digest mismatch: {root}")
    if expected_identity is not None and canonical_json(identity) != canonical_json(expected_identity):
        raise ProvenanceError(f"Resume identity mismatch: {root}; use a new output root")
    environment = identity.get("environment")
    environment_markers = {"schema_version", "environment_id", "reference", "reference_validation",
                           "install_artifacts", "installed_content"}
    if isinstance(environment, dict) and environment_markers.intersection(environment):
        # New evidence is self-contained and must remain semantically valid
        # after a manifest is rehashed. Historical identities without this
        # evidence keep their original, explicitly limited provenance contract.
        # Deleting only schema_version cannot downgrade new evidence to legacy.
        from .environment import EnvironmentError, validate_environment_evidence
        try:
            validate_environment_evidence(environment)
        except EnvironmentError as exc:
            raise ProvenanceError(f"Invalid run environment evidence: {exc}") from exc
    for path in manifest.get("artifacts", {}):
        _safe_relative(path)
    if _artifacts(root) != manifest.get("artifacts"):
        raise ProvenanceError(f"Missing, modified, or unexpected run artifacts: {root}")
    _validate_outputs(root, identity)
    return manifest


class RunTransaction:
    def __init__(self, destination, identity, *, resume=False, project_root=None):
        self.destination = protect_output(destination, project_root)
        self.identity = json.loads(canonical_json(identity))
        self.resume = resume
        self.project_root = project_root

    def preflight(self) -> bool:
        if self.destination.exists():
            validate_complete_run(self.destination, self.identity)
            if not self.resume:
                raise ProvenanceError(f"Run already exists: {self.destination}; use --resume or a new root")
            return True
        return False

    def execute(self, writer: Callable[[Path], None]) -> Path:
        if self.preflight():
            return self.destination
        self.destination.parent.mkdir(parents=True, exist_ok=True)
        stage = Path(tempfile.mkdtemp(prefix=f".staging-{self.destination.name}-", dir=self.destination.parent))
        _write_json(stage / "transaction_state.json", {"status": "running", "run_id": fingerprint(self.identity)})
        try:
            writer(stage)
            _validate_outputs(stage, self.identity)
            artifacts = _artifacts(stage)
            (stage / "transaction_state.json").unlink()
            _write_json(stage / MANIFEST_NAME, {"schema_version": MANIFEST_VERSION, "status": "complete",
                "run_id": fingerprint(self.identity), "identity": self.identity, "artifacts": artifacts,
                "completed_at": datetime.now(timezone.utc).isoformat()})
            validate_complete_run(stage, self.identity)
            # rename is atomic on the same filesystem and cannot replace a nonempty run.
            if self.destination.exists():
                raise ProvenanceError(f"Concurrent destination appeared: {self.destination}")
            os.rename(stage, self.destination)
            return self.destination
        except BaseException as exc:
            if stage.exists():
                (stage / MANIFEST_NAME).unlink(missing_ok=True)
                _write_json(stage / "transaction_state.json", {"status": "interrupted" if isinstance(exc, (KeyboardInterrupt, SystemExit)) else "failed", "run_id": fingerprint(self.identity), "error_type": type(exc).__name__})
            raise


@contextmanager
def _matrix_lock(root: Path):
    """OS locks are released on crashes; no stale PID file needs guessing."""
    path = root / ".matrix.lock"
    with path.open("a+b") as handle:
        try:
            if os.name == "nt":
                import msvcrt
                handle.seek(0)
                if not handle.read(1):
                    handle.write(b"0"); handle.flush()
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise ProvenanceError(f"Another process owns the matrix: {root}") from exc
        try:
            yield
        finally:
            if os.name == "nt":
                import msvcrt
                handle.seek(0); msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def table_contract(frame: pd.DataFrame) -> dict:
    # Serialize and parse the same CSV representation used for publication.
    restored = read_csv_literal(StringIO(frame.to_csv(index=False)))
    return {"rows": len(restored), "columns": list(restored.columns),
            "values_sha256": fingerprint(json.loads(restored.to_json(orient="split", index=False, double_precision=15)))}


class MatrixRun:
    def __init__(self, root, context_identity: dict, identities: Mapping[str, dict], requested, *, resume=False, project_root=None):
        self.root = protect_output(root, project_root)
        self.context = json.loads(canonical_json(context_identity))
        self.identities = {safe_name(name): json.loads(canonical_json(identity)) for name, identity in identities.items()}
        if any(name.casefold() == "reports" for name in self.identities):
            raise ProvenanceError("Reserved experiment name: reports")
        if len({name.casefold() for name in self.identities}) != len(self.identities):
            raise ProvenanceError("Case-insensitive experiment name collision")
        self.requested = [safe_name(name) for name in requested]
        if not self.requested or len(set(self.requested)) != len(self.requested) or set(self.requested) - set(self.identities):
            raise ProvenanceError("Requested experiment names are empty, duplicate, or unknown")
        self.resume = resume
        self.project_root = project_root

    def preflight(self) -> dict[str, Path]:
        if getattr(self, "source_guard", None) is not None:
            self.source_guard()
        found = {}
        if self.root.exists():
            if not self.root.is_dir():
                raise ProvenanceError("Matrix output must be a directory")
            context_path = self.root / "matrix_manifest.json"
            existing = [p for p in self.root.iterdir() if p.name != ".matrix.lock"]
            if existing and not context_path.exists():
                raise ProvenanceError("Existing output lacks a matrix manifest; legacy results are read-only")
            if context_path.exists():
                meta = _read_json(context_path)
                if meta != {"schema_version": "eeg-matrix-1", "context_id": fingerprint(self.context), "identity": self.context}:
                    raise ProvenanceError("Matrix context mismatch; use a new output root")
            for path in self.root.iterdir():
                if path.is_symlink():
                    raise ProvenanceError(f"Matrix control/output is a symlink: {path}")
                if path.name in ("matrix_manifest.json", "CURRENT.json", ".matrix.lock", "reports") or path.name.startswith(".staging-") or path.name.startswith(".CURRENT.json-"):
                    continue
                if path.name not in self.identities:
                    raise ProvenanceError(f"Unexpected experiment/output in matrix: {path.name}")
                validate_complete_run(path, self.identities[path.name])
                found[path.name] = path
            if (self.root / "CURRENT.json").exists():
                resolve_report_root(self.root)
        for name in self.requested:
            if name in found and not self.resume:
                raise ProvenanceError(f"Experiment already complete: {name}; use --resume")
        return found

    def run(self, evaluator: Callable[[str, Path], None], reporter: Callable[[dict[str, Path]], dict[str, pd.DataFrame]], *, required_experiments=None, report_metadata=None) -> tuple[dict[str, Path], Path]:
        # Validate every existing experiment before even creating a staging/snapshot.
        self.preflight()
        self.root.mkdir(parents=True, exist_ok=True)
        with _matrix_lock(self.root):
            self.preflight()
            context_path = self.root / "matrix_manifest.json"
            if not context_path.exists():
                _atomic_json(context_path, {"schema_version": "eeg-matrix-1", "context_id": fingerprint(self.context), "identity": self.context})
            for name in self.requested:
                transaction = RunTransaction(self.root / name, self.identities[name], resume=self.resume, project_root=self.project_root)
                transaction.execute(lambda stage, name=name: evaluator(name, stage))
            inventory = {}
            for name, identity in self.identities.items():
                path = self.root / name
                if path.exists():
                    validate_complete_run(path, identity)
                    inventory[name] = path
            if getattr(self, "source_guard", None) is not None:
                self.source_guard()
            tables = reporter(inventory)
            if getattr(self, "source_guard", None) is not None:
                self.source_guard()
            if not tables:
                raise ProvenanceError("A report must contain at least one table")
            for filename in tables:
                rel = _safe_relative(filename)
                if len(rel.parts) != 1 or rel.suffix != ".csv":
                    raise ProvenanceError("Report tables must be plain CSV filenames")
            required = set(required_experiments or self.identities)
            report_identity = {"kind": "report", "context_id": fingerprint(self.context),
                "inventory": {name: {"run_id": fingerprint(self.identities[name]), "manifest_sha256": sha256_file(path / MANIFEST_NAME)} for name, path in sorted(inventory.items())},
                "coverage": {"available": sorted(inventory), "required": sorted(required), "missing": sorted(required - set(inventory)), "study_matrix_complete": required <= set(inventory)},
                "metadata": report_metadata or {}, "table_contracts": {name: table_contract(frame) for name, frame in sorted(tables.items())}}
            report_id = fingerprint(report_identity)
            report_path = self.root / "reports" / report_id
            def write_report(stage):
                for filename, frame in tables.items():
                    frame.to_csv(stage / filename, index=False, encoding="utf-8-sig")
            RunTransaction(report_path, report_identity, resume=True, project_root=self.project_root).execute(write_report)
            pointer = {"schema_version": "eeg-current-report-1", "report": f"reports/{report_id}", "run_id": report_id}
            current = self.root / "CURRENT.json"
            if not current.exists() or _read_json(current) != pointer:
                _atomic_json(current, pointer)
            return inventory, report_path


def resolve_report_root(root: str | Path, *, allow_legacy=False) -> Path:
    root = Path(root)
    current = root / "CURRENT.json"
    if current.is_symlink() or (root / "reports").is_symlink():
        raise ProvenanceError("Report controls may not be symlinks")
    if current.exists():
        pointer = _read_json(current)
        if pointer.get("schema_version") != "eeg-current-report-1":
            raise ProvenanceError("Unknown CURRENT report pointer")
        relative = _safe_relative(str(pointer.get("report", "")))
        if len(relative.parts) != 2 or relative.parts[0] != "reports" or not re.fullmatch(r"[0-9a-f]{64}", relative.parts[1]):
            raise ProvenanceError("Invalid CURRENT report destination")
        path = root / relative
        manifest = validate_complete_run(path)
        if manifest["identity"].get("kind") != "report" or manifest["run_id"] != pointer.get("run_id") or path.name != manifest["run_id"]:
            raise ProvenanceError("CURRENT report identity mismatch")
        matrix = _read_json(root / "matrix_manifest.json")
        if matrix.get("schema_version") != "eeg-matrix-1" or matrix.get("context_id") != fingerprint(matrix.get("identity", {})):
            raise ProvenanceError("CURRENT matrix identity is invalid")
        if manifest["identity"].get("context_id") != matrix["context_id"]:
            raise ProvenanceError("CURRENT report belongs to a different matrix")
        for name, parent in manifest["identity"].get("inventory", {}).items():
            parent_root = root / safe_name(name)
            parent_manifest = validate_complete_run(parent_root)
            if parent_manifest["run_id"] != parent.get("run_id") or sha256_file(parent_root / MANIFEST_NAME) != parent.get("manifest_sha256"):
                raise ProvenanceError("CURRENT report parent run has changed")
        return path
    if (root / MANIFEST_NAME).exists():
        validate_complete_run(root)
        return root
    if (root / "matrix_manifest.json").exists():
        raise ProvenanceError("Managed matrix has no completed report")
    if allow_legacy and root.is_dir():
        return root
    raise ProvenanceError(f"No verified report at {root}; legacy reads must be explicit")


def legacy_inventory(root: str | Path) -> dict:
    root = Path(root)
    if not root.is_dir():
        raise ProvenanceError(f"Legacy root not found: {root}")
    if (root / MANIFEST_NAME).exists() or (root / "matrix_manifest.json").exists():
        raise ProvenanceError("Managed runs are not legacy inventories")
    return {"schema_version": "eeg-legacy-inventory-1", "status": "legacy_unverified",
            "source_root": str(root.resolve()), "observed_files": _artifacts(root),
            "historical_code": "unknown", "historical_environment": "unknown", "raw_hashes": "unknown",
            "warning": "Checksums identify present bytes only; not historical generation evidence or trusted resume."}


def reference_identity(root: str | Path, scopes) -> dict:
    root = Path(root)
    if (root / MANIFEST_NAME).exists():
        manifest = validate_complete_run(root)
        if manifest["identity"].get("scope_layout") == "flat":
            raise ProvenanceError("Reference root must contain intra/ or inter/; pass the parent of a flat scope run")
        if not set(scopes) <= set(manifest["identity"].get("scopes", [])):
            raise ProvenanceError("Reference does not contain required scopes")
        return {"status": "complete", "run_id": manifest["run_id"], "manifest_sha256": sha256_file(root / MANIFEST_NAME)}
    managed_scopes = {scope: root / scope for scope in scopes if (root / scope / MANIFEST_NAME).exists()}
    if managed_scopes:
        if len(managed_scopes) != len(scopes):
            raise ProvenanceError("Cannot mix managed and unverified scope references")
        components = {}
        for scope, path in managed_scopes.items():
            m = validate_complete_run(path)
            if m["identity"].get("scope_layout") != "flat" or m["identity"].get("scopes") != [scope]:
                raise ProvenanceError("Compound baseline has incompatible scope layout")
            components[scope] = {"run_id": m["run_id"], "manifest_sha256": sha256_file(path / MANIFEST_NAME)}
        return {"status": "complete", "run_id": "compound-" + fingerprint(components), "components": components}
    # Historical references remain explicitly unverified; never eligible for resume.
    files = {}
    for scope in scopes:
        folder = root / scope
        names = [f"{scope}_folds.csv", f"{scope}_predictions.csv", f"{scope}_sfs_path.csv", f"{scope}_feature_frequency.csv", f"{scope}_summary.csv"]
        if scope == "intra":
            names.append("intra_overall_summary.csv")
        for name in names:
            path = folder / name
            _csv(path)
            files[f"{scope}/{name}"] = {"sha256": sha256_file(path), "bytes": path.stat().st_size}
    return {"status": "legacy_unverified", "run_id": "legacy-" + fingerprint(files), "observed_files": files,
            "historical_code": "unknown", "historical_environment": "unknown", "raw_hashes": "unknown"}


def validate_reference_coverage(root: str | Path, identity: dict, expected_feature_set: str | None = None) -> None:
    """Historical anchors must cover the same units, labels and epoch counts.

    This does not upgrade legacy identity: matching counts cannot establish
    that old rows came from identical recordings. That limitation stays visible.
    """
    root = Path(root)
    rows = identity["dataset"]["metadata"]
    metadata = pd.DataFrame(rows)
    manifests = []
    if (root / MANIFEST_NAME).exists():
        manifests = [validate_complete_run(root)]
    else:
        manifests = [validate_complete_run(root / scope) for scope in identity["scopes"]
                     if (root / scope / MANIFEST_NAME).exists()]
    for manifest in manifests:
        reference = manifest["identity"]
        if canonical_json(reference["dataset"]["metadata"]) != canonical_json(rows):
            raise ProvenanceError("Reference dataset/cohort/epoch metadata differs")
        reference_cfg = {k: v for k, v in reference["config"].items() if k != "feature_set"}
        current_cfg = {k: v for k, v in identity["config"].items() if k != "feature_set"}
        if canonical_json(reference_cfg) != canonical_json(current_cfg):
            raise ProvenanceError("Reference evaluation settings differ")
        if expected_feature_set is not None:
            actual = reference["config"].get("feature_set", "all")
            aliases = {"all": "baseline"}
            if aliases.get(actual, actual) != aliases.get(expected_feature_set, expected_feature_set):
                raise ProvenanceError("Reference representation differs from its declared role")
        for scope in reference["scopes"]:
            if scope in identity["scopes"] and canonical_json(reference["splits"][scope]) != canonical_json(identity["splits"][scope]):
                raise ProvenanceError("Reference split membership differs")
    for scope in identity["scopes"]:
        unit = "subject_id" if scope == "intra" else "held_out_subject"
        keys = [unit, "comparison", "outer_fold", "classifier"]
        expected = [{**{k: split[k] for k in keys if k != "classifier"}, "classifier": classifier}
            for split in identity["splits"][scope] for classifier in CLASSIFIERS]
        folds = _csv(root / scope / f"{scope}_folds.csv", keys + ["accuracy"], keys)
        _same_table(folds, pd.DataFrame(expected), keys, f"{scope} reference fold coverage")
        if not folds.accuracy.between(0, 1).all():
            raise ProvenanceError("Reference accuracy is outside [0,1]")
        preds = _csv(root / scope / f"{scope}_predictions.csv", keys + ["subject_id", "true_label", "predicted_label"])
        if not preds.true_label.isin([0, 1]).all() or not preds.predicted_label.isin([0, 1]).all():
            raise ProvenanceError("Reference contains invalid labels")
        count_keys = ["subject_id", "comparison", "true_label", "classifier"]
        expected_rows = pd.concat([metadata[["subject_id", "comparison", "class_label"]].rename(columns={"class_label": "true_label"}).assign(classifier=c) for c in CLASSIFIERS])
        counts = preds.groupby(count_keys).size().reset_index(name="count")
        expected_counts = expected_rows.groupby(count_keys).size().reset_index(name="count")
        _same_table(counts, expected_counts, count_keys, f"{scope} reference class counts")
        recomputed = preds.assign(accuracy=(preds.true_label == preds.predicted_label).astype(float)).groupby(keys).accuracy.mean().reset_index()
        _same_table(folds, recomputed, keys, f"{scope} reference accuracy")
        if scope == "intra":
            expected_summary = folds.groupby(["subject_id", "comparison", "classifier"]).accuracy.agg(["mean", "std", "count"]).reset_index().rename(columns={"mean": "accuracy_mean", "std": "accuracy_std", "count": "n_outer_folds"})
            summary = _csv(root / scope / "intra_summary.csv", expected_summary.columns, ["subject_id", "comparison", "classifier"])
            _same_table(summary, expected_summary, ["subject_id", "comparison", "classifier"], "reference intra summary")
            overall = expected_summary.groupby(["comparison", "classifier"]).accuracy_mean.agg(["mean", "std", "count"]).reset_index().rename(columns={"mean": "participant_accuracy_mean", "std": "participant_accuracy_std", "count": "n_subjects"})
            _same_table(_csv(root / scope / "intra_overall_summary.csv", overall.columns), overall, ["comparison", "classifier"], "reference intra overall")
        else:
            summary = folds.groupby(["comparison", "classifier"]).accuracy.agg(["mean", "std", "count"]).reset_index().rename(columns={"mean": "lopo_accuracy_mean", "std": "lopo_accuracy_std", "count": "n_held_out_subjects"})
            pooled = preds.assign(pooled_epoch_accuracy=(preds.true_label == preds.predicted_label).astype(float)).groupby(["comparison", "classifier"]).pooled_epoch_accuracy.mean().reset_index()
            summary = summary.merge(pooled, on=["comparison", "classifier"], validate="one_to_one")
            _same_table(_csv(root / scope / "inter_summary.csv", summary.columns), summary, ["comparison", "classifier"], "reference inter summary")
