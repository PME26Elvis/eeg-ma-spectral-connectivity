"""Reference-only strict CV reconstruction from immutable handwritten-learner evidence.

Fast provenance checks run first. We then rerun the actual training procedure on
*every* stored inner candidate and each final outer refit, comparing the new
model state and out-of-fold probabilities against the archived values. This is
a reproducibility audit of the original learner, not independent mathematical
proof; separate tiny exhaustive oracles provide the latter.
"""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
import json
import math
from pathlib import Path
import time

import numpy as np
import pandas as pd

from .artifacts import new_output_directory
from .classifier_config import classifier_policy
from .evaluation import _evaluation_indices
from .extract import feature_columns
from .environment import verify_reference_environment
from .models import fit_lightgbm, fit_random_forest
from .provenance import (
    ProvenanceError, canonical_json, code_identity, fingerprint, protect_output,
    read_csv_literal, sha256_file, validate_complete_run,
)
from .sfs import forward_select_lda


SCHEMA = "eeg-strict-train-replay-1"
REL_TOL = 2e-12
ABS_TOL = 2e-14


class StrictReplayError(ValueError):
    """A model, input, fit or reproducibility claim is inconsistent."""


class ReplayResourceLimit(StrictReplayError):
    """The requested audit exceeds explicitly declared resource limits."""


@dataclass(frozen=True)
class ReplayBudget:
    max_outer_folds: int = 8
    max_candidates: int = 8
    max_model_fits: int = 128
    max_projected_nodes: int = 250_000
    max_seconds: float = 600.0

    def __post_init__(self):
        for name in ("max_outer_folds", "max_candidates",
                     "max_model_fits", "max_projected_nodes"):
            item = getattr(self, name)
            if type(item) is not int or item < 1:
                raise ValueError(f"{name} must be a positive integer")
        if (type(self.max_seconds) not in (int, float)
                or not math.isfinite(self.max_seconds) or self.max_seconds <= 0):
            raise ValueError("max_seconds must be finite and positive")


def _equal(actual, expected, label):
    """Strict type-safe recursion with a fixed, published floating tolerance."""
    if type(actual) is not type(expected):
        raise StrictReplayError(f"{label}: JSON type changed {type(actual).__name__} != {type(expected).__name__}")
    if isinstance(expected, dict):
        if set(actual) != set(expected):
            raise StrictReplayError(f"{label}: record fields differ")
        for key in expected:
            _equal(actual[key], expected[key], f"{label}.{key}")
        return
    if isinstance(expected, list):
        if len(actual) != len(expected):
            raise StrictReplayError(f"{label}: list length differs")
        for index, (a, e) in enumerate(zip(actual, expected)):
            _equal(a, e, f"{label}[{index}]")
        return
    if isinstance(expected, float):
        if not (math.isfinite(actual) and math.isfinite(expected)
                and math.isclose(actual, expected, rel_tol=REL_TOL, abs_tol=ABS_TOL)):
            raise StrictReplayError(f"{label}: numeric replay differs")
    elif actual != expected:
        raise StrictReplayError(f"{label}: replay differs")


def _bounds_model(parameters, candidate, rows):
    """Conservative upper limit, not an estimated measured node count."""
    if candidate == "RandomForest":
        leaf_cap = max(1, rows // parameters["min_samples_leaf"])
        if parameters["max_depth"] is not None:
            leaf_cap = min(leaf_cap, 2 ** min(parameters["max_depth"], 30))
        return int(parameters["n_estimators"]) * (2 * leaf_cap - 1)
    return int(parameters["n_estimators"]) * (2 * int(parameters["num_leaves"]) - 1)


def _preflight(identity, folder, budget):
    policy = classifier_policy(identity["config"])
    scopes = identity["scopes"]
    if not isinstance(scopes, list) or any(scope not in {"intra", "inter"} for scope in scopes):
        raise StrictReplayError("Only explicitly versioned intra/inter CV runs can be replayed")
    if not scopes:
        raise StrictReplayError("Empty replay scope")
    models = [m for m in ("RandomForest", "LightGBM") if m in policy["classifiers"]]
    if not models:
        raise StrictReplayError("Strict tree replay requires RF or LightGBM")
    total_outer = sum(len(identity["splits"][scope]) for scope in scopes)
    if total_outer > budget.max_outer_folds:
        raise ReplayResourceLimit(f"outer_folds={total_outer} exceeds max_outer_folds={budget.max_outer_folds}")
    total_fits, projected_nodes, candidates_max = 0, 0, 0
    metadata = pd.DataFrame(identity["dataset"]["metadata"])
    for scope in scopes:
        for split in identity["splits"][scope]:
            outer_size = len(split["train_epoch_ids"])
            for name in models:
                lookup = policy["random_forest" if name == "RandomForest" else "lightgbm"]
                candidates = lookup["candidates"]
                candidates_max = max(candidates_max, len(candidates))
                total_fits += len(candidates) * len(split["inner_splits"]) + 1
                for candidate in candidates:
                    projected_nodes += sum(
                        _bounds_model(candidate, name, len(inner["train_epoch_ids"]))
                        for inner in split["inner_splits"]
                    )
                projected_nodes += max(_bounds_model(candidate, name, outer_size) for candidate in candidates)
    if candidates_max > budget.max_candidates:
        raise ReplayResourceLimit(f"candidate_count={candidates_max} exceeds max_candidates={budget.max_candidates}")
    if total_fits > budget.max_model_fits:
        raise ReplayResourceLimit(f"model_fits={total_fits} exceeds max_model_fits={budget.max_model_fits}")
    if projected_nodes > budget.max_projected_nodes:
        raise ReplayResourceLimit(
            f"projected_nodes_upper_bound={projected_nodes} exceeds max_projected_nodes={budget.max_projected_nodes}"
        )
    return {"scopes": scopes, "models": models, "outer_folds": total_outer,
            "candidate_limit_observed": candidates_max, "complete_model_fits": total_fits,
            "projected_nodes_upper_bound": projected_nodes,
            "resource_policy": {
                "max_outer_folds": budget.max_outer_folds,
                "max_candidates": budget.max_candidates,
                "max_model_fits": budget.max_model_fits,
                "max_projected_nodes": budget.max_projected_nodes,
                "max_seconds": budget.max_seconds,
            }}


def _verify_sfs(identity, frame, split, train_ids, selected_names, inner_splits, folder, scope):
    fcols = feature_columns(frame, str(identity["config"].get("feature_set", "all")))
    mode = classifier_policy(identity["config"])["feature_selection"]
    if mode == "all_declared_features":
        if selected_names != fcols:
            raise StrictReplayError("Full-feature track does not use every ordered feature")
        return {"feature_policy": mode, "retrained_selection": False, "n_features": len(fcols)}
    indexed = frame.set_index("epoch_id")
    X = indexed.loc[train_ids, fcols].to_numpy(float)
    y = indexed.loc[train_ids, "class_label"].to_numpy(int)
    sfs = forward_select_lda(
        X, y, fcols, inner_splits,
        max_features=identity["config"]["sfs"].get("max_features"),
        min_improvement=float(identity["config"]["sfs"].get("min_improvement", 0.0)),
    )
    if sfs.selected_names != selected_names:
        raise StrictReplayError("Fresh training-only SFS selected a different feature subset")
    # Fast validation checks path structure, not whether inner LDA scores
    # were numerically produced by this particular outer-training fold.
    archive = read_csv_literal(folder / f"{scope}_sfs_path.csv")
    unit = "subject_id" if scope == "intra" else "held_out_subject"
    required = {unit, "comparison", "outer_fold", "step", "candidate_index",
                "candidate_name", "cv_accuracy", "improvement", "accepted"}
    if not required.issubset(archive.columns):
        raise StrictReplayError("Archived SFS trace lacks required fields")
    observed = archive.loc[
        archive[unit].astype(str).eq(str(split[unit]))
        & archive["comparison"].astype(str).eq(str(split["comparison"]))
        & archive["outer_fold"].eq(split["outer_fold"])
    ].sort_values("step").reset_index(drop=True)
    expected = sfs.path.sort_values("step").reset_index(drop=True)
    if len(observed) != len(expected) or not len(expected):
        raise StrictReplayError("Archived SFS candidate path coverage differs")
    for key in ("step", "candidate_index", "candidate_name"):
        if observed[key].tolist() != expected[key].tolist():
            raise StrictReplayError(f"Archived SFS {key} differs from actual training")
    accepted = observed["accepted"].astype(str).str.lower().map({"true": True, "false": False}).tolist()
    if accepted != expected["candidate_name"].isin(set(selected_names)).tolist():
        raise StrictReplayError("Archived SFS acceptance was not reproduced")
    for key in ("cv_accuracy", "improvement"):
        # Never turn arbitrary non-numeric strings into NaN: the first
        # SFS improvement is legitimately null and must not conceal tampering.
        try:
            got = pd.to_numeric(observed[key], errors="raise").to_numpy(float)
            wanted = pd.to_numeric(expected[key], errors="raise").to_numpy(float)
        except (TypeError, ValueError) as exc:
            raise StrictReplayError(f"Archived SFS {key} contains a nonnumeric value") from exc
        if not np.allclose(got, wanted, rtol=REL_TOL, atol=ABS_TOL, equal_nan=True):
            raise StrictReplayError(f"Archived SFS {key} was not reproduced from training data")
    folds = read_csv_literal(folder / f"{scope}_folds.csv")
    matching = folds.loc[
        folds[unit].astype(str).eq(str(split[unit]))
        & folds["comparison"].astype(str).eq(str(split["comparison"]))
        & folds["outer_fold"].eq(split["outer_fold"])
    ]
    if (matching.empty or not np.allclose(
            pd.to_numeric(matching["sfs_inner_accuracy"], errors="coerce").to_numpy(float),
            sfs.best_score, rtol=REL_TOL, atol=ABS_TOL)):
        raise StrictReplayError("Archived SFS best score differs from true training-only CV")
    return {"feature_policy": mode, "retrained_selection": True,
            "n_features": len(selected_names), "selected_features": selected_names}


def _replay_scope(root: Path, identity: dict, scope: str, budget: ReplayBudget, begin: float):
    policy = classifier_policy(identity["config"])
    models = [m for m in ("RandomForest", "LightGBM") if m in policy["classifiers"]]
    files = {name: root / scope if identity.get("scope_layout") != "flat" else root
             for name in models}
    first_model = models[0]
    first_source = files[first_model] / ("random_forest_inputs.csv" if first_model == "RandomForest"
                                       else "lightgbm_inputs.csv")
    if not first_source.is_file() or sha256_file(first_source) != identity["dataset"]["table_sha256"]:
        raise StrictReplayError("Strict replay source differs from the archived data identity")
    data = read_csv_literal(first_source, float_precision="round_trip")
    if len(data) != identity["dataset"]["n_rows"] or list(data.columns) != identity["dataset"]["columns"]:
        raise StrictReplayError("Strict replay input schema or row count differs")
    if data.epoch_id.duplicated().any():
        raise StrictReplayError("Replay source has ambiguous epoch identities")
    indexed = data.set_index("epoch_id")
    results = []
    unit = "subject_id" if scope == "intra" else "held_out_subject"
    for ordinal, split in enumerate(identity["splits"][scope]):
        train_ids = split["train_epoch_ids"]
        test_ids = split["test_epoch_ids"]
        seen = set(train_ids)
        if seen.intersection(test_ids) or len(seen) != len(train_ids) or len(test_ids) != len(set(test_ids)):
            raise StrictReplayError("Outer train/test epoch identities overlap")
        for name in models:
            if time.monotonic() - begin >= budget.max_seconds:
                raise ReplayResourceLimit("Runtime budget expired before a full outer refit")
            folder = files[name]
            evidence_name = "random_forest_audit.json" if name == "RandomForest" else "lightgbm_audit.json"
            source_name = "random_forest_inputs.csv" if name == "RandomForest" else "lightgbm_inputs.csv"
            from .provenance import _read_json
            audit = _read_json(folder / evidence_name)
            if sha256_file(folder / source_name) != identity["dataset"]["table_sha256"]:
                raise StrictReplayError("Tree learners did not use identical input source bytes")
            record = audit["folds"][ordinal]
            names = list(record["selected_features"])
            if (names != list(dict.fromkeys(names))
                    or any(name not in identity["feature_order"] for name in names)):
                raise StrictReplayError("Selected feature list is missing, repeated or foreign")
            positions = {key: n for n, key in enumerate(train_ids)}
            inner = [
                (np.asarray([positions[epoch] for epoch in part["train_epoch_ids"]], dtype=int),
                 np.asarray([positions[epoch] for epoch in part["test_epoch_ids"]], dtype=int))
                for part in split["inner_splits"]
            ]
            # Exactly the rows accessible to the original outer-training model.
            Xtrain = indexed.loc[train_ids, names].to_numpy(float)
            ytrain = indexed.loc[train_ids, "class_label"].to_numpy(int)
            Xtest = indexed.loc[test_ids, names].to_numpy(float)
            context = {"scope": scope, "comparison": split["comparison"],
                       "subject": split[unit], "outer_fold": split["outer_fold"]}
            _verify_sfs(identity, data, split, train_ids, names, inner, folder, scope)
            groups = (indexed.loc[train_ids, "subject_id"].astype(str).to_numpy()
                      if scope == "inter" else None)
            if name == "RandomForest":
                fitted = fit_random_forest(identity["config"], Xtrain, ytrain, inner,
                                           context=context, groups=groups)
            else:
                fitted = fit_lightgbm(identity["config"], Xtrain, ytrain, inner,
                                      context=context, groups=groups)
            if time.monotonic() - begin >= budget.max_seconds:
                raise ReplayResourceLimit("Runtime budget expired after an outer candidate+refit batch")
            for field, expected in fitted.audit.items():
                if field not in record:
                    raise StrictReplayError(f"Archived {name} outer audit lacks {field}")
                _equal(record[field], expected, f"{name}/{scope}/outer{ordinal + 1}/{field}")
            predictions = fitted.model.predict(Xtest).tolist()
            probabilities = fitted.model.predict_proba(Xtest).tolist()
            _equal(record["test_predictions"], predictions, f"{name}/outer predictions")
            _equal(record["test_probabilities"], probabilities, f"{name}/outer probabilities")
            _equal(record["test_classes"], fitted.model.classes_.tolist(), f"{name}/classes")
            if name == "LightGBM":
                _equal(record["test_raw_scores"], fitted.model.decision_function(Xtest).tolist(),
                       "LightGBM outer decision scores")
            results.append({
                "scope": scope, "outer_fold": split["outer_fold"],
                "comparison": split["comparison"], "held_out_subject": split[unit],
                "classifier": name, "selected_features": names,
                "n_inner_candidates": len(record["candidates"]),
                "n_inner_folds": len(split["inner_splits"]),
                "model_fits_replayed": len(record["candidates"]) * len(split["inner_splits"]) + 1,
                "n_outer_predictions_checked": len(test_ids),
                "model_sha256": fingerprint(record["model"]),
                "retrained_model_sha256": fingerprint(fitted.model.to_dict()),
                "validation_scope": "full_inner_retraining_and_final_refit_same_original_algorithm",
            })
    return results


def verify_strict_run(path: str | Path, *, budget: ReplayBudget | None = None,
                      report_dir: str | Path | None = None, preflight_only=False) -> dict:
    """Fail closed unless the full immutable run AND actual reference env match.

    Budget refusal and soft runtime expiry never publish a 'complete' report.
    The model training code and original source hashes must match this checkout.
    """
    root = Path(path)
    budget = budget or ReplayBudget()
    begin = time.monotonic()
    manifest = validate_complete_run(root)
    identity = manifest["identity"]
    original_manifest_sha = sha256_file(root / "run_manifest.json")
    plan = _preflight(identity, root, budget)
    if preflight_only:
        return {"status": "not_run_preflight_only", "schema_version": SCHEMA, "plan": plan,
                "run_id": manifest["run_id"]}
    if report_dir is not None:
        target = protect_output(report_dir)
        if target.exists():
            raise StrictReplayError("Verified report destination already exists; refuse before refitting")
    current_code = code_identity()
    if canonical_json(identity.get("code")) != canonical_json(current_code):
        raise StrictReplayError("Strict verification needs exactly the run's original checkout and code identity")
    reference = verify_reference_environment(identity.get("environment"))
    results = []
    for scope in plan["scopes"]:
        results.extend(_replay_scope(root, identity, scope, budget, begin))
    if len(results) != plan["outer_folds"] * len(plan["models"]):
        raise StrictReplayError("Strict replay does not cover the declared complete outer/model matrix")
    checked_fits = sum(result["model_fits_replayed"] for result in results)
    if checked_fits != plan["complete_model_fits"]:
        raise StrictReplayError("Strict replay did not fit all declared inner candidates and outer models")
    # An input or program may change while the expensive verification runs.
    validate_complete_run(root, identity)
    if sha256_file(root / "run_manifest.json") != original_manifest_sha:
        raise StrictReplayError("Original run manifest changed during replay")
    if canonical_json(code_identity()) != canonical_json(current_code):
        raise StrictReplayError("Analysis checkout or source code changed during replay")
    verify_reference_environment(identity.get("environment"))
    response = {
        "schema_version": SCHEMA, "status": "verified_strict_train_replay",
        "run_id": manifest["run_id"], "source_manifest_sha256": original_manifest_sha,
        "reference_environment_id": reference["environment_id"], "reference_lock_id": reference["lock_id"],
        "code_commit": current_code["commit"], "code_source_sha256": current_code["working_source_sha256"],
        "time_seconds": time.monotonic() - begin, "preflight": plan,
        "models": results, "n_complete_refits": checked_fits,
        "tolerances": {"relative": REL_TOL, "absolute": ABS_TOL},
        "verified_claim": "all_candidate_fold_predictions_and_full_final_model_recomputed_using_same_learner",
        "separate_mathematical_oracle": "see bounded independent tiny exhaustive oracle report; not a proof for all full-size models",
    }
    if report_dir is not None:
        with new_output_directory(report_dir) as temporary:
            from .provenance import _write_json
            _write_json(temporary / "strict_report.json", response)
            validate_complete_run(root, identity)
            if (sha256_file(root / "run_manifest.json") != original_manifest_sha
                    or canonical_json(code_identity()) != canonical_json(current_code)):
                raise StrictReplayError("Input run or analysis source changed before strict report publication")
    return response
