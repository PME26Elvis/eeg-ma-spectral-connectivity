"""Explicit classifier rosters and bounded, reproducible forest search policy.

This module is configuration validation, not a model implementation.  The
forest algorithm itself depends only on Python and NumPy.  Omitting the new
configuration fields preserves the original three-classifier experiment.
"""
from __future__ import annotations

from copy import deepcopy
from itertools import product
import json
import math


LEGACY_CLASSIFIERS = ("LDA", "RBF-SVM", "KFDA")
SUPPORTED_CLASSIFIERS = (*LEGACY_CLASSIFIERS, "RandomForest", "LightGBM")
MAX_RF_CANDIDATES = 64
MAX_LIGHTGBM_CANDIDATES = 64
_EXECUTION_PARAMETERS = {"random_state", "n_jobs", "oob_score"}
_LIGHTGBM_EXECUTION_PARAMETERS = {
    "random_state", "n_jobs", "early_stopping_rounds", "early_stopping_min_delta",
}


def _canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def resolve_classifiers(cfg: dict) -> tuple[str, ...]:
    """Return the declared order; reject names that would otherwise be ignored."""
    if not isinstance(cfg, dict):
        raise ValueError("Classifier configuration must be an object")
    value = cfg.get("classifiers", list(LEGACY_CLASSIFIERS))
    if (not isinstance(value, list) or not value
            or any(not isinstance(name, str) or name not in SUPPORTED_CLASSIFIERS for name in value)
            or len(set(value)) != len(value)):
        raise ValueError(f"classifiers must be a nonempty unique list drawn from {SUPPORTED_CLASSIFIERS}")
    if "random_forest" in cfg and "RandomForest" not in value:
        raise ValueError("random_forest configuration requires RandomForest in classifiers")
    if "lightgbm" in cfg and "LightGBM" not in value:
        raise ValueError("lightgbm configuration requires LightGBM in classifiers")
    return tuple(value)


def _class_weight(value, *, name="random_forest"):
    """Keep JSON class keys explicit for this repository's binary 0/1 labels."""
    if value is None or value == "balanced":
        return value
    if not isinstance(value, dict) or not value:
        raise ValueError(f"Integrated {name} class_weight must be null, balanced, or a 0/1 mapping")
    result = {}
    for key, weight in value.items():
        if isinstance(key, bool) or key not in (0, 1, "0", "1") or str(key) in result:
            raise ValueError("Integrated class_weight keys must uniquely identify classes 0 and 1")
        if isinstance(weight, bool) or not isinstance(weight, (int, float)) or not math.isfinite(weight) or weight <= 0:
            raise ValueError("class_weight values must be finite and strictly positive")
        result[str(key)] = float(weight)
    if set(result) != {"0", "1"}:
        raise ValueError("Integrated class_weight must declare both binary classes 0 and 1")
    return result


def forest_constructor_parameters(parameters: dict) -> dict:
    """Translate JSON class labels to the numerical labels consumed by CART."""
    result = deepcopy(parameters)
    if isinstance(result.get("class_weight"), dict):
        result["class_weight"] = {int(key): value for key, value in result["class_weight"].items()}
    return result


def _forest_classifier_policy(cfg: dict) -> dict:
    """Validate all candidate settings before running selection or creating output.

    Candidate order is sorted grid keys × the explicitly supplied value order.
    Ties keep the first candidate.  Seeds and execution settings cannot be tuned.
    The returned value contains only JSON-compatible, detached configuration.
    """
    roster = resolve_classifiers(cfg)
    result = {
        "schema_version": "eeg-classifier-policy-1",
        "classifiers": list(roster),
        "feature_selection": "shared_training_only_lda_sfs",
        "random_forest": None,
    }
    if "RandomForest" not in roster:
        return result
    from .random_forest import ALGORITHM_VERSION, HandcraftedRandomForestClassifier
    supplied = cfg.get("random_forest", {})
    if not isinstance(supplied, dict) or set(supplied) - {"parameters", "grid"}:
        raise ValueError("random_forest accepts only parameters and grid objects")
    parameters = supplied.get("parameters", {})
    grid = supplied.get("grid", {})
    if not isinstance(parameters, dict) or not isinstance(grid, dict):
        raise ValueError("random_forest parameters and grid must be objects")
    defaults = HandcraftedRandomForestClassifier().get_params(deep=False)
    unknown = (set(parameters) | set(grid)) - set(defaults)
    if unknown:
        raise ValueError(f"Unknown random_forest parameters: {sorted(unknown)}")
    if set(grid) & _EXECUTION_PARAMETERS:
        raise ValueError("random_forest grid cannot tune random_state, n_jobs, or oob_score")
    for key, values in grid.items():
        if not isinstance(values, list) or not values:
            raise ValueError(f"random_forest grid.{key} must be a nonempty list")
        try:
            encoded = [_canonical(value) for value in values]
        except (TypeError, ValueError) as exc:
            raise ValueError(f"random_forest grid.{key} contains non-JSON or non-finite values") from exc
        if len(set(encoded)) != len(encoded):
            raise ValueError(f"random_forest grid.{key} contains duplicate values")
    count = math.prod(len(values) for values in grid.values())
    if count > MAX_RF_CANDIDATES:
        raise ValueError(f"random_forest grid exceeds the fixed {MAX_RF_CANDIDATES}-candidate limit")
    resolved = {**defaults, **deepcopy(parameters)}
    resolved["class_weight"] = _class_weight(resolved["class_weight"])
    # A reproducible outer-training experiment must not choose its seed later.
    seed = resolved["random_state"]
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**32:
        raise ValueError("Integrated random_forest random_state must be an integer in [0, 2**32)")
    HandcraftedRandomForestClassifier(**forest_constructor_parameters(resolved)).validate_parameters()
    keys = sorted(grid)
    candidates = []
    for values in product(*(grid[key] for key in keys)):
        candidate = {**deepcopy(resolved), **dict(zip(keys, deepcopy(values)))}
        candidate["class_weight"] = _class_weight(candidate["class_weight"])
        HandcraftedRandomForestClassifier(**forest_constructor_parameters(candidate)).validate_parameters()
        candidates.append(candidate)
    if len({_canonical(value) for value in candidates}) != len(candidates):
        raise ValueError("random_forest grid resolves to duplicate candidates")
    result["random_forest"] = {
        "algorithm_version": ALGORITHM_VERSION,
        "parameters": resolved,
        "grid": deepcopy(grid),
        "candidates": candidates,
        "candidate_order": "sorted_grid_keys_then_declared_value_order",
        "tie_break": "first_maximum_mean_inner_accuracy",
        "scoring": "arithmetic_mean_of_inner_fold_epoch_accuracy",
        "score_scope": "selection_diagnostic_after_shared_sfs_not_independent_performance",
        "seed_policy": "sha256_master_scope_comparison_subject_outer_stage_inner_v1",
        "oob_scope": "training_epoch_diagnostic_not_independent_participant_validation",
    }
    # Detect unsupported non-JSON objects before an expensive evaluation starts.
    try:
        _canonical(result)
    except (TypeError, ValueError) as exc:
        raise ValueError("Classifier policy is not finite JSON-compatible configuration") from exc
    return result


def lightgbm_constructor_parameters(parameters: dict) -> dict:
    """Convert the archived binary class keys without mutating configuration."""
    return forest_constructor_parameters(parameters)


def classifier_policy(cfg: dict) -> dict:
    """Resolve bounded classifier searches while preserving legacy identities.

    The original policy's complete shape remains unchanged unless LightGBM is
    explicitly selected.  Its integrated search never receives an evaluation
    set: all tree counts are selected inside the existing inner partitions.
    """
    result = _forest_classifier_policy(cfg)
    feature_mode = cfg.get("feature_selection_policy", "shared_training_only_lda_sfs")
    if feature_mode not in {"shared_training_only_lda_sfs", "all_declared_features"}:
        raise ValueError("Unknown feature_selection_policy")
    if feature_mode == "all_declared_features":
        if result["classifiers"] != ["RandomForest", "LightGBM"]:
            raise ValueError("all_declared_features is only supported for the explicit RF/LightGBM exploratory track")
        result["feature_selection"] = feature_mode
        result["random_forest"]["score_scope"] = "inner_model_search_without_sfs_not_independent_performance"
    if "LightGBM" not in result["classifiers"]:
        return result
    from .lightgbm import ALGORITHM_VERSION, HandcraftedLightGBMClassifier
    supplied = cfg.get("lightgbm", {})
    if not isinstance(supplied, dict) or set(supplied) - {"parameters", "grid"}:
        raise ValueError("lightgbm accepts only parameters and grid objects")
    parameters, grid = supplied.get("parameters", {}), supplied.get("grid", {})
    if not isinstance(parameters, dict) or not isinstance(grid, dict):
        raise ValueError("lightgbm parameters and grid must be objects")
    defaults = HandcraftedLightGBMClassifier().get_params(deep=False)
    unknown = (set(parameters) | set(grid)) - set(defaults)
    if unknown:
        raise ValueError(f"Unknown lightgbm parameters: {sorted(unknown)}")
    if set(grid) & _LIGHTGBM_EXECUTION_PARAMETERS:
        raise ValueError("lightgbm grid cannot tune seeds, n_jobs, or early-stopping settings")
    for key, values in grid.items():
        if not isinstance(values, list) or not values:
            raise ValueError(f"lightgbm grid.{key} must be a nonempty list")
        try:
            encoded = [_canonical(value) for value in values]
        except (TypeError, ValueError) as exc:
            raise ValueError(f"lightgbm grid.{key} contains non-JSON or non-finite values") from exc
        if len(set(encoded)) != len(encoded):
            raise ValueError(f"lightgbm grid.{key} contains duplicate values")
    if math.prod(len(values) for values in grid.values()) > MAX_LIGHTGBM_CANDIDATES:
        raise ValueError(f"lightgbm grid exceeds the fixed {MAX_LIGHTGBM_CANDIDATES}-candidate limit")
    resolved = {**defaults, **deepcopy(parameters)}
    resolved["class_weight"] = _class_weight(resolved["class_weight"], name="lightgbm")
    seed = resolved["random_state"]
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError("Integrated lightgbm random_state must be an integer in [0, 2**32)")
    if resolved["early_stopping_rounds"] is not None:
        raise ValueError("Integrated lightgbm disables early stopping; select n_estimators through the inner grid")
    if resolved["early_stopping_min_delta"] != 0:
        raise ValueError("Integrated lightgbm disables early stopping and requires early_stopping_min_delta=0")
    HandcraftedLightGBMClassifier(**lightgbm_constructor_parameters(resolved)).validate_parameters()
    keys = sorted(grid)
    candidates = []
    for values in product(*(grid[key] for key in keys)):
        candidate = {**deepcopy(resolved), **dict(zip(keys, deepcopy(values)))}
        candidate["class_weight"] = _class_weight(candidate["class_weight"], name="lightgbm")
        HandcraftedLightGBMClassifier(**lightgbm_constructor_parameters(candidate)).validate_parameters()
        candidates.append(candidate)
    if len({_canonical(value) for value in candidates}) != len(candidates):
        raise ValueError("lightgbm grid resolves to duplicate candidates")
    result["schema_version"] = "eeg-classifier-policy-2"
    result["lightgbm"] = {
        "algorithm_version": ALGORITHM_VERSION,
        "parameters": resolved,
        "grid": deepcopy(grid),
        "candidates": candidates,
        "candidate_order": "sorted_grid_keys_then_declared_value_order",
        "tie_break": "first_maximum_mean_inner_accuracy",
        "scoring": "arithmetic_mean_of_inner_fold_epoch_accuracy",
        "score_scope": "selection_diagnostic_after_shared_sfs_not_independent_performance",
        "seed_policy": "sha256_lightgbm_master_scope_comparison_subject_outer_stage_inner_v1",
        "early_stopping_policy": "disabled_in_integrated_cv_n_estimators_selected_by_inner_search",
    }
    if feature_mode == "all_declared_features":
        result["schema_version"] = "eeg-classifier-policy-3"
        result["lightgbm"]["score_scope"] = "inner_model_search_without_sfs_not_independent_performance"
    try:
        _canonical(result)
    except (TypeError, ValueError) as exc:
        raise ValueError("Classifier policy is not finite JSON-compatible configuration") from exc
    return result


def validate_classifier_config(cfg: dict) -> tuple[str, ...]:
    return tuple(classifier_policy(cfg)["classifiers"])
