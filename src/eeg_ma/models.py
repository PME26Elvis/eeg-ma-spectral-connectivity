from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from hashlib import sha256
import json
from typing import Dict, Iterable, List, Tuple

import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from .config import gamma_grid
from .kfda import KernelFisherDiscriminant


def lda_pipeline() -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LinearDiscriminantAnalysis()),
    ])


def svm_grid_search(cfg: Dict, cv_splits: Iterable[Tuple[np.ndarray, np.ndarray]]) -> GridSearchCV:
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", SVC(kernel="rbf")),
    ])
    grid = {
        "clf__C": [float(x) for x in cfg["rbf_grid"]["C"]],
        "clf__gamma": gamma_grid(cfg),
    }
    return GridSearchCV(
        model,
        param_grid=grid,
        scoring="accuracy",
        cv=list(cv_splits),
        n_jobs=int(cfg.get("n_jobs", -1)),
        refit=True,
        return_train_score=False,
    )


def kfda_grid_search(cfg: Dict, cv_splits: Iterable[Tuple[np.ndarray, np.ndarray]]) -> GridSearchCV:
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", KernelFisherDiscriminant()),
    ])
    grid = {
        "clf__C": [float(x) for x in cfg["rbf_grid"]["C"]],
        "clf__gamma": gamma_grid(cfg),
    }
    return GridSearchCV(
        model,
        param_grid=grid,
        scoring="accuracy",
        cv=list(cv_splits),
        n_jobs=int(cfg.get("n_jobs", -1)),
        refit=True,
        return_train_score=False,
        error_score="raise",
    )


def random_forest_seed(master_seed: int, context: dict, stage: str, inner_fold: int | None = None) -> int:
    """Stable, isolated RNG streams; neither row order nor Python hash is used."""
    required = {"scope", "comparison", "subject", "outer_fold"}
    if (not isinstance(context, dict) or set(context) != required or context["scope"] not in {"intra", "inter"}
            or any(not isinstance(context[key], str) or not context[key] for key in ("comparison", "subject"))
            or type(context["outer_fold"]) is not int or context["outer_fold"] < 1
            or type(master_seed) is not int or not 0 <= master_seed < 2**32):
        raise ValueError("RF seed context must identify scope, comparison, subject, and outer_fold")
    if (stage not in {"inner", "refit"} or (stage == "inner") != (inner_fold is not None)
            or (inner_fold is not None and (type(inner_fold) is not int or inner_fold < 1))):
        raise ValueError("RF seed stage/inner-fold identity is inconsistent")
    value = {"version": "rf-cv-seed-1", "master_seed": master_seed,
             **context, "stage": stage, "inner_fold": inner_fold}
    data = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    return int.from_bytes(sha256(data.encode("utf-8")).digest()[:4], "big")


@dataclass(frozen=True)
class RandomForestSearchResult:
    model: object
    best_params: dict
    best_score: float
    audit: dict


def _forest_splits(cv_splits, n_samples: int, *, groups=None):
    """No dropped validation rows, overlaps, duplicate indices, or foreign rows."""
    parsed = []
    for train, test in cv_splits:
        pair = []
        for value in (train, test):
            indices = np.asarray(value)
            if (indices.ndim != 1 or not len(indices) or indices.dtype.kind not in "iu"
                    or len(np.unique(indices)) != len(indices)
                    or np.any(indices < 0) or np.any(indices >= n_samples)):
                raise ValueError("RF inner split needs unique, nonempty in-range integer indices")
            pair.append(indices.astype(np.int64, copy=True))
        a, b = pair
        if np.intersect1d(a, b).size or len(a) + len(b) != n_samples:
            raise ValueError("RF inner train/test split must disjointly partition outer training rows")
        if groups is not None and (set(groups[a]) & set(groups[b]) or len(set(groups[b])) != 1):
            raise ValueError("RF inner split must hold out exactly one whole participant")
        parsed.append((a, b))
    if not parsed or sorted(np.concatenate([b for _, b in parsed]).tolist()) != list(range(n_samples)):
        raise ValueError("RF inner validation must cover each outer-training row exactly once")
    return parsed


def fit_random_forest(cfg: Dict, X, y, cv_splits, *, context: dict, groups=None) -> RandomForestSearchResult:
    """Manual bounded inner search followed by one outer-training-only refit.

    Only the existing split indices and raw selected training values enter this
    routine.  No scaler, sklearn search, ready-made tree, forest, or metric is
    used by the RF path.  Inner accuracies remain selection diagnostics because
    the shared LDA SFS has already used these inner partitions.
    """
    from .classifier_config import classifier_policy, forest_constructor_parameters
    from .random_forest import ALGORITHM_VERSION, HandcraftedRandomForestClassifier
    policy = classifier_policy(cfg)["random_forest"]
    if policy is None:
        raise ValueError("RandomForest must be explicitly selected")
    random_forest_seed(policy["parameters"]["random_state"], context, "refit")
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y)
    if (X.ndim != 2 or not X.shape[0] or not X.shape[1] or not np.isfinite(X).all()
            or y.ndim != 1 or len(y) != len(X) or not np.isin(y, [0, 1]).all()):
        raise ValueError("RF search requires finite selected features and corresponding binary labels")
    if groups is not None:
        groups = np.asarray(groups)
        if groups.ndim != 1 or len(groups) != len(y):
            raise ValueError("RF inner participant groups must match outer-training rows")
    if context["scope"] == "inter" and groups is None:
        raise ValueError("Inter-participant RF search requires explicit training groups")
    splits = _forest_splits(cv_splits, len(y), groups=groups)
    master = policy["parameters"]["random_state"]
    candidates = policy["candidates"]
    # Reject contextual bounds for *every* candidate before fitting any of them.
    for candidate in candidates:
        model = HandcraftedRandomForestClassifier(**forest_constructor_parameters(candidate))
        for train, _ in splits:
            model.validate_parameters(n_features=X.shape[1], n_samples=len(train))
    records = []
    best_index, best_exact_score = -1, None
    for index, parameters in enumerate(candidates):
        folds, exact_scores = [], []
        for inner_fold, (train, test) in enumerate(splits, start=1):
            seed = random_forest_seed(master, context, "inner", inner_fold)
            fit_params = {**forest_constructor_parameters(parameters), "random_state": seed, "oob_score": False}
            model = HandcraftedRandomForestClassifier(**fit_params).fit(X[train], y[train])
            prediction = model.predict(X[test])
            probabilities = model.predict_proba(X[test])
            exact_accuracy = Fraction(int(np.count_nonzero(prediction == y[test])), len(test))
            exact_scores.append(exact_accuracy)
            accuracy = float(exact_accuracy)
            folds.append({"inner_fold": inner_fold, "seed": seed,
                "train_indices": train.tolist(), "test_indices": test.tolist(),
                "classes": model.classes_.tolist(), "predictions": prediction.tolist(),
                "probabilities": probabilities.tolist(), "accuracy": accuracy,
                "oob_score_requested_during_search": False})
        # Accuracy has integer sufficient statistics. Exact rational ranking
        # prevents float summation from breaking a mathematically equal tie.
        exact_score = sum(exact_scores, Fraction(0)) / len(exact_scores)
        score = float(exact_score)
        records.append({"candidate_index": index, "parameters": parameters,
                        "folds": folds, "mean_inner_accuracy": score})
        if best_exact_score is None or exact_score > best_exact_score:
            best_index, best_exact_score = index, exact_score
    best_score = float(best_exact_score)
    chosen = {**candidates[best_index], "random_state": random_forest_seed(master, context, "refit")}
    final = HandcraftedRandomForestClassifier(**forest_constructor_parameters(chosen)).fit(X, y)
    oob = None
    if chosen["oob_score"]:
        values = np.asarray(final.oob_decision_function_)
        oob = {"scope": policy["oob_scope"], "counts": final.oob_counts_.tolist(),
               "eligible_mask": final.oob_eligible_mask_.tolist(),
               "probabilities": [[float(x) if np.isfinite(x) else None for x in row] for row in values],
               "coverage": float(final.oob_coverage_),
               "accuracy": float(final.oob_score_) if np.isfinite(final.oob_score_) else None}
    audit = {"algorithm_version": ALGORITHM_VERSION, "context": dict(context),
             "search_policy": policy, "candidates": records, "best_candidate_index": best_index,
             "best_score": best_score, "best_params": chosen,
             "training_feature_importances": final.feature_importances_.tolist(),
             "feature_importance_scope": "training_impurity_decrease_not_causal_or_test_permutation_importance",
             "oob": oob, "model": final.to_dict()}
    return RandomForestSearchResult(final, chosen, best_score, audit)
