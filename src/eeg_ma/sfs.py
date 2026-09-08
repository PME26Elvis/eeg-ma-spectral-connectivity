from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


@dataclass
class SFSResult:
    selected_indices: List[int]
    selected_names: List[str]
    best_score: float
    path: pd.DataFrame


def forward_select_lda(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: Sequence[str],
    cv_splits: Iterable[Tuple[np.ndarray, np.ndarray]],
    max_features: Optional[int] = None,
    min_improvement: float = 0.0,
) -> SFSResult:
    """Greedy Sequential Forward Selection using training-only LDA CV accuracy.

    The selected subset is common to LDA / RBF-SVM / KFDA, matching the project
    flow SFS -> classifiers. Selection stops when adding the best remaining
    feature does not improve inner-CV accuracy.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y)
    feature_names = list(feature_names)
    splits = [(np.asarray(a), np.asarray(b)) for a, b in cv_splits]
    n_features = X.shape[1]
    cap = n_features if max_features is None else min(int(max_features), n_features)
    if cap < 1:
        raise ValueError("SFS max_features 必須 >= 1")
    if not splits:
        raise ValueError("SFS 需要至少一個 inner-CV split")

    selected: List[int] = []
    remaining = list(range(n_features))
    current_score = -np.inf
    rows = []

    for step in range(1, cap + 1):
        best_idx = None
        best_score = -np.inf
        for candidate in remaining:
            cols = selected + [candidate]
            score = _cv_lda_accuracy(X[:, cols], y, splits)
            if score > best_score + 1e-15 or (
                abs(score - best_score) <= 1e-15 and (best_idx is None or candidate < best_idx)
            ):
                best_idx = candidate
                best_score = score

        if best_idx is None:
            break

        improvement = np.inf if not selected else best_score - current_score
        rows.append({
            "step": step,
            "candidate_index": int(best_idx),
            "candidate_name": feature_names[best_idx],
            "cv_accuracy": float(best_score),
            "improvement": float(improvement) if np.isfinite(improvement) else np.nan,
        })

        # First feature is always accepted. Thereafter, stop before adding a
        # feature that does not improve the inner-CV score.
        if selected and best_score <= current_score + float(min_improvement) + 1e-15:
            break

        selected.append(best_idx)
        remaining.remove(best_idx)
        current_score = best_score
        if not remaining:
            break

    if not selected:
        # Defensive fallback: accept the first evaluated best feature.
        if not rows:
            raise RuntimeError("SFS 無法評估任何 feature")
        selected = [int(rows[0]["candidate_index"])]
        current_score = float(rows[0]["cv_accuracy"])

    return SFSResult(
        selected_indices=selected,
        selected_names=[feature_names[i] for i in selected],
        best_score=float(current_score),
        path=pd.DataFrame(rows),
    )


def _cv_lda_accuracy(X: np.ndarray, y: np.ndarray, splits) -> float:
    scores = []
    for train_idx, val_idx in splits:
        model = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LinearDiscriminantAnalysis()),
        ])
        model.fit(X[train_idx], y[train_idx])
        pred = model.predict(X[val_idx])
        scores.append(accuracy_score(y[val_idx], pred))
    return float(np.mean(scores))
