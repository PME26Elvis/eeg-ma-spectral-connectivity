from __future__ import annotations

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
