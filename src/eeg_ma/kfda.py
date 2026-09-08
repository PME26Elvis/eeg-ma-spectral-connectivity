from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted


class KernelFisherDiscriminant(BaseEstimator, ClassifierMixin):
    """Regularized binary Kernel Fisher Discriminant classifier with an RBF kernel.

    The discriminant is estimated in the RKHS using the class-mean difference and
    within-class scatter. ``C`` is used as inverse regularization strength
    (lambda = 1 / C), matching the lab poster's C/gamma grid notation.

    This estimator intentionally exposes sklearn-compatible ``C`` and ``gamma``
    parameters so it can be used with GridSearchCV inside the training fold.
    """

    def __init__(self, C: float = 1.0, gamma: float = 1.0):
        self.C = C
        self.gamma = gamma

    def fit(self, X, y):
        X, y = check_X_y(X, y, dtype=float)
        classes = np.unique(y)
        if len(classes) != 2:
            raise ValueError("KFDA 目前只支援二分類")
        if float(self.C) <= 0 or float(self.gamma) <= 0:
            raise ValueError("C 與 gamma 必須 > 0")

        self.classes_ = classes
        self.X_fit_ = X.copy()
        K = rbf_kernel(X, X, gamma=float(self.gamma))
        n = len(X)

        class_means = []
        within = np.zeros((n, n), dtype=float)
        for cls in classes:
            idx = np.flatnonzero(y == cls)
            if len(idx) < 2:
                raise ValueError("KFDA 每個 class 至少需要 2 筆 training samples")
            Kc = K[:, idx]
            mean_vec = Kc.mean(axis=1)
            class_means.append(mean_vec)
            H = np.eye(len(idx)) - np.ones((len(idx), len(idx))) / float(len(idx))
            # Average within-class covariance contribution. Scaling only changes
            # the regularization trade-off, which is tuned by C.
            within += (Kc @ H @ Kc.T) / float(max(len(idx) - 1, 1))

        mean_diff = class_means[1] - class_means[0]
        reg = 1.0 / float(self.C)
        A = within + reg * np.eye(n)
        try:
            alpha = np.linalg.solve(A, mean_diff)
        except np.linalg.LinAlgError:
            alpha = np.linalg.pinv(A, rcond=1e-10) @ mean_diff

        train_scores = alpha @ K
        mu0 = float(np.mean(train_scores[y == classes[0]]))
        mu1 = float(np.mean(train_scores[y == classes[1]]))
        if mu1 < mu0:
            alpha = -alpha
            train_scores = -train_scores
            mu0 = -mu0
            mu1 = -mu1

        self.alpha_ = alpha
        self.threshold_ = 0.5 * (mu0 + mu1)
        self.class_score_means_ = np.array([mu0, mu1], dtype=float)
        self.n_features_in_ = X.shape[1]
        return self

    def decision_function(self, X):
        check_is_fitted(self, ["X_fit_", "alpha_", "threshold_"])
        X = check_array(X, dtype=float)
        K = rbf_kernel(self.X_fit_, X, gamma=float(self.gamma))
        return (self.alpha_ @ K) - self.threshold_

    def predict(self, X):
        scores = self.decision_function(X)
        return np.where(scores >= 0.0, self.classes_[1], self.classes_[0])
