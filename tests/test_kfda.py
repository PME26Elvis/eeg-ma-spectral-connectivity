import numpy as np
from sklearn.datasets import make_circles
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from eeg_ma.kfda import KernelFisherDiscriminant


def test_kfda_learns_nonlinear_circles():
    X, y = make_circles(n_samples=160, factor=0.35, noise=0.04, random_state=7)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, stratify=y, random_state=7)
    scaler = StandardScaler().fit(Xtr)
    Xtr = scaler.transform(Xtr)
    Xte = scaler.transform(Xte)
    model = KernelFisherDiscriminant(C=10.0, gamma=1.0).fit(Xtr, ytr)
    acc = np.mean(model.predict(Xte) == yte)
    assert acc >= 0.9
