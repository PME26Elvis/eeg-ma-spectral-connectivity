import numpy as np
from sklearn.model_selection import StratifiedKFold

from eeg_ma.sfs import forward_select_lda


def test_sfs_selects_informative_feature_first():
    rng = np.random.default_rng(10)
    n = 100
    y = np.repeat([0, 1], n // 2)
    informative = y + rng.normal(scale=0.15, size=n)
    noise1 = rng.normal(size=n)
    noise2 = rng.normal(size=n)
    X = np.column_stack([noise1, informative, noise2])
    cv = list(StratifiedKFold(5, shuffle=True, random_state=1).split(X, y))
    result = forward_select_lda(X, y, ['noise1', 'signal', 'noise2'], cv, max_features=3)
    assert result.selected_names[0] == 'signal'
