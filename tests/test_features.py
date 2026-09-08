import json
from pathlib import Path

import numpy as np

from eeg_ma.features import extract_epoch_features, band_coherence
from eeg_ma.constants import band_items


def _cfg():
    return json.loads(Path('configs/lab_replication.json').read_text(encoding='utf-8'))


def test_feature_dimension_42_plus_126():
    cfg = _cfg()
    fs = cfg['sampling_rate_hz']
    n = int(fs * cfg['epoch_seconds'])
    rng = np.random.default_rng(1)
    epoch = rng.normal(size=(n, 7))
    names, values = extract_epoch_features(epoch, fs, cfg['analysis_channels'], cfg)
    assert len(names) == 168
    assert values.shape == (168,)
    assert sum(name.startswith('BP__') for name in names) == 42
    assert sum(name.startswith('COH__') for name in names) == 126
    assert np.isfinite(values).all()


def test_coherence_identical_signals_is_one():
    cfg = _cfg()
    fs = cfg['sampling_rate_hz']
    n = int(fs * cfg['epoch_seconds'])
    rng = np.random.default_rng(2)
    x = rng.normal(size=n)
    bands = band_items(cfg['bands_hz'])
    coh = band_coherence(x, x, fs, bands, cfg['coherence'])
    for value in coh.values():
        assert 0.999999 <= value <= 1.000001
