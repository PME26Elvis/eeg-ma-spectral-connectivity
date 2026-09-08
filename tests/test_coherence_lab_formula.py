import json
from pathlib import Path

import numpy as np

from eeg_ma.constants import band_items
from eeg_ma.features import _lab_segment_spectra, band_coherence


def _cfg():
    return json.loads(Path("configs/lab_replication.json").read_text(encoding="utf-8"))


def test_lab_segment_spectra_matches_manual_fft_average():
    fs = 8.0
    # Two 1-s subsegments.  Values are intentionally asymmetric so this checks
    # the complex CPSD average, not only identical-signal behavior.
    x = np.array([1, 2, 0, -1, 0, 1, -2, 1, 2, 0, -1, 1, 0, -2, 1, 0], dtype=float)
    y = np.array([0, 1, 1, 0, -1, 0, 2, -1, 1, -1, 0, 2, 1, 0, -1, 1], dtype=float)

    f, gxx, gyy, gxy = _lab_segment_spectra(x, y, fs, segment_seconds=1.0)

    xs = x.reshape(2, 8)
    ys = y.reshape(2, 8)
    X = np.fft.rfft(xs, axis=1)
    Y = np.fft.rfft(ys, axis=1)
    np.testing.assert_allclose(f, np.fft.rfftfreq(8, d=1 / fs))
    np.testing.assert_allclose(gxx, np.mean(np.abs(X) ** 2, axis=0))
    np.testing.assert_allclose(gyy, np.mean(np.abs(Y) ** 2, axis=0))
    np.testing.assert_allclose(gxy, np.mean(X * np.conjugate(Y), axis=0))


def test_band_coherence_matches_tutorial_formula_then_band_mean():
    fs = 8.0
    x = np.array([1, 2, 0, -1, 0, 1, -2, 1, 2, 0, -1, 1, 0, -2, 1, 0], dtype=float)
    y = np.array([0, 1, 1, 0, -1, 0, 2, -1, 1, -1, 0, 2, 1, 0, -1, 1], dtype=float)
    cfg = {"segment_seconds": 1.0, "band_reduce": "mean"}
    bands = [("low", 1.0, 3.0)]

    got = band_coherence(x, y, fs, bands, cfg)["low"]

    X = np.fft.rfft(x.reshape(2, 8), axis=1)
    Y = np.fft.rfft(y.reshape(2, 8), axis=1)
    gxx = np.mean(np.abs(X) ** 2, axis=0)
    gyy = np.mean(np.abs(Y) ** 2, axis=0)
    gxy = np.mean(X * np.conjugate(Y), axis=0)
    coh = np.abs(gxy) ** 2 / (gxx * gyy)
    f = np.fft.rfftfreq(8, d=1 / fs)
    expected = float(np.mean(coh[(f >= 1.0) & (f < 3.0)]))
    assert np.isclose(got, expected)


def test_formal_config_yields_five_nonoverlapping_subsegments():
    cfg = _cfg()
    fs = float(cfg["sampling_rate_hz"])
    n = int(round(fs * float(cfg["epoch_seconds"])))
    segment_n = int(round(fs * float(cfg["coherence"]["segment_seconds"])))
    assert n == 2500
    assert segment_n == 500
    assert n // segment_n == 5
    assert n % segment_n == 0


def test_identical_signals_still_have_unit_coherence():
    cfg = _cfg()
    fs = float(cfg["sampling_rate_hz"])
    n = int(fs * float(cfg["epoch_seconds"]))
    rng = np.random.default_rng(20260909)
    x = rng.normal(size=n)
    coh = band_coherence(x, x, fs, band_items(cfg["bands_hz"]), cfg["coherence"])
    for value in coh.values():
        assert 0.999999 <= value <= 1.000001
