import numpy as np

from eeg_ma.features import dft_squared_spectrum, lab_band_power


def test_frequency_resolution_for_5s_500hz():
    fs = 500.0
    n = int(5 * fs)
    f, _ = dft_squared_spectrum(np.zeros(n), fs)
    assert np.isclose(f[1] - f[0], 0.2)


def test_lab_dft_squared_definition_exact_bin_sine():
    fs = 500.0
    duration = 5.0
    n = int(fs * duration)
    t = np.arange(n) / fs
    x = 2.0 * np.sin(2 * np.pi * 7.0 * t) + 1.0 * np.sin(2 * np.pi * 10.0 * t)
    bands = [("theta", 4.0, 8.0), ("alpha", 8.0, 13.0)]
    bp = lab_band_power(x, fs, bands)

    # Exact-bin real sine: positive-frequency FFT amplitude = N*A/2.
    expected_7 = (n * 2.0 / 2.0) ** 2
    expected_10 = (n * 1.0 / 2.0) ** 2
    assert np.isclose(bp["theta"], expected_7, rtol=1e-10, atol=1e-5)
    assert np.isclose(bp["alpha"], expected_10, rtol=1e-10, atol=1e-5)
    assert np.isclose(bp["theta"] / bp["alpha"], 4.0, rtol=1e-10)
