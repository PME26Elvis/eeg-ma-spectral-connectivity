from __future__ import annotations

from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
from .constants import all_feature_names, band_items, channel_pairs


def dft_squared_spectrum(x: np.ndarray, fs: float) -> Tuple[np.ndarray, np.ndarray]:
    """Lab definition: PSD[ω_k] = |X[k]|^2, using positive-frequency DFT bins.

    No density normalization and no one-sided doubling are applied, because the
    laboratory teaching material explicitly defines the quantity as |X[k]|^2.
    """
    x = np.asarray(x, dtype=float)
    n = x.shape[0]
    X = np.fft.rfft(x)
    f = np.fft.rfftfreq(n, d=1.0 / float(fs))
    return f, np.abs(X) ** 2


def lab_band_power(x: np.ndarray, fs: float, bands: Iterable[Tuple[str, float, float]]) -> Dict[str, float]:
    f, p = dft_squared_spectrum(x, fs)
    out: Dict[str, float] = {}
    for name, lo, hi in bands:
        mask = (f >= lo) & (f < hi)
        if not np.any(mask):
            raise ValueError(f"頻帶 {name} [{lo},{hi}) 沒有任何 DFT bin")
        out[name] = float(np.sum(p[mask]))
    return out


def _lab_segment_spectra(
    x: np.ndarray,
    y: np.ndarray,
    fs: float,
    segment_seconds: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute the lab-tutorial auto/cross spectra averaged over subsegments.

    For each non-overlapping subsegment m:
        Gxx_m(f) = X_m(f) X_m*(f) = |X_m(f)|^2
        Gyy_m(f) = Y_m(f) Y_m*(f) = |Y_m(f)|^2
        Gxy_m(f) = X_m(f) Y_m*(f)

    The teaching slides then average Gxx/Gyy/Gxy over subsegments before
    forming magnitude-squared coherence.  No extra taper/detrend is applied
    here because those operations are not present in the supplied formula.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.ndim != 1 or y.ndim != 1 or x.shape != y.shape:
        raise ValueError("coherence x/y 必須是一維且長度相同")

    segment_samples = int(round(float(segment_seconds) * float(fs)))
    if segment_samples < 2:
        raise ValueError("coherence segment_seconds 太短")
    if len(x) % segment_samples != 0:
        raise ValueError(
            f"5-s trial 長度 {len(x)} samples 無法被 coherence segment "
            f"{segment_samples} samples 整除；為避免偷偷丟資料，請調整 config"
        )

    n_segments = len(x) // segment_samples
    if n_segments < 2:
        raise ValueError(
            "Coherence 至少需要 2 個 subsegments；單一 segment 依教材公式會退化為 1"
        )

    x_segments = x.reshape(n_segments, segment_samples)
    y_segments = y.reshape(n_segments, segment_samples)
    X = np.fft.rfft(x_segments, axis=1)
    Y = np.fft.rfft(y_segments, axis=1)

    gxx_bar = np.mean(np.abs(X) ** 2, axis=0)
    gyy_bar = np.mean(np.abs(Y) ** 2, axis=0)
    gxy_bar = np.mean(X * np.conjugate(Y), axis=0)
    f = np.fft.rfftfreq(segment_samples, d=1.0 / float(fs))
    return f, gxx_bar, gyy_bar, gxy_bar


def band_coherence(
    x: np.ndarray,
    y: np.ndarray,
    fs: float,
    bands: Iterable[Tuple[str, float, float]],
    cfg: Dict,
) -> Dict[str, float]:
    """Lab-tutorial magnitude-squared coherence, then band-wise mean.

    Coh_xy(f) = |mean_m Gxy_m(f)|^2 /
                (mean_m Gxx_m(f) * mean_m Gyy_m(f))

    The supplied teaching material explicitly says to divide one trial into
    N subsegments, compute PSD/CPSD for every subsegment, average those spectra,
    form Coh(f), and finally average Coh(f) inside each frequency band.
    """
    segment_seconds = float(cfg["segment_seconds"])
    f, gxx_bar, gyy_bar, gxy_bar = _lab_segment_spectra(
        x, y, fs, segment_seconds
    )

    denominator = gxx_bar * gyy_bar
    coh = np.zeros_like(denominator, dtype=float)
    valid = denominator > np.finfo(float).tiny
    coh[valid] = (np.abs(gxy_bar[valid]) ** 2) / denominator[valid]
    # Cauchy-Schwarz bounds the theoretical value to [0, 1].  Clip only tiny
    # floating-point excursions, not the underlying spectra.
    coh = np.clip(coh, 0.0, 1.0)

    out: Dict[str, float] = {}
    reducer = str(cfg.get("band_reduce", "mean"))
    for name, lo, hi in bands:
        mask = (f >= lo) & (f < hi)
        vals = coh[mask]
        if len(vals) == 0:
            raise ValueError(f"頻帶 {name} [{lo},{hi}) 沒有任何 coherence bin")
        if reducer == "mean":
            value = float(np.mean(vals))
        elif reducer == "sum":
            value = float(np.sum(vals))
        else:
            raise ValueError(f"未知 coherence.band_reduce={reducer}")
        out[name] = value
    return out


def extract_epoch_features(epoch: np.ndarray, fs: float, channels: Sequence[str], cfg: Dict) -> Tuple[List[str], np.ndarray]:
    epoch = np.asarray(epoch, dtype=float)
    if epoch.ndim != 2 or epoch.shape[1] != len(channels):
        raise ValueError(f"epoch shape 應為 [samples,{len(channels)}]，實際 {epoch.shape}")

    bands = band_items(cfg["bands_hz"])
    names = all_feature_names(channels, bands)
    values: List[float] = []

    # BP: channel-major, then band order.
    for ci, _ch in enumerate(channels):
        bp = lab_band_power(epoch[:, ci], fs, bands)
        values.extend(bp[name] for name, _, _ in bands)

    # COH: pair-major according to channel order, then band order.
    for a, b in channel_pairs(channels):
        ia, ib = channels.index(a), channels.index(b)
        coh = band_coherence(epoch[:, ia], epoch[:, ib], fs, bands, cfg["coherence"])
        values.extend(coh[name] for name, _, _ in bands)

    arr = np.asarray(values, dtype=float)
    if len(names) != 168 or arr.shape != (168,):
        raise AssertionError(f"Feature dimension 應為 168，實際 names={len(names)}, values={arr.shape}")
    if not np.isfinite(arr).all():
        raise ValueError("Feature extraction 產生 NaN/Inf")
    return names, arr
