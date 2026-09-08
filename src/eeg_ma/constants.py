from __future__ import annotations

from itertools import combinations
from typing import Iterable, List, Mapping, Sequence, Tuple

DEFAULT_CHANNELS = ["FP1", "FP2", "F7", "F3", "Fz", "F4", "F8"]
DEFAULT_BANDS = [
    ("delta", 1.0, 4.0),
    ("theta", 4.0, 8.0),
    ("alpha", 8.0, 13.0),
    ("beta_low", 13.0, 20.0),
    ("beta_high", 20.0, 30.0),
    ("gamma", 30.0, 45.0),
]

MARKERS = {
    "session_start": 1,
    "session_end": 2,
    "type1_stage_start": 10,
    "type1_task_onset": 11,
    "type2_task_onset": 12,
    "type2_stage_start": 13,
    "rest1_block_start": 19,
    "rest1_epoch_onset": 20,
    "pre_answer_fixation": 21,
    "rest2_block_start": 23,
    "rest2_epoch_onset": 24,
    "answer_prompt": 30,
    "response_less": 41,
    "response_equal": 42,
    "response_greater": 43,
    "post_answer_fixation": 50,
}


def band_items(bands_hz: Mapping[str, Sequence[float]]) -> List[Tuple[str, float, float]]:
    return [(name, float(bounds[0]), float(bounds[1])) for name, bounds in bands_hz.items()]


def channel_pairs(channels: Sequence[str]) -> List[Tuple[str, str]]:
    return list(combinations(channels, 2))


def bp_feature_names(channels: Sequence[str], bands: Iterable[Tuple[str, float, float]]) -> List[str]:
    band_names = [b[0] for b in bands]
    return [f"BP__{ch}__{band}" for ch in channels for band in band_names]


def coh_feature_names(channels: Sequence[str], bands: Iterable[Tuple[str, float, float]]) -> List[str]:
    band_names = [b[0] for b in bands]
    return [
        f"COH__{a}__{b}__{band}"
        for a, b in channel_pairs(channels)
        for band in band_names
    ]


def all_feature_names(channels: Sequence[str], bands: Iterable[Tuple[str, float, float]]) -> List[str]:
    bands = list(bands)
    return bp_feature_names(channels, bands) + coh_feature_names(channels, bands)
