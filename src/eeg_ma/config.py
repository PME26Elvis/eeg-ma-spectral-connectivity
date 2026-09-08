from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List


def load_config(path: str | Path) -> Dict[str, Any]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        cfg = json.load(f)
    _validate_config(cfg)
    return cfg


def gamma_grid(cfg: Dict[str, Any]) -> List[float]:
    grid = cfg["rbf_grid"]
    if "gamma_values" in grid:
        return [float(x) for x in grid["gamma_values"]]
    base = float(grid["gamma_base"])
    return [base ** int(e) for e in grid["gamma_exponents"]]


def _validate_config(cfg: Dict[str, Any]) -> None:
    required = [
        "sampling_rate_hz", "epoch_seconds", "raw_channels", "analysis_channels", "bandpass",
        "bands_hz", "coherence", "comparisons", "sfs", "cv", "rbf_grid"
    ]
    missing = [k for k in required if k not in cfg]
    if missing:
        raise ValueError(f"Config 缺少欄位: {missing}")
    if len(cfg["raw_channels"]) != 8:
        raise ValueError("HNC raw acquisition 必須定義 8 channels。")
    if len(cfg["analysis_channels"]) != 7:
        raise ValueError("第一階段 replication analysis 必須使用 7 個 frontal channels。")
    if len(cfg["bands_hz"]) != 6:
        raise ValueError("必須定義 6 個頻帶。")
