"""Read-only, assumption-labelled grouped-CV cost planning.

Planning estimates cannot guarantee wall time or RSS; a matching, measured
calibration is mandatory before enforcing any estimated runtime budget.
"""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
import platform
import sys

import numpy as np

from .classifier_config import classifier_policy
from .evaluation import make_evaluation_splits
from .extract import feature_columns
from .provenance import fingerprint


PLAN_VERSION = "eeg-training-cost-plan-1"
CALIBRATION_VERSION = "eeg-benchmark-calibration-1"


class CostPlanError(ValueError):
    pass


class CostLimitExceeded(CostPlanError):
    """Planning rejected a run before training or publication."""


def _positive(value, name):
    if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
        raise CostPlanError(f"{name} must be a finite positive number")
    return float(value)


def machine_identity():
    cpu = platform.processor()
    if sys.platform == "linux":
        try:
            for line in Path("/proc/cpuinfo").read_text().splitlines():
                if line.startswith("model name"):
                    cpu = line.split(":", 1)[1].strip()
                    break
        except OSError:
            pass
    return {
        "machine": platform.machine(),
        "system": platform.system(),
        "cpu_model": cpu or "unknown",
        "available_cpus": os.cpu_count(),
        "affinity_cpus": len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
        "python": ".".join(str(x) for x in sys.version_info[:3]),
        "numpy": np.__version__,
        "blas_thread_env": {name: os.environ.get(name) for name in
                            ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")},
    }


def tree_work_units(kind, rows, dimensions, parameters):
    """Declared monotonic work proxy, not FLOPs or measured seconds."""
    rows = max(1, int(rows))
    dimensions = max(1, int(dimensions))
    if kind == "RandomForest":
        trees = int(parameters["n_estimators"])
        maxf = parameters["max_features"]
        if maxf is None:
            sampled = dimensions
        elif maxf == "sqrt":
            sampled = max(1, int(math.sqrt(dimensions)))
        elif maxf == "log2":
            sampled = max(1, int(math.log2(dimensions)))
        elif type(maxf) is int:
            sampled = min(dimensions, maxf)
        elif type(maxf) is float:
            sampled = max(1, int(math.ceil(maxf * dimensions)))
        else:
            sampled = dimensions
        return int(math.ceil(trees * sampled * rows * math.log2(rows + 1)))
    if kind == "LightGBM":
        rounds = int(parameters["n_estimators"])
        leaves = int(parameters["num_leaves"])
        fraction = float(parameters["feature_fraction"])
        sampled = max(1, math.ceil(dimensions * fraction))
        return int(math.ceil(rounds * sampled * rows * (1 + math.log2(rows + 1))
                             * (1 + math.log2(leaves + 1))))
    raise CostPlanError(f"Unknown handwritten learner: {kind}")


def _node_upper(name, rows, params):
    if name == "RandomForest":
        leaves = max(1, rows // params["min_samples_leaf"])
        if params["max_depth"] is not None:
            leaves = min(leaves, 2 ** min(int(params["max_depth"]), 30))
        return int(params["n_estimators"]) * (2 * leaves - 1)
    return int(params["n_estimators"]) * (2 * int(params["num_leaves"]) - 1)


def plan_evaluation_cost(features, config, *, scope, calibration=None,
                         max_estimated_seconds=None, max_estimated_rss_mib=None):
    """Examine all planned folds/candidates, never fit or publish results."""
    if scope not in ("inter", "intra"):
        raise CostPlanError("scope must be inter or intra")
    policy = classifier_policy(config)
    cols = feature_columns(features, str(config.get("feature_set", "all")))
    if not cols:
        raise CostPlanError("No predeclared features")
    splits = make_evaluation_splits(features, config, scope)
    cap = int(config["sfs"]["max_features"]) if config["sfs"].get("max_features") is not None else len(cols)
    dimensions = len(cols) if policy["feature_selection"] == "all_declared_features" else min(len(cols), cap)
    full_tree_fits, nodes_upper, units, max_rows = 0, 0, 0, 0
    for split in splits:
        outer = len(split["train_epoch_ids"])
        max_rows = max(max_rows, outer)
        for learner, key in (("RandomForest", "random_forest"), ("LightGBM", "lightgbm")):
            details = policy.get(key)
            if details is None:
                continue
            for params in details["candidates"]:
                for inner in split["inner_splits"]:
                    size = len(inner["train_epoch_ids"])
                    units += tree_work_units(learner, size, dimensions, params)
                    full_tree_fits += 1
                    nodes_upper += _node_upper(learner, size, params)
            units += max(tree_work_units(learner, outer, dimensions, p) for p in details["candidates"])
            nodes_upper += max(_node_upper(learner, outer, p) for p in details["candidates"])
            full_tree_fits += 1
    sfs_fits = (0 if policy["feature_selection"] == "all_declared_features"
                else sum(sum(len(cols) - i for i in range(dimensions)) * len(split["inner_splits"])
                         for split in splits))
    input_mib = 8 * len(features) * len(cols) / 2**20
    fold_mib = 8 * max_rows * len(cols) / 2**20
    # Empirical headroom assumption; neither hard memory cap nor OS peak RSS.
    memory_mib = math.ceil(12 * (input_mib + fold_mib) + 256 * nodes_upper / 2**20 + 128)
    report = {
        "schema_version": PLAN_VERSION, "scope": scope,
        "dataset": {"n_rows": len(features), "participants": int(features.subject_id.nunique()),
                    "n_feature_candidates": len(cols), "max_model_features": dimensions,
                    "feature_order_sha256": fingerprint(cols)},
        "policy_sha256": fingerprint(policy), "outer_folds": len(splits),
        "full_tree_model_fits": full_tree_fits, "sfs_inner_model_fits_upper": sfs_fits,
        "tree_work_units": units, "projected_nodes_upper": nodes_upper,
        "estimated_memory_mib": memory_mib,
        "memory_scope": "heuristic_planning_envelope_not_hard_process_RSS_limit",
        "cpu_environment": machine_identity(),
        "calibration": None, "estimated_seconds": None,
        "time_scope": "not_estimated_without_same_host_measured_calibration",
        "status": "preflight_only",
    }
    if calibration is not None:
        saved = json.loads(Path(calibration).read_text(encoding="utf-8"))
        if (saved.get("schema_version") != CALIBRATION_VERSION
                or saved.get("host") != report["cpu_environment"]
                or type(saved.get("seconds_per_unit")) not in (int, float)
                or not math.isfinite(saved["seconds_per_unit"])
                or saved["seconds_per_unit"] <= 0
                or type(saved.get("measurements")) is not int
                or saved["measurements"] < 3
                or type(saved.get("safety_factor")) not in (int, float)
                or not 1 <= saved["safety_factor"] <= 10):
            raise CostPlanError("Runtime calibration missing, invalid or from another machine/environment")
        report["calibration"] = saved
        report["estimated_seconds"] = math.ceil(units * saved["seconds_per_unit"]
                                                 * saved["safety_factor"])
        report["time_scope"] = "empirical_same_host_extrapolation_not_hard_timeout"
    if max_estimated_rss_mib is not None:
        limit = _positive(max_estimated_rss_mib, "max_estimated_rss_mib")
        if memory_mib > limit:
            raise CostLimitExceeded(f"planned memory {memory_mib} MiB exceeds {limit:g} MiB")
    if max_estimated_seconds is not None:
        limit = _positive(max_estimated_seconds, "max_estimated_seconds")
        if report["estimated_seconds"] is None:
            raise CostLimitExceeded("Time budget needs measured same-host calibration; refusing uncalibrated execution")
        if report["estimated_seconds"] > limit:
            raise CostLimitExceeded(f"estimated runtime {report['estimated_seconds']} s exceeds {limit:g} s")
    return report
