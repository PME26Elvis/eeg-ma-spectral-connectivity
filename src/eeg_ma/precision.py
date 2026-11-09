"""Prospective precision planning for a fixed model's participant-macro BA.

No study performance or canonical features are inputs.  Individuals are the
sampling units.  Epochs only generate each individual's two class recalls.
"""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import platform
from typing import Callable

import numpy as np
import scipy
from scipy.stats import binom, norm

REPORT_VERSION = "eeg-precision-plan-1"
CONFIG_VERSION = "eeg-precision-config-1"


def _canonical(value) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _digest(value) -> str:
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def _integer(value, name: str, minimum=1) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _finite(value, name: str, low=None, high=None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value):
        raise ValueError(f"{name} must be finite numeric data")
    value = float(value)
    if (low is not None and value < low) or (high is not None and value > high):
        raise ValueError(f"{name} is outside its permitted interval")
    return value


def wilson_interval(successes: int, trials: int, confidence_level: float = 0.95) -> tuple[float, float]:
    """Pointwise Monte Carlo interval for independent simulation indicators."""
    _integer(trials, "trials")
    _integer(successes, "successes", 0)
    if successes > trials or not 0 < confidence_level < 1:
        raise ValueError("Invalid Wilson interval inputs")
    z = float(norm.ppf((1 + confidence_level) / 2))
    p = successes / trials
    denominator = 1 + z * z / trials
    center = (p + z * z / (2 * trials)) / denominator
    radius = z * np.sqrt(p * (1 - p) / trials + z * z / (4 * trials * trials)) / denominator
    return max(0.0, float(center - radius)), min(1.0, float(center + radius))


def recruitment_cap(target_n: int, retention: float, assurance: float = 0.95) -> int:
    """Smallest integer cap with P(at least target_n evaluable people)>=assurance."""
    _integer(target_n, "target_n")
    retention = _finite(retention, "retention", 0, 1)
    assurance = _finite(assurance, "assurance", 0, 1)
    if retention == 0 or assurance in (0, 1):
        raise ValueError("Need positive retention and assurance strictly between zero and one")
    lower, upper = target_n, target_n
    while binom.sf(target_n - 1, upper, retention) < assurance:
        upper *= 2
        if upper > 10_000_000:
            raise ValueError("Recruitment requirement exceeds supported range")
    while lower < upper:
        middle = (lower + upper) // 2
        if binom.sf(target_n - 1, middle, retention) >= assurance:
            upper = middle
        else:
            lower = middle + 1
    return lower


def _scores(values, name: str) -> np.ndarray:
    result = np.asarray(values, dtype=float)
    if result.ndim != 1 or len(result) < 2:
        raise ValueError(f"{name} must contain at least two participant-level scores")
    if not np.isfinite(result).all() or np.any((result < 0) | (result > 1)):
        raise ValueError(f"{name} requires finite participant BA values in [0,1]; missing cases cannot be dropped")
    return result


def _bootstrap_intervals(primary: np.ndarray, paired_delta: np.ndarray | None, *,
                         resamples: int, confidence: float, rng: np.random.Generator,
                         batch_size: int = 8) -> tuple[np.ndarray, np.ndarray | None]:
    """Independent bootstrap index draws for each simulated study, shared by its models."""
    studies, participants = primary.shape
    bounds = np.empty((studies, 2), dtype=float)
    delta_bounds = np.empty_like(bounds) if paired_delta is not None else None
    quantiles = [(1 - confidence) / 2, (1 + confidence) / 2]
    for start in range(0, studies, batch_size):
        stop = min(start + batch_size, studies)
        indices = rng.integers(0, participants, size=(stop - start, resamples, participants), dtype=np.int32)
        samples = np.take_along_axis(primary[start:stop, None, :], indices, axis=2).mean(axis=2)
        bounds[start:stop] = np.quantile(samples, quantiles, axis=1, method="linear").T
        if paired_delta is not None:
            samples = np.take_along_axis(paired_delta[start:stop, None, :], indices, axis=2).mean(axis=2)
            delta_bounds[start:stop] = np.quantile(samples, quantiles, axis=1, method="linear").T
    return bounds, delta_bounds


def participant_bootstrap(scores, reference=None, *, seed: int, n_resamples: int = 10_000,
                          confidence_level: float = 0.95, subject_ids=None) -> dict:
    """Percentile CI for equally weighted independent participants and paired delta.

    Each array must already contain one BA per person, calculated from both
    classes on identical eligible epochs. This function never pools epochs or
    silently drops missing scores. Degenerate samples remain explicitly flagged.
    """
    primary = _scores(scores, "scores")
    _integer(seed, "seed", 0)
    _integer(n_resamples, "n_resamples", 2)
    if not 0 < confidence_level < 1:
        raise ValueError("confidence_level must be strictly between zero and one")
    if subject_ids is not None:
        ids = list(subject_ids)
        if len(ids) != len(primary) or any(not isinstance(x, str) or not x.strip() for x in ids) or len(set(ids)) != len(ids):
            raise ValueError("Need one unique literal subject ID per participant score")
    delta = None
    if reference is not None:
        reference = _scores(reference, "reference")
        if reference.shape != primary.shape:
            raise ValueError("Paired reference must contain the same participants in the same order")
        delta = primary - reference
    ci, delta_ci = _bootstrap_intervals(primary[None, :], None if delta is None else delta[None, :],
        resamples=n_resamples, confidence=confidence_level, rng=np.random.default_rng(seed), batch_size=1)
    def summary(values, interval):
        degenerate = bool(np.ptp(values) <= 1e-12)
        return {"estimate": float(np.mean(values)), "ci_lower": float(interval[0]),
                "ci_upper": float(interval[1]), "half_width": float((interval[1] - interval[0]) / 2),
                "degenerate": degenerate, "confirmatory_eligible": not degenerate}
    out = {"n_participants": len(primary), "method": "percentile", "confidence_level": confidence_level,
           "bootstrap_resamples": n_resamples, "seed": seed, "primary": summary(primary, ci[0])}
    if delta is not None:
        out["paired_delta"] = summary(delta, delta_ci[0])
    return out


def validate_precision_config(config: dict) -> None:
    if config.get("schema_version") != CONFIG_VERSION:
        raise ValueError("Unsupported precision config schema")
    purpose = config.get("purpose")
    if purpose not in {"design_validation", "smoke_test"}:
        raise ValueError("Precision purpose must be explicit")
    _integer(config["seed"], "seed", 0)
    ns = config["candidate_n"]
    if not isinstance(ns, list) or not ns or ns != sorted(set(ns)):
        raise ValueError("candidate_n must be a sorted nonempty unique integer list")
    for n in ns:
        _integer(n, "candidate_n", 10)
    for key in ("screening_replicates", "validation_replicates", "screening_bootstrap_resamples", "bootstrap_resamples"):
        _integer(config[key], key, 20)
    if purpose == "design_validation" and (config["validation_replicates"] < 1000 or config["bootstrap_resamples"] != 10_000):
        raise ValueError("Design validation requires >=1000 independent studies and the final 10000-resample procedure")
    if config.get("ci_method") != "percentile":
        raise ValueError("Only the prespecified percentile procedure is implemented")
    for key in ("confidence_level", "mc_confidence_level", "coverage_floor", "width_probability_floor", "completion_assurance"):
        value = _finite(config[key], key, 0, 1)
        if value in (0, 1):
            raise ValueError(f"{key} must be strictly between zero and one")
    _finite(config["half_width_target"], "half_width_target", 0.001, 0.5)
    scenarios = config["scenarios"]
    if not isinstance(scenarios, list) or not scenarios:
        raise ValueError("Need explicit precision scenarios")
    ids = [s.get("id") for s in scenarios]
    if any(not isinstance(x, str) or not x.strip() for x in ids) or len(ids) != len(set(ids)):
        raise ValueError("Scenario IDs must be unique nonempty strings")
    for s in scenarios:
        mean = _finite(s["mean"], "mean", 0, 1)
        sd = _finite(s["between_sd"], "between_sd", 0, 0.5)
        if not 0 < mean < 1 or not 0 < sd * sd < mean * (1 - mean):
            raise ValueError("Beta participant moments are infeasible")
        delta = _finite(s["paired_mean_delta"], "paired_mean_delta", 0, mean)
        if delta == mean:
            raise ValueError("Reference mean must be positive")
        rho = _finite(s["epoch_icc"], "epoch_icc", 0, 1)
        if rho == 1:
            raise ValueError("Residual epoch ICC must be below one")
        for key in ("model_shared_weight", "epoch_shared_probability", "dropout_probability", "missing_class_probability"):
            _finite(s[key], key, 0, 1)
        if s["dropout_probability"] == 1 or s["missing_class_probability"] == 1:
            raise ValueError("Scenario must have positive evaluability")
        slope = _finite(s["informative_retention_slope"], "informative_retention_slope", -1, 1)
        retention_at_bounds = [1 - s["dropout_probability"] + slope * (x - mean) for x in (0, 1)]
        if min(retention_at_bounds) < 0 or max(retention_at_bounds) > 1:
            raise ValueError("Linear retention must stay within [0,1] on the entire support; clipping would invalidate analytical truth")
        epochs = s["epochs_per_class"]
        if not isinstance(epochs, list) or len(epochs) != 2:
            raise ValueError("Need two class-specific epoch counts")
        for value in epochs:
            _integer(value, "epochs_per_class", 1)
    for value in config["retention_sensitivity"]:
        if not 0 < _finite(value, "retention_sensitivity", 0, 1) <= 1:
            raise ValueError("Retention sensitivities must be positive")


def scenario_truth(scenario: dict) -> dict:
    """Analytical truth under linear informative selection, never clipped."""
    mean, variance = scenario["mean"], scenario["between_sd"] ** 2
    qc_retention = 1 - scenario["dropout_probability"]
    shift = scenario["informative_retention_slope"] * variance / qc_retention
    evaluable = mean + shift
    scale = 1 - scenario["paired_mean_delta"] / mean
    reference = scale * (scenario["model_shared_weight"] * evaluable + (1 - scenario["model_shared_weight"]) * mean)
    return {"primary_truth_recruited": mean, "primary_truth_evaluable": evaluable,
            "reference_truth_evaluable": reference, "paired_truth_recruited": scenario["paired_mean_delta"],
            "paired_truth_evaluable": evaluable - reference,
            "retention": qc_retention * (1 - scenario["missing_class_probability"]),
            "selection_shift": shift}


def _simulated_scores(scenario: dict, replicates: int, n: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    mean, sd = scenario["mean"], scenario["between_sd"]
    concentration = mean * (1 - mean) / (sd * sd) - 1
    a, b = mean * concentration, (1 - mean) * concentration
    # Rejection samples exactly from the evaluable population; cap completion
    # is assessed separately by its exact binomial probability.
    retained = []
    need = replicates * n
    while need:
        size = max(64, int(np.ceil(need / scenario_truth(scenario)["retention"] * 1.1)))
        p = rng.beta(a, b, size=size)
        retention = 1 - scenario["dropout_probability"] + scenario["informative_retention_slope"] * (p - mean)
        selected = p[(rng.random(size) < retention) & (rng.random(size) >= scenario["missing_class_probability"])]
        selected = selected[:need]
        retained.append(selected)
        need -= len(selected)
    primary = np.concatenate(retained).reshape(replicates, n)
    independent = rng.beta(a, b, size=primary.shape)
    scale = 1 - scenario["paired_mean_delta"] / mean
    reference = scale * (scenario["model_shared_weight"] * primary + (1 - scenario["model_shared_weight"]) * independent)
    primary_ba = np.zeros_like(primary)
    reference_ba = np.zeros_like(reference)
    for epochs in scenario["epochs_per_class"]:
        if scenario["epoch_icc"]:
            phi = 1 / scenario["epoch_icc"] - 1
            def latent(probability):
                result = probability.copy()
                interior = (probability > 0) & (probability < 1)
                result[interior] = rng.beta(probability[interior] * phi, (1 - probability[interior]) * phi)
                return result
            primary_class, reference_class = latent(primary), latent(reference)
        else:
            primary_class, reference_class = primary, reference
        shared = rng.random((replicates, n, epochs))
        independent_errors = rng.random(shared.shape)
        mask = rng.random(shared.shape) < scenario["epoch_shared_probability"]
        reference_uniform = np.where(mask, shared, independent_errors)
        primary_ba += 0.5 * np.mean(shared < primary_class[:, :, None], axis=2)
        reference_ba += 0.5 * np.mean(reference_uniform < reference_class[:, :, None], axis=2)
    return primary_ba, reference_ba


def _cell(config: dict, scenario: dict, n: int, *, stage: str) -> dict:
    replicates = config[f"{stage}_replicates"]
    resamples = config["bootstrap_resamples"] if stage == "validation" else config["screening_bootstrap_resamples"]
    # Each named cell and its score/bootstrap streams are independent and stay
    # unchanged if an unrelated scenario is appended to the config.
    key = _digest({"seed": config["seed"], "stage": stage, "scenario": scenario, "n": n})
    seeds = np.random.SeedSequence([config["seed"], *[int(key[i:i+8], 16) for i in range(0, 32, 8)]]).spawn(2)
    primary, reference = _simulated_scores(scenario, replicates, n, np.random.default_rng(seeds[0]))
    delta = primary - reference
    ci, delta_ci = _bootstrap_intervals(primary, delta, resamples=resamples,
        confidence=config["confidence_level"], rng=np.random.default_rng(seeds[1]))
    truth = scenario_truth(scenario)
    halfwidth = (ci[:, 1] - ci[:, 0]) / 2
    delta_halfwidth = (delta_ci[:, 1] - delta_ci[:, 0]) / 2
    covered = (ci[:, 0] <= truth["primary_truth_evaluable"]) & (truth["primary_truth_evaluable"] <= ci[:, 1])
    recruited_covered = (ci[:, 0] <= truth["primary_truth_recruited"]) & (truth["primary_truth_recruited"] <= ci[:, 1])
    delta_covered = (delta_ci[:, 0] <= truth["paired_truth_evaluable"]) & (truth["paired_truth_evaluable"] <= delta_ci[:, 1])
    width_successes = int(np.sum(halfwidth <= config["half_width_target"]))
    coverage_successes = int(np.sum(covered))
    coverage_interval = wilson_interval(coverage_successes, replicates, config["mc_confidence_level"])
    width_interval = wilson_interval(width_successes, replicates, config["mc_confidence_level"])
    return {"stage": stage, "scenario_id": scenario["id"], "target_n": n, "replicates": replicates,
        "bootstrap_resamples": resamples, "cell_seed_digest": key, **truth,
        "primary_coverage_successes": coverage_successes, "primary_coverage": coverage_successes / replicates,
        "primary_coverage_mc_lower": coverage_interval[0], "primary_coverage_mc_upper": coverage_interval[1],
        "primary_halfwidth_successes": width_successes, "primary_width_pass_rate": width_successes / replicates,
        "primary_width_pass_mc_lower": width_interval[0], "primary_width_pass_mc_upper": width_interval[1],
        "primary_halfwidth_mean": float(np.mean(halfwidth)), "primary_halfwidth_p90": float(np.quantile(halfwidth, .9)),
        "primary_bias_evaluable": float(np.mean(primary) - truth["primary_truth_evaluable"]),
        "primary_bias_recruited": float(np.mean(primary) - truth["primary_truth_recruited"]),
        "primary_coverage_recruited": float(np.mean(recruited_covered)),
        "primary_degenerate_studies": int(np.sum(np.ptp(primary, axis=1) <= 1e-12)),
        "paired_coverage_successes": int(np.sum(delta_covered)), "paired_coverage": float(np.mean(delta_covered)),
        "paired_coverage_mc_lower": wilson_interval(int(np.sum(delta_covered)), replicates, config["mc_confidence_level"])[0],
        "paired_coverage_mc_upper": wilson_interval(int(np.sum(delta_covered)), replicates, config["mc_confidence_level"])[1],
        "paired_halfwidth_mean": float(np.mean(delta_halfwidth)), "paired_halfwidth_p90": float(np.quantile(delta_halfwidth, .9)),
        "paired_bias_evaluable": float(np.mean(delta) - truth["paired_truth_evaluable"]),
        "paired_degenerate_studies": int(np.sum(np.ptp(delta, axis=1) <= 1e-12))}


def _passes(row: dict, config: dict) -> bool:
    coverage_lower = wilson_interval(row["primary_coverage_successes"], row["replicates"], config["mc_confidence_level"])[0]
    width_lower = wilson_interval(row["primary_halfwidth_successes"], row["replicates"], config["mc_confidence_level"])[0]
    return (coverage_lower >= config["coverage_floor"] and width_lower >= config["width_probability_floor"]
            and row["primary_halfwidth_p90"] <= config["half_width_target"]
            and row["primary_degenerate_studies"] == 0)


def run_precision_plan(config: dict, *, progress: Callable[[str], None] | None = None) -> dict:
    validate_precision_config(config)
    config = copy.deepcopy(config)
    emit = progress or (lambda _: None)
    screening, validation = [], []
    for n in config["candidate_n"]:
        emit(f"screening N={n}")
        screening.extend(_cell(config, s, n, stage="screening") for s in config["scenarios"])
    screened = [n for n in config["candidate_n"] if all(_passes(r, config) for r in screening if r["target_n"] == n)]
    selected = None
    if screened:
        for n in [x for x in config["candidate_n"] if x >= min(screened)]:
            emit(f"validation N={n}, {config['validation_replicates']} independent studies per scenario")
            current = []
            for s in config["scenarios"]:
                emit(f"validation N={n} scenario={s['id']}")
                current.append(_cell(config, s, n, stage="validation"))
            validation.extend(current)
            if all(_passes(r, config) for r in current):
                selected = n
                break
    floor = min(scenario_truth(s)["retention"] for s in config["scenarios"])
    cap = recruitment_cap(selected, floor, config["completion_assurance"]) if selected else None
    sensitivity = []
    if selected:
        for retention in sorted(set(config["retention_sensitivity"] + [floor]), reverse=True):
            needed = recruitment_cap(selected, retention, config["completion_assurance"])
            sensitivity.append({"retention": retention, "cap_for_assurance": needed,
                "completion_probability_at_selected_cap": float(binom.sf(selected - 1, cap, retention)),
                "selected_cap_meets_assurance": needed <= cap})
    result = {"schema_version": REPORT_VERSION,
        "status": "smoke_only" if config["purpose"] == "smoke_test" else ("validated_design" if selected else "no_feasible_design"),
        "validation_pass": bool(selected is not None and config["purpose"] == "design_validation"),
        "config": config, "config_sha256": _digest(config),
        "engine_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "environment": {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__},
        "selected_target_n": selected, "recruitment_cap": cap, "design_retention_floor": floor,
        "screening": screening, "validation": validation, "recruitment_sensitivity": sensitivity,
        "limitations": [
            "Finite prospective stress scenarios are assumptions, not estimates from the historical five participants.",
            "Coverage and width intervals are pointwise Monte Carlo uncertainty, not a universal or simultaneous coverage proof.",
            "Precision conditions on attaining target evaluable N; recruitment completion probability is a separate binomial calculation.",
            "Informative QC changes the evaluable estimand; a participant bootstrap does not remove selection bias for all recruited people.",
            "Intervals condition on one fixed training set and model; they do not include model-development or site uncertainty.",
            "Primary adequacy drives N selection; paired-delta precision is reported and is not a guaranteed detection of delta_min.",
        ]}
    result["report_sha256"] = _digest(result)
    return result


def validate_precision_report(report: dict, config: dict | None = None) -> dict:
    """Check a completed design report, including counts/gates, not a passed flag."""
    if report.get("schema_version") != REPORT_VERSION or report.get("status") != "validated_design" or report.get("validation_pass") is not True:
        raise ValueError("Only a completed design-validation report can support a protocol lock")
    payload = {k: v for k, v in report.items() if k != "report_sha256"}
    if report.get("report_sha256") != _digest(payload):
        raise ValueError("Precision report digest mismatch")
    embedded = report["config"]
    validate_precision_config(embedded)
    if embedded["purpose"] != "design_validation":
        raise ValueError("Smoke-only precision evidence cannot authorize a protocol lock")
    if report.get("config_sha256") != _digest(embedded) or (config is not None and _canonical(config) != _canonical(embedded)):
        raise ValueError("Precision config mismatch")
    if report.get("engine_sha256") != hashlib.sha256(Path(__file__).read_bytes()).hexdigest():
        raise ValueError("Precision engine changed; regenerate the report")
    n = report["selected_target_n"]
    if n not in embedded["candidate_n"]:
        raise ValueError("Selected N is not a declared candidate")
    scenarios = {s["id"]: s for s in embedded["scenarios"]}
    for stage in ("screening", "validation"):
        rows = report[stage]
        keys = [(r["target_n"], r["scenario_id"]) for r in rows]
        if len(set(keys)) != len(keys):
            raise ValueError("Duplicate precision cells")
        expected_b = embedded["bootstrap_resamples"] if stage == "validation" else embedded["screening_bootstrap_resamples"]
        for row in rows:
            if row["stage"] != stage or row["scenario_id"] not in scenarios or row["target_n"] not in embedded["candidate_n"]:
                raise ValueError("Unknown precision cell")
            if row["replicates"] != embedded[f"{stage}_replicates"] or row["bootstrap_resamples"] != expected_b:
                raise ValueError("Precision simulation counts do not match config")
            for key, value in scenario_truth(scenarios[row["scenario_id"]]).items():
                if not np.isclose(row[key], value, rtol=0, atol=1e-12):
                    raise ValueError("Precision estimand truth mismatch")
            for prefix, successes_key in (("primary_coverage", "primary_coverage_successes"), ("primary_width_pass", "primary_halfwidth_successes"), ("paired_coverage", "paired_coverage_successes")):
                count = _integer(row[successes_key], successes_key, 0)
                lo, hi = wilson_interval(count, row["replicates"], embedded["mc_confidence_level"])
                if not np.isclose(row[prefix + "_mc_lower"], lo, rtol=0, atol=1e-12) or not np.isclose(row[prefix + "_mc_upper"], hi, rtol=0, atol=1e-12):
                    raise ValueError("Monte Carlo interval disagrees with its success count")
                rate_key = "primary_width_pass_rate" if prefix == "primary_width_pass" else prefix
                if not np.isclose(row[rate_key], count / row["replicates"], rtol=0, atol=1e-12):
                    raise ValueError("Monte Carlo rate disagrees with its success count")
            for key in ("primary_halfwidth_mean", "primary_halfwidth_p90", "paired_halfwidth_mean", "paired_halfwidth_p90"):
                _finite(row[key], key, 0, 1)
            _integer(row["primary_degenerate_studies"], "primary_degenerate_studies", 0)
        ns = embedded["candidate_n"] if stage == "screening" else sorted({r["target_n"] for r in rows})
        if set(keys) != {(candidate, scenario) for candidate in ns for scenario in scenarios}:
            raise ValueError("Missing precision scenario/candidate cells")
    screened = [candidate for candidate in embedded["candidate_n"] if all(_passes(r, embedded) for r in report["screening"] if r["target_n"] == candidate)]
    attempted = sorted({r["target_n"] for r in report["validation"]})
    if not screened or attempted != [candidate for candidate in embedded["candidate_n"] if min(screened) <= candidate <= n]:
        raise ValueError("Selected design does not follow the prespecified screening/validation sequence")
    for candidate in attempted:
        passed = all(_passes(r, embedded) for r in report["validation"] if r["target_n"] == candidate)
        if passed != (candidate == n):
            raise ValueError("Selected design does not match recomputed precision gates")
    floor = min(scenario_truth(s)["retention"] for s in scenarios.values())
    cap = recruitment_cap(n, floor, embedded["completion_assurance"])
    if report["recruitment_cap"] != cap or report["design_retention_floor"] != floor:
        raise ValueError("Recruitment cap/retention does not match exact binomial assurance")
    return {"selected_target_n": n, "recruitment_cap": cap, "confidence_level": embedded["confidence_level"],
        "ci_method": embedded["ci_method"], "bootstrap_resamples": embedded["bootstrap_resamples"],
        "validation_replicates": embedded["validation_replicates"], "seed": embedded["seed"],
        "half_width_target": embedded["half_width_target"], "design_retention_floor": floor,
        "coverage_floor": embedded["coverage_floor"], "width_probability_floor": embedded["width_probability_floor"],
        "config_sha256": report["config_sha256"], "report_sha256": report["report_sha256"],
        "engine_sha256": report["engine_sha256"], "completion_assurance": embedded["completion_assurance"]}
