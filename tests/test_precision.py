from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
from scipy.stats import binom

from eeg_ma.precision import (
    _bootstrap_intervals, _cell, _simulated_scores, participant_bootstrap,
    recruitment_cap, run_precision_plan, scenario_truth, validate_precision_config,
    validate_precision_report, wilson_interval,
)

ROOT = Path(__file__).resolve().parents[1]


def config():
    return json.loads((ROOT / "configs/precision_plan.json").read_text())


def test_paired_bootstrap_uses_one_participant_weight_vector():
    scores = np.array([.25, .50, .75, .9])
    reference = scores - .1
    result = participant_bootstrap(scores, reference, seed=91, n_resamples=500,
                                   subject_ids=["001", "1", "NA", "null"])
    assert result == participant_bootstrap(scores, reference, seed=91, n_resamples=500)
    assert result["primary"]["ci_upper"] > result["primary"]["ci_lower"]
    assert result["paired_delta"]["estimate"] == pytest.approx(.1)
    assert result["paired_delta"]["ci_lower"] == pytest.approx(.1)
    assert result["paired_delta"]["ci_upper"] == pytest.approx(.1)


def test_bootstrap_indices_match_independent_hand_calculation():
    scores = np.array([.1, .5, .8, .9])
    ref = np.array([.2, .4, .3, .7])
    seed, b = 717, 300
    indices = np.random.default_rng(seed).integers(0, 4, size=(1, b, 4), dtype=np.int32)
    expected = np.quantile(scores[indices].mean(axis=2)[0], [.025, .975])
    expected_delta = np.quantile((scores - ref)[indices].mean(axis=2)[0], [.025, .975])
    result = participant_bootstrap(scores, ref, seed=seed, n_resamples=b)
    np.testing.assert_array_equal(expected, [result["primary"]["ci_lower"], result["primary"]["ci_upper"]])
    np.testing.assert_array_equal(expected_delta, [result["paired_delta"]["ci_lower"], result["paired_delta"]["ci_upper"]])


@pytest.mark.parametrize("scores", [[.5], [.5, np.nan], [.5, np.inf], [-.1, .5], [.5, 1.1], [[.5], [.6]]])
def test_missing_or_invalid_scores_cannot_be_silently_dropped(scores):
    with pytest.raises(ValueError):
        participant_bootstrap(scores, seed=1)


def test_degenerate_or_duplicate_participants_are_not_confirmatory():
    result = participant_bootstrap([.8, .8, .8], seed=3, n_resamples=100)
    assert result["primary"]["degenerate"]
    assert not result["primary"]["confirmatory_eligible"]
    with pytest.raises(ValueError, match="unique literal"):
        participant_bootstrap([.6, .7], seed=1, subject_ids=["a", "a"])
    with pytest.raises(ValueError, match="same participants"):
        participant_bootstrap([.6, .7], [.5, .5, .5], seed=1)


def test_recruitment_cap_is_tail_assured_and_minimal():
    cap = recruitment_cap(60, .7125, .95)
    assert cap == 94
    assert binom.sf(59, cap, .7125) >= .95
    assert binom.sf(59, cap - 1, .7125) < .95
    assert cap > int(np.ceil(60 / .7125))
    assert recruitment_cap(60, 1, .95) == 60


def test_informative_loss_truth_distinguishes_target_populations():
    scenario = next(s for s in config()["scenarios"] if s["id"] == "informative-qc-moderate")
    truth = scenario_truth(scenario)
    assert truth["primary_truth_recruited"] == .65
    assert truth["primary_truth_evaluable"] == pytest.approx(.65 + .5 * .2 ** 2 / .75)
    primary, reference = _simulated_scores(scenario, 1000, 80, np.random.default_rng(8))
    assert abs(primary.mean() - truth["primary_truth_evaluable"]) < .005
    assert abs((primary - reference).mean() - truth["paired_truth_evaluable"]) < .005
    assert truth["retention"] == pytest.approx(.7125)


def test_epoch_dependence_increases_uncertainty_without_changing_participant_n():
    scenario = copy.deepcopy(config()["scenarios"][7])
    scenario["epoch_icc"] = 0
    independent, _ = _simulated_scores(scenario, 1000, 40, np.random.default_rng(11))
    scenario["epoch_icc"] = .3
    correlated, _ = _simulated_scores(scenario, 1000, 40, np.random.default_rng(12))
    assert independent.shape == correlated.shape == (1000, 40)
    assert correlated.var() > independent.var() * 1.5


def test_retention_cannot_be_clipped_while_claiming_linear_truth():
    cfg = config()
    cfg["scenarios"][0]["informative_retention_slope"] = .9
    with pytest.raises(ValueError, match="entire support"):
        validate_precision_config(cfg)


def test_smoke_simulation_is_deterministic_and_never_lock_evidence():
    cfg = config()
    cfg.update(purpose="smoke_test", candidate_n=[20], screening_replicates=30,
               validation_replicates=30, screening_bootstrap_resamples=50, bootstrap_resamples=50,
               half_width_target=.4, coverage_floor=.5, width_probability_floor=.5)
    cfg["scenarios"] = cfg["scenarios"][:1]
    a, b = run_precision_plan(cfg), run_precision_plan(cfg)
    assert a == b
    assert a["status"] == "smoke_only"
    with pytest.raises(ValueError, match="design-validation"):
        validate_precision_report(a)


def test_monte_carlo_interval_includes_sampling_uncertainty():
    low, high = wilson_interval(950, 1000)
    assert .93 < low < .95 < high < .97
    assert wilson_interval(1000, 1000)[0] < 1


def test_committed_design_report_is_validated_and_forged_pass_flag_is_rejected():
    path = ROOT / "docs/precision/precision_report.json"
    report = json.loads(path.read_text())
    summary = validate_precision_report(report, config())
    assert summary["bootstrap_resamples"] == 10_000
    assert summary["validation_replicates"] >= 1000
    altered = copy.deepcopy(report)
    altered["validation"][0]["primary_coverage_successes"] = 0
    altered["validation_pass"] = True
    payload = {k: v for k, v in altered.items() if k != "report_sha256"}
    altered["report_sha256"] = hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True,
        separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    with pytest.raises(ValueError, match="Monte Carlo"):
        validate_precision_report(altered)
