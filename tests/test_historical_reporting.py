"""Reproduce frozen numbers and reject plausible-looking but incomplete reports."""
from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd
import pytest

from eeg_ma.historical_reporting import (FINAL_EXPERIMENTS, HISTORICAL_EXPERIMENTS, SOURCE_ROOTS,
    rebuild_historical_results, read_historical_report, validate_historical_report,
    verify_historical_report, write_historical_report)
from eeg_ma.provenance import fingerprint, read_csv_literal, sha256_file
from eeg_ma.result_contract import ResultContractError


ROOT = Path(__file__).resolve().parents[1]
MAPPING = ROOT / "configs/legacy_discovery_mapping.json"


@pytest.fixture(scope="module")
def complete_report():
    return rebuild_historical_results(ROOT / "results", legacy_manifest=MAPPING, allow_legacy=True)


@pytest.fixture(scope="module")
def figures():
    spec = importlib.util.spec_from_file_location("strict_final_figures", ROOT / "scripts/13_make_final_figures.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _partial_sources(tmp_path):
    destination = tmp_path / "archive"
    for name in ("poster_baseline", "plv_alpha_only"):
        relative, _ = SOURCE_ROOTS[name]
        source = ROOT / "results" / relative
        target = destination / relative
        (target / "inter").mkdir(parents=True, exist_ok=True)
        for filename in ("inter_predictions.csv", "inter_folds.csv", "inter_feature_frequency.csv"):
            shutil.copyfile(source / "inter" / filename, target / "inter" / filename)
        if (source / "config_snapshot.json").exists():
            shutil.copyfile(source / "config_snapshot.json", target / "config_snapshot.json")
    return destination


def _partial(root):
    return rebuild_historical_results(root, legacy_manifest=MAPPING, allow_legacy=True,
        experiments=["poster_baseline", "plv_alpha_only"], partial=True)


def test_all_frozen_final_numbers_are_reproduced_from_epoch_predictions(complete_report):
    report = complete_report
    assert report.metadata["study_matrix_complete"] is True
    assert len(report.predictions) == 19800
    assert len(report.participants) == 330
    assert len(report.summary) == len(HISTORICAL_EXPERIMENTS) * 2 * 3
    assert set(report.summary.n_subjects) == {5}
    assert set(report.participants.n_class_0) == set(report.participants.n_class_1) == {30}
    archived = read_csv_literal(ROOT / "results/final_validation/final_validation_summary.csv")
    current = report.summary.loc[report.summary.experiment.isin(FINAL_EXPERIMENTS)]
    keys = ["experiment", "comparison", "classifier"]
    old = archived.set_index(keys).sort_index()
    new = current.set_index(keys).sort_index()
    assert old.index.equals(new.index)
    np.testing.assert_allclose(new.accuracy_mean, old.accuracy_mean, atol=1e-14)
    np.testing.assert_allclose(new.accuracy_std, old.accuracy_std, atol=1e-14)
    np.testing.assert_allclose(new.balanced_accuracy_mean, old.accuracy_mean, atol=1e-14)
    # Absolute checks independent of the reconstruction formulas.
    assert new.loc[("poster_baseline", "rest1_vs_type1", "LDA"), "accuracy_mean"] == pytest.approx(157/300)
    assert new.loc[("plv_alpha_only", "rest1_vs_type1", "LDA"), "accuracy_mean"] == pytest.approx(223/300)
    assert new.loc[("iplv_only", "rest1_vs_type1", "LDA"), "accuracy_mean"] == pytest.approx(134/300)
    assert report.metadata["historical_provenance"] == "legacy_unverified"
    assert not any("ci_low" in column or "ci_high" in column for column in current)
    for parent in report.metadata["parent_runs"].values():
        assert parent["historical_code"] == parent["historical_environment"] == parent["raw_hashes"] == "unknown"
        assert parent["run_id"].startswith("legacy-unknown-run-")
    validate_historical_report(report)


def test_reconstruction_used_the_exact_frozen_inventory_bytes(complete_report):
    original = json.loads((ROOT / "docs/manifests/legacy_discovery_inventory.json").read_text())
    for label, location in complete_report.metadata["source_locations"].items():
        path = Path(location)
        if path.is_relative_to(ROOT / "results"):
            expected = original["inventories"]["results"]["observed_files"][path.relative_to(ROOT / "results").as_posix()]["sha256"]
            assert complete_report.metadata["inputs"][label]["sha256"] == expected == sha256_file(path)


@pytest.mark.parametrize("mutation", ["missing_epoch", "duplicate_epoch", "wrong_subject", "wrong_cohort", "wrong_label", "missing_classifier"])
def test_historical_predictions_fail_closed_at_identity_and_coverage_boundaries(tmp_path, mutation):
    root = _partial_sources(tmp_path)
    path = root / "final_validation/plv_alpha_only/inter/inter_predictions.csv"
    frame = read_csv_literal(path)
    if mutation == "missing_epoch":
        frame = frame.iloc[1:]
    elif mutation == "duplicate_epoch":
        frame = pd.concat([frame, frame.iloc[[0]]], ignore_index=True)
    elif mutation == "wrong_subject":
        frame.loc[0, "subject_id"] = "unrelated-participant"
    elif mutation == "wrong_cohort":
        frame["cohort_id"] = "unrelated-cohort"
    elif mutation == "wrong_label":
        frame.loc[0, "true_label"] = 1 - frame.loc[0, "true_label"]
    else:
        frame = frame.loc[frame.classifier.ne("KFDA")]
    frame.to_csv(path, index=False)
    with pytest.raises(ResultContractError):
        _partial(root)


def test_partial_scope_is_explicit_labelled_and_cannot_replace_any_report(tmp_path):
    root = _partial_sources(tmp_path)
    with pytest.raises(ResultContractError, match="partial"):
        rebuild_historical_results(root, legacy_manifest=MAPPING, allow_legacy=True,
            experiments=["poster_baseline", "plv_alpha_only"])
    report = _partial(root)
    destination = tmp_path / "partial-report"
    write_historical_report(report, destination)
    before = sha256_file(destination / "historical_report.json")
    assert verify_historical_report(destination)["scope"] == "partial_historical_matrix"
    assert len(read_historical_report(destination, verify_sources=False).summary) == 12
    with pytest.raises(FileExistsError, match="already exists"):
        write_historical_report(report, destination)
    assert sha256_file(destination / "historical_report.json") == before
    with pytest.raises(ValueError, match="Protected"):
        write_historical_report(report, ROOT / "results/forbidden-report")


@pytest.mark.parametrize("mutation", ["n_subjects", "protocol_id", "scope", "historical_provenance", "uncertainty"])
def test_rehashed_forged_metadata_still_fails_semantic_validation(tmp_path, mutation):
    report = _partial(_partial_sources(tmp_path))
    destination = tmp_path / "report"
    write_historical_report(report, destination)
    path = destination / "historical_report.json"
    metadata = json.loads(path.read_text())
    metadata[mutation] = {"n_subjects": 900, "protocol_id": "different-protocol",
        "scope": "complete_historical_matrix", "historical_provenance": "verified_managed_runs", "uncertainty": "95% confidence interval"}[mutation]
    metadata["report_id"] = fingerprint({k: v for k, v in metadata.items() if k != "report_id"})
    path.write_text(json.dumps(metadata))
    with pytest.raises(ResultContractError):
        read_historical_report(destination, verify_sources=False)


def test_changed_numeric_artifact_and_rehashed_summary_are_rejected(tmp_path):
    report = _partial(_partial_sources(tmp_path))
    destination = tmp_path / "report"
    write_historical_report(report, destination)
    csv_path = destination / "historical_summary.csv"
    frame = read_csv_literal(csv_path)
    frame.loc[0, "accuracy_mean"] = 0.99
    frame.to_csv(csv_path, index=False)
    with pytest.raises(ResultContractError, match="artifacts"):
        read_historical_report(destination, verify_sources=False)
    manifest_path = destination / "historical_report.json"
    metadata = json.loads(manifest_path.read_text())
    metadata["artifacts"][csv_path.name] = {"sha256": sha256_file(csv_path), "bytes": csv_path.stat().st_size}
    metadata["report_id"] = fingerprint({k: v for k, v in metadata.items() if k != "report_id"})
    manifest_path.write_text(json.dumps(metadata))
    with pytest.raises(ResultContractError, match="prediction-derived"):
        read_historical_report(destination, verify_sources=False)


def test_recorded_original_sources_are_checked_when_verifying(tmp_path):
    root = _partial_sources(tmp_path)
    report = _partial(root)
    destination = tmp_path / "report"
    write_historical_report(report, destination)
    source = root / "inter/inter_predictions.csv"
    source.write_bytes(source.read_bytes() + b"\n")
    with pytest.raises(ResultContractError, match="source bytes changed"):
        verify_historical_report(destination)


@pytest.mark.parametrize("mutation", ["missing_cell", "duplicate_cell", "wrong_cohort", "nonfinite", "different_n"])
def test_direct_plot_helpers_reject_ambiguous_or_incomplete_matrix_inputs(complete_report, figures, tmp_path, mutation):
    frame = complete_report.summary.copy()
    selected = frame.index[frame.experiment.eq("plv_alpha_only") & frame.comparison.eq("rest1_vs_type1")][0]
    if mutation == "missing_cell":
        frame = frame.drop(selected)
    elif mutation == "duplicate_cell":
        frame = pd.concat([frame, frame.loc[[selected]]], ignore_index=True)
    elif mutation == "wrong_cohort":
        frame.loc[selected, "cohort_id"] = "wrong-cohort"
    elif mutation == "nonfinite":
        frame.loc[selected, "accuracy_mean"] = np.nan
    else:
        frame.loc[selected, "n_subjects"] = 4
    with pytest.raises(ResultContractError):
        figures.plot_band_ranking(frame, tmp_path)
    assert not list(tmp_path.glob("*.png"))


def test_plotted_heights_errors_and_saved_sources_are_prediction_derived(complete_report, figures, tmp_path, monkeypatch):
    observed = []
    original_bar = figures.plt.Axes.bar
    def capture(self, x, height, *args, **kwargs):
        observed.append((np.asarray(height, dtype=float), np.asarray(kwargs["yerr"], dtype=float)))
        return original_bar(self, x, height, *args, **kwargs)
    monkeypatch.setattr(figures.plt.Axes, "bar", capture)
    paths = figures.plot_phase_robustness(complete_report.summary, tmp_path, metadata=complete_report.metadata)
    assert len(paths) == 2 and all(path.stat().st_size > 1000 for path in paths)
    experiments = ["poster_baseline", "plv_all_bands", "plv_alpha_only", "iplv_only"]
    for (heights, errors), classifier in zip(observed, figures.CLASSIFIERS):
        # Independently use raw binary outcomes, participant by participant.
        means, stds = [], []
        for experiment in experiments:
            raw = complete_report.predictions.loc[complete_report.predictions.experiment.eq(experiment)
                & complete_report.predictions.comparison.eq("rest1_vs_type1") & complete_report.predictions.classifier.eq(classifier)]
            scores = [sum(part.y_true == part.y_pred) / len(part) for _, part in raw.groupby("subject_id")]
            means.append(np.mean(scores))
            stds.append(np.std(scores, ddof=1))
        np.testing.assert_allclose(heights, means, rtol=0, atol=1e-15)
        np.testing.assert_allclose(errors, stds, rtol=0, atol=1e-15)
    sidecar = json.loads((tmp_path / "final_phase_connectivity_validation.json").read_text())
    assert sidecar["n_subjects"] == 5
    assert sidecar["metric"] == "accuracy"
    assert sidecar["source_sha256"] == complete_report.metadata["source_sha256"]
    assert "sample SD" in sidecar["uncertainty"] and "no CI" in sidecar["uncertainty"]
    assert sidecar["numerical_source_sha256"] == sha256_file(tmp_path / sidecar["numerical_source"])
    before = sha256_file(paths[0])
    with pytest.raises(FileExistsError, match="already exist"):
        figures.plot_phase_robustness(complete_report.summary, tmp_path, metadata=complete_report.metadata)
    assert sha256_file(paths[0]) == before
    protected = ROOT / "results/final_figures/final_phase_connectivity_validation.png"
    before = sha256_file(protected)
    with pytest.raises(ValueError, match="Protected"):
        figures.plot_phase_robustness(complete_report.summary, protected.parent, metadata=complete_report.metadata)
    assert sha256_file(protected) == before


def test_legacy_aggregate_summary_is_not_a_numerical_source(tmp_path):
    root = _partial_sources(tmp_path)
    # A plausible but wrong old aggregate must neither influence nor authorize
    # the reconstructed result. Prediction rows and their coverage are decisive.
    (root / "final_validation/final_validation_summary.csv").write_text("experiment,accuracy_mean\nplv_alpha_only,0.999\n")
    report = _partial(root)
    rows = report.summary.loc[report.summary.experiment.eq("plv_alpha_only") & report.summary.comparison.eq("rest1_vs_type1")]
    assert max(rows.accuracy_mean) < 0.75
    assert all("final_validation_summary.csv" not in value for value in report.metadata["source_locations"].values())


def test_missing_eligible_class_cannot_be_hidden_in_a_balanced_accuracy_figure(complete_report, figures, tmp_path):
    report = copy.deepcopy(complete_report)
    report.summary.loc[report.summary.experiment.eq("plv_alpha_only"), "n_ba_eligible"] = 4
    with pytest.raises(ResultContractError, match="both classes"):
        figures.plot_band_ranking(report.summary, tmp_path, metric="balanced_accuracy")


def test_observed_wrong_representation_is_not_relabelled_by_its_directory(tmp_path):
    root = _partial_sources(tmp_path)
    config = root / "final_validation/plv_alpha_only/config_snapshot.json"
    config.write_bytes((ROOT / "results/final_validation/plv_delta_only/config_snapshot.json").read_bytes())
    with pytest.raises(ResultContractError, match="feature_set|experiment name"):
        _partial(root)


def test_alpha_frequency_must_follow_the_bound_training_selection_evidence(complete_report, tmp_path):
    report = copy.deepcopy(complete_report)
    report.alpha_frequency.loc[0, "selection_count"] += 1
    with pytest.raises(ResultContractError, match="Alpha feature frequency"):
        validate_historical_report(report)
    with pytest.raises(ResultContractError, match="Alpha feature frequency"):
        write_historical_report(report, tmp_path / "forged-alpha")
    assert not (tmp_path / "forged-alpha").exists()


def test_an_undeclared_alpha_table_cannot_enter_a_baseline_only_report(tmp_path):
    report = rebuild_historical_results(ROOT / "results", legacy_manifest=MAPPING, allow_legacy=True,
        experiments=["poster_baseline"], partial=True)
    report.alpha_frequency = pd.DataFrame({"comparison": ["rest1_vs_type1"], "feature": ["PLV__FP1__FP2__alpha"], "selection_count": [1]})
    with pytest.raises(ResultContractError, match="undeclared alpha"):
        validate_historical_report(report)


def test_alpha_subject_figure_cannot_drop_one_person_across_all_classifiers(complete_report, figures, tmp_path):
    deltas = complete_report.paired_deltas.loc[complete_report.paired_deltas.subject_id.ne("irene")]
    with pytest.raises(ResultContractError, match="matrix"):
        figures.plot_alpha_subjects(deltas, tmp_path, metadata=complete_report.metadata)


def test_figure_caption_n_and_subject_roster_must_describe_the_plotted_values(complete_report, figures, tmp_path):
    metadata = copy.deepcopy(complete_report.metadata)
    metadata["n_subjects_by_comparison"]["rest1_vs_type1"] = 999
    with pytest.raises(ResultContractError, match="N contradicts"):
        figures.plot_phase_robustness(complete_report.summary, tmp_path, metadata=metadata)
    assert not list(tmp_path.glob("*.png"))


def test_complete_figure_bundle_is_sealed_after_all_figures_and_verifies(complete_report, figures, tmp_path, monkeypatch):
    import sys
    monkeypatch.setattr(figures, "rebuild_historical_results", lambda *args, **kwargs: complete_report)
    monkeypatch.setattr(sys, "argv", ["figures", "--out-dir", str(tmp_path / "figures")])
    assert figures.main() == 0
    destination = tmp_path / "figures"
    assert len(list(destination.glob("*.png"))) == len(list(destination.glob("*.svg"))) == 5
    verified = verify_historical_report(destination, verify_sources=False)
    assert verified["scope"] == "complete_historical_matrix"
    metadata = json.loads((destination / "historical_report.json").read_text())
    assert "final_story_type1_inter.png" in metadata["artifacts"]
    assert "figure_sources.json" in metadata["artifacts"]


def test_source_changes_after_an_earlier_experiment_are_detected_before_publication(tmp_path, monkeypatch):
    from eeg_ma import historical_reporting
    root = _partial_sources(tmp_path)
    original = historical_reporting._adapt_source
    def mutate_after_read(source, experiment, *args, **kwargs):
        result = original(source, experiment, *args, **kwargs)
        if experiment == "plv_alpha_only":
            baseline = root / "inter/inter_predictions.csv"
            baseline.write_bytes(baseline.read_bytes() + b"\n")
        return result
    monkeypatch.setattr(historical_reporting, "_adapt_source", mutate_after_read)
    with pytest.raises(ResultContractError, match="Source bytes changed"):
        _partial(root)
