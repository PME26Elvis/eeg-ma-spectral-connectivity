"""Report inference is conditional on fixed models and complete synthetic archives.

All cohorts and acquisition attestations here are explicitly synthetic. These
tests exercise the report implementation, never supply real study evidence.
"""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import runpy
import shutil
import sys

import numpy as np
import pandas as pd
import pytest

from eeg_ma import external_reporting as reporting
from eeg_ma.provenance import fingerprint, read_csv_literal, sha256_file
from eeg_ma.result_contract import ResultContractError, read_result_csv
from fixed_model_fixtures import write_json
from test_external_workflow import (external_base, external_case, _forbid_fit,
                                    _rewrite_labels, _score)


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def report_base(external_base, tmp_path_factory):
    root = Path(shutil.copytree(external_base["root"], tmp_path_factory.mktemp("external-report") / "case"))
    case = {**external_base, "root": root}
    patches = pytest.MonkeyPatch()
    try:
        _forbid_fit(patches)
        _score(case)
        manifest = reporting.build_external_report(root / "score", root / "report",
            prediction_dir=root / "predictions", lock_dir=root / "lock")
    finally:
        patches.undo()
    return {**case, "manifest": manifest}


@pytest.fixture
def report_case(report_base, tmp_path):
    root = Path(shutil.copytree(report_base["root"], tmp_path / "case"))
    return {**report_base, "root": root}


def _rehash_report(root):
    path = root / "report_manifest.json"
    manifest = json.loads(path.read_text())
    manifest["artifacts"] = reporting._inventory(root, ignore=("report_manifest.json",))
    manifest.pop("report_id")
    manifest["report_id"] = fingerprint(manifest)
    write_json(path, manifest)


def test_report_is_complete_roster_equal_participant_and_self_contained(report_case, monkeypatch):
    root = report_case["root"]
    _forbid_fit(monkeypatch)
    manifest = reporting.validate_external_report(root / "report")
    summary = json.loads((root / "report/summary.json").read_text())
    participants = read_result_csv(root / "report/participant_results.csv")
    assert manifest["status"] == "complete"
    assert manifest["inference_scope"] == "descriptive_target_shortfall"
    assert manifest["score_id"] == summary["score_id"]
    assert len(participants) == 94 * 4
    assert participants.subject_id.nunique() == 94
    assert len(summary["endpoints"]) == 4 and len(summary["paired_deltas"]) == 2
    assert summary["cohort_flow"]["n_recruited"] == 94
    assert summary["cohort_flow"]["n_evaluable"] == 3
    assert summary["cohort_flow"]["n_without_recorded_session"] == 91
    assert summary["cohort_flow"]["dispositions"] == {"completed": 3, "lost_to_followup": 91}
    assert participants.loc[~participants.evaluable, "balanced_accuracy"].isna().all()
    assert participants.loc[~participants.evaluable, "accuracy"].isna().all()
    assert participants.loc[~participants.evaluable, "ba_status"].eq("missing_both_classes").all()
    assert not summary["interpretation"]["formal_primary_claim_eligible"]
    assert not summary["interpretation"]["precision_target_met"]
    for row in summary["endpoints"]:
        table = participants.loc[(participants.comparison == row["comparison"]) &
                                 (participants.model_role == row["model_role"]) & participants.evaluable]
        assert row["n_evaluable"] == len(table) == 3
        for metric in reporting.METRICS:
            assert row["metrics"][metric]["estimate"] == pytest.approx(table[metric].mean())
    for row in summary["pooled_epoch_metrics"]:
        assert row["is_primary_estimand"] is False and row["confidence_interval"] is None
    assert "model fitting" in (root / "report/report.md").read_text()
    assert summary["inference"]["model_fitting_uncertainty_included"] is False
    assert summary["inference"]["resamples"] == 10_000
    assert summary["inference"]["seed"] == 20261109
    for name in ("score", "predictions", "lock", "models", "development", "prepared"):
        shutil.rmtree(root / name)
    assert reporting.validate_external_report(root / "report") == manifest


def test_bootstrap_limits_and_paired_indices_match_manual_locked_resampling():
    spec = json.loads((ROOT / "configs/confirmatory_protocol.json").read_text())
    primary = pd.DataFrame({"balanced_accuracy": [.60, .85, .95, .45], "accuracy": [.55, .80, .90, .40],
                            "recall_0": [.4, .9, 1., .6], "recall_1": [.8, .8, .9, .3]})
    reference = pd.DataFrame({"balanced_accuracy": [.65, .55, .90, .50], "accuracy": [.6, .5, .85, .45],
                              "recall_0": [.5, .6, .8, .5], "recall_1": [.8, .5, 1., .5]})
    ids = ["001", "1", "NA", "null"]
    results, paired, resampling = reporting._bootstrap_metrics(primary, reference, spec=spec, subject_ids=ids)
    indices = np.random.default_rng(20261109).integers(0, 4, size=(1, 10_000, 4), dtype=np.int32)[0]
    for metric in reporting.METRICS:
        for role, values in (("primary", primary[metric].to_numpy()), ("reference", reference[metric].to_numpy())):
            expected = np.quantile(values[indices].mean(axis=1), [.025, .975], method="linear")
            assert results[role][metric]["estimate"] == pytest.approx(values.mean())
            np.testing.assert_allclose([results[role][metric]["ci_lower"], results[role][metric]["ci_upper"]], expected, rtol=0, atol=1e-15)
        if metric in ("balanced_accuracy", "accuracy"):
            difference = (primary[metric].to_numpy() - reference[metric].to_numpy())[indices].mean(axis=1)
            expected = np.quantile(difference, [.025, .975], method="linear")
            np.testing.assert_allclose([paired[metric]["ci_lower"], paired[metric]["ci_upper"]], expected, rtol=0, atol=1e-15)
            assert paired[metric]["confirmatory_eligible"] is False
    assert resampling["participant_ids_in_order"] == ids
    assert resampling["indices_sha256"] == reporting._sha(indices.astype("<i4").tobytes())
    assert resampling["shared_across_models_and_metrics"] is True


@pytest.mark.parametrize("fault", ["nan", "duplicate_ids", "short_reference", "out_of_range"])
def test_bootstrap_never_drops_missing_or_unpaired_participants(fault):
    spec = json.loads((ROOT / "configs/confirmatory_protocol.json").read_text())
    primary = pd.DataFrame({metric: [.6, .8, .9] for metric in reporting.METRICS})
    reference = primary.copy()
    ids = ["a", "b", "c"]
    if fault == "nan":
        reference.loc[1, "balanced_accuracy"] = np.nan
    elif fault == "duplicate_ids":
        ids[-1] = ids[0]
    elif fault == "short_reference":
        reference = reference.iloc[:2]
    else:
        primary.loc[0, "recall_0"] = 1.1
    with pytest.raises(reporting.ExternalReportError):
        reporting._bootstrap_metrics(primary, reference, spec=spec, subject_ids=ids)


@pytest.mark.parametrize("mode,n,status,scope", [
    ("confirmatory", 60, "estimated", "confirmatory_primary"),
    ("pilot_estimation", 60, "estimated", "descriptive_pilot_only"),
    ("confirmatory", 59, "estimated", "descriptive_target_shortfall"),
    ("confirmatory", 60, "degenerate", "confirmatory_primary"),
    ("confirmatory", 1, "insufficient_participants", "descriptive_target_shortfall"),
    ("confirmatory", 0, "not_estimable", "descriptive_target_shortfall"),
])
def test_favorable_results_cannot_bypass_protocol_scope_guards(mode, n, status, scope):
    spec = json.loads((ROOT / "configs/confirmatory_protocol.json").read_text())
    spec["study"]["mode"] = mode
    primary = {"ci_lower": .75, "ci_upper": .85, "status": status}
    delta = {"ci_lower": .10, "ci_upper": .15}
    if status == "degenerate":
        primary["ci_upper"] = primary["ci_lower"]
    if n < 2:
        primary.update(ci_lower=None, ci_upper=None)
        delta.update(ci_lower=None, ci_upper=None)
    result = reporting._interpret(primary, delta, n=n, spec=spec)
    assert result["inference_scope"] == scope
    assert result["formal_primary_claim_eligible"] == (mode == "confirmatory" and n >= 60 and status == "estimated")
    if mode != "confirmatory" or n < 60:
        assert result["above_chance"] == "descriptive_only"
    if status == "degenerate":
        assert result["above_chance"] == "uncertain_degenerate"
        assert not result["precision_target_met"]
    if n < 60:
        assert not result["precision_target_met"]
    assert result["paired_delta_inference"] == "descriptive_nominal_interval_no_confirmatory_claim"


def test_zero_and_one_participant_have_no_invented_bootstrap_interval():
    spec = json.loads((ROOT / "configs/confirmatory_protocol.json").read_text())
    for n in (0, 1):
        frame = pd.DataFrame({metric: [.9] * n for metric in reporting.METRICS})
        results, paired, indices = reporting._bootstrap_metrics(frame, frame.copy(), spec=spec, subject_ids=["x"] * n)
        for row in [results["primary"]["balanced_accuracy"], paired["balanced_accuracy"]]:
            assert row["ci_lower"] is None and row["ci_upper"] is None
            assert not row["confirmatory_eligible"]
        assert indices["indices_sha256"] is None


@pytest.mark.parametrize("fault", ["missing_epoch", "missing_participant", "nan_prediction", "label_disagreement"])
def test_pairing_requires_exact_epochs_people_labels_and_finite_predictions(report_base, fault):
    root = report_base["root"]
    evidence = reporting._load_evidence(root / "score", root / "predictions", root / "lock")
    predictions = evidence["prediction"]["predictions"].copy()
    reference = predictions.model_role == "reference"
    if fault == "missing_epoch":
        predictions = predictions.drop(predictions.loc[reference].index[0])
    elif fault == "missing_participant":
        predictions = predictions.loc[~(reference & (predictions.subject_id == "external-000"))]
    elif fault == "nan_prediction":
        predictions.loc[predictions.loc[reference].index[0], "predicted_label"] = np.nan
    else:
        # Reassigning an epoch to a different subject is also a label/owner identity conflict.
        predictions.loc[predictions.loc[reference].index[0], "subject_id"] = "external-001"
    evidence["prediction"]["predictions"] = predictions
    flow, _, _ = reporting._flow(evidence)
    with pytest.raises((ResultContractError, reporting.ExternalReportError), match="coverage|exact|binary|inconsistent|differs"):
        reporting._canonical_predictions(evidence, flow)


def test_missing_task_class_is_retained_and_secondary_can_remain_evaluable(external_case, monkeypatch):
    from eeg_ma.external import prepare_external_inputs, predict_external, score_external
    root = external_case["root"]
    _forbid_fit(monkeypatch)
    frame = external_case["external"].copy()
    frame = frame.loc[~((frame.subject_id == "external-000") &
                        (frame.comparison == "rest1_vs_type1") & (frame.class_label == 1))]
    frame.to_csv(root / "missing-class-features.csv", index=False)
    enrollment = json.loads((root / "enrollment.json").read_text())
    session = enrollment["participants"][0]["sessions"][0]
    session["attempts"]["Type1"] = 45
    session["correct"]["Type1"] = session["eligible_correct"]["Type1"] = 0
    session["task_end"]["Type1"] = "attempt_limit"
    write_json(root / "enrollment.json", enrollment)
    prepare_external_inputs(root / "missing-class-features.csv", root / "missing-prepared",
        lock_dir=root / "lock", cohort_manifest_path=root / "cohort.json")
    predict_external(root / "missing-prepared/inputs", root / "missing-predictions",
        lock_dir=root / "lock", cohort_manifest_path=root / "cohort.json")
    score_external(root / "missing-predictions", root / "missing-prepared/labels", root / "missing-score",
        lock_dir=root / "lock", cohort_manifest_path=root / "cohort.json", enrollment_manifest_path=root / "enrollment.json")
    reporting.build_external_report(root / "missing-score", root / "missing-report",
        prediction_dir=root / "missing-predictions", lock_dir=root / "lock")
    participants = read_result_csv(root / "missing-report/participant_results.csv")
    missing = participants.loc[(participants.subject_id == "external-000") & (participants.comparison == "rest1_vs_type1")]
    assert len(missing) == 2 and not missing.evaluable.any()
    assert missing.ba_status.eq("missing_class_1").all()
    assert missing.balanced_accuracy.isna().all() and missing.accuracy.isna().all()
    assert missing.n_class_0.eq(30).all() and missing.n_class_1.eq(0).all()
    assert missing.recall_0.notna().all() and missing.recall_1.isna().all()
    summary = json.loads((root / "missing-report/summary.json").read_text())
    assert summary["cohort_flow"]["n_recruited"] == 94
    assert summary["cohort_flow"]["n_evaluable"] == 2
    assert summary["cohort_flow"]["n_secondary_evaluable"] == 3
    assert len(participants) == 94 * 4


def test_unfavorable_results_still_publish_complete_report(external_case, monkeypatch):
    root = external_case["root"]
    _forbid_fit(monkeypatch)
    _rewrite_labels(root / "prepared/labels", lambda rows: rows.assign(true_label=1 - rows.true_label))
    _score(external_case)
    reporting.build_external_report(root / "score", root / "negative-report",
        prediction_dir=root / "predictions", lock_dir=root / "lock")
    summary = json.loads((root / "negative-report/summary.json").read_text())
    assert len(summary["endpoints"]) == 4 and len(summary["paired_deltas"]) == 2
    assert summary["endpoints"][0]["metrics"]["balanced_accuracy"]["estimate"] < .5
    assert summary["publication_policy"].endswith("regardless_of_sign_or_threshold")


@pytest.mark.parametrize("fault", ["unhashed_table", "rehashed_summary", "rehashed_roster", "rehashed_narrative", "rehashed_figure", "source_model", "source_protocol", "unexpected_file"])
def test_report_rejects_hash_mutation_and_rehashed_numerical_or_scope_forgery(report_case, fault):
    root = report_case["root"] / "report"
    if fault == "unhashed_table":
        (root / "participant_results.csv").write_text((root / "participant_results.csv").read_text() + "\n")
    elif fault == "rehashed_summary":
        summary = json.loads((root / "summary.json").read_text())
        summary["endpoints"][0]["metrics"]["balanced_accuracy"]["estimate"] = .123456
        write_json(root / "summary.json", summary)
    elif fault == "rehashed_roster":
        rows = pd.read_csv(root / "participant_flow.csv", dtype=str, keep_default_na=False)
        rows.iloc[:-1].to_csv(root / "participant_flow.csv", index=False)
    elif fault == "rehashed_narrative":
        (root / "report.md").write_text("# The experiment proves online BCI usability\n")
    elif fault == "rehashed_figure":
        rows = pd.read_csv(root / "figure_data.csv")
        rows.loc[0, "n_participants"] = 180
        rows.to_csv(root / "figure_data.csv", index=False)
    elif fault == "source_model":
        path = root / "sources/predictions/models/primary.json"
        path.write_bytes(path.read_bytes() + b"\n")
    elif fault == "source_protocol":
        path = root / "sources/protocol/protocol.json"
        value = json.loads(path.read_text())
        value["recruitment"]["target_evaluable"] = 3
        write_json(path, value)
    else:
        (root / "unreported.csv").write_text("secret_result\n1\n")
    if fault.startswith("rehashed") or fault == "unexpected_file":
        _rehash_report(root)
    with pytest.raises(reporting.ExternalReportError):
        reporting.validate_external_report(root)


def test_report_never_overwrites_or_mutates_source_archives(report_base):
    root = report_base["root"]
    before = {name: reporting._inventory(root / name) for name in ("score", "predictions", "lock", "report")}
    with pytest.raises(reporting.ExternalReportError, match="already exists"):
        reporting.build_external_report(root / "score", root / "report", prediction_dir=root / "predictions", lock_dir=root / "lock")
    assert before == {name: reporting._inventory(root / name) for name in before}


def test_figure_data_carries_verified_metric_n_ci_and_source_identity(report_base):
    root = report_base["root"] / "report"
    figure = read_result_csv(root / "figure_data.csv")
    metadata = json.loads((root / "figure_metadata.json").read_text())
    assert len(figure) == 16
    assert set(figure.metric) == set(reporting.METRICS)
    assert set(figure.n_participants) == {3}
    assert set(figure.ci_method) == {"participant_percentile_bootstrap"}
    assert metadata["data_sha256"] == sha256_file(root / "figure_data.csv")
    assert metadata["source_participant_sha256"] == sha256_file(root / "participant_results.csv")
    assert figure.protocol_sha256.eq(metadata["protocol_sha256"]).all()
    assert figure.cohort_id.eq(metadata["cohort_id"]).all()


def test_cli_verifies_self_contained_report_and_fails_closed(report_case, monkeypatch, capsys):
    root = report_case["root"]
    entry = runpy.run_path(str(ROOT / "scripts/22_build_report.py"))["main"]
    monkeypatch.setattr(sys, "argv", ["22_build_report.py", "verify-external", "--report", str(root / "report")])
    assert entry() == 0
    result = json.loads(capsys.readouterr().out)
    assert result["status"] == "complete"
    assert result["inference_scope"] == "descriptive_target_shortfall"
    (root / "report/summary.json").unlink()
    assert entry() == 2
    assert json.loads(capsys.readouterr().out)["status"] == "blocked"
