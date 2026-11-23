"""Contract and adversarial verification for an independent S/F research workflow."""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import shutil

import pandas as pd
import pytest

from eeg_ma.classifier_config import classifier_policy
from eeg_ma.evaluation import make_evaluation_splits
from eeg_ma.exploratory import (
    ANALYSIS_ID, ExploratoryError, _config_for_track, _verify_plan,
    build_exploratory_report, validate_exploratory_report,
)
from eeg_ma.provenance import fingerprint, read_csv_literal, sha256_file, validate_complete_run
from eeg_ma.schema import load_feature_table

from exploratory_fixture import create_exploratory_fixture


@pytest.fixture(scope="module")
def synthetic_report(tmp_path_factory):
    root = tmp_path_factory.mktemp("exploratory_")
    source, config, plan, frame = create_exploratory_fixture(root / "source")
    out = root / "report"
    info = build_exploratory_report(source, config, plan, out, allow_synthetic=True)
    return out, source, config, plan, frame, info


def test_complete_two_tracks_with_participant_macro(synthetic_report):
    root, _, _, _, frame, info = synthetic_report
    assert info["status"] == "verified"
    assert info["scope"] == "synthetic_only"
    assert info["n_groups"] == 14 and info["n_participants"] == 3
    assert info["n_predictions"] == len(frame) * 7 == 336
    participants = read_csv_literal(root / "participant_results.csv")
    assert len(participants) == 14 * 3
    summary = read_csv_literal(root / "summary.csv")
    assert set(summary.classifier) == {"LDA", "RBF-SVM", "KFDA", "RandomForest", "LightGBM"}
    assert set(summary.n_subjects) == {3}
    assert set(summary.n_ba_eligible) == {3}
    assert set(summary.n_epochs) == {48}
    pairs = read_csv_literal(root / "paired_participants.csv")
    assert len(pairs) == 2 * 4 * 3
    assert set(pairs.subject_id) == set(frame.subject_id)
    assert "no new hypothesis is chosen" in (root / "report.md").read_text().lower()


def test_full_feature_track_never_fabricates_sfs_accuracy(synthetic_report):
    root, _, _, _, _, _ = synthetic_report
    s = read_csv_literal(root / "S/inter_sfs_path.csv")
    f = read_csv_literal(root / "F/inter_sfs_path.csv")
    assert s.cv_accuracy.notna().all()
    assert f.cv_accuracy.isna().all() and f.improvement.isna().all()
    assert set(f.selection_method) == {"all_declared_features"}
    assert set(read_csv_literal(root / "F/inter_folds.csv").n_selected_features) == {3}
    assert validate_complete_run(root / "F")["status"] == "complete"


def test_reordered_source_rows_preserve_every_inner_and_outer_split(synthetic_report):
    _, _, config, _, frame, _ = synthetic_report
    cfg = json.loads(Path(config).read_text())
    shuffled = frame.sample(frac=1.0, random_state=377).reset_index(drop=True)
    for track in ("S", "F"):
        effective = _config_for_track(cfg, track)
        original = make_evaluation_splits(frame, effective, "inter")
        mixed = make_evaluation_splits(shuffled, effective, "inter")
        assert original == mixed


def test_draft_and_foreign_cohort_block_without_publication(synthetic_report, tmp_path):
    _, source, config, plan, frame, _ = synthetic_report
    cfg = json.loads(Path(config).read_text())
    declared = json.loads(Path(plan).read_text())
    with pytest.raises(ExploratoryError, match="Synthetic-only"):
        _verify_plan(declared, cfg, frame, allow_synthetic=False)
    draft = deepcopy(declared)
    draft["status"] = "draft"
    with pytest.raises(ExploratoryError, match="Draft"):
        _verify_plan(draft, cfg, frame, allow_synthetic=True)
    foreign = deepcopy(declared)
    foreign["cohort_id"] = "another-cohort"
    with pytest.raises(ExploratoryError, match="foreign cohort"):
        _verify_plan(foreign, cfg, frame, allow_synthetic=True)
    output = tmp_path / "should_not_publish"
    with pytest.raises(ExploratoryError):
        build_exploratory_report(source, config, plan, output, allow_synthetic=False)
    assert not output.exists()


def test_f_only_allows_two_declared_tree_models(synthetic_report):
    _, _, config, _, _, _ = synthetic_report
    base = json.loads(Path(config).read_text())
    old = classifier_policy(base)
    assert old["feature_selection"] == "shared_training_only_lda_sfs"
    assert old["schema_version"] == "eeg-classifier-policy-2"
    full = _config_for_track(base, "F")
    policy = classifier_policy(full)
    assert policy["schema_version"] == "eeg-classifier-policy-3"
    assert policy["feature_selection"] == "all_declared_features"
    full["classifiers"].append("LDA")
    with pytest.raises(ValueError, match="only supported"):
        classifier_policy(full)


def _rehashed_copy(synthetic_report, tmp_path, operation):
    original = synthetic_report[0]
    root = tmp_path / ("attacked-" + operation)
    shutil.copytree(original, root)
    if operation == "missing_model":
        table = read_csv_literal(root / "predictions.csv")
        table = table.loc[table["classifier"] != "LightGBM"]
        table.to_csv(root / "predictions.csv", index=False)
    elif operation == "missing_participant":
        table = read_csv_literal(root / "participant_results.csv")
        table = table.loc[table["subject_id"] != table["subject_id"].iloc[0]]
        table.to_csv(root / "participant_results.csv", index=False)
    elif operation == "missing_epoch":
        table = read_csv_literal(root / "predictions.csv")
        table.iloc[1:].to_csv(root / "predictions.csv", index=False)
    elif operation == "changed_eligibility":
        table = read_csv_literal(root / "eligible_epochs.csv")
        table.loc[0, "eligible_epoch_manifest_id"] = "forged"
        table.to_csv(root / "eligible_epochs.csv", index=False)
    elif operation == "wrong_predictions":
        table = read_csv_literal(root / "predictions.csv")
        table.loc[0, "y_pred"] = 1 - int(table.loc[0, "y_pred"])
        table.to_csv(root / "predictions.csv", index=False)
    elif operation == "wrong_source":
        table = read_csv_literal(root / "source_features.csv")
        table.loc[0, "BP__FP1__alpha"] += 0.3
        table.to_csv(root / "source_features.csv", index=False)
    else:
        raise AssertionError(operation)
    manifest = json.loads((root / "report_manifest.json").read_text())
    manifest["files"] = {
        p.relative_to(root).as_posix(): {"sha256": sha256_file(p), "bytes": p.stat().st_size}
        for p in root.rglob("*") if p.is_file() and p.name != "report_manifest.json"
    }
    manifest["report_id"] = fingerprint({
        "version": manifest["schema_version"], "analysis_id": ANALYSIS_ID,
        "study_status": manifest["study_status"], "run_ids": manifest["run_ids"],
        "files": manifest["files"],
    })
    (root / "report_manifest.json").write_text(json.dumps(manifest, sort_keys=True))
    return root


@pytest.mark.parametrize("attack", [
    "missing_model", "missing_participant", "missing_epoch",
    "changed_eligibility", "wrong_predictions", "wrong_source",
])
def test_rehashed_adversarial_archive_is_rejected(synthetic_report, tmp_path, attack):
    altered = _rehashed_copy(synthetic_report, tmp_path, attack)
    with pytest.raises((ExploratoryError, ValueError)):
        validate_exploratory_report(altered)


def test_report_is_immutable_and_rechecks_all_sources(synthetic_report):
    root, *_ = synthetic_report
    before = {str(p.relative_to(root)): sha256_file(p) for p in root.rglob("*") if p.is_file()}
    validate_exploratory_report(root)
    after = {str(p.relative_to(root)): sha256_file(p) for p in root.rglob("*") if p.is_file()}
    assert before == after


def test_never_accept_discovery_legacy_data(synthetic_report):
    _, _, config, plan, frame, _ = synthetic_report
    p = json.loads(Path(plan).read_text())
    c = json.loads(Path(config).read_text())
    bad = frame.copy()
    bad["identity_origin"] = "legacy_manifest"
    with pytest.raises(ExploratoryError, match="Legacy/adapted"):
        _verify_plan(p, c, bad, allow_synthetic=True)
