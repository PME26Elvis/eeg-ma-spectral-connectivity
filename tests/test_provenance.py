from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd
import pytest

from eeg_ma.experiment_runs import execute_evaluation, prepare_matrix, run_single
from eeg_ma.provenance import (
    MatrixRun, ProvenanceError, RunTransaction, build_run_identity, fingerprint,
    legacy_inventory, protect_output, read_csv_literal, resolve_report_root,
    validate_complete_run,
)
from eeg_ma.schema import adapt_legacy_features

CODE = {"commit": "test-commit", "dirty": False, "dirty_fingerprint": "test"}
ENV = {"python": "test", "lock_id": "test-lock"}


def fixture_data():
    rng = np.random.default_rng(44)
    rows = []
    subjects = ["001", "1", "NA"]
    for subject in subjects:
        for y in (0, 1):
            for i in range(12):
                start = (y * 12 + i) * 2500
                rows.append({"subject_id": subject, "comparison": "rest1_vs_type1", "class_label": y,
                    "class_name": "Rest" if y == 0 else "MA", "stage": "Rest1" if y == 0 else "Type1",
                    "epoch_index": i + 1, "attempt_index": 0 if y == 0 else i + 1,
                    "onset_elapsed_s": start / 500, "start_sample": start, "end_sample": start + 2500,
                    "BP__FP1__alpha": rng.normal() + y, "COH__FP1__FP2__alpha": rng.normal() + y,
                    "PLV__FP1__FP2__alpha": rng.uniform(0, 1), "IPLV__FP1__FP2__alpha": rng.uniform(0, 1)})
    mapping = {"schema_version": "legacy-mapping-1", "subjects": {s: {
        "cohort_id": "fixture", "session_id": f"session-{s}", "recording_id": f"record-{s}",
        "source_sha256": "unknown", "units": "unknown", "reference": "unknown",
        "protocol_id": "legacy-fixture", "protocol_version": "001", "channel_order": "unknown"} for s in subjects}}
    frame = adapt_legacy_features(pd.DataFrame(rows), mapping)
    cfg = json.loads(Path("configs/smoke_test.json").read_text())
    cfg["sfs"]["max_features"] = 1
    cfg["cv"]["intra_outer_folds"] = 2
    cfg["cv"]["intra_inner_folds"] = 2
    cfg["rbf_grid"] = {"C": [1.0], "gamma_values": [0.5]}
    cfg["n_jobs"] = 1
    return frame, cfg


@pytest.fixture(scope="module")
def completed_base(tmp_path_factory):
    tmp_path = tmp_path_factory.mktemp("verified-run")
    frame, cfg = fixture_data()
    identity = build_run_identity(frame, cfg, ["inter", "intra"], code=CODE, environment=ENV)
    root = tmp_path / "complete"
    RunTransaction(root, identity).execute(lambda stage: execute_evaluation(stage, frame, identity))
    return root, frame, cfg, identity


@pytest.fixture
def completed(completed_base, tmp_path):
    source, frame, cfg, identity = completed_base
    root = tmp_path / "complete"
    shutil.copytree(source, root)
    return root, frame, cfg, identity


def bytes_tree(root):
    return {str(p.relative_to(root)): p.read_bytes() for p in root.rglob("*") if p.is_file()}


def test_complete_resume_is_immutable_and_literal_ids_survive(completed):
    root, _, _, identity = completed
    before = bytes_tree(root)
    RunTransaction(root, identity, resume=True).execute(lambda _: pytest.fail("should reuse"))
    assert bytes_tree(root) == before
    prediction = read_csv_literal(root / "inter/inter_predictions.csv")
    assert set(prediction.subject_id) == {"001", "1", "NA"}
    assert set(prediction.protocol_version) == {"001"}


@pytest.mark.parametrize("field", ["config", "code", "environment", "feature_order", "inputs", "splits", "references", "dataset"])
def test_identity_changes_fail_before_writes(completed, field):
    root, _, _, identity = completed
    changed = copy.deepcopy(identity)
    changed[field] = {"different": True} if field != "feature_order" else list(reversed(identity[field]))
    before = bytes_tree(root)
    with pytest.raises(ProvenanceError, match="identity mismatch"):
        RunTransaction(root, changed, resume=True).execute(lambda _: pytest.fail("not reached"))
    assert bytes_tree(root) == before


@pytest.mark.parametrize("relative", ["run_manifest.json", "inter/inter_predictions.csv", "inter/inter_sfs_path.csv", "inter/inter_summary.csv", "intra/intra_overall_summary.csv"])
def test_missing_required_artifact_rejected(completed, relative):
    root, _, _, identity = completed
    (root / relative).unlink()
    before = bytes_tree(root)
    with pytest.raises(ProvenanceError):
        RunTransaction(root, identity, resume=True).execute(lambda _: pytest.fail("not reached"))
    assert bytes_tree(root) == before


def test_parseable_truncated_predictions_fail_resume_and_new_publication(completed, tmp_path):
    root, _, _, identity = completed
    path = root / "inter/inter_predictions.csv"
    rows = path.read_bytes().splitlines(keepends=True)
    path.write_bytes(b"".join(rows[:-1]))
    before = bytes_tree(root)
    with pytest.raises(ProvenanceError, match="artifacts"):
        RunTransaction(root, identity, resume=True).execute(lambda _: pytest.fail("must not reuse truncation"))
    assert bytes_tree(root) == before

    # A syntactically valid CSV missing its last row must also fail before a
    # new manifest can bless its hashes as complete.
    destination = tmp_path / "new-run"
    def truncated_writer(stage):
        shutil.copytree(root, stage, dirs_exist_ok=True)
        (stage / "run_manifest.json").unlink()
    with pytest.raises(ProvenanceError, match="row count"):
        RunTransaction(destination, identity).execute(truncated_writer)
    assert not destination.exists()
    assert not list(tmp_path.glob(".staging-new-run-*/run_manifest.json"))


def test_semantically_wrong_output_never_publishes(tmp_path):
    frame, cfg = fixture_data()
    identity = build_run_identity(frame, cfg, ["inter"], code=CODE, environment=ENV)
    destination = tmp_path / "bad"
    def bad_writer(stage):
        execute_evaluation(stage, frame, identity)
        path = stage / "inter/inter_predictions.csv"
        table = read_csv_literal(path)
        table.loc[0, "session_id"] = "wrong-session"
        table.to_csv(path, index=False)
    with pytest.raises(ProvenanceError, match="session_id"):
        RunTransaction(destination, identity).execute(bad_writer)
    assert not destination.exists()
    failed = list(tmp_path.glob(".staging-bad-*/transaction_state.json"))
    assert len(failed) == 1 and json.loads(failed[0].read_text())["status"] == "failed"
    assert not list(tmp_path.glob(".staging-bad-*/run_manifest.json"))


def test_interrupted_stage_not_complete_and_can_retry(tmp_path):
    frame, cfg = fixture_data()
    identity = build_run_identity(frame, cfg, ["inter"], code=CODE, environment=ENV)
    destination = tmp_path / "retry"
    def interrupt(stage):
        (stage / "partial.csv").write_text("a\n1\n")
        raise KeyboardInterrupt()
    with pytest.raises(KeyboardInterrupt):
        RunTransaction(destination, identity).execute(interrupt)
    assert not destination.exists()
    state = next(tmp_path.glob(".staging-retry-*/transaction_state.json"))
    assert json.loads(state.read_text())["status"] == "interrupted"
    RunTransaction(destination, identity, resume=True).execute(lambda stage: execute_evaluation(stage, frame, identity))
    validate_complete_run(destination)


def test_partial_run_inventory_aggregation_never_shrinks(tmp_path):
    frame, cfg = fixture_data()
    defs = [{"name": "bp", "feature_set": "bp"}, {"name": "coh", "feature_set": "coh"}]
    root = tmp_path / "matrix"
    def reporter(inventory):
        return {"summary.csv": pd.DataFrame({"experiment": sorted(inventory), "value": [1] * len(inventory)})}
    def call(selected, resume):
        matrix, evaluator = prepare_matrix(frame, cfg, experiments=defs, selected=selected,
            scopes=["inter"], out_root=root, input_paths={}, reference_paths={}, matrix_config={"experiments": defs},
            resume=resume, code=CODE, environment=ENV)
        return matrix.run(evaluator, reporter)
    call(["bp"], False)
    first_report = resolve_report_root(root)
    first_bytes = bytes_tree(root / "bp")
    call(["coh"], False)
    second_report = resolve_report_root(root)
    assert set(read_csv_literal(second_report / "summary.csv").experiment) == {"bp", "coh"}
    before = bytes_tree(root)
    call(["bp"], True)
    assert bytes_tree(root) == before
    assert bytes_tree(root / "bp") == first_bytes
    assert first_report.exists() and first_report != second_report


def test_matrix_preflights_unselected_incompatible_run_before_writes(tmp_path):
    frame, cfg = fixture_data()
    defs = [{"name": "bp", "feature_set": "bp"}, {"name": "coh", "feature_set": "coh"}]
    root = tmp_path / "matrix"
    kwargs = dict(experiments=defs, scopes=["inter"], out_root=root, input_paths={}, reference_paths={}, matrix_config={}, code=CODE, environment=ENV)
    matrix, evaluator = prepare_matrix(frame, cfg, selected=["coh"], **kwargs)
    matrix.run(evaluator, lambda inv: {"summary.csv": pd.DataFrame({"experiment": list(inv)})})
    (root / "coh/inter/inter_predictions.csv").write_text("broken\n")
    before = bytes_tree(root)
    with pytest.raises(ProvenanceError):
        prepare_matrix(frame, cfg, selected=["bp"], **kwargs)
    assert not (root / "bp").exists() and bytes_tree(root) == before


def test_flat_single_runner_and_protected_canonical_paths(tmp_path):
    frame, cfg = fixture_data()
    output = tmp_path / "single"
    run_single(frame, cfg, "inter", output, input_paths={}, code=CODE, environment=ENV)
    assert (output / "inter_folds.csv").exists()
    validate_complete_run(output)
    project = tmp_path / "repo"
    (project / "results").mkdir(parents=True)
    for path in [project, project / "results", project / "results/new", project / "data/features/new"]:
        with pytest.raises(ProvenanceError):
            protect_output(path, project)
    alias = tmp_path / "alias"
    alias.symlink_to(project / "results", target_is_directory=True)
    with pytest.raises(ProvenanceError):
        protect_output(alias / "new", project)


def test_legacy_inventory_does_not_upgrade_to_trusted_resume(tmp_path):
    old = tmp_path / "old"
    old.mkdir()
    (old / "summary.csv").write_text("value\n1\n")
    inventory = legacy_inventory(old)
    assert inventory["status"] == "legacy_unverified"
    assert inventory["historical_environment"] == "unknown"
    with pytest.raises(ProvenanceError):
        validate_complete_run(old)
    with pytest.raises(ProvenanceError):
        resolve_report_root(old)
    assert resolve_report_root(old, allow_legacy=True) == old
