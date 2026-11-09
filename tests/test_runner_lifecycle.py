"""Exercise the actual CLI orchestration with small, deterministic evaluations."""
from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path
import sys

import pandas as pd
import pytest

from eeg_ma.config import load_config
from eeg_ma.experiment_runs import execute_evaluation
from eeg_ma.provenance import RunTransaction, build_run_identity, read_csv_literal, resolve_report_root, validate_complete_run
from eeg_ma.schema import META_COLUMNS, load_feature_table
from test_provenance import CODE, ENV, fixture_data


def load_script(number):
    path = next(Path("scripts").glob(f"{number:02d}_*.py"))
    spec = importlib.util.spec_from_file_location(f"cli_lifecycle_{number}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def study(tmp_path_factory):
    root = tmp_path_factory.mktemp("runner-study")
    frame, cfg = fixture_data()
    frame["ASYM_LOGRATIO__FP1__FP2__alpha"] = frame["PLV__FP1__FP2__alpha"] - .5
    config_path = root / "config.json"
    config_path.write_text(json.dumps(cfg))
    cfg = load_config(config_path)
    all_path = root / "all.csv"
    frame.to_csv(all_path, index=False)
    frame = load_feature_table(all_path)
    paths = {"all": all_path}
    for name, prefixes in {"baseline": ("BP__", "COH__"), "extensions": ("PLV__", "ASYM_"), "iplv": ("IPLV__",)}.items():
        path = root / f"{name}.csv"
        frame[META_COLUMNS + [c for c in frame if c.startswith(prefixes)]].to_csv(path, index=False)
        paths[name] = path
    references = {}
    for name, feature_set in [("baseline", "all"), ("e2", "baseline_plv"), ("plv", "plv")]:
        effective = copy.deepcopy(cfg)
        effective["feature_set"] = feature_set
        identity = build_run_identity(frame, effective, ["intra", "inter"], code=CODE, environment=ENV)
        path = root / f"reference-{name}"
        RunTransaction(path, identity).execute(lambda stage, identity=identity: execute_evaluation(stage, frame, identity))
        references[name] = path
    return root, config_path, paths, references


def invoke(module, argv, monkeypatch):
    monkeypatch.setattr("eeg_ma.experiment_runs.code_identity", lambda *_: CODE)
    monkeypatch.setattr("eeg_ma.experiment_runs.environment_identity", lambda *_: ENV)
    monkeypatch.setattr(sys, "argv", ["script", *map(str, argv)])
    assert module.main() == 0


@pytest.mark.parametrize("number", [3, 4])
def test_single_scope_cli_preserves_output_layout(study, tmp_path, monkeypatch, number):
    _, cfg, paths, _ = study
    module = load_script(number)
    output = tmp_path / "scope"
    args = ["--features", paths["baseline"], "--config", cfg, "--out-dir", output]
    invoke(module, args, monkeypatch)
    manifest = validate_complete_run(output)
    scope = "intra" if number == 3 else "inter"
    assert manifest["identity"]["scopes"] == [scope]
    assert (output / f"{scope}_predictions.csv").exists()
    before = (output / "run_manifest.json").read_bytes()
    invoke(module, args + ["--resume"], monkeypatch)
    assert (output / "run_manifest.json").read_bytes() == before


@pytest.mark.parametrize("number", [6, 9, 10, 12])
def test_matrix_cli_partial_then_resume_preserves_full_inventory(study, tmp_path, monkeypatch, number):
    _, cfg, paths, refs = study
    module = load_script(number)
    output = tmp_path / "matrix"
    common = ["--config", cfg, "--out-root", output]
    if number == 6:
        args = common + ["--features", paths["baseline"], "--results-root", refs["baseline"]]
        first = ["--feature-sets", "bp"]
        second = ["--feature-sets", "coh"]
        summary_name, expected = "feature_set_comparison.csv", {"all", "bp", "coh"}
        column = "feature_set"
    elif number == 9:
        spec = {"experiments": [
            {"name": "e1_asymmetry", "feature_set": "baseline_asym", "enabled_by_default": True},
            {"name": "e1e2_combined", "feature_set": "baseline_asym_plv", "enabled_by_default": False}]}
        spec_path = tmp_path / "extensions.json"; spec_path.write_text(json.dumps(spec))
        args = common + ["--baseline-features", paths["baseline"], "--extension-features", paths["extensions"],
            "--extension-config", spec_path, "--baseline-results-root", refs["baseline"]]
        first, second = [], ["--include-combined", "--resume"]
        summary_name, expected = "extension_summary.csv", {"poster_baseline", "e1_asymmetry", "e1e2_combined"}
        column = "experiment"
    elif number == 10:
        spec = {"primary_experiments": [
            {"name": "e2a_plv_only", "feature_set": "plv"}, {"name": "e2b_bp_plv", "feature_set": "bp_plv"}]}
        spec_path = tmp_path / "decomposition.json"; spec_path.write_text(json.dumps(spec))
        args = common + ["--baseline-features", paths["baseline"], "--extension-features", paths["extensions"],
            "--experiment-config", spec_path, "--poster-results-root", refs["baseline"], "--e2-full-results-root", refs["e2"]]
        first, second = ["--experiments", "e2a_plv_only"], ["--experiments", "e2b_bp_plv"]
        summary_name, expected = "plv_decomposition_summary.csv", {"poster_baseline", "e2_full_baseline_plv", "e2a_plv_only", "e2b_bp_plv"}
        column = "experiment"
    else:
        spec = {"experiments": [{"name": "plv_alpha_only", "feature_set": "plv_band_alpha"}, {"name": "iplv_only", "feature_set": "iplv"}]}
        spec_path = tmp_path / "validation.json"; spec_path.write_text(json.dumps(spec))
        args = common + ["--baseline-features", paths["baseline"], "--extension-features", paths["extensions"],
            "--iplv-features", paths["iplv"], "--validation-config", spec_path,
            "--poster-results-root", refs["baseline"], "--plv-only-results-root", refs["plv"]]
        first, second = ["--experiments", "plv_alpha_only"], ["--experiments", "iplv_only"]
        summary_name, expected = "final_validation_summary.csv", {"poster_baseline", "plv_all_bands", "plv_alpha_only", "iplv_only"}
        column = "experiment"
    invoke(module, args + first, monkeypatch)
    partial = resolve_report_root(output)
    invoke(module, args + second, monkeypatch)
    complete = resolve_report_root(output)
    assert partial != complete and partial.exists()
    assert set(read_csv_literal(complete / summary_name)[column]) == expected
    before = (complete / "run_manifest.json").read_bytes()
    invoke(module, args + first + ["--resume"], monkeypatch)
    assert resolve_report_root(output) == complete
    assert (complete / "run_manifest.json").read_bytes() == before
