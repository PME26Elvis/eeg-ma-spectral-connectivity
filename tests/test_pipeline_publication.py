from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import pandas as pd
import pytest

from eeg_ma.artifacts import new_output_directory, write_new_csv
from eeg_ma.provenance import ProvenanceError, sha256_file


ROOT = Path(__file__).resolve().parents[1]


def _pipeline():
    spec = importlib.util.spec_from_file_location("pipeline_cli", ROOT / "scripts" / "run_all.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_failed_publication_never_exposes_partial_output(tmp_path):
    target = tmp_path / "published"
    with pytest.raises(RuntimeError, match="interrupted"):
        with new_output_directory(target) as stage:
            (stage / "summary.csv").write_text("incomplete", encoding="utf-8")
            assert not target.exists()
            raise RuntimeError("interrupted")
    assert not target.exists()
    assert not list(tmp_path.glob(".published.staging-*"))


def test_publication_and_csv_refuse_existing_and_canonical_outputs(tmp_path):
    target = tmp_path / "published"
    with new_output_directory(target) as stage:
        (stage / "evidence.txt").write_bytes(b"original")
    with pytest.raises(FileExistsError):
        with new_output_directory(target):
            pytest.fail("writer must not start")
    assert (target / "evidence.txt").read_bytes() == b"original"

    csv = tmp_path / "behavior.csv"
    write_new_csv(pd.DataFrame({"subject_id": ["001", "NA"]}), csv)
    original = csv.read_bytes()
    with pytest.raises(FileExistsError):
        write_new_csv(pd.DataFrame({"subject_id": ["1"]}), csv)
    assert csv.read_bytes() == original
    with pytest.raises(ProvenanceError, match="Protected"):
        write_new_csv(pd.DataFrame({"x": [1]}), ROOT / "results" / "forbidden.csv")
    link = tmp_path / "canonical_alias"
    link.symlink_to(ROOT / "results", target_is_directory=True)
    with pytest.raises(ProvenanceError):
        write_new_csv(pd.DataFrame({"x": [1]}), link / "forbidden.csv")


def test_pipeline_evaluates_exactly_the_newly_extracted_features(tmp_path, monkeypatch):
    module = _pipeline()
    target = tmp_path / "run"
    calls = []

    def run(script, *args):
        options = dict(zip(args[::2], args[1::2]))
        calls.append((script, options))
        assert not target.exists()
        if script == "02_extract_features.py":
            folder = Path(options["--out-dir"])
            folder.mkdir(parents=True)
            (folder / "features_all.csv").write_text("new feature evidence", encoding="utf-8")
        if script.startswith(("03_", "04_")):
            feature = Path(options["--features"])
            assert feature.read_text(encoding="utf-8") == "new feature evidence"
            result = Path(options["--out-dir"])
            result.mkdir(parents=True)
            (result / "complete.txt").write_text(script, encoding="utf-8")

    monkeypatch.setattr(module, "run", run)
    monkeypatch.setattr(sys, "argv", ["run_all", "--raw-root", "new-raw", "--config", "new-config.json", "--out-root", str(target)])
    assert module.main() == 0
    assert len(calls) == 5
    assert calls[3][1]["--features"] == calls[4][1]["--features"]
    assert all(options["--config"] == "new-config.json" for _, options in calls)
    assert (target / "features" / "baseline" / "features_all.csv").is_file()
    assert (target / "results" / "baseline" / "intra" / "complete.txt").is_file()
    assert (target / "results" / "baseline" / "inter" / "complete.txt").is_file()


def test_pipeline_error_does_not_publish_successful_prefix(tmp_path, monkeypatch):
    module = _pipeline()
    target = tmp_path / "failed-run"

    def run(script, *args):
        if script == "03_intra_subject.py":
            raise RuntimeError("evaluation failed")

    monkeypatch.setattr(module, "run", run)
    monkeypatch.setattr(sys, "argv", ["run_all", "--out-root", str(target)])
    with pytest.raises(RuntimeError, match="evaluation failed"):
        module.main()
    assert not target.exists()


def test_committed_legacy_inventory_matches_frozen_research_files():
    inventory = json.loads((ROOT / "docs/manifests/legacy_discovery_inventory.json").read_text(encoding="utf-8"))
    assert inventory["status"] == "legacy_unverified"
    for relative, snapshot in inventory["inventories"].items():
        root = ROOT / relative
        assert snapshot["historical_code"] == "unknown"
        assert snapshot["historical_environment"] == "unknown"
        actual = {path.relative_to(root).as_posix(): path for path in root.rglob("*") if path.is_file()}
        assert set(actual) == set(snapshot["observed_files"])
        for name, observed in snapshot["observed_files"].items():
            assert actual[name].stat().st_size == observed["bytes"], name
            assert sha256_file(actual[name]) == observed["sha256"], name
