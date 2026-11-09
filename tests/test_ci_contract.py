"""Controlled environment comparisons must identify inputs before installation."""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess

import pytest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("prepare_environment_comparison", ROOT / "scripts/20_prepare_environment_comparison.py")
comparison = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(comparison)


def git(path, *args):
    return subprocess.check_output(["git", "-C", str(path), *args], text=True).strip()


def commit(path):
    git(path, "add", ".")
    git(path, "-c", "user.name=Synthetic CI fixture", "-c", "user.email=fixture@example.invalid",
        "commit", "--quiet", "-m", "Synthetic immutable comparison evidence")
    return git(path, "rev-parse", "HEAD")


def checkout(path):
    path.mkdir()
    subprocess.run(["git", "init", "--quiet", str(path)], check=True, capture_output=True)
    (path / "environment").mkdir()
    definition = json.loads((ROOT / "environment/reference.json").read_text())
    for name in ("reference.json", definition["requirements_lock"]["file"]):
        shutil.copyfile(ROOT / "environment" / name, path / "environment" / name)
    fixture = path / "tests/fixtures/numerical/reference.json"
    fixture.parent.mkdir(parents=True)
    fixture.write_text('{"scope":"synthetic comparison-preflight fixture only"}\n')
    commit(path)
    return path


@pytest.fixture
def checkouts(tmp_path):
    return checkout(tmp_path / "analysis"), checkout(tmp_path / "candidate")


def prepare(paths, **overrides):
    analysis, candidate = paths
    values = {"candidate_commit": git(candidate, "rev-parse", "HEAD"), "run_id": "review-001",
              "reason": "Prospective numerical comparison of a dependency update",
              "out_dir": analysis / "outputs/comparison-request"}
    values.update(overrides)
    return comparison.prepare_comparison(analysis, candidate, **values)


@pytest.mark.parametrize("line_ending", [b"\n", b"\r\n"])
def test_comparison_freezes_both_profiles_and_never_overwrites(checkouts, line_ending):
    analysis, candidate = checkouts
    for checkout_root in checkouts:
        definition = checkout_root / "environment/reference.json"
        definition.write_bytes(definition.read_bytes().replace(b"\n", line_ending))
        if line_ending != b"\n":
            commit(checkout_root)
    record = prepare(checkouts)
    destination = analysis / "outputs/comparison-request"
    assert record["analysis_code_commit"] == git(analysis, "rev-parse", "HEAD")
    assert record["candidate_environment_commit"] == git(candidate, "rev-parse", "HEAD")
    assert record["scope"] == "bounded_synthetic_grouped_cv_only"
    for role in ("baseline", "candidate"):
        item = record["environments"][role]
        source = analysis if role == "baseline" else candidate
        assert (destination / item["frozen_definition"]).read_bytes() == (source / "environment/reference.json").read_bytes()
        assert hashlib.sha256((destination / item["frozen_definition"]).read_bytes()).hexdigest() == item["definition_sha256"]
        assert hashlib.sha256((destination / item["frozen_requirements"]).read_bytes()).hexdigest() == item["requirements_sha256"]
    before = (destination / "comparison_request.json").read_bytes()
    with pytest.raises(comparison.ComparisonError, match="new"):
        prepare(checkouts)
    assert (destination / "comparison_request.json").read_bytes() == before
    shutil.rmtree(candidate)
    assert json.loads((destination / "candidate/reference.json").read_text())["python"] == "3.12.14"


@pytest.mark.parametrize("candidate_commit", ["main", "a" * 39, "a" * 40 + "\n", "$(touch sentinel)"])
def test_candidate_ref_is_an_exact_commit_before_any_git_operation(candidate_commit):
    with pytest.raises(comparison.ComparisonError, match="40-character"):
        comparison.validate_request_inputs(candidate_commit, "comparison-1", "Documented update")


@pytest.mark.parametrize("run_id", ["../results", "value\nOTHER=value", "a" * 65])
def test_run_identifiers_cannot_be_paths_or_action_output_injections(run_id):
    with pytest.raises(comparison.ComparisonError, match="identifier"):
        comparison.validate_request_inputs("a" * 40, run_id, "Documented update")


@pytest.mark.parametrize("reason", ["", "   ", "x" * 2001])
def test_comparison_requires_a_bounded_prospective_reason(reason):
    with pytest.raises(comparison.ComparisonError, match="prospective reason"):
        comparison.validate_request_inputs("a" * 40, "comparison-1", reason)


def test_reason_remains_data_and_never_enters_action_outputs(checkouts):
    analysis, _ = checkouts
    reason = 'Literal $(touch should-not-exist) `id`\nINJECTED=value'
    record = prepare(checkouts, reason=reason)
    assert record["reason"] == reason
    output = comparison.github_outputs(record, analysis / "outputs/comparison-request", analysis)
    assert "INJECTED" not in output and "should-not-exist" not in output
    assert set(line.split("=", 1)[0] for line in output.splitlines()) == {
        "baseline_python", "candidate_python", "baseline_lock", "candidate_lock"}
    assert not (analysis / "should-not-exist").exists()


def test_wrong_declared_commit_is_not_replaced_with_current_head(checkouts):
    with pytest.raises(comparison.ComparisonError, match="requested commit"):
        prepare(checkouts, candidate_commit="0" * 40)


def test_uncommitted_candidate_profile_is_not_blessed_by_its_committed_lock(checkouts):
    _, candidate = checkouts
    path = candidate / "environment/reference.json"
    document = json.loads(path.read_text())
    document["reference_id"] += "-uncommitted"
    path.write_text(json.dumps(document))
    with pytest.raises(comparison.ComparisonError, match="differs from its selected commit"):
        prepare(checkouts)


def test_dirty_analysis_fixture_cannot_masquerade_as_the_selected_revision(checkouts):
    analysis, _ = checkouts
    (analysis / "tests/fixtures/numerical/reference.json").write_text('{"changed":true}\n')
    with pytest.raises(comparison.ComparisonError, match="must be clean"):
        prepare(checkouts)


def test_rehashed_candidate_lock_cannot_add_pip_options_or_includes(checkouts):
    _, candidate = checkouts
    path = candidate / "environment/reference.json"
    document = json.loads(path.read_text())
    lock = path.parent / document["requirements_lock"]["file"]
    lock.write_text(lock.read_text() + "--requirement https://example.invalid/unreviewed.txt\n")
    document["requirements_lock"]["sha256"] = hashlib.sha256(lock.read_bytes()).hexdigest()
    path.write_text(json.dumps(document))
    commit(candidate)
    with pytest.raises(ValueError, match="options, URLs and includes"):
        prepare(checkouts)
    assert not (checkouts[0] / "outputs/comparison-request").exists()


def test_candidate_lock_path_traversal_is_rejected_before_install(checkouts):
    _, candidate = checkouts
    path = candidate / "environment/reference.json"
    document = json.loads(path.read_text())
    document["requirements_lock"]["file"] = "../unreviewed.lock"
    path.write_text(json.dumps(document))
    commit(candidate)
    with pytest.raises(ValueError, match="same-directory basename"):
        prepare(checkouts)


@pytest.mark.parametrize("alias", [False, True])
def test_comparison_outputs_cannot_overwrite_canonical_sources(checkouts, alias):
    analysis, _ = checkouts
    canonical = analysis / "results"
    canonical.mkdir()
    target = canonical / "comparison"
    if alias:
        (analysis / "outputs").mkdir()
        (analysis / "outputs/alias").symlink_to(canonical, target_is_directory=True)
        target = analysis / "outputs/alias/comparison"
    with pytest.raises(comparison.ComparisonError, match="outputs/ directory|symlinks"):
        prepare(checkouts, out_dir=target)
    assert not (canonical / "comparison").exists()
