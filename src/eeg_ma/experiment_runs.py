"""Shared lifecycle for the historical evaluation command-line programs."""
from __future__ import annotations

import copy
from pathlib import Path

from .evaluation import run_inter_subject, run_intra_subject
from .provenance import (
    MatrixRun, RunTransaction, build_run_identity, code_identity,
    environment_identity, fingerprint, reference_identity, validate_reference_coverage, input_inventory, ProvenanceError, _write_json,
)


def load_features(path, legacy_manifest=None):
    from .schema import load_feature_table
    return load_feature_table(path, legacy_manifest=legacy_manifest)


def add_identity_arguments(parser, *, resume=True):
    parser.add_argument("--legacy-manifest", default=None,
        help="Explicit legacy subject/session mapping for pre-schema feature CSVs; never inferred from filenames.")
    if resume:
        parser.add_argument("--resume", action="store_true",
            help="Reuse only a complete run with identical input/config/code/schema/split/environment identity.")


def source_inputs(paths, legacy_manifest=None):
    inputs = {name: path for name, path in paths.items() if path is not None}
    if legacy_manifest is not None:
        inputs["legacy_mapping"] = legacy_manifest
    return inputs


def execute_evaluation(stage, features, identity):
    _write_json(stage / "config_snapshot.json", identity["config"])
    _write_json(stage / "split_manifest.json", identity["splits"])
    for scope in identity["scopes"]:
        flat = identity.get("scope_layout") == "flat"
        output = stage / ".evaluation" if flat else stage / scope
        (run_intra_subject if scope == "intra" else run_inter_subject)(features, identity["config"], output)
        if flat:
            for path in output.iterdir():
                path.rename(stage / path.name)
            output.rmdir()


def run_single(features, cfg, scope, output, *, input_paths, resume=False, code=None, environment=None, project_root=None):
    identity = build_run_identity(features, cfg, [scope], input_paths=input_paths,
        experiment={"name": f"baseline_{scope}", "feature_set": cfg.get("feature_set", "all")},
        code=code if code is not None else code_identity(project_root),
        environment=environment if environment is not None else environment_identity(project_root),
        project_root=project_root, scope_layout="flat")
    def writer(stage):
        if input_inventory(input_paths) != identity["inputs"]:
            raise ProvenanceError("Input bytes changed before evaluation")
        execute_evaluation(stage, features, identity)
        if input_inventory(input_paths) != identity["inputs"]:
            raise ProvenanceError("Input bytes changed during evaluation")
    return RunTransaction(output, identity, resume=resume, project_root=project_root).execute(writer)


def prepare_matrix(features, cfg, *, experiments, selected, scopes, out_root,
                   input_paths, reference_paths, matrix_config, resume=False,
                   code=None, environment=None, project_root=None):
    """Capture one invocation snapshot and preflight the entire matrix."""
    names = [str(x["name"]) for x in experiments]
    if len(set(names)) != len(names):
        raise ValueError("Duplicate experiment definitions")
    code = code if code is not None else code_identity(project_root)
    environment = environment if environment is not None else environment_identity(project_root)
    references = {name: reference_identity(path, scopes) for name, path in reference_paths.items()}
    identities = {}
    for experiment in experiments:
        effective = copy.deepcopy(cfg)
        effective["feature_set"] = str(experiment["feature_set"])
        identities[str(experiment["name"])] = build_run_identity(features, effective, scopes,
            input_paths=input_paths, references=references, experiment=experiment,
            context_config=matrix_config, code=code, environment=environment, project_root=project_root)
    first = next(iter(identities.values()))
    roles = {"poster_baseline": "baseline", "e2_full_baseline_plv": "baseline_plv", "plv_all_bands": "plv"}
    for name, path in reference_paths.items():
        validate_reference_coverage(path, first, roles.get(name))
    context = {"context_version": "experiment-matrix-1", "base_config": cfg,
        "matrix_config": matrix_config, "inputs": first["inputs"], "dataset": first["dataset"],
        "split_sha256": first["split_sha256"], "scopes": list(scopes),
        "references": references, "code": code, "environment": environment,
        "experiment_identity_ids": {name: fingerprint(identity) for name, identity in identities.items()}}
    matrix = MatrixRun(out_root, context, identities, selected, resume=resume, project_root=project_root)
    def source_guard():
        if input_inventory(input_paths) != first["inputs"]:
            raise ProvenanceError("Input bytes changed during this invocation")
        now = {name: reference_identity(path, scopes) for name, path in reference_paths.items()}
        if now != references:
            raise ProvenanceError("Reference bytes changed during this invocation")
    matrix.source_guard = source_guard
    matrix.preflight()
    def evaluator(name, stage):
        source_guard()
        execute_evaluation(stage, features, identities[name])
        source_guard()
    return matrix, evaluator
