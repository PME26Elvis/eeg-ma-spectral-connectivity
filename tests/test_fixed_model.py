"""Fixed fitted-state roundtrips and adversarial training/artifact boundaries."""
from copy import deepcopy
import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd
import pytest
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler

import eeg_ma.fixed_model as models
from eeg_ma.provenance import fingerprint, sha256_file
from eeg_ma.schema import load_feature_table
from fixed_model_fixtures import make_feature_table, write_development_fixture, write_json


@pytest.fixture(scope="module")
def fitted(tmp_path_factory):
    root = tmp_path_factory.mktemp("fixed-model")
    path, spec, frame = write_development_fixture(root / "source")
    pipeline_factory = models.lda_pipeline
    estimators = []
    def factory():
        estimator = pipeline_factory()
        estimators.append(estimator)
        return estimator
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(models, "lda_pipeline", factory)
        result = models.fit_development(path, root / "bundle", require_reference_environment=False)
    assert len(estimators) == 4
    return root, path, spec, frame, result, estimators


def _model(fitted, role="primary"):
    return models.load_fixed_model(fitted[0] / "bundle" / f"{role}_model.json")


def _rehash(document):
    document.pop("model_id", None)
    document["model_id"] = fingerprint(document)
    return document


@pytest.mark.parametrize("role,offset", [("primary", 0), ("reference", 2)])
def test_roundtrip_matches_actual_fitted_sklearn_on_new_subjects(fitted, role, offset):
    loaded = _model(fitted, role)
    external = make_feature_table(["new-001", "new-NA"], "synthetic-external", seed=814)
    document = loaded.document
    assert document["execution_scope"] == "software_validation"
    assert document["training_subject_ids"] == ["001", "1", "NA"]
    assert len(loaded.feature_order) == (21 if role == "primary" else 168)
    assert not hasattr(loaded, "fit") and not hasattr(loaded, "partial_fit")
    for number, comparison in enumerate(("rest1_vs_type1", "rest2_vs_type2")):
        values = external.loc[external.comparison == comparison, list(loaded.feature_order)]
        indices = document["comparisons"][comparison]["selected_indices"]
        estimator = fitted[5][offset + number]
        np.testing.assert_array_equal(loaded.predict(values, comparison=comparison), estimator.predict(values.to_numpy(float)[:, indices]))
        np.testing.assert_allclose(loaded.decision_function(values, comparison=comparison), estimator.decision_function(values.to_numpy(float)[:, indices]), rtol=0, atol=1e-14)
        restored = models.FixedLDAModel(document)
        np.testing.assert_array_equal(restored.predict(values, comparison=comparison), loaded.predict(values, comparison=comparison))
    document["class_mapping"]["0"] = "changed externally"
    assert loaded.document["class_mapping"] == models.CLASS_MAPPING


def test_predict_only_state_never_calls_fitting(fitted, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("External rows must never enter fit/selection")
    monkeypatch.setattr(StandardScaler, "fit", forbidden)
    monkeypatch.setattr(LinearDiscriminantAnalysis, "fit", forbidden)
    monkeypatch.setattr(models, "forward_select_lda", forbidden)
    frame = make_feature_table(["held-out"], "new-cohort", seed=15)
    for role in models.ROLES:
        loaded = _model(fitted, role)
        for comparison in loaded.document["comparisons"]:
            values = frame.loc[frame.comparison == comparison, list(loaded.feature_order)]
            assert len(loaded.predict(values, comparison=comparison)) == len(values)


@pytest.mark.parametrize("fault", ["reordered", "extra_label", "missing", "nonfinite"])
def test_inference_requires_exact_numeric_predictor_projection(fitted, fault):
    loaded = _model(fitted)
    frame = fitted[3][list(loaded.feature_order)].copy()
    if fault == "reordered":
        frame = frame[frame.columns[::-1]]
    elif fault == "extra_label":
        frame["true_label"] = 1
    elif fault == "missing":
        frame = frame.iloc[:, :-1]
    else:
        frame.iloc[0, 0] = np.nan
    with pytest.raises(models.FixedModelError):
        loaded.predict(frame, comparison="rest1_vs_type1")


@pytest.mark.parametrize("fault", ["class_mapping", "feature_order", "selection_names", "selection_index", "scaler_variance", "classifier_coefficients", "class_prior", "inner_membership", "missing_session", "diagnostic_score", "diagnostic_improvement", "analysis_content", "schema_content", "source_protocol"])
def test_rehash_does_not_make_contradictory_executable_evidence_valid(fitted, fault):
    document = _model(fitted).document
    state = document["comparisons"]["rest1_vs_type1"]
    if fault == "class_mapping":
        document["class_mapping"]["0"] = "MentalArithmetic"
    elif fault == "feature_order":
        document["feature_order"].reverse()
    elif fault == "selection_names":
        state["selected_names"][0] = document["feature_order"][-1] + "-wrong"
    elif fault == "selection_index":
        state["selected_indices"][0] = len(document["feature_order"])
    elif fault == "scaler_variance":
        state["scaler"]["scale"][0] *= 2
    elif fault == "classifier_coefficients":
        state["classifier"]["coef"][0][0] += 1
    elif fault == "class_prior":
        state["classifier"]["priors"] = [.1, .9]
    elif fault == "inner_membership":
        state["inner_splits"][0]["train_epoch_ids"].append(state["inner_splits"][0]["validation_epoch_ids"][0])
    elif fault == "missing_session":
        state["training_epochs"] = state["training_epochs"][:-1]
    elif fault == "diagnostic_score":
        state["selection_diagnostics"]["sfs_inner_accuracy"] = .123
    elif fault == "diagnostic_improvement":
        state["selection_diagnostics"]["path"][0]["improvement"] = .123
    elif fault == "analysis_content":
        document["analysis_config"]["sfs"]["max_features"] = 2
    elif fault == "schema_content":
        document["feature_schema"]["class_mapping"]["1"] = "wrong"
    else:
        document["source_protocol_text"] += "\n"
    with pytest.raises(models.FixedModelError):
        models.validate_model(_rehash(document))


def test_expected_hash_and_role_and_schema_bind_the_loaded_artifact(fitted, tmp_path):
    source = fitted[0] / "bundle/primary_model.json"
    with pytest.raises(models.FixedModelError, match="byte hash"):
        models.load_fixed_model(source, expected_sha256="f" * 64)
    with pytest.raises(models.FixedModelError, match="role"):
        models.load_fixed_model(source, expected_role="reference")
    schema = _model(fitted).document["feature_schema"]
    schema["primary_features"] = schema["primary_features"][::-1]
    with pytest.raises(models.FixedModelError, match="expected schema"):
        models.load_fixed_model(source, expected_feature_schema=schema)
    alias = tmp_path / "alias.json"
    alias.symlink_to(source)
    with pytest.raises(models.FixedModelError, match="symlink"):
        models.load_fixed_model(alias)
    assert models.load_fixed_model(source, expected_sha256=sha256_file(source)).model_id


def test_grouped_training_membership_is_stable_under_row_reordering(fitted, tmp_path):
    path, _, frame = write_development_fixture(tmp_path / "reordered", frame=fitted[3].sample(frac=1, random_state=17))
    models.fit_development(path, tmp_path / "fit", require_reference_environment=False)
    for role in models.ROLES:
        original = _model(fitted, role).document
        reordered = models.load_fixed_model(tmp_path / "fit" / f"{role}_model.json").document
        assert original["comparisons"] == reordered["comparisons"]
        for state in reordered["comparisons"].values():
            membership = {row["epoch_id"]: row for row in state["training_epochs"]}
            for split in state["inner_splits"]:
                assert set(split["train_subject_ids"]).isdisjoint(split["validation_subject_ids"])
                assert set(split["train_epoch_ids"]).isdisjoint(split["validation_epoch_ids"])
                heldout = split["validation_subject_ids"][0]
                assert {membership[e]["session_id"] for e in split["validation_epoch_ids"]} == {"session-1", "session-2"}
                assert all(membership[e]["subject_id"] != heldout for e in split["train_epoch_ids"])


@pytest.mark.parametrize("fault", ["source_hash", "subject_boundary", "comparison_missing", "schema_order"])
def test_development_input_fails_before_any_training(tmp_path, monkeypatch, fault):
    path, spec, _ = write_development_fixture(tmp_path / "source")
    if fault == "source_hash":
        with (path.parent / "baseline.csv").open("a") as stream:
            stream.write("\n")
    elif fault == "subject_boundary":
        manifest_path = path.parent / "development_manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["subject_ids"] = ["001", "1", "unseen"]
        write_json(manifest_path, manifest)
        spec["freeze"]["artifacts"]["development_manifest"]["sha256"] = sha256_file(manifest_path)
        write_json(path, spec)
    elif fault == "comparison_missing":
        data_path = path.parent / "development_data_manifest.json"
        data = json.loads(data_path.read_text())
        for entry in data["sources"]:
            source = path.parent / entry["path"]
            frame = load_feature_table(source)
            frame.loc[frame.comparison == "rest1_vs_type1"].to_csv(source, index=False)
            entry["sha256"] = sha256_file(source)
        write_json(data_path, data)
        spec["freeze"]["artifacts"]["development_data_manifest"]["sha256"] = sha256_file(data_path)
        manifest_path = path.parent / "development_manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["data_sha256"] = sha256_file(data_path)
        write_json(manifest_path, manifest)
        spec["freeze"]["artifacts"]["development_manifest"]["sha256"] = sha256_file(manifest_path)
        write_json(path, spec)
    else:
        schema_path = path.parent / "feature_schema.json"
        schema = json.loads(schema_path.read_text())
        schema["primary_features"].reverse()
        write_json(schema_path, schema)
        spec["freeze"]["artifacts"]["feature_schema"]["sha256"] = sha256_file(schema_path)
        write_json(path, spec)
    def forbidden(*args, **kwargs):
        raise AssertionError("Validation failure must precede fit")
    monkeypatch.setattr(models, "forward_select_lda", forbidden)
    with pytest.raises(models.FixedModelError):
        models.fit_development(path, tmp_path / "rejected", require_reference_environment=False)
    assert not (tmp_path / "rejected").exists()


def test_model_bundle_never_overwrites_existing_evidence(fitted):
    before = {p.name: p.read_bytes() for p in (fitted[0] / "bundle").iterdir()}
    with pytest.raises(FileExistsError):
        models.fit_development(fitted[1], fitted[0] / "bundle", require_reference_environment=False)
    assert before == {p.name: p.read_bytes() for p in (fitted[0] / "bundle").iterdir()}


def test_chance_level_fitted_classifier_is_not_a_failed_engineering_gate(tmp_path):
    frame = make_feature_table(["d1", "d2", "d3"], "chance-development")
    schema = json.loads((Path(__file__).resolve().parents[1] / "configs/confirmatory_feature_schema.json").read_text())
    # Identical class-conditional feature distributions yield a rank-zero binary
    # LDA rule. Its valid predictions have chance accuracy; no performance gate.
    values = frame.epoch_index.map({1: 0.0, 2: 1.0, 3: 0.0, 4: 1.0}).to_numpy()
    for name in schema["primary_features"] + schema["reference_features"]:
        frame[name] = values
    path, _, _ = write_development_fixture(tmp_path / "source", frame=frame)
    with np.errstate(invalid="ignore"):
        models.fit_development(path, tmp_path / "bundle", require_reference_environment=False)
    loaded = models.load_fixed_model(tmp_path / "bundle/primary_model.json")
    pred = loaded.predict(frame[list(loaded.feature_order)], comparison="rest1_vs_type1")
    assert (pred == frame.class_label.to_numpy()).mean() == .5
    assert loaded.document["comparisons"]["rest1_vs_type1"]["classifier"]["explained_variance_ratio"] == [None]


def test_every_development_inner_scaler_and_classifier_sees_only_its_training_fold(tmp_path, monkeypatch):
    """Inspect actual SFS and final-fit calls, not only the exported membership."""
    path, _, _ = write_development_fixture(tmp_path / "source")
    _, _, _, frame, _, _ = models._load_development(path)
    original_selector = models.forward_select_lda
    original_scaler = StandardScaler.fit
    original_classifier = LinearDiscriminantAnalysis.fit
    context = {"X": None, "y": None, "allowed": None, "scaled": None}
    counts = {"selection": 0, "inner": 0, "final": 0, "classifier": 0}

    def select(X, y, names, splits, **kwargs):
        splits = list(splits)
        # Locate the exact comparison/role matrix from declared source epochs.
        candidates = [part for _, part in frame.groupby("comparison", sort=True)
                      if np.array_equal(part[list(names)].to_numpy(float), X)]
        assert len(candidates) == 1
        part = candidates[0]
        groups = part.subject_id.to_numpy()
        assert len(splits) == 3
        for train, valid in splits:
            assert set(groups[train]).isdisjoint(groups[valid])
            assert len(set(groups[valid])) == 1
            assert set(part.iloc[valid].session_id) == {"session-1", "session-2"}
        context.update(X=X, y=y, allowed=[train for train, _ in splits] + [np.arange(len(X))], scaled=None)
        counts["selection"] += 1
        return original_selector(X, y, names, splits, **kwargs)

    def scale(self, X, y=None, **kwargs):
        assert context["scaled"] is None
        matches = [(rows, column) for rows in context["allowed"]
                   if len(rows) == len(X)
                   for column in range(context["X"].shape[1])
                   if np.array_equal(context["X"][rows, column:column + 1], X)]
        assert len(matches) == 1, "Scaler did not receive one exact allowed development training subset"
        rows, _ = matches[0]
        np.testing.assert_array_equal(y, context["y"][rows])
        fitted = original_scaler(self, X, y, **kwargs)
        context["scaled"] = (self.transform(X), y.copy())
        counts["final" if len(rows) == len(context["X"]) else "inner"] += 1
        return fitted

    def classify(self, X, y, **kwargs):
        assert context["scaled"] is not None
        wanted_x, wanted_y = context["scaled"]
        np.testing.assert_array_equal(X, wanted_x)
        np.testing.assert_array_equal(y, wanted_y)
        context["scaled"] = None
        counts["classifier"] += 1
        return original_classifier(self, X, y, **kwargs)

    monkeypatch.setattr(models, "forward_select_lda", select)
    monkeypatch.setattr(StandardScaler, "fit", scale)
    monkeypatch.setattr(LinearDiscriminantAnalysis, "fit", classify)
    models.fit_development(path, tmp_path / "bundle", require_reference_environment=False)
    assert counts == {"selection": 4, "inner": (21 + 168) * 2 * 3, "final": 4,
                      "classifier": (21 + 168) * 2 * 3 + 4}
    assert context["scaled"] is None


def test_semantically_wrong_training_labels_are_rejected_even_with_recomputed_ids(tmp_path, monkeypatch):
    from eeg_ma.schema import epoch_identity, validate_feature_table
    frame = make_feature_table(["d1", "d2", "d3"], "synthetic-development")
    frame.loc[frame.class_label == 0, "class_name"] = "MentalArithmetic"
    for index, row in frame.iterrows():
        for key, value in epoch_identity(row, row).items():
            frame.at[index, key] = value
    validate_feature_table(frame)  # Structurally intact, but the scientific label mapping is false.
    path, _, _ = write_development_fixture(tmp_path / "source", frame=frame)
    def forbidden(*args, **kwargs):
        raise AssertionError("Class-mapping failure must precede fitting")
    monkeypatch.setattr(models, "forward_select_lda", forbidden)
    with pytest.raises(models.FixedModelError, match="class mapping"):
        models.fit_development(path, tmp_path / "rejected", require_reference_environment=False)
    assert not (tmp_path / "rejected").exists()


@pytest.mark.parametrize("fault", ["units", "reference", "channel_order", "sample_length"])
def test_raw_training_acquisition_must_match_its_frozen_contract(tmp_path, monkeypatch, fault):
    from eeg_ma.schema import epoch_identity, validate_feature_table
    frame = make_feature_table(["d1", "d2", "d3"], "synthetic-development")
    if fault == "units":
        frame.units = "mV"
    elif fault == "reference":
        frame.reference = "different reference"
    elif fault == "channel_order":
        frame.channel_order = frame.channel_order.map(lambda value: json.dumps(json.loads(value)[::-1]))
    else:
        frame.end_sample -= 1
        for index, row in frame.iterrows():
            for key, value in epoch_identity(row, row).items():
                frame.at[index, key] = value
    validate_feature_table(frame)  # All declared source bytes are internally valid.
    path, _, _ = write_development_fixture(tmp_path / "source", frame=frame)
    def forbidden(*args, **kwargs):
        raise AssertionError("Acquisition mismatch must precede fitting")
    monkeypatch.setattr(models, "forward_select_lda", forbidden)
    with pytest.raises(models.FixedModelError, match="acquisition metadata"):
        models.fit_development(path, tmp_path / "rejected", require_reference_environment=False)
    assert not (tmp_path / "rejected").exists()


@pytest.mark.parametrize("fault", ["code", "protocol", "config", "features"])
def test_source_mutation_during_fit_cannot_publish_stale_provenance(fitted, tmp_path, monkeypatch, fault):
    path, _, _ = write_development_fixture(tmp_path / "source")
    original_identity = models.code_identity()
    mutation = {"happened": False}

    def identity():
        result = deepcopy(original_identity)
        if fault == "code" and mutation["happened"]:
            result["commit"] = "f" * 40
        return result

    def reuse_fitted(frame, names, cfg):
        # Reuse actual fitted numerical state to focus this regression on the
        # post-fit publication boundary, without repeating 1,134 SFS fits.
        role = "primary" if len(names) == 21 else "reference"
        comparison = frame.comparison.iloc[0]
        offset = (0 if role == "primary" else 2) + (0 if comparison == "rest1_vs_type1" else 1)
        if not mutation["happened"] and fault != "code":
            source = {"protocol": path, "config": path.parent / "analysis_config.json",
                      "features": path.parent / "baseline.csv"}[fault]
            source.write_bytes(source.read_bytes() + b"\n")
        mutation["happened"] = True
        return _model(fitted, role).document["comparisons"][comparison], fitted[5][offset]

    monkeypatch.setattr(models, "code_identity", identity)
    monkeypatch.setattr(models, "_fit_comparison", reuse_fitted)
    with pytest.raises(models.FixedModelError, match="changed during"):
        models.fit_development(path, tmp_path / "rejected", require_reference_environment=False)
    assert mutation["happened"]
    assert not (tmp_path / "rejected").exists()
