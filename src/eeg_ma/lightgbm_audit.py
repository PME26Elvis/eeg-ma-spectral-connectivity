"""Source-bound replay for handwritten binary histogram boosting archives.

The verifier reconstructs training bins and sampling, then checks each stored
tree against derivatives of the preceding ensemble on the original training
rows.  It also replays held-out predictions.  Inner candidate records are
checked for membership, arithmetic and selection consistency; this is not an
inner-model refit or a proof that every greedy split was globally optimal.
"""
from __future__ import annotations

from fractions import Fraction
import json
import math

import numpy as np
import pandas as pd

from .classifier_config import classifier_policy, lightgbm_constructor_parameters
from .provenance import (
    ProvenanceError, _read_json, _same_table, canonical_json,
    read_csv_literal, sha256_file,
)


VERIFICATION_SCOPE = (
    "source_bound_training_bins_derivatives_sampling_tree_statistics_and_outer_prediction_replay_"
    "plus_inner_selection_record_consistency_not_inner_refit_or_greedy_optimality"
)
IMPORTANCE_SCOPE = "training_newton_gain_not_causal_or_test_permutation_importance"


def _equal(observed, expected, name):
    if canonical_json(observed) != canonical_json(expected):
        raise ProvenanceError(f"LightGBM {name} differs")


def _array(value, name):
    """Reject raw JSON booleans/strings before NumPy can coerce them."""
    def check(item):
        if isinstance(item, (list, tuple)):
            for child in item:
                check(child)
        elif isinstance(item, np.ndarray):
            if item.dtype.kind not in "iuf":
                raise ProvenanceError(f"LightGBM {name} must contain only numbers")
        elif isinstance(item, (bool, np.bool_)) or not isinstance(item, (int, float, np.integer, np.floating)):
            raise ProvenanceError(f"LightGBM {name} must contain only numbers")
    check(value)
    result = np.asarray(value, dtype=np.float64)
    if not np.isfinite(result).all():
        raise ProvenanceError(f"LightGBM {name} must be finite")
    return result


def _numeric(observed, expected, name):
    actual, target = _array(observed, name), _array(expected, name)
    if actual.shape != target.shape or not np.allclose(actual, target, rtol=2e-12, atol=2e-14):
        raise ProvenanceError(f"LightGBM {name} differs from source replay")


def _sigmoid(raw):
    raw = np.asarray(raw, dtype=np.float64)
    result = np.empty_like(raw)
    positive = raw >= 0
    result[positive] = 1 / (1 + np.exp(-raw[positive]))
    exponential = np.exp(raw[~positive])
    result[~positive] = exponential / (1 + exponential)
    return result


def _probabilities(probabilities, raw_scores, classes, predictions, n_rows, name):
    values, raw = _array(probabilities, name), _array(raw_scores, name)
    if (values.shape != (n_rows, 2) or raw.shape != (n_rows,)
            or np.any(values < 0) or np.any(values > 1)):
        raise ProvenanceError(f"LightGBM {name} has invalid binary probability/score dimensions")
    positive = _sigmoid(raw)
    _numeric(values, np.column_stack((1 - positive, positive)), f"{name} sigmoid")
    _equal(predictions, np.asarray(classes)[np.argmax(values, axis=1)].tolist(), f"{name} argmax")
    return values, raw


def _weights(y, parameters):
    weights = np.ones(len(y), dtype=np.float64)
    class_weight = parameters["class_weight"]
    if class_weight == "balanced":
        counts = np.bincount(y, minlength=2)
        weights = len(y) / (2 * counts[y])
    elif isinstance(class_weight, dict):
        weights = np.asarray([class_weight[int(label)] for label in y], dtype=np.float64)
    return weights


def _loss(raw, y, weights):
    losses = np.maximum(raw, 0) - y * raw + np.log1p(np.exp(-np.abs(raw)))
    total = math.fsum(weights.tolist())
    # Raw weighted losses can overflow even when their weighted mean is a
    # representable number. Normalize before multiplying, as the objective
    # defines, and use compensated sums for cancellation-sensitive evidence.
    return math.fsum((weights / total * losses).tolist())


def _verify_training(model, X, y):
    """Verify stored node evidence against source data and ensemble prefixes."""
    from .lightgbm import (
        HESSIAN_FLOOR, iteration_seed_sequence, sampled_features, sampling_plan,
    )
    from .lightgbm_histogram import BinMapper, FeatureBundler
    parameters = model.get_params(deep=False)
    weights = _weights(y, parameters)
    eligible = np.flatnonzero(weights > 0)
    serialized = model.to_dict()
    _equal(serialized["training_n_samples"], len(y), "source training sample count")
    _numeric(serialized["class_weight_factors"],
             [float(weights[y == label][0]) for label in (0, 1)], "source class-weight factors")
    _numeric(serialized["training_class_mass"],
             [math.fsum(weights[y == label].tolist()) for label in (0, 1)], "source effective class mass")
    _equal(model.training_eligible_indices_.tolist(), eligible.tolist(), "eligible training rows")
    mapper = BinMapper.fit(X[eligible], max_bin=parameters["max_bin"],
                           min_data_in_bin=parameters["min_data_in_bin"])
    _equal(model.bin_mapper_.to_dict(), mapper.to_dict(), "training-only bin thresholds")
    bins = mapper.transform(X)
    layout = FeatureBundler.fit(bins[eligible], mapper, enable_bundle=parameters["enable_bundle"])
    _equal(model.bundle_layout_.to_dict(), layout.to_dict(), "training-only exclusive feature bundles")
    positive, negative = math.fsum(weights[y == 1].tolist()), math.fsum(weights[y == 0].tolist())
    base_score = math.log(positive) - math.log(negative)
    _numeric(model.base_score_, base_score, "weighted training prior")
    raw = np.full(len(y), base_score, dtype=np.float64)
    losses = [_loss(raw, y, weights)]
    trees = model.estimators_
    if model.best_iteration_ != len(trees) or model.validation_loss_ is not None:
        raise ProvenanceError("Integrated LightGBM archive must use every fitted tree without early stopping")
    # Only materialized trees need seeds. A legitimate prior-only model can
    # stop after one attempt despite an arbitrarily large requested budget.
    expected_seeds = iteration_seed_sequence(parameters["random_state"], len(trees))
    _equal(np.asarray(model.iteration_seeds_).tolist(), np.asarray(expected_seeds)[:len(trees)].tolist(), "iteration seed sequence")
    if len(model.sampling_records_) != len(trees) or len(model.feature_subsets_) != len(trees):
        raise ProvenanceError("LightGBM iteration audit coverage differs from fitted trees")
    gains = []
    split_counts = np.zeros(X.shape[1], dtype=np.int64)
    for iteration, tree in enumerate(trees):
        if (int(np.count_nonzero(tree.children_left_ < 0)) > parameters["num_leaves"]
                or (parameters["max_depth"] is not None and tree.max_depth > parameters["max_depth"])):
            raise ProvenanceError("LightGBM tree exceeds its declared leaf or depth limit")
        probability = _sigmoid(raw)
        complement = _sigmoid(-raw)
        gradients = np.where(y == 1, -complement, probability) * weights
        hessians = np.maximum(probability * complement, HESSIAN_FLOOR) * weights
        seed = int(expected_seeds[iteration])
        planned = sampling_plan(eligible, gradients, parameters, seed)
        _equal(model.sampling_records_[iteration], planned, "source-derived row sampling")
        selected_features = sampled_features(X.shape[1], parameters["feature_fraction"], seed)
        _equal(np.asarray(model.feature_subsets_[iteration]).tolist(), np.asarray(selected_features).tolist(), "iteration feature sample")
        indices = np.asarray(planned["sampled_indices"], dtype=np.int64)
        factors = np.asarray(planned["multipliers"], dtype=np.float64)
        sampled_gradient, sampled_hessian = gradients[indices] * factors, hessians[indices] * factors
        routes = [(0, np.arange(len(indices), dtype=np.int64), 0)]
        while routes:
            node, members, depth = routes.pop()
            _equal(int(tree.depth_[node]), depth, "tree path depth")
            G = math.fsum(sampled_gradient[members].tolist())
            H = math.fsum(sampled_hessian[members].tolist())
            _equal(int(tree.n_node_samples_[node]), len(members), "sampled node row count")
            _numeric(tree.sum_gradient_[node], G, "source-derived node gradient")
            _numeric(tree.sum_hessian_[node], H, "source-derived node Hessian")
            shrunk_gradient = math.copysign(max(abs(G) - parameters["lambda_l1"], 0.), G)
            denominator = math.fsum((H, parameters["lambda_l2"]))
            step = -shrunk_gradient / denominator
            if parameters["max_delta_step"] > 0:
                step = float(np.clip(step, -parameters["max_delta_step"], parameters["max_delta_step"]))
            _numeric(tree.value_[node], step, "Newton node value")
            left, right = int(tree.children_left_[node]), int(tree.children_right_[node])
            if left < 0:
                continue
            feature, threshold = int(tree.feature_[node]), int(tree.threshold_bin_[node])
            if feature not in selected_features:
                raise ProvenanceError("LightGBM split uses a feature outside its sampled subset")
            _numeric(tree.threshold_[node], mapper.thresholds_[feature][threshold], "raw split threshold")
            mask = bins[indices[members], feature] <= threshold
            left_members, right_members = members[mask], members[~mask]
            for child, child_members in ((left, left_members), (right, right_members)):
                if (len(child_members) < parameters["min_data_in_leaf"]
                        or math.fsum(sampled_hessian[child_members].tolist()) < parameters["min_sum_hessian_in_leaf"]):
                    raise ProvenanceError("LightGBM split violates sampled count or Hessian constraints")
                routes.append((child, child_members, depth + 1))
            def objective(child):
                magnitude = abs(float(tree.value_[child]))
                softened = max(abs(float(tree.sum_gradient_[child])) - parameters["lambda_l1"], 0.)
                denominator = math.fsum((float(tree.sum_hessian_[child]), parameters["lambda_l2"]))
                # Equivalent to -2Gv-(H+L2)v²-2L1|v|, avoiding cancellation
                # of large G/L1 terms and overflow of intermediate squares.
                return math.fsum((magnitude * softened,
                                  magnitude * (softened - denominator * magnitude)))
            left_score, right_score, parent_score = objective(left), objective(right), objective(node)
            gain = math.fsum((left_score, right_score, -parent_score))
            _numeric(tree.gain_[node], gain, "Newton split gain")
            cancellation_floor = 128 * np.finfo(float).eps * math.fsum((abs(left_score), abs(right_score), abs(parent_score)))
            if gain <= max(parameters["min_gain_to_split"], cancellation_floor):
                raise ProvenanceError("LightGBM accepted a nonpositive or subthreshold split gain")
            _numeric(math.fsum((float(tree.sum_gradient_[left]), float(tree.sum_gradient_[right]))), G, "parent-child gradient conservation")
            _numeric(math.fsum((float(tree.sum_hessian_[left]), float(tree.sum_hessian_[right]))), H, "parent-child Hessian conservation")
            gains.append((feature, float(tree.gain_[node])))
            split_counts[feature] += 1
        raw += parameters["learning_rate"] * tree.predict_binned(bins)
        losses.append(_loss(raw, y, weights))
    _numeric(model.training_loss_, losses, "training log-loss curve")
    _numeric(model.decision_function(X), raw, "full training score replay")
    importance = np.zeros(X.shape[1], dtype=np.float64)
    if gains:
        maximum_gain = max(gain for _, gain in gains)
        buckets = [[] for _ in range(X.shape[1])]
        for feature, gain in gains:
            buckets[feature].append(gain / maximum_gain)
        importance = np.asarray([math.fsum(bucket) for bucket in buckets])
        importance /= math.fsum(importance.tolist())
    _numeric(model.feature_importances_, importance, "training gain importances")
    _equal(model.feature_split_counts_.tolist(), split_counts.tolist(), "training feature split counts")


def validate_lightgbm_evidence(folder, scope, identity, folds, predictions):
    from .lightgbm import ALGORITHM_VERSION, HandcraftedLightGBMClassifier
    from .models import lightgbm_seed
    from .schema import validate_feature_table
    source = folder / "lightgbm_inputs.csv"
    if not source.is_file() or sha256_file(source) != identity["dataset"]["table_sha256"]:
        raise ProvenanceError("LightGBM source tensor differs from the run input table identity")
    frame = read_csv_literal(source, float_precision="round_trip")
    validate_feature_table(frame)
    if list(frame.columns) != identity["dataset"]["columns"]:
        raise ProvenanceError("LightGBM source columns differ from the run input schema")
    _same_table(frame, pd.DataFrame(identity["dataset"]["metadata"]), ["epoch_id"], "LightGBM source epoch metadata")
    indexed = frame.set_index("epoch_id", drop=False)
    policy = classifier_policy(identity["config"])
    boosting_policy = policy["lightgbm"]
    audit = _read_json(folder / "lightgbm_audit.json")
    if not isinstance(audit, dict) or set(audit) != {"schema_version", "scope", "classifier_policy", "input_table_sha256", "verification_scope", "folds"}:
        raise ProvenanceError("LightGBM audit schema differs")
    _equal(audit["schema_version"], "eeg-lightgbm-cv-1", "audit schema")
    _equal(audit["scope"], scope, "audit scope")
    _equal(audit["classifier_policy"], policy, "audit classifier policy")
    _equal(audit["input_table_sha256"], identity["dataset"]["table_sha256"], "audit source identity")
    _equal(audit["verification_scope"], VERIFICATION_SCOPE, "verification scope")
    records, splits = audit["folds"], identity["splits"][scope]
    if not isinstance(records, list) or len(records) != len(splits):
        raise ProvenanceError("LightGBM audit does not cover every outer fold")
    unit = "subject_id" if scope == "intra" else "held_out_subject"
    expected_fields = {"algorithm_version", "context", "search_policy", "candidates", "best_candidate_index", "best_score", "best_params", "training_feature_importances", "feature_importance_scope", "model", "identity_status", "selected_features", "train_epoch_ids", "test_epoch_ids", "test_classes", "test_predictions", "test_probabilities", "test_raw_scores"}
    for record, split in zip(records, splits):
        if not isinstance(record, dict) or set(record) != expected_fields:
            raise ProvenanceError("LightGBM outer-fold audit fields differ")
        context = {"scope": scope, "comparison": split["comparison"], "subject": split[unit], "outer_fold": split["outer_fold"]}
        _equal(record["context"], context, "outer-fold identity")
        _equal(record["algorithm_version"], ALGORITHM_VERSION, "algorithm version")
        _equal(record["search_policy"], boosting_policy, "search policy")
        _equal(record["identity_status"], "schema_validated", "source identity status")
        for field in ("train_epoch_ids", "test_epoch_ids"):
            _equal(record[field], split[field], field)
        selected_fold = folds.loc[(folds.classifier == "LightGBM") & (folds[unit].astype(str) == str(split[unit]))
            & (folds.comparison == split["comparison"]) & (folds.outer_fold == split["outer_fold"])].iloc[0]
        names = str(selected_fold.selected_features).split(";")
        _equal(record["selected_features"], names, "selected features")
        train_ids, test_ids = split["train_epoch_ids"], split["test_epoch_ids"]
        Xtrain, Xtest = indexed.loc[train_ids, names].to_numpy(dtype=float), indexed.loc[test_ids, names].to_numpy(dtype=float)
        ytrain, ytest = indexed.loc[train_ids, "class_label"].to_numpy(dtype=int), indexed.loc[test_ids, "class_label"].to_numpy(dtype=int)
        positions = {epoch: index for index, epoch in enumerate(train_ids)}
        candidates = record["candidates"]
        if not isinstance(candidates, list) or len(candidates) != len(boosting_policy["candidates"]):
            raise ProvenanceError("LightGBM audit candidate matrix is incomplete")
        means = []
        for index, (candidate, parameters) in enumerate(zip(candidates, boosting_policy["candidates"])):
            if not isinstance(candidate, dict) or set(candidate) != {"candidate_index", "parameters", "folds", "mean_inner_accuracy"}:
                raise ProvenanceError("LightGBM candidate audit fields differ")
            _equal(candidate["candidate_index"], index, "candidate order")
            _equal(candidate["parameters"], parameters, "candidate parameters")
            if not isinstance(candidate["folds"], list) or len(candidate["folds"]) != len(split["inner_splits"]):
                raise ProvenanceError("LightGBM candidate inner fold coverage differs")
            scores = []
            for inner_index, (observed, inner) in enumerate(zip(candidate["folds"], split["inner_splits"]), start=1):
                if not isinstance(observed, dict) or set(observed) != {"inner_fold", "seed", "train_indices", "test_indices", "classes", "predictions", "probabilities", "raw_scores", "accuracy", "early_stopping_used"}:
                    raise ProvenanceError("LightGBM inner-fold fields differ")
                a = [positions[epoch] for epoch in inner["train_epoch_ids"]]
                b = [positions[epoch] for epoch in inner["test_epoch_ids"]]
                _equal(observed["inner_fold"], inner_index, "inner fold order")
                _equal(observed["train_indices"], a, "inner training membership")
                _equal(observed["test_indices"], b, "inner validation membership")
                _equal(observed["seed"], lightgbm_seed(boosting_policy["parameters"]["random_state"], context, "inner", inner_index), "inner seed")
                _equal(observed["early_stopping_used"], False, "inner early-stopping policy")
                classes = np.unique(ytrain[a]).tolist()
                _equal(observed["classes"], classes, "inner classes")
                _probabilities(observed["probabilities"], observed["raw_scores"], classes, observed["predictions"], len(b), "inner predictions")
                score = Fraction(int(np.count_nonzero(np.asarray(observed["predictions"]) == ytrain[b])), len(b))
                _numeric(observed["accuracy"], float(score), "inner accuracy")
                scores.append(score)
            mean = sum(scores, Fraction(0)) / len(scores)
            _numeric(candidate["mean_inner_accuracy"], float(mean), "candidate mean accuracy")
            means.append(mean)
        best = max(range(len(means)), key=means.__getitem__)
        _equal(record["best_candidate_index"], best, "first best candidate")
        _numeric(record["best_score"], float(means[best]), "chosen inner accuracy")
        _numeric(selected_fold.grid_inner_accuracy, float(means[best]), "fold inner accuracy")
        chosen = {**boosting_policy["candidates"][best], "random_state": lightgbm_seed(boosting_policy["parameters"]["random_state"], context, "refit")}
        _equal(record["best_params"], chosen, "refit parameters")
        _equal(json.loads(selected_fold.best_params), chosen, "fold refit parameters")
        model = HandcraftedLightGBMClassifier.from_dict(record["model"])
        _equal(model.get_params(deep=False), lightgbm_constructor_parameters(chosen), "stored model parameters")
        _equal(model.classes_.tolist(), np.unique(ytrain).tolist(), "model classes")
        if model.n_features_in_ != len(names):
            raise ProvenanceError("LightGBM model features differ from the selected subset")
        _verify_training(model, Xtrain, ytrain)
        _equal(record["test_classes"], model.classes_.tolist(), "test classes")
        probabilities, raw = _probabilities(record["test_probabilities"], record["test_raw_scores"],
            model.classes_.tolist(), record["test_predictions"], len(test_ids), "outer predictions")
        _numeric(raw, model.decision_function(Xtest), "source-bound outer raw scores")
        _numeric(probabilities, model.predict_proba(Xtest), "source-bound outer probabilities")
        _equal(record["test_predictions"], model.predict(Xtest).tolist(), "source-bound outer predictions")
        published = predictions.loc[predictions.classifier == "LightGBM"].set_index("epoch_id").loc[test_ids]
        _equal(published.predicted_label.tolist(), record["test_predictions"], "published outer predictions")
        _numeric(selected_fold.accuracy, float(np.mean(model.predict(Xtest) == ytest)), "source-bound outer accuracy")
        _numeric(record["training_feature_importances"], model.feature_importances_, "training gain importance")
        _equal(record["feature_importance_scope"], IMPORTANCE_SCOPE, "importance scope")
