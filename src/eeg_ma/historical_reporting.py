"""Reconstruct descriptive historical LOPO reports without refitting any model.

Legacy row positions are resolved only through an explicitly supplied feature
mapping.  The resulting content IDs describe observed archives; they do not
recover missing historical code, environments, raw hashes or fitted models.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
from io import StringIO
import json
from pathlib import Path
import re
from typing import Sequence

import numpy as np
import pandas as pd

from .artifacts import new_output_directory
from .constants import DEFAULT_CHANNELS, channel_pairs
from .extract import feature_columns
from .provenance import (PROJECT_ROOT, fingerprint, input_inventory, read_csv_literal,
                         protect_output, resolve_report_root, sha256_file, validate_complete_run)
from .result_contract import (EPOCH_COLUMNS, RESULT_GROUP_COLUMNS, ResultContractError,
    paired_participant_deltas, participant_results, summarize_participants,
    read_result_csv, validate_predictions, validate_result_matrix)
from .schema import META_COLUMNS, assert_same_epochs, load_feature_table


CLASSIFIERS = ("KFDA", "LDA", "RBF-SVM")
COMPARISONS = ("rest1_vs_type1", "rest2_vs_type2")
BAND_EXPERIMENTS = tuple(f"plv_{band}_only" for band in
                         ("delta", "theta", "alpha", "beta_low", "beta_high", "gamma"))
FINAL_EXPERIMENTS = ("poster_baseline", "plv_all_bands", *BAND_EXPERIMENTS, "iplv_only")
STORY_EXPERIMENTS = ("poster_baseline", "e1_asymmetry", "e2_plv", "plv_all_bands", "plv_alpha_only", "iplv_only")
HISTORICAL_EXPERIMENTS = (*FINAL_EXPERIMENTS, "e1_asymmetry", "e2_plv")
SOURCE_ROOTS = {
    "poster_baseline": (".", "baseline"),
    "plv_all_bands": ("plv_decomposition/e2a_plv_only", "extension"),
    "e1_asymmetry": ("extensions/e1_asymmetry", "extension"),
    "e2_plv": ("extensions/e2_plv", "extension"),
    **{name: (f"final_validation/{name}", "extension") for name in BAND_EXPERIMENTS},
    "iplv_only": ("final_validation/iplv_only", "iplv"),
}
REPORT_SCHEMA = "eeg-historical-participant-report-1"
REPORT_AGGREGATION = "equal_participant_weight; sessions pooled within participant"
REPORT_UNCERTAINTY = "between-participant sample SD (ddof=1), descriptive; no LOPO confidence interval"
REPORT_BA_RULE = "Requires both true classes; otherwise missing, never replaced by chance"
REPORT_METRICS = ["participant accuracy", "participant balanced accuracy", "class recalls", "confusion counts"]
FEATURE_SETS = {"poster_baseline": "baseline", "plv_all_bands": "plv", "e1_asymmetry": "baseline_asym",
    "e2_plv": "baseline_plv", "iplv_only": "iplv",
    **{name: "plv_band_" + name.removeprefix("plv_").removesuffix("_only") for name in BAND_EXPERIMENTS}}
SELECTION_COLUMNS = ["held_out_subject", "comparison", "classifier", "outer_fold", "accuracy", "selected_features", "n_selected_features"]


@dataclass
class HistoricalReport:
    predictions: pd.DataFrame
    eligible_epochs: pd.DataFrame
    participants: pd.DataFrame
    summary: pd.DataFrame
    paired_deltas: pd.DataFrame
    alpha_frequency: pd.DataFrame
    metadata: dict
    alpha_training_selections: pd.DataFrame = field(default_factory=lambda: pd.DataFrame(columns=SELECTION_COLUMNS))

    def tables(self) -> dict[str, pd.DataFrame]:
        return {"predictions.csv": self.predictions,
                "eligible_epochs.csv": self.eligible_epochs,
                "participant_results.csv": self.participants,
                "historical_summary.csv": self.summary,
                "paired_participant_deltas.csv": self.paired_deltas,
                "alpha_feature_frequency.csv": self.alpha_frequency,
                "alpha_training_selections.csv": self.alpha_training_selections}


def _equal_table(left: pd.DataFrame, right: pd.DataFrame, keys: list[str], label: str) -> None:
    if left.duplicated(keys).any() or right.duplicated(keys).any():
        raise ResultContractError(f"{label} contains duplicate keys")
    try:
        pd.testing.assert_frame_equal(left.sort_values(keys).reset_index(drop=True)[right.columns],
            right.sort_values(keys).reset_index(drop=True), check_dtype=False, atol=1e-12, rtol=1e-10)
    except (AssertionError, KeyError) as exc:
        raise ResultContractError(f"{label} differs from validated prediction-derived values or exact epoch identity") from exc


def _managed_matrices(root: Path, *, partial: bool) -> dict:
    parents = {}
    for family in ("extensions", "plv_decomposition", "final_validation"):
        folder = root / family
        if (folder / "CURRENT.json").exists() or (folder / "matrix_manifest.json").exists():
            report = resolve_report_root(folder)
            manifest = validate_complete_run(report)
            coverage = manifest["identity"].get("coverage", {})
            if not partial and not coverage.get("study_matrix_complete", False):
                raise ResultContractError(f"{family} manifest describes an incomplete study matrix")
            parents[family] = {"run_id": manifest["run_id"], "manifest_sha256": sha256_file(report / "run_manifest.json")}
    return parents


def _read_required(path: Path) -> pd.DataFrame:
    if not path.is_file() or path.is_symlink():
        raise ResultContractError(f"Missing required prediction/selection source: {path}")
    payload = path.read_bytes()
    frame = read_csv_literal(StringIO(payload.decode("utf-8-sig")))
    frame.attrs["source_sha256"] = hashlib.sha256(payload).hexdigest()
    return frame


def _validate_representation(declaration: dict, experiment: str) -> bool:
    if not isinstance(declaration, dict):
        raise ResultContractError("Observed experiment/config declaration must be an object")
    accepted_names = {experiment}
    if experiment == "poster_baseline":
        accepted_names.add("baseline_inter")
    if experiment == "plv_all_bands":
        accepted_names.add("e2a_plv_only")
    descriptions = [declaration]
    for key in ("config", "experiment"):
        if key in declaration:
            if not isinstance(declaration[key], dict):
                raise ResultContractError("Observed experiment/config declaration must be an object")
            descriptions.append(declaration[key])
    observed = False
    for item in descriptions:
        if "feature_set" in item:
            observed = True
            actual = "baseline" if item["feature_set"] == "all" else item["feature_set"]
            if actual != FEATURE_SETS[experiment]:
                raise ResultContractError(f"{experiment} representation/feature_set contradicts its declared report role")
    named = declaration.get("experiment", {})
    if "name" in named and named["name"] not in accepted_names:
        raise ResultContractError(f"{experiment} experiment name contradicts its declared report role")
    return observed


def _adapt_source(root: Path, experiment: str, features: pd.DataFrame, *,
                  analysis_id: str, eligible: pd.DataFrame, allow_legacy: bool,
                  sources: dict[str, Path], candidate_features: Sequence[str] | None = None,
                  source_reads: dict[str, str] | None = None) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    prediction_path = root / "inter/inter_predictions.csv"
    fold_path = root / "inter/inter_folds.csv"
    raw = _read_required(prediction_path)
    folds = _read_required(fold_path)
    sources[f"{experiment}/predictions"] = prediction_path
    sources[f"{experiment}/folds"] = fold_path
    if source_reads is not None:
        source_reads[f"{experiment}/predictions"] = raw.attrs["source_sha256"]
        source_reads[f"{experiment}/folds"] = folds.attrs["source_sha256"]
    parent_root = root if (root / "run_manifest.json").exists() else root / "inter"
    if (parent_root / "run_manifest.json").exists():
        manifest = validate_complete_run(parent_root)
        if manifest["identity"].get("kind") != "evaluation" or "inter" not in manifest["identity"].get("scopes", []):
            raise ResultContractError("Prediction parent manifest is not an inter-subject evaluation")
        _validate_representation(manifest["identity"], experiment)
        if candidate_features is not None and set(manifest["identity"]["feature_order"]) != set(candidate_features):
            raise ResultContractError(f"{experiment} managed candidate feature order belongs to a different representation")
        parent_metadata = pd.DataFrame(manifest["identity"]["dataset"]["metadata"])
        _equal_table(parent_metadata, features[META_COLUMNS], ["epoch_id"], "Parent cohort/eligible epoch metadata")
        parent = {"status": "complete", "run_id": manifest["run_id"],
                  "manifest_sha256": sha256_file(parent_root / "run_manifest.json")}
        sources[f"{experiment}/manifest"] = parent_root / "run_manifest.json"
        if source_reads is not None:
            source_reads[f"{experiment}/manifest"] = parent["manifest_sha256"]
    else:
        if not allow_legacy:
            raise ResultContractError("Historical sources lack a run manifest; explicitly allow legacy results")
        config = root / "config_snapshot.json"
        observed = {"predictions_sha256": raw.attrs["source_sha256"], "folds_sha256": folds.attrs["source_sha256"]}
        if config.exists():
            try:
                config_bytes = config.read_bytes()
                declaration = json.loads(config_bytes.decode("utf-8"))
            except (OSError, ValueError) as exc:
                raise ResultContractError("Unreadable historical config declaration") from exc
            _validate_representation(declaration, experiment)
            observed["config_snapshot_sha256"] = hashlib.sha256(config_bytes).hexdigest()
            sources[f"{experiment}/config_snapshot"] = config
            if source_reads is not None:
                source_reads[f"{experiment}/config_snapshot"] = observed["config_snapshot_sha256"]
        parent = {"status": "legacy_unverified", "run_id": "legacy-unknown-run-" + fingerprint(observed),
            "observed_sources": observed, "historical_code": "unknown", "historical_environment": "unknown",
            "raw_hashes": "unknown", "serialized_models": "unavailable",
            "identity_meaning": "Content IDs identify observed archives; historical model and execution identities remain unknown."}
    required = {"subject_id", "held_out_subject", "comparison", "outer_fold", "classifier", "true_label", "predicted_label"}
    if required - set(raw):
        raise ResultContractError(f"{experiment} prediction schema is missing {sorted(required - set(raw))}")
    if not raw.subject_id.eq(raw.held_out_subject).all():
        raise ResultContractError(f"{experiment} has a foreign held-out subject/cohort")
    fold_numbers = pd.to_numeric(raw.outer_fold, errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(fold_numbers).all() or np.any(fold_numbers < 1) or np.any(fold_numbers != np.floor(fold_numbers)):
        raise ResultContractError(f"{experiment} has invalid LOPO fold identifiers")
    if not set(raw.comparison) <= set(COMPARISONS) or not set(raw.classifier) <= set(CLASSIFIERS):
        raise ResultContractError(f"{experiment} has an unexpected comparison or classifier")
    fold_keys = ["held_out_subject", "comparison", "classifier"]
    if not set(fold_keys + ["outer_fold", "accuracy", "selected_features"]) <= set(folds) or folds.duplicated(fold_keys).any():
        raise ResultContractError(f"{experiment} has missing or duplicate participant fold keys")
    if candidate_features is not None:
        candidates = set(candidate_features)
        for fold in folds.to_dict("records"):
            selected = fold["selected_features"].split(";")
            if not selected or len(selected) != len(set(selected)) or set(selected) - candidates or fold.get("n_selected_features") != len(selected):
                raise ResultContractError(f"{experiment} training selection is duplicate or outside its declared candidate representation")
    if (raw.groupby(fold_keys, dropna=False).outer_fold.nunique() != 1).any():
        raise ResultContractError(f"{experiment} splits a held-out participant across multiple LOPO folds")
    memberships = raw[["comparison", "outer_fold", "held_out_subject"]].drop_duplicates()
    if memberships.duplicated(["comparison", "outer_fold"]).any():
        raise ResultContractError(f"{experiment} uses one LOPO fold for multiple participants")
    observed_folds = raw[fold_keys + ["outer_fold"]].drop_duplicates()
    _equal_table(folds, observed_folds, fold_keys, f"{experiment} fold membership")
    lookup = features.set_index("epoch_id", drop=False)
    rows = []
    local_tables = {comparison: features.loc[features.comparison.eq(comparison)].reset_index(drop=True)
                    for comparison in COMPARISONS}
    manifest_ids = eligible.groupby("comparison").eligible_epoch_manifest_id.first().to_dict()
    for row in raw.to_dict("records"):
        if "epoch_id" in raw:
            if row["epoch_id"] not in lookup.index:
                raise ResultContractError(f"{experiment} predicts an unknown epoch or wrong reference cohort")
            epoch = lookup.loc[row["epoch_id"]]
            for column in set(META_COLUMNS) & set(raw):
                if column == "onset_elapsed_s":
                    if not np.isclose(row[column], epoch[column], atol=1e-9, rtol=0):
                        raise ResultContractError(f"{experiment} prediction epoch metadata differs")
                elif row[column] != epoch[column]:
                    raise ResultContractError(f"{experiment} prediction epoch/cohort metadata differs: {column}")
        else:
            if parent["status"] != "legacy_unverified" or "global_row_index" not in raw:
                raise ResultContractError(f"{experiment} has no stable epoch ID or explicit legacy row mapping")
            if not features.identity_origin.eq("legacy_manifest").all():
                raise ResultContractError("Historical positional predictions require an explicit legacy feature mapping, not a newly declared reference cohort")
            index = row["global_row_index"]
            local = local_tables[row["comparison"]]
            if not isinstance(index, (int, np.integer)) or index < 0 or index >= len(local):
                raise ResultContractError(f"{experiment} has an invalid comparison-local historical epoch index")
            epoch = local.iloc[index]
        for column in set(META_COLUMNS) & set(raw):
            if column == "onset_elapsed_s":
                matches = np.isclose(row[column], epoch[column], atol=1e-9, rtol=0)
            else:
                matches = row[column] == epoch[column]
            if not matches:
                raise ResultContractError(f"{experiment} prediction epoch/cohort metadata differs: {column}")
        if row["subject_id"] != epoch.subject_id or row["comparison"] != epoch.comparison or row["true_label"] != epoch.class_label:
            raise ResultContractError(f"{experiment} has wrong reference cohort, subject, label or historical epoch ordering")
        model_identity = {"run_id": parent["run_id"], "held_out_subject": row["subject_id"],
                          "comparison": row["comparison"], "classifier": row["classifier"], "outer_fold": row["outer_fold"]}
        rows.append({"analysis_id": analysis_id, "run_id": parent["run_id"], "experiment": experiment,
            "classifier": row["classifier"], **{c: epoch[c] for c in ("cohort_id", "protocol_id", "comparison", "subject_id", "session_id", "epoch_id")},
            "model_id": ("legacy-unknown-model-" if parent["status"] == "legacy_unverified" else "lopo-model-content-") + fingerprint(model_identity),
            "outer_fold": int(row["outer_fold"]),
            "y_true": row["true_label"], "y_pred": row["predicted_label"],
            "eligible_epoch_manifest_id": manifest_ids[row["comparison"]]})
    canonical = pd.DataFrame(rows)
    expected_groups = pd.DataFrame([{ "analysis_id": analysis_id, "run_id": parent["run_id"], "experiment": experiment,
        "cohort_id": features.cohort_id.iloc[0], "protocol_id": features.protocol_id.iloc[0], "comparison": comparison, "classifier": classifier}
        for comparison in COMPARISONS for classifier in CLASSIFIERS])
    canonical = validate_predictions(canonical, eligible_epochs=eligible, expected_groups=expected_groups)
    units = participant_results(canonical)
    recomputed = units.rename(columns={"subject_id": "held_out_subject"})[fold_keys + ["accuracy"]]
    _equal_table(folds, recomputed, fold_keys, f"{experiment} archived fold accuracy")
    if sha256_file(prediction_path) != raw.attrs["source_sha256"] or sha256_file(fold_path) != folds.attrs["source_sha256"]:
        raise ResultContractError(f"{experiment} prediction/fold source bytes changed during validation")
    return canonical, folds, parent


def _frequency_from_folds(folds: pd.DataFrame) -> pd.DataFrame:
    # SFS is shared by the three classifiers.  Each held-out subject contributes
    # at most once to a selected feature, not three times and not once per epoch.
    keys = ["held_out_subject", "comparison"]
    if (folds.groupby(keys).selected_features.nunique() != 1).any():
        raise ResultContractError("Alpha classifiers disagree on their shared training SFS selection")
    records = []
    for row in folds.drop_duplicates(keys).to_dict("records"):
        names = row["selected_features"].split(";")
        if not names or len(names) != len(set(names)) or any(not name.startswith("PLV__") or not name.endswith("__alpha") for name in names):
            raise ResultContractError("Alpha fold selections are duplicate or outside the alpha PLV family")
        records.extend({"comparison": row["comparison"], "feature": name} for name in names)
    return pd.DataFrame(records).groupby(["comparison", "feature"]).size().reset_index(name="selection_count")


def _alpha_frequency(root: Path, folds: pd.DataFrame, sources: dict[str, Path], source_reads: dict[str, str] | None = None) -> pd.DataFrame:
    expected = _frequency_from_folds(folds)
    source = root / "inter/inter_feature_frequency.csv"
    actual = _read_required(source)
    _equal_table(actual, expected, ["comparison", "feature"], "Alpha feature frequency")
    sources["plv_alpha_only/feature_frequency"] = source
    if source_reads is not None:
        source_reads["plv_alpha_only/feature_frequency"] = actual.attrs["source_sha256"]
    return expected


def _selection_digest(folds: pd.DataFrame) -> str:
    columns = [c for c in SELECTION_COLUMNS if c != "accuracy"]
    return fingerprint(folds[columns].sort_values(["held_out_subject", "comparison", "classifier"]).to_dict("records"))


def rebuild_historical_results(results_root: str | Path, *,
        baseline_features: str | Path = PROJECT_ROOT / "data/features/features_all.csv",
        extension_features: str | Path = PROJECT_ROOT / "data/features/extensions/features_extensions.csv",
        iplv_features: str | Path = PROJECT_ROOT / "data/features/final_validation/features_iplv.csv",
        legacy_manifest: str | Path | None = None, allow_legacy: bool = False,
        experiments: Sequence[str] | None = None, partial: bool = False) -> HistoricalReport:
    """Validate the full declared historical matrix and rebuild every score.

    An explicit partial scope is allowed, with the baseline present for exact
    paired deltas.  It is labelled partial and published only to a new folder.
    No complete report or source archive is ever replaced.
    """
    root = Path(results_root)
    selected = list(HISTORICAL_EXPERIMENTS if experiments is None else experiments)
    if not selected or len(selected) != len(set(selected)) or set(selected) - set(HISTORICAL_EXPERIMENTS):
        raise ResultContractError("Historical experiment scope is empty, duplicate or unknown")
    complete = set(selected) == set(HISTORICAL_EXPERIMENTS)
    if not complete and not partial:
        raise ResultContractError("Incomplete historical matrix requires an explicit partial report scope")
    if "poster_baseline" not in selected:
        raise ResultContractError("Historical reports require the poster_baseline reference for exact pairing")
    matrix_parents = _managed_matrices(root, partial=partial)
    paths = {"baseline": Path(baseline_features), "extension": Path(extension_features), "iplv": Path(iplv_features)}
    needed = {SOURCE_ROOTS[name][1] for name in selected} | {"baseline"}
    sources = {f"features/{family}": paths[family] for family in sorted(needed)}
    if legacy_manifest is not None:
        sources["legacy_mapping"] = Path(legacy_manifest)
    before = input_inventory(sources)
    source_reads = {name: record["sha256"] for name, record in before.items()}
    feature_sets = {family: load_feature_table(paths[family], legacy_manifest=legacy_manifest) for family in sorted(needed)}
    base = feature_sets["baseline"]
    if base.cohort_id.nunique() != 1 or base.protocol_id.nunique() != 1 or set(base.comparison) != set(COMPARISONS):
        raise ResultContractError("Historical reference must identify one cohort/protocol and both declared comparisons")
    for family, features in feature_sets.items():
        try:
            assert_same_epochs(base, features)
        except ValueError as exc:
            raise ResultContractError(f"{family} features differ from the reference cohort or eligible epoch identities") from exc
    eligible = base[["cohort_id", "protocol_id", "comparison", "subject_id", "session_id", "epoch_id", "class_label"]].rename(columns={"class_label": "y_true"}).copy()
    for comparison, frame in eligible.groupby("comparison"):
        manifest_id = "eligible-epochs-" + fingerprint(frame.sort_values("epoch_id").to_dict("records"))
        eligible.loc[eligible.comparison.eq(comparison), "eligible_epoch_manifest_id"] = manifest_id
    analysis_id = "historical-lopo-report-" + fingerprint({"schema": REPORT_SCHEMA,
        "estimand": "equal-participant accuracy and balanced accuracy", "eligible_epochs": eligible.sort_values("epoch_id").to_dict("records")})
    predictions, parents, alpha_folds = [], {}, None
    for name in selected:
        relative, family = SOURCE_ROOTS[name]
        canonical, folds, parent = _adapt_source(root / relative, name, feature_sets[family],
            analysis_id=analysis_id, eligible=eligible, allow_legacy=allow_legacy, sources=sources,
            candidate_features=feature_columns(pd.DataFrame(columns=list(dict.fromkeys([*base.columns, *feature_sets[family].columns]))), FEATURE_SETS[name]),
            source_reads=source_reads)
        predictions.append(canonical)
        parents[name] = parent
        if name == "plv_alpha_only":
            alpha_folds = folds
    joined = pd.concat(predictions, ignore_index=True)
    units = participant_results(joined, eligible_epochs=eligible)
    summary = summarize_participants(units)
    reference = joined.loc[joined.experiment.eq("poster_baseline")]
    deltas = [paired_participant_deltas(reference, joined.loc[joined.experiment.eq(name)]) for name in selected if name != "poster_baseline"]
    paired = pd.concat(deltas, ignore_index=True) if deltas else pd.DataFrame(columns=RESULT_GROUP_COLUMNS + [
        "subject_id", "reference_analysis_id", "reference_run_id", "reference_experiment",
        "eligible_epoch_manifest_id", "eligible_epochs_sha256", "n_epochs", "accuracy_reference",
        "accuracy_experiment", "delta_accuracy", "balanced_accuracy_reference",
        "balanced_accuracy_experiment", "delta_balanced_accuracy"])
    alpha_frequency = (_alpha_frequency(root / "final_validation/plv_alpha_only", alpha_folds, sources, source_reads)
                       if alpha_folds is not None else pd.DataFrame(columns=["comparison", "feature", "selection_count"]))
    inputs = input_inventory(sources)
    if set(inputs) != set(source_reads) or any(inputs[name]["sha256"] != digest for name, digest in source_reads.items()):
        raise ResultContractError("Source bytes changed during report generation")
    selections = alpha_folds[SELECTION_COLUMNS].copy() if alpha_folds is not None else pd.DataFrame(columns=SELECTION_COLUMNS)
    subject_ids = {comparison: sorted(rows.subject_id.unique()) for comparison, rows in eligible.groupby("comparison")}
    metadata = {"schema_version": REPORT_SCHEMA, "analysis_id": analysis_id,
        "scope": "complete_historical_matrix" if complete else "partial_historical_matrix",
        "study_matrix_complete": complete, "required_experiments": list(HISTORICAL_EXPERIMENTS),
        "selected_experiments": selected, "missing_experiments": sorted(set(HISTORICAL_EXPERIMENTS) - set(selected)),
        "comparisons": list(COMPARISONS), "classifiers": list(CLASSIFIERS),
        "cohort_id": base.cohort_id.iloc[0], "protocol_id": base.protocol_id.iloc[0],
        "n_subjects": int(base.subject_id.nunique()), "n_sessions": len(base[["subject_id", "session_id"]].drop_duplicates()),
        "subject_ids_by_comparison": subject_ids,
        "n_subjects_by_comparison": {comparison: len(subjects) for comparison, subjects in subject_ids.items()},
        "alpha_selection_sha256": _selection_digest(selections),
        "alpha_candidate_features": feature_columns(feature_sets["extension"], "plv_band_alpha") if alpha_folds is not None else [],
        "n_eligible_epochs": len(eligible), "inputs": inputs, "source_sha256": fingerprint(inputs),
        "source_locations": {name: str(path.resolve()) for name, path in sources.items()},
        "reconstruction": {"results_root": str(root.resolve()),
            "baseline_features": str(paths["baseline"].resolve()), "extension_features": str(paths["extension"].resolve()),
            "iplv_features": str(paths["iplv"].resolve()), "legacy_manifest": str(Path(legacy_manifest).resolve()) if legacy_manifest is not None else None,
            "allow_legacy": allow_legacy, "experiments": selected, "partial": partial},
        "parent_runs": parents, "parent_matrix_reports": matrix_parents,
        "metrics": list(REPORT_METRICS),
        "aggregation": REPORT_AGGREGATION,
        "uncertainty": REPORT_UNCERTAINTY,
        "balanced_accuracy_eligibility": REPORT_BA_RULE,
        "prediction_identity": "Legacy comparison-local row positions are matched to explicit feature/session/epoch metadata; raw acquisition identity remains unverified.",
        "historical_provenance": "legacy_unverified" if any(p["status"] == "legacy_unverified" for p in parents.values()) else "verified_managed_runs",
        "interpretation": "Historical discovery estimates are exploratory and do not estimate independent external performance."}
    return HistoricalReport(joined, eligible, units, summary, paired, alpha_frequency, metadata, selections)


def validate_historical_report(report: HistoricalReport) -> None:
    """Verify report tables are still exactly derived from their prediction rows."""
    for field, expected in {"schema_version": REPORT_SCHEMA, "aggregation": REPORT_AGGREGATION,
        "uncertainty": REPORT_UNCERTAINTY, "balanced_accuracy_eligibility": REPORT_BA_RULE,
        "metrics": REPORT_METRICS, "required_experiments": list(HISTORICAL_EXPERIMENTS)}.items():
        if report.metadata.get(field) != expected:
            raise ResultContractError(f"Historical report {field} contradicts the participant estimand/reporting contract")
    participants = participant_results(report.predictions, eligible_epochs=report.eligible_epochs)
    _equal_table(report.participants, participants, RESULT_GROUP_COLUMNS + ["subject_id"], "Participant report")
    _equal_table(report.summary, summarize_participants(participants), RESULT_GROUP_COLUMNS, "Summary report")
    expected = pd.DataFrame([{ "experiment": experiment, "classifier": classifier, "comparison": comparison}
        for experiment in report.metadata["selected_experiments"] for classifier in CLASSIFIERS for comparison in COMPARISONS])
    validate_result_matrix(report.summary, expected, keys=["experiment", "classifier", "comparison"])
    selected = report.metadata["selected_experiments"]
    complete = set(selected) == set(HISTORICAL_EXPERIMENTS)
    if (len(selected) != len(set(selected)) or set(selected) - set(HISTORICAL_EXPERIMENTS)
            or report.metadata["study_matrix_complete"] != complete
            or report.metadata["scope"] != ("complete_historical_matrix" if complete else "partial_historical_matrix")
            or report.metadata["missing_experiments"] != sorted(set(HISTORICAL_EXPERIMENTS) - set(selected))):
        raise ResultContractError("Historical report has an inconsistent complete/partial matrix scope")
    if set(report.summary.cohort_id) != {report.metadata["cohort_id"]} or set(report.summary.protocol_id) != {report.metadata["protocol_id"]}:
        raise ResultContractError("Historical report cohort/protocol identity differs from source metadata")
    eligible = report.eligible_epochs
    if (report.metadata["n_subjects"] != eligible.subject_id.nunique()
            or report.metadata["n_sessions"] != len(eligible[["subject_id", "session_id"]].drop_duplicates())
            or report.metadata["n_eligible_epochs"] != len(eligible)
            or report.metadata["comparisons"] != list(COMPARISONS)
            or report.metadata["classifiers"] != list(CLASSIFIERS)):
        raise ResultContractError("Historical report N/session/epoch count or declared matrix metadata is false")
    expected_subjects = {comparison: sorted(rows.subject_id.unique()) for comparison, rows in eligible.groupby("comparison")}
    if report.metadata["subject_ids_by_comparison"] != expected_subjects or report.metadata["n_subjects_by_comparison"] != {comparison: len(subjects) for comparison, subjects in expected_subjects.items()}:
        raise ResultContractError("Historical report participant roster by comparison differs from eligible epochs")
    epoch_columns = [c for c in EPOCH_COLUMNS if c != "eligible_epoch_manifest_id"]
    for _, epochs in eligible.groupby("comparison"):
        expected_manifest_id = "eligible-epochs-" + fingerprint(epochs[epoch_columns].sort_values("epoch_id").to_dict("records"))
        if set(epochs.eligible_epoch_manifest_id) != {expected_manifest_id}:
            raise ResultContractError("Historical eligible epoch manifest ID differs from its exact identities and labels")
    expected_analysis = "historical-lopo-report-" + fingerprint({"schema": REPORT_SCHEMA,
        "estimand": "equal-participant accuracy and balanced accuracy", "eligible_epochs": eligible.sort_values("epoch_id").to_dict("records")})
    if report.metadata["analysis_id"] != expected_analysis or set(report.predictions.analysis_id) != {expected_analysis}:
        raise ResultContractError("Historical analysis identity differs from its metric/epoch specification")
    if set(report.metadata["parent_runs"]) != set(selected):
        raise ResultContractError("Historical parent run inventory differs from selected experiments")
    for name, parent in report.metadata["parent_runs"].items():
        rows = report.predictions.loc[report.predictions.experiment.eq(name)]
        if set(rows.run_id) != {parent["run_id"]}:
            raise ResultContractError("Historical predictions do not belong to their declared parent run")
        if parent["status"] == "legacy_unverified":
            observed = parent["observed_sources"]
            if parent["run_id"] != "legacy-unknown-run-" + fingerprint(observed):
                raise ResultContractError("Legacy parent content ID differs from its observed source hashes")
            for suffix, field in [("predictions", "predictions_sha256"), ("folds", "folds_sha256")]:
                if observed[field] != report.metadata["inputs"][f"{name}/{suffix}"]["sha256"]:
                    raise ResultContractError("Legacy parent source hashes differ from the input inventory")
            if "config_snapshot_sha256" in observed and observed["config_snapshot_sha256"] != report.metadata["inputs"][f"{name}/config_snapshot"]["sha256"]:
                raise ResultContractError("Legacy parent config hash differs from the input inventory")
            if any(parent.get(field) != value for field, value in {
                "historical_code": "unknown", "historical_environment": "unknown", "raw_hashes": "unknown", "serialized_models": "unavailable"}.items()):
                raise ResultContractError("Legacy historical provenance must remain explicitly unknown")
        elif parent["status"] != "complete":
            raise ResultContractError("Unknown historical parent run status")
        elif parent["manifest_sha256"] != report.metadata["inputs"][f"{name}/manifest"]["sha256"]:
            raise ResultContractError("Managed parent manifest hash differs from the input inventory")
        if "outer_fold" not in rows:
            raise ResultContractError("Historical prediction provenance lacks its original outer fold")
        for record in rows[["run_id", "subject_id", "comparison", "classifier", "outer_fold", "model_id"]].drop_duplicates().to_dict("records"):
            identity = {"run_id": record["run_id"], "held_out_subject": record["subject_id"],
                "comparison": record["comparison"], "classifier": record["classifier"], "outer_fold": record["outer_fold"]}
            prefix = "legacy-unknown-model-" if parent["status"] == "legacy_unverified" else "lopo-model-content-"
            if record["model_id"] != prefix + fingerprint(identity):
                raise ResultContractError("Historical model content identity differs from its parent run/fold")
    reference = report.predictions.loc[report.predictions.experiment.eq("poster_baseline")]
    expected_deltas = [paired_participant_deltas(reference, report.predictions.loc[report.predictions.experiment.eq(name)])
        for name in report.metadata["selected_experiments"] if name != "poster_baseline"]
    if expected_deltas:
        _equal_table(report.paired_deltas, pd.concat(expected_deltas, ignore_index=True), RESULT_GROUP_COLUMNS + ["subject_id"], "Paired report")
    elif not report.paired_deltas.empty:
        raise ResultContractError("Historical report contains an undeclared paired comparison")
    if fingerprint(report.metadata["inputs"]) != report.metadata["source_sha256"]:
        raise ResultContractError("Historical report source inventory hash differs")
    if set(report.metadata["source_locations"]) != set(report.metadata["inputs"]):
        raise ResultContractError("Historical source locations do not cover their input inventory")
    for entry in report.metadata["inputs"].values():
        if not re.fullmatch(r"[0-9a-f]{64}", entry.get("sha256", "")) or type(entry.get("bytes")) is not int or entry["bytes"] < 1:
            raise ResultContractError("Historical source inventory is malformed")
    provenance = "legacy_unverified" if any(parent["status"] == "legacy_unverified" for parent in report.metadata["parent_runs"].values()) else "verified_managed_runs"
    if report.metadata["historical_provenance"] != provenance:
        raise ResultContractError("Historical provenance label contradicts its parent run evidence")
    selections = report.alpha_training_selections
    if _selection_digest(selections) != report.metadata["alpha_selection_sha256"]:
        raise ResultContractError("Alpha training selection evidence differs from its recorded content hash")
    if "plv_alpha_only" not in selected:
        if not selections.empty or not report.alpha_frequency.empty or report.metadata["alpha_candidate_features"]:
            raise ResultContractError("Historical report has undeclared alpha selection/frequency evidence")
    else:
        keys = ["held_out_subject", "comparison", "classifier"]
        units = participants.loc[participants.experiment.eq("plv_alpha_only")].rename(columns={"subject_id": "held_out_subject"})
        _equal_table(selections, units[keys + ["accuracy"]], keys, "Alpha selection participant coverage/accuracy")
        predictions = report.predictions.loc[report.predictions.experiment.eq("plv_alpha_only")].rename(columns={"subject_id": "held_out_subject"})
        _equal_table(selections, predictions[keys + ["outer_fold"]].drop_duplicates(), keys, "Alpha selection fold coverage")
        candidates = report.metadata["alpha_candidate_features"]
        expected_candidates = {f"PLV__{first}__{second}__alpha" for first, second in channel_pairs(DEFAULT_CHANNELS)}
        if set(candidates) != expected_candidates or len(candidates) != len(expected_candidates):
            raise ResultContractError("Alpha candidate representation is missing or duplicate")
        for row in selections.to_dict("records"):
            names = row["selected_features"].split(";")
            if set(names) - set(candidates) or len(names) != row["n_selected_features"]:
                raise ResultContractError("Alpha training selection is outside its declared candidate representation")
        _equal_table(report.alpha_frequency, _frequency_from_folds(selections), ["comparison", "feature"], "Alpha feature frequency")


def write_report_contents(report: HistoricalReport, directory: Path) -> None:
    """Write to a caller-owned new staging folder, after checking derivations."""
    directory = protect_output(directory)
    if (directory / "historical_report.json").exists() or any((directory / name).exists() for name in report.tables()):
        raise FileExistsError("Historical report tables already exist; choose a new immutable output directory")
    validate_historical_report(report)
    for name, frame in report.tables().items():
        frame.to_csv(directory / name, index=False, encoding="utf-8-sig")
    metadata = {key: value for key, value in report.metadata.items() if key not in {"artifacts", "report_id"}}
    metadata["artifacts"] = {path.relative_to(directory).as_posix(): {"sha256": sha256_file(path), "bytes": path.stat().st_size}
        for path in sorted(directory.rglob("*")) if path.is_file() and path.name != "historical_report.json"}
    metadata["report_id"] = fingerprint(metadata)
    (directory / "historical_report.json").write_text(json.dumps(metadata, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def write_historical_report(report: HistoricalReport, out_dir: str | Path) -> Path:
    with new_output_directory(out_dir) as stage:
        write_report_contents(report, stage)
        read_historical_report(stage, verify_sources=False)
    return Path(out_dir)


def read_historical_report(path: str | Path, *, verify_sources: bool = True) -> HistoricalReport:
    """Verify immutable artifacts, recompute tables, and by default verify sources.

    ``verify_sources=False`` is an explicit detached content check only.  The
    default requires the original recorded inputs and reconstructs the report
    from them, so rehashing fabricated tables cannot make them source evidence.
    """
    root = Path(path)
    if root.is_symlink() or not root.is_dir():
        raise ResultContractError("Historical report directory is missing or a symlink")
    try:
        metadata = json.loads((root / "historical_report.json").read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ResultContractError("Unreadable historical report manifest") from exc
    if not isinstance(metadata, dict) or metadata.get("schema_version") != REPORT_SCHEMA or metadata.get("report_id") != fingerprint({k: v for k, v in metadata.items() if k != "report_id"}):
        raise ResultContractError("Historical report manifest identity is invalid")
    actual = {}
    for item in sorted(root.rglob("*")):
        if item.is_symlink():
            raise ResultContractError("Historical report artifact may not be a symlink")
        if item.is_file() and item.name != "historical_report.json":
            actual[item.relative_to(root).as_posix()] = {"sha256": sha256_file(item), "bytes": item.stat().st_size}
    if actual != metadata.get("artifacts"):
        raise ResultContractError("Historical report contains missing, changed or unexpected artifacts")
    filenames = ["predictions.csv", "eligible_epochs.csv", "participant_results.csv", "historical_summary.csv",
                 "paired_participant_deltas.csv", "alpha_feature_frequency.csv", "alpha_training_selections.csv"]
    if not set(filenames) <= set(actual):
        raise ResultContractError("Historical report is missing required result tables")
    tables = [read_result_csv(root / filename) for filename in filenames]
    report = HistoricalReport(*tables[:6], metadata, tables[6])
    try:
        validate_historical_report(report)
    except (KeyError, TypeError) as exc:
        raise ResultContractError("Historical report metadata is incomplete or malformed") from exc
    if verify_sources:
        try:
            observed = input_inventory(metadata["source_locations"])
        except (OSError, ValueError) as exc:
            raise ResultContractError("Original historical source inputs are unavailable for source verification") from exc
        if observed != metadata["inputs"]:
            raise ResultContractError("Original historical source bytes changed after this report")
        rebuilt = rebuild_historical_results(**metadata["reconstruction"])
        keys = {"predictions.csv": RESULT_GROUP_COLUMNS + ["epoch_id"], "eligible_epochs.csv": ["epoch_id"],
            "participant_results.csv": RESULT_GROUP_COLUMNS + ["subject_id"], "historical_summary.csv": RESULT_GROUP_COLUMNS,
            "paired_participant_deltas.csv": RESULT_GROUP_COLUMNS + ["subject_id"], "alpha_feature_frequency.csv": ["comparison", "feature"],
            "alpha_training_selections.csv": ["held_out_subject", "comparison", "classifier"]}
        for filename, expected in rebuilt.tables().items():
            _equal_table(report.tables()[filename], expected, keys[filename], f"Historical original-source reconstruction: {filename}")
        if any(metadata.get(key) != value for key, value in rebuilt.metadata.items()):
            raise ResultContractError("Historical reconstructed source/parent/report metadata differs")
    return report


def verify_historical_report(path: str | Path, *, verify_sources: bool = True) -> dict:
    report = read_historical_report(path, verify_sources=verify_sources)
    return {"status": "verified", "report_id": report.metadata["report_id"],
        "scope": report.metadata["scope"], "n_subjects": report.metadata["n_subjects"],
        "historical_provenance": report.metadata["historical_provenance"],
        "source_verification": "reconstructed_from_original_inputs" if verify_sources else "detached_content_only"}
