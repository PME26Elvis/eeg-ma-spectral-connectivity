"""Two predeclared exploratory EEG classifier tracks, never a new A/B endpoint.

S: shared inner-training LDA-SFS -> all five classifiers.
F: every declared feature -> handwritten RandomForest and LightGBM only.

All evaluation fitting is delegated to the existing immutable grouped-LOPO
runner. This module handles prospective scope, source identity, complete
cross-track coverage, participant-level summaries and atomic report evidence.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import re
from pathlib import Path
import shutil

import numpy as np
import pandas as pd

from .artifacts import new_output_directory
from .classifier_config import classifier_policy
from .evaluation import make_evaluation_splits
from .experiment_runs import execute_evaluation
from .extract import feature_columns
from .provenance import (
    ProvenanceError, RunTransaction, build_run_identity, code_identity,
    environment_identity, fingerprint, input_inventory, protect_output,
    read_csv_literal, sha256_file, validate_complete_run,
)
from .result_contract import (
    EPOCH_COLUMNS, PREDICTION_COLUMNS, RESULT_GROUP_COLUMNS,
    ResultContractError, participant_results, summarize_participants,
    validate_predictions,
)
from .schema import load_feature_table, validate_feature_table


PLAN_VERSION = "eeg-exploratory-five-classifier-plan-1"
REPORT_VERSION = "eeg-exploratory-five-classifier-report-1"
ANALYSIS_ID = "prospective_five_classifier_exploratory_lopo"
S_CLASSIFIERS = ("LDA", "RBF-SVM", "KFDA", "RandomForest", "LightGBM")
F_CLASSIFIERS = ("RandomForest", "LightGBM")
TRACKS = {"S": list(S_CLASSIFIERS), "F": list(F_CLASSIFIERS)}
TRACK_POLICIES = {"S": "shared_training_only_lda_sfs", "F": "all_declared_features"}
DEFAULT_PAIRS = [
    {"reference": "S:LDA", "target": "S:RandomForest"},
    {"reference": "S:LDA", "target": "S:LightGBM"},
    {"reference": "S:RandomForest", "target": "F:RandomForest"},
    {"reference": "S:LightGBM", "target": "F:LightGBM"},
]
TABLE_NAMES = ("eligible_epochs.csv", "predictions.csv", "participant_results.csv",
               "summary.csv", "paired_participants.csv", "paired_summary.csv")
BARE_ELIGIBILITY = [c for c in EPOCH_COLUMNS if c != "eligible_epoch_manifest_id"]


class ExploratoryError(ValueError):
    """This exploratory output is incomplete, ambiguous or out of scope."""


def _read_json(path: Path):
    def reject_duplicates(pairs):
        value = {}
        for key, item in pairs:
            if key in value:
                raise ExploratoryError(f"Duplicate JSON key {key!r}: {path}")
            value[key] = item
        return value

    try:
        return json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=reject_duplicates)
    except (OSError, ValueError) as exc:
        raise ExploratoryError(f"Invalid JSON input: {path}") from exc


def _write_json(path: Path, value):
    path.write_text(json.dumps(value, sort_keys=True, ensure_ascii=False,
                               allow_nan=False, indent=2) + "\n", encoding="utf-8")


def _write_csv(path: Path, frame: pd.DataFrame):
    frame.to_csv(path, index=False, encoding="utf-8", float_format="%.12g",
                 lineterminator="\n")


def _is_literal(value):
    return isinstance(value, str) and value and value == value.strip()


def _read_eligibility(path: Path) -> pd.DataFrame:
    """Independently prepared expected cohort/epoch roster; never infer completion from predictions."""
    try:
        frame = read_csv_literal(path)
    except (OSError, ValueError) as exc:
        raise ExploratoryError("Cannot read independent eligibility manifest") from exc
    if list(frame.columns) != BARE_ELIGIBILITY or frame.empty:
        raise ExploratoryError("Eligibility manifest must contain the exact ordered cohort/session/epoch/label columns")
    for col in [c for c in BARE_ELIGIBILITY if c != "y_true"]:
        if any(not _is_literal(v) for v in frame[col]):
            raise ExploratoryError("Eligibility manifest has invalid literal IDs")
    if (frame["y_true"].isna().any() or not frame["y_true"].isin([0, 1]).all()
            or frame["epoch_id"].duplicated().any()):
        raise ExploratoryError("Eligibility manifest contains missing/duplicate epochs or invalid binary truth")
    return frame


def _verify_plan(plan: dict, cfg: dict, features: pd.DataFrame,
                 eligibility: pd.DataFrame, *, allow_synthetic: bool):
    fields = {"schema_version", "analysis_id", "status", "cohort_id", "protocol_id",
              "comparisons", "feature_set", "tracks", "paired_comparisons",
              "registration_evidence", "predeclared_scope", "eligibility_sha256"}
    if not isinstance(plan, dict) or set(plan) != fields:
        raise ExploratoryError("Exploratory plan has missing or unknown fields")
    if plan["schema_version"] != PLAN_VERSION or plan["analysis_id"] != ANALYSIS_ID:
        raise ExploratoryError("Exploratory plan version/analysis ID is not recognized")
    if (not _is_literal(plan["cohort_id"]) or not _is_literal(plan["protocol_id"])
            or plan["cohort_id"].lower().startswith(("discovery", "legacy", "replace"))
            or plan["protocol_id"].lower().startswith("replace")):
        raise ExploratoryError("Declare real, previously unseen cohort and acquisition protocol IDs")
    if plan["predeclared_scope"] != "descriptive_exploration_not_confirmatory":
        raise ExploratoryError("Exploratory scope must not be reported as confirmatory")
    if plan["tracks"] != TRACKS:
        raise ExploratoryError("Both predefined S/F tracks and their precise classifier rosters are required")
    pairs = plan["paired_comparisons"]
    allowed = {f"{track}:{name}" for track, roster in TRACKS.items() for name in roster}
    if (not isinstance(pairs, list) or not pairs
            or any(not isinstance(p, dict) or set(p) != {"reference", "target"}
                   or p["reference"] not in allowed or p["target"] not in allowed
                   or p["reference"] == p["target"] for p in pairs)
            or len({(p["reference"], p["target"]) for p in pairs}) != len(pairs)):
        raise ExploratoryError("Declare nonduplicate eligible paired comparisons before evaluation")
    comparisons = [item["name"] for item in cfg["comparisons"]]
    if (not isinstance(plan["comparisons"], list) or not plan["comparisons"]
            or len(set(map(str, plan["comparisons"]))) != len(plan["comparisons"])
            or plan["comparisons"] != comparisons
            or plan["comparisons"] != sorted(features["comparison"].unique().tolist())):
        raise ExploratoryError("The complete predeclared comparison matrix must match the input data/config")
    if plan["feature_set"] != cfg.get("feature_set", "all"):
        raise ExploratoryError("Declared feature set differs from the analysis configuration")
    if (not isinstance(plan["eligibility_sha256"], str)
            or re.fullmatch(r"[0-9a-f]{64}", plan["eligibility_sha256"]) is None):
        raise ExploratoryError("A frozen SHA-256 for the independent eligibility manifest is required")
    expected = features[
        ["cohort_id", "protocol_id", "comparison", "subject_id", "session_id",
         "epoch_id", "class_label"]
    ].rename(columns={"class_label": "y_true"})
    observed = eligibility[BARE_ELIGIBILITY]
    order = BARE_ELIGIBILITY
    if (expected.sort_values(order).reset_index(drop=True).to_dict("records")
            != observed.sort_values(order).reset_index(drop=True).to_dict("records")):
        raise ExploratoryError("Feature input does not match the independent frozen eligibility roster")
    if cfg.get("classifiers") != list(S_CLASSIFIERS) or "feature_selection_policy" in cfg:
        raise ExploratoryError("Base analysis config must explicitly list the five standard S-track classifiers")
    if not feature_columns(features, plan["feature_set"]):
        raise ExploratoryError("No candidates for the frozen feature family")
    if set(features["cohort_id"]) != {plan["cohort_id"]} or set(features["protocol_id"]) != {plan["protocol_id"]}:
        raise ExploratoryError("A foreign cohort or acquisition protocol was mixed into this analysis")
    if set(features["identity_origin"]) != {"raw_metadata"}:
        raise ExploratoryError("Legacy/adapted discovery features are forbidden in prospective exploration")
    if not set(features["qc_status"]).issubset({"passed", "passed_with_warnings"}):
        raise ExploratoryError("Exploratory input contains unverified epochs")
    if features["subject_id"].nunique() < 3:
        raise ExploratoryError("Grouped outer/inner participant CV needs at least three distinct subjects")
    for (comparison, subject), group in features.groupby(["comparison", "subject_id"]):
        if set(group["class_label"]) != {0, 1}:
            raise ExploratoryError(f"Both class labels are required for {subject}/{comparison}")
    status = plan["status"]
    evidence = plan["registration_evidence"]
    if status == "synthetic_only":
        if not allow_synthetic or evidence is not None:
            raise ExploratoryError("Synthetic-only run requires explicit --allow-synthetic and no registration claim")
    elif status == "prospective_exploratory_frozen":
        if (not isinstance(evidence, dict)
                or set(evidence) != {"frozen_at", "reference", "operator"}
                or any(not _is_literal(evidence[k]) for k in evidence)):
            raise ExploratoryError("A prospective run needs explicit operator-entered freeze evidence")
    else:
        raise ExploratoryError("Draft/unrecognized plan cannot run; freeze an exploratory plan first")
    # Validate both actual estimator policies before creating any output.
    for track in TRACKS:
        effective = _config_for_track(cfg, track)
        classifier_policy(effective)
        make_evaluation_splits(features, effective, "inter")


def _config_for_track(config: dict, track: str) -> dict:
    effective = deepcopy(config)
    if track == "S":
        effective["classifiers"] = list(S_CLASSIFIERS)
    elif track == "F":
        effective["classifiers"] = list(F_CLASSIFIERS)
        effective["feature_selection_policy"] = "all_declared_features"
        effective["sfs"] = {**effective["sfs"], "max_features": None}
    else:
        raise ExploratoryError(f"Unknown track {track}")
    return effective


def _eligible(independent: pd.DataFrame, plan: dict) -> pd.DataFrame:
    parts = []
    for comparison in plan["comparisons"]:
        bare = independent.loc[independent["comparison"] == comparison].copy()
        identifier = fingerprint(bare.sort_values("epoch_id").to_dict("records"))
        bare["eligible_epoch_manifest_id"] = identifier
        parts.append(bare)
    return pd.concat(parts, ignore_index=True)[EPOCH_COLUMNS].sort_values(
        ["comparison", "subject_id", "session_id", "epoch_id"]).reset_index(drop=True)


def _run_track(root: Path, track: str, features: pd.DataFrame, config: dict, plan: dict,
               *, code: dict, environment: dict):
    effective = _config_for_track(config, track)
    inputs = {"features": root / "source_features.csv",
              "analysis_config": root / "analysis_config.json", "plan": root / "plan.json",
              "eligibility": root / "eligibility_source.csv"}
    identity = build_run_identity(
        features, effective, ["inter"], input_paths=inputs,
        experiment={"name": f"exploratory_track_{track}", "feature_set":plan["feature_set"]},
        context_config={"analysis_id": ANALYSIS_ID, "track": track,
                        "plan_sha256": sha256_file(inputs["plan"]),
                        "scope": "exploratory_not_A_or_B"},
        code=code, environment=environment, scope_layout="flat",
    )

    def writer(stage):
        if input_inventory(inputs) != identity["inputs"]:
            raise ExploratoryError("Input bytes changed before grouped evaluation")
        execute_evaluation(stage, features, identity)
        if input_inventory(inputs) != identity["inputs"]:
            raise ExploratoryError("Input bytes changed while fitting")

    return RunTransaction(root / track, identity, resume=False).execute(writer)


def _materialize(root: Path):
    plan = _read_json(root / "plan.json")
    config = _read_json(root / "analysis_config.json")
    features = load_feature_table(root / "source_features.csv")
    validate_feature_table(features)
    independent = _read_eligibility(root / "eligibility_source.csv")
    if sha256_file(root / "eligibility_source.csv") != plan["eligibility_sha256"]:
        raise ExploratoryError("Independent eligibility manifest bytes differ from the frozen plan")
    _verify_plan(plan, config, features, independent, allow_synthetic=True)
    eligible = _eligible(independent, plan)
    lookup = features.set_index("epoch_id", drop=False)
    frames, expected_groups = [], []
    run_ids = {}
    for track, roster in TRACKS.items():
        manifest = validate_complete_run(root / track)
        identity = manifest["identity"]
        if (identity["config"] != _config_for_track(config, track)
                or identity["experiment"] != {"name": f"exploratory_track_{track}",
                                               "feature_set": plan["feature_set"]}
                or identity["context_config"] != {
                    "analysis_id": ANALYSIS_ID, "track": track,
                    "plan_sha256": sha256_file(root / "plan.json"),
                    "scope": "exploratory_not_A_or_B",
                }):
            raise ExploratoryError("Track run identity differs from the archived exploratory plan")
        for name, path in (("features", "source_features.csv"),
                           ("analysis_config", "analysis_config.json"), ("plan", "plan.json"),
                           ("eligibility", "eligibility_source.csv")):
            if identity["inputs"][name]["sha256"] != sha256_file(root / path):
                raise ExploratoryError(f"Track {track} uses different {name} bytes")
        if identity["dataset"]["n_rows"] != len(features):
            raise ExploratoryError("Track dataset contains incomplete epoch coverage")
        run_id = str(manifest["run_id"])
        run_ids[track] = run_id
        pred = read_csv_literal(root / track / "inter_predictions.csv")
        if set(pred["classifier"]) != set(roster):
            raise ExploratoryError(f"Track {track} has missing or foreign classifiers")
        for comparison in plan["comparisons"]:
            for classifier in roster:
                expected_groups.append({
                    "analysis_id": ANALYSIS_ID, "run_id": run_id,
                    "cohort_id": plan["cohort_id"], "protocol_id": plan["protocol_id"],
                    "experiment": f"exploratory_track_{track}",
                    "classifier": classifier, "comparison": comparison,
                })
        if not set(["subject_id", "session_id", "epoch_id", "comparison",
                    "true_label", "predicted_label", "outer_fold",
                    "held_out_subject", "classifier"]).issubset(pred):
            raise ExploratoryError("A track is missing prediction or epoch identity fields")
        for record in pred.to_dict("records"):
            epoch = str(record["epoch_id"])
            if epoch not in lookup.index:
                raise ExploratoryError("Prediction has an epoch not in the input cohort")
            source = lookup.loc[epoch]
            if (str(record["subject_id"]) != str(source["subject_id"])
                    or str(record["held_out_subject"]) != str(source["subject_id"])
                    or str(record["session_id"]) != str(source["session_id"])
                    or str(record["comparison"]) != str(source["comparison"])
                    or int(record["true_label"]) != int(source["class_label"])):
                raise ExploratoryError("A held-out prediction changed source epoch identity or truth")
        enriched = pred.copy()
        enriched["analysis_id"] = ANALYSIS_ID
        enriched["run_id"] = run_id
        enriched["cohort_id"] = plan["cohort_id"]
        enriched["protocol_id"] = plan["protocol_id"]
        enriched["experiment"] = f"exploratory_track_{track}"
        enriched["y_true"] = enriched["true_label"].astype(int)
        enriched["y_pred"] = enriched["predicted_label"].astype(int)
        enriched["model_id"] = enriched.apply(
            lambda row: "unarchived-learner-fold-" + fingerprint({
                "run_id": run_id, "classifier": row["classifier"],
                "comparison": row["comparison"], "fold": int(row["outer_fold"]),
            }), axis=1)
        eligible_ids = eligible.groupby("comparison")["eligible_epoch_manifest_id"].first().to_dict()
        enriched["eligible_epoch_manifest_id"] = enriched["comparison"].map(eligible_ids)
        frames.append(enriched[PREDICTION_COLUMNS].copy())
    predictions = pd.concat(frames, ignore_index=True)
    expected = pd.DataFrame(expected_groups, columns=RESULT_GROUP_COLUMNS)
    predictions = validate_predictions(predictions, eligible_epochs=eligible, expected_groups=expected)
    participants = participant_results(predictions, eligible_epochs=eligible, expected_groups=expected)
    summary = summarize_participants(participants)
    participants = participants.sort_values(
        ["experiment", "comparison", "classifier", "subject_id"]).reset_index(drop=True)
    summary = summary.sort_values(["experiment", "comparison", "classifier"]).reset_index(drop=True)

    pairs, aggregations = [], []
    by_key = {(str(row["experiment"]), str(row["classifier"]), str(row["comparison"]),
               str(row["subject_id"])): row for row in participants.to_dict("records")}
    for comparison in plan["comparisons"]:
        subjects = sorted(features.loc[features.comparison.eq(comparison), "subject_id"].unique().tolist())
        for item in plan["paired_comparisons"]:
            ref_track, ref_classifier = item["reference"].split(":", 1)
            tgt_track, tgt_classifier = item["target"].split(":", 1)
            delta_values = []
            for subject in subjects:
                reference = by_key[(f"exploratory_track_{ref_track}", ref_classifier, comparison, subject)]
                target = by_key[(f"exploratory_track_{tgt_track}", tgt_classifier, comparison, subject)]
                if (reference["eligible_epochs_sha256"] != target["eligible_epochs_sha256"]
                        or reference["n_epochs"] != target["n_epochs"]
                        or reference["model_ids_json"] == ""):
                    raise ExploratoryError("Paired subjects have nonmatching eligible epoch coverage")
                a, b = reference["balanced_accuracy"], target["balanced_accuracy"]
                delta = float(b) - float(a) if pd.notna(a) and pd.notna(b) else np.nan
                if pd.notna(delta):
                    delta_values.append(delta)
                pairs.append({
                    "comparison": comparison, "reference": item["reference"], "target": item["target"],
                    "subject_id": subject, "reference_ba": a, "target_ba": b, "delta_ba": delta,
                    "eligible_epoch_manifest_id": reference["eligible_epoch_manifest_id"],
                })
            aggregations.append({
                "comparison": comparison, "reference": item["reference"], "target": item["target"],
                "n_participants": len(subjects), "n_evaluable_pairs": len(delta_values),
                "mean_delta_ba": float(np.mean(delta_values)) if delta_values else np.nan,
                "sd_delta_ba": float(np.std(delta_values, ddof=1)) if len(delta_values) > 1 else np.nan,
                "interpretation": "descriptive_exploratory_not_confirmatory",
            })
    pairs = pd.DataFrame(pairs).sort_values(
        ["comparison", "reference", "target", "subject_id"]).reset_index(drop=True)
    paired_summary = pd.DataFrame(aggregations).sort_values(
        ["comparison", "reference", "target"]).reset_index(drop=True)
    return {
        "eligible_epochs.csv": eligible,
        "predictions.csv": predictions,
        "participant_results.csv": participants,
        "summary.csv": summary,
        "paired_participants.csv": pairs,
        "paired_summary.csv": paired_summary,
    }, run_ids


def _render_report(root: Path, tables: dict, run_ids: dict) -> str:
    plan = _read_json(root / "plan.json")
    feature = load_feature_table(root / "source_features.csv")
    lines = [
        "# Prospective five-classifier comparison — exploratory only",
        "",
        f"Scope: **{plan['status']}**. Analysis ID: \`{ANALYSIS_ID}\`.",
        "This report is **not** Protocol A confirmation, Protocol B control analysis,",
        "a discovery-cohort rerun, or evidence that any classifier generalizes.",
        "No model is declared the winner; no new hypothesis is chosen from test accuracy.",
        "",
        f"Cohort: \`{plan['cohort_id']}\`; acquisition protocol: \`{plan['protocol_id']}\`.",
        f"Participants: {feature.subject_id.nunique()}; sessions: "
        f"{len(feature[['subject_id', 'session_id']].drop_duplicates())}; epochs: {len(feature)}.",
        "Participants, not epochs, are the reporting unit; all sessions from a person",
        "stay together in every outer leave-one-participant-out split.",
        "",
        "## Predeclared tracks",
        "",
        f"- S / \`{run_ids['S']}\`: LDA-SFS inside outer training, followed by five classifiers.",
        f"- F / \`{run_ids['F']}\`: all declared features without SFS; RF and LightGBM only.",
        "  F's inherited CSV *sfs_path* is a typed inclusion trace with null SFS scores.",
        "Both tracks use the same independent eligible epoch manifest. Inner grid",
        "accuracies are selection diagnostics, not an independent performance estimate.",
        "",
        "## Participant-macro summaries",
        "",
        "| Comparison | Track | Classifier | N | BA mean | Accuracy mean | Pooled accuracy |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    for r in tables["summary.csv"].to_dict("records"):
        fmt = lambda x: "NA" if pd.isna(x) else f"{float(x):.4f}"
        track = str(r["experiment"]).removeprefix("exploratory_track_")
        lines.append(f"| {r['comparison']} | {track} | {r['classifier']} | {r['n_subjects']} | "
                     f"{fmt(r['balanced_accuracy_mean'])} | {fmt(r['accuracy_mean'])} | "
                     f"{fmt(r['pooled_epoch_accuracy'])} |")
    lines += ["", "## Predeclared paired differences", "",
              "Each delta = target BA − reference BA for the **same** participants and epochs.",
              "A negative value, zero, and a null when a class is absent are all reported.",
              "No retrospective winner or inferential CI is implied.", "",
              "| Comparison | Reference → target | N evaluable | Mean Δ BA |",
              "|---|---|---:|---:|"]
    for row in tables["paired_summary.csv"].to_dict("records"):
        value = "NA" if pd.isna(row["mean_delta_ba"]) else f"{float(row['mean_delta_ba']):+.4f}"
        lines.append(f"| {row['comparison']} | {row['reference']} → {row['target']} | "
                     f"{row['n_evaluable_pairs']} | {value} |")
    lines += ["", "## Verification and limitations", "",
              "- The run archive contains the exact source features, analysis config, and plan.",
              "- Both track subruns are separately immutable and validated against their",
              "  stored predictions, outer/inner split manifests and tree-model audits.",
              "- The complete expected track/classifier/comparison/epoch matrix is enforced;",
              "  no inner joins can silently discard a participant or an epoch.",
              "- Participant-macro BA/accuracy and pooled-epoch metrics are separate.",
              "- The reported learner-fold identifiers identify prediction groups; they",
              "  are **not** serialized fitted LDA/SVM/KFDA model artifacts.",
              "- Five discovery subjects and their historical result files are not modified.",
              "- Synthetic-only evidence is not human EEG research evidence.",
              "- Even on real prospective data this is descriptive exploration, not",
              "  prospective Protocol A's frozen primary/secondary endpoint.",
              ""]
    return "\n".join(lines)


def _inventory(root: Path):
    items = {}
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ExploratoryError("Symlink inside comparison archive is forbidden")
        if path.is_file() and path.relative_to(root).as_posix() != "report_manifest.json":
            relative = path.relative_to(root).as_posix()
            items[relative] = {"sha256": sha256_file(path), "bytes": path.stat().st_size}
    return items


def validate_exploratory_report(path: str | Path) -> dict:
    root = Path(path)
    manifest = _read_json(root / "report_manifest.json")
    if (not isinstance(manifest, dict)
            or set(manifest) != {"schema_version", "status", "analysis_id", "study_status",
                                 "report_id", "run_ids", "files"}
            or manifest["schema_version"] != REPORT_VERSION
            or manifest["status"] != "complete"
            or manifest["analysis_id"] != ANALYSIS_ID):
        raise ExploratoryError("Exploratory report manifest is incomplete or has an invalid scope")
    current = _inventory(root)
    if current != manifest["files"]:
        raise ExploratoryError("Archive content was removed, added, or modified")
    derived, run_ids = _materialize(root)
    if run_ids != manifest["run_ids"] or _read_json(root / "plan.json")["status"] != manifest["study_status"]:
        raise ExploratoryError("Report manifest identity differs from source-bound track runs")
    for name, frame in derived.items():
        actual = (root / name).read_bytes()
        expected = frame.to_csv(index=False, encoding="utf-8", float_format="%.12g",
                                lineterminator="\n").encode("utf-8")
        if actual != expected:
            raise ExploratoryError(f"Derived report values do not agree with complete predictions: {name}")
    if (root / "report.md").read_text(encoding="utf-8") != _render_report(root, derived, run_ids):
        raise ExploratoryError("Narrative is inconsistent with participant-level results")
    expected_id = fingerprint({
        "version": REPORT_VERSION, "analysis_id": ANALYSIS_ID,
        "study_status": manifest["study_status"], "run_ids": run_ids,
        "files": current,
    })
    if manifest["report_id"] != expected_id:
        raise ExploratoryError("Report identity does not match archived evidence")
    return {"status": "verified", "report_id": expected_id, "run_ids": run_ids,
            "n_groups": len(derived["summary.csv"]),
            "n_participants": derived["participant_results.csv"]["subject_id"].nunique(),
            "n_predictions": len(derived["predictions.csv"]),
            "scope": manifest["study_status"]}


def build_exploratory_report(feature_path: str | Path, config_path: str | Path,
                             plan_path: str | Path, eligibility_path: str | Path,
                             out_dir: str | Path, *, allow_synthetic=False) -> dict:
    paths = [Path(value) for value in (feature_path, config_path, plan_path, eligibility_path)]
    source_hashes = [sha256_file(p) for p in paths]
    config, plan = _read_json(paths[1]), _read_json(paths[2])
    features = load_feature_table(paths[0])
    validate_feature_table(features)
    independent = _read_eligibility(paths[3])
    if sha256_file(paths[3]) != plan["eligibility_sha256"]:
        raise ExploratoryError("Independent eligibility manifest digest does not match the frozen plan")
    _verify_plan(plan, config, features, independent, allow_synthetic=allow_synthetic)
    target = protect_output(out_dir)
    code = code_identity()
    environment = environment_identity()
    with new_output_directory(target) as stage:
        for source, name in zip(paths, ("source_features.csv", "analysis_config.json",
                                       "plan.json", "eligibility_source.csv")):
            shutil.copyfile(source, stage / name)
        if [sha256_file(p) for p in paths] != source_hashes:
            raise ExploratoryError("Source bytes changed while being archived")
        for track in TRACKS:
            _run_track(stage, track, features, config, plan, code=code, environment=environment)
        derived, run_ids = _materialize(stage)
        for name, frame in derived.items():
            _write_csv(stage / name, frame)
        (stage / "report.md").write_text(_render_report(stage, derived, run_ids), encoding="utf-8")
        files = _inventory(stage)
        report_id = fingerprint({
            "version": REPORT_VERSION, "analysis_id": ANALYSIS_ID,
            "study_status": plan["status"], "run_ids": run_ids, "files": files,
        })
        _write_json(stage / "report_manifest.json", {
            "schema_version": REPORT_VERSION, "status": "complete", "analysis_id": ANALYSIS_ID,
            "study_status": plan["status"], "report_id": report_id,
            "run_ids": run_ids, "files": files,
        })
        validate_exploratory_report(stage)
    if [sha256_file(p) for p in paths] != source_hashes:
        raise ExploratoryError("Source changed after publication; retain the archived source")
    return validate_exploratory_report(target)
