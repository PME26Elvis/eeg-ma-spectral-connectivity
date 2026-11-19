"""Hand-computable participant estimands and adversarial identity boundaries."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from eeg_ma.result_contract import (EPOCH_COLUMNS, RESULT_GROUP_COLUMNS, ResultContractError,
    paired_participant_deltas, participant_results, read_result_csv,
    summarize_participants, validate_predictions, validate_result_matrix)


def predictions():
    # Person 001: three of four correct, BA=(2/3+1)/2=5/6, two sessions.
    # Person NA: zero of two correct, BA=0. Epoch weighting would give 1/2;
    # equal participant weighting gives 3/8 accuracy and 5/12 BA.
    records = []
    for subject, outcomes in [("001", [("a", 0, 0), ("a", 0, 0), ("b", 0, 1), ("b", 1, 1)]),
                              ("NA", [("a", 0, 1), ("a", 1, 0)])]:
        for number, (session, truth, prediction) in enumerate(outcomes):
            records.append({"analysis_id": "analysis", "run_id": "run", "cohort_id": "cohort", "protocol_id": "protocol",
                "experiment": "baseline", "classifier": "LDA", "comparison": "rest_task", "model_id": f"model-{subject}",
                "subject_id": subject, "session_id": session, "epoch_id": f"epoch-{subject}-{number}",
                "y_true": truth, "y_pred": prediction, "eligible_epoch_manifest_id": "manifest"})
    return pd.DataFrame(records)


def test_sessions_are_pooled_within_person_and_people_have_equal_weight():
    units = participant_results(predictions())
    first = units.set_index("subject_id").loc["001"]
    assert first.n_sessions == 2 and first.n_epochs == 4
    assert [first.tn, first.fp, first.fn, first.tp] == [2, 1, 0, 1]
    assert first.recall_0 == pytest.approx(2/3)
    assert first.recall_1 == 1
    assert first.accuracy == 3/4
    assert first.balanced_accuracy == pytest.approx(5/6)
    summary = summarize_participants(units).iloc[0]
    assert summary.n_subjects == 2  # Different LOPO model IDs are not separate groups.
    assert summary.accuracy_mean == 3/8
    assert summary.balanced_accuracy_mean == pytest.approx(5/12)
    assert summary.pooled_epoch_accuracy == 1/2
    assert summary.pooled_epoch_balanced_accuracy == 1/2
    assert summary.pooled_recall_0 == summary.pooled_recall_1 == 1/2
    assert summary.accuracy_std == pytest.approx(np.sqrt(9/32))
    assert not any("ci" in column for column in summary.index)


@pytest.mark.parametrize("missing", [0, 1])
def test_missing_class_keeps_accuracy_but_never_fills_balanced_accuracy_with_chance(missing):
    frame = predictions()
    frame = frame.loc[~(frame.subject_id.eq("NA") & frame.y_true.eq(missing))]
    units = participant_results(frame)
    participant = units.set_index("subject_id").loc["NA"]
    assert participant.accuracy == 0
    assert np.isnan(participant.balanced_accuracy)
    assert np.isnan(participant[f"recall_{missing}"])
    assert participant.ba_status == f"missing_class_{missing}"
    summary = summarize_participants(units).iloc[0]
    assert summary.n_subjects == 2 and summary.n_ba_eligible == 1 and summary.n_ba_ineligible == 1
    assert summary.balanced_accuracy_mean == pytest.approx(5/6)
    assert np.isnan(summary.balanced_accuracy_std)


def test_literal_identifiers_survive_csv_roundtrip(tmp_path):
    frame = pd.concat([predictions().assign(subject_id=subject, epoch_id=lambda f: subject + f.epoch_id,
                       run_id="001", analysis_id="NA", model_id="nan", eligible_epoch_manifest_id="001")
                       for subject in ("001", "1", "NA", "nan")], ignore_index=True)
    path = tmp_path / "literal.csv"
    frame.to_csv(path, index=False)
    loaded = read_result_csv(path)
    assert set(loaded.subject_id) == {"001", "1", "NA", "nan"}
    assert set(loaded.run_id) == {"001"} and set(loaded.model_id) == {"nan"}
    assert participant_results(loaded).subject_id.nunique() == 4


@pytest.mark.parametrize("column,value", [("subject_id", 1), ("model_id", ""), ("protocol_id", " protocol"), ("y_pred", 2), ("y_true", np.nan), ("y_pred", True)])
def test_missing_or_invalid_identity_and_label_never_gets_coerced(column, value):
    frame = predictions().astype({column: object})
    frame.loc[0, column] = value
    with pytest.raises(ResultContractError):
        participant_results(frame)


def test_duplicate_prediction_epoch_is_rejected_even_if_its_owner_is_renamed():
    frame = predictions()
    extra = frame.iloc[[0]].assign(subject_id="intruder", session_id="other")
    with pytest.raises(ResultContractError, match="duplicate"):
        participant_results(pd.concat([frame, extra]))


def test_expected_matrix_and_manifest_detect_entire_missing_groups_and_epochs():
    one = predictions()
    two = one.assign(classifier="KFDA")
    complete = pd.concat([one, two], ignore_index=True)
    expected = complete[RESULT_GROUP_COLUMNS].drop_duplicates()
    eligible = one[EPOCH_COLUMNS]
    with pytest.raises(ResultContractError, match="matrix"):
        validate_predictions(one, eligible_epochs=eligible, expected_groups=expected)
    with pytest.raises(ResultContractError, match="coverage"):
        validate_predictions(complete.iloc[1:], eligible_epochs=eligible, expected_groups=expected)
    checked = validate_predictions(complete.sample(frac=1, random_state=7), eligible_epochs=eligible, expected_groups=expected)
    assert len(checked) == 12


@pytest.mark.parametrize("mutation", ["participant", "epoch", "cohort", "label", "manifest", "analysis"])
def test_pairing_rejects_equal_size_but_different_eligible_identity(mutation):
    reference = predictions()
    target = reference.assign(run_id="target-run", experiment="target", model_id="target-model").copy()
    if mutation == "participant":
        target.loc[target.subject_id.eq("NA"), "subject_id"] = "other"
    elif mutation == "epoch":
        target.loc[0, "epoch_id"] = "replacement-epoch"
    elif mutation == "cohort":
        target["cohort_id"] = "foreign"
    elif mutation == "label":
        target.loc[0, "y_true"] = 1
    elif mutation == "analysis":
        target["analysis_id"] = "another-estimand"
    else:
        target["eligible_epoch_manifest_id"] = "different-manifest"
    with pytest.raises(ResultContractError, match="differs"):
        paired_participant_deltas(reference, target)


def test_pairing_has_no_inner_join_or_duplicate_first_fallback():
    reference = predictions()
    target = reference.assign(run_id="target", experiment="target")
    with pytest.raises(ResultContractError):
        paired_participant_deltas(reference, target.loc[target.subject_id.ne("NA")])
    with pytest.raises(ResultContractError):
        paired_participant_deltas(reference, pd.concat([target, target.iloc[[0]]]))
    target["y_pred"] = target.y_true
    delta = paired_participant_deltas(reference, target).set_index("subject_id")
    assert delta.loc["001", "delta_accuracy"] == 1/4
    assert delta.loc["001", "delta_balanced_accuracy"] == pytest.approx(1/6)
    assert delta.loc["NA", "delta_accuracy"] == 1


def test_derived_summary_rejects_forged_metrics_and_duplicate_participants():
    units = participant_results(predictions())
    forged = units.copy()
    forged.loc[0, "accuracy"] = 1
    with pytest.raises(ResultContractError, match="confusion"):
        summarize_participants(forged)
    with pytest.raises(ResultContractError, match="duplicate"):
        summarize_participants(pd.concat([units, units.iloc[[0]]]))


def test_result_matrix_rejects_duplicates_before_any_selection():
    cells = predictions()[RESULT_GROUP_COLUMNS].drop_duplicates()
    with pytest.raises(ResultContractError, match="duplicate"):
        validate_result_matrix(pd.concat([cells, cells]), cells)
