# 獨立受試者驗證協定草案

**狀態：DRAFT；2026-11-09。** 本文件是可實作的下一階段設計，尚未完成註冊、模型凍結或新受試者收案。文件合併不等於這些工作已完成。

研究方向見 [roadmap](roadmap.md)；本次程式與資料稽核見 [2026-11-09 audit](roadmap_audit_2026-11-09.md)。
歷史方法以 [`method_definition.md`](method_definition.md) 為準；已完成的探索結果見 [`final_results_summary.md`](final_results_summary.md)。

## 1. 目的與既有停止規則

主要問題：**使用歷史五位受試者訓練並固定的 α-PLV 模型，能否辨識獨立新受試者在指定實驗流程中的 Rest 與 Type1 狀態？**

本研究先確認分類結果能否外推到新受試者；生理來源需要另外控制的實驗。
歷史資料的 74.1% 是三個分類器結果的描述平均，並非單一可部署模型，也不是獨立確認的目標值；nested CV 不消除跨實驗選擇造成的偏差。[1]
所有舊五人的頻帶與特徵探索已完成；遵守 [`final_validation.md`](final_validation.md) 的 stop rule。
本協定不授權在舊五人上重新排列特徵、挑選新分類器或進行大量搜尋。
若發現方法錯誤，保存原結果、修正原因與新版本，不把修正後的探索結果改稱獨立驗證。

## 2. Acquisition 決策：先完成 A，再解答 B

| 協定 | 角色與設計 | 可回答的問題 |
|---|---|---|
| **A：歷史條件對齊的新 cohort** | Primary；沿用 Rest1 → Type1 → Rest2 → Type2、5 秒心算、相同通道與可核對的量測條件 | 固定模型是否能辨識新人的相同實驗流程 |
| **B：受控的任務流程** | 預先定義的 secondary session 或下一個 study；重複 blocks、平衡順序、匹配視覺與作答條件 | 分類訊息是否能在減少流程混淆後保留 |

A 沿用 150 秒 Rest 分為 30 個 epochs、每種任務累積 30 題答對的歷史取樣規則。
收案前須另訂可接受的最長 session／最多嘗試數及無法完成時的處理；不能無限延長直到符合資料數。
A 仍存在 fixed-order、time-on-task、視覺與 block 差異；即使結果良好，也只能支持 **protocol discrimination**。[2]
不能由 A 單獨確認純心算負荷、神經功能連結，或宣稱已排除疲勞與眼動。

B 必須在資料檢視前決定 block 數、順序分配、washout、視覺控制與分析方式。
只交換 Type1／Type2 順序不能完整解決 Rest／MA 的時間位置和視覺差異。
B 若由同一批人完成，預設先完成 A，再於固定間隔後做 B，避免 B 的先前練習改變 A 的主要驗證條件；同人兩個 session 仍只算一位受試者。
B 內部的 condition/block 順序另做平衡；若研究問題要求 A／B session 也 counterbalance，需另訂 study 並承認 acquisition-history 差異。
B 使用不同 `protocol_id`，單獨報告；不把 A／B 合併成較高分數，不因 A 失敗改稱 B 是 primary。
若量測條件或任務有實質改變，明確標為 protocol transport，不能稱完全相同條件的 replication。

## 3. Development、test 與估計對象

| 分析 | 訓練／測試方式 | Estimand |
|---|---|---|
| **固定模型外部測試** | 舊五人 development；一次 fit 後，所有新受試者只 predict | 這個固定模型對新 cohort 的表現；primary |
| **擴充樣本 nested LOPO** | 新舊資料合併，每折重新在 training subjects 內選特徵與 fit | 固定學習流程以 N−1 人訓練時的跨人表現；secondary |
| **校正／alignment** | 使用新受試者事先指定的 calibration 資料 | 具有該資料存取條件的表現；獨立實驗 |

LOPO 會改變 training size 及每折模型，不能替代固定模型外部測試，也不能用來補救其失敗。
若新增 development participants 再 fit，必須另留完全未見的 confirmatory cohort，並重新記錄 freeze。
同一人的所有 sessions 共享 `subject_id`；任何 train／test 邊界都以人為單位排除交集。
禁止使用新 test cohort 的標籤、分布或最佳結果調整 primary 模型。

## 4. Primary 模型的完整定義

| 環節 | 固定決策 |
|---|---|
| 比較 | A 的 Rest1 vs Type1；Type2 為 secondary |
| 通道 | FP1、FP2、F7、F3、Fz、F4、F8；Cz 不進 primary 特徵 |
| Epoch | 500 Hz、5 秒；保留歷史事件對齊定義 |
| 前處理 | 四階 Butterworth 2–50 Hz、`sosfiltfilt`；預設無 notch |
| 相位特徵 | 連續訊號再經四階 Butterworth 8–13 Hz、`sosfiltfilt`、Hilbert phase；21 組 PLV pairs |
| SFS | 只在 development subjects 內使用 LOGO；LDA inner-CV accuracy 準則 |
| SFS 停止 | `max_features=null`、`min_improvement=0`；第一個特徵接受，其後無嚴格改善即停止；維持 `1e-15` 容差與索引 tie-breaking |
| 標準化 | SFS 各 inner fold 只 fit 該 training split；最終 selected subset 只在完整 development fit scaler |
| 分類器 | LDA：`solver=svd`、`shrinkage=None`、`priors=None`、`tol=1e-4`；完整參數與套件版本寫入 artifact |
| Prediction | 使用固定 feature order、scaler、selected indices、class mapping 與 LDA 決策規則 |

LDA 被指定為 primary 是為減少調參自由度；不宣稱它比歷史 0.747 的 SVM 或 0.733 的 KFDA 更好。
SFS 在 development 全體內重新完成一次，再 fit 最終模型；不以外層 LOPO 的選取次數人工拼出「最佳 pairs」。
Primary 維持現有 SFS accuracy 準則；reporting 的 balanced accuracy 不代表偷偷更換 selector objective。
SVM／KFDA 若列 secondary，必須在 test 資料可用前鎖定其範圍，不能由 test 最佳結果晉升 primary。

此流程是 **correct-only、offline、zero-calibration** 的分類研究。
心算主要分析限 `analysis_eligible=true` 且 `response_correct=true`；這不代表所有嘗試或即時使用的準確率。
固定的 continuous filtering／Hilbert 不會學習 test labels，但使用整段記錄，不能宣稱 causal online inference。
禁止在新受試者資料上重新估 scaler、SFS、classifier 或 EA；全段 target-fit ICA 也需另列資料使用條件。

## 5. Primary 與 secondary endpoints

對可評估受試者 i，先計算其 Rest 與 MA 的 recall，再定義：

```text
BA_i = (Recall_Rest_i + Recall_MA_i) / 2
Primary = mean_i(BA_i)
Delta_i = BA_i(alpha-PLV + SFS + LDA) - BA_i(BP+COH + SFS + LDA)
Key secondary = mean_i(Delta_i)
```

主要報告 participant-macro BA，每位受試者權重相同；pooled epoch accuracy 只作附表。
歷史每人每類均 30 epochs，accuracy 與 BA 一致；未來固定 QC 後不平衡時，BA 更清楚呈現兩類辨識。
Reference model 使用歷史 42 BP＋126 COH、同一 development cohort、既有 training-only SFS 與 LDA。
兩個模型在同一套事先固定的 eligible test epochs 上比較，禁止各選有利的樣本。
Reference 同樣需保存固定 fitted artifact；不能拿舊 LOPO 平均 0.523 代替新 cohort baseline prediction。

次要表格包含每人每類 recall、confusion matrix、eligible epochs、排除數與 paired Δ。
Type2、其他模型及 B 分別標為 secondary；其 nominal CI 不當成多重比較校正後的確認性發現。
目前只設一個 primary endpoint。若要對 secondary 做正式檢定，須在收案前固定 family 與 multiplicity 方法。

## 6. 執行介面：fit、predict、score 分開

以下是待實作的介面契約，不代表 repo 目前已有這些命令：

```text
fit(development, frozen_protocol) -> model_artifact + training_manifest
predict(model_artifact, test_features) -> immutable_predictions + prediction_manifest
score(immutable_predictions, locked_labels, locked_eligibility) -> external_report
```

`fit` 必須驗證 development subject IDs、feature schema、class mapping、資料與設定 hash。
`predict` 的模型輸入只接受固定的 feature columns，不接收 class、stage、檔名或 correctness 作為 predictor。
Epoch 定位與 correctness eligibility 可由獨立的 protocol metadata 流程建立；不能依模型分數修改。
先對 protocol-valid epochs 生成預測，再以預先固定規則套用 correctness／QC mask，完整保留分母。
若看得到實驗事件而無法做到人員完全盲化，誠實記錄限制；仍須維持模型輸入與評分的隔離。

預測輸出至少包含 `subject_id`、`session_id`、`protocol_id`、`epoch_id`、`model_id`、prediction 與 artifact hash。
先寫出 predictions 及 SHA-256 manifest，再啟用含標籤的 `score`；相同 run ID 禁止覆寫。
`score` 須驗證模型、protocol、epoch keys、predictions、labels、eligibility hashes 與預定 cohort 邊界。
同一份輸入的重算可驗證一致性；任何變更須新 run ID、記錄原因，不能覆蓋原來主要分析。
主要報告產生後鎖定該版本；後續修正與 secondary analyses 以 amendment 追蹤。

## 7. QC、missing data 與 repeated sessions

收案前固定 acquisition reference、ground、signal units、sampling rate、channel order 與 event alignment。
實際 reference／units 尚未核對；不能以猜測值把本草案升級為 acquisition-ready。
Raw continuity、finite values、flat channels、sampling-rate 一致性、event／trial 對齊須成為 fail-fast checks。
Artifact amplitude、flatness duration、接觸品質等數值門檻由 acquisition feasibility 確定並在收案前封存。
禁止利用新 cohort 的 accuracy 或 class-separation 最佳點選 QC threshold。

所有剔除須保存 `reason_code`、適用範圍、原始 epoch keys 與可評估分母。
Subject-level 排除、session 重錄條件、每類最少有效 epochs 亦須事前固定。
某人缺少任一類或不滿最少數量時，該 primary BA 為不可評估，不能補成 0.5 或只算剩下的一類。
此類個案仍列入招募／流失表；不宣稱排除後結果代表所有招募者，也不假設缺失完全隨機。
若預測缺漏來自程式或 artifact 不相容，應使 run 失敗並調查，不能把低分個案當 QC 排除。

A 的 primary 預設每人只使用第一個符合事先 QC 的 A session。
僅在預定的技術失敗條件下允許 replacement；不因 accuracy 差而重錄或選另一 session。
若正式協定改用多個 A sessions，必須在收案前固定 participant-level 聚合方式，再對人等權平均。
B 另算同人的 B endpoint；paired A／B 比較也以人為單位，絕不能把兩 sessions 算成兩人。

不使用人工挑選高分 epochs、label-aware artifact removal 或事後改變 correct-only 定義。
若要評估所有有效心算嘗試，應事前指定為獨立 secondary endpoint，並保留錯誤作答資訊。

## 8. 不確定性與可支持的推論

固定模型外部測試完成後，primary 採 **participant bootstrap**：以人有放回抽樣，每次保留其全部資料。
建議鎖定 10,000 次、固定 seed、兩側 95% percentile CI；正式參數與適用性須通過收案前 simulation gate。
每次先形成每人的 BA，再取 macro mean；paired Δ 使用同一抽樣人的兩模型結果。
若納入多 sessions，抽樣單位仍是 participant cluster；不能把 epochs 或 sessions 當獨立抽樣單位。

此 CI 估計固定 training set 與固定模型下，新 cohort 受試者抽樣的不確定性。
它不包含換一批 training subjects、換實驗中心或換硬體所產生的全部不確定性。
Bootstrap 需要受試者抽樣單位可合理視為獨立；若存在家庭、班級或其他群集，須事前修訂抽樣模型。
小樣本 bootstrap 可能覆蓋率不足；不能因得到數字就宣稱精確，應由預定 simulation 決定可用性。

**不得把歷史五個 LOPO fold 的 SD/√5 直接當作本試驗 CI。**
LOPO 模型共享大量 training data；其折間變異不是固定模型新 cohort 的獨立誤差估計。[5]
也不得以 300／600 epochs 當獨立 N，或以三個分類器當三次獨立重複。
歷史數字及擴充樣本 LOPO 先保留描述性報告；需要正式推論時另定符合其重抽樣結構的方法。
固定 block 設計不允許任意 epoch-label permutation 被解讀成控制時間混淆後的證據。

## 9. 收案數與 precision simulation gate

**目標 N、招募上限、每類最低 epoch 數目前待定；它們是收案前 gate，不是現在阻擋 roadmap PR 的問題。**
先決定希望達到的 participant-macro BA CI 半寬，以及有研究意義的 baseline 增益門檻 `delta_min`。
以多組合理的 participant variation、類別 epoch 數、QC 流失率與 BA 分布模擬候選 N。
受試者內的 epochs 不假設互相獨立；需要時用相關 trial／block 模型產生資料。
模擬須評估 CI 覆蓋率、寬度、missing-class 影響與 paired Δ 的精度，不只算「高於 0.5 的 power」。
不可只代入舊五人的 0.741 或其最小 SD；已經過 representation selection 的觀察可能過度樂觀。

輸出 simulation 設定、seed、候選 N、敏感度範圍、選定 N 與預定招募上限。
停止條件依可評估受試者數或事先寫定的招募上限；不能一邊看 accuracy，一邊決定是否繼續收案。
若資源使精度目標無法達成，事前改列 estimation／pilot study，保留結果而不冒稱充足確認。

## 10. 結果判讀：統計支持、實用價值、未定論

| 層級 | 事先固定的判讀方式 |
|---|---|
| 統計支持 | 在協定與抽樣假設成立時，primary 95% CI 下界高於 0.5，支持此 acquisition protocol 下的 above-chance discrimination |
| 實用增益 | key secondary 的 paired Δ 與事前固定 `delta_min` 比較；須同時呈現 CI，不只看平均改善 |
| 未定論 | CI 跨過 chance／實用門檻，或樣本與缺失使精度不足；不以「沒有顯著」直接宣稱完全無效 |
| 未達目標 | 若 CI 與預定目標明確不相容，報告未達目標及適用範圍，不更換 primary 重新宣布成功 |

Above-chance 不自動等於具有實用價值；相對 baseline 改善也不自動等於絕對表現足夠。
`delta_min` 必須在解盲前訂定；若另需絕對可用性門檻，須同時寫入，不能事後以最接近結果的值取代。
Secondary 的實用門檻評估是估計性判讀，除非預先定義 multiplicity 方法，不另宣稱校正後顯著。
A 與 B 不可事後擇優；A 的良好結果仍受 fixed order／視覺條件限制，B 的差異需依其設計解讀。
不把分類成功等同於已證實神經因果連結；iPLV 結果也不能單獨定位 artifact 或真實同步來源。[3,4]

## 11. Freeze manifest 與啟動條件

正式 freeze manifest 至少包含以下欄位；`pending` 欄位清空後才能升級狀態：

```yaml
status: draft
protocol_id: pending
development_subject_manifest_sha256: pending
confirmatory_cohort_definition: pending
acquisition_reference_and_units: pending
qc_rules_and_minimum_class_counts: pending
primary_endpoint: participant_macro_balanced_accuracy
primary_model: alpha_plv_sfs_lda
secondary_reference: bp_coh_sfs_lda
target_evaluable_subjects: pending
recruitment_cap_and_stop_rule: pending
delta_min: pending
uncertainty_and_simulation_report: pending
code_config_environment_hashes: pending
fitted_model_and_feature_schema_hashes: pending
```

收案前確認 acquisition／QC／樣本數／終點等欄位完整，並記錄協定簽定與版本時間。
新 test 資料進入分析前確認 fitted artifacts、訓練資料邊界、hash guards 與資料契約完整。
Prediction 前確認 test subject／session IDs 與 development 無交集；score 前確認 predictions 已封存。
正式註冊若採外部服務，另填真實註冊 URL／時間；本文件不預填或暗示已完成。

## 12. 實作驗收與變更規則

- 合成 fixture 證明更改 test labels 不會改變 fitted artifacts 或 predictions；只影響評分。
- Fit spy／明確介面證明 test data 未進 scaler、SFS、classifier；schema 缺欄、錯序或 ID 交集必須拒絕。
- Test epoch 重排只改變列順序，不改變依 `epoch_id` 對齊的預測；duplicate keys 必須失敗。
- Prediction／model／eligibility hash 任一不符時拒絕 score，不靜默重算後覆寫。
- 驗證 missing-class 不會產生假 BA；多 session 不會增加 participant N 或改變等權原則。
- 兩模型 paired Δ 必須使用同一 eligibility manifest；不完整 predictions 不能被默默丟棄。
- 外部測試與 expanded LOPO 使用不同 analysis IDs、輸出路徑與報告標題。

看到新 cohort 的績效後，若修改模型、QC、選特徵、門檻或 protocol，該 cohort 轉為 development。
保留原主要結果與 amendment；下一次確認需新的未見 holdout，不重新包裝同一批資料。

## 參考來源

1. Cawley & Talbot (2010), [On Over-fitting in Model Selection and Subsequent Selection Bias in Performance Evaluation](https://www.jmlr.org/papers/v11/cawley10a.html)：有限資料上的流程選擇仍可能造成評估偏差。
2. White & Power (2023), [k-Fold Cross-Validation Can Significantly Over-Estimate True Classification Accuracy in Common EEG-Based Passive BCI Experimental Designs](https://pmc.ncbi.nlm.nih.gov/articles/PMC10346713/)：長 block 切分 epochs 的時間相關問題；block-wise CV 也不是保證無偏的解方。
3. Bastos & Schoffelen (2016), [A Tutorial Review of Functional Connectivity Analysis Methods and Their Interpretational Pitfalls](https://www.frontiersin.org/journals/systems-neuroscience/articles/10.3389/fnsys.2015.00175/full)：reference、volume conduction、SNR、common-input 與樣本限制。
4. Bruña et al. (2018), [Phase Locking Value revisited: teaching new tricks to an old dog](https://arxiv.org/abs/1710.08037)：PLV、iPLV／ciPLV 的方法與 zero-lag sensitivity。
5. Varoquaux (2018), [Cross-validation failure: Small sample sizes lead to large error bars](https://arxiv.org/abs/1706.07581)：小樣本 CV 的不確定性，以及不能直接由折間標準誤宣稱精度的原因。
