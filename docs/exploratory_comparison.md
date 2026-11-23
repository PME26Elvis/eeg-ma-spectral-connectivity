# 五分類器 S/F 雙軌：新受試者探索性 LOPO 比較

對應 [Issue #24](https://github.com/PME26Elvis/eeg-ma-spectral-connectivity/issues/24)，需先完成 [Issue #22](https://github.com/PME26Elvis/eeg-ma-spectral-connectivity/issues/22) 的 LightGBM 主線整合。

**此功能不是確認性 Protocol A、不是 B 相位控制、不是 2026-10-21 向教授報告的原始研究；亦不允許拿原五人 discovery 再挑模型。** 它提供一個新 cohort 上可驗證的探索性比較流程。合成測試沒有任何真實 EEG 準確率主張。

## 1. 明列的研究對象與七個比較格

兩個 track 使用**相同、獨立先宣告的**比較、受試者、sessions、eligible epochs、輸入 feature family 與 outer/inner participant LOPO membership：

| Track | Feature policy | Classifiers | 這回答什麼問題 |
| --- | --- | --- | --- |
| S | 共同 training-only LDA forward SFS | LDA、RBF-SVM、KFDA、手刻 RF、手刻 LightGBM | 相同 LDA 選定特徵下分類器差異 |
| F | 全部預先宣告的候選特徵，**不執行 SFS** | 手刻 RF、手刻 LightGBM | 樹模型接受完整候選特徵時的描述差異 |

每個 comparison 有七個固定格，並不是七個獨立研究。固定配對示例：S:RF − S:LDA、S:LightGBM − S:LDA、F:RF − S:RF、F:LightGBM − S:LightGBM。要比較別的配對，必須**先在 plan 列出**；所有零、負、不可估值都保留，不可看完結果挑「模型冠軍」。

F track 在現有評估 CSV 中用帶有 <code>selection_method=all_declared_features</code> 的 inclusion trace 表示全特徵，<code>cv_accuracy</code>／<code>improvement</code>／<code>sfs_inner_accuracy</code> 全為 null；它**不是做了 SFS 卻取得空分數**。配套 classifier-policy-3、evaluation-contract-3 及 run validator 會拒絕遺漏／改順序的特徵、非空的虛構 SFS 分數或把 LDA 也加入 F。未設定此模式的原三分類器／RF-only／五分類器 S 保持原 contract。

## 2. Split、調參與數據的邊界

由 <code>stable_epoch_id_v1</code> 建立 outer LOPO；所有同一人的 sessions 同側。每一折內：
1. 只用 outer-training 對 S 執行 LDA-SFS；F 直接使用已宣告的全部特徵。
2. RF／LightGBM 在 each inner-training fold 重新 fit；LightGBM 的 bins／EFB、RF bootstrap 隨訓練資料內產生。SVM／KFDA 的 scaler／grid 同樣僅使用當折 training rows。
3. Inner search 完成後僅以 outer-training refit，最後才預測 outer held-out participant；不得用 outer labels、outer features 做 SFS、分箱、調參或 early stopping。
4. 即使 CSV row 順序改變，外層與內層 epoch-id membership 不應改變。輸出及封存仍包含原始資料 bytes 與配置 hashes，row reorder 導致 input bytes／run identity 改變是**可追查的現象**，不是 splitter 改變。
5. Inner accuracies 是同一批 folds 上選模的 diagnostics，不可以拿來當獨立表現估計或 confirmatory CI。

## 3. 計畫與資料的完整性

權威模板：[five_classifier_exploratory.plan.json](../configs/five_classifier_exploratory.plan.json)。預設為 <code>draft</code>／placeholder cohort，**不能直接執行**。必須由新研究操作者在取得模型結果前固定：

- 新的 cohort 與 acquisition protocol ID；一個預先指定的 feature_set／comparisons；**ordered_features 精確且有序的候選欄位清單**；完整 S/F roster 與配對名單。不能只寫 `feature_set=all` 就允許後續依分數移除欄位。
- **獨立先固定的 eligibility CSV**（不可從最終 predictions 推導）列出完整 cohort 的所有預期 epochs，欄位順序為 `cohort_id,protocol_id,comparison,subject_id,session_id,epoch_id,y_true`。將該檔實際 bytes 的 SHA-256 填入 `plan.eligibility_sha256`；check/run/verify 會核對 eligibility vs features、每個格子的 predictions，缺整位 participant 或單一 epoch 均拒絕。此核對只能驗證已封存的 roster，真正的凍結時間和獨立性需由研究者提供外部事前證據。
- 已通過 schema 2.0／raw_metadata QA 的來源，以及確認不是原五人 discovery 或曾見過結果的受試者。
- 真實新研究標記為 <code>prospective_exploratory_frozen</code>，填入 <code>registration_evidence.frozen_at/reference/operator</code>；程式只驗證聲明格式、身份一致與檔案雜湊。**這不是實際倫理核准、資料未暴露或現場設備驗證的替代品**，仍須依 #23、既有 A/B 與院校規範完成。
- 只有測試時可用 <code>synthetic_only</code>，而且必須明確傳入 <code>--allow-synthetic</code>。這一狀態永遠不代表完成真人驗證。

設定使用新的 [五分類器 profile](../configs/lightgbm.json) 或從它另存先固定的副本。原 A 的固定 LDA 路線、B 控制計畫、原三分類器結果均不能被這份 profile 修改。

## 4. CLI

在專案 root、正確安裝 Python dependencies 後：

~~~bash
python scripts/25_compare_classifiers.py check \
  --features /path/to/new-schema2-features.csv \
  --config configs/lightgbm.json \
  --plan /path/to/declared-exploratory-plan.json \
  --eligibility /path/to/frozen-eligibility.csv

python scripts/25_compare_classifiers.py run \
  --features /path/to/new-schema2-features.csv \
  --config configs/lightgbm.json \
  --plan /path/to/frozen-exploratory-plan.json \
  --eligibility /path/to/frozen-eligibility.csv \
  --out-dir outputs/exploratory/cohort-run-001

python scripts/25_compare_classifiers.py verify \
  --report outputs/exploratory/cohort-run-001

# 不接收真實 EEG 的自動工程煙霧測試
python tests/run_five_classifier_exploratory_smoke.py \
  --out-dir outputs/exploratory/synthetic-smoke-001
~~~

路徑必須是全新輸出路徑；現存 output 會被拒絕覆寫。建立全套流程使用 staged → complete validation → atomic publish，不會更新原 results 或 features 目錄。<code>check</code> 只驗證規則、**不訓練**。

## 5. 報告來源與統計單位

| 封存檔案 | 驗證功能 |
| --- | --- |
| <code>source_features.csv</code>、<code>analysis_config.json</code>、<code>plan.json</code>、<code>eligibility_source.csv</code> | 原始 bytes，cohort／protocol／特徵與分析政策 |
| <code>S/</code>、<code>F/</code> | 兩個完整的 #7 不可變 evaluation run，各有 held-out/inner splits、逐折結果、模型來源稽核 |
| <code>cohort_flow.csv</code> | 每個 comparison 從獨立 roster 列出**預期 eligible 受試者 N、sessions、epochs、兩類別數**；未提供的 recruited／excluded 人數明記 unknown，**不假裝已統計完整收案流程** |
| <code>eligible_epochs.csv</code>、<code>predictions.csv</code> | 事先固定 cohort／comparison／epoch，完整 model×comparison×epoch 覆蓋 |
| <code>participant_results.csv</code>、<code>summary.csv</code> | TP／TN／FP／FN、每人 accuracy／BA／recalls、participant macro／SD 與獨立 pooled epoch metrics |
| <code>paired_participants.csv</code>、<code>paired_summary.csv</code> | 每個事前配對的人數、相同 eligibility 下的 BA 差與負／零／null |
| <code>report.md</code>、<code>report_manifest.json</code> | 非確認性敘述、run IDs、每個檔案 SHA-256、content-addressed report ID |

在每一個 group 中，**同一人所有 sessions 先合計 confusion counts，再算 BA**，一人一票；pooled epoch accuracy 與 participant macro 分欄。少一個 classifier、participant、epoch、comparison 或不同 eligibility，完整性驗證會失敗，不能靠 inner join 丟掉缺失。

驗證器重讀 S/F run manifests 與 RF／LightGBM 檔案來源稽核，由**保存的 predictions** 重新產生所有六張 CSV 與 markdown，與封存的每個 byte hash 對照，不能只相信 report.json 上寫的 passed。

這些模型 fold IDs 是可辨識的**預測分組**，不等於已序列化的 LDA／SVM／KFDA fitted estimator。現有 fast auditor 不會重 fit 全部 inner candidates，嚴格數學訓練重播另由 [Issue #25](https://github.com/PME26Elvis/eeg-ma-spectral-connectivity/issues/25) 追蹤。

## 6. 交付界線

本 Issue 完成的是新的工程與探索性方法；**不修改 243 份歷史 canonical artifacts，不新增原五人效能宣稱，不產生 A 的正式 CI 或任何科學顯著性結論**。正式 prospective 研究應另有事前固定的註冊、收案、倫理及未見 holdout；有新的方法假設必須開新版本，而不是同 cohort 不斷選比較好看的結果。
