# 受試者結果契約與可驗證報告

本文件對應 [#12](https://github.com/PME26Elvis/eeg-ma-spectral-connectivity/issues/12)。歷史結果由現有逐 epoch predictions 重建；固定模型的新受試者報告則讀取 [#11 評分封存](fixed_external.md)。兩者共用身分、計數與配對契約，保留各自的估計對象與證據限制。

歷史成果沒有重新訓練或新增實驗。`results/`、`data/features/` 與原有 [discovery inventory](manifests/legacy_discovery_inventory.json) 維持原始 bytes；新產物寫入全新的 `outputs/` 目錄。

## 1. 每個數字從哪裡來

`result_contract.py` 的標準 prediction row 包含以下欄位：

| 類別 | 欄位與意義 |
| :-- | :-- |
| 分析身分 | `analysis_id`、`run_id`、`cohort_id`、`protocol_id`、`experiment`、`classifier`、`comparison` |
| 模型 | `model_id`；LOPO 每個 held-out participant 可有不同模型，不能以 model ID 把同一人的結果拆成多個人 |
| 資料身分 | `subject_id`、`session_id`、`epoch_id`、`eligible_epoch_manifest_id` |
| 結果 | `y_true`、`y_pred`，僅允許二元數值標籤 0／1 |

識別碼以 literal string 讀取，`001`、`1`、`NA` 不互相合併。每一個分析組合與 epoch 的 key 唯一；同一 epoch 不能在另一模型中改換受試者、session、comparison、真值或 eligibility 身分。預期分析矩陣及 eligible epoch manifest 必須獨立提供，觀察到的 rows 不能自行證明「已完整」。

歷史預測的 comparison-local row position，透過 [明確 legacy mapping](../configs/legacy_discovery_mapping.json) 與已封存 feature tables 對齊。三組 features 的實際 epoch 身分先核對一致，才可接回預測。既有 configuration 若明示不同 experiment／feature family，會直接拒絕；缺失的舊 baseline configuration 仍記為未知。

對舊檔新增 content ID 只識別現在可讀取的 bytes。它不回填未知的 raw acquisition、歷史執行環境、原始程式版本，亦不假裝保存了不存在的 fitted model。這些結果仍是 `legacy_unverified`。

## 2. 統計單位與缺失值

每位受試者先累計自己所有 eligible sessions 的 `tn`、`fp`、`fn`、`tp`，再計算：

\[
\mathrm{accuracy}_s=\frac{TP_s+TN_s}{TP_s+TN_s+FP_s+FN_s},\qquad
\mathrm{BA}_s=\frac12\left(\frac{TN_s}{TN_s+FP_s}+\frac{TP_s}{TP_s+FN_s}\right).
\]

BA 必須同時有兩個 true classes；缺任何一類就不可評估。CSV 使用空值、JSON 使用 `null`，不補 0.5，也不把單一 class recall 當成 BA。招募、QA、不足正確試次、未完成與沒有可用 session 的人仍保留於 flow，並顯示對應 endpoint 的分母及原因。

Participant macro 是每人一票；同人多 sessions 不增加 N。Pooled epoch accuracy／BA／recalls 另列，不代替 primary estimand。例如兩人的 accuracy 分別是 2/2 與 1/10，participant macro 是 0.55，pooled epoch accuracy 是 3/12 = 0.25。這是統計單位不同造成的差異，不是格式上的四捨五入。

歷史五人 LOPO 只提供描述平均與**受試者間 sample SD（ddof=1）**。不以 `SD / sqrt(5)`、重複 sessions 或 600 epochs 製造 population CI；LOPO 訓練集重疊及同 cohort 的歷次選擇也不會因為換報表而消失。

## 3. Paired Δ 的必要條件

`paired_participant_deltas()` 要求相同 analysis、cohort、protocol、classifier、comparison，以及完整相同的 participant、session、epoch、真值與 eligible-manifest 身分。Reference 和 target 的 run／experiment 身分各自保留；不能拿另一個 cohort 的 reference、拿 fixed-model external 與 retrained LOPO 混配，或靠 inner join 丟掉少了一邊的人。

Reference 缺列、重複鍵、不同 eligible epochs 或不相容 analysis 都會明確失敗。不同 protocol 的 A/B transport 另外以人配對，不能宣稱它們測了相同 epochs；[B 控制計畫](phase_controls.md) 對這個不同估計對象另行標示。

## 4. 重建歷史報告與正式圖表

在 repository root 執行下列命令，每次使用新的 output directory：

```bash
python scripts/22_build_report.py historical \
  --allow-legacy-results \
  --legacy-manifest configs/legacy_discovery_mapping.json \
  --out-dir outputs/historical-report-001

python scripts/22_build_report.py verify-historical \
  --report outputs/historical-report-001

python scripts/13_make_final_figures.py \
  --allow-legacy-results \
  --legacy-manifest configs/legacy_discovery_mapping.json \
  --out-dir outputs/final-figures-001

python scripts/22_build_report.py verify-historical \
  --report outputs/final-figures-001
```

完整的既有圖表範圍是 **11 個具名分析 × 2 comparisons × 3 classifiers = 66 cells**；由 **19,800 筆已封存模型預測**重建 **330 筆 participant results**。19,800 是跨模型／分析重複出現的預測列數，不是新增 epochs 或獨立人數。所有 cell 的 participant N 仍是 5。

正式效能圖直接讀取重建的 participant summary。SFS network 圖的資料則是與 prediction fold／participant 對齊的 training selections，明示其數值為選取次數，不能解讀為 classification accuracy 或腦部連結強度。Alpha candidate pair、每折選取清單與彙整頻率也會交叉驗證。

每張圖旁保留實際數值 CSV 和 JSON metadata；caption／metadata 記錄 metric、該 comparison 的 participant N、SD、cohort、protocol、來源 inventory hash 及數值 CSV hash。缺少任一必需 cell、非有限值、矛盾重複參照，或整個 participant 從三個 classifiers 一起消失，都會阻止正式輸出。

`historical_report.json` 在全部表格、圖像與 sidecars 寫完後才封存 inventory，並在原子發布前重讀原始來源驗證。`verify-historical` 預設重建原始來源；若原始路徑已移除，可使用 `--detached` 明確只檢查封存檔的內部一致性。Detached 模式不能宣稱重新核對了原始資料。

局部報告必須明確宣告，例如：

```bash
python scripts/22_build_report.py historical \
  --allow-legacy-results \
  --legacy-manifest configs/legacy_discovery_mapping.json \
  --experiments poster_baseline plv_alpha_only \
  --partial-report \
  --out-dir outputs/partial-baseline-alpha-001
```

其 scope 為 `partial_historical_matrix`，列出已納入與缺失分析。既有 complete report 和 canonical inventory 不會被縮寫或覆蓋。正式 output directory 一旦存在，任何再次寫入均拒絕。

## 5. 固定模型外部報告

先完成 [fit → lock → predict → score](fixed_external.md)，再建立報告：

```bash
python scripts/22_build_report.py external \
  --scores outputs/external-score-001 \
  --predictions outputs/external-predictions-001 \
  --lock-dir outputs/study-lock-001 \
  --out-dir outputs/external-report-001

python scripts/22_build_report.py verify-external \
  --report outputs/external-report-001
```

入口重新驗證 predictions、模型、鎖定協定、closed enrollment、labels 與被授權的 outcome-access event。報告包含完整 sources 副本；刪除原始 output directories 後，仍可從封存來源重算。報告與驗證過程不 fit、不重新選特徵，也不開啟新的預測／解盲交易。

主要產物如下：

| 產物 | 內容 |
| :-- | :-- |
| `predictions.csv`、`eligible_epochs.csv` | 統一 schema 的完整預測與 eligible epoch 身分 |
| `participant_results.csv` | 全部招募者 × 兩 comparisons × 兩模型；計數、每人指標與 endpoint eligibility |
| `participant_flow.csv`、`session_flow.csv` | 招募、失訪／排除原因、selected session、replacement 與 QA evidence |
| `summary.json` | Participant macro、class recalls、paired Δ、分開的 pooled epoch metrics、CI 與協定解讀 |
| `figure_data.csv`、`figure_metadata.json` | 數值已驗證的作圖輸入、metric、N、CI 設定及來源 hash |
| `report.md`、`report_manifest.json` | 完整敘述、限制、不可變來源 inventory 與 report identity |

Bootstrap 完全沿用 A 協定：**10,000 次、seed 20261109、雙側 95% participant percentile interval、linear quantiles**。每個 comparison 依排序後的相同 participant IDs 產生 resampling indices，primary／reference 與 paired Δ 共用同一批抽樣；indices 的 shape、dtype、hash 也會保留。類別、session 或 epoch 不是獨立抽樣單位。

CI 的範圍是：**給定已固定 development models，新 QA-evaluable participants 的抽樣不確定性**。它不包含模型重訓、feature selection、重新設計流程或母體中所有被排除者的不確定性。Pooled epoch 指標不附偽裝成 participant uncertainty 的 CI。

Pilot、目標 N 未達、少於兩位可評估者、degenerate interval 與 precision shortfall 都由協定 guard 標示；不因漂亮點估計而恢復確認性資格。主要 endpoint 的解讀沿用原本 above-chance／研究門檻／precision 規則，secondary／reference／paired Δ 的 nominal intervals 保持描述性。負結果與不可評估結果仍完整發布。

這份報告的 complete 範圍是 **固定模型 Protocol A endpoints**。Expanded nested LOPO 需要另一份重訓學習流程的描述報告；B 控制實驗也不會被外部報告自動啟用。

## 6. 驗證邊界

Tests 比較實際數值、完整 roster、來源 hashes、配對鍵和錯誤行為。包含 unequal epoch counts、多 sessions、缺 class、missing participant、conflicting duplicate、錯誤 feature family、資料／敘述／作圖 metadata 被修改後重新 hash 的情境。CI 另外實際重建 frozen report 與圖表；沒有以「圖片存在」作為唯一驗收。

Hash 與本地 ledger 是可重現、可檢查的研究契約，不是電子簽章或人類未曾私下看結果的證明。無法追溯的歷史資訊，以及尚未取得的新 cohort／設備事實，都維持明確未知。
