# 可追溯執行與歷史資料遷移

本頁配合 Issues [#7](https://github.com/PME26Elvis/eeg-ma-spectral-connectivity/issues/7) 與 [#8](https://github.com/PME26Elvis/eeg-ma-spectral-connectivity/issues/8)。研究方法及歷史分數不變；新增的契約讓錯誤輸入、混用結果與不相容續跑在產生正式輸出前失敗。

## 1. 新收案資料的完整流程

每個 raw session 需提供明確的 schema 2.0 metadata；格式、時間對齊、marker policy 與拒絕條件見 [資料契約](data_contract.md)。同一人的 `subject_id` 必須跨 sessions／cohorts 保持相同文字身分，`001`、`1`、`NA` 等字串不會被自動改成數字或缺值。

安裝專案後，在 repository 根目錄執行：

```bash
python scripts/run_all.py \
  --raw-root /path/to/validated-sessions \
  --config configs/lab_replication.json \
  --out-root outputs/study-001
```

`run_all.py` 明確將當次生成的 `features/baseline/features_all.csv` 傳給 intra／inter runner。所有步驟在同檔案系統的暫存目錄執行，全部成功後才發布 `outputs/study-001/`。任一步驟失敗皆不發布半套資料；目的地已存在則拒絕覆寫。這個整批入口不提供半途續跑，較長的評估可改用下列獨立 runner。

已抽取的 schema 2.0 特徵可直接評估：

```bash
python scripts/03_intra_subject.py \
  --features outputs/study-001/features/baseline/features_all.csv \
  --config configs/lab_replication.json \
  --out-dir outputs/analysis-002/intra

python scripts/04_inter_subject.py \
  --features outputs/study-001/features/baseline/features_all.csv \
  --config configs/lab_replication.json \
  --out-dir outputs/analysis-002/inter
```

同一命令增加 `--resume`，只有輸入 bytes、effective config、程式 commit／working tree、環境、欄位順序、epoch metadata、參考結果與所有 splits 完全相同，且輸出檔完整通過驗證，才會重用既有 run。請先固定 checkout 與環境再執行；程式或設定改變時使用新的 output root。完整細節見 [run provenance](run_provenance.md)。

## 2. 歷史 feature tables 的明確身分

`data/features/` 是封存的五人 discovery tables。它們沒有原始 session／recording 身分、raw hash、單位或參考電極等完整資訊，因此一般 loader 會拒絕默認推測。已提供 [legacy mapping](../configs/legacy_discovery_mapping.json)，明確宣告合成的 legacy session／recording namespace，並將無法證實的 acquisition metadata 記為 `unknown`。

此 mapping 綁定三張已提交的 combined CSV 的 SHA-256；搬移檔案不影響讀取，改動 bytes 則必須先重新稽核並建立另一份明確 mapping。載入後增加 schema metadata，不修改原 CSV，也不把歷史資料標為通過 raw QA。`identity_origin=legacy_manifest`、`qc_status=not_revalidated` 會跟隨資料進入預測與 manifest。

```python
from eeg_ma.schema import load_feature_table, assert_same_epochs

mapping = "configs/legacy_discovery_mapping.json"
baseline = load_feature_table("data/features/features_all.csv", legacy_manifest=mapping)
extensions = load_feature_table(
    "data/features/extensions/features_extensions.csv", legacy_manifest=mapping
)
assert_same_epochs(baseline, extensions)
```

必要的具名重現工作可對 scripts 03／04／06／09／10／12 傳入 `--legacy-manifest configs/legacy_discovery_mapping.json`，並使用新的 `--out-dir` 或 `--out-root`。同一批五人資料的探索停止規則維持有效；本次修正沒有重新跑研究矩陣或選出新最佳模型。

現存 `results/` 與 `data/features/` 的 [逐檔 inventory](manifests/legacy_discovery_inventory.json) 標為 `legacy_unverified`。現存檔案雜湊可證明本次修正沒有改動封存內容，不能證明當年的程式版本、環境、raw 身分或執行成功狀態，也不能用來通過 trusted resume。

## 3. 部分實驗與報告

矩陣 runner 將每個 experiment 存為獨立、不可變的完成目錄。報告從相容的全部已完成 inventory 重建，發布到 `reports/<report_id>/`，最後才原子更新 `CURRENT.json`。只要求一個已完成 experiment 的 `--resume`，不會把其他已完成 experiments 從報告刪除。初次只完成部分矩陣時，coverage metadata 會列出 missing experiments；它不會被標成完整 study matrix。

程式讀取最新完整報告時應使用 resolver：

```python
from eeg_ma.provenance import resolve_report_root

report = resolve_report_root("outputs/results/final_validation")
summary_path = report / "final_validation_summary.csv"
```

不要假設新 summary 仍寫在 family root。Managed matrix 缺少完整報告或報告被修改時 resolver 會拒絕，不會悄悄讀取另一份舊 summary。

## 4. 圖表與行為摘要

scripts 07／13 讀取經驗證的 managed run／CURRENT report，輸出到新的 `outputs/figures/` 子目錄；全部圖表產生成功後才發布。`figure_sources.json` 記錄實際使用的 CSV hashes。繪圖不再回寫來源 `feature_set_comparison.csv`。scripts 05 的行為摘要也使用共同 session QA，保留 cohort／session／recording 身分，並拒絕覆寫既有 CSV。

要從已封存的研究結果重畫圖，必須明確承認 historical execution provenance 未驗證：

```bash
python scripts/07_make_figures.py \
  --results-root results --allow-legacy-results \
  --out-dir outputs/figures/discovery-baseline

python scripts/13_make_final_figures.py \
  --results-root results --allow-legacy-results \
  --out-dir outputs/figures/discovery-final
```

新 baseline 與 feature-set matrix 可分別指定：

```bash
python scripts/07_make_figures.py \
  --results-root outputs/study-001/results/baseline \
  --feature-results-root outputs/results/feature_sets \
  --out-dir outputs/figures/study-001
```

完整 result contract、參考結果相容性與圖表缺列／重複列的研究報告政策，仍由 [#12](https://github.com/PME26Elvis/eeg-ma-spectral-connectivity/issues/12) 追蹤；本次原子輸出與來源驗證不代表新 cohort 的推論契約已完成。

## 5. 驗證邊界

自動測試使用合成 raw／feature 資料重現缺陷，並以封存 feature tables 驗證 legacy adaptation 的 600 列 epoch 對齊和數值保留。沒有公開 raw EEG 的條件下，本次不聲稱重新驗證過歷史採集時鐘、reference、artifact、排除規則或原始執行環境。新 cohort 的固定模型、前瞻 protocol、鎖定環境及受試者層級推論仍依原 roadmap 後續 Issues 完成。
