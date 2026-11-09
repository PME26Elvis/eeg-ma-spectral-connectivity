# Run identity、可信任 resume 與原子報告

此文件對應 [Issue #7](https://github.com/PME26Elvis/eeg-ma-spectral-connectivity/issues/7)。操作入口與 #8 的資料契約見 [integrity workflow](integrity_workflow.md)。歷史研究的特徵定義、SFS、分類器與數值方法不因本機制改變。

## 1. 輸入、執行與報告的身分

新的 evaluation runner 接受 schema 2.0 feature table。歷史 CSV 必須以 `--legacy-manifest configs/legacy_discovery_mapping.json` 明確對應 cohort／session／recording；這項轉換只在記憶體發生，不改寫歷史 CSV。對照表見 [legacy discovery mapping](../configs/legacy_discovery_mapping.json)。

`build_run_identity` 在同一次 invocation 固定以下證據：

| 類別 | 記錄內容 |
| :-- | :-- |
| Code | Git commit、相關 source dirty 狀態、working source files hashes、dirty fingerprint |
| Environment | Python、platform、scientific package versions、lock files hash；沒有 lock 明記 `unknown` |
| Method | 完整 effective config、experiment 定義、matrix config；`feature_set=all` 仍只代表 BP＋COH |
| Input | CSV／config／legacy mapping 的實際 bytes SHA-256 與大小 |
| Feature table | ordered columns、dtypes、全表 hash、完整 epoch metadata、ordered selected feature names |
| Partition | 與 evaluation 同源的 outer／inner split membership，記錄 epoch IDs 與 subjects |
| References | Managed reference run IDs 與 manifest hashes，或明確標記 `legacy_unverified` 的現存檔案 hashes |

Identity 的 canonical JSON SHA-256 是 `run_id`。Absolute output path 和完成時間不參與這個 ID。Column order、資料 bytes、metadata、split、code 或環境的改變會產生不同 identity；不是只比較 JSON 檔名或 `random_state`。

同次 invocation 不在每個 experiment 重新讀取 code／environment 身分，避免一個矩陣的不同項目混用不同捕捉時點。執行中不應編輯程式或輸入檔；跨 invocation 有任何研究程式或依賴改變時，使用新的 output root。這是嚴格的研究執行識別，並非聲稱所有環境變動都必然影響數值。

## 2. 一個 run 如何變成 complete

`RunTransaction` 先檢查 destination，再在同一 filesystem 的 `.staging-...` 目錄執行。正式 destination 在完成前不存在。

1. 若 destination 存在，先驗證完整 manifest、identity、所有 artifacts hashes 和 CSV 契約。沒有 `--resume` 也不會覆寫它。
2. 若是新 run，staging 狀態為 `running`，寫入固定 config／split snapshots 與 evaluation outputs。
3. 驗證完整 fold 與 prediction coverage、每個 classifier、epoch labels／metadata、selected features、SFS paths、frequency、逐折 accuracy、subject summaries。Accuracy 和 summaries 由 predictions／folds 重新核對。
4. 驗證通過後，最後寫入 `run_manifest.json` 的 `status=complete`，再以同 filesystem directory rename 一次發布整個 run。
5. 發生一般 exception 留下 `failed` staging；KeyboardInterrupt／SystemExit 留下 `interrupted` staging。若 OS 直接終止 process，未發布 staging 仍不被視為 complete；不會拿其中一張 folds CSV resume。

Staging 可以供診斷，但不會自動當作完成實驗。重新執行會從新的 staging 開始；這個版本提供 **experiment-level resume**，沒有宣稱 fold-level checkpoint。Failed／interrupted staging 的清理由操作人員在確認不再需要後進行。

Complete run 的任何 required file 遺失、截斷、內容／metadata 被更動或多出未記錄 artifact，都會拒絕 reuse。讀取 CSV 會保留 `001`、`1`、`NA` 等 literal identifiers，不讓 pandas 的預設 NA／數值推斷改變 subject identity。

## 3. Matrix 的預檢與 partial execution

scripts 06／09／10／12 共用 `prepare_matrix` 與 `MatrixRun`。執行前預檢整個既存 inventory，包含未列在本次 `--experiments` 的實驗。任何舊項目失配時，尚未開始第一個新 experiment，也不會先覆寫 snapshot。

Matrix 的 immutable `matrix_manifest.json` 綁定完整 experiment universe 與 context。Requested subset 不改變 context。每個完成 experiment 存在自己的目錄：

```text
outputs/results/final_validation/
  matrix_manifest.json
  plv_alpha_only/
    run_manifest.json
    config_snapshot.json
    split_manifest.json
    inter/
  iplv_only/
    run_manifest.json
    ...
  reports/<report-id>/
    run_manifest.json
    final_validation_summary.csv
    ...
  CURRENT.json
```

聚合階段讀取 **所有同 context、已驗證完整的 experiments**。先只完成 alpha，後來完成 iPLV，再只 resume alpha，報告仍同時包含兩者。未執行項目列在報告 manifest 的 `coverage.missing`；`study_matrix_complete` 只在 required experiment set 完整時成立。Report 本身成功產生與整個研究矩陣已跑完是兩種不同狀態。

每一份報告亦以 staging → validation → atomic directory rename 發布到 `reports/<report-id>`。最後才原子更新 `CURRENT.json`。舊報告不刪除；中途失敗時讀者仍能看到原來完整的版本，避免多張 summary／delta／ranking 半新半舊。

讀取新版報告使用 `resolve_report_root(root)`，會驗證 CURRENT、report identity、檔案 hashes 與 CSV contracts。有 managed matrix 卻缺 CURRENT 時會報錯，不會偷偷改讀同名舊表。讀取原本未被管理的歷史目錄必須明確指定 `allow_legacy=True`，這只允許 read-only access，不會升級為可信任 resume。

Matrix writers 以 OS file lock 排除同時更新同一 root；process 結束後 OS 釋放鎖，不以猜測 PID 自動刪除未知鎖檔。Complete experiment dir 的 publication 也不能取代已存在的非空 run。

## 4. CLI 路徑與相容性

| Script | 新預設輸出 | 保留的語意 |
| :-- | :-- | :-- |
| `03_intra_subject.py` | `outputs/results/baseline/intra` | `--out-dir` 仍直接包含 `intra_*.csv`，加上 immutable manifest |
| `04_inter_subject.py` | `outputs/results/baseline/inter` | `--out-dir` 仍直接包含 `inter_*.csv`，加上 immutable manifest |
| `06_compare_feature_sets.py` | `outputs/results/feature_sets` | `--results-root` 為 read-only baseline；新增獨立 `--out-root` |
| `09_run_extension_experiments.py` | `outputs/results/extensions` | 原有 default／combined experiment 選擇與 reference roots |
| `10_run_plv_decomposition.py` | `outputs/results/plv_decomposition` | 原有 primary／secondary／explicit experiment 選擇與 reference roots |
| `12_run_final_validation.py` | `outputs/results/final_validation` | 原有 inter-only bounded matrix 與 stop rule |

六支 runner 均提供 `--resume` 與 `--legacy-manifest`。新版 matrix 的總表在 `CURRENT.json` 指向的完整 report directory，CLI 會印出實際路徑。請勿再假設 matrix root 下同名 CSV 就是最新報告。

Canonical `results/`、`data/features/`、`data/raw/` 及其祖先均不能作為新輸出；透過 symlink 指回它們也會拒絕。Explicit `--out-root results/final_validation` 不會解除保護。變更方法／設定後，請使用具名新 root，例如：

```bash
python scripts/12_run_final_validation.py \
  --legacy-manifest configs/legacy_discovery_mapping.json \
  --out-root outputs/results/final-validation-audit-001 \
  --experiments plv_alpha_only
```

此例是操作語法，不是建議在舊五人開啟新搜尋。正式研究仍遵守既有 stop rule；工程驗證使用小型 synthetic fixtures。

## 5. 歷史 inventory 的誠實界線

已觀察的歷史 bytes 可以盤點，但不能倒推出原先的 raw hash、執行環境或生成時設定。`14_inventory_legacy.py` 是 explicit inventory 操作：

```bash
python scripts/14_inventory_legacy.py \
  --root results \
  --out outputs/inventory/discovery-results-001.json
```

它不改動 source tree，不覆寫既存 inventory，且保留 `status=legacy_unverified`、`historical_code=unknown`、`historical_environment=unknown`、`raw_hashes=unknown`。舊 inventory 永遠不被 `RunTransaction` 當作 complete run。

[本次 discovery inventory](manifests/legacy_discovery_inventory.json) 同樣只證明盤點當時的檔案。歷史 reference tables 可作明確標記的比較錨點；runner 會檢查它們的 fold、subject、class-count coverage，但相同 counts 不能證明相同 raw source。Managed references 則還要核對 dataset metadata、split membership、方法設定與所宣稱的 feature representation。

## 6. 測試邊界

測試包含真實的小型 SFS／LDA／SVM／KFDA 執行，以及六支 CLI 的 orchestration、partial aggregation、immutable resume、缺檔／錯 schema／錯 metadata、canonical path 和 interruption。這些測試不重跑正式五人研究矩陣，也不宣稱完成 private raw-to-result reproduction。

本機制是防止意外混用、無聲覆寫與不完整產物被當完成的工程契約；manifest 是可稽核的來源記錄，不是由外部信任方簽章的防篡改證明。
