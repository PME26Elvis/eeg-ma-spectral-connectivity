# 手刻 RF／LightGBM 的嚴格訓練重播與獨立數學 oracle

對應 Issue [#25](https://github.com/PME26Elvis/eeg-ma-spectral-connectivity/issues/25)。先前 validate_complete_run／RF source audit／LightGBM source audit 會驗證來源、模型結構、inner 選模記錄及 outer 預測，但不會重訓全部 inner candidates。本工作新增成本較高的完整重訓，以及另一份真正獨立的 tiny exhaustive oracle；**請勿混淆兩種證據**。

## 1. 三層檢查的精確涵義

| 檢查 | 實際 coverage | 不涵蓋的內容 |
| --- | --- | --- |
| Fast source-bound audit | 完整 run inventory、input/schema、LOPO train/test、RF bootstrap/OOB/節點統計、LightGBM bins/GOSS/EFB/G/H、outer predictions，及 inner records 自洽性 | 不是所有 inner candidate 的重新訓練或獨立的每一分裂最優性證明 |
| **strict_train_replay** | 在原 commit／source fingerprint、實際核實的 CPython 3.12.14 hash-locked reference environment 下，**重訓所有候選 × inner folds 與 final refits**，核對新舊模型 JSON、所有 inner validation 機率、原始分數、種子、最佳候選、外層預測及重要度；共同 S 軌會重算 training-only LDA-SFS，F 軌核對完整特徵 | 是同一套原演算法的再次執行，不是全新演算法的獨立證明 |
| **Independent tiny exhaustive oracle** | 另一個數學實作，tiny CART 採 Fraction 精確 Gini 枚舉所有 feature/boundary/節點；tiny booster 採直接逐列導數／Newton objective、所有合法 bin 分割／全域 leaf-wise 候選枚舉，覆蓋 L1/L2、步長截斷、GOSS 與 EFB 小型輸入 | 只保證指定小型 fixture，不聲稱大型模型所有節點已完成數學證明。GOSS sampled indices、feature masks、bin mapping 仍依 fast source audit＋strict refit 核對 |

## 2. CLI

**只在原始訓練 checkout 的相同程式檔案，以及真正核對的 reference environment 執行。** 若來源是別的 commit，請先 checkout 該 commit 再核實對應依賴。S/F 的 run root 必須是已完整發布的子 run，不是摘要 CSV 或尚未完成的資料夾。

~~~bash
# 僅估算模型數和節點成本，不做訓練也不宣稱通過
python scripts/26_strict_verify.py verify \
  --run outputs/exploratory/new-cohort/S \
  --preflight-only

# 指定上限後重訓所有候選
python scripts/26_strict_verify.py verify \
  --run outputs/exploratory/new-cohort/S \
  --out-dir outputs/verification/S-strict-001 \
  --max-outer-folds 20 --max-candidates 8 \
  --max-model-fits 300 --max-projected-nodes 1000000 \
  --max-seconds 900

# 另外執行真正獨立的小型數學 oracle
python scripts/26_strict_verify.py oracles \
  --out-dir outputs/verification/tiny-oracles-001
~~~

預設最大值：8 outer folds、每個 learner 8 candidates、128 complete fits、250,000 projected nodes、600 秒。**超限的 not_run_resource_limit（exit 3）不會建立成功報告；環境或內容錯誤的 failed_integrity_or_environment（exit 2）也不會。只有全部重新訓練完成才是 verified_strict_train_replay（exit 0）。** 單純 preflight 則回報 not_run_preflight_only。

成本為所有 outer folds × 所有 candidate × inner folds 再加每折每模型一次 final refit；估算節點是保守上界而非真實時間或記憶體。max_seconds 是**模型批次之間檢查的 soft timeout**，無法保證單次長 fit 即時終止；GitHub Actions 另設 step/job hard timeout，且保留中止診斷。

報告 strict_report.json 記錄原始 run manifest SHA-256、run ID、code commit/工作檔 checksum、reference environment/lock ID、每個 scope／fold／模型的候選與 fit 次數、原模型與重訓 JSON fingerprint、固定 numeric tolerance：rtol=2e-12、atol=2e-14。只成功發布於指定的全新獨立 output directory，**原 complete run 不會被改寫**。

## 3. 新增的自動化驗證

Reference CI 使用 [合成五模型 S/F smoke](../tests/run_five_classifier_exploratory_smoke.py) 產生 3 位 synthetic subjects、6 sessions、48 epochs；在每一個 S/F track 上重新執行 3 個 outer folds × 2 learners ×（2 candidates × 2 inner folds＋1 final）＝**30 次完整 refit／track**。本 smoke 不 mock 模型、環境、訓練時間或程式身分，並保存所有輸入、輸出與參數。

[Strict CI acceptance](../tests/run_strict_replay_smoke.py) 還會修改一筆 inner RF 機率，但保留 argmax／inner accuracy 和機率總和，並重新計算 output artifact SHA；fast validator 可接受這種**數值自洽卻非原訓練輸出**的篡改，strict replay 會從真實 X/y 重新訓練而拒絕，且不得建立假 verified 產物。這個測試也確保 run 的原始 bytes 沒有被 strict 修改。

參數／結構／隨機數／bootstrap／OOB／GOSS、zero gain XOR、常數、sample weights、class imbalance、L1/L2、float boundaries、serial/threaded、不可變 JSON 等舊 tests 繼續保留。新的 independent tiny oracle 比較數學枚舉結果；它的報告記錄**實際覆蓋的樹／節點／分裂數**，不能假裝已驗證大型 RF/LightGBM 全部分裂。

## 4. 手動執行與下載證據

[Manual strict training replay](../.github/workflows/strict-replay.yml) 是 workflow_dispatch。手動設定最大 fits、projected nodes 和秒數；workflow 會：

1. 以 wheel hashes 從零建立 CPython 3.12.14 reference environment。
2. 實際執行 independent oracle 及 synthetic S/F complete run。
3. 重新 fit 兩條軌道每個 candidate/inner/final model，驗證全部結果。
4. 封存 reference install receipt、環境證據、完整來源、S/F model／audit、strict report 與 oracle 報告，保存於 GitHub Actions artifacts（90 天）。

**手動 workflow 的輸入是 synthetic，不會自動讀私有原始 EEG。** 若需驗證自己已有的完整新研究 run，請在原研究環境直接執行 CLI，並保留相關來源及核准。舊五人的原始 243 份 artifacts、A/B protocol、README 既有教授簡報 snapshot 均不變。

## 5. 科學與工程限制

Strict replay 驗證的是**原模型的數值／資料／訓練程序重現**，不是更多 EEG 受試者、模型可泛化的證據，也不是新模型會比舊模型更好的宣稱。跨 compiler／platform／依賴版本的近似吻合須使用另立受控比較程序，不能以此冒稱相同的實驗環境。

獨立數學證明的範圍是**tiny exhaustive fixtures**；大型參數搜尋的模型仍由 fast source audit ＋ full inner refit 提供源頭、訓練與預測證據。如果要對所有大型節點做不同語言實作的形式驗證，需要另設有界、可執行的新工作項目。
