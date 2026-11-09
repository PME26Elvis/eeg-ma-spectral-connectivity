# 固定模型 external inference 與 split 契約

本文件對應 [Issue #11](https://github.com/PME26Elvis/eeg-ma-spectral-connectivity/issues/11)。實作入口為 [`fixed_model.py`](../src/eeg_ma/fixed_model.py)、[`external.py`](../src/eeg_ma/external.py) 與 [`21_fixed_external.py`](../scripts/21_fixed_external.py)。它將 development fitting、模型封存、外部預測與解盲評分分開，並接上 [protocol lock](protocol_gate.md) 及 [reference environment](reproducibility.md)。

目前 repo 的正式研究仍是 **DRAFT／not_registered／blocked**。Synthetic 測試會實際訓練模型與產生預測，但其中的 acquisition、核准及參與者均明示為測試資料。這些測試沒有替真實研究補上證據，也沒有再次搜尋原五人的特徵或更新歷史分數。

## 1. 訓練與測試各自回答什麼問題

| 流程 | `analysis_id` | 訓練資料與輸出 |
| --- | --- | --- |
| 固定模型外部測試 | `fixed_model_external` | 只在 development fit；新 cohort 使用既定 scaler、subset、LDA；寫入獨立 `outputs/fixed-external/...` |
| 跨受試者 LOPO | `grouped_lopo_learning_procedure` | 每個 outer fold 在該折的 training subjects 重新選特徵及 fit；沿用 CV runner 的獨立輸出 |
| 受試者內 nested CV | `within_subject_nested_cv` | 估計同一受試者內的 epoch generalization；不能宣稱跨人驗證 |

Fixed external 的報表標題為 `Fixed development models on an independent external cohort`。Grouped LOPO 的標題為 `Grouped LOPO learning-procedure evaluation`，並透過 `evaluation_contract.json` 保存 estimand、分組政策及分數用途。把新舊資料合併再跑 LOPO，估計的是重新學習流程；它不能替代已凍結模型對獨立新人的驗證。

Primary 是 **21 個 α-PLV candidates → SFS → StandardScaler＋LDA**；reference 是 **42 BP＋126 COH candidates → 獨立 SFS → StandardScaler＋LDA**。兩個模型各自包含 `rest1_vs_type1` 與 `rest2_vs_type2` 的已擬合狀態。A／Type1 是主要 endpoint，Type2 保持 secondary；模型種類及分數高低不改變這個順位。

## 2. Development fit 與可執行 artifact

`fit_development(protocol_path, out_dir)` 只讀取協定指向、且 hash 相符的 development dataset inventory、subject manifest、analysis config、feature schema，以及需要時的 legacy mapping。所有 comparison 都要包含足夠的 development subjects 及兩個類別；literal IDs 如 `001`、`1`、`NA` 不會被 CSV 型別推斷合併。

SFS 使用 development-only LeaveOneGroupOut。Subject 是最高分組單位，同一人的所有 sessions 一起進入該折 train 或 validation；每次 inner StandardScaler／LDA 都只 fit 當次 training rows。選定 subset 後，最終 scaler／LDA 才使用全部 development rows。scikit-learn 的資料外洩指引要求 preprocessing 與 feature selection 只由 training data 學得，後續資料沿用該 transformation。[1][2]

匯出採用可檢查的 **JSON 數值狀態**，不是 pickle／joblib。此版本僅支援協定指定的 binary SVD LDA；它不是任意 sklearn estimator 的通用序列化工具。`FixedLDAModel` 只公開 `predict`、`decision_function` 與可複製的文件／身分資訊，沒有 `fit` 或 `partial_fit`。讀取時驗證：

| 契約 | 保存及驗證的內容 |
| --- | --- |
| 特徵／類別 | 完整 ordered candidate features、selected indices／names、`0=Rest`／`1=MentalArithmetic`、schema 原文與 SHA-256 |
| Scaler | fitted mean、variance、scale、樣本數、固定參數；檢查 variance／scale 關係 |
| LDA | classes、coef、intercept、means、priors、scalings、xbar、solver 參數及 fitted dimensions；檢查 fitted state 的相互一致性 |
| Training | 全部 development subjects、每個 comparison 的穩定 epoch／session identity、inner train／validation memberships、SFS trace |
| Provenance | 原始 protocol／config／schema bytes、data／development／eligibility hashes、code commit／source fingerprint、實際 environment 及 reference lock identity |
| Artifact | canonical `model_id`、原始 JSON bytes SHA-256，以及模型與 training manifest 之間的交叉綁定 |

推論只依選取後的欄位計算 `(X - mean) / scale`，再使用已保存的 LDA coefficient／intercept；decision score 嚴格大於 0 才是 class 1，平手為 class 0。匯出時直接與原 fitted sklearn pipeline 比對 predictions，測試另以新 subjects 比對 round-trip。[3]

Bundle 包含 `primary_model.json`、`reference_model.json`、兩份 `*_training_manifest.json`、`environment.json`、`bundle_manifest.json` 與供稽核的輸入文件副本。`bundle_manifest.artifacts` 的五個角色可接入 protocol freeze；`input_evidence` 是訓練時的原始輸入副本。**不要直接把 development dataset inventory 的 locator 改成副本路徑**：原始 inventory 內相對 source paths 的基準目錄仍有意義。正式 lock 另會保存實際 source bytes。

### 避免 model 與 lock 互相依賴

模型先於 lock 產生，因此它不能預先持有尚不存在的最終 `lock_id`。`protocol_training_contract` 會保存完整科學決策，僅排除 `status`、`approval`、`study.registration`、`study.outcomes_accessed` 與 `freeze.artifacts` locators。模型另外保存 fit 當下的完整 source protocol bytes/hash，以及實際資料、設定、schema、環境的 hashes。

這讓操作者可填入新產生的 model／environment 路徑後建立 lock；變更 acquisition、eligibility、endpoints、分析政策或 protocol version 仍會改變 training contract，不能讓舊模型冒用新決策。Study document version `1.2` 與 A acquisition protocol version `1` 分開保存，不能互相代替。

新的 `freeze.model_artifact_policy=executable_fixed_lda_v1` 要求真實可驗證的模型 JSON、reference training scope、clean identifiable fit code 與相符 training manifest。歷史 opaque-model archives 仍可稽核，但不能啟用新的 external inference。Python API 的 `require_reference_environment=False` 僅產生 `software_validation` artifact；正式 lock 與 external prediction 都拒絕此 scope，CLI 不提供略過 reference gate 的開關。

## 3. 操作順序

先完成 [reference environment 安裝及實際驗證](reproducibility.md)。以下路徑是操作者提供的真實研究輸入，repo 沒有附上已完成核准的 live-study protocol。每個輸出都必須是新路徑；既有結果不會被覆寫。

### A. 訓練、匯出、鎖定

先確認 acquisition 與所有科學決策已定案，使用 clean、可識別的 source checkout 執行正式 fit。Acquisition 是 training contract 的一部分；若 fit 後再改 units、reference 或其他科學決策，就必須重新訓練相符模型。

```bash
python scripts/21_fixed_external.py fit-development \
  --protocol private/study/protocol.json \
  --out-dir outputs/fixed-external/development-models
```

從 bundle 的 `artifacts` 讀取五個新檔案的實際路徑／SHA-256，填入 protocol 的 `environment`、`primary_model`、`reference_model`、`primary_training_manifest`、`reference_training_manifest`。保留原本的 development／analysis／schema source locators。模型產生後可補齊獨立的核准／registration 記錄及 artifact locators；所有真實證據齊備後，同樣在 clean source checkout 執行：

```bash
python scripts/16_protocol.py lock \
  --protocol private/study/protocol.json \
  --out-dir outputs/fixed-external/protocol-lock

python scripts/16_protocol.py launch \
  --lock-dir outputs/fixed-external/protocol-lock \
  --cohort private/study/cohort.json
```

`launch` 與後續 inference 都檢查完整 cumulative cohort，並排除本代／祖先 development 及已 exposed subjects。收案及所有 sessions 依 #9 的規則執行，完整 enrollment ledger 在解盲前定案。

### B. 資料保管者分開 features 與 labels

新資料先經既有 shared whole-recording QA 與 feature extraction。`prepare-inputs` 由可接觸事件／correctness 的資料保管者執行，接收完整 schema 2.0 feature table；它只接受 `identity_origin=raw_metadata`。

```bash
python scripts/21_fixed_external.py prepare-inputs \
  --features private/study/external_features.csv \
  --lock-dir outputs/fixed-external/protocol-lock \
  --cohort private/study/cohort.json \
  --out-dir private/study/prepared
```

產物分為 `prepared/inputs/` 與 `prepared/labels/`。前者只有 `features.csv`、`input_manifest.json`，後者只有 `labels.csv`、`labels_manifest.json`。Labels 以原始 stable `epoch_id` 回接；不能因更正標籤而重新生成 epoch ID。

推論輸入包含 identity 欄位與 exact ordered **168 reference＋21 primary feature columns**。Identity 包含 cohort、subject、session、recording、epoch、comparison、protocol ID/version、source hash、sample interval、units、reference、channel order。Comparison 只路由至對應已擬合狀態；各 learner 實際接收的 DataFrame 只有自身的 ordered feature columns。`class_label`、`true_label`、`class_name`、`stage`、correctness 或任意額外欄位都不能出現在 inference CSV。

這是 **correct-only、offline、zero-calibration** 的流程。Feature extraction 按事前事件／correctness 政策決定可分析 trials；它不是從無標記 raw EEG 自動辨識 task，也不能改稱 causal online inference。QA／enrollment 的真實性仍由原始記錄與操作者負責。

### C. 先保存固定 predictions

```bash
python scripts/21_fixed_external.py predict-external \
  --inputs private/study/prepared/inputs \
  --lock-dir outputs/fixed-external/protocol-lock \
  --cohort private/study/cohort.json \
  --out-dir outputs/fixed-external/predictions

python scripts/21_fixed_external.py verify-predictions \
  --predictions outputs/fixed-external/predictions \
  --lock-dir outputs/fixed-external/protocol-lock
```

`predict-external` 沒有 labels 或 enrollment 參數。移走整個 labels directory 仍可產生相同 predictions。它不 fit scaler、selector、classifier、EA、ICA 或 target-specific transformation，也不做 rest calibration。

新的 inference 要驗證**當前實際 reference runtime**，其 `lock_id` 與模型 training reference lock 必須相同；host-specific `environment_id` 可以不同，但完整當次環境會保存。執行 source 必須乾淨、可識別，且 source-file fingerprint 與 protocol lock 相符。既有 archives 的驗證則依已保存證據，不要求驗證者目前位於 training host。[4]

Prediction directory 保存 label-free inputs、兩個精確模型副本、`predictions.csv` 與 `prediction_manifest.json`。每個 epoch 必須各有 primary／reference 一筆 prediction，追溯 model、protocol lock、cohort、subject、session、recording、epoch；manifest 另保存時間、environment、code、inventory hashes 與 `prediction_id`。

驗證器使用保存的模型及 feature values 重算 predictions，不只比對 hash。缺列、重複列、錯誤類別、變更大小寫的 literal ID、非二元浮點值或重新 hash 的錯誤預測都失敗。產物由暫存目錄完整驗證後原子發布；發布期間持有與 outcome ledger 相同的 OS lock，防止解盲事件搶在最後檢查與發布之間提交。

### D. 解盲後評分

```bash
python scripts/21_fixed_external.py score-external \
  --predictions outputs/fixed-external/predictions \
  --labels private/study/prepared/labels \
  --lock-dir outputs/fixed-external/protocol-lock \
  --cohort private/study/cohort.json \
  --enrollment private/study/enrollment.json \
  --out-dir outputs/fixed-external/scores

python scripts/21_fixed_external.py verify-score \
  --scores outputs/fixed-external/scores \
  --predictions outputs/fixed-external/predictions \
  --lock-dir outputs/fixed-external/protocol-lock
```

Scoring 先驗證完整 prediction bundle，再稽核 closed enrollment，將 **prediction ID、lock ID、完整 cohort bytes hash、全部 subject IDs、prediction timestamp** 綁進 outcome-access event，最後才讀 labels。請由這個 scoring 命令登記 access；舊 `record-outcome-access` 的手動事件沒有 prediction binding，不能事後補稱已授權任意 predictions。

若 labels 無效或寫檔失敗，已提交的 exposure event 保留。修復資料後可用**同一份** predictions、同一份 cohort／enrollment bytes，在新 output path 重試；不能重訓、換 predictions 或重開收案。每次修正標籤會產生新的 labels／score identity。這是可稽核的更正紀錄，不授權依分數挑標籤。Outcome access 後新的 prepare／predict 會被拒絕。

Labels 必須精確一對一覆蓋所有 predicted epochs，且 class mapping、input ID、cohort hash 一致。Scoring 獨立核對 first QA-valid session、所有合格 Rest／correct-task counts、replacement 及可評估性，禁止另外傳入選擇性 epoch mask。兩模型使用同一組 epochs；prediction 缺失是分析失敗。

## 4. 評分產物與 #12 邊界

`participant_scores.csv` 保存每人／comparison／model 的 confusion counts、epoch counts、evaluable flag、BA 與 accuracy。每人等權，summary 是可評估 participants 的平均；並保存相同 participants 的 paired primary-minus-reference BA。不同人有 20、25、30 題 task 時，仍不以 pooled epochs 改變人與人的權重。

所有 recruited participants 都在 flow 中。失訪、撤回或不滿 eligibility 的 endpoint 保留不可評估狀態，score 為空，不補成 0.5。`score_manifest.json` 保存 prediction／labels／event IDs、完整 enrollment audit、summary、paired deltas 與 artifact inventory；`verify-score` 重算所有 arithmetic 與 eligibility。

本 issue 交付 **participant point estimates**，`inference_status` 明示 uncertainty／publication integration 屬 [#12](https://github.com/PME26Elvis/eeg-ma-spectral-connectivity/issues/12)。既定的 participant bootstrap、CI、precision shortfall、multiplicity 與 research-target interpretation 仍以 [confirmatory protocol](confirmatory_protocol.md) 為準，不能把這份 score summary 當成已完成確認性報告。

## 5. Grouped CV 的 split 稽核

[`evaluation.py`](../src/eeg_ma/evaluation.py) 對有 stable epoch IDs 的 schema 2.0 rows，先以 epoch ID 排序再建立 split，記錄 `stable_epoch_id_v1`。重排 CSV rows 會改變原始檔案 hash，因此需要新的 run identity；它不改變以 epoch IDs 表示的 train／test membership。

Outer LOPO 保留一整位 subject 的全部 sessions，inner splits 只能從該 outer-training set 取資料。Split manifest 同時保存 train／test epoch IDs、subject IDs、fold identity 及 split policy。Provenance validator 會由凍結 metadata／config 重建 memberships；僅重算被竄改 manifest 的 hash 不能讓錯誤 inner／outer 邊界通過。

`grid_inner_accuracy` 明示為 **在 SFS 選擇後重用 inner folds 的 hyperparameter-selection diagnostic**；`sfs_inner_accuracy` 同樣是 selection diagnostic。兩者都不是不受選擇影響的泛化估計；外層 held-out performance 才用作該 CV 流程的測試指標。此 issue 沒有替換歷史 SFS／grid 演算法或改寫舊研究的 estimand。

沒有 stable epoch IDs 的 legacy caller 保持 `legacy_positional_v1` 與原 positional／seed 行為。新保存的 run 使用 `identity_version=evaluation-2` 並綁定 evaluation contract；舊 `evaluation-1` archives 仍可讀，已封存的 feature／result／figure bytes 不變。

## 6. 驗證證據與可重跑測試

```bash
python -m pytest tests/test_fixed_model.py tests/test_external_workflow.py \
  tests/test_external_protocol.py tests/test_split_contract.py
```

Focused tests 包含實際全 21／168 candidate schema 的 bounded SFS fit、兩個 sessions 的分組、所有 inner scaler／LDA fit rows 與 labels spies、sklearn／JSON round-trip、class mapping／schema／state/hash 攻擊、六個 CLI subcommands、獨立 enrollment 與解盲 ledger、labels permutation、feature perturbation、unequal-count participant weighting，以及 row-reorder／rehashed split 攻擊。Synthetic SFS 只限制測試的 `max_features`，正式 config 仍保留 `max_features=null`。

Reference CI 使用真正的 hash-locked runtime；compatibility CI 的 lifecycle fixture 只在 reference probe 不相符時提供明示 synthetic environment evidence，以驗證 gate 與資料流，不聲稱相容性環境已是研究 reference。兩種 CI 的 acquisition／人員核准資料都是 synthetic。

Reference CI 另執行 [`run_external_reference_smoke.py`](../tests/run_external_reference_smoke.py)，從真正 clean checkout 以 subprocess 跑 fit／lock／prepare／predict／verify，不替換 code identity、clock 或 environment verifier；模型、lock、predictions、命令輸出與 `validation_report.json` 一起保留在 `outputs/reference/fixed-external/`。這個即時檢查使用尚未完成收案的 synthetic roster，驗證 score 在讀取 labels 或記錄 exposure 前被阻擋；完整 successful scoring 由前述明示 synthetic chronology 的 lifecycle tests 驗收，不偽造即時執行中尚未經過的 acquisition 時間。

工程驗收沒有 accuracy threshold：合法的 chance-level／rank-zero LDA 仍能匯出、載入及預測。驗證的是數值與資料邊界的正確性，不以測試資料高分宣稱科學成功。

## 參考來源

1. scikit-learn, [Common pitfalls: data leakage and consistent preprocessing](https://scikit-learn.org/stable/common_pitfalls.html).
2. scikit-learn, [LeaveOneGroupOut](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.LeaveOneGroupOut.html).
3. scikit-learn 1.8, [LinearDiscriminantAnalysis](https://scikit-learn.org/1.8/modules/generated/sklearn.discriminant_analysis.LinearDiscriminantAnalysis.html)；本 repo 的 exact class-tie 規則另由 fitted pipeline round-trip 測試驗證。
4. scikit-learn, [Model persistence](https://scikit-learn.org/stable/model_persistence.html)：persisted estimator 的環境相容性及 pickle-based 格式的限制；本 repo 另外定義受限的 JSON 數值契約。
