# 資料契約：session、epoch identity 與匯入品質檢查

本文件定義 schema **2.0** 的新資料匯入，以及明確宣告來源限制的歷史 feature adapter。它保留既有 BP、COH、PLV、iPLV 的數學方法；不增加研究矩陣、不改寫 `data/features/` 或 `results/` 中的 canonical 檔案。

## 1. 兩種來源必須區分

| 來源 | identity_origin | 原始資料品質狀態 | 可做的事 |
| :-- | :-- | :-- | :-- |
| 原始 EEG 經完整新匯入流程 | `raw_metadata` | `passed`／`passed_with_warnings` | 追溯來源檔案、事件時鐘、session 與 epoch |
| 舊 feature CSV 經明示 mapping | `legacy_manifest` | `not_revalidated` | 保留舊數值，建立跨 feature family 的一致 epoch namespace |

Legacy adapter 沒有重新看到 raw EEG，因此不能證明 acquisition units、reference、raw session IDs 或 raw hash。它不會把 feature CSV 的 hash 冒充 raw hash，也不會把新 namespace 宣稱成找回的原始 session identity。

## 2. Raw session 的必要檔案

每個 session 目錄仍包含 `eeg_raw.csv`、`events.csv`、`trials.csv`、`rest_epochs.csv`、`session_info.txt`。新匯入另外要求明確 metadata，可放在同目錄的 `session_metadata.json`；整批共用值也可由 config 的 `ingestion.metadata` 提供，但不能與 sidecar 衝突。

以下是格式範例。IDs 與 protocol 名稱必須換成實際研究的宣告值；`unknown` 是明確的未知狀態。

```json
{
  "schema_version": "2.0",
  "cohort_id": "cohort-confirmatory-01",
  "subject_id": "participant-001",
  "session_id": "session-01",
  "protocol_id": "mental-arithmetic-rest-task",
  "protocol_version": "1",
  "units": "unknown",
  "reference": "unknown",
  "channel_order": ["FP1", "FP2", "F7", "F3", "Fz", "F4", "F8", "Cz"]
}
```

`units`、`reference` 若可從 acquisition 紀錄確認，應填確切值；未知時保留 warning，不猜測或自動換算。`subject_id` 與 `session_id` 會和所有 CSV 的對應欄位及 `session_info.txt` 中已提供的身分核對。沒有明確 cohort、protocol 或 channel order 時，strict 匯入不會從資料夾名稱推測。

`recording_id` 由 cohort、subject、session 與 **實際解析的 raw CSV bytes SHA-256** 決定；`source_sha256` 是該 raw hash。若 sidecar 自行宣告 recording ID 或 raw hash，必須與實際來源相符。

所有 IDs 以 literal strings 讀取，`001`、`1`、`NA` 和 `null` 不會被 pandas 自動轉成同一個值或缺值。空白 ID、混入其他 subject/session、重複 CSV 欄名及不明 boolean 值都會拒絕。

## 3. 時鐘、marker 與 QA 政策

未新增 `ingestion` 設定時採以下 strict defaults：

```json
{
  "ingestion": {
    "marker_policy": "hardware_required",
    "clock_policy": "session_start",
    "timestamp_policy": "require_utc"
  }
}
```

這是加在完整既有 config 內的片段，不是可單獨執行的完整 config。

| 政策 | 預設 | 明示例外 | 例外留下的紀錄 |
| :-- | :-- | :-- | :-- |
| marker status | `hardware_required` | `software_timing` | policy code；未送出 marker 另列 warning |
| raw timeline origin | `session_start` | `legacy_zero` | policy code；只有缺 SessionStart 時採 elapsed origin 0 |
| UTC 對照 | `require_utc` | `elapsed_only` | policy code；明確容許缺少 UTC 對照 |

`legacy_zero` 不允許多個 SessionStart，也不允許 SessionStart phase 與 marker 1 指向不同事件。`software_timing` 不代表 marker failure 已被修好，只表示這次分析明確採用允許 software event timing 的政策。若提供了 UTC，仍會檢查格式與 clock 一致性；不會把沒有 offset 的時間默認成 UTC。

時鐘檢查包括：

- Raw sample_index 必須從 0 開始、為整數且逐筆連續。
- Raw elapsed_s 必須從 0 開始、每個差分為正，且每個 sample period 都符合 Fs；不只檢查 median。
- Session 與 config Fs 必須一致。
- Events elapsed clock 必須非遞減；選出的 epoch 不可超出 raw 範圍或互相重疊。
- UTC 與 elapsed clock 的差值需一致；trial calculation_onset、rest onset_utc 與各自 event 對齊。
- 預設 sample-period tolerance 為 `max(1e-9, 0.001/Fs)` 秒；clock tolerance 為 `max(0.005, 1/Fs)` 秒。若要改，必須明示 `sample_period_tolerance_s`／`clock_tolerance_s` 並留下 config。

以 SessionStart 對齊 raw sample 0 是明確的軟體 alignment convention，並不自動證明硬體 marker latency 或時鐘漂移已經實測校正。

## 4. 事件必須一對一，數量相同仍可能不合格

Task trial 使用 `(task_type, attempt_index)` 檢查唯一性；每個 calculation event 需有唯一對應 trial。多一個、少一個、同一事件複製兩次，都不會因總數剛好符合設定而通過。

所有 phase=`Calculation` 的事件都會核對 task_type、stage 與 configured task marker；未知 marker 不會被篩掉後忽略。所有 Rest 事件與 rest manifest rows 也必須屬於 configured rest markers。其他正常 response／fixation 等 control events 可保留。

Rest manifest 的 stage、epoch index、duration 與 onset 會逐一核對事件；SessionStart marker 與 phase 必須識別同一組事件。Raw 的非有限值、flat channel、channel order 不一致也會形成 QA error。

### 正確試次選擇與排除數量

沿用既有 `analysis_eligible=true` 且 `response_correct=true` 的選擇規則。每種 task 都輸出 total attempts、selected attempts、excluded ineligible、excluded incorrect，且四者可依下式核對：

`total = selected + excluded_ineligible + excluded_incorrect`

兩種排除理由互斥：先判定 `TRIAL_INELIGIBLE`；其餘 eligible 但答錯者才列 `INCORRECT_RESPONSE`。`excluded_trials.csv` 保存 cohort／subject／session／recording、task type、attempt index、獨立的 `trial_key`、reason code 與 trials.csv hash。未選試次不會被賦予假的 selected-epoch ID。

這些是既有 correct-only 分析規則的可稽核記錄；不是新增的 artifact rejection 或排除策略。發現 QA error 時，整批 extraction 停止，不會偷偷丟掉不合格 epochs 再繼續。

## 5. Feature schema 與精確對齊

每列保留原本十個 epoch 欄位，並加入以下 metadata：

| 欄位 | 意義 |
| :-- | :-- |
| schema_version | 目前為字串 `2.0` |
| cohort_id／subject_id／session_id | 研究批次、同一人的身分與量測場次 |
| recording_id／source_sha256 | 來源 recording identity 與 raw bytes hash |
| block_id | 該 recording 中的 protocol stage identity |
| event_key／epoch_id | 可重建的事件與 sample interval identity |
| protocol_id／protocol_version | 實驗協定宣告 |
| channel_order | JSON array，保存實際宣告與檢查過的 raw 通道順序 |
| units／reference | 明確 acquisition metadata，或明示 unknown |
| identity_origin | `raw_metadata` 或 `legacy_manifest` |
| qc_status／qc_reason_codes | 品質狀態與 JSON reason-code array |

Epoch ID 不依賴 feature-family CSV hash，因此相同 epoch 的 BP/COH、PLV、iPLV 可用同一 ID 對齊。IDs 由 session/recording scope、event metadata 與 sample interval 決定；重排 feature table rows 不會改變 identity。

Merge 前要求兩張表的 epoch ID 集合**完全相同**，且所有非特徵 metadata 一致。缺少、多出、重複、class label／sample interval／來源 metadata 不一致，都拒絕；不以 inner join 或 left join 靜默丟列。只有 onset 浮點序列化容許絕對誤差 `1e-9` 秒。

同 recording 的 raw source、units、reference、protocol 與 channel order 必須一致。Raw table 的 recording_id 也會從 raw hash 與 session identity 重算，防止 hash 改了卻保留舊 recording ID。

## 6. 多 session 與寫入行為

同一人的多場 session 可以匯入；跨 cohort 仍應使用同一個 subject_id，讓受試者層級分組有正確依據。不同 raw 目錄若聲稱同一 cohort／subject／session，會拒絕。

Per-session filenames 使用可讀 subject prefix 加完整 identity hash，避免 `a/b` 與 `a?b` 正規化後碰撞，也不再用同一 `{subject}_features.csv` 覆寫第二場。

所有 sessions 先完成共用 QA，之後才計算與發佈輸出。輸出先放 staging directory，全部完成後一起發佈；現有非空目錄與 canonical `data/features/`、`data/raw/`、`results/` 受到保護，symlink 路徑也不接受。

每個 feature output directory 包含：

- 該 family 的 combined CSV 與各 session CSV。
- `raw_validation.csv`、`excluded_trials.csv`。
- `feature_manifest.json`：schema、effective extraction config、session metadata、所有來源檔案 hashes、QA、輸出檔 hashes。
- PLV／iPLV 另保留原有簡短 family manifest CSV，方便檢視維度。

預設新輸出位置為 `outputs/features/baseline/`、`outputs/features/extensions/`、`outputs/features/final_validation/`。再次提取時選新的 output directory；不默默覆寫之前結果。

## 7. 可執行入口

在每個 raw session 放入正確 sidecar，並確認完整原始來源檔案後：

```bash
python scripts/01_validate_raw.py --raw-root data/raw --config configs/lab_replication.json --out outputs/qa/check-01.csv
python scripts/02_extract_features.py --raw-root data/raw --config configs/lab_replication.json --out-dir outputs/features/baseline-01
python scripts/08_extract_extension_features.py --raw-root data/raw --baseline-features outputs/features/baseline-01/features_all.csv --out-dir outputs/features/extensions-01
python scripts/11_extract_iplv_features.py --raw-root data/raw --baseline-features outputs/features/baseline-01/features_all.csv --out-dir outputs/features/iplv-01
```

QA error 會使 validate CLI 回傳 nonzero；即使全批都在 load 階段失敗，QA CSV 仍有 error rows。無 raw sessions 時回傳 2。Raw extensions 要求可驗證、完全匹配的 v2 baseline；不能把未驗證的 legacy mapping 自動視為匹配真實 raw identity。

## 8. Legacy feature adapter

在分析既有 canonical features 時，透過 runner 的 `--legacy-manifest configs/legacy_discovery_mapping.json` 明確啟用 adapter。該 manifest 對三張 canonical combined CSV 的 bytes hashes 做 pinning，並明示五人的 synthetic legacy session/recording namespace。輸入 bytes 改變便拒絕。

自訂 manifest 的最小格式如下；一人多場的混合 legacy table 不能用這個 single-session adapter 猜測拆分，必須先建立受控的逐列 migration。

```json
{
  "schema_version": "legacy-mapping-1",
  "subjects": {
    "participant-001": {
      "cohort_id": "discovery-legacy",
      "session_id": "declared-legacy-session",
      "recording_id": "declared-unverified-recording",
      "source_sha256": "unknown",
      "units": "unknown",
      "reference": "unknown",
      "protocol_id": "legacy-rest-task",
      "protocol_version": "unknown",
      "channel_order": "unknown"
    }
  }
}
```

可加入 `feature_inputs` 字典，以檔案描述對應 SHA-256 的方式限制允許的來源 bytes。Adapter 僅新增 metadata，不修改既有十欄、feature 數值或 canonical 檔案；mapping 與來源 feature-file hashes 都保留在載入結果的 provenance attrs，供 runner 記錄。

## 9. 程式介面與驗證界線

`schema.META_COLUMNS` 是完整 v2 metadata；`LEGACY_META_COLUMNS` 保存歷史十欄。`load_feature_table(path, legacy_manifest=...)` 讀取並驗證來源；`validate_feature_table(frame)` 檢查 schema；`assert_same_epochs(left, right)` 檢查精確對齊。

`qa.validate_session(session, cfg)` 回傳 `QAReport`，包含 typed findings、metrics、exclusions；`.raise_for_errors()` 是共同阻擋條件。`qa.preflight_sessions(raw_root, cfg)` 先檢查整批並拒絕 duplicate sessions；所有 raw feature extractors 與 behavioral 路徑共用此介面。

本次測試包含合成多 session、事件與時鐘對抗案例、三個 extraction families 的 epoch 一致性，以及三張 canonical 600-row tables 的無損 adapter。這些驗證沒有重跑真實 raw EEG，也沒有將新的合成通過結果宣稱為原始 acquisition 的再次驗證。
