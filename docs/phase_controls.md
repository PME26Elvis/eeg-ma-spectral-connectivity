# 前瞻 B：有界、相互匹配的相位指標控制

## 1. 目前完成的範圍與啟動條件

本工作提供 **B 研究設計檢查、前瞻計畫封存、輸入授權，以及 PLV／iPLV／ciPLV 的匹配數值產物**。目前真實研究設定仍是 draft；新資料、設備／reference／ground、有效 rank、可用標註、收案數與資源、核准及註冊證據必須實際完成。合成測試通過不會把研究改為 ready，也不代表已完成收案、EEG acquisition、B 模型訓練或科學驗證。

實作與設定：

| 用途 | 權威檔案 |
|---|---|
| B 的固定設計與尚待決定項目 | [phase_control_plan.json](../configs/phase_control_plan.json) |
| 計畫、證據、完整 session history、arm 與停止規則驗證 | [control_plan.py](../src/eeg_ma/control_plan.py) |
| 單一 pair 的數值公式與 undefined 狀態 | [phase_metrics.py](../src/eeg_ma/phase_metrics.py) |
| 匹配矩陣、共同 QC、LOPO manifest、輸出與驗證 | [phase_controls.py](../src/eeg_ma/phase_controls.py) |
| 前瞻授權、實際 phase tensor 與可獨立重算的完整封存 | [control_extraction.py](../src/eeg_ma/control_extraction.py) |
| 可重建的目前啟動條件檢查 | [phase_control_readiness.json](phase_control_readiness.json) |
| 操作入口 | [23_phase_controls.py](../scripts/23_phase_controls.py) |
| 既有 A primary 的獨立協定 | [confirmatory_protocol.md](confirmatory_protocol.md) |

歷史原五人的 representation exploration 在此停止。不得以這個入口繼續搜尋其頻帶、pairs、特徵、分類器或 ciPLV 門檻。新 B cohort 必須排除歷史 discovery、development 及先前已看過結果的人；穩定的 literal participant identity 由完整跨 cohort registry 核對。重新錄製或同一人另做 B，不增加獨立受試者數。

B 使用獨立的 **prospective_control_grouped_lopo** analysis ID。此工作不會啟用或改寫 A 協定中的 B flag；A 的固定模型 primary 與既有歷史數值產物維持其原本身份。

## 2. B 的研究問題與 acquisition 草案

B 比較 CopyControl 與 Type1，目的是在更匹配的視覺／反應條件下，檢查狀態辨識對相位 representation、reference 與預先定義 artifact 排除的敏感性。結果不能識別特定神經源、證明因果連結，或把 PLV 與 iPLV 的差額解釋為某個 artifact／零延遲來源的百分比。

| 項目 | v1 固定內容 |
|---|---|
| 比較與 class mapping | copy_control_vs_type1；0＝CopyControl，1＝Type1 |
| CopyControl | 抄錄畫面數字，不做心算；採相同數字輸入介面 |
| Type1 | 三位數減法、兩次借位 |
| Blocks／epochs | 共 6 blocks，每 block 10 個 epochs；每條件最多 30 個 |
| 每次評分視窗 | 5 秒，500 Hz，即 2,500 samples；只含計算／控制視窗，不含之後的反應 |
| 視覺與反應匹配 | 相同版面、亮度與 5 秒顯示時長；相同輸入介面與反應時間窗 |
| 反應時間 | 收案前依實際可行性填定一個共同值；目前未填，不由程式猜測 |
| Block 間 washout | 30 秒 |
| 條件順序 | CopyControl／Type1 交替的兩種起始序列；每兩個收案 slots 做一次平衡隨機排列 |
| 分派 | 收案前以固定 seed 產生完整 cap slots；依完整招募順序使用，不看分類表現重排 |
| Session | 每人最多一個 B session；不得以低 accuracy、答對數或數值結果挑另一場 |
| Eligibility | 兩條件均採正確 trials；每類最低 20、最高 30 個 |

若預先選擇同人 A/B 描述比較，順序固定為 A-before-B，並保留每人的全部 sessions、技術失敗、失訪、退出及 task shortfall。這個固定順序仍可能與練習、疲勞及 time-on-task 混雜；同人的兩次量測不是兩個獨立的人，也不形成隨機化的 A/B 因果比較。預設同人 A/B 分支為 disabled。

前處理設計為 continuous 2–50 Hz 四階濾波，再作 continuous 8–13 Hz 四階濾波及 Hilbert phase，最後才切 epochs。此計畫是 offline；整段訊號處理不能標成 causal streaming。**CLI extract 只接收已依該規則處理的相位張量，不負責這段 acquisition 或前處理。**

## 3. 僅允許預先決定的 3／6／9 格矩陣

每個 arm 固定比較 3 種 representations，每種固定 21 個 alpha channel pairs，分類器只規劃 LDA。下列兩項敏感度都必須在收案／結果存取前，根據 feasibility 明確選 enabled 或 disabled，並記錄理由；pending 不可 lock。

| Arm | 格數 | 啟用前提與可解釋範圍 |
|---|---:|---|
| recorded_reference | 3 | 保存實際 acquisition reference 的 baseline；必須有 reference／ground 證據 |
| seven_frontal_average | +3 | 可行性已核對後才啟用；只對指定七個額部電極作平均重參考 |
| annotation_clean | +3 | 有對結果盲化的 annotation、固定 ontology 與來源證據；排除任何與預定 artifact 標註重疊的 epoch |

只有 baseline 為 **3 格**；加任一敏感度為 **6 格**；兩者都啟用為 **9 格**。不新增「重參考 × artifact」交叉 arm，不擴充其他 band、分類器、source-space、wPLI、新特徵、校正／EA 或 streaming 搜尋。

### 同 arm 的匹配

PLV、iPLV、ciPLV 必須共用：

- 完全相同的 phase samples、epoch／recording identity 與 channel order。
- alpha 8–13 Hz、相同的 21 pairs 與相同候選順序。
- 來源 QC、ciPLV undefined 共同排除、被保留的完整 epoch／participant 身份。
- 相同 outer／inner participant folds、前處理契約、選模規則與分類器。

鎖定選模計畫為 training-only LDA forward selection，最多 21 個候選，內層 objective 為 **participant-macro balanced accuracy**，min_improvement＝0，tie 依既定 pair 順序。先保留第一個非空最佳 subset，其後必須嚴格改善才繼續；StandardScaler 只 fit 該 training partition。LDA 使用 svd、tol＝0.0001，priors 取 training class frequencies，不搜尋分類器超參數。共用的是候選空間、規則與 folds；不同 representation 在 training-only 選擇下可能留下不同 pairs。此版本尚未提供 B 模型 trainer，不能把已建立特徵矩陣描述成已完成 nested training。

### 跨 arm 的配對限制

矩陣完整性只保證所有預定格都有列出，並檢查同 arm 的匹配。若不同 arm 的 QC／ciPLV 有效性留下不同 epochs，估計對象也改變；僅因使用相同 participant IDs，不能直接宣稱是完全匹配的跨 arm 差值。

設定中的 **within_arm_exact_epochs_cross_arm_delta_requires_separate_exact_pair_validation** 要求跨 arm 差值另經明確的相同人、相同 epoch 與 baseline 限縮驗證。[result_contract.py](../src/eeg_ma/result_contract.py) 的 **paired_participant_deltas** 對不一致的人或 epochs 拒絕計算，不能先作 inner join 再假裝配對完整。完整矩陣檢查回傳 **cross_arm_paired_deltas_authorized=false**，不會自動授權差值。Artifact 排除本身會改變保留樣本，須同時呈現保留量與排除理由。

### 七額部平均不是全頭平均參考

七通道平均重參考矩陣為 \(R=I-\mathbf{1}\mathbf{1}^{T}/7\)，其 rank 為 6；因此重參考後 rank 上限為原始量測 rank 與 6 的較小值。這是線性代數上限，不是量測結果；實際有效 rank 仍須有 feasibility 證據。

這七個額部通道的空間範圍不支持「全頭平均 reference」的說法，也不會自動排除眼動／肌電。平均重參考應作用於電位訊號，再依鎖定流程取得相位；不能平均或相減已經算出的 phase／PLV 來冒充重參考。target-fit ICA、ASR、alignment 與 channel interpolation 均 disabled。未量測的 EOG、reference、units、ground、rank 或 artifact 門檻不得補造。

## 4. PLV、iPLV、ciPLV 與數值邊界

對同一 epoch、同一 pair 的樣本令

\[
\Delta_t=\phi_{a,t}-\phi_{b,t},\qquad
z=\frac{1}{T}\sum_{t=1}^{T}e^{i\Delta_t}.
\]

本專案的無方向、非負 convention 為

\[
\mathrm{PLV}=|z|,\qquad
\mathrm{iPLV}=|\operatorname{Im}z|,\qquad
\mathrm{ciPLV}=\frac{|\operatorname{Im}z|}
{\sqrt{1-(\operatorname{Re}z)^2}}.
\]

ciPLV 依 Bruña、Maestú 與 Pereda 的定義；本專案明確採 imaginary component 的絕對值。[1] iPLV 是 **先平均有正負號的 imaginary component，再取絕對值**，不是逐樣本取絕對值後平均。有限非退化輸入的三種 magnitude 都在 [0,1]。

實作以補償加總估計 mean，並以等價恆等式計算 denominator，避免接近 0／π 時的相減消去：

\[
d^2=
\frac{1}{T}\sum_t\sin^2\Delta_t+
\frac{1}{T}\sum_t\left(\cos\Delta_t-\overline{\cos\Delta}\right)^2
=1-\left(\overline{\cos\Delta}\right)^2.
\]

| 狀態／規則 | 明確行為 |
|---|---|
| \(d>10^{-6}\) | 依公式計算 ciPLV；記錄 defined |
| \(d\le10^{-6}\) | ciPLV＝null／None，記錄 undefined 與原因 |
| 整段固定 0 或 π lag | 以 float64 precision 記錄 real-unit-mean 退化；不補 0 |
| near-degenerate lag | 記錄 denominator at or below prespecified minimum；不加 epsilon |
| 混合 0／π samples | 依整段 mean 判定；不因單一樣本位於實軸就自動判 undefined |
| nonfinite／不合法 shape | 拒絕整個輸入；不靜默刪 sample 或 impute |
| 純浮點邊界超出 | 僅容許固定 64 machine-epsilon 範圍內的 roundoff clipping，並逐次記錄；更大違反直接失敗 |

**\(10^{-6}\) 是本研究固定的數值有效性政策，不是 artifact 門檻，也不是論文提供的最佳生理 cutoff。** 它不可依分類表現調整。Denominator 沒有加 epsilon；undefined 不會被寫成 0、0.5、NaN 或無來源的「修正值」。

每個通過來源 QC 的 epoch 都保留 21 個 pair 的 mean real／imaginary、denominator、樣本數、threshold、defined／undefined、reason 及 roundoff 紀錄。任一 pair 的 ciPLV undefined，該 epoch 對三種 representations 一起排除，即使 PLV／iPLV 本身仍可算。

例如固定非零且非 π 的 lag，PLV＝1、iPLV＝\(|\sin\Delta|\)、ciPLV＝1；0 與 π 的退化另按上述政策處理。[1] 零延遲不敏感的論證依賴訊號混合假設；這些指標不能保證每個非零 lag 都是神經作用，也不提供 artifact 來源的定量分解。

## 5. 精確輸入 schemas 與公共 API

以下是版本化欄位契約；所有列出的欄位均須存在，不接受額外欄位。完整可執行合成範例位於 [phase_fixture](../tests/test_phase_controls.py)。例子中的身份與 SHA-256 僅屬明確標記的 synthetic fixture，不可移作真人資料或核准證據。

### 5.1 Phase tensor 與 epoch manifest

Phase tensor 為 real numeric radians，shape **[epochs, samples, 7]**，所有值有限。固定 channel order 為 **FP1、FP2、F7、F3、Fz、F4、F8**；兩兩組合依此順序產生 21 pairs。真實 B 每 epoch 固定 2,500 samples；合成工程 fixture 使用 64 samples，不聲稱是生理 alpha EEG。

Epoch manifest 的頂層只有：

    schema_version = "eeg-matched-phase-epochs-1"
    epochs = ordered list of epoch records

每筆 epoch record 的精確欄位：

| 欄位 | 型別／限制 |
|---|---|
| epoch_id | 唯一 literal string，順序恰好對應 tensor 第一軸 |
| cohort_id | literal string；一個 bundle 只能有一個 cohort |
| subject_id | 穩定 literal string；不轉數字、不去除前導零 |
| session_id | literal string；真實 B 必須匹配完整 history 所選 session |
| recording_id | literal string |
| recording_sha256 | 64 個小寫十六進位字元；相同 recording 身份不得指向不同 digest |
| start_sample、end_sample | literal integers；start ≥0；end−start 等於樣本軸長度 |
| comparison | literal string；真實 B 固定 copy_control_vs_type1 |
| class_label | literal integer 0 或 1；bool 不視為合法 label |
| qc_passed | literal boolean |
| qc_reason_codes | 不重複的 literal strings；pass 時為空，fail 時至少一個原因 |

Literal IDs 使用下列規則，不含空白。重複 epoch_id、同一 source sample span 換 ID 重複使用、混合 cohort、來源錄製 hash 衝突，都會被拒絕。

    [A-Za-z0-9][A-Za-z0-9_.:-]*

### 5.2 Nested participant LOPO manifest

Split manifest 的頂層只有：

    schema_version = "eeg-matched-phase-splits-1"
    splits = list of outer split records

Outer record 精確欄位為 **split_id、train_subject_ids、test_subject_ids、inner_splits**；inner record 精確欄位為 **split_id、train_subject_ids、validation_subject_ids**。所有 subject lists 必須是唯一、非空的 literal IDs。

每個 outer test 留出一人，outer training 為其他全部人；整體 outer test 恰好覆蓋每人一次。每個 inner validation 留出一名 outer-training participant，其餘 outer-training participants 為 inner training；所有 inner validations 恰好覆蓋該 outer training set 一次。Outer test person 不得出現在任何對應 inner partition。至少需要三個 participants 才能形成這個數值契約；真實研究的 N 仍由獨立前瞻 target／cap 決定，三人合成 fixture 不是樣本數建議。

QC 後只將保留的 epoch IDs 對應回既有 splits，不重新產生 folds，也不默默刪除受試者。缺類、完全無有效 epochs、或任一人每類少於預定最低數，均保留明確 findings，並令 bundle.evaluable＝false。

### 5.3 Input policy

Policy 的精確欄位如下：

| 欄位 | 固定值／內容 |
|---|---|
| schema_version | eeg-matched-phase-policy-1 |
| analysis_id | prospective_control_grouped_lopo |
| evidence_kind | synthetic_fixture 或 prospective_data |
| protocol_lock_id | synthetic 可為 null；prospective 必須為已驗證的 B lock ID |
| code_sha256、environment_sha256 | 實際程式與環境的 canonical identity digests |
| preprocessing_sha256 | 真實 B 綁定 arm contract、acquisition／reference、前處理與標註決策 |
| selection_sha256 | 鎖定 selection contract 的 digest |
| epoch_manifest_sha256、split_manifest_sha256 | 明確提供之完整 ordered manifests 的 canonical digests |
| channels | 上述固定七通道順序 |
| band_hz | [8.0, 13.0] |
| ciplv_min_denominator | 0.000001 |
| undefined_ciplv_policy | exclude_epoch_from_all_representations |
| min_epochs_per_class | synthetic_fixture＝1；prospective_data＝20 |
| max_epochs_per_class | 30 |

真實 policy 由 **authorize_phase_inputs** 從已驗證 B lock 建立，不應手填一組看似有效的 hashes 繞過 gate。來源每人每類超過 30 個 epochs 直接拒絕；共同 QC 後低於最低數時，保留數值／排除證據並標為不可評估，不能靠刪人或補 chance performance 修復。

### 5.4 公共函式與輸出

    from eeg_ma.phase_controls import (
        canonical_sha256,
        validate_matched_phase_manifests,
        extract_matched_alpha_features,
        export_matched_phase_bundle,
        verify_matched_phase_bundle,
    )

    audit = validate_matched_phase_manifests(
        epochs, splits, input_policy=policy, n_samples=n_samples
    )
    bundle = extract_matched_alpha_features(
        phases, epochs, splits, input_policy=policy
    )
    bundle.require_evaluable()
    export_matched_phase_bundle(bundle, output_directory)
    restored = verify_matched_phase_bundle(output_directory)

低階 extractor 是純數值元件；上述程式片段本身不驗證真人研究許可。真實資料應走後述 prospective CLI。不可評估的 bundle 仍可先 export 保存稽核；**require_evaluable** 是執行分析前的阻擋，不要求刪除這些紀錄。

Bundle 提供 **plv、iplv、ciplv、matrix(name)、epoch_ids、manifest、evaluable、require_evaluable()**。每個 matrix 為相同 N×21，資料儲存在不可寫的 bytes backing；每次取得 manifest 會得到獨立複本。特徵欄位格式為 **METRIC__channel_a__channel_b__alpha**，METRIC 分別取 PLV、IPLV、CIPLV，pairs 依固定通道組合順序排列。

數值 bundle 目錄精確包含：

    manifest.json
    plv.npy
    iplv.npy
    ciplv.npy

manifest.json 的外層為 **eeg-matched-phase-export-1**，包含 bundle、files 與 export_sha256；內層 bundle schema 為 **eeg-matched-phase-bundle-1**。內容保留 source phases 的 float64 bytes／shape hash、完整 epoch／split manifests、共同 QC、accepted identities、逐 pair 數值 audit、未評估原因、三個矩陣 hashes、實際執行 Python／NumPy 版本與數值實作來源 hashes。

Export 使用新目錄 transaction，拒絕覆寫並保護歷史 results、raw 與 features 來源樹。Verify 檢查 inventory、hash、shape／bounds、公式與 status、一致的共同排除、欄位順序及 frozen split resolution；這是內部一致性驗證，不是原始 acquisition 的獨立鑑定。

## 6. 真實 B 所需的外部證據與限制

### 完整 session history

**eeg-control-session-history-1** 頂層精確欄位為 **schema_version、cohort_id、closed、participants、stop_reason、stop_reason_detail**。closed 必須為 true，participants 必須涵蓋所有已招募者，依實際招募時間排序，不能只提供有好結果的人。

每人精確欄位為 **subject_id、recruited_at、disposition、sessions**。Disposition 可為 completed、technical_failure、withdrawn、lost 或 task_shortfall。每個 session 的精確欄位為：

    session_id, protocol, started_at, qa_passed, qc_reason_codes,
    sequence_index, eligible_epochs_per_class, recording_id,
    recording_sha256, qa_evidence_sha256, eligibility_manifest_sha256

Protocol 為 B，或事先啟用 paired 分支時的 A；eligible_epochs_per_class 精確含字串鍵 "0" 與 "1"，值為 0–30 的 integers。B 的 sequence_index 必須符合該收案 slot 的鎖定分派。整段 recording QA fail 時有效數必須為零。Eligibility identity 由 **session_epoch_identity** 綁定原始正確 trial 身份，不因敏感度 arm 的後續 QC flags 而重定義原始收案資料。

招募與 session 時間須在 lock 之後、實際 audit time 之前；同人的已規劃 A 必須先於 B。停止理由固定分為 target_reached、cap_reached 或 administrative_shortfall；行政提前終止必須有實際原因並保留全體人數，不容許因分數低而延長收案。Target 指共同 phase QC 前已通過 B QA 且達到最低正確 epoch 數的人；cap 按全體招募的唯一人計數。共同 phase QC 後的損失仍須另報。

### Annotation manifest

只在已啟用 annotation_clean 時提供 **eeg-control-annotation-1**，其精確欄位為 **schema_version、ontology、blinded_to_outcomes、excluded_epoch_ids、evidence_sha256**。Ontology 必須和 lock 相同，blinded_to_outcomes 必須為 true，excluded_epoch_ids 必須都存在於完整 epoch manifest。被標註排除者須明確具有 **BLINDED_ARTIFACT_ANNOTATION** reason，且 qc_passed＝false；三種 representations 共用此決定。

### Gate 能驗證與不能驗證的事

計畫 lock 綁定實際 feasibility、data-use approval、participant registry 檔案的 bytes，完整固定矩陣、平衡分派、程式來源與已驗證 reference environment。授權再對照 cohort、session、recording、correct-epoch identity、class counts、完整 roster、arm 決策與 annotation 身份；不得使用停留在 draft 的計畫或結果已存取後才補的 lock。

Hashes 能檢查檔案一致性，不能認證證詞的真實性。Phase samples 本身不能證明設備型號、units、reference／ground、正確試次、盲化標註或實際 event timing；資料提供者必須保留對應 raw recording、QA 與 eligibility 證據。不得把 synthetic fixture 的任意 source identifiers 或測試 hashes 當成真人 acquisition lineage。

## 7. CLI：檢查、工程驗證與未來真實輸入

以下命令均從 repository root 執行，並使用 [reference environment 定義](../environment/reference.json) 已驗證的 interpreter、安裝 receipt 與執行緒設定。所有輸出必須使用尚不存在的新目錄。正式 freeze／prospective authorization 另要求 clean committed code 與鎖定來源一致；synthetic mode 會如實記錄工作中的程式來源，不冒充正式 lock。

### 7.1 檢查目前 draft

    python scripts/23_phase_controls.py check \
      --plan configs/phase_control_plan.json \
      --out-dir outputs/phase-controls/readiness-v1

目前應回傳 **exit code 2** 與 blocked findings。這是預期 gate 行為，不能略過後改標 ready。Check 不建立 control_plan_lock。

### 7.2 可執行的 synthetic smoke

    python tests/run_phase_controls_smoke.py \
      --out-dir outputs/reference-local/phase-controls

此 smoke 以 seed 固定的 12 epochs、3 個合成 participant IDs、7 通道相位亂數執行實際 subprocess CLI；64-sample fixture 用於數值工程檢查，不模擬或聲稱已取得真實 8–13 Hz EEG。它會：

- 驗證當前 interpreter 符合 reference environment，保留真實 code／environment 身份。
- 確認真實 draft check 仍回傳 blocked／exit 2。
- 執行 extract 與 verify-features，取得三個 12×21 矩陣。
- 以獨立的直接 complex-exponential 公式比對普通非退化 fixture，避免直接拿同一數值函式當 oracle。
- 檢查共同 epoch／split／provenance hashes、不可寫 array／metadata，並實際確認再次輸出被拒絕且原檔 bytes 未變。

產物包含 inputs、features、logs、draft-readiness、reference_environment.json 與 validation_report.json。報告不含模型 accuracy、科學效果或偽造核准。

可使用 smoke 建立的合成 inputs，另作一次獨立數值輸出：

    python scripts/23_phase_controls.py extract \
      --mode synthetic-fixture \
      --phases outputs/reference-local/phase-controls/inputs/phases.npy \
      --epochs outputs/reference-local/phase-controls/inputs/epochs.json \
      --splits outputs/reference-local/phase-controls/inputs/splits.json \
      --out-dir outputs/phase-controls/synthetic-review-v1

    python scripts/23_phase_controls.py verify-features \
      --bundle outputs/phase-controls/synthetic-review-v1

Synthetic mode 不接受真實 lock、participant history、annotations 或敏感度 arm。把真實資料換個檔名送進 synthetic mode 不會獲得 prospective 科學分析資格。

### 7.3 完成真實證據後的命令形式

下列 inputs／輸出名稱只是操作位置；本文不附真實核准、註冊、人員名單或收案資料。計畫尚未 ready 時不能完成 freeze。

    python scripts/23_phase_controls.py freeze-plan \
      --plan inputs/reviewed_B_plan.json \
      --out-dir outputs/phase-controls/B-lock-v1

    python scripts/23_phase_controls.py verify-plan \
      --lock-dir outputs/phase-controls/B-lock-v1

    python scripts/23_phase_controls.py authorize-inputs \
      --lock-dir outputs/phase-controls/B-lock-v1 \
      --epochs inputs/B-epochs.json \
      --splits inputs/B-splits.json \
      --session-history inputs/B-session-history.json \
      --arm recorded_reference \
      --out-dir outputs/phase-controls/B-authorization-v1

    python scripts/23_phase_controls.py extract \
      --mode prospective \
      --lock-dir outputs/phase-controls/B-lock-v1 \
      --phases inputs/B-alpha-phases.npy \
      --epochs inputs/B-epochs.json \
      --splits inputs/B-splits.json \
      --session-history inputs/B-session-history.json \
      --arm recorded_reference \
      --out-dir outputs/phase-controls/B-recorded-v1

    python scripts/23_phase_controls.py verify-features \
      --bundle outputs/phase-controls/B-recorded-v1

只有 lock 已列出的 arm 可使用。Seven-frontal-average arm 必須提供已依該重參考規則取得的 phase tensor；annotation_clean 必須另外提供 **--annotations inputs/B-annotations.json**。CLI 不會替操作者完成重參考、標註、EEG acquisition 或模型訓練。

Prospective extract 以新的外層 archive 一併保存 authorization、session history、已封存 B plan、實際輸入的 **phases.npy**、兩個數值模組的 source snapshots，以及內層 **features/** 數值 bundle；相較之下，synthetic extract 直接輸出四檔的數值 bundle。Prospective verify 從保存的 phase tensor 重算三組 features，核對外層證據與數值，並將原 runtime／程式觀察保留為歷史身分，無須讀取者使用相同環境或執行封存程式。浮點比較僅允許明列的 `64 × float64 epsilon` 絕對 roundoff；identities、counts、排除與 partitions 仍須完全一致。

若只驗證 features/，不能宣稱已驗證完整前瞻授權。Archive 的 phase tensor 也不會重建或替代 raw recording；原始錄製檔、正確試次／event timing 及 QA evidence 仍須按各自來源 hash 保存。這個 archive 沒有 B model fitting 或 classification performance 結果。

## 8. 報告單位、不確定性與停止規則

| 分析 | 對象與訓練邊界 | 本專案允許的報告 |
|---|---|---|
| A fixed-model external primary | 已在 development 固定的模型；新參與者只作外部預測 | 依 A 已鎖定方法作 participant-level 指標與 CI；不混入 B |
| B prospective grouped nested LOPO | 每個 outer fold 只以其他 B 參與者重新選擇／訓練；所有 sessions 以人分組 | 每人 balanced accuracy、accuracy、recall_control、recall_task、confusion counts；等人權重描述平均與 SD |
| Synthetic phase smoke | 相位數值／檔案契約 | 數值誤差、身份／hash、gate、immutability；沒有科學表現估計 |

B 的 outer training sets 重疊，fold 結果不是可直接當成 IID replicates 的樣本。v1 不把 across-fold SD 當 SE，也不對這些分數套用 A 的固定模型 IID participant bootstrap 並稱為有效 CV confidence interval。交叉驗證變異數具有資料重用造成的相關性；其一般估計限制由原始研究討論。[2]

報告必須完整列出全部預定 cells，包含負向、零差異、不支持假設與不可評估結果；每格保留 analysis／run／model／lock IDs、完整 recruited participant roster、epoch／split／preprocessing／selection 及 predictions source hashes。缺類或不足最低數保留原因、招募與 QA denominators，不填 chance score、不刪列美化。**validate_control_result_matrix** 需要獨立驗證的 lock ID 與完整 roster；它只驗證矩陣與來源契約，不能取代預測紀錄的數值檢查或授權跨 arm 差值。

原五人不再探索。新 B 依預先 target／cap／行政停止規則結束，完成已鎖定矩陣後停止；不得因某格較好就擴展另一頻帶、分類器、QC 門檻、參考方式或 participant 子集。新的假說需要新的研究設計及未暴露 holdout。

## 9. 可重查的測試與證據

[test_phase_metrics.py](../tests/test_phase_metrics.py) 覆蓋 0／π、正負非零 lag、已知混合角度、imaginary cancellation、near-degenerate／threshold 邊界、bounds／symmetry、nonfinite 與 shape 拒絕。[test_phase_controls.py](../tests/test_phase_controls.py) 覆蓋共同排除、身份／source span、LOPO leakage／coverage、post-QC 每人最低量、完整空保留 artifact、immutability、hash 與數值 audit 篡改檢查。

[test_control_plan.py](../tests/test_control_plan.py) 檢查真實 draft 仍阻擋、矩陣與排程範圍、前瞻證據、session／participant 身份與停止規則；[run_phase_controls_smoke.py](../tests/run_phase_controls_smoke.py) 則執行實際 reference-runtime CLI。這些是工程可重查證據，未取代新 B cohort、實際 acquisition feasibility 或未來 B model-training 驗證。

[test_control_extraction.py](../tests/test_control_extraction.py) 另從 synthetic prospective-context archive 重算實際 phase tensor，拒絕內容自洽但來自錯誤 tensor 的重新 hash 產物，並測試 source 變動時的原子取消、完整不可評估 roster、annotation 排除與跨環境封存驗證。

### 一手來源

1. Bruña R, Maestú F, Pereda E. **Phase locking value revisited: teaching new tricks to an old dog.** Journal of Neural Engineering. 2018;15(5):056011. [DOI: 10.1088/1741-2552/aacfe4](https://doi.org/10.1088/1741-2552/aacfe4)。[作者公開稿，section II.C、equations 13–14](https://arxiv.org/abs/1710.08037)。
2. Bengio Y, Grandvalet Y. **No Unbiased Estimator of the Variance of K-Fold Cross-Validation.** Journal of Machine Learning Research. 2004;5:1089–1105. [期刊原文與 PDF](https://jmlr.org/papers/v5/grandvalet04a.html)。
