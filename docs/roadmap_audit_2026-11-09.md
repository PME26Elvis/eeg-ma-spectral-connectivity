# 後續規劃的程式與資料契約審查

> 此頁是規劃時的版本限定稽核證據，保留當時的缺陷與重現結果。Issues #7／#8 的後續修正與使用方式見 [資料契約](data_contract.md)、[run provenance](run_provenance.md) 及 [完整流程](integrity_workflow.md)。

日期：2026-11-09（Asia/Taipei）  
檢查基準：[`df6b1394f0f68c4ce5a3c6659f09cc7fb918ecc1`](https://github.com/PME26Elvis/eeg-ma-spectral-connectivity/commit/df6b1394f0f68c4ce5a3c6659f09cc7fb918ecc1)

> 本文件是後續研究與工程規劃的證據附錄。此次只新增規劃文件；下列程式缺陷尚未修正，也沒有重新執行正式研究矩陣。

## 1. 審查結論與適用範圍

目前最應優先處理的是 **resume 的來源一致性、事件與 epoch 的唯一性，以及多 session 的資料契約**。這些工作支撐既有結果的可追溯性與新受試者驗證，並不需要增加新的特徵或分類器。

LOPO 外層確實以受試者隔離，標準化也放在訓練流程中。此次沒有找到足以直接認定既有 LOPO 結果失效的證據。需要區分「已確認的程式行為」、「研究設計限制」和「未來功能」。

本次檢查涵蓋 `src/eeg_ma/` 的 IO、epoching、BP/COH、PLV/iPLV、SFS、分類與驗證，以及 configs、主要 runners 和關鍵 tests。針對高風險邊界條件，另使用現有 synthetic fixture 在暫存目錄復現；未修改正式 data/results。

| 證據 | 本次觀察 | 可支持的結論 |
| :-- | :-- | :-- |
| 基準單元／整合測試 | **29 passed** | 現有測試在此次環境可通過 |
| 三張 canonical feature tables | 各 **600 rows、5 subjects** | 基本研究規模一致 |
| 兩種重複鍵檢查 | 三張表均 **0 duplicates** | 未觀察到下述合成重複事件案例已發生於正式表格 |
| main 最新 CI | [tests / 39532859563](https://github.com/PME26Elvis/eeg-ma-spectral-connectivity/actions/runs/39532859563) 為 success | 該 commit 的既有 CI 通過 |
| 原始 EEG 全流程重跑 | **未執行** | 不能宣稱重新驗證了真實 raw acquisition 與全部正式結果 |

此次測試環境為 Python **3.12.14**、NumPy **2.3.5**、pandas **2.2.3**、SciPy **1.17.0**、scikit-learn **1.8.0**。這是當前環境的相容性驗證，**不是原始研究執行環境的精確重建**。

三張表分別為 `data/features/features_all.csv`、`data/features/extensions/features_extensions.csv`、`data/features/final_validation/features_iplv.csv`。重複鍵檢查使用現有 `ALIGN_COLUMNS`，以及 `(subject_id, comparison, class_label, start_sample, end_sample)`。

## 2. 已復現的邊界案例

以下全部使用合成輸入或正式結果的暫存副本；沒有改動 canonical 輸出。數值描述的是程式行為，不是正式 EEG 資料品質的判定。

| 案例 | 合成操作 | 實際結果 | 風險分類 |
| :-- | :-- | :-- | :-- |
| 不相容 resume | 複製 alpha experiment 的 folds，將 C 改為 `[987.0]`，執行 `--resume` | exit 0；folds bytes 不變；snapshot 改成新 C；沒有 predictions 仍接受 | 已確認 provenance 缺陷 |
| 重複 task event | 第二個 task event 換成第一個的完整複本，事件數仍為 2 | counts pass；attempts `[1,1]`；start samples `[5000,5000]`；features 完全相同 | 已確認事件驗證缺口 |
| sample index 缺口 | 在第 100 row 後讓 sample_index 跳一格 | continuity=false；仍輸出 8 個 finite epochs | 已確認 QA 未形成阻擋條件 |
| marker status 為 false | 將一個 task event 的 marker_sent 改為 false | 該試次仍被提取 | 缺少明確處理政策 |
| 同人兩場 session | 同一 subject 放兩份合成 session | combined 16 epochs；per-subject CSV 只剩 8；features 無 session_id | 已確認輸出覆寫與 schema 限制 |

正式三張 feature tables 沒有觀察到重複 alignment keys 或重複 sample intervals；既有 raw validation 記錄亦未顯示上述 continuity／finite／marker 異常。因此，**不能將這些合成復現解讀為既有結果已經損壞**。

## 3. P0：讓 resume 驗證同一個實驗

### 證據與影響

[`12_run_final_validation.py:239–258`](https://github.com/PME26Elvis/eeg-ma-spectral-connectivity/blob/df6b1394f0f68c4ce5a3c6659f09cc7fb918ecc1/scripts/12_run_final_validation.py#L239-L258) 先寫新的 config snapshot，再以 folds 檔是否存在決定重用。

[`10_run_plv_decomposition.py:300–323`](https://github.com/PME26Elvis/eeg-ma-spectral-connectivity/blob/df6b1394f0f68c4ce5a3c6659f09cc7fb918ecc1/scripts/10_run_plv_decomposition.py#L300-L323) 有同型行為，只額外要求 intra 與 inter folds 都存在。

因此，設定或輸入改變後，舊結果可能被重新標示為新設定的產物。僅靠 folds 存在，也不能證明 predictions、SFS、summary 已完整完成。

[`evaluation.py:207–214`](https://github.com/PME26Elvis/eeg-ma-spectral-connectivity/blob/df6b1394f0f68c4ce5a3c6659f09cc7fb918ecc1/src/eeg_ma/evaluation.py#L207-L214) 依序寫入各種輸出，尚無最後才寫入的 completion manifest。

### 預期設計

- run identity 包含程式版本、effective config、輸入 hashes、feature schema 與欄位順序、cohort/epoch manifest、分析模式及環境資訊。
- 在任何 snapshot／結果寫入前驗證 identity；不相符就拒絕重用，或輸出到新 run。
- 完成的 snapshot 保持不可變；resume 不應先覆寫它。
- 先寫暫存輸出，驗證全部必備檔案後，最後建立 completion manifest。
- legacy 結果建立來源清單時，缺少的歷史資訊標為 unknown，不追填成已證實的 provenance。

### 驗收

- [ ] 相同 identity 可重用，既有檔案 hashes 不變。
- [ ] 改 C、SFS、feature 值、欄位順序、受試者或 epoch 集合時，拒絕重用。
- [ ] 缺 predictions／SFS／summary、folds 截斷或沒有 completion marker 時，不視為完成。
- [ ] 中斷後可恢復，且 partial run 不會被標示成完整研究結果。
- [ ] canonical 與新 run 路徑有明確保護規則，沒有無意覆寫。

## 4. P0／P1：把事件數量檢查提升為事件身分檢查

### 證據與影響

[`preprocess.py:73–88`](https://github.com/PME26Elvis/eeg-ma-spectral-connectivity/blob/df6b1394f0f68c4ce5a3c6659f09cc7fb918ecc1/src/eeg_ma/preprocess.py#L73-L88) 將 eligible attempts 轉成集合，再篩選事件並檢查數量；[`145–150`](https://github.com/PME26Elvis/eeg-ma-spectral-connectivity/blob/df6b1394f0f68c4ce5a3c6659f09cc7fb918ecc1/src/eeg_ma/preprocess.py#L145-L150) 不驗證事件的一對一對應。

[`qa.py:15–40`](https://github.com/PME26Elvis/eeg-ma-spectral-connectivity/blob/df6b1394f0f68c4ce5a3c6659f09cc7fb918ecc1/src/eeg_ma/qa.py#L15-L40) 的 protocol pass 也主要依 counts 判定。因此 duplicate event 與 missing trial 可以互相抵銷。

若重複片段進入 random intra CV，同一 raw interval 的複本可能分到 train/test。現有 alignment key 包含 epoch_index，不能單獨取代 raw interval uniqueness。

此外，[`io.py:63–66`](https://github.com/PME26Elvis/eeg-ma-spectral-connectivity/blob/df6b1394f0f68c4ce5a3c6659f09cc7fb918ecc1/src/eeg_ma/io.py#L63-L66) 取第一個 subject_id，未驗證各來源檔案的身分一致性；[`preprocess.py:63–64`](https://github.com/PME26Elvis/eeg-ma-spectral-connectivity/blob/df6b1394f0f68c4ce5a3c6659f09cc7fb918ecc1/src/eeg_ma/preprocess.py#L63-L64) 在沒有 SessionStart 時默認 offset 為 0。

### 驗收

- [ ] 每個 eligible trial 對應唯一 calculation event，且不存在額外、重複或缺少的事件。
- [ ] events、trials、rest manifest 與 session metadata 的 subject/session identity 一致。
- [ ] 同 session 的 epoch 原始 sample intervals 不重複；不允許的重疊明確報錯。
- [ ] SessionStart 的存在性、唯一性與 clock mapping 規則有明確版本。
- [ ] `rest_epochs.csv` 與 events 有一致性檢查。
- [ ] duplicate-event/missing-event、混入其他 session、時間倒退與 off-by-one 都有測試。
- [ ] 舊 fixture／舊資料需要的無 SessionStart 行為，限定於明確命名的 legacy policy。

## 5. P1：讓 QA 成為一致的執行條件

[`io.py:88–110`](https://github.com/PME26Elvis/eeg-ma-spectral-connectivity/blob/df6b1394f0f68c4ce5a3c6659f09cc7fb918ecc1/src/eeg_ma/io.py#L88-L110) 回傳 continuity、finite、flat channels 與 marker 狀態；[`extract.py:30–37`](https://github.com/PME26Elvis/eeg-ma-spectral-connectivity/blob/df6b1394f0f68c4ce5a3c6659f09cc7fb918ecc1/src/eeg_ma/extract.py#L30-L37) 記錄後直接繼續處理。

[`01_validate_raw.py:39–46`](https://github.com/PME26Elvis/eeg-ma-spectral-connectivity/blob/df6b1394f0f68c4ce5a3c6659f09cc7fb918ecc1/scripts/01_validate_raw.py#L39-L46) 只有 exception 導致 nonzero exit；顯示 CHECK 的資料仍可能 exit 0。PLV/iPLV 的 extraction 也尚未共用一致 QA gate。

marker_sent=false 不等於 EEG event 必然無效，因為 software event timing 仍可能可用。需要明確定義研究究竟要求 hardware marker，或允許何種 software alignment，以及如何留下證據。

### 驗收

- [ ] QA 結果區分 error、warning 與符合明確政策的允許案例，包含 reason codes。
- [ ] sample gaps／clock discontinuity 不會僅以固定 Fs 的 row index 換算悄悄通過。
- [ ] finite、flat channel、時間單調性、取樣率與 marker policy 均有定義。
- [ ] 所有 feature extractors 共用 QA 決策；validate CLI 有 error 時回 nonzero。
- [ ] 不合格資料有排除記錄與數量，不因默默丟列而讓 cohort 改變。

## 6. P1：多 session 與穩定 epoch identity

[`extract.py:14–17`](https://github.com/PME26Elvis/eeg-ma-spectral-connectivity/blob/df6b1394f0f68c4ce5a3c6659f09cc7fb918ecc1/src/eeg_ma/extract.py#L14-L17) 的 metadata 沒有 session_id；[`64–70`](https://github.com/PME26Elvis/eeg-ma-spectral-connectivity/blob/df6b1394f0f68c4ce5a3c6659f09cc7fb918ecc1/src/eeg_ma/extract.py#L64-L70) 以 subject 命名 per-session 輸出，造成重複 subject 的檔案覆寫。

相對地，[`extensions.py:290–297`](https://github.com/PME26Elvis/eeg-ma-spectral-connectivity/blob/df6b1394f0f68c4ce5a3c6659f09cc7fb918ecc1/src/eeg_ma/extensions.py#L290-L297) 與 [`final_validation.py:130–138`](https://github.com/PME26Elvis/eeg-ma-spectral-connectivity/blob/df6b1394f0f68c4ce5a3c6659f09cc7fb918ecc1/src/eeg_ma/final_validation.py#L130-L138) 直接拒絕同人第二場 session。這是下一輪跨日／重複量測研究前必須統一的邊界。

### 驗收

- [ ] 新 schema 包含 cohort_id、subject_id、session_id、recording_id/block_id 與 epoch_id；run_id 另指分析執行。
- [ ] epoch 可追溯 raw source、sample interval 與 event/trial 身分。
- [ ] 多 session 不覆寫；所有 feature family、predictions 與 exclusions 用 epoch_id 對齊。
- [ ] LOPO 最高隔離單位仍是 subject，同人的不同 session 不分到 train/test。
- [ ] cross-session 另定評估目標，不與新受試者泛化混報。
- [ ] 同人兩場、重複 attempt numbers、檔名正規化碰撞與 channel order 改變都有測試。
- [ ] legacy 的 8 raw／7 analysis channels、6 bands、168 baseline features 保留明確版本，新協定另立契約。

## 7. CV：保留已正確的邊界，補上可機器驗證的不變條件

### 已有的合理設計

[`evaluation.py:142–157`](https://github.com/PME26Elvis/eeg-ma-spectral-connectivity/blob/df6b1394f0f68c4ce5a3c6659f09cc7fb918ecc1/src/eeg_ma/evaluation.py#L142-L157) 的 outer LOPO 保留整位 subject；inner LOPO 只使用 outer training groups。

[`sfs.py:102–112`](https://github.com/PME26Elvis/eeg-ma-spectral-connectivity/blob/df6b1394f0f68c4ce5a3c6659f09cc7fb918ecc1/src/eeg_ma/sfs.py#L102-L112) 在每個 inner training fold fit scaler/LDA；[`models.py:16–60`](https://github.com/PME26Elvis/eeg-ma-spectral-connectivity/blob/df6b1394f0f68c4ce5a3c6659f09cc7fb918ecc1/src/eeg_ma/models.py#L16-L60) 以 Pipeline 包住分類器的標準化。

[`evaluation.py:252–275`](https://github.com/PME26Elvis/eeg-ma-spectral-connectivity/blob/df6b1394f0f68c4ce5a3c6659f09cc7fb918ecc1/src/eeg_ma/evaluation.py#L252-L275) 以 held-out subjects 彙整 LOPO，另列 pooled accuracy。SFS frequency 也避免因三個分類器共用 subset 而重複計數。

### 需要精確描述的限制

**Inner selection optimism。** [`evaluation.py:151–178`](https://github.com/PME26Elvis/eeg-ma-spectral-connectivity/blob/df6b1394f0f68c4ce5a3c6659f09cc7fb918ecc1/src/eeg_ma/evaluation.py#L151-L178) 先以 inner CV 在 outer training 選出 SFS subset，再重用同組 inner splits 調 SVM/KFDA。inner validation labels 已參與選特徵，因此 grid_inner_accuracy 是選擇過程的 diagnostic，不能當成獨立泛化估計。**這不等於 outer-test label leakage。**

**Intra 時間依賴。** [`evaluation.py:46–56`](https://github.com/PME26Elvis/eeg-ma-spectral-connectivity/blob/df6b1394f0f68c4ce5a3c6659f09cc7fb918ecc1/src/eeg_ma/evaluation.py#L46-L56) 使用 shuffled StratifiedKFold；[`preprocess.py:28–49`](https://github.com/PME26Elvis/eeg-ma-spectral-connectivity/blob/df6b1394f0f68c4ce5a3c6659f09cc7fb918ecc1/src/eeg_ma/preprocess.py#L28-L49) 對整段濾波；[`extensions.py:145–154`](https://github.com/PME26Elvis/eeg-ma-spectral-connectivity/blob/df6b1394f0f68c4ce5a3c6659f09cc7fb918ecc1/src/eeg_ma/extensions.py#L145-L154) 對整段窄頻濾波與 Hilbert。

因此，既有 intra 結果對應同 session 的離線 epoch 分類，不能直接等同未來時點、跨日或即時預測。LOPO 的 train/test 不共享 recording 時，也不能僅因 full-record preprocessing 就宣判跨受試者洩漏。

### 驗收

- [ ] 輸出 outer/inner split manifests，含穩定 epoch IDs 與 subject/session IDs。
- [ ] train/test subject 與受限 block/session 集合 disjoint，有直接 assertions。
- [ ] 改 held-out labels/features，不改變 training fits、SFS selections 與參數。
- [ ] 改 inner validation 值，不改變該折 scaler 的 training statistics。
- [ ] 輸入 row order 改變後，可依穩定 IDs 重建相同 split。
- [ ] inner scores 標為 selection diagnostics；如要改 joint search，另立事前固定的新協定。
- [ ] prospective cohort 不參與 feature family、頻帶或 classifier 的事後選擇。

## 8. 其他會影響研究產物的工程缺口

**Partial summary。** [`12_run_final_validation.py:273–289`](https://github.com/PME26Elvis/eeg-ma-spectral-connectivity/blob/df6b1394f0f68c4ce5a3c6659f09cc7fb918ecc1/scripts/12_run_final_validation.py#L273-L289) 只彙整本次選中的 experiments，寫回固定檔名。在暫存目錄以 `--experiments plv_alpha_only --resume` 復現，summary 由完整矩陣的 9 experiments（7 實驗＋2 references）縮成 3（alpha＋2 references）。若沿用 canonical output root，完整 summary 會被部分結果取代，其餘實驗目錄卻仍存在。應分開 partial execution 與完整 aggregate report，後者從已驗證的 run inventory 建立。

**Figure integrity。** [`13_make_final_figures.py:64–70`](https://github.com/PME26Elvis/eeg-ma-spectral-connectivity/blob/df6b1394f0f68c4ce5a3c6659f09cc7fb918ecc1/scripts/13_make_final_figures.py#L64-L70) 對缺列使用 NaN、重複列取 first；[`115–117`](https://github.com/PME26Elvis/eeg-ma-spectral-connectivity/blob/df6b1394f0f68c4ce5a3c6659f09cc7fb918ecc1/scripts/13_make_final_figures.py#L115-L117) 也未先驗證重複 reference 是否矛盾。正式圖表入口應檢查唯一 keys、完整矩陣、finite values 與相容 run IDs，並明示 error bars 是 participant SD 或指定 CI。這是防線缺口，未據此推論已提交圖表的數字錯誤。

**Environment identity。** [`pyproject.toml`](https://github.com/PME26Elvis/eeg-ma-spectral-connectivity/blob/df6b1394f0f68c4ce5a3c6659f09cc7fb918ecc1/pyproject.toml) 使用寬依賴範圍，[`tests.yml`](https://github.com/PME26Elvis/eeg-ma-spectral-connectivity/blob/df6b1394f0f68c4ce5a3c6659f09cc7fb918ecc1/.github/workflows/tests.yml) 在 Python 3.10/3.12 重新解析相容套件。既有 snapshots 缺少 code/data/environment 的完整生成身分。因此應增加固定研究環境，同時保留相容性 CI；未知的歷史環境不可用此次安裝版本回填。

驗收以 partial-run 不破壞完整 inventory、矛盾／缺漏圖表輸入明確失敗、固定環境能重建小型數值 fixture 為主。昂貴的真實資料重算不需成為每次文件 PR 的 gate。

## 9. 建議工程順序與完成界線

| 順序 | 可交付工作 | 完成後解決的問題 |
| :-- | :-- | :-- |
| 1 | immutable manifest、fingerprint resume、完整輸出檢查 | 舊結果不會被錯標成新設定產物 |
| 2 | versioned session/epoch schema、事件對應、QA gate | 新批次資料可一致匯入與追溯 |
| 3 | split manifests、CV invariants、frozen-model evaluation 契約 | 新受試者驗證有可稽核的隔離邊界 |
| 4 | 依獨立驗證協定執行、產出結果與限制 | 支持事先定義的研究結論 |

以上工作應保留原有五人探索結果與方法版本，不以新程式默默覆寫舊表格。只有確認實際影響既有結果的方法錯誤時，才建立明確命名的修正版分析，保留原結果及差異說明。

新增單元測試與合成資料可以驗證工程契約；既有 feature tables 可以做 schema／結果一致性稽核。需要真實 raw EEG 的 clock、reference、artifact 與重新提取驗證，仍須在有對應資料的受控執行環境完成。

此審查的完成條件是讓後續工作有具體證據、依賴與驗收標準；**不代表上述修正已實作，也不代表新的獨立研究已完成**。
