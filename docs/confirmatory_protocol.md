# 獨立受試者驗證協定 v1：設計已定義，正式收案仍受 gate 管理

**版本 1.1；2026-11-09；狀態 DRAFT／尚未註冊。** #9 完成可執行的協定、樣本數證據、收案稽核與鎖定／修訂機制；#10 加入 verified reference environment 與 training lock identity 的必要契約，研究終點、N／cap 與數值方法不變。實際設備驗證、研究核准、固定模型與其 training environment 證據尚未齊備，不能把文件合併稱為已鎖定、已註冊或已開始收案。

機器可讀的唯一決策來源是 [`confirmatory_protocol.json`](../configs/confirmatory_protocol.json)；操作見 [protocol gate](protocol_gate.md)，數值證據見 [precision report](precision/README.md)。歷史方法與結果仍分別以 [`method_definition.md`](method_definition.md)、[`final_results_summary.md`](final_results_summary.md) 為準。

已保存目前真實 draft 的 [readiness 檢查](protocol_readiness.json)，包含 protocol／validator hashes、已存在檔案的 hash 檢查及未齊備欄位。它是 blocked 狀態的可重查證據，不是正式 study lock。

## 1. 問題、估計對象與研究邊界

主要問題：**只用歷史 development participants 訓練後固定的 α-PLV＋SFS＋LDA，對獨立新受試者在 A 流程下的 Rest1／Type1 分類表現如何？** Primary 是符合預定 QA、完成可分析資料的新受試者之平均 balanced accuracy；它以正確心算作答為條件，並不估計所有招募者或所有嘗試的表現。

歷史五人的 74.1% 是三個分類器的描述平均，不能當成這個固定 LDA 的預期表現或新 cohort 成功保證。原五人的 representation exploration 已停止，本工作不新增其頻帶、特徵或模型搜尋。有限資料上的方法選擇與 nested CV 是不同層次的問題，後者不會消除跨實驗選擇偏差。[1]

| 分析 | 資料與訓練邊界 | 可估計的對象 |
|---|---|---|
| 固定模型外部測試 | development-only fit 一次；新 cohort 只 predict | 這兩個固定 fitted models 對 QA-evaluable 新人的表現；A primary |
| 擴充樣本 nested LOPO | 合併可用 development 資料，每折只在 training subjects 選特徵與 fit | 在不同 training size 下重新學習的流程；獨立、描述性 secondary |
| calibration／alignment | 另定可存取的新受試者校正資料 | 具有該校正條件的表現；不併入 zero-calibration primary |

所有 cohort／session 共用穩定、literal `subject_id`。重新錄製或另做 B 不會創造新的人；development／test 以人排除交集。若擴充 development 再 fit，需新模型、新 freeze，以及完全未見的 holdout。

## 2. A 與 B 的角色、順序及權重

| 決策 | A：`eeg-ma-A-capped-v1` | B：`eeg-ma-B-control-v1` |
|---|---|---|
| 角色 | 唯一 primary，啟用 | 下一個 study，`enabled=false` |
| 主要流程 | Rest1 → Type1 → Rest2 → Type2 | 草案為 6 blocks、每 block 10 epochs、30 秒 washout |
| 任務 | Type1：三位數、兩次借位；Type2：兩位數、一次借位；mini-chain 4 | arithmetic 與非心算 copy control 匹配顯示、數字輸入與反應時間 |
| 順序 | 固定歷史任務順序；A 先於同人的 B | 收案前平衡隨機分派 RestControl／Type1 起始、交替的兩種序列 |
| session 選取 | 第一個通過結構 QA 的 A session；只允許指定技術 replacement | 必須另完成 feasibility 與新協定 lock 才能收集 |
| primary 權重 | 每人 1 | 0；同人的 B 對獨立 N 增量為 0 |

A 保留每段 Rest 150 秒／30 個 5 秒 epochs，每個 task 目標 30 題正確，但增加明確嘗試與時間上限。因此稱為 **歷史任務對齊、具有上限的驗證**，不稱完全相同條件 replication。設備／reference 等對齊尚須實際核對。

只做 instructions／response interface 說明，不加入可評分的心算練習。A 仍含 fixed order、time-on-task、視覺與 block 差異；結果只支持該 acquisition protocol 的狀態辨識。跨人測試不會自行排除這些混淆。[2] B 的完整 acquisition feasibility 尚未完成，不得在 A 不理想後啟動 B、改稱 primary 或挑 A／B 較好者。

## 3. 固定模型與 ordered feature contract

| 環節 | 前瞻決策 |
|---|---|
| Primary | A Rest1 vs Type1；α-PLV 21 pairs → development-only SFS → StandardScaler＋LDA |
| Reference | 同一 development cohort 的 42 BP＋126 COH → 獨立 development-only SFS → StandardScaler＋LDA |
| 通道 | raw：FP1、FP2、F7、F3、Fz、F4、F8、Cz；analysis 排除 Cz，保留前 7 個 |
| 時間與前處理 | 500 Hz、5 秒；四階 Butterworth 2–50 Hz、`sosfiltfilt`；無 notch |
| 相位特徵 | continuous signal 再經四階 8–13 Hz `sosfiltfilt`、Hilbert phase，21 組 PLV |
| Selector | development-only LOGO；inner accuracy；`max_features=null`、`min_improvement=0`，沿用 `1e-15` 改善容差與索引 tie-breaking |
| Scaler | SFS inner fold 只 fit 該 training split；最終 subset 只在完整 development fit |
| LDA | `solver=svd`、`shrinkage=None`、`priors=None`、`tol=1e-4`；完整 fitted 參數、類別與版本入 manifest |
| Test 輸入 | frozen feature order、selected indices、class mapping；不得 target-fit scaler、SFS、classifier、EA、ICA 或 ASR |

選 LDA 是減少可選流程，不宣稱它是舊資料的最佳分類器。不得用 LOPO 選取頻率人工拼出最佳 pairs。Primary 的 BA reporting 不改變 SFS 的 inner accuracy objective。

實際 feature order 已封存在 [`confirmatory_feature_schema.json`](../configs/confirmatory_feature_schema.json)。development manifest 綁定目前檔案的 SHA-256、既有 subject IDs 及 legacy mapping；未知 historical raw lineage 保留 `unknown`，不補造 acquisition provenance。

本研究為 **correct-only、offline、zero-calibration**。Continuous filtering／Hilbert 使用整段 recording，不能稱 causal online inference。不得把 class、stage、檔名、correctness 或其他 label metadata 餵入模型；事件／correctness 只供預定 eligibility 與評分。

## 4. Acquisition 與數值 QA

正式 lock 要求有實際 device、signal units、reference、ground、channel order、Fs、clock／hardware marker 驗證證據；不得由通道名或歷史 CSV 猜測。核准人、核准時間、研究倫理狀態與資源承諾也須真實填入。

| 項目 | v1 決策 |
|---|---|
| Sample continuity | index 從 0 連續；sample gaps 容許 0 |
| Finite signal | nonfinite 值容許 0；8 raw channels 必須全數存在且同序 |
| Flat channel | raw-channel SD ≤ `1e-15` 判為 flat；只是既有數值 sanity check，不是生理 artifact 門檻 |
| Sampling rate | 500 Hz，差異容差 `1e-6` Hz |
| Raw sample period | 0.002 秒，絕對容差 `2e-6` 秒 |
| Event／UTC alignment | SessionStart elapsed origin；UTC 必填；clock tolerance 0.005 秒 |
| Hardware marker | `hardware_required`，不得靜默改用 software timing |
| Structural failure | 整個 recording 拒絕，不做事後 segment salvage |
| Amplitude／artifact 操作 | amplitude rejection、target-fit ICA／ASR、channel interpolation 均 disabled |

不憑沒有 units／設備依據的數字添加振幅門檻。MNE 的範例是方法說明，並不替本設備選出已驗證的 threshold。[3] 這個決定保留歷史流程對齊，代價是不能宣稱已去除眼動、肌電或其他生理 artifact；量測控制另屬 B／後續研究。

[`confirmatory_analysis.json`](../configs/confirmatory_analysis.json) 保留歷史數值方法，但將通用 raw QA 的 `expected_epochs_per_class` 設為 `null`，讓合法的 20–29 題任務結果能通過結構檢查。**這不取消 cohort policy**：正式流程必須依本文件第 5 節稽核數量與停止原因，再進入固定模型分析。歷史 `lab_replication.json` 不變。

## 5. Eligibility、任務上限與 replacement

| 決策 | 固定值／規則 |
|---|---|
| 每 task 目標 | 30 題正確；每次計算 5 秒 |
| 每 task 最多嘗試 | 45 次；無論答對率，不得超過 |
| Session 最長 | 1,800 秒，從正式開始至結束，排除設備準備 |
| Primary 最低有效量 | Type1 ≥20 題，且 Rest1 完整 150 秒／30 epochs；上限每類 30 |
| Correct-only | 通過整段 QA 後，所有正式正確 trials 都須 `analysis_eligible=true`；禁止選擇性 epoch mask |
| Partial task | 20–29 題只在達到 45 次嘗試或 1,800 秒上限時保留；不能任意提早停在高分片段 |
| Replacement | 最多 1 次，前次開始起 7 天內；只因已列舉的技術 QA failure |
| Session 選取 | 第一個通過結構 QA 的 A session；答對數不足也不能跳過它換一場 |
| Missing endpoint | 缺類／不足最低量 ⇒ 該 endpoint 不可評估；留在 flow，不能補 0.5 |

Minimum 20 不授權把 Rest1 改為 100 秒。因未啟用逐 epoch artifact rejection，通過結構 QA 的 acquired rest epochs 必須全部 eligible。Task 的 raw correct 數與 eligible correct 數亦須相等。

可 replacement 的 error codes 僅為 `DEVICE_DISCONNECT`、`SAMPLE_INDEX`、`SAMPLE_CLOCK`、`NONFINITE_SIGNAL`、`FLAT_CHANNEL`、`TASK_EVENT_TRIAL_SET`、`SESSION_START`、`MARKER_NOT_SENT`。其他 invalid recording 仍可排除，但不能自行取得重錄資格。QA 已通過、答對數不足、模型 accuracy 低或撤回後均不得重錄。Technical replacement 與原 session 同一 subject，不新增 N。

Type2 **未完成或答對數不足**，在 recording 仍通過整段結構 QA 時不會刪掉有效的 Type1 primary。若 Type2 時段出現 sample gap／nonfinite／clock 等結構錯誤，仍拒絕整個 recording；v1 不支援事後挽救前半段。兩模型必須共用同一 eligible epoch manifest，prediction 缺失或 artifact 不一致是分析失敗，不能包成 subject QC exclusion。

## 6. Endpoint、bootstrap 與判讀門檻

對可評估受試者 i：

```text
BA_i = (Recall_Rest_i + Recall_MA_i) / 2
Primary = mean_i(BA_i)
Delta_i = BA_i(primary fixed model) - BA_i(reference fixed model)
Key secondary = mean_i(Delta_i)
```

人等權；epochs／sessions 不作獨立 N。Primary 只一個，為 A Rest1 vs Type1 的 participant-macro BA。Paired Δ、Type2、expanded nested LOPO 均描述性 secondary；其 nominal CI 不作 multiplicity-adjusted 確認性宣稱。Reference 要在新 cohort 產生固定模型預測，不能用歷史 LOPO 平均替代。

CI 固定為 **participant percentile bootstrap、10,000 resamples、seed 20261109、兩側 95%**。先算每人的 BA，再抽人；兩模型用完全相同的 participant indices 產生 paired Δ。Percentile 不是對所有分布都優良的通用選擇，因此本設計以明確 simulation gate 檢查適用情境，不能只因套件提供選項便宣稱充分。[4]

| 判讀 | 事前固定規則 |
|---|---|
| Above-chance 統計支持 | confirmatory mode、達成 N、非退化 CI，且 primary CI 下界 **>0.50** |
| 絕對研究目標 | 門檻 **BA=0.65**；下界 >0.65：supported；上界 <0.65：not_met；其餘 uncertain |
| Paired 增益研究目標 | 門檻 **Δ=0.05**；同樣比較 CI 上下界，但一律是描述性判讀 |
| 精度 | primary CI 半寬 ≤0.08、N≥60、非退化；獨立於 above-chance 判讀 |
| Degenerate | 分數或 CI spread ≤`1e-12`，不做確認性／精度充分宣稱 |
| Pilot 或 cap 導致 N 不足 | CI／interval position 可描述，`inference_scope` 明列 pilot／target shortfall，不冒稱完成 primary confirmation |

等於門檻仍為 uncertain。0.65 與 0.05 是前瞻的研究價值目標，未經真實使用者／BCI 決策成本驗證，不能稱實用性標準。Above-chance、絕對目標、相對增益與精度是四個獨立欄位，沒有「任一通過就成功」的總開關。

CI 只描述固定 development set／固定模型下、新的獨立 participant sampling；不涵蓋換 training cohort、中心或硬體的所有不確定性。若以家庭／班級等群集招募，需在收案前改抽樣模型。不得用歷史 LOPO fold SD/√5、600 epochs 或三種 classifier 來產生新 cohort 的獨立 N／CI。[5]

## 7. 樣本數與招募上限：已完成的數值證據

**v1：目標 60 位 QA-evaluable participants，最多招募 94 位獨立 participants。** 這是在候選 N={20,40,60,80,100} 與 16 個明列情境中，通過 gate 的最小候選值，不是所有可能分布下的數學最小 N。

[`precision_plan.json`](../configs/precision_plan.json) 以 latent participant BA 均值 0.50／0.65／0.80、SD 0.10／0.20、每類 20／30 epochs、額外 within-class epoch ICC 0／0.10／0.30、QC attrition、missing-class 及 paired-model correlation 做情境分析。這些是假設範圍，不從舊五人結果估參數。

Informative QC 情境允許 retention 與 latent performance 相關；計算 **留存可評估者** 的解析真值與 coverage，另外報 all-recruited 真值偏移。它不假設 QC 隨機，也不把可評估者 CI 說成全體招募者的 CI。

| 數值驗證 | 結果 |
|---|---|
| Screening | 每個 N／scenario 1,000 個模擬 studies、每次 1,000 bootstrap |
| 獨立 final validation | 選定 N60，16 scenarios 各 1,000 studies；每次完整 10,000 bootstrap；獨立 seed stream |
| Primary coverage | 0.934–0.956；最低 pointwise 95% Wilson MC 下界 0.9169 |
| Primary CI 半寬 p90 | 最大 0.07440，低於 0.08 |
| 半寬 ≤0.08 機率 | 0.992–1.000；最低 pointwise 95% MC 下界 0.9843 |
| Paired Δ CI 半寬 p90 | 最寬約 0.10354；不能宣稱有足夠 power 確認 Δ0.05 |

Gate 要求每個情境 primary coverage 的 MC 下界 ≥0.90，且達到半寬目標機率的 MC 下界 ≥0.90。這是預先宣告的 suitability floor，**不是已證明精確 95% coverage**；各 MC bounds 也是 pointwise，沒有宣稱全情境 family-wise 保證。初次 400-study screening 因 MC 下界不足未選出 N；只增加 screening 到 1,000、保留 seed／情境／門檻與原失敗結果，完整歷程見 [planning history](precision/development/history.json)。

招募 cap 用精確 binomial tail：[6]

```text
retention_floor = (1 - 0.25 QC loss) * (1 - 0.05 missing-class loss) = 0.7125
cap = smallest m with P[Binomial(m, 0.7125) >= 60] >= 0.95 = 94
P[reach 60 by cap 94 | these assumptions] = 0.95330
```

不以 `60 / retention` 的期望值代替完成機率。若實際 retention 只有 0.60，cap94 的完成機率約 0.2584；只有 0.50 時約 0.0048。這些是模型假設敏感度，不是觀察到的 attrition。到 cap 仍未滿 60 就停止並報 precision shortfall，不能看模型分數後補招至成功。

## 8. Score-blind 收案、停止與 cumulative ledger

停止新招募的條件為 **first of N60 可評估者或 94 位 unique recruited participants**。每次 audit 必須給完整 cumulative roster，不得分成數批各自重用 cap。已在停止時間前招募者可完成預定 session／合規 replacement，全部 eligible 人都列入，不挑恰好 60 人；因此 final N 可以略超過 60，但 recruited N 絕不超過 94。

每人有 UTC `recruited_at`、`closed_at`、population eligibility attestation 與 `pending/completed/withdrawn/lost_to_followup` disposition。`closed_at` 表示資料與 QA／disposition 已盲於模型結果而定案；到達 N 的時間以第 60 位 completed evaluable participant 的該時間為準。不得延後登錄 QA 結論來迴避停止。停止後新 recruit 必須失敗；cap 到達但仍有 pending participants 時，只停止招募，**不允許 outcome access**。

[`cohort_policy.assess_enrollment`](../src/eeg_ma/cohort_policy.py) 驗證 literal unique IDs、所有 session 上限、first valid session、task stop reasons、withdrawal、replacement window、Rest 完整度、correct-only、一人一權重及停止時間。其欄位是 operator-attested metadata；每個 session 綁定實際 shared raw preflight report 的 `qa.evidence_sha256`。它驗證記錄的一致性，不能取代 raw QA 或證明人工輸入真實。完整 manifest 會在 outcome event 內保存原始 bytes 與 hash。

Manifest schema：`eeg-enrollment-1`，頂層只有 `schema_version`、`cohort_id`、`outcomes_accessed:false`、`participants`。Participant 僅有 `subject_id`、`recruited_at`、`population_eligible:true`、`disposition`、`closed_at`、`sessions`。Session 僅有：

```text
session_id / protocol_id / attempt_number / started_at / duration_seconds
qa: passed (bool), reason_codes (error-code list), evidence_sha256
attempts / correct / eligible_correct: Type1, Type2 的 integer counts
rest_acquired / rest_eligible / rest_seconds: Rest1, Rest2 的 counts 或秒數
task_end: Type1, Type2 各一個 quota_reached / attempt_limit /
          session_time_limit / not_started / technical_failure / withdrawal
```

禁止額外分數、預測或 accuracy 欄位；bool 不當 integer。Synthetic 完整範例見 [`test_cohort_policy.py`](../tests/test_cohort_policy.py)，測試資料不是已招募參與者。

```bash
PYTHONPATH=src python scripts/17_audit_enrollment.py \
  --lock outputs/protocol-v1 \
  --enrollment private/enrollment.json \
  --out-dir outputs/enrollment-audit-001
```

每次新輸出，不能覆蓋舊 audit。`stop_recruitment`、`outcome_ready` 分別報告停止與可解盲；後者還需所有已招募者定案。失訪、撤回、缺類與技術失敗留在 flow，不填 chance score，不推論 excluded subjects 的模型表現。

## 9. 真正的 lock 與啟動 gate

`validate → lock → launch → audit enrollment → record-outcome-access` 是現有操作順序；詳見 [protocol gate 操作](protocol_gate.md)。這些命令不能取代 acquisition、人員核准或 #11 的固定模型執行管線。

| 證據 | v1 repo 狀態 | Lock 所需真實內容 |
|---|---|---|
| 協定、QA、N／cap、endpoints | 已定義 | snapshot 與 SHA-256 |
| Simulation | 已生成、通過 16 scenarios | report、原 config／engine 身分、完整驗證數值 |
| Development boundary | 已綁定現有 feature bytes／subject IDs／legacy mapping | 真實資料檔與 manifest，不能只宣告 hash 而無檔案 |
| Feature schema／analysis config | 已生成 | exact ordered features、class mapping、數值方法 |
| Acquisition | **未完成** | device、units、reference、ground、歷史對齊／Fs／clock 等實證及核准 |
| Environment | **#10 reference stack／驗證機制已交付；實際 training evidence 隨 #11 產生** | 精確 runtime／dependency manifest、已驗證的 reference lock；不能以 CI／模擬環境冒充模型 training environment |
| Primary／reference fitted models | **未完成，#11** | binary artifacts、development-only training manifests、feature/config/data/eligibility hashes |
| 人員／倫理／資源 | **未完成** | 真實負責人、核准時間、倫理判定與 reference、資源承諾 |
| Registration | `not_registered` | 若採外部註冊，另填實際 URL／時間；local lock 不等於註冊 |

任何必要欄位 missing、null、unknown、false verification 或 artifact 缺失都阻擋 lock。鎖定時使用 clean Git commit 與 working-source fingerprint，複製實際 artifacts／development sources，建立不可原地覆寫的 snapshot；不反序列化不可信的模型 binary。之後驗證靠已封存 bytes，不依賴外部來源仍存在。

Launch roster 必須與 cohort ID、A protocol ID 一致；與 development／歷次 exposed subjects 不得交集。`record-outcome-access` 要求同一 full roster 的 enrollment audit 已達 stop 且沒有 pending，再記錄機器 UTC、原始 manifest bytes、SHA 與 hash-chain event。解盲後原 lock 不再接受招募。所有可見紀錄可稽核，但本地 ledger 不是數位簽章或防惡意管理員機制，無法證明有人從未私下看過結果。

## 10. Amendment、pilot 與後續實作邊界

收案前資源不足時，應先改 `study.mode=pilot_estimation`、重新指定可負擔 N／較寬精度目標、重新產生匹配該 N／cap 的 simulation report，再以新 version lock。單改 mode 不會讓 N20 重用 N60 的證據；仍要如實通過所宣告的 interval suitability gate。保留 parent lock／原計畫，結果只作 estimation，不以 pilot 名稱補救已看過的失敗 primary。

任何 amendment 要有理由、parent lock identity、時間及 changed sections。Outcome access 後的模型、QC、特徵、threshold 或 protocol 更動，必須記錄舊 test subjects 已 exposed，改用新的 cohort ID 與未見 holdout；不能把 `outcomes_accessed` 改回 false 清除歷史。即使只是修正文案，本地流程也保守地要求 post-outcome 新 lock 的新 holdout，不允許重開原收案。

#11 尚須完成 `fit → immutable predict → locked score`：development-only fit spies、label permutation 不改 predictions、ordered schema／subject boundary、prediction／label／eligibility hashes、兩模型相同 epochs、missing prediction fail closed。#12 尚須將本協定的人層級 CI／flow／interpretation 整合成最終研究報告。現在可驗證 #9 的協定工程與 synthetic 情境，尚不能宣稱真實固定模型外部驗證已完成。

## 參考來源

1. Cawley & Talbot (2010), [On Over-fitting in Model Selection and Subsequent Selection Bias in Performance Evaluation](https://www.jmlr.org/papers/v11/cawley10a.html).
2. White & Power (2023), [k-Fold Cross-Validation Can Significantly Over-Estimate True Classification Accuracy in Common EEG-Based Passive BCI Experimental Designs](https://www.mdpi.com/1424-8220/23/13/6077). DOI: 10.3390/s23136077.
3. MNE, [Rejecting bad data spans and breaks](https://mne.tools/stable/auto_tutorials/preprocessing/20_rejecting_bad_data.html)：threshold 的操作與適用條件，不能把 tutorial 數值視為本設備的已驗證參數。
4. SciPy, [bootstrap](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.bootstrap.html)：percentile／BCa、paired resampling、degenerate distribution 的區別。
5. Varoquaux (2018), [Cross-validation failure: Small sample sizes lead to large error bars](https://arxiv.org/abs/1706.07581).
6. SciPy, [binom](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.binom.html)：以 `sf(target-1, cap, retention)` 計算完成率。
7. Center for Open Science, [Preregistration](https://www.cos.io/initiatives/prereg)：事前計畫、已看過資料與後續 deviations 的區分。
