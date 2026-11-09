# 獨立新受試者研究：participant-level precision plan

本文件是 [confirmatory protocol](../confirmatory_protocol.md) 的設計證據。完成的有限情境模擬選出 **60 位可評估的新受試者**；若每位招募者可評估的機率至少為 **0.7125**，**招募上限 94 人**給出約 **95.33%** 達成 60 位的機率。所有數字都是在收案前明確選定的設計假設或合成模擬輸出，沒有讀取既有 EEG、canonical features、舊 cohort 表現或新 cohort outcome。

這項設計通過的目標是 primary participant-macro balanced accuracy (BA) 的 CI 精度。配對模型差值 Δ 的精度另列，並沒有因此宣稱足以辨識任何特定小幅改善。完整機器證據在 [precision_report.json](precision_report.json)、全部 80 個 screening cells 與 16 個 final validation cells 在 [scenarios.csv](scenarios.csv)，所有假設在 [precision_plan.json](../../configs/precision_plan.json)。

## 1. Estimand、獨立單位與 CI

每位受試者先各自計算兩類 recall，再等權平均：

$$
BA_i=\frac12\left(\frac{\sum_j I(\widehat y_{i0j}=0)}{m_{i0}}+
\frac{\sum_j I(\widehat y_{i1j}=1)}{m_{i1}}\right),\qquad
\widehat\theta=\frac1N\sum_{i=1}^N BA_i.
$$

目標是同一個已凍結模型在**符合預先規定、可評估族群**中的平均 BA。每位受試者的權重相同；epoch 數量只影響其 BA 的測量精度。這裡不把同一人的 epochs 當作獨立受試者。

推論使用雙側 95% **participant percentile bootstrap**：每次有放回抽取 N 個受試者索引，計算平均，取 bootstrap 分布的 2.5% 與 97.5% 分位數；分位數演算法固定為 NumPy `method="linear"`。實際推論與最終規劃驗證都使用 **10,000 resamples**。配對 Δ 使用同一位受試者、同一批合格 epochs、同一組 resampled participant indices，對 `BA_primary − BA_reference` 的平均建立 CI。此配對重抽原則與 [SciPy bootstrap 文件](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.bootstrap.html) 一致；cluster cases bootstrap 的抽樣單位應反映研究抽樣設計，見 [Loy & Korobova 的原始方法文章](https://journal.r-project.org/articles/RJ-2023-015/)。

`participant_bootstrap` 拒絕缺失、非有限、範圍外分數與重複 subject ID。零變異的 primary 或 paired Δ 各自標記 `degenerate=true` 與 `confirmatory_eligible=false`；不把零寬 CI 說成有力的確認證據。兩類最低 epoch 門檻與 participant-level eligibility 由 protocol 在傳入此函式前檢查。`confirmatory_eligible` 只是此函式的局部檢查，不能取代完整 protocol gate。

Percentile 是本版預先固定的方法，並非一概比 BCa、basic 或其他方法更好。研究問題是這個特定方法在下列有限假設下是否有足夠精度；不會依新 cohort 表現切換方法。

## 2. 生成機制：有界分數、人際差異、相關 epochs

令參與者的 primary 正確率為

$$
p_i\sim\operatorname{Beta}(\mu t,(1-\mu)t),\qquad
 t=\frac{\mu(1-\mu)}{\sigma^2}-1.
$$

因此每人機率在 [0,1]，招募族群的均值為 μ、潛在人際 SD 為 σ。每人的每一類另有 class probability：當 residual epoch correlation ρ 大於 0，

$$
q_{ic}\mid p_i\sim\operatorname{Beta}\left(p_i(1/\rho-1),(1-p_i)(1/\rho-1)\right),
\qquad Y_{icj}\mid q_{ic}\sim\operatorname{Bernoulli}(q_{ic}).
$$

ρ=0 時令 `q_ic=p_i`。同一類 epochs 因共享 q 而相關；ρ 指的是**給定 participant probability 後的同類 residual ICC**。加入人際差異後的總邊際相關不等於表中的 ρ。兩類共享同一 p、各自抽 q，然後各自算 recall。這是明確的 stress model，不是已驗證的 EEG 誤差分布。

參考模型設 `p_ind` 為獨立的同分布 participant probability，並使用

$$
p_{ref,i}=(1-\delta/\mu)\{w p_i+(1-w)p_{ind,i}\}.
$$

δ 為招募族群的預設平均差值；w 是機率混合權重，不直接等同模型分數的相關係數。部分 epochs 的 Bernoulli uniform draws 由兩模型共用，比例為 `epoch_shared_probability`。一般情境使用 w=0.5、epoch sharing=0.5、δ=0.05；chance 情境使用 δ=0。弱配對情境使用 w=0、epoch sharing=0；強配對使用 w=1、epoch sharing=1。class probability 的 residual draws 仍各自抽取。

## 3. QC attrition 與兩種不同的真值

每人的 QC 通過機率預設為

$$
r(p)=1-d+\lambda(p-\mu),
$$

d 是平均 QC loss，另有獨立的 missing-class 機率 m。設定驗證要求整個 p∈[0,1] 上的 r(p) 都在 [0,1]；沒有先 clip 再錯用線性解析真值。總平均 retention 為 `(1−d)(1−m)`。

當 QC 與 latent performance 有關，bootstrap 所估的可評估族群均值是

$$
\theta_{evaluable}=\mu+\frac{\lambda\sigma^2}{1-d},
$$

而全體招募族群均值仍為 μ。參考模型及 paired Δ 的 evaluable truth 依同一 selection distribution 解析計算。模擬以 rejection sampling 取得恰好 N 位可評估的人，評估的是**已達成 N 的條件下** CI 的精度；達成 N 的機率在第 6 節另外計算。

情境 15 的 evaluable BA 真值為 0.676667、recruited 真值為 0.650000；情境 16 分別為 0.521333 與 0.500000。相應 CI 對 evaluable 真值的實際模擬 coverage 是 0.944 與 0.943，對 recruited 真值卻只有 **0.846 與 0.906**。這個差異顯示 informative QC 的選擇偏差；participant bootstrap 本身不會修正此偏差。Primary acceptance 一律對 evaluable truth 判斷，recruited coverage 是清楚標示的 sensitivity output。

## 4. 固定的有限 stress scenarios

均值 0.50 / 0.65 / 0.80、人際 SD 0.10 / 0.20、residual ICC 0 / 0.1 / 0.3、20 或 30 epochs/class，以及不等 epoch 數、低/高配對與 informative QC，組成以下 **16 個明列情境**。它不是完整 Cartesian grid，也沒有宣稱覆蓋所有可能母群體。最低、最高 epoch 數對應本版 protocol 的可評估範圍；QC loss 與 missing-class 比例是規劃假設，尚非現場估計。

| # | Scenario ID | μ | SD | Residual ρ | Epochs/class | QC loss | Missing class | QC slope |
|---|---|---:|---:|---:|---|---:|---:|---:|
| 1 | `chance-low-variation-independent-epochs` | 0.50 | 0.10 | 0.0 | 30 / 30 | 0.00 | 0.00 | 0.0 |
| 2 | `moderate-low-variation` | 0.65 | 0.10 | 0.1 | 30 / 30 | 0.10 | 0.00 | 0.0 |
| 3 | `high-mean-low-variation` | 0.80 | 0.10 | 0.1 | 30 / 30 | 0.10 | 0.00 | 0.0 |
| 4 | `chance-high-variation` | 0.50 | 0.20 | 0.1 | 30 / 30 | 0.10 | 0.00 | 0.0 |
| 5 | `moderate-high-variation` | 0.65 | 0.20 | 0.1 | 30 / 30 | 0.10 | 0.00 | 0.0 |
| 6 | `high-mean-high-variation` | 0.80 | 0.20 | 0.1 | 30 / 30 | 0.10 | 0.00 | 0.0 |
| 7 | `chance-correlated-minimum-epochs` | 0.50 | 0.20 | 0.3 | 20 / 20 | 0.25 | 0.05 | 0.0 |
| 8 | `moderate-correlated-minimum-epochs` | 0.65 | 0.20 | 0.3 | 20 / 20 | 0.25 | 0.05 | 0.0 |
| 9 | `high-mean-correlated-minimum-epochs` | 0.80 | 0.20 | 0.3 | 20 / 20 | 0.25 | 0.05 | 0.0 |
| 10 | `moderate-correlated-low-variation` | 0.65 | 0.10 | 0.3 | 20 / 20 | 0.25 | 0.05 | 0.0 |
| 11 | `moderate-independent-minimum-epochs` | 0.65 | 0.20 | 0.0 | 20 / 20 | 0.10 | 0.05 | 0.0 |
| 12 | `unequal-class-epoch-counts` | 0.65 | 0.20 | 0.3 | 30 / 20 | 0.25 | 0.05 | 0.0 |
| 13 | `weak-model-pairing` | 0.65 | 0.20 | 0.3 | 20 / 20 | 0.25 | 0.05 | 0.0 |
| 14 | `strong-model-pairing` | 0.65 | 0.20 | 0.3 | 20 / 20 | 0.25 | 0.05 | 0.0 |
| 15 | `informative-qc-moderate` | 0.65 | 0.20 | 0.3 | 20 / 20 | 0.25 | 0.05 | 0.5 |
| 16 | `informative-qc-chance` | 0.50 | 0.20 | 0.3 | 20 / 20 | 0.25 | 0.05 | 0.4 |

## 5. N 選擇與最終模擬證據

候選 N 固定為 20、40、60、80、100。每個情境先以 1,000 次獨立合成研究、每研究 1,000 bootstrap resamples 作 screening。從最小通過 screening 的 N 起，以與 screening 分開的亂數 streams 做最終驗證：每情境 **1,000 次獨立合成研究 × 10,000 bootstrap resamples**。依候選 N 順序進行，第一個所有情境均通過的 N 為選定設計。

每個情境必須同時符合：

1. Primary nominal 95% CI 的 coverage，**95% Wilson Monte Carlo interval 下界 ≥0.90**。
2. CI half-width ≤0.08 的機率，其 **95% Wilson Monte Carlo interval 下界 ≥0.90**。
3. 模擬 CI half-width 的第 90 百分位 ≤0.08。
4. 沒有 primary zero-variance study。

Wilson interval 用各次獨立模擬成功/失敗的二項計數計算，含 Monte Carlo 誤差；其方法可見 [SciPy `proportion_ci` 文件](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats._result_classes.BinomTestResult.proportion_ci.html)。**0.90 是事前選定的 adequacy tolerance，不是 nominal 95% coverage 已被證明。** 這些 MC intervals 都是逐情境、逐指標的 pointwise intervals，沒有同時涵蓋 16 情境的 95% familywise 保證。候選選擇與反覆規劃亦不能解讀成一項獨立的 confirmatory coverage 試驗。

完成的 screening：

| Candidate N | Lowest coverage MC lower | Largest half-width p90 | Lowest width-pass MC lower | Screening result |
|---|---:|---:|---:|---|
| 20 | 0.8734 | 0.1338 | 0.0016 | fail |
| 40 | 0.9070 | 0.0923 | 0.2889 | fail |
| 60 | 0.9048 | 0.0746 | 0.9830 | pass |
| 80 | 0.9224 | 0.0639 | 0.9962 | pass |
| 100 | 0.9158 | 0.0570 | 0.9962 | pass |

N=60 是第一個 screening 通過的候選，並在獨立 final validation 通過。其最終結果如下；coverage 以 evaluable truth 為準：

| Scenario # | Primary coverage | 95% MC lower | Primary half-width p90 | Width-pass 95% MC lower | Paired Δ half-width p90 |
|---|---:|---:|---:|---:|---:|
| 1 | 0.942 | 0.9258 | 0.0331 | 0.9962 | 0.0276 |
| 2 | 0.956 | 0.9414 | 0.0429 | 0.9962 | 0.0499 |
| 3 | 0.948 | 0.9324 | 0.0396 | 0.9962 | 0.0449 |
| 4 | 0.936 | 0.9191 | 0.0634 | 0.9962 | 0.0602 |
| 5 | 0.943 | 0.9269 | 0.0629 | 0.9962 | 0.0597 |
| 6 | 0.941 | 0.9246 | 0.0621 | 0.9962 | 0.0551 |
| 7 | 0.949 | 0.9336 | 0.0744 | 0.9870 | 0.0844 |
| 8 | 0.945 | 0.9291 | 0.0732 | 0.9912 | 0.0823 |
| 9 | 0.949 | 0.9336 | 0.0690 | 0.9944 | 0.0729 |
| 10 | 0.934 | 0.9169 | 0.0594 | 0.9962 | 0.0779 |
| 11 | 0.945 | 0.9291 | 0.0579 | 0.9962 | 0.0452 |
| 12 | 0.945 | 0.9291 | 0.0724 | 0.9927 | 0.0813 |
| 13 | 0.940 | 0.9235 | 0.0731 | 0.9962 | 0.1035 |
| 14 | 0.943 | 0.9269 | 0.0731 | 0.9927 | 0.0702 |
| 15 | 0.944 | 0.9280 | 0.0713 | 0.9912 | 0.0815 |
| 16 | 0.943 | 0.9269 | 0.0744 | 0.9843 | 0.0842 |

最終最低 primary coverage 為 0.934、最低 MC 下界為 0.9169；最大 primary half-width p90 為 0.0744。所有情境均無 primary degenerate studies。這支持所宣告有限假設下的 primary precision 規劃。

弱模型配對下 paired Δ 的 half-width p90 為 **0.1035**。因此即使 primary BA 精度合格，仍可能無法對 0.05 的改善給出排除零的區間。這裡的 δ=0.05 是 sensitivity input，沒有被解讀成已證實的 effect、最小有意義差異或檢定力保證；paired-delta coverage、bias 與 degenerate counts 全部保留於機器報告。

## 6. 招募上限與完成機率

最差宣告 retention 為 `(1−0.25)(1−0.05)=0.7125`。在招募者之間獨立、每人相同邊際 evaluability probability 的假設下，取最小整數 C，使

$$
P\{\operatorname{Binomial}(C,0.7125)\ge60\}\ge0.95.
$$

使用二項 survival function 精確計算尾端機率，方法定義見 [SciPy binomial 文件](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.binom.html)。得到 **C=94，完成機率 0.953301**；93 人尚不滿足門檻。單純將 60 除以 retention 後向上取整只描述平均需求，不能提供同樣的完成機率。

| Retention assumption | Cap needed for ≥95% completion | Completion probability at cap 94 | Meets assurance |
|---|---:|---:|---|
| 1.0000 | 60 | 1.000000 | yes |
| 0.9000 | 71 | 1.000000 | yes |
| 0.7500 | 89 | 0.994268 | yes |
| 0.7125 | 94 | 0.953301 | yes |
| 0.6000 | 114 | 0.258408 | no |
| 0.5000 | 139 | 0.004774 | no |

Retention 若只有 0.60，94 人上限的完成機率降為 0.2584；若為 0.50，降為 0.0048。這是具實質影響的可行性假設，必須在 launch gate 中誠實處理，不能把未知現場 attrition 當成已觀察到 0.7125。若收案前的盲態技術可行性無法支持此假設，應先建立新 protocol revision；已看到 outcome 後不能追改招募上限、排除規則或 precision target。

## 7. 模擬開發歷史：一次提高 screening 精度

初次規劃使用 400 次研究、每次 1,000 resamples screening；固定的 16 情境、候選 N、亂數 seed 與 acceptance thresholds 均與目前相同。這次 screening **沒有任何候選被認證通過**，沒有啟動 final validation。

| Candidate N | Initial 400-study coverage lower minimum | Initial largest half-width p90 | Initial width-pass lower minimum |
|---|---:|---:|---:|
| 20 | 0.8639 | 0.1332 | 0.0014 |
| 40 | 0.8892 | 0.0919 | 0.3001 |
| 60 | 0.8978 | 0.0752 | 0.9578 |
| 80 | 0.8864 | 0.0644 | 0.9905 |
| 100 | 0.8978 | 0.0573 | 0.9905 |

當 N≥60，width 標準已通過，但每個候選至少有一個情境的 coverage MC 下界低於 0.90。例如 N=60 的一個情境是 371/400=0.9275，其 Wilson 下界 0.8978。這個結果本身無法區分低於目標的 coverage 與 Monte Carlo 誤差；沒有把它刪掉或說成已通過。

接著**一次**把 screening study 數提高到 1,000；維持 seed、所有情境、CI 方法、1,000 個 screening bootstrap resamples、最終 1,000×10,000 procedure，以及全部門檻。沒有換 seed、降低 coverage 門檻或繼續調整直到通過。Study count 會改變隨機生成陣列尺寸，所以第二次 run 是可重現的較大 simulation，不能把它當成初次樣本的簡單追加。分開的 final-validation streams 在此之後驗證 N=60。

完整初次報告、80 個 cells、初次設定、與初次 engine hash 相符的原始碼 snapshot 均保留於 [development/](development/)。[history.json](development/history.json) 逐項記錄這次更動與兩版 hashes。Engine 的另一項差異只是在 report validator 回傳 metadata 加入四個 gate-binding fields；生成分數與 CI 的演算法未改變。這是新 cohort 收案前的模擬方法開發歷史，不是已在獨立 registry 預註冊的統計驗證。

## 8. 重現、完整性與限制

在 repository 的 Python 環境中，從 repo root 執行：

```bash
python scripts/15_plan_precision.py \
  --config configs/precision_plan.json \
  --out-dir outputs/precision-reproduction
```

輸出目錄必須尚不存在；CLI 不覆寫先前報告。正式規劃運算有 80 個 screening cells 與 16 個 final cells，應預留足夠運算時間。單元測試使用小型 `smoke_test` 設定檢查程式路徑；smoke-only report 會被 protocol lock validator 拒絕，不能當作正式設計證據。

報告記錄 master seed、每個具名 cell 的 seed digest、兩階段 study/bootstrap 次數、設定、環境與 hashes。同一設定、原始碼與環境可重現相同機器報告；各 cell 分開派生 RNG，添加無關情境不會改變既有情境的 streams。報告 hash 對移除自身 `report_sha256` 欄位後的 canonical JSON 計算；engine hash 對 `src/eeg_ma/precision.py` 的 bytes 計算。

`validate_precision_report` 會驗證 config/engine/report hashes、完整且不重複的 cell grid、解析真值、由成功次數重算的 MC intervals、screening/final 選擇順序、最終門檻與最小 exact-binomial cap。單獨把 `validation_pass` 改成 true 或刪除不通過情境不能授權 protocol lock。Hash 和結構檢查提供來源綁定與內部一致性，並不代替獨立重跑，也不是對蓄意重寫整套證據的數位簽章。

本報告環境與身份：

- Python **3.12.14**；NumPy **2.3.5**；SciPy **1.17.0**。
- Master seed：`20261109`。
- Config SHA-256：`fa2d3549fe9d0193a7ea7cb823c8f734b8c234bd6ff02bd2bc12d91af68bca7f`。
- Engine SHA-256：`8bf8ecf23aa0fc19c584a31da2dc51d2c19319abda4909ed6a4575c1cbc1aa78`。
- Report SHA-256：`1db7c5e29481e172ed10cdfa075679a143835723e02d2a018e1c3add7c4e16b5`。

這個結果的適用範圍仍有限：假設獨立受試者、固定訓練資料與模型、宣告的有界 latent family、QC selection mechanism 與兩類測量機制。它沒有包含重新訓練變異、跨站點群聚、設備偏移、所有可能的尾部分布、時間性 drift 或任何未列入的 selection mechanism。各情境的人際 SD 不包括全部 epoch measurement noise；observed participant BA variance 另由 class/epoch 層生成。因此沒有用舊 5 人的平均或 SD 取代不確定的下一批參與者分布，也沒有把有限情境通過解讀成對任意真實分布的保證。

以具體 precision 目標與適用的 simulation 規劃 external validation 規模，亦符合 [Snell et al. (2021) 原始 simulation study](https://pmc.ncbi.nlm.nih.gov/articles/PMC8352630/) 所討論的設計方向；該文的臨床預測與事件數建議不直接移植到本 EEG 任務。硬體、任務與 label 取得、倫理核准、participant overlap 與 frozen artifact 可用性仍由主 protocol 的 launch gate 分別確認。Precision report 通過不表示已完成這些現場條件或已完成新 cohort 驗證。
