# 最終書面報告草稿

> 本文以目前 5 位受試者的正式結果為基礎，整理成可直接改寫為期末書面報告的版本。所有延伸分析皆沿用既定 preprocessing、training-only SFS、LDA / RBF-SVM / KFDA 與 leakage-safe validation。數值採目前 repo 中的 canonical results。由於樣本數有限，所有 extension 與 final validation 的解讀維持描述性與探索性層級。

## 1. 研究目標與分析策略

本專題先重現實驗室既有 mental-arithmetic EEG 分析流程，再根據 baseline 結果規劃後續 feature representation 實驗。分析重點包含兩個問題：

1. 受試者內 Rest / Mental Arithmetic 是否具有可分類性。
2. 相同 feature representation 是否能泛化到未出現在 training set 的受試者。

為了區分這兩個問題，使用：

- intra-subject：每位受試者個別做 5-fold cross-validation。
- inter-subject：Leave-One-Participant-Out cross-validation（LOPO-CV）。

後續 extension 均以固定 baseline 為 reference，保留相同 classifier、SFS 與 validation 邏輯，只改變候選 feature representation。這樣可以追蹤每次修改對結果造成的影響，也能降低因同時更動多個分析環節而無法歸因的問題。

## 2. 資料與實驗設計

目前共 5 位受試者：`irene`、`jh`、`lin`、`pftest`、`yen`。

實驗流程為：

```text
Rest1 → Type1 → Rest2 → Type2
```

每位受試者正式分析資料包含：

- Rest1：30 個 5-s epochs。
- Type1：30 個答對 trial 的 5-s calculation epochs。
- Rest2：30 個 5-s epochs。
- Type2：30 個答對 trial 的 5-s calculation epochs。

因此每位受試者有 120 個正式 epochs，5 位受試者共 600 個 epochs。

EEG 以 HNC EAmp 8-channel 系統記錄：

```text
FP1, FP2, F7, F3, Fz, F4, F8, Cz
```

第一階段 replication 與後續 feature analysis 使用 7 個 frontal channels：

```text
FP1, FP2, F7, F3, Fz, F4, F8
```

Cz 保留於 raw data，但不進入目前的 168-feature baseline pipeline。

## 3. Baseline preprocessing 與 feature extraction

### 3.1 Preprocessing

連續 EEG 使用：

```text
4th-order Butterworth 2–50 Hz band-pass
→ zero-phase sosfiltfilt
→ event-aligned 5-s epochs
```

正式 Mental Arithmetic epoch 只使用 `analysis_eligible=true` 且答對的 trial。

### 3.2 Band Power

依實驗室教材採 DFT 後的頻帶能量加總：

```text
X[k] = DFT{x[n]}
PSD[f_k] = |X[k]|²
BP_band = Σ PSD[f_k]
```

六個 frequency bands 為：

```text
delta      1–4 Hz
theta      4–8 Hz
alpha      8–13 Hz
beta-low  13–20 Hz
beta-high 20–30 Hz
gamma     30–45 Hz
```

7 channels × 6 bands = 42 BP features。

### 3.3 Coherence

Coherence 依實驗室教材的 segmented-FFT magnitude-squared coherence 計算：

```text
Gxx = |X|²
Gyy = |Y|²
Gxy = X·Y*
Coh(f) = |mean(Gxy)|² / (mean(Gxx) · mean(Gyy))
```

5-s epoch 分成 5 個 non-overlapping 1-s segments；此 segment length 為 repo 明示的 implementation choice。每個頻帶內再對 coherence bins 取平均。

7 channels 共有 21 組 channel pairs：

```text
21 pairs × 6 bands = 126 COH features
```

Baseline 每個 epoch 共：

```text
42 BP + 126 COH = 168 features
```

## 4. Feature selection、classifier 與 validation

### 4.1 SFS

Sequential Forward Selection 在每個 outer training fold 內執行，使用 training-only inner-CV LDA accuracy 作為 greedy selection criterion。三種 classifier 共用同一個 outer fold 的 selected feature subset。

### 4.2 Classifiers

使用：

- LDA
- RBF-SVM
- KFDA

RBF-SVM 與 KFDA 的 hyperparameter search 只在 outer training data 的 inner validation 中完成。RBF grid 沿用實驗室海報設定：

```text
C = [0.1, 1, 10, 50, 100, 1000]
gamma = {1.05^-100, 1.05^-90, ..., 1.05^100}
```

### 4.3 Validation

Intra-subject 使用 stratified 5-fold CV。

Inter-subject 使用 LOPO-CV：每次保留 1 位受試者作為 outer test，其餘 4 位為 training。SFS 與 SVM/KFDA tuning 在 training subjects 內再用 Leave-One-Group-Out validation。

Normalization、SFS 與 tuning 都只從 outer training data fit，避免 test leakage。

## 5. Baseline 結果

### 5.1 Intra-subject

| Comparison | KFDA | LDA | RBF-SVM |
|---|---:|---:|---:|
| Rest1 vs Type1 | 0.823 | 0.803 | 0.817 |
| Rest2 vs Type2 | 0.717 | 0.710 | 0.713 |

### 5.2 Inter-subject LOPO

| Comparison | KFDA | LDA | RBF-SVM |
|---|---:|---:|---:|
| Rest1 vs Type1 | 0.533 | 0.523 | 0.533 |
| Rest2 vs Type2 | 0.490 | 0.527 | 0.517 |

Baseline 顯示受試者內分類明顯高於跨受試者分類。這個差異成為後續 extension 的主要出發點：目前 feature representation 對單一受試者具有辨識力，但在不同受試者之間缺乏穩定泛化。

## 6. Baseline feature-family decomposition

將 baseline 拆成 BP-only、COH-only 與 BP+COH 後，inter-subject 結果仍大致落在 chance 附近。沒有單一 baseline feature family 能明顯改善跨受試者泛化。

這一步保留了兩個資訊：

1. BP 與 COH 都能在部分受試者內提供可分性。
2. 目前 cross-subject 問題不能只靠移除 BP 或移除 COH 解決。

因此後續改從新的 relative / phase representation 測試。

## 7. E1：Frontal hemispheric asymmetry

E1 利用 frontal montage 的左右對稱 electrode pairs：

```text
FP1 ↔ FP2
F3  ↔ F4
F7  ↔ F8
```

每個 pair、每個 band 計算：

```text
log(BP_right) - log(BP_left)
```

共 18 asymmetry features，正式 E1 為 BP+COH+Asymmetry。

Rest1 vs Type1 inter-subject 相較 baseline 的平均變化：

```text
KFDA    -0.013
LDA     -0.007
RBF-SVM -0.023
```

Rest2 vs Type2 只有約 +0.01 至 +0.02 的小幅變化。E1 沒有形成穩定跨受試者增益，因此後續不再擴大 asymmetry variants。

## 8. E2：Phase Locking Value

PLV 用來描述兩個 channel 之間 phase difference 的穩定程度：

```text
PLV = |mean(exp(j * Δphi))|
```

對 21 組 channel pairs 與 6 個 bands，共 126 PLV features。PLV extraction 使用 continuous band-pass、Hilbert analytic phase，再依正式 5-s epoch 計算。

Feature sanity check：

```text
600 epochs
126 PLV features
NaN = 0
PLV range = [0.005693, 0.999230]
median = 0.734884
PLV > 0.90 = 9.52%
PLV > 0.95 = 2.60%
PLV > 0.99 = 0.176%
```

沒有觀察到大量 feature 飽和在 1 的數值異常。

### 8.1 Rest1 vs Type1

| Classifier | Poster baseline | BP+COH+PLV | Δ |
|---|---:|---:|---:|
| KFDA | 0.533 | 0.667 | +0.133 |
| LDA | 0.523 | 0.687 | +0.163 |
| RBF-SVM | 0.533 | 0.660 | +0.127 |

三種 classifier 都出現相同方向的提升。Per-subject 檢查顯示 `irene`、`lin`、`pftest`、`yen` 有明顯提升，`jh` 約持平。

SFS recurring features 主要集中在 alpha-band PLV，例如：

```text
PLV__Fz__F8__alpha      4/5 LOPO folds
PLV__F3__F4__alpha      3/5
PLV__FP2__F7__alpha     3/5
PLV__F3__Fz__alpha      2/5
```

### 8.2 Rest2 vs Type2

同一 E2 extension 對 Rest2 vs Type2 幾乎沒有改善。這表示 PLV 的主要 observation 集中在 Type1 comparison。

## 9. PLV decomposition

為了確認 E2 的增益來源，固定 evaluation protocol，只改 feature family：

```text
PLV only
BP + PLV
COH + PLV
Asymmetry + PLV
BP + COH + Asymmetry + PLV
```

### 9.1 Rest1 vs Type1 inter-subject

| Feature set | KFDA | LDA | RBF-SVM |
|---|---:|---:|---:|
| Poster baseline | 0.533 | 0.523 | 0.533 |
| BP+COH+PLV | 0.667 | 0.687 | 0.660 |
| PLV only | **0.707** | 0.667 | **0.690** |
| BP+PLV | 0.687 | 0.640 | 0.650 |
| COH+PLV | 0.663 | 0.683 | 0.647 |
| ASYM+PLV | 0.673 | **0.703** | 0.680 |
| All extensions | 0.620 | 0.647 | 0.610 |

PLV-only 已保留完整 E2 的主要跨受試者增益。加入 BP、COH 或 asymmetry 後，沒有形成三種 classifier 一致的額外提升。

全部 312 features 的 E3 在 Type1 intra-subject 可達約 0.857–0.880，但 inter-subject 下降到約 0.610–0.647。這個對比顯示較大的候選 feature pool 對 within-subject discrimination 有幫助，但在目前小樣本 LOPO setting 中可能增加 subject-specific information 與 feature-selection instability。

## 10. Final validation：PLV band specificity

前一階段 SFS 多次選到 alpha-band PLV，因此最後一輪將六個 bands 全部做 single-band ablation。每一組只有 21 個 PLV pair features，沿用相同 LOPO / inner SFS / classifier pipeline。

### 10.1 Rest1 vs Type1

| PLV band | KFDA | LDA | RBF-SVM | 三模型平均 |
|---|---:|---:|---:|---:|
| **Alpha** | **0.733** | **0.743** | **0.747** | **0.741** |
| Gamma | 0.727 | 0.653 | 0.680 | 0.687 |
| Beta-high | 0.637 | 0.620 | 0.653 | 0.637 |
| Delta | 0.593 | 0.570 | 0.613 | 0.592 |
| Beta-low | 0.583 | 0.590 | 0.580 | 0.584 |
| Theta | 0.547 | 0.557 | 0.543 | 0.549 |

All-band PLV-only：

```text
KFDA    0.707 ± 0.128
LDA     0.667 ± 0.116
RBF-SVM 0.690 ± 0.126
```

Alpha-only：

```text
KFDA    0.733 ± 0.081
LDA     0.743 ± 0.065
RBF-SVM 0.747 ± 0.069
```

Alpha-only 在三種 classifier 都高於 Poster baseline，並且高於 all-band PLV 的 mean accuracy。跨 held-out subjects 的 standard deviation 也較低。Gamma 是第二高的 single-band representation，但跨 classifier 與跨 subject 的穩定度較 alpha 弱。

### 10.2 Rest2 vs Type2

Alpha-only 仍是六個 bands 中最高的一組：

```text
KFDA    0.570
LDA     0.600
RBF-SVM 0.603
三模型平均 0.591
```

Type2 整體效果較弱，因此報告以 Type1 作為主要 phase-connectivity observation，Type2 保留為 secondary comparison。

## 11. Final validation：iPLV robustness

為了檢查 PLV observation 對 zero-lag component 的依賴程度，最後加入：

```text
iPLV = |Im(mean(exp(j * Δphi)))|
```

此 representation 會壓低 exact 0-lag 與 π-lag phase locking。

### Rest1 vs Type1

| Representation | KFDA | LDA | RBF-SVM |
|---|---:|---:|---:|
| Poster baseline | 0.533 | 0.523 | 0.533 |
| PLV all bands | 0.707 | 0.667 | 0.690 |
| Alpha PLV | 0.733 | 0.743 | 0.747 |
| iPLV | 0.473 | 0.447 | 0.440 |

相較 all-band PLV：

```text
KFDA    -0.233
LDA     -0.220
RBF-SVM -0.250
```

PLV 的跨受試者增益沒有在 iPLV 中保留。現有結果顯示目前有效的 PLV discrimination 高度依賴 zero- 或 near-zero-lag phase-locking component。

這個現象有多種可能來源：

- common reference
- volume conduction
- shared signal source
- frontal artifact
- 真正接近零相位差的同步神經活動

目前 7-channel、5-subject 的資料無法把這些來源分離，因此最終報告將 PLV 結果主要解讀為 classification representation observation；對 functional connectivity 的生理意義維持保守描述。

## 12. Behavioral context

Type2 的 behavioral accuracy 對所有受試者都接近 ceiling，反應時間也普遍比 Type1 快。這提供一個合理的背景：Type2 任務可能較容易，Rest2 / Type2 的 EEG state difference 因此較弱。

這項 behavioral observation 目前只能作為解釋背景，不能單獨證明 Type2 EEG classification 較弱的因果原因。

## 13. 討論

### 13.1 Within-subject 與 cross-subject 的差異

Baseline 在 intra-subject 有明顯可分性，但 inter-subject 接近 chance，顯示 subject variability 對目前 BP / COH representation 影響很大。

後續 E1、E2 與 decomposition 的結果進一步顯示：某些 feature 可以在同一人內提供辨識資訊，但未必在不同人之間保持相同結構。PLV 在 Type1 上提供較穩定的跨人 representation，尤其 alpha-band PLV。

### 13.2 Alpha-band PLV

Alpha-only PLV 在三種 classifier 的平均表現都高於 baseline、all-band PLV 與其他單一 bands，並且跨 subject standard deviation 較低。這與前一階段 SFS recurring alpha features 一致，因此形成一條連續的分析證據鏈：

```text
E2 中 PLV 改善
→ SFS recurrence 指向 alpha
→ PLV-only 保留主要增益
→ six-band validation 中 alpha 排名最高
```

目前結果支持 alpha-band phase-locking 是本資料中值得注意的 representation。樣本數與探索流程限制仍需保留。

### 13.3 iPLV robustness 的意義

iPLV 沒有保留 PLV gain，表示目前分類可用的 phase-locking information 主要存在於 zero-/near-zero-lag component。這降低了直接把 PLV network 解讀成生理 connectivity 的可信度。

此結果也提供明確的研究邊界：classification performance 與 physiological connectivity interpretation 應分開描述。PLV 可以是有效的分類 feature；其生理來源仍需更多 channels、不同 reference strategy、更多 subjects 或專門的 connectivity control 才能確認。

## 14. 研究限制

目前主要限制為：

- 受試者只有 5 位，LOPO 外層只有 5 個 held-out units。
- E1、E2、decomposition 與 final validation 都使用同一批資料。
- 沒有 independent cohort 進行 confirmatory validation。
- 已探索多種 feature representations，因此最高 accuracy 不能視為獨立確認的 generalization estimate。
- 只使用 7 frontal channels 做第一階段 analysis。
- PLV 受 reference、volume conduction、common source 與 frontal artifact 影響。
- iPLV 只能壓低 zero-lag component，不能完全排除 volume conduction 或 reference bias。
- Type2 behavioral performance 接近 ceiling，兩個 task 的難度與認知負荷並不完全等同。

## 15. 結論

本專題先以實驗室 Poster/Lab pipeline 建立固定 baseline，再根據結果逐步測試新的 feature representation。

目前可支持的結論為：

1. BP+COH baseline 在受試者內具有明顯 Rest / Mental Arithmetic 可分性，跨受試者 LOPO 泛化較弱。
2. Frontal hemispheric BP asymmetry 在目前資料上沒有帶來穩定的跨人增益。
3. PLV 對 Rest1 vs Type1 的 inter-subject classification 有明顯改善，三種 classifier 與多數 held-out subjects 的方向一致。
4. PLV-only 已能保留主要 improvement，較大的 mixed feature pool 沒有帶來一致的 cross-subject優勢。
5. Band-wise validation 顯示 alpha-only PLV 是目前平均表現最高且較穩定的 single-band representation。
6. iPLV-only 接近或低於 baseline，顯示目前 PLV gain 依賴 zero-/near-zero-lag component，因此 functional-connectivity 生理解讀需保持保守。

在目前 5-subject dataset 上，feature exploration 至此停止。後續工作集中在圖表、投影片、口頭報告與書面整理。若未來增加新的受試者或獨立 cohort，應固定目前方法後再做 confirmatory validation。

## 16. 建議報告圖表

主報告可優先使用：

```text
results/final_figures/final_story_type1_inter.*
results/final_figures/final_plv_band_comparison_type1.*
results/final_figures/final_alpha_vs_baseline_by_subject.*
results/final_figures/final_phase_connectivity_validation.*
results/final_figures/final_alpha_plv_sfs_network.*
```

Behavioral context 可搭配：

```text
results/figures/behavioral_accuracy.*
results/figures/behavioral_rt_median.*
```

完整結果表與 implementation details 放入 appendix，可避免主文被過多數值與超參數細節切碎。