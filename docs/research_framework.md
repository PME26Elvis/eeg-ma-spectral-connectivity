# 專題研究分析框架

本文件把整個專題的分析順序、每一步的依據、觀察結果與後續推導串起來。後續投影片與口頭報告可直接沿用此架構。詳細數值整理見 [`final_results_summary.md`](final_results_summary.md)。

## 1. 研究起點：建立可對照的 Poster / Lab baseline

先依照學長海報與實驗室教材建立固定 baseline：

```text
2–50 Hz preprocessing
→ 5-s epochs
→ 42 BP + 126 COH
→ training-only SFS
→ LDA / RBF-SVM / KFDA
→ intra-subject 5-fold CV
→ inter-subject LOPO-CV
```

海報沒有明定的實作細節，例如 COH segment length、KFDA regularization mapping，均在 repo 中明確記錄。SFS、z-score 與 model tuning 只在 training data 內 fit，避免 test leakage。

### Baseline observation

- Intra-subject：Rest / Mental Arithmetic 有明顯可分性。
- Inter-subject：兩個 comparison 多數接近 chance。

這顯示目前的主要瓶頸集中在跨受試者泛化。後續實驗固定 preprocessing、CV 與 classifiers，優先改變 feature representation。

## 2. BP-only / COH-only：先確認 baseline feature family

Baseline 的兩個主要 feature family 分開測試：

```text
BP only
COH only
BP + COH
```

結果顯示三種設定在 inter-subject 都沒有形成明顯穩定優勢，Rest1 vs Type1 與 Rest2 vs Type2 仍大致接近 chance。

由此可知，跨受試者問題沒有因單獨保留 BP 或 COH 而解決。後續需要測試新的 representation。

## 3. E1：Frontal hemispheric asymmetry

7 個 frontal channels 中有三組自然左右配對：

```text
FP1 ↔ FP2
F3  ↔ F4
F7  ↔ F8
```

E1 使用：

```text
log(BP_right) - log(BP_left)
```

共 18 個 asymmetry features。

### 假說

不同受試者的絕對 BP 尺度可能差異較大；左右相對差異可能保留較一致的 task-related information。

### 結果

E1 沒有穩定改善 inter-subject classification。Rest1 vs Type1 三種 classifier 都小幅下降，Rest2 vs Type2 僅有約 +0.01 至 +0.02 的小幅變化。

### 推導

目前資料不支持 frontal BP asymmetry 作為主要跨受試者改善來源，因此後續沒有沿此方向增加更多 asymmetry variants。

## 4. E2：Phase Locking Value

第二個 extension 引入 PLV：

```text
PLV = |mean(exp(j * Δphi))|
```

7 channels 形成 21 組 pairs，搭配 6 bands，共 126 個 PLV features。PLV 使用 continuous band-pass + Hilbert analytic phase，再切 5-s epoch。

### 假說

BP 與 COH 都包含振幅相關資訊；PLV 直接描述 phase difference 的穩定程度。若 phase synchrony 的跨人一致性較高，PLV 可能改善 LOPO generalization。

### 結果

Rest1 vs Type1 inter-subject：

```text
Poster baseline
KFDA 0.533 / LDA 0.523 / SVM 0.533

BP+COH+PLV
KFDA 0.667 / LDA 0.687 / SVM 0.660
```

三種 classifier 都提高。4/5 位 held-out subjects 有明顯改善，1 位約持平。

Rest2 vs Type2 沒有同等幅度的提升。

SFS 中多個 recurring PLV features 集中在 alpha band，例如：

```text
Fz–F8 alpha
F3–F4 alpha
FP2–F7 alpha
F3–Fz alpha
```

### 推導

E2 顯示 PLV 值得進一步拆解。需要確認增益來自 PLV 本身，或來自與 BP / COH 的 feature combination。

## 5. PLV decomposition：拆解 improvement source

固定相同 evaluation pipeline，測試：

```text
PLV only
BP + PLV
COH + PLV
ASYM + PLV
BP + COH + ASYM + PLV
```

### Rest1 vs Type1 inter-subject

```text
PLV only       0.707 / 0.667 / 0.690
BP + PLV       0.687 / 0.640 / 0.650
COH + PLV      0.663 / 0.683 / 0.647
ASYM + PLV     0.673 / 0.703 / 0.680
All extensions 0.620 / 0.647 / 0.610
```

順序為 KFDA / LDA / RBF-SVM。

PLV-only 已保留完整 E2 的主要跨受試者增益。加入 BP、COH 或 asymmetry 沒有形成三種 classifier 一致的額外改善。

All-extensions E3 在 Type1 intra-subject 可達約 0.857–0.880，但 inter-subject 降到約 0.610–0.647。這表示額外 feature candidates 對 within-subject discrimination 有幫助時，跨受試者泛化仍可能因 subject-specific information、redundancy 或 SFS instability 而下降。

SFS family usage 也顯示 PLV 在所有含 PLV 的 Type1 inter-subject 實驗中都出現在 5/5 LOPO folds。

### 推導

PLV 本身已足以解釋主要 improvement。前面 SFS 又反覆指向 alpha，因此最後只保留兩個直接由結果推導出的驗證問題：

1. PLV 是否具有明顯 band specificity？
2. PLV gain 在抑制 exact zero-lag component 後是否仍存在？

## 6. Final validation A：PLV band specificity

六個既有 bands 全部單獨測試，每個實驗只有 21 個 PLV pair features。

### Rest1 vs Type1

| Band | KFDA | LDA | RBF-SVM | 三模型平均 |
|---|---:|---:|---:|---:|
| Alpha | 0.733 | 0.743 | 0.747 | **0.741** |
| Gamma | 0.727 | 0.653 | 0.680 | 0.687 |
| Beta-high | 0.637 | 0.620 | 0.653 | 0.637 |
| Delta | 0.593 | 0.570 | 0.613 | 0.592 |
| Beta-low | 0.583 | 0.590 | 0.580 | 0.584 |
| Theta | 0.547 | 0.557 | 0.543 | 0.549 |

Alpha-only 同時具有較高 mean accuracy 與較低 held-out-subject standard deviation：

```text
Alpha-only
KFDA    0.733 ± 0.081
LDA     0.743 ± 0.065
RBF-SVM 0.747 ± 0.069

All-band PLV
KFDA    0.707 ± 0.128
LDA     0.667 ± 0.116
RBF-SVM 0.690 ± 0.126
```

這與前面 SFS recurring alpha features 的觀察一致。Gamma 是第二高的單一 band，但跨 classifier 與 held-out subjects 的變異較大。

Rest2 vs Type2 中，alpha 也是六個 bands 中最高的一組，三模型平均約 0.591；整體效果仍弱於 Type1。

## 7. Final validation B：iPLV robustness

使用：

```text
iPLV = |Im(mean(exp(j * Δphi)))|
```

此表示會壓低 exact 0-lag 與 π-lag phase locking，用來檢查目前 PLV observation 對 zero-lag component 的依賴程度。

### Rest1 vs Type1

| Representation | KFDA | LDA | RBF-SVM |
|---|---:|---:|---:|
| Poster baseline | 0.533 | 0.523 | 0.533 |
| PLV all bands | 0.707 | 0.667 | 0.690 |
| PLV alpha only | 0.733 | 0.743 | 0.747 |
| iPLV only | 0.473 | 0.447 | 0.440 |

PLV gain 沒有在 iPLV 中保留。相較 all-band PLV，iPLV 約下降 0.22–0.25 accuracy。

### 解讀

目前有效的 PLV discrimination 高度依賴 zero-lag 或接近 zero-lag 的 phase-locking component。可能來源包括：

- common reference
- volume conduction
- shared / common signal source
- 真正同步且接近零相位差的神經活動
- frontal artifact 等共同成分

目前 7-channel、5-subject 資料無法把這些來源分離。因此 alpha-PLV 可作為有辨識力的 feature observation，生理 connectivity 的解讀需保持保守。

## 8. 完整研究推導鏈

```text
Poster / Lab baseline
↓
within-subject 可分，cross-subject 接近 chance
↓
BP-only / COH-only 拆解仍無明顯改善
↓
E1 asymmetry：沒有穩定跨人增益
↓
E2 PLV：Type1 inter-subject 明顯改善
↓
per-subject + SFS：改善跨多數 held-out subjects，alpha PLV recurring
↓
PLV decomposition：PLV-only 已保留主要增益
↓
Band-wise validation：alpha-only 最穩定，三模型平均約 0.741
↓
iPLV robustness：zero-lag suppression 後效果消失
↓
停止 feature engineering，進入結果統整與報告
```

## 9. 最終報告可採用的結論層級

### 可以直接陳述的資料結果

- Baseline intra-subject accuracy 明顯高於 inter-subject accuracy。
- E1 asymmetry 沒有穩定改善 cross-subject result。
- PLV 對 Rest1 vs Type1 的 LOPO accuracy 有一致提升。
- PLV-only 保留主要提升。
- Alpha-only PLV 在六個單一 bands 中平均表現最高，三種 classifier 結果接近。
- iPLV-only 接近或低於 Poster baseline。

### 適合用「顯示、支持、可能」描述的解讀

- PLV 可能比 baseline BP/COH 保留更多跨受試者共通資訊。
- Alpha-band phase locking 可能是 Type1 cross-subject discrimination 的主要來源之一。
- 額外 BP/COH/asymmetry candidates 可能帶入較多 subject-specific 或 redundant information。
- iPLV 結果顯示 PLV discrimination 對 zero-/near-zero-lag component 有高度依賴。

### 不適合由目前資料推出的結論

- PLV 已被證明為一般 EEG mental-arithmetic 的最佳 feature。
- Alpha PLV 具有確定的神經因果機制。
- Fz–F8、F3–F4 等 connections 代表確定的 functional connectivity pathway。
- 約 74% accuracy 是可外推到新 cohort 的 confirmatory performance。

## 10. 限制與停止點

主要限制：

- n=5，outer LOPO 只有 5 個 held-out units。
- 所有 extension 都在同一批資料上探索。
- 沒有 independent cohort。
- frontal 7-channel montage 限制空間解析度。
- common reference、volume conduction、shared source 與 artifact 會影響 phase-connectivity interpretation。
- Type2 behavioral accuracy 接近 ceiling，任務負荷與 EEG separability 可能較弱。

目前已完成 baseline replication、feature decomposition、兩個有理據的 extensions、PLV ablation 與 zero-lag robustness。這批資料的 feature-engineering 階段在此結束。後續工作為最終圖表、投影片與報告整理；若未來增加新受試者，再使用預先固定的方法做 confirmatory validation。
