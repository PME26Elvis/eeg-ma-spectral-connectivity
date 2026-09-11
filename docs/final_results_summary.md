# 最終成果分析摘要

本文件整理目前 5 位受試者資料的主要分析結果，作為後續投影片與報告的數值依據。所有 extension 與 validation 都沿用既定的 preprocessing、training-only SFS、LDA / RBF-SVM / KFDA，以及 inter-subject LOPO 架構。資料量有限，因此以下結果以描述性、探索性解讀為主。

## 1. Poster / Lab baseline

Baseline 使用：

```text
2–50 Hz preprocessing
→ 5-s epochs
→ 42 BP + 126 COH = 168 features
→ training-only SFS
→ LDA / RBF-SVM / KFDA
→ intra-subject 5-fold CV
→ inter-subject LOPO-CV
```

5 位受試者共 600 個正式 epoch。每個 comparison、每位受試者都有 30 Rest + 30 Mental Arithmetic。

### Inter-subject LOPO

| Comparison | KFDA | LDA | RBF-SVM |
|---|---:|---:|---:|
| Rest1 vs Type1 | 0.533 | 0.523 | 0.533 |
| Rest2 vs Type2 | 0.490 | 0.527 | 0.517 |

### Intra-subject

| Comparison | KFDA | LDA | RBF-SVM |
|---|---:|---:|---:|
| Rest1 vs Type1 | 0.823 | 0.803 | 0.817 |
| Rest2 vs Type2 | 0.717 | 0.710 | 0.713 |

這組結果建立了後續分析方向：同一受試者內的 Rest / Mental Arithmetic 有明顯可分性，跨受試者泛化接近 chance。後續實驗因此集中在 feature representation 與 subject variability。

## 2. BP / COH decomposition

將 baseline 拆成 BP-only、COH-only 與 BP+COH 後，Type1 / Type2 的 inter-subject 結果仍大致接近 chance。這表示跨受試者問題沒有因單獨保留 BP 或 COH 而明顯改善。

## 3. E1：Frontal hemispheric asymmetry

E1 由三組左右對稱 frontal electrodes 建立 18 個 log-ratio BP asymmetry features：

```text
FP1 ↔ FP2
F3  ↔ F4
F7  ↔ F8

log(BP_right) - log(BP_left)
```

E1 使用 baseline + asymmetry。Rest1 vs Type1 的 inter-subject 變化為：

```text
KFDA    -0.013
LDA     -0.007
RBF-SVM -0.023
```

Rest2 vs Type2 僅有約 +0.01 至 +0.02 的小幅變化。此資料上沒有觀察到穩定的 asymmetry 跨受試者增益。

## 4. E2：加入 PLV

PLV 定義：

```text
PLV = |mean(exp(j * Δphi))|
```

7 個 frontal channels 形成 21 組 channel pairs，搭配 6 個 bands，共 126 PLV features。PLV feature extraction 使用 continuous band-pass + Hilbert analytic phase，再切正式 5-s epoch。

PLV sanity check：

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

沒有看到整批 PLV 飽和在 1 的數值異常。

### Rest1 vs Type1：inter-subject

| Classifier | Poster baseline | BP+COH+PLV | Δ |
|---|---:|---:|---:|
| KFDA | 0.533 | 0.667 | +0.133 |
| LDA | 0.523 | 0.687 | +0.163 |
| RBF-SVM | 0.533 | 0.660 | +0.127 |

4/5 位 held-out subjects 在三種 classifier 都有明顯改善；`jh` 約持平。SFS recurring features 也以 alpha-band PLV 為主，包含：

```text
PLV__Fz__F8__alpha      4/5 LOPO folds
PLV__F3__F4__alpha      3/5
PLV__FP2__F7__alpha     3/5
PLV__F3__Fz__alpha      2/5
```

### Rest2 vs Type2

加入 PLV 後沒有形成相同幅度的跨受試者提升。

## 5. PLV decomposition

為了拆解 E2 的增益來源，固定 evaluation protocol，只改候選 feature families。

| Experiment | Feature set | Candidate features |
|---|---|---:|
| E2a | PLV only | 126 |
| E2b | BP + PLV | 168 |
| E2c | COH + PLV | 252 |
| E2d | Asymmetry + PLV | 144 |
| E3 | BP + COH + Asymmetry + PLV | 312 |
| E2 full | BP + COH + PLV | 294 |

### Rest1 vs Type1：inter-subject

| Feature set | KFDA | LDA | RBF-SVM |
|---|---:|---:|---:|
| Poster baseline | 0.533 | 0.523 | 0.533 |
| BP+COH+PLV | 0.667 | 0.687 | 0.660 |
| PLV only | **0.707** | 0.667 | **0.690** |
| BP+PLV | 0.687 | 0.640 | 0.650 |
| COH+PLV | 0.663 | 0.683 | 0.647 |
| ASYM+PLV | 0.673 | **0.703** | 0.680 |
| All extensions | 0.620 | 0.647 | 0.610 |

PLV-only 已保留完整 E2 的主要跨受試者增益。BP、COH 或 asymmetry 加入後沒有形成三種 classifier 一致的額外提升。E3 在 intra-subject Type1 可達約 0.857–0.880，但 inter-subject 降到約 0.610–0.647，顯示額外 candidate features 在小樣本跨受試者設定中可能增加 feature-selection instability 或 subject-specific discrimination。

SFS family usage 也顯示 PLV 在所有含 PLV 的 Type1 inter-subject 實驗中都出現在 5/5 outer folds。COH 在有提供時也常被保留；BP 與 asymmetry 的 fold presence 較低。

## 6. Final validation：PLV band specificity

前一階段的 recurring SFS features 多集中於 alpha，因此最後一輪固定比較六個單一 band。每個實驗只有 21 個 PLV pair features，仍使用相同 inter-subject LOPO / inner group-aware SFS / tuning。

### Rest1 vs Type1

| PLV band | KFDA | LDA | RBF-SVM | 三模型平均 |
|---|---:|---:|---:|---:|
| **Alpha** | **0.733** | **0.743** | **0.747** | **0.741** |
| Gamma | 0.727 | 0.653 | 0.680 | 0.687 |
| Beta-high | 0.637 | 0.620 | 0.653 | 0.637 |
| Delta | 0.593 | 0.570 | 0.613 | 0.592 |
| Beta-low | 0.583 | 0.590 | 0.580 | 0.584 |
| Theta | 0.547 | 0.557 | 0.543 | 0.549 |

All-band PLV-only reference：

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

Alpha-only 在三種 classifier 都高於 Poster baseline，也高於 all-band PLV 的 mean accuracy。相較 all-band PLV 的 paired mean delta：

```text
KFDA    +0.027
LDA     +0.077
RBF-SVM +0.057
```

三種 classifier 的 alpha-only 結果彼此接近，held-out-subject 間的 standard deviation 也比 all-band PLV 小。這與前面 SFS 對 alpha PLV 的 recurring selection 相互一致。Gamma 是第二高的單一 band，但 LDA / SVM 的 mean 較 alpha 低，且 KFDA / SVM 的跨受試者變異較大。

### Rest2 vs Type2

Alpha-only 仍是六個 bands 中最高的一組：

```text
KFDA    0.570
LDA     0.600
RBF-SVM 0.603
三模型平均 0.591
```

Type2 的幅度低於 Type1，跨受試者結果整體仍較弱，因此報告中以 Type1 為主要 phase-connectivity observation，Type2 作為 secondary comparison。

## 7. Final validation：iPLV robustness

最後一輪使用：

```text
iPLV = |Im(mean(exp(j * Δphi)))|
```

此指標會壓低 exact 0-lag / π-lag phase locking。它用於檢查 PLV observation 對 zero-lag component 的依賴程度。

### Rest1 vs Type1

| Classifier | Poster baseline | PLV-only | iPLV-only |
|---|---:|---:|---:|
| KFDA | 0.533 | 0.707 | 0.473 |
| LDA | 0.523 | 0.667 | 0.447 |
| RBF-SVM | 0.533 | 0.690 | 0.440 |

相較 all-band PLV：

```text
KFDA    -0.233
LDA     -0.220
RBF-SVM -0.250
```

相較 Poster baseline：

```text
KFDA    -0.060
LDA     -0.077
RBF-SVM -0.093
```

PLV 的跨受試者增益沒有在 iPLV 中保留。這表示目前有效的 PLV discrimination 高度依賴 zero-lag 或接近 zero-lag 的 phase-locking component。這個 observation 同時有多種可能來源，包括 common reference、volume conduction、共同訊號來源，以及真正同步但接近零相位差的神經活動。現有 7-channel、5-subject 資料不能分離這些來源，因此 final report 應把 physiological-connectivity interpretation 保持在保守層級。

## 8. 最終可支持的工作結論

1. Poster/Lab BP+COH baseline 在 within-subject 有明顯可分性，cross-subject 泛化接近 chance。
2. Frontal BP asymmetry 在目前資料上沒有穩定跨受試者改善。
3. PLV 對 Rest1 vs Type1 的 inter-subject LOPO 提供明顯增益，且 improvement 在多數 held-out subjects 與三種 classifier 中方向一致。
4. PLV decomposition 顯示 PLV-only 已能保留主要增益；增加更多 feature families 沒有帶來一致的 cross-subject improvement。
5. Band-wise validation 顯示 alpha-only PLV 是目前最穩定、平均表現最高的單一頻帶，結果與先前 alpha feature recurrence 一致。
6. iPLV-only 回到約 chance 或低於 Poster baseline，顯示 PLV gain 依賴 zero-/near-zero-lag component。這限制了對 PLV 結果的生理 connectivity 解讀。
7. Type2 的 phase-connectivity improvement 較弱，主要 phase-connectivity finding 目前集中在 Rest1 vs Type1。

## 9. 研究限制

- 受試者只有 5 位，LOPO 外層只有 5 個 held-out units。
- E1、E2、decomposition 與 final validation 都在同一批資料上完成，沒有獨立 cohort。
- 多個 feature representations 已被探索，因此最高 accuracy 不能視為獨立 confirmatory estimate。
- PLV 受到 reference、volume conduction、common source 與 frontal artifact 的可能影響。
- iPLV 只提供一個 zero-lag-suppressing robustness check，不能完全排除 volume conduction 或 reference bias。
- Type2 behavioral accuracy 接近 ceiling，且 RT 較 Type1 快，任務負荷差異可能影響 EEG state separability。
- 第一階段只使用 7 frontal channels；結果不代表全頭皮 connectivity pattern。

## 10. 分析停止點

目前資料已完成：baseline replication、feature-family decomposition、asymmetry extension、PLV extension、PLV decomposition、band-wise validation 與 iPLV robustness check。

在沒有方法或程式錯誤的前提下，這批 5-subject 資料停止新增 feature family、classifier 與大量超參數搜尋。後續工作集中在：

```text
整理正式結果
→ 產生最終圖表
→ 統整研究流程與限制
→ 製作投影片 / 書面報告
```

若後續增加新的受試者或獨立 cohort，再用固定方法進行 confirmatory validation，會比繼續在目前 5 人資料上增加 exploratory variants 更有資訊價值。
