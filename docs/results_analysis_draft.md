# 成果分析草稿（持續更新）

> 本文件是後續投影片 / 報告用的工作草稿。目的不是提早下定論，而是把每一階段可重現、值得保留的結果與目前解讀記錄下來。正式報告時應再依完整實驗結果、圖表與樣本數限制重新整理。

## 1. Poster/Lab baseline

Baseline 維持學長海報 / 實驗室教材的核心分析流程：

```text
2–50 Hz preprocessing
→ 5-s epochs
→ 42 BP + 126 COH = 168 features
→ training-only SFS
→ LDA / RBF-SVM / KFDA
→ intra-subject 5-fold CV
→ inter-subject LOPO-CV
```

目前 5 位受試者、每位兩個 comparison，各 comparison 30 Rest + 30 Mental Arithmetic，共 600 epochs。

Baseline inter-subject LOPO 大致落在 chance level 附近：

| Comparison | KFDA | LDA | RBF-SVM |
|---|---:|---:|---:|
| Rest1 vs Type1 | 0.533 | 0.523 | 0.533 |
| Rest2 vs Type2 | 0.490 | 0.527 | 0.517 |

Baseline intra-subject 則明顯較高：

| Comparison | KFDA | LDA | RBF-SVM |
|---|---:|---:|---:|
| Rest1 vs Type1 | 0.823 | 0.803 | 0.817 |
| Rest2 vs Type2 | 0.717 | 0.710 | 0.713 |

工作解讀：目前資料顯示 within-subject 可分性明顯高於 cross-subject 泛化；因此後續 extension 優先探索能否降低 subject-specific variability，而不是單純調 classifier 超參數。

---

## 2. E1：Frontal Hemispheric Asymmetry

E1 新增 3 組 frontal homologous pairs：

```text
FP1 ↔ FP2
F3  ↔ F4
F7  ↔ F8
```

每組 6 bands，預設採：

```text
log(BP_right) - log(BP_left)
```

共 18 asymmetry features；正式 E1 為：

```text
Baseline BP + COH + Asymmetry
168 + 18 = 186 features
```

### E1 結果

E1 沒有顯示穩定的 inter-subject 改善。

Rest1 vs Type1 的 inter LOPO 相較 baseline：

- KFDA：Δ = -0.013
- LDA：Δ = -0.007
- RBF-SVM：Δ = -0.023

Rest2 vs Type2：

- KFDA：Δ = +0.010
- LDA：Δ = +0.020
- RBF-SVM：Δ = +0.010

Intra-subject 亦僅有小幅正負變化。

目前工作結論：**在這 5 位受試者上，單純加入 frontal BP hemispheric asymmetry 沒有提供穩定的跨受試者增益。** 這不代表 asymmetry 在一般 EEG 任務無效，只代表此資料、此 electrode layout、此 task / validation setting 下沒有清楚改善。

---

## 3. E2：Phase Locking Value (PLV)

E2 新增：

```text
21 channel pairs × 6 frequency bands = 126 PLV features
```

定義：

```text
PLV = |mean(exp(j * (phase_a - phase_b)))|
```

每個 band 在 continuous preprocessed EEG 上額外 band-pass，Hilbert transform 取得 analytic phase，再於正式 5-s epoch 計算 PLV。

正式 E2：

```text
Baseline BP + COH + PLV
168 + 126 = 294 features
```

### 3.1 Feature extraction sanity check

5 subjects / 600 epochs：

```text
asymmetry_features = 18
plv_features       = 126
NaN                = 0
PLV min             = 0.005693
PLV max             = 0.999230
```

PLV distribution：

| Quantile | PLV |
|---:|---:|
| 0% | 0.005693 |
| 1% | 0.176134 |
| 5% | 0.358073 |
| 25% | 0.604993 |
| 50% | 0.734884 |
| 75% | 0.832669 |
| 90% | 0.897433 |
| 95% | 0.928573 |
| 99% | 0.970557 |
| 100% | 0.999230 |

High-PLV fractions：

```text
PLV > 0.90 : 9.52%
PLV > 0.95 : 2.60%
PLV > 0.99 : 0.176%
```

目前沒有看到「大量 feature 飽和在 1」的明顯異常；少量非常高的 PLV 仍需在最終討論中注意 common reference / volume conduction / zero-lag synchronization 等可能性。

### 3.2 E2 classification result

#### Inter-subject: Rest1 vs Type1

| Classifier | Baseline | Baseline + PLV | Δ accuracy |
|---|---:|---:|---:|
| KFDA | 0.533 | **0.667** | **+0.133** |
| LDA | 0.523 | **0.687** | **+0.163** |
| RBF-SVM | 0.533 | **0.660** | **+0.127** |

三種 classifier 都出現一致方向、幅度明顯的提升。

#### Inter-subject: Rest2 vs Type2

| Classifier | Baseline | Baseline + PLV | Δ accuracy |
|---|---:|---:|---:|
| KFDA | 0.490 | 0.493 | +0.003 |
| LDA | 0.527 | 0.527 | 0.000 |
| RBF-SVM | 0.517 | 0.517 | 0.000 |

Type2 comparison 幾乎沒有改善。

### 3.3 Per-subject inter-subject delta：Rest1 vs Type1

PLV 的 Type1 improvement 並非單一受試者拉高平均值。

| Held-out subject | KFDA Δ | LDA Δ | RBF-SVM Δ |
|---|---:|---:|---:|
| irene | +0.217 | +0.233 | +0.200 |
| jh | -0.017 | 0.000 | -0.017 |
| lin | +0.117 | +0.117 | +0.117 |
| pftest | +0.133 | +0.183 | +0.133 |
| yen | +0.217 | +0.283 | +0.200 |

因此目前是 **4/5 位 held-out subjects 明顯改善，jh 約持平 / 微幅下降**。這比只看 group mean 更支持「PLV 可能增加跨受試者共通資訊」的工作假說。

但樣本數仍只有 5 人，且已探索多種 feature sets，因此目前只應描述為明顯、跨多數受試者一致的 exploratory trend，不宣稱統計顯著或普遍成立。

### 3.4 SFS feature frequency

Rest1 vs Type1 的 inter-subject SFS 中，最常被選到的 features 已由 PLV 主導，且主要集中在 alpha band：

| Feature | Selection count / 5 LOPO folds |
|---|---:|
| PLV__Fz__F8__alpha | **4** |
| PLV__F3__F4__alpha | **3** |
| PLV__FP2__F7__alpha | **3** |
| PLV__F3__Fz__alpha | **2** |

其他 PLV features 多為 1 次。

Rest2 vs Type2 中，PLV 沒有形成穩定 feature selection：

```text
PLV__F3__Fz__delta       1
PLV__F4__F8__alpha       1
PLV__F7__F3__alpha       1
PLV__FP2__F3__beta_low   1
```

這與 classification 結果一致：**Type1 的 PLV extension 有穩定 signal，Type2 沒有。**

目前最值得後續追蹤的是 alpha-band phase synchrony，尤其：

```text
Fz–F8
F3–F4
FP2–F7
F3–Fz
```

這些 feature 目前只能視為 SFS stability / hypothesis-generating 結果；5-fold LOPO selection count 不足以支持生理因果解釋。

---

## 4. 目前階段性解讀

截至 E1 / E2，可形成以下工作敘事：

1. Poster/Lab baseline 的 intra-subject classification 明顯高於 inter-subject，顯示 subject variability 是主要瓶頸之一。
2. 加入 frontal hemispheric BP asymmetry（E1）沒有穩定提升跨人表現。
3. 加入 PLV（E2）對 Rest1 vs Type1 的 inter-subject LOPO 有明顯改善，三種 classifier 同方向，且 4/5 held-out subjects 有明顯提升。
4. 同一 PLV extension 對 Rest2 vs Type2 幾乎無改善，因此不是一般性的 feature-count 增加效果。
5. Type1 的 SFS recurring features 主要是 alpha-band PLV，支持「phase synchrony 可能提供原 BP / COH 未捕捉到的跨受試者資訊」這個後續假說。

注意：上述是探索性結果，不應在最終報告中表述成 PLV 已被證明優於 COH。下一階段必須拆解 PLV 本身與 BP/COH 的互補關係。

---

## 5. 下一階段實驗優先序

為了回答 E2 improvement 的來源，下一批最有資訊量的實驗建議為：

### E2a — PLV only

```text
126 PLV
```

回答：PLV 自己是否已足以提供 Type1 inter-subject discrimination？

### E2b — BP + PLV

```text
42 BP + 126 PLV = 168
```

回答：COH 是否其實不是 E2 improvement 必要成分？

### E2c — COH + PLV

```text
126 COH + 126 PLV = 252
```

回答：兩種 connectivity representation 是否互補？

### E3 — Baseline + Asymmetry + PLV

```text
42 BP + 126 COH + 18 Asymmetry + 126 PLV = 312
```

優先度低於 E2a / E2b / E2c。E1 已顯示 asymmetry 沒有穩定單獨增益，因此 combined 主要用來確認 SFS 是否會自動忽略 asymmetry，而不是期待它一定再提高 accuracy。

後續若 PLV 結果仍穩定，再考慮 subject calibration / Euclidean Alignment。這類方法會改變「held-out subject 是否允許提供 unlabeled / baseline calibration data」的研究問題，因此不應與目前 strict LOPO baseline 混在一起。

---

## 6. Reproducibility / reporting policy

- 原始 raw EEG 永遠不進 Git。
- 可版本化：extracted feature tables、正式 result CSV、figure、config snapshot。
- 每一批 extension 應保存：feature set 定義、CV / seed、per-unit delta、summary、SFS frequency。
- 不因 accuracy 不理想而事後更改 baseline preprocessing / validation 定義。
- 最終投影片 / 報告應同時呈現改善與沒有改善的實驗（例如 E1），避免只挑最高 accuracy 結果。
- `n=5` 是目前最重要限制之一；平均 accuracy 應搭配 per-subject / held-out-subject 結果一起呈現。

---

## 7. PLV decomposition：E2a / E2b / E2c / E2d / E3

為了拆解 E2 的改善來源，固定原本的 preprocessing、nested SFS、intra 5-fold、inter LOPO、classifier/grid 與 random state，只改候選 feature families。

| Experiment | Feature set | Candidate features |
|---|---|---:|
| E2a | PLV only | 126 |
| E2b | BP + PLV | 168 |
| E2c | COH + PLV | 252 |
| E2d | Asymmetry + PLV | 144 |
| E3 | BP + COH + Asymmetry + PLV | 312 |
| E2 full reference | BP + COH + PLV | 294 |

### 7.1 Inter-subject：Rest1 vs Type1

| Experiment | KFDA | LDA | RBF-SVM |
|---|---:|---:|---:|
| Poster baseline | 0.533 | 0.523 | 0.533 |
| E2 full: BP+COH+PLV | 0.667 | 0.687 | 0.660 |
| **E2a: PLV only** | **0.707** | 0.667 | **0.690** |
| E2b: BP+PLV | 0.687 | 0.640 | 0.650 |
| E2c: COH+PLV | 0.663 | 0.683 | 0.647 |
| E2d: ASYM+PLV | 0.673 | **0.703** | 0.680 |
| E3: all extensions | 0.620 | 0.647 | 0.610 |

E2a 相較 Poster baseline 的 mean gain：

```text
KFDA    +0.173
LDA     +0.143
RBF-SVM +0.157
```

而 E2a 相較完整 E2（BP+COH+PLV）：

```text
KFDA    +0.040
LDA     -0.020
RBF-SVM +0.030
```

因此 **PLV-only 已幾乎完整保留 E2 的跨受試者增益，甚至在 KFDA / RBF-SVM 上略高於完整 E2**。這是目前最重要的 decomposition 結果：Type1 inter-subject improvement 的主要來源看起來是 PLV 本身，而不是單純增加 feature 數量或必須依賴 BP / COH 才成立。

E2b（BP+PLV）沒有穩定優於 PLV-only，因此 BP 在這個 cross-subject setting 下不是必要成分。E2c（COH+PLV）平均 accuracy 與完整 E2 很接近，但沒有超越 PLV-only；不過它的 inter-subject standard deviation 明顯較低（KFDA/LDA/SVM 約 0.042/0.057/0.036），相較 PLV-only 約 0.128/0.116/0.126，可能表示 COH 與 PLV 的組合在不同 held-out subjects 間較穩定。這個「mean accuracy vs between-subject stability」的差異值得在後續投影片保留，但 n=5 下不能過度解讀。

E2d（ASYM+PLV）在 LDA 得到 0.703，為該 classifier 的最高值，但 KFDA / RBF-SVM 並沒有相同優勢，因此目前不應宣稱 asymmetry 與 PLV 有穩定互補作用。

E3 將所有 feature families 一次加入後，inter-subject Type1 反而降至約 0.61–0.65，低於 E2 full 與 PLV-only。這說明「更多 candidate features」不等於更好的跨人泛化；在只有 5 位受試者時，額外 subject-specific / redundant candidates 可能增加 feature-selection instability 或 overfitting 風險。

### 7.2 Inter-subject：Rest2 vs Type2

所有 decomposition 仍大致落在 chance 附近。PLV-only 約：

```text
KFDA    0.530
LDA     0.570
RBF-SVM 0.557
```

雖略高於部分 baseline 數值，但 standard deviation 大，其他 feature combinations 也沒有一致提升，因此目前仍沒有足夠證據支持 Type2 cross-subject improvement。這延續先前 E2 結果：PLV 的明顯效果主要是 Type1-specific，而不是任何 Rest-vs-MA comparison 都會提高。

### 7.3 Intra-subject 與 inter-subject 的對比

Type1 intra-subject 中，E3（全部 features）反而是表現最好的組合之一：

```text
KFDA    0.857
LDA     0.860
RBF-SVM 0.880
```

但同一 E3 在 inter-subject 明顯低於 PLV-only。

這形成一個很重要的工作解讀：**BP / COH / asymmetry 等額外 feature family 可能含有對單一受試者很有辨識力、但跨受試者不穩定的資訊；PLV 相對更有機會捕捉跨人共通的 task-related representation。** 這個結論仍是 exploratory，但它能同時解釋「更多 features 幫助 intra、卻傷害 inter」的現象，比單純追求最高 accuracy 更有研究價值。

### 7.4 SFS feature-family usage

Rest1 vs Type1 的 inter LOPO 中，只要 PLV 存在，它在 **5/5 outer folds 都被 SFS 選到**：

| Experiment | Family | Selection occurrences | Fold presence |
|---|---|---:|---:|
| E2 full | PLV | 23 | 5/5 |
| E2 full | COH | 16 | 5/5 |
| E2 full | BP | 3 | 2/5 |
| E2a | PLV | 37 | 5/5 |
| E2b | PLV | 32 | 5/5 |
| E2b | BP | 4 | 3/5 |
| E2c | PLV | 20 | 5/5 |
| E2c | COH | 15 | 5/5 |
| E2d | PLV | 34 | 5/5 |
| E2d | ASYM | 3 | 3/5 |
| E3 | PLV | 19 | 5/5 |
| E3 | COH | 12 | 5/5 |
| E3 | ASYM | 2 | 2/5 |
| E3 | BP | 1 | 1/5 |

這再次支持 PLV 是目前最穩定的 feature family。BP 與 ASYM 在 PLV 存在時只偶爾被選；COH 在有提供時仍會出現在 5/5 folds，表示它可能含有補充資訊，但目前沒有帶來一致的 mean-accuracy 增益。

### 7.5 目前較強的工作結論

截至 PLV decomposition，較合理的研究敘事更新為：

1. Poster/Lab baseline 的 cross-subject performance 接近 chance，但 within-subject classification 明顯較高。
2. E1 hemispheric asymmetry 本身沒有穩定改善 inter-subject performance。
3. 加入 PLV 後，Type1 inter-subject accuracy 有明顯、跨 classifier 的提升。
4. Decomposition 顯示 **PLV-only 已足以重現甚至略超過完整 E2 的 Type1 cross-subject performance**，因此 PLV 很可能是主要 improvement source。
5. BP / asymmetry 並不是此 improvement 的必要成分；COH 可能降低 held-out-subject 間的 variance，但沒有明確提高 mean accuracy。
6. 全部 feature families 對 intra-subject 可以更好，對 inter-subject 卻更差，支持「額外 features 含較多 subject-specific discrimination，而 PLV 較可能保留 cross-subject invariant information」的工作假說。
7. Type2 仍沒有類似 cross-subject gain，因此上述現象目前只能描述為 Type1-specific。

以上仍屬同一批 5 subjects 上的 post-hoc exploratory analyses；不能把 decomposition 中最高的 0.70 左右 accuracy 當成獨立 confirmatory validation，也不應從多個 feature sets 中只挑最高數字而忽略完整 experiment matrix。

### 7.6 下一階段建議

Feature-family decomposition 已經回答最核心的「PLV improvement 從哪裡來」問題。下一階段優先度建議改為：

1. **Band-wise PLV ablation**：預先固定六個既有 bands，各自只用 21 個 PLV pair features，檢查 alpha 是否真的承擔主要 Type1 cross-subject signal，而不是只因 SFS 偶然偏好 alpha。
2. **Zero-lag-robust phase-connectivity robustness check**：新增例如 PLI / imaginary-PLV / wPLI 類指標，確認 PLV gain 是否仍存在於較不受 common reference / volume conduction / zero-lag synchronization 影響的表示法中。
3. 上述 robustness 若仍支持 phase connectivity，再考慮 **subject calibration / Euclidean Alignment**。這會改變 validation regime，必須獨立標示為 calibrated / domain-adapted setting，不和目前 strict zero-calibration LOPO 混為同一結果。

這個優先序比繼續任意組合更多 feature families 更有科學資訊量。
