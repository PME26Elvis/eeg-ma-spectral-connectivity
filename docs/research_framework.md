# 專題研究分析框架：從 Poster baseline 到 phase-connectivity validation

這份文件的用途不是列出所有程式功能，而是把目前專題中每一次分析的**依據、推導邏輯、得到的結果、以及為什麼會進到下一步**串起來。後續投影片 / 口頭報告可以直接沿用這條故事線。

---

## 0. 研究起點：先重現既有實驗室流程

本專題不是從自由調參開始，而是先建立一個明確 baseline：

```text
2–50 Hz preprocessing
→ 5-s epochs
→ BP + COH
→ SFS
→ LDA / RBF-SVM / KFDA
→ intra-subject 5-fold
→ inter-subject LOPO
```

目的有兩個：

1. 讓結果可對照學長海報 / 實驗室教材。
2. 後續所有修改都有一個固定 reference，而不是看到 accuracy 不理想就一直改流程。

### Baseline observation

- intra-subject：明顯高於 chance。
- inter-subject：大多接近 chance。

因此第一個真正研究問題不是「classifier 不夠強嗎？」，而是：

> **同一人的 Rest / Mental Arithmetic 可以分，但跨人泛化失敗，是否代表 feature representation 太受個體差異影響？**

這決定了後續方向應優先改 representation，而不是只繼續加 classifier。

---

## 1. 先拆 baseline：BP-only / COH-only

在新增 feature 前，先把 baseline 的兩個主要 feature family 分開：

```text
BP only
COH only
BP + COH
```

### 邏輯

如果其中一種本身已經非常穩定，就沒有理由立刻增加更多 feature。

### Observation

- intra-subject 各 feature set 多數仍有一定可分性。
- inter-subject BP / COH / BP+COH 都沒有形成明顯穩定優勢，仍接近 chance。

### 推導

因此跨受試者問題不像只是「BP 或 COH 選錯一個」，而比較像既有表示仍保留太多 subject-specific information。

---

## 2. E1：Hemispheric Asymmetry

第一個 extension 使用 frontal electrode layout 天然存在的左右對稱：

```text
FP1 ↔ FP2
F3  ↔ F4
F7  ↔ F8
```

使用：

```text
log(BP_right) - log(BP_left)
```

### 假說

絕對 BP 在不同受試者間可能尺度差異很大；左右相對差異可能較具有 subject-invariant 性質。

### 結果

E1 沒有穩定改善 inter-subject classification；Type1 甚至略降，Type2 只有極小變化。

### 解讀

這是一個有價值的負結果：

> **在目前 7 frontal channels、這兩種 mental-arithmetic comparison 與 n=5 下，單純 frontal BP asymmetry 並沒有解決跨受試者泛化。**

因此不繼續把 asymmetry 當主要方向。

---

## 3. E2：Phase Locking Value (PLV)

第二個 extension 從另一個角度處理 subject variability：

- COH 同時受振幅與 phase relationship 影響。
- PLV 只描述 phase difference 的穩定程度。

因此加入：

```text
PLV = |mean(exp(j * Δphi))|
```

共：

```text
21 channel pairs × 6 bands = 126 PLV features
```

### 假說

若跨人的主要差異之一來自 amplitude scale，而 task-related phase synchrony 比較具有共通性，PLV 可能比 BP / COH 更適合 cross-subject representation。

### 結果

對 `Rest1 vs Type1`：

- Poster baseline 約 52–53%。
- BP+COH+PLV 約 66–69%。
- 三個 classifier 同方向改善。
- 4/5 held-out subjects 明顯改善，1 人約持平。

對 `Rest2 vs Type2`：

- 幾乎沒有相同提升。

### 推導

這排除了最簡單的「只要增加 feature 數量就會變好」解釋，因為同一 feature extension 並沒有讓 Type2 一起上升。

而 SFS 又反覆選到多個 alpha-band PLV feature，因此形成下一個假說：

> **Type1 的跨受試者訊號可能主要存在於 frontal phase synchronization，且 alpha band 特別值得注意。**

---

## 4. PLV decomposition：確認 improvement 到底從哪裡來

看到 E2 提升後，不能直接宣稱「PLV 很好」，因為 E2 是：

```text
BP + COH + PLV
```

所以需要拆解：

```text
PLV only
BP + PLV
COH + PLV
ASYM + PLV
BP + COH + ASYM + PLV
```

### 主要問題

1. PLV 本身是否已足夠？
2. BP / COH 是否與 PLV 有必要的互補？
3. feature 越多是否真的越好？

### 結果

`Rest1 vs Type1` inter-subject：

- PLV-only 約 67–71%。
- BP+PLV / COH+PLV / ASYM+PLV 大致沒有穩定超越 PLV-only。
- 全 312 features 的 E3 反而下降到約 61–65%。

同時 E3 的 intra-subject 表現仍很好。

### 目前最重要的解讀

這形成一個比「PLV accuracy 比較高」更完整的故事：

> **加入更多 subject-specific feature 可以維持甚至提高 within-subject discrimination，但未必改善 cross-subject generalization；相較之下，PLV-only 是更精簡、但跨人較有效的 representation。**

因此在目前資料上，PLV improvement 的主要來源看起來就是 phase-locking representation 本身，而不是依賴 BP / COH / asymmetry 的聯合堆疊。

注意：這仍是同一批 5 subjects 上的 exploratory decomposition，不是 independent confirmation。

---

## 5. 為什麼不繼續排列更多 feature combination

做到這裡後，繼續測：

```text
BP + alpha-PLV
COH + alpha-PLV
某兩個 bands + 某一種 classifier
更多 asymmetry variant
更多 kernel / C / gamma
...
```

雖然可能找到更高 accuracy，但研究價值會快速下降，而且增加 researcher degrees of freedom。

目前真正還有方法學意義、而且能直接從已有結果推導出的問題只剩兩個：

### A. Alpha specificity

SFS 多次選中 alpha PLV，所以需要用六個 band 全部做單獨 ablation，確認 alpha 是否真的特別，而不是事後只挑 alpha。

### B. Zero-lag robustness

PLV 可能受到 common reference / volume conduction / common-source zero-lag synchronization 影響。因此使用 iPLV 做一次 zero-lag-suppressing robustness check。

這兩個問題回答完後，就不再做新的 feature engineering。

---

## 6. Final validation 的角色

最後一輪固定為：

```text
PLV delta-only
PLV theta-only
PLV alpha-only
PLV beta-low-only
PLV beta-high-only
PLV gamma-only

iPLV-only
```

而且只跑 inter-subject LOPO，因為現在真正要驗證的就是 cross-subject PLV observation。

這一輪不是「最後再找一次最高 accuracy」，而是：

1. 驗證 band specificity。
2. 檢查 PLV 結果對 zero-lag suppression 是否 robust。
3. 給研究故事一個自然終點。

---

## 7. Stop rule 與最終報告邏輯

完成 final validation 後：

> **除非發現方法 / code bug，不再對同一批 5 subjects 做新的 feature family、classifier 或大量參數搜尋。**

最終報告應以這條邏輯呈現：

```text
先重現 Poster/Lab baseline
↓
發現 intra 好、inter 差
↓
推測 subject variability / representation 是瓶頸
↓
拆 BP / COH → 沒有解決
↓
E1 asymmetry → 沒有穩定改善
↓
E2 PLV → Type1 inter 明顯改善
↓
per-subject + SFS 檢查 → 改善不是單一 subject，且 alpha PLV recurring
↓
PLV decomposition → PLV-only 已足夠；feature 越多不等於跨人越好
↓
Final validation → alpha specificity + zero-lag robustness
↓
停止優化，統整限制與研究意義
```

### 應主動講出的限制

- 只有 5 位受試者。
- 所有 extension / decomposition 都在同一批資料上探索。
- 沒有獨立 cohort 做 confirmatory validation。
- PLV / iPLV 都不能直接等同生理 causal connectivity。
- common reference、volume conduction、眼動 / frontal artifact 仍是解讀限制。
- Type2 behavior 有明顯 ceiling effect，可能降低不同 state 之間的生理差異。

### 本專題真正的重點

不是宣稱找到一個 universal 70% EEG classifier，而是能完整說明：

> **如何從既有 baseline 出發，根據結果辨認瓶頸、提出有理由的 feature hypothesis、用 controlled ablation 拆解 improvement，再用最後的 robustness check 確認哪些結論可以講、哪些只能保守描述。**

這也是後續投影片最值得呈現的主軸。
