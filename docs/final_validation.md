# 最後一輪方法驗證：PLV band specificity + iPLV robustness

本文件定義本專題在同一批 5 位受試者資料上的**最後一輪 feature-engineering / representation 實驗**。

這一輪不是繼續追逐最高 accuracy，而是回答前一階段 PLV 結果留下的兩個方法學問題。完成後，除非發現程式或方法定義錯誤，**停止在相同 5 位受試者上新增 feature family / classifier 組合**，轉入結果統整、圖表、投影片與報告撰寫。

---

## 1. 為什麼還需要最後一輪

目前已完成：

1. Poster/Lab baseline：BP + COH。
2. BP-only / COH-only decomposition。
3. E1：frontal hemispheric asymmetry。
4. E2：新增 PLV。
5. PLV decomposition：PLV-only、BP+PLV、COH+PLV、ASYM+PLV、all extensions。

主要觀察為：

- Poster baseline 的 inter-subject LOPO 接近 chance。
- E1 asymmetry 沒有穩定改善。
- E2 在 `Rest1 vs Type1` 顯示明顯跨受試者改善。
- PLV-only 幾乎保留甚至略高於完整 E2 的效果。
- SFS recurring PLV features 多集中於 alpha band。
- 同一 PLV extension 對 `Rest2 vs Type2` 沒有相同程度改善。

因此此時再排列更多 BP / COH / asymmetry 組合的資訊價值已經很低。真正尚未回答的是：

1. **band specificity**：PLV effect 是否主要由 alpha band 承擔？
2. **zero-lag robustness**：PLV effect 是否可能主要來自 zero-lag synchronization / common-source effect？

這兩個問題直接從前一輪結果推導，具有明確分析邏輯，因此保留為最後一輪。

---

## 2. 實驗 A：PLV band-wise ablation

現有 PLV：

```text
21 channel pairs × 6 bands = 126 features
```

最後一輪把六個 band 各自獨立送入相同 pipeline：

```text
PLV-delta only      = 21
PLV-theta only      = 21
PLV-alpha only      = 21
PLV-beta_low only   = 21
PLV-beta_high only  = 21
PLV-gamma only      = 21
```

每組都沿用：

```text
training-only SFS
→ LDA / RBF-SVM / KFDA
→ outer LOPO
→ inner Leave-One-Group-Out tuning / SFS
```

### 目的

不是從六個 band 中挑最高的當作「新最佳模型」，而是驗證前面 SFS 所形成的 hypothesis：

> Type1 跨受試者可分資訊是否真的主要集中於 alpha-band phase synchronization？

因此六個 band 必須全部跑，避免只針對 alpha 做事後確認造成更嚴重的 cherry-picking。

### 解讀

- 若 alpha-only 明顯高於其他 bands，且接近 all-band PLV：支持 alpha specificity 的探索性證據。
- 若多個 bands 都類似：前面的 alpha SFS recurrence 可能只是 feature-selection instability / small-sample variation。
- 若 alpha-only 明顯低於 all-band PLV：表示跨 band 互補可能比單一 alpha 更重要。

這些都屬描述性 / hypothesis-generating 解讀，不宣稱正式統計顯著。

---

## 3. 實驗 B：iPLV-only robustness check

普通 PLV：

```text
PLV = |mean(exp(j * Δphi))|
```

會對接近 0° 的穩定 phase difference 給高值。因此 common reference、volume conduction、common source 等因素有可能貢獻高 PLV。

本輪新增：

```text
iPLV = |Im(mean(exp(j * Δphi)))|
```

其特性是：

- exact zero-lag phase locking → iPLV = 0
- exact pi-lag phase locking → iPLV = 0
- non-zero-lag stable phase relation 可保留

這裡的 iPLV 只作為 **zero-lag-suppressing robustness metric**。它不能消除所有 volume conduction / reference bias，也不應被描述成「真實 connectivity 的直接量測」。

### extraction

與既有 PLV 完全對齊：

```text
continuous raw
→ baseline 2–50 Hz preprocessing
→ 每個 frequency band 再做 continuous band-pass
→ Hilbert analytic phase
→ 正式 5-s epoch
→ 21 channel pairs × 6 bands
→ 126 iPLV features
```

因此 PLV / iPLV 的主要差別只在 phase-locking summary statistic。

### 解讀

- 若 iPLV-only 仍明顯高於 Poster baseline：PLV signal 不是完全依賴 exact zero-lag synchronization。
- 若 iPLV-only 接近 chance：PLV improvement 可能高度依賴 zero-lag component；報告需把 common-source / reference effect 視為主要限制。
- 若 iPLV 比 PLV 更好，也不將它當成新的 optimization target；只描述為 robustness observation。

---

## 4. 為什麼最後一輪只做 inter-subject

前一階段的核心研究問題已經從「能不能在同一人內分類」轉成：

> 為什麼 Poster baseline 跨人失敗，而 PLV 對 Type1 跨人改善？

Band specificity 與 iPLV 都是在驗證這個 cross-subject observation，因此最後一輪只跑 inter-subject LOPO。

這不是為了省算力而改 protocol，而是刻意避免再產生大量與主要問題無關的 secondary 結果。

---

## 5. 執行方式

先抽 iPLV：

```bash
python scripts/11_extract_iplv_features.py
```

預期：

```text
subjects=5
epochs=600
iplv_features=126
NaN=0
Inf=0
iPLV range=[0, 1] 內
```

接著跑最後矩陣：

```bash
python scripts/12_run_final_validation.py
```

中斷後可：

```bash
python scripts/12_run_final_validation.py --resume
```

正式矩陣固定為：

```text
6 × PLV single-band
+ 1 × iPLV-only
```

不再自動加入其他 feature permutations。

主要輸出：

```text
results/final_validation/
├── final_validation_summary.csv
├── delta_vs_poster_by_unit.csv
├── delta_vs_poster_summary.csv
├── delta_vs_plv_all_by_unit.csv
├── delta_vs_plv_all_summary.csv
├── plv_band_descriptive_ranking.csv
├── experiment_matrix_snapshot.json
├── plv_delta_only/
├── plv_theta_only/
├── plv_alpha_only/
├── plv_beta_low_only/
├── plv_beta_high_only/
├── plv_gamma_only/
└── iplv_only/
```

---

## 6. Stop rule

這一點是正式研究設計的一部分：

> **完成此 final-validation matrix 後，在沒有方法 bug 的前提下，不再針對同一批 5 subjects 新增 feature family、classifier 或大量參數搜尋。**

原因：

- `n=5` 是主要限制。
- 同一資料上持續嘗試方法會增加 researcher degrees of freedom。
- 專題的主要價值應是能清楚說明「問題 → 假說 → 實驗 → 結果 → 下一個問題」的分析框架，而不是在小資料上把 accuracy 最大化。

完成後工作轉為：

1. 將 final validation 結果補入 `docs/results_analysis_draft.md`。
2. 整理一份簡潔研究故事線。
3. 製作最終圖表。
4. 撰寫投影片 / 報告。

若未來增加新受試者，才適合重新進行 confirmatory validation 或 domain-alignment 類實驗。
