# 最終投影片建議架構

本文件把目前分析整理成可直接用於口頭報告的順序。建議主報告控制在 8–10 張內容投影片；完整數值表與額外圖表放 appendix。

## Slide 1 — 實驗資料與問題設定

說明：

- 5 位受試者。
- HNC 8-channel recording；第一階段分析使用 7 frontal channels。
- Rest1 vs Type1、Rest2 vs Type2。
- 每位受試者每個 comparison：30 Rest + 30 correct Mental Arithmetic epochs。
- 每個 epoch 5 s。

圖：簡單 protocol flow 或 electrode schematic。

## Slide 2 — Poster / Lab baseline

流程：

```text
2–50 Hz
→ 5-s epoch
→ BP + COH
→ SFS
→ LDA / RBF-SVM / KFDA
→ intra 5-fold / inter LOPO
```

數值重點：

- Type1 intra 約 0.80–0.82。
- Type1 inter 約 0.52–0.53。
- Type2 intra 約 0.71。
- Type2 inter 約 0.49–0.53。

這一頁建立後續研究問題：跨受試者泛化明顯弱於受試者內分類。

建議圖：既有 `classification_all_intra` + `classification_all_inter`，或整理成單一簡化圖。

## Slide 3 — Baseline feature family 與 E1

依序說明：

1. BP-only / COH-only 沒有解決 inter-subject 問題。
2. 因 frontal montage 有自然左右配對，測試 hemispheric asymmetry。
3. E1 asymmetry 的 inter-subject improvement 很小或略為下降。

這一頁可以用一個小表格呈現，不需要放太多圖。

## Slide 4 — E2：加入 PLV

動機：

- BP 著重頻帶能量。
- COH 同時包含振幅與 phase relation。
- PLV 直接描述 phase difference 的穩定程度。

Type1 inter-subject：

```text
Poster baseline ≈ 0.52–0.53
BP+COH+PLV      ≈ 0.66–0.69
```

補充：4/5 held-out subjects 有明顯改善，三種 classifier 同方向。

建議圖：`final_story_type1_inter`。

## Slide 5 — PLV decomposition

比較：

```text
PLV only
BP + PLV
COH + PLV
ASYM + PLV
All extensions
```

關鍵 observation：

- PLV-only 已保留主要 cross-subject improvement。
- All-extensions 對 intra 很好，對 inter 反而下降。
- SFS 在含 PLV 的 Type1 experiments 中每個 LOPO fold 都有選到 PLV family。

這一頁解釋 feature representation 與跨人泛化之間的關係。

## Slide 6 — Band-wise final validation

六個單一 PLV band 都跑相同 LOPO pipeline。

Type1 三模型平均：

```text
alpha      0.741
gamma      0.687
beta-high  0.637
delta      0.592
beta-low   0.584
theta      0.549
```

Alpha-only：

```text
KFDA    0.733 ± 0.081
LDA     0.743 ± 0.065
RBF-SVM 0.747 ± 0.069
```

建議圖：`final_plv_band_comparison_type1`。

如果版面允許，可在旁邊放 `final_alpha_plv_sfs_network`，表示 SFS 常用的 frontal alpha connections。網路圖只代表 feature-selection frequency，不代表因果 connectivity。

## Slide 7 — Zero-lag robustness：iPLV

Type1：

| Representation | KFDA | LDA | SVM |
|---|---:|---:|---:|
| Poster baseline | .533 | .523 | .533 |
| PLV all bands | .707 | .667 | .690 |
| Alpha PLV | .733 | .743 | .747 |
| iPLV | .473 | .447 | .440 |

建議圖：`final_phase_connectivity_validation`。

解讀：PLV 的 classification gain 高度依賴 zero- / near-zero-lag phase-locking component。可能來源包含 common reference、volume conduction、shared signal source、同步神經活動與 frontal artifact。現有資料無法進一步拆分來源，因此 connectivity 的生理解讀維持保守。

## Slide 8 — Type1 / Type2 對照與 behavioral context

Type2 behavioral accuracy 接近 ceiling、reaction time 較快。Type2 的 phase-connectivity cross-subject improvement 也較弱。

這一頁用來說明兩種 arithmetic task 的結果不同，並把 behavioral observation 與 EEG 分類結果放在同一個背景下。

建議圖：既有 `behavioral_accuracy`、`behavioral_rt_median`；若時間有限可只留 RT 圖加簡表。

## Slide 9 — 研究結論

可用四點：

1. Poster BP+COH pipeline 在受試者內可分，跨受試者泛化弱。
2. Asymmetry 沒有提供穩定 improvement；PLV 對 Type1 LOPO 有明顯增益。
3. Decomposition 與 band-wise validation 指向 alpha-band PLV 為目前最穩定的 representation。
4. iPLV robustness 顯示 PLV gain 依賴 zero-/near-zero-lag component，因此目前結果適合作為 classification feature observation，functional-connectivity 生理解讀需要保守。

## Slide 10 — 限制與後續工作

限制：

- n=5。
- 所有 extensions 來自同一批資料。
- 沒有 independent cohort。
- 7 frontal channels 空間資訊有限。
- phase metrics 受 reference / volume conduction / common source 影響。

後續工作：

- 增加受試者數量。
- 固定目前方法後做 independent / confirmatory validation。
- 有更多資料後再評估 domain alignment、更多 channels 或更完整的 connectivity analysis。

## Appendix 建議

可放：

- 完整 BP-only / COH-only table。
- E1 / E2 per-subject delta。
- PLV distribution sanity check。
- PLV decomposition 全矩陣。
- Type2 final validation 全表。
- SFS feature-frequency tables。
- preprocessing / COH / PLV / iPLV 公式與 implementation details。

這樣主報告保留研究推導，appendix 保留可追溯的技術細節。
