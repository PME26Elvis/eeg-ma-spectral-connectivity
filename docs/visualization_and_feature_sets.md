# BP-only / COH-only 與視覺化

本文件描述 feature extraction 完成後的延伸分析。`data/features/features_all.csv` 已包含未標準化的 42 BP + 126 COH features，因此 **BP-only / COH-only 不需要重新讀 raw EEG，也不需要重新抽 feature**。

## 1. Feature-set comparison

執行：

```bash
python scripts/06_compare_feature_sets.py
```

此腳本會沿用 `configs/lab_replication.json` 的：

- 5 位受試者
- Rest1 vs Type1、Rest2 vs Type2
- SFS
- LDA
- RBF-SVM + inner grid search
- KFDA + inner grid search
- intra-subject outer 5-fold CV
- inter-subject outer LOPO-CV
- training-fold-only normalization / feature selection / tuning

唯一改變的是 `feature_set`：

```text
BP only  = 42 features
COH only = 126 features
BP+COH   = 168 features（既有 canonical results）
```

BP-only / COH-only 結果分別寫入：

```text
results/feature_sets/bp/
├─ config_snapshot.json
├─ intra/
└─ inter/

results/feature_sets/coh/
├─ config_snapshot.json
├─ intra/
└─ inter/
```

三種 feature set 的整合表：

```text
results/feature_set_comparison.csv
```

`config_snapshot.json` 會保存實際執行的完整設定，避免之後忘記某次結果是用哪一個 feature set。

> 正式 RBF grid 與 SFS 會重跑，因此這一步可能花較久。這是正常的；不要用 smoke config 的結果當正式報告數字。

## 2. 圖表

BP-only / COH-only 完成後執行：

```bash
python scripts/07_make_figures.py
```

預設輸出到：

```text
results/figures/
```

每張圖同時輸出：

- PNG：方便直接放投影片 / Word
- SVG：方便後續向量編輯或排版

### 2.1 BP+COH classification overview

```text
classification_all_intra.png/.svg
classification_all_inter.png/.svg
```

- intra：5 位受試者的 participant-level accuracy mean ± SD
- inter：5 個 held-out participants 的 LOPO accuracy mean ± SD

### 2.2 BP / COH / BP+COH 比較

每個 comparison 與 validation scope 分開畫：

```text
feature_sets_intra_rest1_vs_type1.*
feature_sets_intra_rest2_vs_type2.*
feature_sets_inter_rest1_vs_type1.*
feature_sets_inter_rest2_vs_type2.*
```

每張圖比較：

- BP only
- COH only
- BP+COH

以及三種 classifier：

- LDA
- RBF-SVM
- KFDA

這類圖最接近實驗室海報中「不同 feature 類型 + classifier performance」的呈現用途。

### 2.3 SFS top features

```text
sfs_top_intra_rest1_vs_type1.*
sfs_top_intra_rest2_vs_type2.*
sfs_top_inter_rest1_vs_type1.*
sfs_top_inter_rest2_vs_type2.*
```

- intra：先把各 subject 的 outer-fold selection count 加總，再排序
- inter：直接使用五個 outer LOPO folds 的 selection count

這些圖表示「feature 在 SFS 中被重複選到的頻率」，不是 feature effect size，也不能直接解讀成因果重要性。

## 3. Connectivity scalp/network 圖

輸出：

```text
connectivity_inter_rest1_vs_type1.*
connectivity_inter_rest2_vs_type2.*
```

圖只使用 `inter_feature_frequency.csv` 中的 COH features，並以 7 個 frontal electrodes：

```text
FP1, FP2, F7, F3, Fz, F4, F8
```

畫成固定的 frontal 10-20 schematic positions。

- node = electrode
- edge = 被 SFS 選到的 COH channel pair
- edge width = selection count
- edge label = frequency band × selection count

此圖是 **connectivity network schematic**，不是由 scalp voltage interpolation 產生的傳統 topomap。因本研究的主要 connectivity feature 是 pair-wise coherence，用 network edge 的方式更忠實；若後續報告需要真正的 BP topomap，可再另外新增 channel-wise BP change / statistic topography。

## 4. Behavioral 圖

輸出：

```text
behavioral_accuracy.*
behavioral_rt_median.*
```

- accuracy：Type1 / Type2 每位受試者
- RT：只使用回答正確 trial 的 median reaction time

RT 圖優先使用 median，是因為 unlimited-answer task 容易出現少數長 RT outliers；原始 mean 仍保留於 CSV，可在文字敘述時一起報告。

## 5. Top-N

預設 top 12 個 SFS features / COH connections：

```bash
python scripts/07_make_figures.py --top-n 12
```

若圖太擁擠，可改：

```bash
python scripts/07_make_figures.py --top-n 8
```

此參數只影響畫圖，不改變任何分類結果。

## 6. Git 管理

目前 repo 的策略：

- `data/raw/`：永遠不進 Git
- `data/features/`：進 Git
- `results/`：正式 canonical results 與 figures 可進 Git
- `results/scratch/`、`results/tmp/`：忽略

因此正式 BP-only / COH-only 結果與 figures 跑完並確認後，可以直接：

```bash
git add results/
git commit -m "Add BP COH comparison results and analysis figures"
git push
```

## 7. 解讀原則

本專題沒有要求分類率必須達到特定門檻。若 intra-subject 明顯高於 inter-subject，而 LOPO 接近 chance level，仍可合理解釋為：

- 個體內 EEG pattern 較穩定
- participant-to-participant domain shift 明顯
- 目前只有 5 位受試者，跨人模型泛化資料量有限

因此不應為了提高分類率而事後反覆更動 preprocessing、band definition 或 CV 規則。正式報告應優先保留事先定義的 pipeline，再如實討論結果與限制。
