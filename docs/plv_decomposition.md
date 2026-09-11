# PLV decomposition：E2a / E2b / E2c / E3

本文件定義 E2 發現之後的下一批 feature-family decomposition。目的不是繼續無限制搜尋最高 accuracy，而是回答一個已經由 E2 結果提出的具體問題：**Rest1 vs Type1 的跨受試者改善究竟主要來自 PLV 本身，還是 PLV 與既有 BP / COH 的互補？**

Poster/Lab baseline 與 E1/E2 原結果全部保持不變。這批實驗沿用相同的 preprocessing、epochs、SFS、classifier、RBF grid、random state、intra 5-fold 與 inter LOPO；唯一改變的是 SFS 可見的 candidate feature families。

## 1. Frozen references

兩個 reference 不在本腳本重跑：

```text
poster_baseline         = BP + COH             = 168
E2 full baseline+PLV    = BP + COH + PLV       = 294
```

這使新的 ablation 直接與已經保存的 canonical result 做 paired comparison。

## 2. Primary experiments

### E2a — PLV only

```text
PLV = 126
```

回答：PLV 自己是否已足以產生 Type1 的 cross-subject discrimination？

若 E2a 接近完整 E2，代表 PLV 很可能是主要資訊來源；若明顯較差，表示 E2 improvement 需要其他 family 配合。

### E2b — BP + PLV

```text
BP 42 + PLV 126 = 168
```

回答：COH 是否為 E2 improvement 的必要成分？

此組特別有利於公平比較，因為總 candidate dimension 恰好與 Poster baseline 168 相同；若表現提升，不能單純歸因於 candidate features 數量增加。

### E2c — COH + PLV

```text
COH 126 + PLV 126 = 252
```

回答：兩種 connectivity representation 是否互補？COH 同時反映 cross-spectrum magnitude / phase relationship，PLV 則只看 phase-difference consistency。

### E3 — BP + COH + Asymmetry + PLV

```text
42 BP + 126 COH + 18 ASYM + 126 PLV = 312
```

E1 的 asymmetry 單獨加入 baseline 並沒有穩定 inter-subject 增益，因此 E3 主要不是期待再創新高，而是觀察 nested SFS 在 PLV 存在時是否會穩定保留任何 asymmetry feature。

## 3. Secondary exploratory experiment

### E2d — Asymmetry + PLV

```text
18 ASYM + 126 PLV = 144
```

這是次要實驗，不屬於 E2a/b/c 的核心 decomposition。它測試「relative power representation + phase synchrony」是否可在不使用 absolute BP / COH 的情況下工作。

因為 E1 已經是負結果，E2d 必須標示為 secondary / exploratory，不能在最後只因 accuracy 高就重新包裝成預先主要假設。

## 4. 執行

前提：本地已經有

```text
data/features/features_all.csv
data/features/extensions/features_extensions.csv
results/intra/...
results/inter/...
results/extensions/e2_plv/...
```

不需要重新跑 raw feature extraction。

只跑四個 primary experiments：

```bash
python scripts/10_run_plv_decomposition.py
```

算力與時間足夠時，可連 secondary E2d 一起跑：

```bash
python scripts/10_run_plv_decomposition.py --include-secondary
```

若長時間運算中斷，已完成 experiment 可以保留，續跑時使用：

```bash
python scripts/10_run_plv_decomposition.py --include-secondary --resume
```

也可以只指定特定 experiment：

```bash
python scripts/10_run_plv_decomposition.py \
  --experiments e2a_plv_only e2b_bp_plv
```

## 5. Output structure

```text
results/plv_decomposition/
├── experiment_matrix_snapshot.json
├── plv_decomposition_summary.csv
├── delta_vs_poster_by_unit.csv
├── delta_vs_poster_summary.csv
├── delta_vs_e2_full_by_unit.csv
├── delta_vs_e2_full_summary.csv
├── sfs_feature_family_usage.csv
├── e2a_plv_only/
│   ├── config_snapshot.json
│   ├── intra/
│   └── inter/
├── e2b_bp_plv/
├── e2c_coh_plv/
├── e3_all_extensions/
└── e2d_asym_plv/        # 只有 --include-secondary 才會產生
```

每個 experiment 的 `intra/` / `inter/` 都沿用原 evaluation pipeline，因此保留 folds、predictions、summary、SFS path 與 feature frequency。

## 6. Paired comparison

除了平均 accuracy，腳本會保存每一位 participant / held-out participant 的 paired delta。

### Against Poster baseline

```text
delta_vs_poster_by_unit.csv
delta_vs_poster_summary.csv
```

回答每個新 representation 相對原 168-feature baseline 改變多少。

### Against full E2

```text
delta_vs_e2_full_by_unit.csv
delta_vs_e2_full_summary.csv
```

回答刪掉 BP 或 COH 之後，是否仍能保留 E2 的增益。

這比單看 group mean 更重要，因為目前只有 5 位受試者。

## 7. SFS feature-family usage

腳本額外輸出：

```text
sfs_feature_family_usage.csv
```

包含：

- `selection_occurrences`：該 family 被 accepted feature 選中的總次數
- `fold_presence_count`：有至少一個該 family feature 被選中的 outer units 數
- `n_outer_units`
- `fold_presence_fraction`

對 inter-subject 每個 comparison 而言，`n_outer_units=5`。例如 E3 中若 ASYM 的 `fold_presence_count=0/5`，代表 SFS 在 PLV 存在時完全沒有保留 asymmetry；若 PLV 是 5/5 而 BP/COH 只偶爾出現，則支持 PLV 是較穩定的 cross-subject information source。

## 8. Study hierarchy / interpretation rule

本階段預先固定：

```text
Primary: E2a, E2b, E2c, E3
Secondary: E2d
```

不要因結果出來後再改哪一組叫「主要實驗」。

Rest1 vs Type1 是由 E2 結果提出 PLV hypothesis 的主要 follow-up comparison；Rest2 vs Type2 仍完整跑、完整報告，作為 task-specificity 對照。

因為 PLV hypothesis 是看過 E2 結果後才形成，這批仍屬 **exploratory follow-up on the same five subjects**，不是獨立 confirmatory validation。最終報告應明確區分：

1. Poster/Lab baseline replication
2. E1/E2 exploratory extensions
3. E2a/b/c/E3 post-hoc decomposition of the observed PLV effect

真正的 confirmatory evidence 需要新的受試者或獨立資料。

## 9. 下一步決策邏輯

跑完後優先回答：

1. **PLV-only 是否接近 full E2？**
2. **BP+PLV 或 COH+PLV 哪個更接近 / 超過 full E2？**
3. **Type1 improvement 是否仍跨多數 held-out subjects？**
4. **SFS family usage 是否穩定由 PLV 主導？**
5. **Type2 是否仍沒有同樣改善？**
6. **E3 中 ASYM 是否實際被 SFS 使用？**

若 PLV 結果經這些 ablations 仍穩定，再考慮 subject calibration / Euclidean Alignment。那會改變 held-out subject 可提供多少 calibration data 的研究問題，因此應另開一個 domain-alignment 階段，而不是混進目前 strict LOPO。
