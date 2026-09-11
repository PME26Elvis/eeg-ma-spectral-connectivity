# eeg-ma-spectral-connectivity

HNC Mental Arithmetic EEG 的可重現分析 repo。README 只描述資料、前處理、feature extraction、分類驗證與視覺化流程，不包含研究動機。

## 分析範圍

固定處理兩個二分類：

1. `Rest1 vs Type1`
2. `Rest2 vs Type2`

第一階段 replication analysis 使用 7 個 frontal channels：

```text
FP1, FP2, F7, F3, Fz, F4, F8
```

`Cz` 保留於 raw EEG，但不進 168-feature pipeline。

整體流程：

```text
HNC v0.15 raw session
→ raw / protocol validation
→ continuous 2–50 Hz band-pass
→ 5-s epoching
→ 42 BP + 126 COH = 168 features
→ SFS
→ LDA / RBF-SVM / KFDA
→ intra-subject 5-fold CV
→ inter-subject LOPO-CV
→ BP-only / COH-only comparison
→ classification / SFS / connectivity / behavioral figures
```

## 1. Git 與資料目錄

Raw EEG **不進 Git**：

```text
data/raw/
├─ subject01/
│  ├─ eeg_raw.csv
│  ├─ events.csv
│  ├─ trials.csv
│  ├─ rest_epochs.csv
│  └─ session_info.txt
├─ subject02/
│  └─ ...
...
└─ subject05/
   └─ ...
```

程式會遞迴尋找 `eeg_raw.csv`，資料夾名稱不需固定。

Git 策略：

- `data/raw/`：忽略
- `data/features/`：版本控制
- `results/`：正式 canonical results 與 figures 版本控制
- `results/scratch/`、`results/tmp/`：忽略

## 2. 安裝

Python 3.10+：

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e ".[dev]"
pytest
```

Windows PowerShell 啟用環境可改為：

```powershell
.\.venv\Scripts\Activate.ps1
```

## 3. Raw validation

```bash
python scripts/01_validate_raw.py
```

檢查：

- sampling rate / `elapsed_s`
- sample index continuity
- 8 個 HNC raw channels
- NaN / Inf / flat channel
- Rest1 / Type1 / Rest2 / Type2 有效 epoch 數
- `marker_sent`

主要輸出：

```text
results/raw_validation.csv
data/features/raw_validation.csv
```

## 4. Preprocessing + Feature Extraction

```bash
python scripts/02_extract_features.py
```

預設正式設定：

```text
continuous raw EEG
→ 4th-order Butterworth 2–50 Hz
→ zero-phase sosfiltfilt
→ 依 SessionStart / event elapsed time切 5-s epochs
```

Rest：marker `20 / 24`。

Mental Arithmetic：marker `11 / 12`，只使用：

```text
analysis_eligible = true
response_correct = true
```

### Band Power

依實驗室教材：

```text
X[k] = DFT{x[n]}
PSD[f_k] = |X[k]|²
BP_band = Σ PSD[f_k]
```

5 s × 500 Hz：

```text
N = 2500
Fr = Fs / N = 0.2 Hz
```

六頻帶：

```text
δ       1–4 Hz
θ       4–8 Hz
α       8–13 Hz
β_low  13–20 Hz
β_high 20–30 Hz
γ      30–45 Hz
```

7 channels × 6 bands = **42 BP features**。

### Coherence

依實驗室教材採 segmented-FFT magnitude-squared coherence：

```text
5-s trial → M 個 subsegments
Gxx = |X|²
Gyy = |Y|²
Gxy = X·Y*
先跨 subsegments 平均 Gxx / Gyy / Gxy
Coh(f) = |mean(Gxy)|² / (mean(Gxx)·mean(Gyy))
最後在各 band 內取平均
```

教材未指定 subsegment 長度；正式 config 明示採：

```text
1.0 s
non-overlapping
raw FFT / rectangular window
```

因此 5-s epoch → 5 segments；COH frequency grid = 1 Hz。

7 channels 共 `C(7,2)=21` pairs：

```text
21 × 6 = 126 COH features
```

總計：

```text
42 + 126 = 168 features / epoch
```

輸出：

```text
data/features/<subject>_features.csv
data/features/features_all.csv
data/features/raw_validation.csv
```

Feature CSV 保存**未 z-score**數值；normalization 只在各 CV training fold 內 fit，避免 leakage。

## 5. Intra-subject

```bash
python scripts/03_intra_subject.py
```

每位 participant、每個 comparison 分開：

```text
outer Stratified 5-fold CV
└─ outer training fold
   ├─ inner CV SFS（LDA accuracy）
   ├─ LDA
   ├─ RBF-SVM + inner grid search
   └─ KFDA + inner grid search
→ outer held-out fold accuracy
```

主要輸出：

```text
results/intra/intra_folds.csv
results/intra/intra_predictions.csv
results/intra/intra_summary.csv
results/intra/intra_overall_summary.csv
results/intra/intra_sfs_path.csv
results/intra/intra_feature_frequency.csv
```

## 6. Inter-subject

```bash
python scripts/04_inter_subject.py
```

outer validation 使用 **LOPO-CV**：

```text
held-out participant = test
其餘 participants = training
```

training participants 內的 SFS 與 SVM/KFDA tuning 也使用 `LeaveOneGroupOut`，避免 subject leakage。

主要輸出：

```text
results/inter/inter_folds.csv
results/inter/inter_predictions.csv
results/inter/inter_summary.csv
results/inter/inter_sfs_path.csv
results/inter/inter_feature_frequency.csv
```

## 7. Behavioral summary

```bash
python scripts/05_behavioral.py
```

輸出：

```text
results/behavioral_summary.csv
```

包含 Type1 / Type2：

- accuracy
- error rate
- correct-trial RT mean / median
- attempts required to reach 30 correct

## 8. SFS 與 classifier 設定

分析順序固定：

```text
SFS → LDA / RBF-SVM / KFDA
```

SFS 使用 training-only LDA inner-CV accuracy 做 greedy forward selection；加入最佳候選 feature 若不再提高 inner-CV accuracy 即停止。三種 classifier 共用該 outer fold 的 selected subset。

RBF grid 依實驗室海報：

```text
C = [0.1, 1, 10, 50, 100, 1000]
gamma = {1.05^-100, 1.05^-90, ..., 1.05^90, 1.05^100}
```

RBF-SVM 直接使用 `C / gamma`；KFDA 使用相同 RBF gamma，並明示定義 `lambda = 1/C`。

正式設定：

```text
configs/lab_replication.json
```

快速 smoke test：

```text
configs/smoke_test.json
```

smoke config 只用來驗證程式鏈，不可當正式結果。

## 9. BP-only / COH-only comparison

既有 `features_all.csv` 已含完整 BP / COH，因此**不用重新抽 raw feature**。

執行：

```bash
python scripts/06_compare_feature_sets.py
```

此腳本沿用完全相同的 CV / SFS / classifier / grid search，只分別把候選 feature 限制成：

```text
BP only  = 42
COH only = 126
BP+COH   = 168（既有 canonical results）
```

新增：

```text
results/feature_sets/bp/
results/feature_sets/coh/
results/feature_set_comparison.csv
```

每個 BP/COH result root 都保存 `config_snapshot.json`。

## 10. 視覺化 / 圖表

BP-only / COH-only 跑完後：

```bash
python scripts/07_make_figures.py
```

或調整顯示的 top features：

```bash
python scripts/07_make_figures.py --top-n 8
```

輸出到：

```text
results/figures/
```

每張圖同時產生 PNG 與 SVG。

包含：

### Classification

```text
classification_all_intra.*
classification_all_inter.*
feature_sets_intra_rest1_vs_type1.*
feature_sets_intra_rest2_vs_type2.*
feature_sets_inter_rest1_vs_type1.*
feature_sets_inter_rest2_vs_type2.*
```

### SFS top features

```text
sfs_top_intra_rest1_vs_type1.*
sfs_top_intra_rest2_vs_type2.*
sfs_top_inter_rest1_vs_type1.*
sfs_top_inter_rest2_vs_type2.*
```

### Connectivity scalp/network

```text
connectivity_inter_rest1_vs_type1.*
connectivity_inter_rest2_vs_type2.*
```

使用 7 frontal channel 的固定 schematic 10-20 位置：

- node = electrode
- edge = SFS 選到的 COH pair
- edge width = outer LOPO selection count
- edge label = band × selection count

這是 **connectivity network schematic**，不是將 scalp voltage 插值的傳統 topomap；對 pair-wise COH feature 的呈現更直接。

### Behavioral

```text
behavioral_accuracy.*
behavioral_rt_median.*
```

RT 圖優先使用 correct-trial median，避免 unlimited answer 中少數極長 RT 把圖拉歪。

完整說明見 [`docs/visualization_and_feature_sets.md`](docs/visualization_and_feature_sets.md)。

## 11. 一次跑原始核心流程

```bash
python scripts/run_all.py
```

第一次新環境仍建議按 `01 → 02 → 03/04/05` 分開跑，先確認 raw 與 168 features 正常。

`06` 的 BP-only / COH-only 會重新做完整 SFS + grid search，因此運算量較大；不應因分類率不佳而事後反覆改 preprocessing / CV 設定。

## 12. 目前刻意不做

- ICA：不在目前 replication pipeline
- Cz feature：第一階段不納入
- 全資料先 z-score：禁止，避免 leakage
- raw data commit：禁止
- 為了提高 accuracy 事後調整 band / CV 定義：不做

更細的公式與來源中「有明定 / 未明定」的邊界見 [`docs/method_definition.md`](docs/method_definition.md)。

## 13. E1 / E2：baseline 之外的 feature extensions

目前 Poster/Lab baseline 已 freeze；以下方法**不是學長海報原流程**，而是針對跨受試者泛化問題新增的專題 extension：

```text
E1: BP + COH + frontal hemispheric asymmetry
E2: BP + COH + Phase Locking Value (PLV)
```

E1 使用三組 frontal homologous pairs：

```text
FP1 ↔ FP2
F3  ↔ F4
F7  ↔ F8
```

正式預設使用 `log(BP_right)-log(BP_left)`，共 `3 × 6 = 18` 個 asymmetry features。

E2 對 21 組 channel pairs、六頻帶，以 continuous band-pass → Hilbert phase 計算：

```text
PLV = |mean(exp(j*(phi_a-phi_b)))|
```

共 `21 × 6 = 126` 個 PLV features。

本機有 raw data 時先抽 extension features：

```bash
python scripts/08_extract_extension_features.py
```

再跑與 baseline 完全相同的 nested intra / LOPO inter evaluation：

```bash
python scripts/09_run_extension_experiments.py
```

預設只跑 E1 與 E2 分開比較；確認後才可選擇：

```bash
python scripts/09_run_extension_experiments.py --include-combined
```

詳細公式、避免 leakage 的設計、輸出檔案與結果解讀方式見 [`docs/extensions_e1_e2.md`](docs/extensions_e1_e2.md)。
