# eeg-ma-spectral-connectivity

HNC Mental Arithmetic EEG 的可重現分析 repo。README 只描述資料格式、前處理、feature extraction、intra-subject / inter-subject 分析與輸出，不包含研究動機。

## 分析範圍

固定處理兩個主要二分類：

1. `Rest1 vs Type1`
2. `Rest2 vs Type2`

第一階段 replication analysis 使用 7 個 frontal channels：

```text
FP1, FP2, F7, F3, Fz, F4, F8
```

`Cz` 保留於 raw EEG，但不進第一階段 168-feature pipeline。

完整流程：

```text
HNC v0.15 raw session
→ raw / protocol validation
→ continuous 2–50 Hz band-pass
→ 5-s epoching
→ 42 Band Power + 126 Coherence = 168 features
→ feature CSV（可進 Git）
→ SFS
→ LDA / RBF-SVM / KFDA
→ intra-subject 5-fold CV
→ inter-subject LOPO-CV
```

專題計畫指定的前處理、5-s epochs、正確 MA trials、168 features、SFS、三種 classifier、5-fold 與 LOPO 都在此 pipeline 中保留。

## 1. Raw data 不進 Git

`.gitignore` 已排除 `data/raw/*`。

把五位受試者各自的完整 session folder 放進：

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

程式會遞迴尋找 `eeg_raw.csv`，資料夾名稱不需要符合固定格式。

本 repo 直接支援目前 v0.15 面板輸出欄位，包括：

```text
eeg_raw.csv
sample_index, elapsed_s, FP1, FP2, F7, F3, Fz, F4, F8, Cz

events.csv
session_id, subject_id, stage, phase, task_type, attempt_index,
utc_timestamp, elapsed_ms, marker_code, marker_sent, marker_message

trials.csv
session_id, subject_id, task_type, attempt_index, ...,
response_correct, analysis_eligible, reaction_time_ms, calculation_onset, ...
```

## 2. 安裝

建議 Python 3.10+。

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -e .
```

開發 / 測試：

```powershell
pip install -e ".[dev]"
pytest
```

## 3. 先驗證五份 raw data

```powershell
python scripts/01_validate_raw.py
```

輸出：

```text
results/raw_validation.csv
```

會檢查：

- 500 Hz 與 `elapsed_s` 是否一致
- sample index 是否連續
- 8 個 HNC raw channels 是否存在；Feature Extraction 再只取 7 個 frontal channels
- NaN / Inf / flat channel
- Rest1 / Type1 / Rest2 / Type2 是否各有預期 30 個有效 epochs
- `marker_sent` 狀態

## 4. Feature Extraction

```powershell
python scripts/02_extract_features.py
```

### Preprocessing

預設 replication config：

```text
continuous raw EEG
→ 4th-order Butterworth 2–50 Hz
→ zero-phase sosfiltfilt
→ 以正式 SessionStart marker 校正 raw time-zero，再依 events.elapsed_ms 切 5-s epochs
```

Rest：

- Rest1 marker `20`
- Rest2 marker `24`

Mental Arithmetic：

- Type1 calculation onset marker `11`
- Type2 calculation onset marker `12`
- 只保留 `analysis_eligible=true` 且 `response_correct=true`

### Band Power

教材定義：

```text
X[k] = DFT{x[n]}
PSD[f_k] = |X[k]|²
BP_band = Σ PSD[f_k]
```

5 秒 epoch、500 Hz：

```text
N = 2500
Fr = Fs/N = 0.2 Hz
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

7 × 6 = **42 BP features**。

### Coherence

7 channels 共 `C(7,2)=21` pairs；每 pair 六頻帶：

21 × 6 = **126 COH features**。

依補充的實驗室教材，COH 採 segmented-FFT magnitude-squared coherence，而不是直接呼叫 SciPy Welch coherence：

```text
一個 5-s trial → 切成 M 個 subsegments
每段：Gxx=|X|², Gyy=|Y|², Gxy=X·Y*
先跨 subsegments 平均 Gxx / Gyy / Gxy
Coh(f)=|mean(Gxy)|² / (mean(Gxx)·mean(Gyy))
最後對指定 band 內的 Coh(f) 取平均
```

教材沒有指定 subsegment 秒數；正式 config 明示採 **1.0 s、non-overlapping、raw FFT（無額外 Hann window）**，因此每個 5-s trial 有 5 個 subsegments。這個秒數是可設定的實作參數，不冒充成教材已指定的值。 在這個正式設定下，COH 頻率格點為 **1 Hz**；Band Power 仍使用完整 5-s epoch，因此 BP 的頻率解析度仍是 **0.2 Hz**。

### Total

```text
42 + 126 = 168 features / epoch
```

輸出：

```text
data/features/<subject>_features.csv
data/features/features_all.csv
data/features/raw_validation.csv
```

`data/features/` **沒有被 `.gitignore` 排除**，所以 feature CSV 可以版本控制。

## 5. 為什麼 feature CSV 不先 z-score

計畫要求 normalization、feature selection、model tuning 都只能由 training fold 估計。

因此 feature CSV 保存未標準化 BP / COH；`StandardScaler` 在每個 CV training fold 內 fit，再套用 validation / test。這樣不會把 held-out data 的 mean/std 洩漏進 training。

## 6. Intra-subject

```powershell
python scripts/03_intra_subject.py
```

每位 participant、每個 comparison 分開做：

```text
outer Stratified 5-fold CV
└─ outer training fold
   ├─ inner CV SFS（LDA accuracy）
   ├─ LDA
   ├─ RBF-SVM + inner grid search
   └─ KFDA + inner grid search
→ outer held-out fold accuracy
```

輸出：

```text
results/intra/intra_folds.csv
results/intra/intra_predictions.csv
results/intra/intra_summary.csv
results/intra/intra_overall_summary.csv
results/intra/intra_sfs_path.csv
results/intra/intra_feature_frequency.csv
```

## 7. Inter-subject

```powershell
python scripts/04_inter_subject.py
```

outer validation 使用 **LOPO-CV**：

```text
held-out participant = test
其餘 participants = training
```

training participants 裡的 SFS 與 SVM/KFDA tuning 也使用 `LeaveOneGroupOut`，避免同一受試者的 epochs 同時出現在 inner train / validation。

輸出：

```text
results/inter/inter_folds.csv
results/inter/inter_predictions.csv
results/inter/inter_summary.csv
results/inter/inter_sfs_path.csv
results/inter/inter_feature_frequency.csv
```

## 8. SFS

分析順序固定為：

```text
SFS → LDA / RBF-SVM / KFDA
```

SFS 使用 training-only LDA inner-CV accuracy 做 greedy forward selection。加入最佳候選 feature 若不再提高 inner-CV accuracy 即停止，得到該 outer fold 的 selected subset。

這個「LDA 作為共同 SFS criterion」是目前資料來源下的明確實作選擇：海報將流程寫成 `SFS → LDA / Non-linear SVM / KFDA`，且 Figure 4 明確描述 inter-subject LDA 的 SFS 結果，但沒有提供 SVM/KFDA 各自重新跑 SFS 的規則。因此正式版先用同一個 training-only subset 公平比較三種 classifier；若之後取得實驗室舊程式，可只替換 SFS evaluator，不必重做 feature extraction。

因此每個 outer fold 都會重新做 SFS，不會在全資料先選一次 features。

## 9. SVM / KFDA grid search

海報中的 RBF grid：

```text
C = [0.1, 1, 10, 50, 100, 1000]
gamma = {1.05^-100, 1.05^-90, ..., 1.05^90, 1.05^100}
```

設定位於：

```text
configs/lab_replication.json
```

RBF-SVM 使用 `SVC(kernel="rbf")`。

KFDA 為 repo 內的 binary regularized Kernel Fisher Discriminant：RBF kernel 使用同一 gamma，並定義 `lambda = 1/C` 來對應海報的 C grid。

## 10. Behavioral summary

```powershell
python scripts/05_behavioral.py
```

輸出：

```text
results/behavioral_summary.csv
```

包含：

- Type1 / Type2 accuracy
- error rate
- correct-trial reaction time mean / median
- attempts required to reach 30 correct

## 11. 一次跑完整流程

```powershell
python scripts/run_all.py
```

先建議單獨跑 `01_validate_raw.py` 與 `02_extract_features.py`；確認五位資料與 168 features 都正常，再跑模型分析。

> 正式 `lab_replication.json` 使用海報完整 RBF grid，並在每個 outer training fold 內做 SFS / tuning，因此運算量會明顯大於 smoke test。第一次在新環境請先用 `configs/smoke_test.json` 驗證程式鏈，再跑正式設定。

## 12. 正式版與 smoke config

正式方法：

```text
configs/lab_replication.json
```

快速驗證程式可跑：

```text
configs/smoke_test.json
```

`smoke_test.json` 只縮小 SFS / RBF grid 以節省時間，不應拿來當正式結果。

## 13. 可重現性與 Git 管理

- `data/raw/`：由 `.gitignore` 排除，不進 Git。
- `data/features/`：刻意不排除，可提交 BP / COH feature tables。
- `results/`：預設排除，避免每次 CV 重跑造成大量 diff；正式要保存特定結果時可另外建立 release/tag 或移出 ignore。
- `.github/workflows/tests.yml`：GitHub Actions 會在 Python 3.10 / 3.12 執行單元與 synthetic end-to-end tests。
- `pytest` 包含頻率解析度、教材 DFT-squared BP、COH、168 維、KFDA、SFS、v0.15 event/epoch 對齊與 intra/inter smoke tests。

## 14. 目前刻意沒有做的處理

- ICA：不在目前 v0.15 計畫的 replication pipeline 中
- Cz feature：第一階段不納入
- 全資料先 z-score：禁止，避免 leakage
- raw data commit：禁止

更細的公式、來源中有明定與沒有明定的實作邊界，見 [`docs/method_definition.md`](docs/method_definition.md)。
