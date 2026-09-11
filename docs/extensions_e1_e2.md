# E1 / E2 延伸實驗方法

本文件描述 **Poster/Lab baseline 之外** 的兩個延伸實驗。Baseline 的 BP、COH、SFS、LDA / RBF-SVM / KFDA、intra 5-fold 與 inter LOPO 定義不修改；E1 / E2 只新增 feature representation，並沿用相同的 leakage-safe validation framework。

## 0. Baseline 保持不變

Baseline 仍是：

```text
2–50 Hz continuous preprocessing
→ 5-s Rest / correct Mental Arithmetic epochs
→ 42 BP + 126 COH = 168 features
→ training-fold-only z-score
→ training-only SFS
→ LDA / RBF-SVM / KFDA
→ intra-subject 5-fold / inter-subject LOPO
```

`feature_set="all"` 的語意仍固定為 BP+COH，不會因為加入 E1/E2 而偷偷把新 feature 混進 baseline。

---

## 1. E1：Frontal Hemispheric Asymmetry

### 1.1 電極對

利用 7 個 frontal channels 中三組自然左右同源位置：

```text
FP1 ↔ FP2
F3  ↔ F4
F7  ↔ F8
```

每組以 `(left, right)` 表示。

### 1.2 正式預設：log-ratio asymmetry

對每個 epoch、每個 frequency band：

```text
ASYM = log(BP_right) - log(BP_left)
     = log(BP_right / BP_left)
```

程式使用非常小的 `epsilon=1e-12` 防止數值上的 log(0)，但正常 BP 資料應為正值。

三組左右電極 × 六頻帶：

```text
3 × 6 = 18 asymmetry features
```

欄位例如：

```text
ASYM_LOGRATIO__F3__F4__alpha
```

第二個 channel 是 right，因此上例代表 `log(BP_F4)-log(BP_F3)`。

### 1.3 可選 normalized difference

程式亦支援：

```text
(BP_right - BP_left) / (BP_right + BP_left)
```

但 E1 正式預設先使用 log-ratio，避免同時改兩個因素。若日後要比較 asymmetry 定義，應把它視為額外 sensitivity experiment。

### 1.4 E1 分類 feature set

E1 不是用 18 features 取代 baseline，而是主要比較：

```text
Baseline:  BP + COH                     = 168
E1:        BP + COH + Asymmetry         = 186
```

同時 repo 支援 `asym` / `bp_asym` 等 feature set，方便日後做消融，但不屬於目前 primary E1 comparison。

---

## 2. E2：Phase Locking Value (PLV)

### 2.1 定義

對兩通道在指定頻帶的 analytic phase：

```text
PLV = | mean_t exp(j (phi_a(t) - phi_b(t))) |
```

理論範圍：

```text
0 ≤ PLV ≤ 1
```

PLV 只使用 phase difference 的一致性，不直接使用訊號振幅大小。

### 2.2 Phase 取得方式

為了避免每個 5-s epoch 個別 zero-phase filtering 所造成的邊界 transient，E2 採：

```text
raw continuous EEG
→ baseline 既有 2–50 Hz continuous preprocessing
→ 每個 frequency band 再對「整段 continuous signal」做 4th-order Butterworth zero-phase band-pass
→ Hilbert transform
→ analytic phase
→ 依 baseline 完全相同的 event/sample index 切 5-s epoch
→ pair-wise PLV
```

這表示 Rest / MA epoch 的 onset、長度、eligibility 與 baseline 完全一致，只增加 phase-derived features。

六頻帶仍使用 baseline 相同定義：

```text
delta       1–4 Hz
theta       4–8 Hz
alpha       8–13 Hz
beta_low   13–20 Hz
beta_high  20–30 Hz
gamma      30–45 Hz
```

因最前面仍有 baseline 2-Hz high-pass，delta 的 1–2 Hz 同樣會被抑制；這與 baseline BP 的既有方法一致。

### 2.3 Feature 數量

7 channels 共有 21 pairs：

```text
21 × 6 = 126 PLV features
```

欄位例如：

```text
PLV__F3__F4__alpha
```

### 2.4 E2 分類 feature set

主要比較：

```text
Baseline:  BP + COH                     = 168
E2:        BP + COH + PLV               = 294
```

程式也支援 `plv`、`coh_plv` 等消融模式，但不是目前 primary E2 comparison。

---

## 3. Validation：與 baseline 使用完全相同的框架

E1 / E2 都不改 validation protocol：

### Intra-subject

```text
outer stratified 5-fold
→ outer training 內做 z-score / SFS / SVM-KFDA grid search
→ outer held-out fold 評估
```

### Inter-subject

```text
outer LOPO
→ held-out participant 完全不參與 training scaler / SFS / tuning
→ training participants 內用 LeaveOneGroupOut 做 inner validation
```

`random_state`、fold 數、RBF grid、SFS stopping rule 都沿用 `configs/lab_replication.json`，因此 E1/E2 與 baseline 的差異主要來自 feature representation，而不是 evaluation protocol 改變。

本階段**不加入** subject-specific baseline correction、Euclidean Alignment 或其他會改變 held-out subject calibration privilege 的方法。

---

## 4. 執行順序（WSL2 / Linux）

先同步 repo 與環境：

```bash
git pull origin main
source .venv/bin/activate
pip install -e ".[dev]"
pytest
```

### Step A：抽 E1 / E2 features

需要本機 `data/raw/` 仍保有五位受試者 raw session：

```bash
python scripts/08_extract_extension_features.py
```

預期：

```text
subjects=5
epochs=600
asymmetry_features=18
plv_features=126
NaN=0
PLV range=[0.xxxxxx, 0.xxxxxx]
```

輸出：

```text
data/features/extensions/
├─ <subject>_extensions.csv
├─ features_extensions.csv
└─ extension_manifest.csv
```

這些檔案不含 raw waveform，可像既有 BP/COH feature CSV 一樣版本控制。

### Step B：跑 E1 / E2 正式分析

```bash
python scripts/09_run_extension_experiments.py
```

預設只跑：

```text
E1 = baseline + asymmetry
E2 = baseline + PLV
```

它們各自重新做完整 training-only SFS / grid search，並沿用原本 intra / inter validation。

輸出：

```text
results/extensions/
├─ e1_asymmetry/
│  ├─ config_snapshot.json
│  ├─ intra/
│  └─ inter/
├─ e2_plv/
│  ├─ config_snapshot.json
│  ├─ intra/
│  └─ inter/
├─ extension_summary.csv
├─ extension_delta_by_unit.csv
└─ extension_delta_summary.csv
```

其中 `extension_delta_by_unit.csv` 會直接把每位 participant 的 extension accuracy 與既有 poster baseline 做配對差值，避免只看平均 accuracy。

若 E1/E2 分開看完後，才想探索兩者同時加入：

```bash
python scripts/09_run_extension_experiments.py --include-combined
```

這會額外加入：

```text
BP + COH + Asymmetry + PLV = 312 features
```

不建議一開始就只看 combined，否則無法知道改善或退步是由哪個 extension 造成。

---

## 5. 如何解讀結果

本階段不以「一定提升分類率」為成功條件。應至少同時看：

- intra / inter 平均 accuracy
- `accuracy_std`
- 每位 held-out participant 的 `delta_accuracy`
- Rest1 vs Type1 / Rest2 vs Type2 是否一致
- LDA / RBF-SVM / KFDA 是否呈現一致方向
- SFS 是否反覆選到同類 asymmetry / PLV features

只有 5 位受試者時，某一個 participant 就能明顯改變 LOPO 平均值，所以 `extension_delta_by_unit.csv` 比單一平均 accuracy 更重要。

---

## 6. Baseline 與研究貢獻的界線

報告建議明確區分：

```text
Poster/Lab baseline
= BP + COH → SFS → LDA/RBF-SVM/KFDA → 5-fold/LOPO

E1（本專題 extension）
= baseline + hemispheric asymmetry

E2（本專題 extension）
= baseline + PLV
```

E1/E2 不應被寫成學長海報原本的方法。它們的用途是針對目前 baseline 顯示的跨受試者泛化困難，測試較具相對性 / phase synchronization 的 representation 是否能降低 subject variability。
