# 方法定義與實作對照

本文件只記錄分析規格與程式中的明確實作選擇。

## 1. Epoch 與頻率解析度

- Sampling rate：`Fs = 500 Hz`
- Epoch length：`t = 5 s`
- 每 epoch：`N = Fs × t = 2500 samples`
- 頻率解析度：`Fr = Fs / N = 1 / t = 0.2 Hz`
- 第 `k` 個 DFT bin 對應：`f_k = k × Fr`

## 2. Band Power：依實驗室教材定義

教材定義：

```text
X[k] = DFT{x[n]}
PSD[f_k] = |X[k]|^2
BP_band = Σ PSD[f_k],  f_k ∈ band
```

程式因此刻意採：

```python
X = np.fft.rfft(x)
power = np.abs(X) ** 2
BP = power[(f >= low) & (f < high)].sum()
```

沒有額外做 PSD density normalization，也沒有 one-sided ×2。這是為了重現教材公式；嚴格從訊號處理名詞來說，它是「DFT squared magnitude 的頻帶總和」，與以 `V²/Hz` 為單位的 Welch PSD density 不是同一個 scaling。由於本研究每個 epoch 的 `Fs` 與 `N` 固定，這個 scaling 對同一 pipeline 的分類比較是一致的。

頻帶採半開區間 `[low, high)`，避免邊界重複：

| Band | Hz |
|---|---:|
| δ | 1–4 |
| θ | 4–8 |
| α | 8–13 |
| β_low | 13–20 |
| β_high | 20–30 |
| γ | 30–45 |

因 preprocessing 先做 2–50 Hz band-pass，因此 δ 的 1–2 Hz 部分會被濾波器抑制；頻帶定義仍照教材保留 1–4 Hz。

## 3. Coherence：依實驗室教材的 segmented-FFT 定義

本次補充教材已把 COH 的計算順序定義清楚，因此 repo 不再使用 SciPy `signal.coherence` / Welch 預設實作。

對一個 5-s analysis trial/epoch 的兩通道訊號 `x[n]`、`y[n]`，先把 trial 切成 `M` 個等長、**不重疊** subsegments。對第 `m` 個 subsegment：

```text
X_m(f) = DFT{x_m[n]}
Y_m(f) = DFT{y_m[n]}

Gxx_m(f) = X_m(f) X_m*(f) = |X_m(f)|²
Gyy_m(f) = Y_m(f) Y_m*(f) = |Y_m(f)|²
Gxy_m(f) = X_m(f) Y_m*(f)
```

接著依教材 **先跨 subsegments 平均 auto/cross spectra**：

```text
Gxx_bar(f) = mean_m Gxx_m(f)
Gyy_bar(f) = mean_m Gyy_m(f)
Gxy_bar(f) = mean_m Gxy_m(f)
```

再計算 magnitude-squared coherence：

```text
Coh_xy(f) = |Gxy_bar(f)|² / (Gxx_bar(f) Gyy_bar(f))
```

最後對感興趣頻帶內所有 `Coh_xy(f)` 頻率點取平均，得到該 pair、該 band 的單一 COH feature。這正對應教材 Step 1–6。

教材**沒有標出 subsegment 的實際秒數 / M 值**。為了讓 5-s trial 有多個 spectra 可平均且不引入教材沒有寫的 overlap/window，正式 config 明示採：

- `segment_seconds = 1.0 s`
- 5-s trial → `M = 5` 個 non-overlapping subsegments
- rectangular / raw FFT（不額外套 Hann window、不 detrend）
- 每個 band 對 coherence bins 取 arithmetic mean

`1.0 s` 是目前唯一仍屬 repo 的明示實作參數，不宣稱為教材既定值；若之後拿到實驗室舊 code，只需改 `coherence.segment_seconds`（或對照其 segmentation）即可。

特別注意：若不做 Step 4 的跨 segment 平均、只對單一 segment 直接代入 `|Gxy|²/(Gxx Gyy)`，非零頻點會數學上退化成 1；所以本 repo 嚴格保留教材的「先分段、先平均 spectra、再算 Coh」順序。

## 4. 前處理

依專題計畫：

1. continuous EEG 先做 2–50 Hz band-pass
2. 再依 event onset 切 5-s epochs
3. 以正式 SessionStart marker（code 1）的 `elapsed_ms` 作為 raw timeline zero；因面板在該 marker 前立即清除 EEG buffer，可消除 ProtocolController 建立到正式 recording start 之間的數毫秒 offset
4. Rest：marker 20 / 24
5. MA：marker 11 / 12，且只使用 `analysis_eligible=true`、`response_correct=true` trials
5. 第一階段只用 `FP1, FP2, F7, F3, Fz, F4, F8`；Cz 不進 168 features

Band-pass 類型與階數在來源中未指定。repo 預設採 4th-order Butterworth + `sosfiltfilt` zero-phase filtering，並在 config 中明示。

60-Hz notch **不是** v0.15 計畫中的必要步驟，因此 `lab_replication.json` 預設 `notch_hz: null`。若之後決定納入，只需改 config，不必改 feature 程式。

## 5. 168 features

### BP

`7 channels × 6 bands = 42`

排序：channel-major，再依 band 順序。

### COH

`C(7,2) = 21 pairs`

`21 pairs × 6 bands = 126`

排序：依 channel list 產生 lexicographic combinations，再依 band 順序。

### Total

`42 + 126 = 168`

欄位名稱例如：

```text
BP__FP1__delta
BP__FP1__theta
...
COH__FP1__FP2__delta
COH__FP1__FP2__theta
...
```

## 6. z-score 與 data leakage

專題計畫要求 normalization、feature selection、model tuning 都只能由 training fold 估計。

repo 因此不在 feature CSV 產生時對全體受試者先做 z-score，而是在 CV 的每個 training fold 內使用 `StandardScaler`：

```text
outer training fold
  → SFS inner CV
  → StandardScaler fit only on inner/train
  → classifier / grid search
outer held-out fold
  → transform with training scaler
  → prediction
```

Feature CSV 保存的是未 z-score 的 BP / COH，避免把 test information 寫死進可版本化資料。

## 7. SFS

流程依專題規格：`SFS → LDA / RBF-SVM / KFDA`。

repo 使用 LDA 的 training-only inner-CV accuracy 做 greedy Sequential Forward Selection，得到一組共同 selected subset，再讓三種 classifier 在同一 subset 上比較。

停止規則：加入最佳剩餘 feature 若無法提高 inner-CV accuracy，就停止；此規則在來源沒有明定，因此明確寫在 config / code，而不是隱藏假設。

## 8. RBF grid

依海報：

```text
C = [0.1, 1, 10, 50, 100, 1000]
gamma = {1.05^-100, 1.05^-90, ..., 1.05^90, 1.05^100}
```

共 6 × 21 = 126 組。

RBF-SVM 直接使用 `C` / `gamma`。

KFDA 的標準 formulation 使用 regularization parameter；為對應海報的 `C` grid，本 repo 定義 `lambda = 1 / C`，RBF kernel 使用同一 `gamma` grid。這個 mapping 在程式與輸出中明示。

## 9. Intra-subject

每位受試者、每個 comparison 分開：

- outer：stratified 5-fold CV，輸出 unbiased fold accuracy
- outer training fold 內：
  - SFS：inner stratified CV
  - LDA：selected subset fit
  - RBF-SVM：selected subset 上 inner grid search
  - KFDA：selected subset 上 inner grid search
- outer test fold 完全不參與 scaler / SFS / grid search

## 10. Inter-subject

- outer：LOPO-CV，每次整位 participant held out
- outer training participants 內：
  - SFS 的 inner validation 使用 `LeaveOneGroupOut`
  - SVM / KFDA grid search 也使用 training participants 的 `LeaveOneGroupOut`
- held-out participant 不參與 scaler / SFS / tuning

這樣避免把同一受試者不同 epochs 同時放進 inner train 與 validation 而產生 subject leakage。

## 11. 海報 / 計畫與 repo 的優先順序

本 repo 對方法定義採以下優先順序：

1. 本次專題 v0.15 計畫（兩個 Rest-vs-MA comparison、5-s epoch、2–50 Hz、7 frontal channels、168 features、SFS、5-fold / LOPO）
2. 本次提供的實驗室教材公式（DFT、`PSD=|X[k]|²`、頻率解析度、BP band-sum、六頻帶）
3. 學長海報中可辨識的 grid 與 validation 流程
4. 來源未明定之處才採標準且可設定的實作，並在本文件明示

因此不會把來源沒有說明的 ICA、notch、COH subsegment 秒數、KFDA regularization mapping 等偷偷寫成「實驗室規範」。
