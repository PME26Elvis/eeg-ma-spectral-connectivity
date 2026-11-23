# 手刻 LightGBM 式分類器：方法、驗證與使用

一般受試者內／跨受試者 CV 可明確選用 `LightGBM`。本專案的 [`HandcraftedLightGBMClassifier`](../src/eeg_ma/lightgbm.py) 及 [histogram／EFB 模組](../src/eeg_ma/lightgbm_histogram.py) **只使用 Python 標準函式庫與 NumPy**，自行實作 binary logistic boosting、Newton leaf outputs、histogram 分箱、leaf-wise best-first 生長、histogram subtraction、GOSS 與零衝突 EFB。Inner-CV 搜尋同樣自行實作，沒有呼叫現成 LightGBM、XGBoost、sklearn tree／boosting estimator 或其搜尋器。

加入這個選項的理由是讓研究工具具備另一種樹模型：RF 將獨立抽樣的樹取平均，boosting 則逐輪依當前預測的梯度更新。是否能改善新資料表現仍須以適當的 held-out 評估回答。本文與 CI 使用 synthetic inputs，沒有新增真實 EEG 成績，也沒有以歷史五人的分類結果挑選這份 profile。

## 1. 直接使用

依 [reference environment 指南](reproducibility.md) 安裝專案。下列 feature table 必須是自己的 schema 2.0、已通過 QA 的資料；output 路徑必須尚未存在。

```bash
# 五分類器：LDA、RBF-SVM、KFDA、手刻 RF、手刻 LightGBM
python scripts/04_inter_subject.py \
  --config configs/lightgbm.json \
  --features outputs/new-cohort/features/features_all.csv \
  --out-dir outputs/results/new-cohort-five-models/inter

# 只選用 LightGBM；特徵選擇仍沿用 training-only LDA SFS
python scripts/03_intra_subject.py \
  --config configs/lightgbm.json \
  --classifiers LightGBM \
  --features outputs/new-cohort/features/features_all.csv \
  --out-dir outputs/results/new-cohort-lightgbm/intra
```

`03`、`04`、`06`、`09`、`10`、`12` 及 `run_all.py` 共用 `--classifiers`。名稱精確為 `LDA`、`RBF-SVM`、`KFDA`、`RandomForest`、`LightGBM`。Flag 覆蓋 config 的 roster，並移除被排除的 RF／LightGBM 專屬設定；effective config 完整存入 identity。直接提供孤立的 `lightgbm` 區塊卻未選用該模型會報錯。

未指定新欄位的既有設定保留原三分類器；[`random_forest.json`](../configs/random_forest.json) 保留四分類器。Matrix reference 必須具有相同 roster，不能把缺少模型的列靜默取交集。通用圖表可呈現五分類器；歷史圖表與固定 external workflow 仍使用各自既定定義。

### 獨立 Python API

分類器本身不需要 scaler、LDA 或 sklearn；以下範例可直接訓練與非可執行 JSON 往返。

```python
import json
import numpy as np
from eeg_ma.lightgbm import HandcraftedLightGBMClassifier

rng = np.random.default_rng(7)
X = rng.normal(size=(160, 6))
y = (X[:, 0] + 0.7 * X[:, 1] > 0).astype(int)
model = HandcraftedLightGBMClassifier(
    n_estimators=32, learning_rate=0.1, num_leaves=5,
    min_data_in_leaf=4, lambda_l2=1.0, random_state=7,
).fit(X, y)

payload = json.dumps(model.to_dict(), allow_nan=False)
restored = HandcraftedLightGBMClassifier.from_dict(json.loads(payload))
np.testing.assert_array_equal(restored.predict_proba(X), model.predict_proba(X))
```

## 2. 實作範圍與設定

這是 **LightGBM 式二分類器**，不是官方 LightGBM Python 套件的完整重製或相容替代品。支援有限實數 features 與兩個同型別 labels；兩類均須具有正有效訓練質量。沒有 categorical splits、missing-value routing、multiclass／ranking、DART、GPU、distributed training、monotonic constraints 或官方模型格式互通。模型、預測及速度也不宣稱與官方實作相同。

[`configs/lightgbm.json`](../configs/lightgbm.json) 保留原訊號、SFS、其他分類器及 RF profile，明確加入第五模型。LightGBM 為 64 輪、learning rate 0.05、最多 7 leaves、最大深度 4、每葉至少 5 個保留 rows、L2 正則化及完整資料。Grid 是 `lambda_l2=[1.0, 5.0]` × `num_leaves=[3, 7]`，四個候選；這是可檢視的起始設定，不是已證明最佳的 EEG 參數。

| 參數 | 核心預設 | 本實作語意 |
| :-- | :-- | :-- |
| `n_estimators` | `64` | 最多嘗試的 boosting rounds；無有效正增益 split 時提前停止 |
| `learning_rate` | `0.05` | 乘在每棵樹葉輸出上的 shrinkage |
| `num_leaves` | `7` | 每棵樹最多葉節點數，至少 2 |
| `max_depth` | `None` | 根深度 0；profile 額外限制為 4 |
| `max_bin` | `63` | 每特徵最多 bins，支援 2–65,535 |
| `min_data_in_bin` | `1` | 可分成多 bins 時各 bin 的最少 training rows |
| `min_data_in_leaf` | `5` | 每個子葉最少實際保留 rows；不以 GOSS correction 放大 counts |
| `min_sum_hessian_in_leaf` | `0.001` | 子葉最少有效 Hessian 總量 |
| `lambda_l1`／`lambda_l2` | `0.0`／`1.0` | Newton leaf objective 的非負 L1／L2 penalty |
| `min_gain_to_split` | `0.0` | split gain 必須嚴格高於此門檻及數值 cancellation floor |
| `max_delta_step` | `10.0` | shrinkage 前的葉輸出絕對值上限；0 表示不設此上限 |
| `feature_fraction` | `1.0` | 每輪無放回選取 `max(1, ceil(fraction × p))` 個特徵，同樹共用 |
| `bagging_fraction` | `1.0` | bagging 每輪無放回保留 `ceil(fraction × eligible rows)` 筆 |
| `data_sample_strategy` | `"bagging"` | `bagging` 或 `goss`；GOSS 時 bagging fraction 必須為 1 |
| `top_rate`／`other_rate` | `0.2`／`0.1` | GOSS 高梯度保留及低梯度抽樣比例，精確計數見下文 |
| `enable_bundle` | `True` | training-only、零衝突的 EFB packing |
| `random_state` | `0` | 固定整數 master seed；managed CV 限制在 `[0, 2**32)` |
| `n_jobs` | `1` | 正整數或 -1；標準函式庫 threads 平行聚合不同 bundles 的 histograms |
| `class_weight` | `None` | `None`、`"balanced"` 或明確兩類權重；另有 `fit(..., sample_weight=...)` |
| `early_stopping_rounds` | `None` | 僅 standalone API 可對明示 validation set 啟用 |
| `early_stopping_min_delta` | `0.0` | standalone validation loss 的最低改善量；managed CV 固定為 0 |

Config 的 `lightgbm.parameters` 為基礎設定，`lightgbm.grid` 覆蓋指定維度。最多 **64 個候選**，依排序後的 grid keys 及各 value 的宣告順序展開。所有候選先驗證，再開始 SFS／訓練；未知欄位、重複候選、布林冒充數字、非有限數值、不合法組合及空 roster 均拒絕。

每個 candidate 使用相同 inner splits 及每折 seed；以各 inner fold 的 epoch accuracy 算術平均選擇，精確平手保留第一個。排序以 `Fraction(correct_count, n_test)` 精確計算，最後才輸出 float。Seed、`n_jobs` 與 early-stopping 設定不得放進 grid。Managed CV 允許以 inner grid 選 `n_estimators`，但禁止 `eval_set`／early stopping；不把 outer test 當成另一個調參集合。

## 3. Boosting 與 leaf-wise tree 的數學語意

將排序後第二類編為 $y_i=1$，有效權重為 $w_i$。初始 raw score 是 training class masses 的 log odds：

$$
F_0=\log W_1-\log W_0.
$$

第 $m$ 輪由此前的 ensemble score 計算：

$$
p_i=\sigma(F_{m-1}(x_i)),\qquad
 g_i=w_i(p_i-y_i),\qquad
 h_i=w_i\max\{\sigma(F_{m-1}(x_i))\sigma(-F_{m-1}(x_i)),10^{-12}\}.
$$

程式分別計算正／負方向的 stable sigmoid；正類梯度使用 `-sigmoid(-raw)`，避免大正 raw score 時 `p - 1` 被捨入為零。Hessian floor 是明示的數值 safeguard，並非未截斷 logistic loss 的精確二階導數。Log loss 使用穩定的 softplus 表達式。無法可靠表示的權重質量或中間結果會拒絕，不產生 NaN 模型。

令節點梯度／Hessian 總量為 $G,H$，則 shrinkage 前的值為：

$$
v=-\frac{\operatorname{sign}(G)\max(|G|-\lambda_1,0)}{H+\lambda_2},
$$

再依 `max_delta_step` 限制其範圍。分裂使用與**實際 clipped value 一致**的二倍 objective improvement：

$$
S(G,H,v)=-2Gv-(H+\lambda_2)v^2-2\lambda_1|v|,\qquad
\operatorname{gain}=S_L+S_R-S_P.
$$

沒有 clipping 時等價於常見的 thresholded-gradient Newton score。接受 split 需同時滿足 counts、Hessian、depth、leaf budget 與嚴格正增益。數值 cancellation floor 為 $128\epsilon(|S_L|+|S_R|+|S_P|)$；gain 必須同時大於這個 floor 及 `min_gain_to_split`。純 XOR 在初始所有單一分裂均零增益時可合法保留 prior-only model；此 boosting 規則與 RF 的零增益 CART 規則不同。

每個待分裂葉先找最佳 histogram boundary，然後比較**所有候選葉**的最佳 gain，選擇最大者繼續生長，直到 leaf budget 或其他停止條件。Tie band 以全域最大 gain 的 $64\epsilon$ 相對尺度為共同基準，依較小 feature index、bin boundary、leaf ID 決定，避免逐對近似平手累積漂移。預測以 `x <= threshold` 向左：

$$
F_m(x)=F_{m-1}(x)+\eta\,v_{m,\ell(x)},\qquad
\widehat p_1(x)=\sigma(F_m(x)).
$$

對可能因 histogram roundoff 改變最大 gain 或平手結果的候選，先由 summation error envelope 及 objective 對 G／H 的變化界限篩出，再用實際 row derivatives 重算。最終排序仍採上述 $64\epsilon$ band，不用擴大平手範圍掩蓋累加誤差。

最終類別由機率 argmax 決定，平手選排序後第一類。`feature_importances_` 是 active model 中各特徵 split gain 加總後正規化，`feature_split_counts_` 另保存分裂次數；沒有分裂時均為零。這是 training importance，不是 held-out permutation importance 或電極連結的因果證據。

## 4. Histogram、GOSS 與 EFB

### Training-only bins 與 histogram subtraction

每個 feature 只從當次 fit 的正有效權重 rows 建立 deterministic rank／quantile-style bins。相同原始值不拆到不同 bins；rank targets 無法同時滿足 `min_data_in_bin` 時可得到少於 `max_bin` 的 bins。低 cardinality 且 counts 足夠時保留所有有效邊界。常數特徵只有一 bin。

Threshold 使用安全中點 `lo/2 + hi/2`；若相鄰 float 的捨入使中點落在 `[lo, hi)` 之外，取 `lo`，維持原有分組。Test／validation features 僅套用已訓練的 cuts，不影響分箱。

Histogram 保存每個 bin 的實際 count、gradient sum 與 Hessian sum；依 cumulative statistics 搜尋分裂。Gradient prefixes 使用補償累加，對 cancellation 敏感的候選再由原始 row derivatives 重算，避免 bin 內或跨 bins 的大數抵銷掩蓋小而有效的 gain。節點分開後先直接建立較小 child 的 histograms，再由 parent 減去它得到另一 child；若 subtraction 有無法信任的 cancellation，就直接重建該 child。EFB default-bin 的 subtraction 同樣有直接重算 fallback。這項保護避免為了節省聚合工作而接受負 Hessian 或虛假的空 bin 質量。

為降低 bin 內累加誤差，每個需要的 bundle 先對 packed codes 做一次 stable sort，再對連續 bin groups 用 `math.fsum` 聚合 G／H，counts 則用精確整數 `bincount`。對 $r$ 個節點 rows、$b$ 個 packed bins，每個 bundle 的這一步為 $O(r\log r+b)$ 時間及 $O(r+b)$ 暫存空間；不是宣稱與一般線性 histogram accumulation 相同成本。EFB 仍共用 bundle 聚合，subtraction 仍避免直接重建每個 child；這是本純 Python 實作選擇的精度與成本取捨。

### GOSS 的精確計數

GOSS 以**含 sample／class weights 的梯度絕對值**排序，高梯度平手依原始 row index 決定。設 eligible row 數為 $N$：

- 高梯度保留數為 `min(N - 1, ceil(top_rate × N))`。
- 低梯度抽樣數為 `min(N - top_count, ceil(other_rate × N))`，從剩餘 rows 無放回抽樣。
- 高梯度 rows 的 multiplier 是 1；低梯度抽中 rows 的 gradient 與 Hessian 同乘 `low_pool_size / low_sample_count`。

低梯度 pool 按原始 row index 排序後抽樣，保存的 selected／top indices 也採此排序。這是使用實際整數抽樣數的 correction；不把有限樣本的 rounding 隱藏在理論比例裡。所有 rounds 都使用這個規則，**沒有 warmup**。`min_data_in_leaf` 仍計算真實保留 rows，不能把 correction 當作複製出的 observations。

預設 strategy 為完整資料的 bagging，GOSS 必須明確啟用。這對本 repo 的小樣本起始設定較容易檢視；沒有宣稱抽樣一定改善泛化或速度。

### 零衝突 EFB

本實作依 training bin 的 nondefault activation counts 降序、feature index 平手，greedy 地把彼此從不共同 active 的 features 放入同一 bundle。只接受**零衝突** packing，不採近似允許衝突的 EFB，也不宣稱得到最少 bundle 數。

Default 是原始值 0 所屬的 bin，負值與正值共存時它可能在中間。Packed code 0 表示整個 bundle 都在各自 default bin，其他非預設 bins 使用不重疊的 offsets；由 packed histogram 還原原 feature 的 monotonic bin statistics。這是對已建立 bins 的 lossless encoding，分箱本身仍是近似。

Inference 使用普通 per-feature bins，**不要求新資料維持 training exclusivity**；測試時同一 bundle 內多個 features 同時 active 仍可預測。零有效權重 rows 不參與分箱、bundle discovery 或 packed training matrix，避免被排除的 outlier 影響 layout。Dense EEG features 可能無法壓縮成較少 bundles，不能僅因開啟 EFB 就宣稱加速。

## 5. 權重、亂數、狀態與 early stopping

`sample_weight` 有限且非負，兩個類別乘上 class factors 後都必須保有正質量。`balanced` 以 sample weight 大於 0 的 rows 計算類別頻數，再乘上 sample weight。Bins 使用 eligible rows 的**計數**，不依權重大小移動 quantiles；權重進入 prior、梯度、Hessian 與 weighted log loss。Standalone `score` 在明示 sample weights 時才採 weighted accuracy。

每輪 seed 由 master `SeedSequence` 產生，row sampling 與 feature sampling 再使用分開的子 streams。增加 rounds 保留既有 prefix；serial／threaded histograms 依固定順序歸約，不消耗 process-global NumPy 或 Python RNG。Managed CV 再以獨立 SHA-256 domain 與 scope／comparison／subject／outer fold／inner-or-refit 推導 seeds，與 RF 的 domain 分開。

Fitted arrays 以不可寫入的 bytes backing 保存，公開 array view 具有獨立 shape metadata。Fit 建立完整新狀態後才替換舊模型；失敗的 refit 保留上一份可用狀態。成功的 `set_params` 使模型回到未 fit 狀態，export 使用 fit 當時的 parameter snapshot。

Standalone API 可使用 `fit(X, y, eval_set=(X_valid, y_valid), eval_sample_weight=...)`，並明確設定 `early_stopping_rounds`。Validation 只用來比較 weighted log loss，不參與 bins、EFB、class-weight fitting 或 gradients。Baseline 也參加比較，所以 `best_iteration_` 可以是 0；模型保留所有實際訓練 rounds 及完整 loss history，prediction 使用最佳 prefix。`n_estimators_` 是保存的實際樹數，可能大於 active 的 `best_iteration_`。這個選擇規則仍依賴 validation labels，因此不能把該 validation score 當作全新獨立 test score。

**EEG integrated CV 停用此功能**。如需選 rounds，使用既有 inner grid；最後只以完整 outer-training refit。Inter 的 outer／inner 都依受試者分組，同人的全部 sessions 同側。Intra 保持同人內 epoch-stratified CV，沒有改成 leave-session-out。各模型共用 outer-training LDA SFS；inner scores 是該 shared SFS 之後的 selection diagnostics，outer held-out predictions 才定義 CV performance。

## 6. API、JSON 與來源證據

| API／屬性 | 用途 |
| :-- | :-- |
| `fit(X, y, sample_weight=None, *, eval_set=None, eval_sample_weight=None)` | Binary training；選用明示 standalone validation |
| `decision_function`／`predict_proba`／`predict`／`score` | Raw scores、類別機率、labels、accuracy |
| `get_params`／`set_params`／`validate_parameters` | 設定快照、狀態失效與預檢 |
| `to_dict`／`from_dict` | 嚴格版本化的純資料 JSON |
| `bin_mapper_`／`bundle_layout_` | Train-only cuts、counts、default bins 與 bundles |
| `estimators_`／`iteration_seeds_` | 實際保存的 trees 與 seeds |
| `sampling_records_`／`feature_subsets_` | 每輪保留 rows、GOSS multipliers 與 feature masks |
| `base_score_`／`best_iteration_`／`n_estimators_` | Weighted prior、active prefix 與實際保存樹數 |
| `training_loss_`／`validation_loss_` | 包含 baseline 的完整 weighted loss history |
| `feature_importances_`／`feature_split_counts_` | Active model 的 training gain／split diagnostics |

Algorithm 標記為 `eeg-handcrafted-lightgbm-1`，模型 schema 為 `eeg-handcrafted-lightgbm-json-1`；bins 與 EFB 各有獨立版本。JSON 不執行 pickle／Python code，loader 檢查 raw types、完整欄位、dimensions、labels、tree topology、depth／leaf bounds、threshold／bin 對應、parent-child count／G／H conservation、Newton values／gains、sampling／seed 結構、history 與 importance。配置尺寸先對照實際 materialized payload，不信任孤立的巨大 dimensions。

`X` 需是有限實數二維矩陣；NaN、inf、complex、string／object features 會拒絕。整數超過 float64 精確範圍時需呼叫端明確處理。Labels 支援同型別 bool、整數、有限浮點或非空 Unicode，恰有兩類；mixed labels 不被隱式字串化。Standalone model 只知道 feature dimension，呼叫端需保持欄位順序；managed CV 同時封存 ordered names。

### Managed archive

明確選用 LightGBM 使用 `eeg-classifier-policy-2`，由原 `evaluation-3` identity 綁定完整 roster／policy；未選用時保留既有 policy-1 的完整形狀。每個啟用 scope 新增：

| Artifact | 證據 |
| :-- | :-- |
| `lightgbm_inputs.csv` | 完整 feature table，literal metadata／float round-trip；SHA 必須與 identity 的 table SHA 相符 |
| `lightgbm_audit.json` | 候選順序、inner membership／seeds／scores／predictions／raw scores、最佳參數、selected features、outer IDs／預測及 final model |

Validator 從原始來源重建 bins 與 EFB，按 final ensemble 的 prefix 計算每輪 weighted gradients／Hessians，重新產生 row／feature sampling，再依 stored tree routing 核對所有 node counts、G／H、Newton outputs、split gains、training loss、importance 及 outer predictions。Inner records 另驗證完整 membership、accuracy、exact candidate mean、first-maximum 決策及 seeds。重新計算 manifest hashes 不能修復與來源矛盾的模型或預測。

Audit 明確宣告：**重播 final model 的來源與訓練 statistics，加上 inner-selection record consistency；不重新 fit 所有 inner models，也不重搜每個 greedy optimum。** 純模型 JSON 缺少原始 X／y，不能單獨證明 training loss 或 split 來自某份 EEG；managed source replay 補上此範圍。Manifest 是稽核記錄，不是外部簽章。原子發布、resume 與 roster coverage 見 [run provenance](run_provenance.md)。

## 7. 工程驗證與成本

測試分為 [核心](../tests/test_lightgbm_core.py)、[bins／EFB／histograms](../tests/test_lightgbm_histogram.py)、[獨立數學 oracle](../tests/test_lightgbm_oracles.py)、[CV／封存](../tests/test_lightgbm_integration.py) 與 [CLI](../tests/test_lightgbm_cli.py)。重點包含：

- 用 `Fraction` 及原始 feature masks 窮舉正則化／clipped Newton splits，獨立核對整棵第一輪 best-first tree。
- 以 direct histogram aggregation 對照 EFB／subtraction，涵蓋負值、零值、常數、相鄰與極端 float、cancellation fallback。
- 重算 boosting prefix 的 GOSS weighted gradients、實際抽樣數與 correction；驗證獨立 row／feature RNG。
- 檢查 training-only bins、零權重排除、held-out overlapping features、parallel／prefix reproducibility、唯讀狀態與失敗 refit。
- 真實 CV membership、手刻搜尋、strict JSON／重新 hash 的 artifact 攻擊與既有三／四模型相容性。

Reference CI 的 [真實 smoke](../tests/run_lightgbm_smoke.py) 從 3 位 synthetic people、6 sessions、48 epochs 的 raw inputs 產生 QA／features，再實際呼叫 `03`／`04` 五分類器 subprocess。每個 scope 240 筆 predictions，intra 30／inter 15 個 classifier-fold rows；同時驗證 RF／LightGBM 封存、safe JSON、resume 不改 bytes、拒絕覆寫及通用圖表。沒有 mock estimator、環境、code identity 或 clock。

同一 smoke 保存 dense 300 × 126 的 16-round bagging／feature-sampling 模型及 sparse 240 × 12 的 12-round GOSS／EFB 模型與 inputs，檢查 serial／threaded、bundled／unbundled 及未見過的 feature overlap。這些 rounds 是要求上限，無正 gain 時可提早停止。觀察時間僅為該次有限 workload 的工程紀錄，不設定 accuracy／速度門檻，也不推論真實資料表現。

Python 與 NumPy 實作可檢視每項規則，但沒有官方編譯式實作的效能保證。整體 CV 成本乘上 candidates、inner／outer folds，SFS 與其他分類器另有成本；EFB 只有實際存在 exclusivity 才壓縮，threads 也不保證所有 workload 都加速。使用新資料時應先做 bounded smoke，再事先固定合理搜尋範圍。

```bash
python -m pytest -q tests/test_lightgbm_core.py \
  tests/test_lightgbm_histogram.py tests/test_lightgbm_oracles.py \
  tests/test_lightgbm_integration.py tests/test_lightgbm_cli.py
python tests/run_lightgbm_smoke.py --out-dir outputs/lightgbm/smoke-001
```

## 方法來源

1. Ke et al. (2017). *LightGBM: A Highly Efficient Gradient Boosting Decision Tree*. NeurIPS. [原論文](https://proceedings.neurips.cc/paper_files/paper/2017/file/6449f44a102fde848669bdd9eb6b76fa-Paper.pdf)。GOSS 與 EFB 的方法來源；本文明列自己的 finite-count correction、zero-conflict layout 與其他實作選擇。
2. LightGBM 官方 [Features](https://lightgbm.readthedocs.io/en/stable/Features.html) 與 [Parameters](https://lightgbm.readthedocs.io/en/stable/Parameters.html)。提供 histogram、leaf-wise、subtraction 及參數的背景。本實作的預設、量化、ties、sampling、停止與封存契約以此文件及 versioned source 為準，不等同官方 API。
