# 手刻 Random Forest：演算法、評估與封存契約

`RandomForest` 現在可由一般 intra／inter CV runners 選用，也可直接透過 Python API 訓練。森林核心 [`random_forest.py`](../src/eeg_ma/random_forest.py) **只依賴 Python 標準函式庫與 NumPy**；CART、bootstrap、節點特徵抽樣、機率投票、OOB、importance 及模型讀寫均由本專案實作。RF 的 inner-CV 搜尋也以明確迴圈實作，不呼叫現成 tree／forest estimator、`GridSearchCV` 或分類 metric。

這項能力加入新的可選分析。原三分類器的預設、歷史結果，以及 A 的固定 LDA 模型與 B 的 LDA 控制計畫各自維持原定義。以下驗證使用 synthetic inputs，沒有增加真實 EEG 準確率主張。

## 1. 直接使用

先依 [固定環境指南](reproducibility.md) 安裝專案。將下例的 feature 路徑換成自己的 **schema 2.0、已通過 QA** 的新資料；output 必須是尚未存在的路徑。

```bash
# 四分類器：LDA、RBF-SVM、KFDA、手刻 RandomForest
python scripts/04_inter_subject.py \
  --config configs/random_forest.json \
  --features outputs/new-cohort/features/features_all.csv \
  --out-dir outputs/results/new-cohort-rf/inter

# 只執行 RandomForest；仍使用同一個 training-only LDA SFS
python scripts/03_intra_subject.py \
  --config configs/random_forest.json \
  --classifiers RandomForest \
  --features outputs/new-cohort/features/features_all.csv \
  --out-dir outputs/results/new-cohort-rf-only/intra
```

`03`、`04`、`06`、`09`、`10`、`12` 及 `run_all.py` 均接受 `--classifiers`，名稱大小寫需精確符合 `LDA`、`RBF-SVM`、`KFDA`、`RandomForest`。Flag 覆蓋 config 中的名單，effective config 完整寫入 run identity。明確排除 RF 時，CLI 同時移除不再使用的 `random_forest` 區塊；直接提供含 RF 設定卻未選用 RF 的 config 則會報錯。

完整 raw pipeline 可使用：

```bash
python scripts/run_all.py \
  --raw-root /path/to/validated-new-raw \
  --config configs/random_forest.json \
  --classifiers LDA RBF-SVM KFDA RandomForest \
  --out-root outputs/pipeline/new-cohort-rf
```

一般 CV 的 `--resume` 會驗證完整來源與產物後重用，不會覆寫或重新訓練。Matrix runners 的 reference 也必須具有相同 classifier roster；四分類器分析不能把歷史三分類器 reference 的缺列靜默丟掉。通用分類圖表可呈現四個模型；歷史研究的固定圖表入口仍對應原來的三模型矩陣。

### 小型獨立 Python 範例

森林本身不需要 LDA、SFS、standardization 或 sklearn。以下以 XOR 示範單棵 CART 的零增益第一層，以及非可執行 JSON 的往返：

```python
import json
import numpy as np
from eeg_ma.random_forest import HandcraftedRandomForestClassifier

X = np.array([[0., 0.], [0., 1.], [1., 0.], [1., 1.]])
y = np.array([0, 1, 1, 0])
model = HandcraftedRandomForestClassifier(
    n_estimators=1, bootstrap=False, max_features=None, random_state=7,
).fit(X, y)

np.testing.assert_array_equal(model.predict(X), y)
payload = json.dumps(model.to_dict(), allow_nan=False)
restored = HandcraftedRandomForestClassifier.from_dict(json.loads(payload))
np.testing.assert_array_equal(restored.predict_proba(X), model.predict_proba(X))
```

## 2. 設定與搜尋規則

[`configs/random_forest.json`](../configs/random_forest.json) 是獨立 profile：保留既有訊號、SFS 與其他分類器設定，明確選用四個模型，RF 為 128 棵樹、`sqrt` 特徵抽樣、bootstrap、OOB 診斷與固定 master seed。RF grid 是 `max_depth=[8, null]` × `min_samples_leaf=[1, 3]`，共四個候選。它是可檢視的起始設定，沒有以本研究 EEG 成績挑選或宣稱最佳。

| 參數 | 核心預設 | 定義 |
| :-- | :-- | :-- |
| `n_estimators` | `128` | 正整數，獨立訓練的 CART 數量 |
| `max_depth` | `None` | 根為深度 0；`None` 不另設深度上限，否則為正整數 |
| `min_samples_split` | `2` | 節點至少有多少個正有效權重 draws 才可分裂 |
| `min_samples_leaf` | `1` | 每個子節點最少的 draws 數；bootstrap 重複抽中會重複計數 |
| `max_features` | `"sqrt"` | 每節點獨立抽樣；`sqrt`／`log2` 向下取整且至少 1；`None` 為全部；正整數不得超過特徵數；`(0,1]` 浮點值取 `floor(fraction × p)` 且至少 1 |
| `bootstrap` | `True` | 對正有效權重的原始 rows 均勻、有放回抽樣；`False` 使用全部 eligible rows |
| `max_samples` | `None` | bootstrap 的 draws 數；`None` 為 eligible row 數，整數不得超過該數，`(0,1]` 浮點值取 `ceil(fraction × eligible rows)` |
| `class_weight` | `None` | 不加權、`"balanced"` 或明確類別權重；另可於 `fit` 傳入 `sample_weight` |
| `min_impurity_decrease` | `0.0` | `[0,1]` 實數；以整棵樹根節點權重正規化的最低 Gini 降幅 |
| `random_state` | `0` | 固定整數 seed，範圍 `[0, 2**64)`；整合式 CV master seed 進一步限制為 `[0, 2**32)` |
| `oob_score` | `False` | bootstrap 才能啟用；輸出 OOB counts、機率、coverage 與診斷 accuracy |
| `n_jobs` | `1` | 正整數或 `-1`；標準函式庫 threads 平行建樹，`-1` 使用可觀察 CPU 數 |

`bootstrap=False` 時不得設定 `max_samples` 或開啟 OOB。布林值不冒充整數參數。未知欄位、非有限數值、空或重複 roster、空 grid、重複候選及不合法參數均明確拒絕。依當次選定特徵數／inner-training 大小才能決定的 bounds，會在第一個 RF candidate fit 之前整體預檢。

Config 的 `random_forest.parameters` 是基礎參數；`random_forest.grid` 僅覆蓋列出的維度。最多 **64 個候選**；依「排序後的 grid keys × 原先宣告的 value 順序」產生候選。每個候選使用完全相同的 inner splits 與各折 seed；以 **各 inner fold 的 epoch accuracy 算術平均**選最佳，精確平手保留第一個候選。實際排序用標準函式庫 `Fraction(correct_count, n_test)` 計算精確平均，最後才轉 float 寫出，避免浮點加總將數學上的平手變成勝負。`random_state`、`n_jobs`、`oob_score` 不可放進 grid，以免搜尋隨機種子或執行設定。

Integrated class weights 接受 `null`、`"balanced"` 或同時具有 `"0"`／`"1"` 的正權重 mapping。獨立核心支援多類別及零類別權重；整合評估仍使用本研究二分類 label 契約。`sample_weight` 屬獨立 API，不會在 CV 流程偷偷推導受試者權重。

## 3. 手刻 CART 與森林的精確語意

### 分裂與機率

對節點 $t$，令類別 $k$ 的加權質量為 $W_{t,k}$，總質量為 $W_t$。實作使用：

$$
G(t)=1-\sum_k\left(\frac{W_{t,k}}{W_t}\right)^2,\qquad
\Delta(t)=G(t)-\frac{W_L}{W_t}G(L)-\frac{W_R}{W_t}G(R).
$$

每個候選特徵先 stable sort，再用累積 class masses 枚舉相鄰不同值的有效邊界；這是精確 sorted-boundary 搜尋，沒有 histogram approximation。每特徵的搜尋暫存空間為 $O(n)$，避免配置 samples × classes 的累積矩陣。每個 split 需同時滿足 leaf count、depth、非純節點及

$$
\frac{W_t}{W_{\mathrm{root}}}\Delta(t)
\;\ge\;\texttt{min\_impurity\_decrease}.
$$

預設門檻 0 允許零增益第一層，後續才能學到 XOR 等互動；不以「當下 gain 必須嚴格大於零」錯誤提前停止。以 iterative stack 建樹，不依賴 Python recursion depth。

節點特徵先以其獨立 RNG 排出 permutation，檢查前 `max_features` 個；只有其中完全沒有符合 leaf count 的不同值分界時，才往後找第一個具有有效邊界的特徵。不是遇到低增益就一直擴大抽樣。Tie 的共同基準是所有已檢查有效邊界中的最大 gain；距離此最大值不超過 `64 × float64 eps` 的候選，依較小 feature index、再較小排序邊界決定。先保留每特徵的最大 gain scalar，再重掃被選中的特徵，避免逐次近似平手導致門檻漂移，或保留全部特徵的排序資料。

切分規則為 `x <= threshold` 向左。Threshold 以 `lo/2 + hi/2` 計算，若相鄰浮點數的捨入使它不在 `[lo, hi)`，使用 `lo`；因此極端值、subnormal 與沒有可表示中點的相鄰值仍維持原始分組。

樹的葉節點輸出 class masses 正規化後的機率，森林取各樹機率的平均：

$$
\widehat p_k(x)=\frac{1}{B}\sum_{b=1}^B
\frac{W_{b,\ell_b(x),k}}{W_{b,\ell_b(x)}}.
$$

這裡明確採 **soft probability averaging**；Breiman 原論文分類森林的定義使用各樹 class vote，兩者在葉節點不純時可能得到不同分類。本實作不宣稱與其他函式庫逐 bit 或逐個預測相同。最終 `argmax` 平手採排序後 `classes_` 的第一類。

### 權重、亂數與狀態

`sample_weight` 需有限、非負且具有正總量。`balanced` 以 input sample weight 大於 0 的 rows 計算各類別筆數，類別倍率為 `n_active / (n_present_classes × class_count)`，再乘上 sample weight。有效權重為 0 的 rows 不參與抽樣或分裂，但其原始類別仍保留於 `classes_`。Bootstrap 的抽樣機率是 eligible rows 的均勻分布，權重另用於 class masses，兩個機制不混為一談。

權重總量、乘法 underflow／overflow，以及每棵樹實際重複 draws 的質量均有數值檢查。無法以 float64 可靠表示時會失敗，不會悄悄得到 NaN tree。整體權重縮放 `1e-200` 到 `1e200` 的代表案例有獨立 oracle 覆蓋。

每棵樹由 `SeedSequence` 產生獨立 seed，bootstrap 與節點特徵抽樣再分開 RNG。結果依固定 tree order 歸約，`n_jobs=1` 與 `2` 相同；增加 `n_estimators` 保留既有 tree prefix；不消耗 process-global NumPy／Python RNG。CV 再以 master seed、scope、comparison、subject、outer fold、inner/refit 階段推導 seed，記入 audit。固定參數與資料在已驗證環境內可重現；跨 NumPy／平台版本仍以 [reference numerical CI](reproducibility.md) 驗證。

Fitted arrays 採不可寫入的 bytes backing，公開取值另提供獨立 shape metadata，不能藉 `setflags` 或 `.shape` 改掉已訓練模型。失敗的 refit 保留上一個有效模型；通過驗證的 `set_params` 會使它回到未訓練狀態。Export 使用 fit 當時的參數快照，不把事後直接修改的 constructor attributes 冒充原訓練設定。

## 4. EEG 評估邊界

每個 outer fold 先用 **該 fold 的 training data** 執行既有 LDA SFS，所有選用的分類器共用同一組 ordered features。RF 直接使用這些特徵值，不需要 scaler。Inter 的 outer test 為整位受試者；inner tuning 也按受試者留一人，同人的全部 sessions 位於同側。Intra 保持既有同人內 epoch-stratified CV，不宣稱 leave-session-out。

RF 搜尋只拿到 outer-training `X`／`y`，選參數後才用完整 outer-training refit，最後預測 held-out epochs。搜尋期間關閉 OOB，OOB 不參與選參數。Inner score 仍是 **共用 SFS 之後的 selection diagnostic**，不是重新估計整套特徵選擇程序的獨立分數，也不是 participant-macro balanced accuracy。此 CV 流程的 test performance 由 outer held-out predictions 定義。

### OOB 與 importance 的解讀

一筆 row 的 OOB 機率只平均沒有抽到該 row 的樹。`oob_counts_` 保存樹數；零票 row 的機率為 NaN，JSON 為 `null`，不補成零或多數類。`oob_eligible_mask_` 排除零有效權重 rows；`oob_coverage_` 為 eligible rows 中有票者比例。`oob_score_` 是這些有票 rows 的 **unweighted accuracy**；完全沒有 eligible OOB votes 時為 NaN／JSON `null`。

EEG epochs 可能來自同人，且共同 SFS 已看過 outer-training 資料，所以 OOB 是訓練診斷，不能取代 participant LOPO，也不是特徵選擇之外的獨立評估。Class weights 不會把此診斷自動改成 weighted 或 balanced accuracy。

`feature_importances_` 由每個 node 的 root-mass-weighted Gini decrease 加總至對應特徵，每棵樹先正規化，再平均並正規化整個森林；沒有有效 split 時為全零。它是 training impurity importance，不是 held-out permutation importance，也不表示電極對的神經因果關係。CV audit 同時保存 selected feature names 及重要度，解讀時只對應那個 fold 的選定欄位。

## 5. API 與模型封存

| API／屬性 | 用途 |
| :-- | :-- |
| `fit(X, y, sample_weight=None)` | 二分類／多類別訓練；支援單一類別 |
| `predict(X)`／`predict_proba(X)` | 類別與依 `classes_` 排列的機率 |
| `score(X, y, sample_weight=None)` | Accuracy；明確提供 sample weights 時採加權平均 |
| `get_params(deep=True)`／`set_params(**params)` | 參數快照／驗證後設定 |
| `validate_parameters(n_features=None, n_samples=None)` | 不 fit 的設定預檢 |
| `to_dict()`／`from_dict(payload)` | 嚴格版本化、純資料 JSON 往返 |
| `estimators_`／`bootstrap_indices_`／`tree_seeds_` | 不可變 trees、原始 row draws 與逐樹 seeds |
| `classes_`／`n_classes_`／`n_features_in_`／`n_samples_fit_` | 訓練維度與類別順序 |
| `feature_importances_`／`oob_*` | 上節定義的訓練診斷；OOB 屬性只在啟用時存在 |

每棵 `CARTTree` 可檢視 children、feature、threshold、樣本數、加權樣本數、class counts、impurity、depth arrays，並直接 `predict_proba`。所有屬性均可由 JSON 重建。

輸入 `X` 需是有限實數的二維矩陣，以 float64 計算。沒有 missing-value routing、類別欄位編碼或自動補值；NaN、inf、complex、object/string features 會拒絕。超過 float64 精確整數範圍的 integer features 必須先由呼叫端明確處理。`y` 需是同型別 bool、整數、有限浮點數（最多 float64）或非空 Unicode labels；例如 `[1, "1"]` 不可被隱式合併為同一類。Standalone model 只記錄 feature dimension，使用者需保持欄位順序；managed CV 另將 ordered names 綁入 identity。

模型標記為 `algorithm_version=eeg-handcrafted-random-forest-1`、`schema_version=eeg-handcrafted-forest-json-1`。Loader 不執行 pickle 或任意 Python code，會檢查欄位集合、類別、tree topology、父子樣本／質量守恆、Gini、depth／leaf／gain 限制、seed／bootstrap 一致性、importance 與 OOB 結構。尺寸先對照實際 payload 長度，避免只憑偽造的 scalar dimensions 進行巨量配置。

**模型 JSON 本身沒有原始訓練 X。** 因此它可驗證模型結構與內部一致性，不能單靠 JSON 證明 split 確實來自某份 EEG。Managed CV 另外封存來源並執行下一節的 replay；兩者的證據範圍分開記錄。

## 6. Run identity 與可重播證據

沒有明確 `classifiers` 欄位的舊設定保持 `evaluation-2`。明確 roster 使用 `evaluation-3`，identity 與 `evaluation_contract.json` 綁定完整 classifier policy；缺少任一宣告 classifier 的 fold／prediction 都會拒絕完成或 resume。

每個啟用 RF 的 scope 另產生兩份受 manifest hash 保護的檔案：

| Artifact | 內容與驗證 |
| :-- | :-- |
| `random_forest_inputs.csv` | 完整輸入 feature table，其序列化 SHA 必須等於 identity 的 table SHA；保留 literal metadata 與 round-trip float values |
| `random_forest_audit.json` | 完整候選與逐 inner-fold seeds／membership／predictions／probabilities／scores、最佳候選／refit 參數、selected features、outer train/test epoch IDs、最終 forest JSON、importance 與 OOB |

Validator 重建 inner／outer partitions、核對 candidate 順序及 seed 推導、重算每折 accuracy／候選平均／first-maximum 決策。它還會從來源 X routing 每棵 final tree 的 bootstrap draws，重算節點 class masses／樣本數、outer predictions／probabilities，以及全部 OOB votes／coverage／accuracy。即使重新計算外層 manifest hashes，語義不一致的產物仍會失敗。

Audit 的 `verification_scope` 明示為 **source-bound outer-model prediction replay + inner-selection record consistency**；驗證器不重新 fit 所有 inner forests，也不宣稱重新搜尋 final tree 的每個最優分裂。Manifest 是可稽核來源記錄，不是外部簽章或防止有意偽造全部來源的證明。完整原子發布與相容性規則見 [run provenance](run_provenance.md)。

## 7. 驗證與執行成本

測試在 [`test_random_forest_core.py`](../tests/test_random_forest_core.py)、[`test_random_forest_oracles.py`](../tests/test_random_forest_oracles.py)、[`test_random_forest_integration.py`](../tests/test_random_forest_integration.py) 與 [`test_random_forest_cli.py`](../tests/test_random_forest_cli.py)：

- 用 `Fraction` 有理數與窮舉邊界獨立驗證 weighted multiclass Gini，不用現成 forest 當正確答案。
- 重算 bootstrap node masses、leaf probabilities、soft voting、每樹 importance 與 OOB；驗證 XOR、常數特徵、單一類別、極端／相鄰 float、權重縮放及零票情況。
- 驗證 serial／threaded／tree-prefix 重現、global RNG 隔離、唯讀資料與 shape 隔離，以及失敗 refit 的狀態保留。
- 破壞 JSON schema／topology／mass／seed／OOB、feature 來源、inner membership 與 classifier coverage，確認重新 hash 仍無法繞過一致性檢查。
- 追蹤真實 fit membership，擾動 held-out features，驗證群組邊界；禁止 RF path 呼叫現成 sklearn learner／search。
- 既有三分類器 numerical fixture 與歷史 inventory 繼續執行，防止加入 RF 改變原數值。

Reference CI 另執行 [`run_random_forest_smoke.py`](../tests/run_random_forest_smoke.py)：實際 synthetic raw → QA → features → 四分類器 `03`／`04` subprocess → archive replay → immutable resume → plots。Fixture 為 3 位 synthetic people、每人 2 sessions、48 epochs；每個 scope 192 筆 predictions，intra 24／inter 12 個 classifier-fold rows。所有模型與搜尋都實際執行，沒有 mock estimator、clock 或 code identity。

同一 smoke 再以 300 × 126 synthetic matrix 建立 32 棵 threaded trees，保存可重建 inputs／模型與觀察到的時間，不設 accuracy 或速度門檻。這是有限工程 workload，不能推論真實 EEG 表現或所有硬體上的耗時。樹的訓練以逐節點排序為主，Python threads 的加速受工作量與 NumPy 執行影響；沒有宣稱能匹配編譯式函式庫的效能。完整搜尋成本另乘上候選數、inner folds 與 outer folds，且原 SFS／其他模型仍有各自成本。

可在已驗證 reference interpreter 執行：

```bash
python -m pytest -q tests/test_random_forest_core.py \
  tests/test_random_forest_oracles.py tests/test_random_forest_integration.py \
  tests/test_random_forest_cli.py
python tests/run_random_forest_smoke.py \
  --out-dir outputs/random-forest/smoke-001
```

## 演算法來源

1. Breiman, L. (2001). *Random Forests*. Machine Learning, 45, 5–32. [作者原文](https://www.stat.berkeley.edu/~breiman/randomforest2001.pdf)、[DOI](https://doi.org/10.1023/A:1010933404324)。Bootstrap 與隨機特徵的森林架構依此建立；上文另明列本實作的 probability-averaging variant。
2. Breiman & Cutler. [Random Forests — Description](https://www.stat.berkeley.edu/~breiman/RandomForests/cc_home.htm)。提供 bootstrap、node-level random features、OOB 與 Gini importance 的原始方法說明；本專案對相關 EEG epochs 與共同 SFS 的推論限制依第 4 節處理。
