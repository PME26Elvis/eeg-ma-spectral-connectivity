# 固定研究環境與數值重現驗證

本文件對應 [Issue #10](https://github.com/PME26Elvis/eeg-ma-spectral-connectivity/issues/10)。Reference environment 是**自本版本起可驗證的新研究執行環境**；它不回填原五人研究的歷史環境，也不聲稱已重建未公開的 raw EEG 全流程。

## 1. 三種證據，各自回答不同問題

| 檢查 | 輸入與環境 | 通過的意義 |
|---|---|---|
| Compatibility CI | Python 3.10／3.12，依 `pyproject.toml` 寬版本範圍解析 | 支援範圍內的 API、資料契約與失敗處理沒有 regression；不是固定歷史環境 |
| Reference numerical CI | 精確 CPython 3.12.14、Linux x86_64、hash-pinned wheels、單執行緒 | 在宣告的 reference stack 下，真實 synthetic pipeline 符合事先保存的數值與身分契約 |
| Controlled dependency comparison | 同一分析程式與 fixture，兩個明確版本的 environment 定義 | 依賴改變造成的數值、選擇、預測與報表差異有可檢查紀錄；不自動接受新 baseline |

`pyproject.toml` 保留廣泛相容性範圍。正式研究使用 [`environment/reference.json`](../environment/reference.json) 及其指向的 hash lock；單純安裝寬版本 dependencies 不能取得 reference 驗證狀態。

## 2. Reference stack 與可重建範圍

| 元件 | 固定版本 |
|---|---:|
| CPython | 3.12.14 |
| NumPy | 2.3.5 |
| pandas | 2.2.3 |
| SciPy | 1.17.0 |
| scikit-learn | 1.8.0 |
| joblib | 1.5.3 |
| Matplotlib | 3.10.8 |
| pytest | 8.4.2 |

完整 lock 同時固定所有解析後的 transitive dependencies 與 pip／setuptools／wheel；上表只是方便閱讀的主要元件，不能代替 lock。每個允許的 wheel 有實際 SHA-256，安裝時拒絕未列出的 dependency resolution、不同 artifact 或 source build。[1][2]

Reference 的 platform target 是 Linux x86_64。CI 固定 Ubuntu 24.04 runner label；runner image、kernel、libc、CPU 與 native libraries 的實際資訊另記錄在環境證據中。Hosted runner label 仍可能更新，因此不宣稱已固定完整 VM image，亦不要求不同 CPU／BLAS 的結果逐 bit 相同。

Windows 原生環境可用於相容性開發。要重建這個 Linux wheel lock，請使用相符的 Linux／WSL 環境或 reference CI，不要把 Linux-only lock 當成跨 OS 通用 lock。

## 3. 安裝與驗證

以下命令在 repository root、Linux x86_64 執行。`python3.12` 必須實際指向 **3.12.14**；不要用其他 patch version 冒充。使用新的隔離 venv，避免修改系統 Python。

```bash
python3.12 -m venv .venv
. .venv/bin/activate
python -c 'import platform; assert platform.python_version() == "3.12.14"'

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONHASHSEED=0
export MPLBACKEND=Agg

mkdir -p outputs/install-reference
python -m pip install --force-reinstall --require-hashes \
  --only-binary=:all: --no-deps \
  --report outputs/install-reference/pip-install.json \
  -r environment/reference-linux-x86_64-py312.lock
python -m pip install --no-deps --no-build-isolation -e .
python -m pip check

export EEG_MA_INSTALL_REPORT="$PWD/outputs/install-reference/pip-install.json"
python scripts/18_environment.py verify --out outputs/environment-reference.json
python scripts/19_validate_numerics.py \
  --out-dir outputs/numerics-reference --require-reference-environment
```

`--force-reinstall` 讓 receipt 包含所有固定 dependencies，而非只列出原環境缺少的套件。`--require-hashes` 檢查取得的 artifacts；`--only-binary` 禁止臨時 source compilation；`--no-build-isolation --no-deps` 防止 editable project installation 偷偷解析另一套 build/runtime dependencies。Local project 的身分由 Git commit 與 source fingerprint 表示。[1][2]

環境檢查輸出必須寫入新路徑；同名既有證據不會被覆寫。每次正式 run 都保留安裝 receipt、環境 JSON、source commit、輸入與設定身分。不要把 receipt 從另一個不相干的環境複製過來當作實際安裝證明。

## 4. Run／model 的環境身分

環境證據分開記錄「預定環境」與「目前觀察到的環境」：

- `lock_id` 識別 reference definition 與 requirements 的內容；只表示有一個明確的預定 stack。
- `environment_id` 識別觀察到的 Python、packages、native libraries、執行緒設定、安裝來源及驗證結果。
- Reference status 必須由實際檢查產生。只有檔案存在、版本字串相同，或手寫 `passed=true` 都不足以取得正式驗證。
- 原 W1 run identity 使用這份完整環境證據。Dependency、lock、runtime 或 thread configuration 改變後，舊 output 不會被當成相同 run 直接 resume。

版本之外還檢查安裝 receipt 的 wheel hash、package metadata 及已安裝內容。Native library 記錄包含 BLAS／OpenMP 實作、版本、雜湊及 thread count；capture 先一致地載入數值 backend，避免第一次模型運算才出現新 library、造成假環境變動。

正式 protocol 的 `freeze.environment_policy` 為 `verified_reference_environment`。Primary／reference training manifests 同時綁定：

| 欄位 | 必須一致的證據 |
|---|---|
| `environment_sha256` | 完整環境 JSON 的原始 bytes |
| `environment_id` | 完整觀察證據的 canonical identity |
| `environment_lock_id` | 該環境所驗證的 reference lock identity |
| 其他既有 training hashes | Development data、feature order/schema、analysis config、eligibility 與實際 fitted model |

保存後的 protocol lock 驗證使用其內嵌 definition、requirements、receipt 與觀察證據，不要求驗證者目前恰好也在相同 Python 上。相反地，開始新的 reference run 時必須檢查**當前實際環境**。這兩種檢查不能互相代替。

`configs/confirmatory_protocol.json` 的 environment artifact 仍保留待實際 model training 產生的證據位置。Reference CI 的測試環境不是尚未訓練的模型之 training history；#11 產生 fitted artifacts 時必須同時保存其真實環境。scikit-learn 也不保證不同套件版本間的 persisted model loading 相容性。[3]

## 5. Synthetic raw-to-report fixture

Public CI 只用程式產生的 deterministic synthetic acquisition；不下載、不假設可取得 private raw EEG，也不使用原五人的 performance 調整 fixture。

流程使用 repository 的實際 QA、feature extractors、grouped evaluation 與 publication validators：

1. 建立 3 個 synthetic subjects、每人 2 個 sessions，共 48 個 epochs 的 raw samples、events、trials、rest epochs 與 metadata。
2. 透過共用 QA 檢查 event/trial bijection、sample continuity、clock、身份與 channel contract。
3. 提取完整 168 個 baseline 與 144 個 extension features，保持既有七 analysis channels 的方法契約。
4. 明確固定小型 evaluation subset、最多一個 SFS feature、單一 RBF grid point，再執行外層 LOPO 與 training-only inner grouped CV。
5. 執行 baseline／phase 兩個 feature families 與 LDA／SVM／KFDA 三個 classifiers，共 18 個外層 folds、288 筆 predictions；保存 stable epoch identity、splits、run manifest，再產生並驗證 report。
6. 以保存的 expected values、tolerances、選取結果與 coverage 契約比較。

這是 **grouped-CV regression fixture**。它不等於 #11 的固定模型 external prediction／score 管線，也不是 N60 的 confirmatory study。Fixture 的小規模與有限 SFS/grid 是測試設計；正式研究的設定不會被改成 smoke-test 參數。

### 數值容差與離散結果

浮點 features／summary 以事前保存的 absolute／relative tolerance 比較；資料身分、feature order、split membership、prediction coverage、選取特徵與離散 predictions 則使用對應的精確契約。容差有明確理由與範圍，不以「檔案存在」或寬鬆到失去意義的 allclose 判定成功。

完整 expected values 與產生時的實際環境證據保存於 [`tests/fixtures/numerical/reference.json`](../tests/fixtures/numerical/reference.json)。目前容差如下；比較公式為 `abs(actual - expected) <= atol + rtol * abs(expected)`。

| 比較內容 | `rtol` | `atol` | 理由 |
|---|---:|---:|---|
| 全部 312 個 features 的 mean／std／min／max，以及預先指定 sentinel features 的逐 epoch 值 | `1e-8` | `1e-10` | 容納 FFT、filter、Hilbert、power 的有限浮點差異 |
| Summary mean／std／counts／pooled accuracy | `1e-12` | `1e-14` | 僅容許彙總 roundoff；counts 另有完整 coverage 契約 |
| Fold accuracy、predicted labels | `0` | `0` | 離散預測結果必須相同 |
| Subject／session／epoch identity、feature order、selected features、prediction coverage | 精確比較 | 精確比較 | 不屬於浮點誤差 |

不同 platform／BLAS 可能有數值差異，因此整份 CSV、runtime JSON 或 run ID 不要求跨主機 byte-identical。依賴升級若改變 feature selection 或 predictions，應留在 comparison evidence 中接受審查，不能自動重寫 expected fixture 來讓 CI 變綠。

## 6. 高風險 regression 對照

| 風險 | 既有與新增驗證 |
|---|---|
| 舊環境 output 被錯誤 resume | Run identity 對 environment／lock／config／data／split 改變拒絕重用 |
| Partial／被改寫的 report | 完整 coverage、predictions-derived metrics、artifact hashes 與 publication transaction |
| Duplicate／missing events 或 sample gap | 共用 raw QA，失敗時不發布 feature prefix |
| 同人的不同 sessions 滲入 test/train | Outer subject grouping 與 inner split membership，fit-boundary spy |
| Feature order／schema 漂移 | Exact ordered schema 與 locked numerical fixture |
| 缺少或矛盾的報表列 | Prediction coverage／重算數值與 validated report 檢查 |
| 有 lock 但實際環境不符 | Python／packages／threads／wheel receipt／installed content／native backend gate |
| Model 與環境證據不相干 | `environment_sha256`、`environment_id` 及 `environment_lock_id`，重新 hash 不會修復語義錯配 |

Tests 使用相對應的破壞情境驗證拒絕路徑。原有 canonical inventory test 持續核對歷史 feature/results bytes，且 historical environment 欄位仍為 `unknown`／`legacy_unverified`。

## 7. CI 與依賴升級的操作原則

Compatibility 與 reference numerical jobs 名稱、環境來源與證據分開。自動 CI 只有有限 synthetic workload，不執行正式研究的完整 SFS／grid。Docs-only 修改不觸發昂貴研究矩陣；更新 numerical fixture、環境 lock、source、tests 或 workflow 時才觸發相應驗證。

手動 dependency comparison 需要 explicit candidate commit、run identity 與變更理由；分析程式與 synthetic input 固定，只從指定候選版本取得 environment 定義。輸入由資料欄位處理，不拼入可執行 shell。Reference 與 candidate 分別使用隔離環境、完整 receipt 與不可覆寫輸出，保存各自 lock／environment identity 及比較結果。

### 操作受控比較

Workflow 合併到 default branch 後，在 GitHub **Actions → Controlled dependency comparison → Run workflow** 選擇分析程式版本，填入以下欄位：

| 輸入 | 要求 |
|---|---|
| `candidate_commit` | 同一 repository 中含候選 `environment/reference.json` 與 lock 的完整 40 字元 commit SHA |
| `run_id` | 唯一比較名稱，1–64 個英文字母、數字、連字號或底線 |
| `reason` | 在查看候選結果之前寫明依賴變更理由 |

Workflow 只讀取候選版本的 environment 定義；兩次執行使用所選分析版本的同一份 source 與 numerical fixture。Candidate 會重用並核對 baseline 實際 raw bytes，避免依賴變動連 synthetic input generation 都一起改掉。保存的 dependency comparison 同時帶 baseline 證據，可重新計算比較結果，不能只修改外層 hash 或 `passed` 欄位。

下載 `dependency-comparison-<run_id>-<attempt>` Actions artifact，檢查 `request/comparison_request.json`、兩份 install receipts／environment evidence，以及 `baseline/numerical_report.json` 與 `candidate/numerical_report.json`。Artifact 名稱中的 `run_id` 是 GitHub Actions run ID；自訂比較名稱保存在 request 內。數值漂移會保留診斷並令 job 失敗，不會覆寫 reviewed reference。Reference evidence 與 dependency-comparison artifacts 保留 90 天；需長期保存的正式證據應另行封存。

本機可先執行同一 preflight；analysis checkout 與 candidate environment 檔案都必須符合宣告的 commit，output 必須是新的 `outputs/` 子目錄：

```bash
python scripts/20_prepare_environment_comparison.py \
  --candidate-checkout .environment-candidate \
  --candidate-commit "$CANDIDATE_COMMIT" \
  --run-id "$COMPARISON_RUN_ID" --reason="$COMPARISON_REASON" \
  --out-dir outputs/dependency-comparison/request
```

依保存的 `request/baseline/` 與 `request/candidate/` profiles，各建立一個符合指定 Python patch 的 venv。分別設定 `EEG_MA_REFERENCE_SPEC` 與 `EEG_MA_INSTALL_REPORT`，依第 3 節方式安裝及驗證。Baseline 用 script 19 正常執行；candidate 額外提供 `--compare-to outputs/dependency-comparison/baseline/numerical_report.json`。完整可執行步驟見 [workflow](../.github/workflows/dependency-comparison.yml)。

本次不提供可直接在 public CI 啟動 private-raw 正式研究的 workflow。若未來新增，必須明確提供 input／run／protocol 身分、授權資料位置及保存產物的規則；不能默認 raw 可公開下載，也不能將日常 PR CI 變成新的研究搜尋。

依賴升級步驟：建立新 environment definition／hash lock，執行原 fixture與受控比較，檢查科學數值和離散 predictions 的差異，經 review 後以新 commit 採用。舊 lock／run evidence 不覆寫；新研究需要新的環境身分。

## 8. 適用限制與下一個工作包

CI 綠燈代表這個有限 synthetic fixture 和現有契約在指定環境下通過。它不能證明真實 acquisition facts、原始 EEG 品質、研究倫理核准、固定模型泛化或 online BCI efficacy。

#10 交付新研究環境與數值 CI；#11 仍需完成 development-only fit／export／predict／score，#12 仍需完成最終 participant-level reporting integration。前瞻協定仍由 [protocol gate](protocol_gate.md) 管理，模型、acquisition、核准等必要證據不足時，正式研究保持 blocked。

## 官方依據

1. pip, [Repeatable installs](https://pip.pypa.io/en/stable/topics/repeatable-installs/): exact versions、transitive dependencies、hashes 與 platform-specific wheels 的範圍。
2. pip, [Secure installs](https://pip.pypa.io/en/stable/topics/secure-installs/): `--require-hashes`、所有 dependencies 的 pins／hashes、`--only-binary`。
3. scikit-learn, [Model persistence](https://scikit-learn.org/stable/model_persistence.html): 持久化模型與環境版本的相容性限制。
