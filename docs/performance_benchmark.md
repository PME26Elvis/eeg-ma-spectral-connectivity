# 手刻 Random Forest／LightGBM：效能基準、資源預檢與數值一致性

本文件對應 [Issue #26](https://github.com/PME26Elvis/eeg-ma-spectral-connectivity/issues/26)。**資料全為 synthetic engineering workloads，絕非 EEG 泛化準確率證據。** 不會覆寫原 243 份歷史資料／結果、A/B 協定或教授簡報 commit。

## 1. 可重現效能基準

[benchmarks/run.py](../benchmarks/run.py) 使用固定 seed=260126 生成無個資的 dense／sparse／correlated／constant features，並保存 input NPZ 與 SHA-256。Full preset 是 **16 個代表性 workloads**（rows 300、600、3,000、6,000；features 21、126、168、312，每組指定一種 shape），不是完整 64 種笛卡兒積。兩個 learner 各跑 jobs=1／2／4、一次 warm-up 與 **3 次非 warm-up**獨立 subprocess 測量。基準模型限成本：RF 六棵、depth 5；LightGBM 六輪、最多五葉，並非研究預設超參數。

每筆保存 CPU/NumPy/BLAS/affinity、fit wall/CPU、predict、JSON export+load、source-bound model audit、process peak RSS、model JSON bytes、完整模型與預測 SHA-256。峰值 RSS 包含 Python import／套件載入，不能冒稱單次 fit 的淨記憶體分配。輸出 summary.json、summary.csv、measurements.jsonl、calibration.json、選配 cProfile；三個樣本的 empirical p95 僅描述此次測量，不是可靠 tail-latency 統計。

~~~bash
# 有界 smoke（CI）
python benchmarks/run.py suite --preset ci --jobs 1 2 --repeats 3 \
  --warmups 1 --profile --out-dir outputs/bench/ci-001

# 全組合，建議 Linux/WSL2 + 固定研究環境；此模式計算量較大
python benchmarks/run.py suite --preset full --jobs 1 2 4 \
  --repeats 3 --warmups 1 --timeout 240 --profile \
  --out-dir outputs/bench/full-001

# 比較兩個「相同 runner、相同 fixture/CPU/Python/BLAS」的結果
python benchmarks/run.py compare \
  --baseline outputs/bench/before --candidate outputs/bench/after \
  --output outputs/bench/comparison.json
~~~

[同機 baseline→candidate 工作流程](../.github/workflows/performance.yml) 對 PR base 與本次程式在**同一 GitHub runner**執行三次測量，檢查完全相同的 inputs、模型 JSON、predict_proba bytes，並保存 cProfile。數值不一致會直接失敗；效能幅度不以不穩定的共享 runner 門檻控制 correctness。[完整 16-case benchmark](../.github/workflows/performance-full.yml) 改採手動 workflow_dispatch，分 rows=300/600/3000/6000 四個 CI jobs，可選 1/2/4 workers 且有每次 fit timeout。不同 CI jobs 不能互相比絕對秒數。

## 2. Resource planning／dry-run

[performance.py](../src/eeg_ma/performance.py) 以 schema 2.0 feature 身分、實際 participant-grouped LOPO／intra splits、每折候選網格、特徵數及 trees/rounds 計算結構性 work units、預期 fit 數量和 heuristic RSS envelope。沒有真實測量的時間欄位是 null，並不是 0 秒。

~~~bash
python scripts/04_inter_subject.py \
  --features outputs/new/features/features_all.csv \
  --config configs/lightgbm.json \
  --out-dir outputs/new/results/inter \
  --dry-run --cost-plan-out outputs/cost/plan-001

# 超限會以 exit 2 阻擋訓練，不發布 complete result。
python scripts/04_inter_subject.py \
  --features outputs/new/features/features_all.csv \
  --config configs/lightgbm.json \
  --out-dir outputs/new/results/inter \
  --cost-max-estimated-rss-mib 2048 \
  --cost-plan-out outputs/cost/memory-001

# 只在同 CPU、相同 learner source、已做真實 benchmark 校準的
# tree-only（不含 LDA-SFS/SVM/KFDA）分析上使用時間預檢
python scripts/04_inter_subject.py \
  --features outputs/new/features/features_all.csv \
  --config configs/tree-only-previously-frozen.json \
  --out-dir outputs/new/results/inter-tree \
  --cost-calibration outputs/bench/ci-001/calibration.json \
  --cost-max-estimated-seconds 1800
~~~

成本 flags 均為 opt-in；不設定時原三／四／五模型的 CLI、run identity、resume 與分析邏輯不變。Dry-run 不 fit、不創建研究 output。另行保存的 cost_plan.json 成功時是 preflight_only；被 budget 拒絕時是 blocked，且 complete_run_published=false。即使 user 指定完整結果目錄也不可建立假 complete run。

### 校準與估計的邊界

- Work units 只是每次 fit 的結構性比例指標，**不是實測 seconds、FLOPs 或 GPU hours**。Runtime calibration 必須對應相同 CPU／Python／NumPy／BLAS、同一 RF／LightGBM 程式 bytes、保存的 measurement JSONL checksum 和逐筆 work-unit 比例。內建 safety factor 2，仍是外推，並非硬性 runtime timeout。
- 只有 tree-only 且不執行 SFS 的流程才能使用此樹模型時間校準做**整條分析**的時間預檢；若同時有 LDA-SFS、SVM、KFDA，時間 budget 一律 fail closed（其訓練時間未被這份 calibration 測量），但仍列出各類模型／SFS 的 fit 數量。
- Memory planning 是 heuristic envelope，不是 Linux cgroup/RLIMIT 形式的 RSS hard limit。需強制機器資源上限時還應使用 OS/container 工具；測量不能假裝比預估更有保證。
- 任務的最長運算時間可能超出 soft preflight 推估。所有大量 benchmark 另由獨立 process timeout 和 GitHub job timeout 保護。

## 3. Hotspot 的保數值最佳化

RF CART [原始實作](../src/eeg_ma/random_forest.py) 每個節點要檢查數個 sampled features。優化將 class labels 的 unique set 提升到每個節點只算一次，同時快取原始最高 gain 的 sorted feature scan；若全域 tolerance band tie-breaking 選的是不同、較小 feature index，**仍走原本二次掃描**。全部 Gini、threshold、隨機數、bootstrap、OOB、class weights、保留／停止規則都未更改。

同 runner baseline/after 測量須提供**至少三次**真實數值與 peak RSS；Issue 的理想目標是中位數縮短至少 20% 且 RSS 不惡化。若量測沒有達成，應報告負結果、實際瓶頸與標準差，**不允許**換不等價演算法、改參數或放寬數值 golden 來達標。

本 PR 的相容性 CI／固定 [reference numerical CI](../.github/workflows/reference-numerics.yml) 必須保留原有 numerical golden、RF/LightGBM independent oracles、serial-threaded 一致性、safe JSON、source audit、#25 strict train replay；並核對 243 個 canonical artifact SHA 不變。即使 smoke 暫時通過，也不能因此推論新資料的 accuracy、模型優越性或正式研究統計結論。
