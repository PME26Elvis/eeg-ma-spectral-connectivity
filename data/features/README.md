# Feature tables

此目錄用來存放可版本控制的 feature CSV。

執行：

```powershell
python scripts/02_extract_features.py
```

會產生：

```text
<subject>_features.csv
features_all.csv
raw_validation.csv
```

每個 epoch 一列。Metadata 後接：

- 42 個 `BP__...` 欄位
- 126 個 `COH__...` 欄位

總計 168 個 EEG features。

這些檔案不包含原始逐 sample EEG 波形，可加入 Git；`data/raw/` 則永久忽略。
