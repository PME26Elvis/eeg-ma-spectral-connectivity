# Raw data 放置方式

此目錄內容被 `.gitignore` 排除，**不要把受試者 raw EEG commit 到 Git**。

每一位受試者保留收案面板輸出的完整 session 資料夾，例如：

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
└─ subject05/
   └─ ...
```

資料夾名稱可以任意；程式會遞迴尋找 `eeg_raw.csv`，並把其所在資料夾視為一個 session。
