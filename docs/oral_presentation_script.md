# 口頭報告講稿草稿

> 依 `docs/presentation_plan.md` 的 10 張內容投影片整理。文字設計成可以直接口頭說明，再依實際報告時間刪減。建議主報告約 8–10 分鐘，Q&A 另外保留。

## Slide 1 — 資料與問題設定

這個專題使用 HNC EAmp 8-channel EEG，第一階段分析選 7 個 frontal channels：FP1、FP2、F7、F3、Fz、F4、F8。

目前共有 5 位受試者。每位受試者的流程是 Rest1、Type1、Rest2、Type2。Rest1 和 Rest2 各切成 30 個 5 秒 epoch；Type1 和 Type2 只取答對 trial 的 5 秒 calculation period，各 30 個。因此每位受試者有 120 個正式 epoch，總共有 600 個 epoch。

我主要做兩個二分類：Rest1 vs Type1、Rest2 vs Type2。分析同時看 intra-subject 與 inter-subject，前者是同一個人的資料內做 5-fold CV，後者使用 leave-one-participant-out，也就是每次留一個完全沒進 training 的受試者當 test。

## Slide 2 — Poster / Lab baseline

分析先從實驗室既有流程開始。連續 EEG 做 2–50 Hz band-pass，再切成 5 秒 epoch。Feature 包含 42 個 Band Power 與 126 個 Coherence，共 168 個。

接著做 training-only SFS，再分別送進 LDA、RBF-SVM、KFDA。SFS、normalization 和 model tuning 都限制在 outer training data 裡面，避免 test leakage。

Baseline 的結果很明顯：Type1 的 intra-subject accuracy 大約 0.80 到 0.82，但 inter-subject 只有大約 0.52 到 0.53。Type2 也有類似差距，intra 大約 0.71，inter 大約 0.49 到 0.53。

這表示同一個人的 Rest 和 Mental Arithmetic 是有可分性的，但換到另一個人時，原本的 BP+COH feature 泛化得不好。後面的分析就沿著這個問題往下做。

## Slide 3 — BP / COH decomposition 與 E1 asymmetry

我先把 baseline 拆成 BP-only、COH-only、BP+COH，看是不是某一個 feature family 特別拖累跨人表現。結果三種在 inter-subject 都還是接近 chance，所以沒有看到單靠 BP 或單靠 COH 就能解決的現象。

接著利用 frontal montage 的左右對稱性做 E1 hemispheric asymmetry。三組 electrode pair 是 FP1-FP2、F3-F4、F7-F8，每個 band 計算 log(BP right) 減 log(BP left)，總共 18 個 asymmetry features。

E1 的 inter-subject 改善很小，Type1 甚至略微下降；Type2 只有大約 1 到 2 個百分點的變化。這組結果就記錄成一個負結果，後續沒有再擴大 asymmetry variants。

## Slide 4 — E2：加入 PLV

第二個 extension 是 Phase Locking Value。PLV 直接描述兩個 channel 的 phase difference 是否穩定。

公式是：PLV 等於 mean of exp(j delta phi) 的 magnitude。

7 個 frontal channels 有 21 組 pair，再乘 6 個 frequency bands，所以總共有 126 個 PLV features。

在 Type1 的 inter-subject LOPO，加入 PLV 後三個 classifier 都有明顯提升。Poster baseline 大約 0.52 到 0.53，BP+COH+PLV 變成大約 0.66 到 0.69。

Per-subject 檢查也顯示 5 個 held-out subjects 裡面有 4 個明顯改善，剩下 1 個大致持平。這表示 group mean 的提升沒有只靠單一受試者拉高。

Type2 沒有出現同樣幅度的改善，所以 PLV 的主要 observation 集中在 Rest1 vs Type1。

## Slide 5 — PLV decomposition

E2 的 feature set 是 BP+COH+PLV，所以我下一步把它拆開，分別測 PLV only、BP+PLV、COH+PLV、asymmetry+PLV，以及全部 features。

Type1 inter-subject 的結果顯示 PLV-only 已經可以做到大約 0.67 到 0.71，和完整 E2 相近，KFDA 和 SVM 甚至略高。

加入更多 feature families 沒有形成一致的額外提升。全部 312 features 的 E3，在 intra-subject Type1 可以到大約 0.86 到 0.88，但 inter-subject 反而掉到大約 0.61 到 0.65。

這裡可以看到一個很重要的差異：一些 feature 對單一受試者很有辨識力，但在不同受試者之間未必穩定。PLV-only 在目前資料上是一個比較精簡、跨人結果也比較好的 representation。

另外，SFS 在含 PLV 的 Type1 experiments 中，每個 LOPO fold 都有選到 PLV family。前面最常出現的單一 features 也集中在 alpha band，因此最後做 frequency-band validation。

## Slide 6 — Band-wise PLV validation

這一輪把六個 bands 全部單獨跑一次，每一組只使用 21 個 PLV pair features。

Type1 三個 classifier 的平均結果是：alpha 0.741、gamma 0.687、beta-high 0.637、delta 0.592、beta-low 0.584、theta 0.549。

Alpha-only 的三個 classifier 分別是 KFDA 0.733、LDA 0.743、SVM 0.747，而且 standard deviation 大約 0.065 到 0.081，比 all-band PLV 還低。

所以前面 SFS recurring alpha features，到了完整 six-band comparison 仍然得到一致方向的結果。這讓 alpha-band PLV 成為目前資料裡最穩定的 single-band representation。

旁邊的 network 圖只表示哪些 alpha PLV pair 被 SFS 反覆選到，以及 selection frequency。這張圖不用解讀成因果 connectivity。

## Slide 7 — iPLV robustness

PLV 有一個重要的限制：如果兩個 channel 有穩定的 zero-lag phase relation，PLV 也會很高。Common reference、volume conduction、shared source 都可能造成這種情況。

所以最後用 iPLV 做 robustness check。iPLV 取 mean complex phase-locking vector 的 imaginary part magnitude，exact zero-lag 和 pi-lag 會被壓低。

Type1 的結果很清楚：Poster baseline 大約 0.52 到 0.53；PLV-only 約 0.67 到 0.71；alpha PLV 約 0.73 到 0.75；iPLV 則只有大約 0.44 到 0.47。

因此目前 PLV 的分類增益高度依賴 zero- 或 near-zero-lag component。

這個結果本身不能判定來源。可能包含 reference、volume conduction、共同訊號來源、frontal artifact，也可能包含真正接近零相位差的同步神經活動。以目前 7 channels、5 subjects 的資料無法把這些因素分離，所以報告中把 PLV 當作有效 classification representation，對 functional connectivity 的生理解讀保持保守。

## Slide 8 — Type1 / Type2 與 behavioral context

Type2 的 behavioral accuracy 幾乎都是滿分，而且 median reaction time 比 Type1 快很多。

這表示 Type2 對受試者來說比較容易，也可能對應到較弱的 task-state difference。EEG 分類也呈現相同方向：Type2 的 phase-connectivity improvement 明顯小於 Type1。

目前這只能當作一致的 behavioral context，還不能從現有資料證明「因為 Type2 比較簡單，所以 EEG 一定比較難分」。

## Slide 9 — 研究結論

這個專題目前有幾個主要結果。

第一，Poster / Lab 的 BP+COH baseline 在受試者內有明顯可分性，但跨受試者泛化接近 chance。

第二，frontal hemispheric asymmetry 沒有帶來穩定改善；PLV 對 Type1 的 inter-subject LOPO 有明顯增益，而且在多數 held-out subjects 與三個 classifiers 中方向一致。

第三，PLV decomposition 顯示 PLV-only 已能保留主要 improvement，加入更多 feature families 對 cross-subject 沒有一致好處。

第四，band-wise validation 中 alpha-only PLV 的平均表現最高，而且跨 subject variation 較低。

第五，iPLV robustness 顯示這個 PLV gain 高度依賴 zero-/near-zero-lag component，所以目前最安全的結論是：alpha-band PLV 對這批 Type1 EEG 提供了有用的跨受試者 classification feature；它的生理 connectivity 來源還需要更多資料確認。

## Slide 10 — 限制與後續工作

目前最大的限制是受試者只有 5 位。LOPO 的外層只有 5 個 test units，而且 E1、E2、decomposition、band-wise validation 都是在同一批資料上完成。

第二，只有 7 個 frontal channels，因此 spatial connectivity 的解讀能力有限。

第三，PLV 類 feature 會受 reference、volume conduction、common source 與 frontal artifact 影響。iPLV 只是一個 robustness check，不能完全解決這些問題。

後續如果能增加受試者，我會先固定目前方法，使用新的資料做 confirmatory validation。資料量夠大之後，再考慮 domain alignment、更多 channels 或更完整的 connectivity analysis。

目前這批 5-subject data 的 feature exploration 已停止，後續工作集中在結果呈現與報告整理。

---

# 短版 5 分鐘講法

如果報告時間被壓縮到約 5 分鐘，可以保留以下順序：

1. 30 秒：資料、兩個 comparison、intra vs LOPO。
2. 45 秒：Poster baseline，重點講 intra 約 0.8、inter 約 0.52。
3. 30 秒：E1 asymmetry 無明顯改善。
4. 60 秒：E2 PLV 將 Type1 inter 拉到約 0.66–0.69。
5. 45 秒：PLV-only 保留主要增益，all-features 對 inter 反而較差。
6. 45 秒：alpha-only 約 0.74，六 bands 中最高。
7. 45 秒：iPLV 約 0.44–0.47，說明 zero-lag dependence 與解讀限制。
8. 30 秒：n=5、7 frontal channels、後續需要 independent subjects。

# 口頭用語注意事項

- Accuracy 統一說「約 0.xx」或「約 xx%」，避免在口頭報告塞過多小數。
- 不說「證明 alpha PLV 最好」，改說「在目前 5 位受試者的 exploratory analysis 中，alpha-only 的結果最高且較穩定」。
- 不把 SFS network 稱作 brain connectivity map；可說「SFS selected PLV-pair network schematic」。
- iPLV 結果可說明 zero-lag dependence，但不要直接說 PLV 是 artifact。
- Type2 behavioral ceiling 可作背景，不直接宣稱 causal explanation。