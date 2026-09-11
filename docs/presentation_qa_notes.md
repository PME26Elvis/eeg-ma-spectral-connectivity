# 報告 Q&A 準備筆記

這份筆記整理目前專題最可能被問到的方法與結果問題。回答以目前 repo 的正式定義為準，避免在現場臨時把 exploratory result 說得太強。

## Q1. 為什麼只用 7 個 frontal channels？Cz 為什麼不用？

這一階段以實驗室原本的 frontal feature pipeline 為 baseline，使用 FP1、FP2、F7、F3、Fz、F4、F8。Cz 有完整保留在 raw EEG，沒有納入目前 168-feature replication pipeline。這樣可以先維持與既有流程的可比性，也避免在 extension 階段同時更動 channel set 與 feature representation。

如果之後增加資料量，可以把 Cz 或更多 channels 視為另一個預先定義的 extension。

## Q2. 為什麼只分析答對的 Mental Arithmetic trials？

目前目標是比較穩定的 Rest state 與正確完成 mental-arithmetic calculation 的 EEG。錯誤 trial 可能混入理解錯誤、計算策略失敗或注意力中斷等額外因素，因此保留完整行為紀錄，但 primary EEG classification 只使用 `analysis_eligible=true` 且答對的 5-s calculation epoch。

## Q3. 為什麼 Rest1 對 Type1、Rest2 對 Type2，不把全部 Rest 合在一起？

這是目前 protocol 的 primary comparison。Rest1 與 Type1、Rest2 與 Type2 在實驗時序上相鄰，可以降低跨較長 session 時段的 drift、疲勞與狀態變化影響。兩個 comparison 分開分析，也能保留不同 task difficulty 的資訊。

## Q4. 為什麼要同時做 intra-subject 和 inter-subject？

兩者回答不同問題。

- intra-subject：同一受試者內，Rest 與 Mental Arithmetic 是否可分。
- inter-subject：從其他受試者學到的模型，是否能泛化到一個完全沒進 training 的人。

Baseline 的主要 observation 就來自這個差異：intra 約 0.7–0.8 以上，inter 接近 0.5。

## Q5. 為什麼 inter-subject 用 LOPO？

受試者數只有 5 位，LOPO 可以讓每一位受試者都輪流成為完整 held-out test subject。這比把不同人的 epochs 隨機混在一般 k-fold 裡更適合評估真正的 cross-subject generalization，因為後者會讓同一人的資料同時出現在 train 與 test。

## Q6. SFS 有沒有 data leakage？

沒有直接使用 outer test data 做 SFS。每個 outer fold 裡，SFS 只在 outer training data 上執行；inter-subject setting 中，inner validation 也按 subject groups 分開。Normalization 與 SVM/KFDA tuning 同樣限制在 outer training data。

## Q7. 為什麼三個 classifier 共用同一組 SFS features？

目前實驗室來源沒有提供每一種 classifier 各自的 feature-selection objective。Repo 明確採 training-only LDA inner-CV 作為統一 SFS criterion，再讓 LDA、RBF-SVM、KFDA 使用相同 selected subset。這樣可以降低「每個 classifier 都重新挑 feature」帶來的額外自由度，並讓 classifier comparison 更直接。

## Q8. Coherence 的 1 秒 segment 是學長海報規定的嗎？

不是。實驗室教材定義了 averaged spectral quantities 與 coherence 公式，但沒有指定 segment duration。Repo 將 1.0 s、non-overlapping、rectangular FFT 明確記錄為 implementation choice。5-s epoch 因此有 5 個 segments。

報告中要把這點列在 implementation details，不要寫成學長原程式已確定採 1 秒。

## Q9. 為什麼 E1 asymmetry 沒有效果還要放？

E1 是從 baseline observation 推導出的合理 hypothesis：不同受試者的 absolute BP scale 差異可能很大，左右相對值可能較穩定。結果沒有形成穩定跨人改善，所以它提供了一個有用的負結果，也說明後續為什麼轉向 phase representation。

## Q10. 為什麼會想到 PLV？

Baseline 的 BP 著重能量，COH 同時包含 amplitude 與 phase relationship。跨受試者的 amplitude scale 變異可能很大，因此測試只看 phase-difference stability 的 PLV，觀察它是否能提供較穩定的 cross-subject representation。

## Q11. PLV 為什麼看起來比 COH 好？

目前可以說 PLV 在這批資料的 Rest1 vs Type1 LOPO 中提供較高分類結果。不能把這個結果延伸成一般性的「PLV 優於 COH」。樣本數只有 5，而且 extension 都在同一批資料上探索。

PLV-only 約 0.67–0.71，Poster baseline 約 0.52–0.53；這是目前資料上的描述性 observation。

## Q12. 為什麼 alpha-only 會比 all-band PLV 還好？

SFS 在前一階段已經多次選到 alpha-band PLV features。Six-band validation 把六個頻帶全部獨立跑，alpha-only 的三模型平均約 0.741，是目前最高的 single-band result，且 held-out-subject standard deviation 較 all-band PLV 低。

可能原因包括 alpha band 的 task-related phase pattern在這批資料中較穩定，以及移除較弱或 subject-specific bands 後候選空間更簡潔。n=5 下不能把這些解釋當成已驗證機制。

## Q13. Gamma 也很高，為什麼主要講 alpha？

Gamma 是第二高的 single-band，三模型平均約 0.687。Alpha 約 0.741，而且三個 classifier 的結果更接近、跨 subject standard deviation 也較低；此外前一階段的 SFS recurrence 已經獨立指向多個 alpha PLV feature。這些結果一起讓 alpha 成為主要 observation。Gamma 可以在 appendix 或討論中保留。

## Q14. iPLV 掉到 chance 附近代表 PLV 是假的嗎？

iPLV 結果表示 PLV 的分類增益高度依賴 zero- 或 near-zero-lag component。這個 component 可能來自 common reference、volume conduction、shared source、frontal artifact，也可能包含真正接近零相位差的同步神經活動。

目前資料不能把來源分離，因此不能直接說 PLV 是 artifact。比較合適的說法是：PLV 對 classification 有用，但生理 connectivity 的解讀受到 zero-lag dependence 限制。

## Q15. 為什麼不用 wPLI、PLI、EA、ICA，再多試幾個？

目前只有 5 位受試者，已經完成 baseline、feature-family decomposition、E1、E2、PLV decomposition、band-wise validation 與 iPLV robustness。繼續在同一批小資料上增加方法會擴大 researcher degrees of freedom，而且難以區分真正 generalization 與 dataset-specific tuning。

目前設定了 stop rule。新的方法比較適合留到增加受試者或取得 independent cohort 後再驗證。

## Q16. Alpha-only 0.74 可以說是最終模型 accuracy 嗎？

可以說它是目前 exploratory LOPO matrix 中的最高、較穩定 representation；不把它當作 independent confirmatory estimate。Six-band validation 仍然使用同一批 5 subjects，而且前面的探索已經提示 alpha，因此它屬於有邏輯依據的 follow-up validation，但不是外部獨立驗證。

## Q17. 為什麼 Type2 的結果比較弱？

Behavioral data 顯示 Type2 幾乎 100% 正確，而且 reaction time 普遍比 Type1 快，代表任務較容易。這可以作為 Type2 EEG state separability 較弱的背景因素。

目前不能從這批資料直接證明 task difficulty 是分類差異的因果原因。

## Q18. 有做統計顯著性檢定嗎？

目前主要呈現 descriptive LOPO accuracy、held-out-subject variability、paired per-subject delta 與 SFS recurrence。Outer LOPO 只有 5 個 subjects，因此正式推論統計的 power 很有限，而且已經探索多種 feature sets。

報告中不主張統計顯著；以 exploratory trend、cross-classifier consistency、per-subject consistency 與 method robustness 作為主要證據。

## Q19. 60 Hz line noise 會不會有影響？

先前 raw diagnostic 有觀察到 60 Hz mains component。Formal pipeline 依既定計畫使用 2–50 Hz band-pass，正式 feature bands 最高到 45 Hz，因此 60 Hz 本身位於分析頻帶之外。Formal pipeline 沒有在後期為了 accuracy 額外加入 notch，避免改動 baseline 定義。

如果未來重新設計 protocol，可以預先把 notch/filter sensitivity analysis 寫入方法，而不是看到結果後再加入。

## Q20. Delta 是 1–4 Hz，但 preprocessing 從 2 Hz 開始，怎麼解釋？

這是目前計畫定義中的一個限制：band label 使用 1–4 Hz，continuous preprocessing high-pass 是 2 Hz，因此實際保留下來的 delta information 主要從 2 Hz 以上開始。Repo 保留原本定義並在方法文件中明示，沒有在分析中途改 band boundaries。

## Q21. 下一步最值得做什麼？

最有資訊價值的下一步是增加新的受試者，固定目前方法後做 confirmatory validation。若資料量足夠，再評估 domain alignment、更多 electrode channels、reference strategy 或更完整的 connectivity analysis。

這可以直接檢查目前 alpha-PLV observation 是否能在新的 subjects 上重現。