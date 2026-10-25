# POC results on synthetic data

Reference test set (severity 1): 600 normal and 360 faulty curves across 2 materials
(30 per fault type per material). Models trained per material on
500 historical curves with 3% unlabelled faults mixed in.

**These numbers come from synthetic curves written by the same person who wrote the detector.
They show the pipeline works end to end. They are not evidence of performance on real tests.**

## Flagging quality (pooled, severity 1)

| layer | tp | fp | fn | tn | precision | recall | f1 | fpr |
|---|---|---|---|---|---|---|---|---|
| Rules only | 354 | 3 | 6 | 597 | 0.992 | 0.983 | 0.987 | 0.005 |
| ML only | 269 | 1 | 91 | 599 | 0.996 | 0.747 | 0.854 | 0.002 |
| Rules + ML | 358 | 4 | 2 | 596 | 0.989 | 0.994 | 0.992 | 0.007 |

ML score on its own (threshold-free): ROC AUC 0.987, average precision 0.983.

## Recall by fault type (severity 1)

| fault | Rules only | ML only | Rules + ML |
|---|---|---|---|
| grip_slippage | 1.000 | 0.933 | 1.000 |
| misalignment | 0.983 | 1.000 | 1.000 |
| premature_fracture | 0.917 | 0.967 | 0.967 |
| extensometer_slip | 1.000 | 0.517 | 1.000 |
| noise_spikes | 1.000 | 0.350 | 1.000 |
| preload_offset | 1.000 | 0.717 | 1.000 |

## Cause and segment attribution (flagged faults only, severity 1)

Overall cause correct: 0.980; segment overlaps the true one: 0.989
(ML-only flags carry no specific cause or segment by design, so they count as incorrect here).

| fault | flagged | cause correct | segment hit |
|---|---|---|---|
| grip_slippage | 60 | 0.950 | 1.000 |
| misalignment | 60 | 0.983 | 0.983 |
| premature_fracture | 58 | 0.948 | 0.948 |
| extensometer_slip | 60 | 1.000 | 1.000 |
| noise_spikes | 60 | 1.000 | 1.000 |
| preload_offset | 60 | 1.000 | 1.000 |

## What happens when faults get subtler

Severity scales fault magnitude (1 = clear, 0.25 = subtle). Rules + ML.

| severity | precision | recall | fpr | ml_auc |
|---|---|---|---|---|
| 1.000 | 0.989 | 0.994 | 0.007 | 0.987 |
| 0.500 | 0.991 | 0.906 | 0.005 | 0.959 |
| 0.250 | 0.996 | 0.728 | 0.002 | 0.901 |

Recall by fault type:

| fault | severity 1 | severity 0.5 | severity 0.25 |
|---|---|---|---|
| grip_slippage | 1.000 | 0.867 | 0.833 |
| misalignment | 1.000 | 0.667 | 0.117 |
| premature_fracture | 0.967 | 0.967 | 0.900 |
| extensometer_slip | 1.000 | 1.000 | 1.000 |
| noise_spikes | 1.000 | 0.933 | 0.783 |
| preload_offset | 1.000 | 1.000 | 0.733 |
