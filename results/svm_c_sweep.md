# Linear SVM: choice of C, 5-fold cross-validation at SNR 0 dB

Same inputs and outer folds as the main results, which use C = 0.01 (model.SVM_C).

## C sweep (sensitivity, not tuning)

Mean held-out accuracy and SD for each C. Every C is scored on the same folds, so this shows how much the accuracy depends on C. The best row is not an unbiased estimate of a tuned SVM, because C would be chosen on the folds it is scored on; see the nested CV below for that.

### A: raw I/Q

| C | Mean 5-fold accuracy | SD |
|---:|---:|---:|
| 0.01 | 17.9% | 0.4% |
| 0.1 | 17.9% | 0.4% |
| 1 | 17.9% | 0.5% |
| 10 | 17.9% | 0.5% |
| 100 | 17.9% | 0.5% |

### B: amplitude + instantaneous frequency [A, cos Δφ, sin Δφ]

| C | Mean 5-fold accuracy | SD |
|---:|---:|---:|
| 0.01 | 39.3% | 0.8% |
| 0.1 | 38.8% | 1.0% |
| 1 | 38.6% | 0.9% |
| 10 | 38.6% | 0.9% |
| 100 | 38.6% | 0.9% |

### C: power-law FFT log|FFT(x^k)|, k = 1, 2, 4, 8

| C | Mean 5-fold accuracy | SD |
|---:|---:|---:|
| 0.01 | 81.4% | 1.1% |
| 0.1 | 80.3% | 0.9% |
| 1 | 78.6% | 0.6% |
| 10 | 77.7% | 0.6% |
| 100 | 77.6% | 0.5% |

## Nested cross-validation (tuned C, unbiased)

Inside each outer training fold, an inner stratified 5-fold CV picks C from {0.01, 0.1, 1, 10, 100} (the smaller C on a tie). The chosen C is refit on the whole outer training fold and scored once on the outer test fold, which plays no part in choosing it. Each cell shows the held-out accuracy and the C chosen for that fold.

| Input | Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 | Mean ± SD |
|---|---:|---:|---:|---:|---:|---:|
| A: raw I/Q | 17.4% (C=0.01) | 18.3% (C=0.01) | 17.9% (C=0.1) | 18.6% (C=100) | 17.4% (C=0.1) | 17.9% ± 0.4% |
| B: amplitude + instantaneous frequency [A, cos Δφ, sin Δφ] | 39.3% (C=0.01) | 40.2% (C=0.01) | 38.1% (C=0.01) | 38.8% (C=0.01) | 40.1% (C=0.01) | 39.3% ± 0.8% |
| C: power-law FFT log|FFT(x^k)|, k = 1, 2, 4, 8 | 81.6% (C=0.01) | 80.0% (C=0.01) | 82.9% (C=0.01) | 80.3% (C=0.01) | 82.1% (C=0.01) | 81.4% ± 1.1% |
