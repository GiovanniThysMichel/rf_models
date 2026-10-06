# 5-fold cross-validation at SNR 0 dB

Held-out accuracy per fold, and mean ± SD over the folds. Every sample is RMS-normalised, and for each input both models see the same values on the same stratified folds.

| Model | Input | Parameters | Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 | Mean ± SD |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Linear SVM, C=0.01 | A: raw I/Q | 2,827 | 17.4% | 18.3% | 18.0% | 18.4% | 17.4% | 17.9% ± 0.4% |
| 1D CNN | A: raw I/Q | 219 | 66.6% | 66.5% | 64.3% | 64.9% | 66.2% | 65.7% ± 0.9% |
| Linear SVM, C=0.01 | B: amplitude + instantaneous frequency [A, cos Δφ, sin Δφ] | 4,202 | 39.3% | 40.2% | 38.1% | 38.8% | 40.1% | 39.3% ± 0.8% |
| 1D CNN | B: amplitude + instantaneous frequency [A, cos Δφ, sin Δφ] | 275 | 77.7% | 76.5% | 78.0% | 74.0% | 75.9% | 76.4% ± 1.4% |
| Linear SVM, C=0.01 | C: power-law FFT log\|FFT(x^k)\|, k = 1, 2, 4, 8 | 5,643 | 81.6% | 80.0% | 82.9% | 80.3% | 82.1% | 81.4% ± 1.1% |
| 1D CNN | C: power-law FFT log\|FFT(x^k)\|, k = 1, 2, 4, 8 | 331 | 78.7% | 77.4% | 79.2% | 78.3% | 78.5% | 78.4% ± 0.6% |
