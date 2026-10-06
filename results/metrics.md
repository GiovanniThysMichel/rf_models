# Precision, recall and F1, 5-fold cross-validation at SNR 0 dB

Macro averages are the unweighted mean over the 11 classes, computed on each held-out fold and reported as mean ± SD over the folds. Every held-out fold has the same number of samples per class, so macro recall equals accuracy. Per-class values use the pooled out-of-fold predictions. A class that is never predicted gets precision 0.

## Macro averages

| Model | Input | Precision | Recall | F1 |
|---|---|---:|---:|---:|
| Linear SVM, C=0.01 | A: raw I/Q | 16.5% ± 0.2% | 17.9% ± 0.4% | 16.4% ± 0.4% |
| 1D CNN | A: raw I/Q | 64.4% ± 0.8% | 65.7% ± 0.9% | 64.5% ± 0.9% |
| Linear SVM, C=0.01 | B: amplitude + instantaneous frequency [A, cos Δφ, sin Δφ] | 37.0% ± 0.8% | 39.3% ± 0.8% | 37.3% ± 0.7% |
| 1D CNN | B: amplitude + instantaneous frequency [A, cos Δφ, sin Δφ] | 75.7% ± 1.4% | 76.4% ± 1.4% | 75.9% ± 1.4% |
| Linear SVM, C=0.01 | C: power-law FFT log\|FFT(x^k)\|, k = 1, 2, 4, 8 | 81.0% ± 1.1% | 81.4% ± 1.1% | 81.0% ± 1.1% |
| 1D CNN | C: power-law FFT log\|FFT(x^k)\|, k = 1, 2, 4, 8 | 78.1% ± 0.7% | 78.4% ± 0.6% | 77.9% ± 0.6% |

## Per class

### Linear SVM, C=0.01 on A: raw I/Q

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| 8PSK | 9.3% | 8.2% | 8.7% | 1,000 |
| AM-DSB | 35.4% | 65.5% | 46.0% | 1,000 |
| AM-SSB | 16.8% | 26.1% | 20.5% | 1,000 |
| BPSK | 10.9% | 12.3% | 11.6% | 1,000 |
| CPFSK | 9.1% | 7.9% | 8.5% | 1,000 |
| GFSK | 11.7% | 10.0% | 10.8% | 1,000 |
| PAM4 | 15.1% | 10.5% | 12.4% | 1,000 |
| QAM16 | 19.9% | 13.3% | 15.9% | 1,000 |
| QAM64 | 21.6% | 10.9% | 14.5% | 1,000 |
| QPSK | 10.9% | 10.0% | 10.4% | 1,000 |
| WBFM | 20.6% | 22.3% | 21.4% | 1,000 |

### 1D CNN on A: raw I/Q

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| 8PSK | 37.2% | 22.6% | 28.1% | 1,000 |
| AM-DSB | 50.0% | 57.0% | 53.3% | 1,000 |
| AM-SSB | 93.4% | 100.0% | 96.6% | 1,000 |
| BPSK | 55.9% | 65.3% | 60.2% | 1,000 |
| CPFSK | 59.2% | 73.8% | 65.7% | 1,000 |
| GFSK | 68.8% | 78.4% | 73.3% | 1,000 |
| PAM4 | 85.9% | 73.8% | 79.4% | 1,000 |
| QAM16 | 77.9% | 84.4% | 81.0% | 1,000 |
| QAM64 | 93.9% | 97.8% | 95.8% | 1,000 |
| QPSK | 40.6% | 35.8% | 38.1% | 1,000 |
| WBFM | 44.1% | 33.7% | 38.2% | 1,000 |

### Linear SVM, C=0.01 on B: amplitude + instantaneous frequency [A, cos Δφ, sin Δφ]

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| 8PSK | 26.9% | 16.2% | 20.2% | 1,000 |
| AM-DSB | 24.2% | 30.8% | 27.1% | 1,000 |
| AM-SSB | 82.9% | 99.6% | 90.5% | 1,000 |
| BPSK | 40.0% | 26.7% | 32.0% | 1,000 |
| CPFSK | 16.5% | 13.8% | 15.0% | 1,000 |
| GFSK | 18.4% | 17.1% | 17.7% | 1,000 |
| PAM4 | 68.0% | 90.8% | 77.8% | 1,000 |
| QAM16 | 40.4% | 32.3% | 35.9% | 1,000 |
| QAM64 | 45.1% | 61.5% | 52.0% | 1,000 |
| QPSK | 22.5% | 13.1% | 16.6% | 1,000 |
| WBFM | 21.7% | 30.5% | 25.4% | 1,000 |

### 1D CNN on B: amplitude + instantaneous frequency [A, cos Δφ, sin Δφ]

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| 8PSK | 50.0% | 44.7% | 47.2% | 1,000 |
| AM-DSB | 53.2% | 63.3% | 57.8% | 1,000 |
| AM-SSB | 93.8% | 99.0% | 96.3% | 1,000 |
| BPSK | 82.6% | 88.9% | 85.6% | 1,000 |
| CPFSK | 87.6% | 91.3% | 89.4% | 1,000 |
| GFSK | 75.3% | 81.2% | 78.2% | 1,000 |
| PAM4 | 97.7% | 97.9% | 97.8% | 1,000 |
| QAM16 | 97.1% | 93.3% | 95.2% | 1,000 |
| QAM64 | 95.2% | 96.7% | 95.9% | 1,000 |
| QPSK | 51.5% | 47.3% | 49.3% | 1,000 |
| WBFM | 48.0% | 36.8% | 41.7% | 1,000 |

### Linear SVM, C=0.01 on C: power-law FFT log|FFT(x^k)|, k = 1, 2, 4, 8

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| 8PSK | 72.9% | 64.2% | 68.3% | 1,000 |
| AM-DSB | 60.4% | 65.2% | 62.7% | 1,000 |
| AM-SSB | 93.5% | 98.5% | 96.0% | 1,000 |
| BPSK | 94.0% | 96.0% | 95.0% | 1,000 |
| CPFSK | 94.9% | 97.5% | 96.2% | 1,000 |
| GFSK | 83.3% | 89.4% | 86.3% | 1,000 |
| PAM4 | 93.9% | 96.9% | 95.4% | 1,000 |
| QAM16 | 86.8% | 76.8% | 81.5% | 1,000 |
| QAM64 | 87.7% | 93.0% | 90.2% | 1,000 |
| QPSK | 70.4% | 72.7% | 71.6% | 1,000 |
| WBFM | 52.6% | 45.0% | 48.5% | 1,000 |

### 1D CNN on C: power-law FFT log|FFT(x^k)|, k = 1, 2, 4, 8

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| 8PSK | 51.7% | 48.8% | 50.2% | 1,000 |
| AM-DSB | 62.1% | 80.1% | 70.0% | 1,000 |
| AM-SSB | 92.8% | 99.5% | 96.0% | 1,000 |
| BPSK | 91.4% | 93.5% | 92.4% | 1,000 |
| CPFSK | 72.8% | 76.2% | 74.5% | 1,000 |
| GFSK | 82.6% | 87.5% | 85.0% | 1,000 |
| PAM4 | 98.0% | 97.6% | 97.8% | 1,000 |
| QAM16 | 94.6% | 94.1% | 94.3% | 1,000 |
| QAM64 | 96.3% | 93.9% | 95.1% | 1,000 |
| QPSK | 52.9% | 48.8% | 50.8% | 1,000 |
| WBFM | 62.9% | 42.7% | 50.9% | 1,000 |
