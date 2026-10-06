# RadioML Classification Assessment

## 1. Problem
Classify 11 modulation types at SNR = 0 dB.

## 2. Dataset
- RadioML 2016.10a, 0 dB SNR slice
- 11 classes × 1,000 examples = 11,000 examples
- Each example: 128 complex samples, stored as 2 × 128 (I, Q)

## 3. Data preparation
- **Load** the pickle, a dict keyed by (modulation, SNR); each value has shape (1000, 2, 128).
- **Filter** to 0 dB SNR, dropping the other 19 levels.
- **Normalization** RadioML 2016.10a is already normalized during dataset generation by dividing each complex vector by the sum of its sample magnitudes. As an additional preprocessing step, each loaded example is rescaled by the RMS computed jointly over its 256 I/Q scalar values. Thus, the combined I/Q array has unit mean-square while preserving the relative I/Q structure and phase. This removes overall received-signal scale, such as that introduced by gain, while preserving the within-sample characteristics that distinguish modulation types.
- **Build the input**, one per experiment, from x[n] = I[n] + jQ[n]:
  - **A: raw I/Q** (2 × 128)
  - **B: amplitude + instantaneous frequency** (3 × 127): [A, cos Δφ, sin Δφ], with A[n] = |x[n]| and Δφ[n] = angle(x[n+1]·x*[n]). Encoding Δφ as cos and sin removes the jump between +π and −π.
  - **C: power-law FFT** (4 × 128): log|FFT(xᵏ)| for k = 1, 2, 4, 8. Raising M-PSK to the M-th power leaves a spectral line (BPSK in x², QPSK in x⁴, 8PSK in x⁸).
- **Standardise** each channel using the mean and std of the fold's training examples only, so nothing about the held-out examples leaks into training.



## 4. CNN diagram 
- Batch 64, lr 3e-3, 150 epochs
```
input   [N, C, L]    C = 2 / 3 / 4, L = 128 / 127 / 128 for A / B / C
  ↓  Conv1d(C → 8, kernel 7, stride 2)
        [N, 8, 61]
  ↓  ReLU
  ↓  global average pooling over time
        [N, 8]
  ↓  Linear(8 → 11)
output  [N, 11]      one logit per class
```
- Model note: For additional experiments B and C, only in_channels changes from 2 to 3 or 4, respectively; the CNN topology is otherwise unchanged.
- Preprocessing note: Each observation is independently RMS-normalized to reduce sensitivity to absolute received-signal amplitude and emphasize modulation structure.


## 5. Cross-validation methodology
- Stratified 5-fold CV, shuffled with seed 42: each fold trains on 8,800 examples (800 per class) and holds out 2,200 (200 per class).
- The CNN and the SVM use identical folds and identical inputs.
- SVM C sweep: nested CV, with an inner CV inside each training fold to pick C; standardisation is refit on each inner training split.



## 6. Reproduction
1. Set up the environment:
   ```bash
   python3 -m venv venv && source venv/bin/activate
   pip install -r requirements.txt
   ```
2. Download RadioML 2016.10a from [zenodo.org/records/18397070](https://zenodo.org/records/18397070) and extract it so the pickle is at `data/18397070/RML2016.10a/RML2016.10a_dict_optimized.pkl`.
3. Run from the project root:
   ```bash
   python -m src.train            # CNN and SVM (C = 0.01) on inputs A, B and C
   python -m src.train --c-sweep  # same, plus the SVM sweep over C ∈ {0.01, 0.1, 1, 10, 100} with nested CV
   ```

Results are written to `results/` (`cv_summary.md`, `metrics.md`, `svm_c_sweep.md`, `figures/`).

## 7. Results
Precision, recall and F1 (macro and per class) are in [metrics.md](results/metrics.md); confusion matrices and CNN training curves are in [figures/](results/figures/).

### Required Results (RAW IQ + best-C Linear + mean accuracy)
| Model | Input | Parameters | Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 | Mean ± SD |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Linear SVM, C=0.01 | A: raw I/Q | 2,827 | 17.4% | 18.3% | 18.0% | 18.4% | 17.4% | 17.9% ± 0.4% |
| 1D CNN | A: raw I/Q | 219 | 66.6% | 66.5% | 64.3% | 64.9% | 66.2% | **65.7% ± 0.9%** |

- Summary: 1D CNN on RAW IQ input performs better than SVM across folds.
- Best C: on raw I/Q every C in {0.01, 0.1, 1, 10, 100} gives 17.9% ([svm_c_sweep.md](results/svm_c_sweep.md)), and nested CV, which picks C inside each training fold, also gives 17.9% ± 0.4%. C = 0.01 is used for every input because it is best on B and C.

### Results: experiments B and C
| Model | Input | Parameters | Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 | Mean ± SD |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Linear SVM, C=0.01 | B: amplitude + instantaneous frequency | 4,202 | 39.3% | 40.2% | 38.1% | 38.8% | 40.1% | 39.3% ± 0.8% |
| 1D CNN | B: amplitude + instantaneous frequency | 275 | 77.7% | 76.5% | 78.0% | 74.0% | 75.9% | 76.4% ± 1.4% |
| Linear SVM, C=0.01 | C: power-law FFT | 5,643 | 81.6% | 80.0% | 82.9% | 80.3% | 82.1% | **81.4% ± 1.1%** |
| 1D CNN | C: power-law FFT | 331 | 78.7% | 77.4% | 79.2% | 78.3% | 78.5% | 78.4% ± 0.6% |

- **B, amplitude + instantaneous frequency:** both models improve on raw I/Q (CNN 65.7% → 76.4%, SVM 17.9% → 39.3%), and the CNN beats the SVM on every fold.
- **C, power-law FFT:** both models improve again (CNN 76.4% → 78.4%, SVM 39.3% → 81.4%), and the SVM now beats the CNN on every fold. The SVM on C is the best result overall.


## 8. Recommendations, including tradeoffs between models
- **Raw I/Q.** On the required raw I/Q input, the CNN reaches 65.7% mean five-fold accuracy versus 17.9% for the linear SVM: even a one-layer nonlinear feature extractor learns modulation structure that is not linearly separable in raw I/Q.
- **Engineered features.** With power-law FFT features the SVM jumps from 17.9% to 81.4% and beats the CNN (78.4%) on every fold. This is the core tradeoff: learned nonlinear feature extraction versus domain-informed feature engineering.
- **Training vs size.** The SVM is simpler and faster to train: its objective is convex and C is the only hyperparameter tuned here. The CNN requires a learning rate, batch size, epoch count, and random initialization, but it is far smaller (219–331 parameters vs. 2,827–5,643).
- **Next step.** Beyond the specified architecture, evaluate a 2- or 3-layer CNN. On raw I/Q, training and held-out loss plateau together around 0.9–1.0, so the current network underfits rather than overfits, and more capacity is likely to help.
- **Takeaway.** When raw I/Q must be used directly, I recommend the CNN. When simple, fast training is a priority and RF-specific preprocessing is acceptable, the linear SVM on power-law FFT features is the better choice.
