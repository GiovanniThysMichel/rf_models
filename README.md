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
- **Normalise power:** divide each example by its RMS over I and Q, giving unit average power.
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
   python -m src.train            # CNN and SVM (C = 1) on inputs A, B and C
   python -m src.train --c-sweep  # same, plus the SVM sweep over C ∈ {0.01, 0.1, 1, 10, 100} with nested CV
   ```

Results are written to `results/` (`cv_summary.md`, `svm_c_sweep.md`, `figures/`).

## 7. Results
### Required Results (RAW IQ + best-C Linear + mean accuracy)
### Results Experiment B and C

## 8. Recommendations, including tradeoffs between models
