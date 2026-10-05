import argparse
import csv
import sys
import warnings
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from pathlib import Path
from joblib import Parallel, delayed
from sklearn.exceptions import ConvergenceWarning

if __name__ == '__main__' and not __package__:
    # Run as a plain script (python src/train.py): make the relative imports below resolve
    # against the src package, as they do under python -m src.train.
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    __package__ = 'src'

from .dataset import (MODULATIONS, IQ_CHANNELS, AP_CHANNELS, POWER_CHANNELS, RadioMLDataset,
                      get_kfold_dataloaders, get_kfold_indices, standardize_channels)
from .evaluate import (evaluate, predict, classification_metrics, macro_metrics_per_fold,
                       plot_confusion_matrix, plot_cv_curves)
from .model import ModulationCNN, make_linear_svm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / 'data/18397070/RML2016.10a/RML2016.10a_dict_optimized.pkl'
C_VALUES = (0.01, 0.1, 1, 10, 100)

# CNN training: Adam with cosine learning-rate decay over EPOCHS
EPOCHS = 150
LR = 3e-3
BATCH_SIZE = 64

# Experiments A, B, C: the input sets both models are compared on.
# key (used in file names and --inputs) -> (label, channels)
INPUTS = {
    'A_iq':        ('A: raw I/Q', IQ_CHANNELS),
    'B_amp_phase': ('B: amplitude + instantaneous frequency [A, cos Δφ, sin Δφ]', AP_CHANNELS),
    'C_power':     ('C: power-law FFT log|FFT(x^k)|, k = 1, 2, 4, 8', POWER_CHANNELS),
}



def train(model, train_loader, val_loader, epochs=EPOCHS, lr=LR, device='cuda',
          checkpoint_path=None, criterion=None):
    """
    Train for a fixed number of epochs and return the per-epoch history.

    val_loader is only monitored, never used to pick an epoch: under cross-validation
    it is the held-out fold, and keeping the best epoch on it would bias the fold
    score upward. The final weights are saved to checkpoint_path, if given.
    """
    if criterion is None:
        criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    model.to(device)
    history = {'train_loss': [], 'val_loss': [], 'val_acc': []}


    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(x)
        train_loss /= len(train_loader.dataset)

        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        lr_scheduler.step()

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        print(f"Epoch {epoch:02d}/{epochs} | "
              f"train_loss: {train_loss:.4f} | "
              f"val_loss: {val_loss:.4f} | "
              f"val_acc: {val_acc:.3f}")

    if checkpoint_path is not None:
        checkpoint_path = Path(checkpoint_path)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), checkpoint_path)

    return history


def cross_validate(model_fn, data_path, n_splits=5, snr=0, channels=IQ_CHANNELS, epochs=EPOCHS,
                   lr=LR, batch_size=BATCH_SIZE, device='cuda', checkpoint_dir='checkpoints', seed=42):
    """
    Stratified k-fold cross-validation on the samples at a single SNR level.

    channels picks the model input (see dataset.make_channels): raw I/Q by default, or
    e.g. AP_CHANNELS or POWER_CHANNELS.

    model_fn() must return a freshly initialised model with in_channels=len(channels);
    it is called once per fold so no weights carry over between folds.

    Returns a dict with:
        histories:      per-fold training histories (see train())
        fold_acc:       held-out accuracy of each fold's final model
        labels, preds:  out-of-fold labels and predictions over the whole dataset
        n_params:       number of trainable parameters in the model
    """
    torch.backends.cudnn.deterministic = True  # bit-for-bit repeatable results for the same seed
    torch.backends.cudnn.benchmark = False

    dataset = _load_snr_dataset(data_path, snr, channels)
    folds = get_kfold_dataloaders(dataset, n_splits=n_splits, batch_size=batch_size, seed=seed)
    results = {'histories': [], 'fold_acc': [], 'labels': [], 'preds': []}

    for fold, (train_loader, val_loader) in enumerate(folds, start=1):
        print(f"\n=== Fold {fold}/{n_splits} | "
              f"train: {len(train_loader.dataset):,} | val: {len(val_loader.dataset):,} ===")
        torch.manual_seed(seed + fold)
        model = model_fn()
        history = train(model, train_loader, val_loader, epochs=epochs, lr=lr, device=device,
                        checkpoint_path=Path(checkpoint_dir) / f'fold{fold}.pt')

        labels, preds = predict(model, val_loader, device)
        results['histories'].append(history)
        results['fold_acc'].append(float((labels == preds).mean()))
        results['labels'].append(labels)
        results['preds'].append(preds)

    results['n_params'] = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return _summarize(results, n_splits)


def cross_validate_svm(model_fn, data_path, n_splits=5, snr=0, channels=IQ_CHANNELS, seed=42):
    """
    Stratified k-fold cross-validation of a scikit-learn classifier on exactly the input
    cross_validate() gives the CNN for the same arguments — same SNR slice, channels,
    normalisation, per-fold standardisation and folds — flattened to (N, len(channels) * L).

    model_fn() must return a fresh, unfitted estimator (e.g. make_linear_svm).

    Returns a dict with fold_acc, labels and preds, as in cross_validate(), and n_params:
    the size of the fitted model's weights and biases.
    """
    dataset = _load_snr_dataset(data_path, snr, channels)
    y = dataset.labels.numpy()

    results = {'fold_acc': [], 'labels': [], 'preds': []}

    for fold, (train_idx, val_idx) in enumerate(get_kfold_indices(y, n_splits, seed), start=1):
        X = standardize_channels(dataset.data, train_idx).flatten(1).numpy()
        model = model_fn()
        model.fit(X[train_idx], y[train_idx])
        preds = model.predict(X[val_idx])

        results['fold_acc'].append(float((preds == y[val_idx]).mean()))
        results['labels'].append(y[val_idx])
        results['preds'].append(preds)
        print(f"Fold {fold}/{n_splits} | train: {len(train_idx):,} | val: {len(val_idx):,} | "
              f"val_acc: {results['fold_acc'][-1]:.3f}")

    results['n_params'] = int(model.coef_.size + model.intercept_.size)
    return _summarize(results, n_splits)


def cross_validate_svm_nested(data_path, c_values=C_VALUES, n_splits=5, n_inner_splits=5, snr=0,
                              channels=IQ_CHANNELS, seed=42, n_jobs=-1):
    """
    Nested cross-validation of the linear SVM with C chosen from c_values: an unbiased estimate
    of the accuracy of the whole "choose C by cross-validation, then train" procedure, which the
    best row of a C sweep on the same folds is not.

    The outer folds are the same as in cross_validate() and cross_validate_svm() for the same
    seed. Inside each outer training fold, an inner stratified n_inner_splits-fold CV (channels
    standardised on each inner training split) scores every C. The C with the best mean inner
    accuracy (the smaller C on a tie) is refit on the whole outer training fold and scored once
    on the outer test fold, which plays no part in choosing C. The inner fits run in parallel
    on n_jobs CPU cores.

    Returns a dict with fold_acc and best_C per outer fold, labels and preds (as in
    cross_validate()), and unconverged_fits: how many fits stopped at the solver's iteration limit.
    """
    c_values = sorted(c_values)
    dataset = _load_snr_dataset(data_path, snr, channels)
    y = dataset.labels.numpy()
    results = {'fold_acc': [], 'best_C': [], 'labels': [], 'preds': [], 'unconverged_fits': 0}

    for fold, (train_idx, test_idx) in enumerate(get_kfold_indices(y, n_splits, seed), start=1):
        jobs = []
        for inner_train, inner_val in get_kfold_indices(y[train_idx], n_inner_splits, seed):
            inner_train, inner_val = train_idx[inner_train], train_idx[inner_val]
            X = standardize_channels(dataset.data, inner_train).flatten(1).numpy()
            jobs += [delayed(_fit_and_score)(C, X, y, inner_train, inner_val) for C in c_values]
        scores = Parallel(n_jobs=n_jobs)(jobs)
        inner_acc = np.array([acc for acc, _, _ in scores]).reshape(n_inner_splits, len(c_values)).mean(axis=0)
        best_C = c_values[int(np.argmax(inner_acc))]  # argmax keeps the first, i.e. smallest, C on a tie

        X = standardize_channels(dataset.data, train_idx).flatten(1).numpy()
        acc, preds, unconverged = _fit_and_score(best_C, X, y, train_idx, test_idx)
        results['unconverged_fits'] += sum(u for _, _, u in scores) + unconverged
        results['fold_acc'].append(acc)
        results['best_C'].append(best_C)
        results['labels'].append(y[test_idx])
        results['preds'].append(preds)
        print(f"Outer fold {fold}/{n_splits} | inner CV accuracy per C: "
              f"{', '.join(f'{C:g}: {a:.3f}' for C, a in zip(c_values, inner_acc))} | "
              f"selected C={best_C:g} | held-out acc: {acc:.3f}")

    return _summarize(results, n_splits)


def _fit_and_score(C, X, y, train_idx, test_idx):
    """Fit make_linear_svm(C) on X[train_idx]; return (accuracy, predictions) on X[test_idx] and
    whether the solver stopped at its iteration limit."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always', ConvergenceWarning)
        model = make_linear_svm(C=C).fit(X[train_idx], y[train_idx])
    preds = model.predict(X[test_idx])
    unconverged = any(issubclass(w.category, ConvergenceWarning) for w in caught)
    return float((preds == y[test_idx]).mean()), preds, unconverged


def run_experiments(data_path=DATA_PATH, out_dir=PROJECT_ROOT / 'results', fig_dir=None,
                    checkpoint_dir=PROJECT_ROOT / 'checkpoints', n_splits=5, snr=0, epochs=EPOCHS,
                    lr=LR, batch_size=BATCH_SIZE, C=1.0, c_sweep=False, c_values=C_VALUES,
                    inputs=None, device='cuda', seed=42):
    """
    The linear SVM and the CNN on each input set in inputs (keys of INPUTS; all of them by
    default): two cross-validated runs per input. For each input, both models see identical
    values on identical folds.

    For every run the confusion matrix of the out-of-fold predictions is saved to
    fig_dir/<cnn|svm>_<input key>_confusion_matrix.png (fig_dir defaults to out_dir/figures),
    and for the CNN also its per-fold training curves (..._training_curves.png; the SVM has
    no epochs). CNN fold checkpoints go to checkpoint_dir/cnn_<input key>/. The results table
    (parameters, accuracy per fold, mean ± SD) is written to out_dir/cv_summary.md and
    out_dir/cv_summary.csv, and returned. Precision, recall and F1 (macro averages over the
    folds, and per class) are written to out_dir/metrics.md and out_dir/metrics.csv.

    With c_sweep=True, also runs run_c_sweep() over c_values: the SVM's C sweep and the
    nested CV that tunes C.
    """
    summary, metrics = [], []
    fig_dir = Path(fig_dir) if fig_dir is not None else Path(out_dir) / 'figures'
    fig_dir.mkdir(parents=True, exist_ok=True)

    for input_name, model_name in [(i, m) for i in (inputs or INPUTS) for m in ('svm', 'cnn')]:
        input_label, channels = INPUTS[input_name]
        model_label = '1D CNN' if model_name == 'cnn' else f'Linear SVM, C={C:g}'
        title = f"{model_label} on {input_label}"
        print(f"\n########## {title} ##########")

        if model_name == 'cnn':
            results = cross_validate(
                lambda: ModulationCNN(num_classes=len(MODULATIONS), in_channels=len(channels)),
                data_path, n_splits=n_splits, snr=snr, channels=channels, epochs=epochs, lr=lr,
                batch_size=batch_size, device=device, checkpoint_dir=Path(checkpoint_dir) / f'cnn_{input_name}',
                seed=seed)
            n_params = results['n_params']
            plt.close(plot_cv_curves(results['histories'], title=title,
                                     save_path=fig_dir / f'cnn_{input_name}_training_curves.png'))
        else:
            results = cross_validate_svm(lambda: make_linear_svm(C=C), data_path, n_splits=n_splits, snr=snr,
                                         channels=channels, seed=seed)
            n_params = results['n_params']

        acc = np.array(results['fold_acc'])
        plt.close(plot_confusion_matrix(
            results['labels'], results['preds'],
            title=f"{title}\n{n_splits}-fold CV at SNR {snr} dB: {acc.mean()*100:.1f}% ± {acc.std()*100:.1f}%",
            save_path=fig_dir / f'{model_name}_{input_name}_confusion_matrix.png'))

        summary.append({'model': model_label, 'input': input_label, 'parameters': n_params,
                        **{f'fold_{k}': round(a, 4) for k, a in enumerate(acc, start=1)},
                        'mean': round(acc.mean(), 4), 'sd': round(acc.std(), 4)})

        fold_macro = macro_metrics_per_fold(results['labels'], results['preds'], results['fold_sizes'])
        metrics.append({'model': model_label, 'input': input_label,
                        'macro': {k: np.array([m[k] for m in fold_macro]) for k in ('precision', 'recall', 'f1')},
                        'per_class': classification_metrics(results['labels'], results['preds'])['per_class']})

    with open(Path(out_dir) / 'cv_summary.csv', 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)

    table = _results_table(summary, n_splits, snr)
    (Path(out_dir) / 'cv_summary.md').write_text(table)
    print('\n' + table)

    _write_metrics(metrics, out_dir, n_splits, snr)

    if c_sweep:
        run_c_sweep(data_path, out_dir=out_dir, c_values=c_values, n_splits=n_splits, snr=snr,
                    inputs=inputs, seed=seed)
    return summary


def run_c_sweep(data_path=DATA_PATH, out_dir=PROJECT_ROOT / 'results', c_values=C_VALUES,
                n_splits=5, n_inner_splits=5, snr=0, inputs=None, seed=42):
    """
    How the linear SVM depends on C, on each input set in inputs (keys of INPUTS; experiments
    A, B and C by default), with the same inputs and outer folds as run_experiments():

      1. C sweep: cross_validate_svm() for every C in c_values. Every C is scored on the same
         held-out folds, so this shows how sensitive the accuracy is to C. Its best row is not
         an unbiased estimate, because C would be chosen on the folds it is scored on.
      2. Nested CV: cross_validate_svm_nested(), which chooses C inside each outer training
         fold and scores it on the untouched outer fold: the unbiased accuracy of the tuned SVM.

    Both tables are written to out_dir/svm_c_sweep.md, with the numbers in
    out_dir/svm_c_sweep.csv and out_dir/svm_nested_cv.csv. Returns (sweep_rows, nested_rows).
    """
    input_sets = [INPUTS[key] for key in (inputs or INPUTS)]
    sweep_rows, nested_rows = [], []

    for input_label, channels in input_sets:
        for C in c_values:
            print(f"\n########## Linear SVM, C={C:g} on {input_label} ##########")
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always', ConvergenceWarning)
                results = cross_validate_svm(lambda: make_linear_svm(C=C), data_path, n_splits=n_splits,
                                             snr=snr, channels=channels, seed=seed)
            acc = np.array(results['fold_acc'])
            sweep_rows.append({'input': input_label, 'C': C,
                               **{f'fold_{k}': round(a, 4) for k, a in enumerate(acc, start=1)},
                               'mean': round(acc.mean(), 4), 'sd': round(acc.std(), 4),
                               'unconverged_folds': sum(issubclass(w.category, ConvergenceWarning) for w in caught)})

    for input_label, channels in input_sets:
        print(f"\n########## Linear SVM, nested CV over C on {input_label} ##########")
        results = cross_validate_svm_nested(data_path, c_values=c_values, n_splits=n_splits,
                                            n_inner_splits=n_inner_splits, snr=snr, channels=channels, seed=seed)
        acc = np.array(results['fold_acc'])
        nested_rows.append({'input': input_label,
                            **{f'fold_{k}': round(a, 4) for k, a in enumerate(acc, start=1)},
                            **{f'C_fold_{k}': C for k, C in enumerate(results['best_C'], start=1)},
                            'mean': round(acc.mean(), 4), 'sd': round(acc.std(), 4),
                            'unconverged_fits': results['unconverged_fits']})

    for name, rows in [('svm_c_sweep.csv', sweep_rows), ('svm_nested_cv.csv', nested_rows)]:
        with open(Path(out_dir) / name, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    table = _c_sweep_table(sweep_rows, nested_rows, sorted(c_values), n_splits, n_inner_splits, snr)
    (Path(out_dir) / 'svm_c_sweep.md').write_text(table)
    print('\n' + table)
    return sweep_rows, nested_rows


def _c_sweep_table(sweep_rows, nested_rows, c_values, n_splits, n_inner_splits, snr):
    c_list = ', '.join(f'{C:g}' for C in c_values)
    lines = [f'# Linear SVM: choice of C, {n_splits}-fold cross-validation at SNR {snr} dB', '',
             'Same inputs and outer folds as the main results, which use C = 1, fixed in advance.', '',
             '## C sweep (sensitivity, not tuning)', '',
             'Mean held-out accuracy and SD for each C. Every C is scored on the same folds, so this shows how '
             'much the accuracy depends on C. The best row is not an unbiased estimate of a tuned SVM, '
             'because C would be chosen on the folds it is scored on; see the nested CV below for that.']
    for input_label in dict.fromkeys(r['input'] for r in sweep_rows):
        lines += ['', f'### {input_label}', '',
                  f'| C | Mean {n_splits}-fold accuracy | SD |', '|---:|---:|---:|']
        for r in (r for r in sweep_rows if r['input'] == input_label):
            lines.append(f"| {r['C']:g} | {r['mean'] * 100:.1f}% | {r['sd'] * 100:.1f}% |")

    lines += ['', '## Nested cross-validation (tuned C, unbiased)', '',
              f'Inside each outer training fold, an inner stratified {n_inner_splits}-fold CV picks C from '
              f'{{{c_list}}} (the smaller C on a tie). The chosen C is refit on the whole outer training fold and '
              f'scored once on the outer test fold, which plays no part in choosing it. Each cell shows the held-out '
              f'accuracy and the C chosen for that fold.', '',
              '| Input | ' + ' | '.join(f'Fold {k}' for k in range(1, n_splits + 1)) + ' | Mean ± SD |',
              '|---|' + '---:|' * (n_splits + 1)]
    for r in nested_rows:
        folds = ' | '.join(f"{r[f'fold_{k}'] * 100:.1f}% (C={r[f'C_fold_{k}']:g})" for k in range(1, n_splits + 1))
        lines.append(f"| {r['input']} | {folds} | {r['mean'] * 100:.1f}% ± {r['sd'] * 100:.1f}% |")

    notes = [f"- C sweep, C = {r['C']:g} on {r['input']}: the solver stopped at its iteration limit before "
             f"converging on {r['unconverged_folds']} of {n_splits} folds." for r in sweep_rows if r['unconverged_folds']]
    notes += [f"- Nested CV on {r['input']}: {r['unconverged_fits']} of "
              f"{n_splits * (n_inner_splits * len(c_values) + 1)} fits stopped at the solver's iteration limit."
              for r in nested_rows if r['unconverged_fits']]
    if notes:
        lines += ['', *notes]
    return '\n'.join(lines) + '\n'


def _results_table(summary, n_splits, snr):
    lines = [f'# {n_splits}-fold cross-validation at SNR {snr} dB', '',
             'Held-out accuracy per fold, and mean ± SD over the folds. Every sample is RMS-normalised, '
             'and for each input both models see the same values on the same stratified folds.', '',
             '| Model | Input | Parameters | ' + ' | '.join(f'Fold {k}' for k in range(1, n_splits + 1)) + ' | Mean ± SD |',
             '|---|---|---:|' + '---:|' * (n_splits + 1)]
    for row in summary:
        folds = ' | '.join(f"{row[f'fold_{k}'] * 100:.1f}%" for k in range(1, n_splits + 1))
        input_label = row['input'].replace('|', '\\|')  # e.g. log|FFT(x^k)| must not split the table cell
        lines.append(f"| {row['model']} | {input_label} | {row['parameters']:,} | {folds} | "
                     f"{row['mean'] * 100:.1f}% ± {row['sd'] * 100:.1f}% |")
    return '\n'.join(lines) + '\n'


def _write_metrics(metrics, out_dir, n_splits, snr):
    """
    metrics.csv: per run, a 'macro' row (mean and SD over the folds) and one row per class
    (pooled out-of-fold predictions). metrics.md: the same as tables.
    """
    fields = ['model', 'input', 'class', 'precision', 'recall', 'f1', 'precision_sd', 'recall_sd', 'f1_sd', 'support']
    rows = []
    for m in metrics:
        rows.append({'model': m['model'], 'input': m['input'], 'class': 'macro',
                     **{k: round(v.mean(), 4) for k, v in m['macro'].items()},
                     **{f'{k}_sd': round(v.std(), 4) for k, v in m['macro'].items()}})
        rows += [{'model': m['model'], 'input': m['input'], 'class': mod,
                  **{k: round(v[k], 4) for k in ('precision', 'recall', 'f1')}, 'support': v['support']}
                 for mod, v in m['per_class'].items()]
    with open(Path(out_dir) / 'metrics.csv', 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    pct = lambda v: f'{v * 100:.1f}%'
    lines = [f'# Precision, recall and F1, {n_splits}-fold cross-validation at SNR {snr} dB', '',
             'Macro averages are the unweighted mean over the 11 classes, computed on each held-out fold '
             'and reported as mean ± SD over the folds. Every held-out fold has the same number of samples '
             'per class, so macro recall equals accuracy. Per-class values use the pooled out-of-fold '
             'predictions. A class that is never predicted gets precision 0.', '',
             '## Macro averages', '',
             '| Model | Input | Precision | Recall | F1 |', '|---|---|---:|---:|---:|']
    for m in metrics:
        cells = ' | '.join(f"{pct(v.mean())} ± {pct(v.std())}" for v in m['macro'].values())
        input_label = m['input'].replace('|', '\\|')
        lines.append(f"| {m['model']} | {input_label} | {cells} |")
    lines += ['', '## Per class']
    for m in metrics:
        lines += ['', f"### {m['model']} on {m['input']}", '',
                  '| Class | Precision | Recall | F1 | Support |', '|---|---:|---:|---:|---:|']
        lines += [f"| {mod} | {pct(v['precision'])} | {pct(v['recall'])} | {pct(v['f1'])} | {v['support']:,} |"
                  for mod, v in m['per_class'].items()]
    table = '\n'.join(lines) + '\n'
    (Path(out_dir) / 'metrics.md').write_text(table)
    print('\n' + table)


def _load_snr_dataset(data_path, snr, channels=IQ_CHANNELS):
    dataset = RadioMLDataset(data_path, snr_range=(snr, snr), channels=channels)
    counts = np.bincount(dataset.labels.numpy(), minlength=len(MODULATIONS))
    print(f"SNR {snr} dB: {len(dataset):,} samples | channels: {dataset.channels} | "
          f"per class: {dict(zip(MODULATIONS, counts.tolist()))}")
    return dataset


def _summarize(results, n_splits):
    results['fold_sizes'] = [len(labels) for labels in results['labels']]
    results['labels'] = np.concatenate(results['labels'])
    results['preds'] = np.concatenate(results['preds'])

    acc = np.array(results['fold_acc'])
    print(f"\n{n_splits}-fold CV accuracy: {acc.mean():.3f} ± {acc.std():.3f} "
          f"(folds: {', '.join(f'{a:.3f}' for a in acc)})")
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Cross-validate the linear SVM and the 1D CNN on experiments A (raw I/Q), '
                    'B (amplitude + instantaneous frequency) and C (power-law FFT).')
    parser.add_argument('--inputs', nargs='+', choices=list(INPUTS), default=list(INPUTS),
                        help='experiments to run (default: all three)')
    parser.add_argument('--out-dir', type=Path, default=PROJECT_ROOT / 'results',
                        help='where to write the results tables and figures (default: results/)')
    parser.add_argument('--c-sweep', action='store_true',
                        help=f'also cross-validate the linear SVM for each C in {", ".join(f"{c:g}" for c in C_VALUES)}, '
                             'plus a nested CV that tunes C without bias; writes svm_c_sweep.md, '
                             'svm_c_sweep.csv and svm_nested_cv.csv')
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    run_experiments(out_dir=args.out_dir, inputs=args.inputs, c_sweep=args.c_sweep)
