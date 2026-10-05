import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, precision_recall_fscore_support

from .dataset import MODULATIONS



def evaluate(model, loader, criterion, device):
    """Mean loss and accuracy of model over loader."""
    model.eval()
    total_loss, correct = 0.0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            total_loss += criterion(logits, y).item() * len(x)
            correct += (logits.argmax(1) == y).sum().item()
    return total_loss / len(loader.dataset), correct / len(loader.dataset)


def predict(model, loader, device):
    """(labels, preds) as numpy arrays, in loader order."""
    model.eval()
    all_labels, all_preds = [], []
    with torch.no_grad():
        for x, y in loader:
            all_preds.append(model(x.to(device)).argmax(1).cpu().numpy())
            all_labels.append(y.numpy())
    return np.concatenate(all_labels), np.concatenate(all_preds)


def classification_metrics(labels, preds):
    """
    Precision, recall and F1 for each class, and their macro averages (the unweighted mean
    over the classes). A class that is never predicted gets precision 0.

    Returns {'per_class': {modulation: {'precision', 'recall', 'f1', 'support'}},
             'macro': {'precision', 'recall', 'f1'}}.
    """
    precision, recall, f1, support = precision_recall_fscore_support(
        labels, preds, labels=range(len(MODULATIONS)), zero_division=0)
    per_class = {mod: {'precision': float(precision[i]), 'recall': float(recall[i]),
                       'f1': float(f1[i]), 'support': int(support[i])}
                 for i, mod in enumerate(MODULATIONS)}
    macro = {'precision': float(precision.mean()), 'recall': float(recall.mean()), 'f1': float(f1.mean())}
    return {'per_class': per_class, 'macro': macro}


def macro_metrics_per_fold(labels, preds, fold_sizes):
    """Macro precision, recall and F1 of each fold, from out-of-fold labels/preds concatenated
    fold by fold (as returned by the cross-validation functions, with their fold_sizes)."""
    bounds = np.cumsum([0, *fold_sizes])
    return [classification_metrics(labels[a:b], preds[a:b])['macro'] for a, b in zip(bounds[:-1], bounds[1:])]


def plot_confusion_matrix(labels, preds, title='Confusion Matrix', save_path=None):
    """
    Row-normalised confusion matrix. Pass the out-of-fold labels/preds from
    cross_validate() to get one matrix over every sample, each predicted by the
    fold model that did not train on it.
    """
    cm = confusion_matrix(labels, preds, labels=range(len(MODULATIONS)), normalize='true')
    fig, ax = plt.subplots(figsize=(11, 9))
    disp = ConfusionMatrixDisplay(cm, display_labels=MODULATIONS)
    disp.plot(ax=ax, colorbar=False, cmap='Blues', values_format='.2f')
    ax.set_title(title, fontsize=14)
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150)
    return fig


def plot_cv_curves(histories, title=None, save_path=None):
    """Train loss, held-out loss and held-out accuracy per epoch, one line per fold."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    if title:
        fig.suptitle(title, fontsize=14)

    for fold, h in enumerate(histories, start=1):
        epochs = range(1, len(h['train_loss']) + 1)
        axes[0].plot(epochs, h['train_loss'], label=f'Fold {fold}')
        axes[1].plot(epochs, h['val_loss'], label=f'Fold {fold}')
        axes[2].plot(epochs, [a * 100 for a in h['val_acc']], label=f'Fold {fold}')

    for ax, ylabel, title in zip(axes,
                                 ['Cross-Entropy Loss', 'Cross-Entropy Loss', 'Accuracy (%)'],
                                 ['Training Loss', 'Held-out Loss', 'Held-out Accuracy']):
        ax.set_xlabel('Epoch')
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend()

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    return fig
