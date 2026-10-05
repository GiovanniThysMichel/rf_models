import pickle
import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import Dataset, DataLoader, TensorDataset

MODULATIONS = ['8PSK', 'AM-DSB', 'AM-SSB', 'BPSK', 'CPFSK', 'GFSK', 'PAM4', 'QAM16', 'QAM64', 'QPSK', 'WBFM']
MOD_TO_IDX = {m: i for i, m in enumerate(MODULATIONS)}
SNR_VALUES = list(range(-20, 20, 2))  # -20 to +18 dB in steps of 2

# Model inputs for experiments A, B and C (see make_channels)
IQ_CHANNELS = ('I', 'Q')                                                  # A: raw I/Q, 2 x 128
AP_CHANNELS = ('amplitude', 'cos_phase_change', 'sin_phase_change')       # B: [A, cos Δφ, sin Δφ], 3 x 127
POWER_CHANNELS = ('log_fft_x', 'log_fft_x2', 'log_fft_x4', 'log_fft_x8')  # C: log|FFT(x^k)|, k = 1, 2, 4, 8, 4 x 128



class RadioMLDataset(Dataset):
    def __init__(self, path, snr_range=None, channels=IQ_CHANNELS):
        """
        Args:
            path:            path to the RML2016.10a dict pickle
            snr_range:       optional (min_snr, max_snr) to filter by SNR, inclusive;
                             use (snr, snr) for a single SNR level
            channels:        model input channels, see make_channels(); raw I and Q by default.
                             Each sample is scaled to unit RMS power over I and Q first.
        """
        with open(path, 'rb') as f:
            raw = pickle.load(f, encoding='latin1')

        samples, labels, snrs = [], [], []
        for (mod, snr), arr in raw.items():
            if snr_range is not None and not (snr_range[0] <= snr <= snr_range[1]):
                continue
            samples.append(arr)
            labels.append(np.full(len(arr), MOD_TO_IDX[mod]))
            snrs.append(np.full(len(arr), snr))

        iq = np.concatenate(samples).astype(np.float32)  # (N, 2, 128)
        iq /= np.sqrt(np.mean(iq ** 2, axis=(1, 2), keepdims=True)) + 1e-8  # unit RMS power over I and Q

        self.channels = list(channels)
        self.data = torch.from_numpy(make_channels(iq, self.channels))  # (N, len(channels), L)
        self.labels = torch.from_numpy(np.concatenate(labels)).long()
        self.snrs = np.concatenate(snrs)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


def make_channels(iq, channels):
    """
    Build model inputs of shape (N, len(channels), L) from IQ samples of shape (N, 2, 128),
    with x[n] = I[n] + jQ[n]:

        'I', 'Q'            the IQ samples themselves
        'amplitude'         A[n] = |x[n]|
        'cos_phase_change', cos Δφ[n] and sin Δφ[n], with Δφ[n] = angle(x[n+1] x*[n]) the
        'sin_phase_change'  instantaneous frequency over the 127 sample steps; together they
                            encode Δφ without the jump between +π and -π
        'log_fft_x', ...,   log|FFT(x^k)| for k = 1, 2, 4, 8: the magnitude spectrum of x^k, centred
        'log_fft_x8'        on DC, with 1e-8 added before the log. Raising M-PSK to the M-th power
                            leaves a spectral line: BPSK in x^2, QPSK in x^4, 8PSK in x^8.

    All channels are cut to the shortest one, so L is 127 when cos/sin_phase_change are used
    (AP_CHANNELS, with A[n] kept for n = 0..126) and 128 otherwise.
    """
    x = iq[:, 0] + 1j * iq[:, 1]
    arrays = [_channel(iq, x, name) for name in channels]
    length = min(a.shape[1] for a in arrays)
    return np.stack([a[:, :length] for a in arrays], axis=1).astype(np.float32)


def _channel(iq, x, name):
    if name == 'I':
        return iq[:, 0]
    if name == 'Q':
        return iq[:, 1]
    if name == 'amplitude':
        return np.abs(x)
    if name == 'cos_phase_change':
        return np.cos(_phase_change(x))
    if name == 'sin_phase_change':
        return np.sin(_phase_change(x))
    if name.startswith('log_fft_x'):
        xk = x ** int(name[len('log_fft_x'):] or 1)
        return np.log(np.abs(np.fft.fftshift(np.fft.fft(xk, axis=1), axes=1)) + 1e-8)
    raise ValueError(f"unknown channel {name!r}")


def _phase_change(x):
    return np.angle(x[:, 1:] * np.conj(x[:, :-1]))  # (N, 127)


def get_kfold_indices(labels, n_splits=5, seed=42):
    """
    Stratified k-fold split of the sample labels.

    At a single SNR level RML2016.10a has 1000 samples per class (11 x 1000 = 11,000),
    so with n_splits=5 every fold trains on 800 and holds out 200 samples per class.

    Returns a list of (train_idx, val_idx), one per fold. The val folds are disjoint
    and together cover the whole dataset exactly once. The CNN and the SVM both split
    through here, so for the same seed they see identical folds.
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return list(skf.split(np.zeros(len(labels)), labels))


def standardize_channels(data, train_idx):
    """
    Standardise each channel of data (N, C, L) with the mean and std of the training
    samples train_idx, so the channels share a common scale and nothing about the
    held-out samples leaks into training. The CNN and the SVM both get their input from
    here, so for the same fold they see identical values.
    """
    train_idx = torch.as_tensor(train_idx)
    mean = data[train_idx].mean(dim=(0, 2), keepdim=True)
    std = data[train_idx].std(dim=(0, 2), keepdim=True)
    return (data - mean) / std


def get_kfold_dataloaders(dataset, n_splits=5, batch_size=256, seed=42):
    """
    Stratified k-fold split of a RadioMLDataset (see get_kfold_indices), with each fold's
    channels standardised on its training samples (see standardize_channels).

    Returns a list of (train_loader, val_loader), one per fold.
    """
    folds = []
    for fold, (train_idx, val_idx) in enumerate(get_kfold_indices(dataset.labels.numpy(), n_splits, seed)):
        train_idx, val_idx = torch.from_numpy(train_idx), torch.from_numpy(val_idx)
        data = standardize_channels(dataset.data, train_idx)

        generator = torch.Generator().manual_seed(seed + fold)
        train_loader = DataLoader(TensorDataset(data[train_idx], dataset.labels[train_idx]),
                                  batch_size=batch_size, shuffle=True, generator=generator, num_workers=0)
        val_loader   = DataLoader(TensorDataset(data[val_idx], dataset.labels[val_idx]),
                                  batch_size=batch_size, shuffle=False, num_workers=0)
        folds.append((train_loader, val_loader))

    return folds
