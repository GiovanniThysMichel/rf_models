import torch
import torch.nn as nn
from sklearn.svm import LinearSVC

class ModulationCNN(nn.Module):
    """
    Minimal 1D CNN for radio modulation classification from IQ samples or features derived from them.

    Conv1d(in_channels -> 8, kernel 7, stride 2) -> ReLU -> global average pooling over time
    -> Linear(8 -> num_classes). 

    Input: (batch, in_channels, L), L = 128 for experiments A and C, 127 for B
    Output: (batch, num_classes) logits
    """

    def __init__(self, num_classes=11, in_channels=2):
        super().__init__()

        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=8,
            kernel_size=7,
            stride=2
        )

        self.relu = nn.ReLU()
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(8, num_classes)

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        x = self.pool(x)
        x = x.squeeze(-1)
        return self.fc(x)


def make_linear_svm(C=1.0):
    """
    Linear SVM for radio modulation classification, on the same input channels as
    ModulationCNN, flattened.
    Input: (n_samples, in_channels * L), i.e. 256 (A), 381 (B) or 512 (C) features
    Output: (n_samples,) class indices
    """
    return LinearSVC(C=C)