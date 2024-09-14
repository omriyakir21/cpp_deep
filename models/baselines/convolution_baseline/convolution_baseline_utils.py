import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..','..','..'))
import paths
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import precision_recall_curve, auc
from ignite.engine import Engine, Events
from ignite.handlers import EarlyStopping
from ignite.metrics import Metric


AMINO_ACIDS = 'ACDEFGHIKLMNPQRSTVWY'
AA_TO_INDEX = {aa: idx for idx, aa in enumerate(AMINO_ACIDS)}

def one_hot_encode(sequence):
    """One-hot encode a sequence of amino acids."""
    encoding = np.zeros((len(sequence), len(AMINO_ACIDS)))
    for i, aa in enumerate(sequence):
        if aa in AA_TO_INDEX:
            encoding[i, AA_TO_INDEX[aa]] = 1
    return encoding

def pad_sequence(encoded_sequence, max_length=50):
    """Pad the one-hot encoded sequence with zeros to a fixed length."""
    if len(encoded_sequence) >= max_length:
        return encoded_sequence[:max_length]
    else:
        padding = np.zeros((max_length - len(encoded_sequence), len(AMINO_ACIDS)))
        return np.vstack((encoded_sequence, padding))
    
def process_sequences(sequences):
    """Process a list of amino acid sequences by one-hot encoding and padding."""
    processed_sequences = []
    for seq in sequences:
        encoded = one_hot_encode(seq)
        padded = pad_sequence(encoded)
        processed_sequences.append(padded)
    return np.array(processed_sequences)

class CNNModel(nn.Module):
    def __init__(self, filters=32, kernel_size=3, num_layers=1):
        super(CNNModel, self).__init__()
        layers = []
        length = 50
        in_channels = 20
        for _ in range(num_layers):
            layers.append(nn.Conv1d(in_channels=in_channels, out_channels=filters, kernel_size=kernel_size))
            layers.append(nn.ReLU())
            layers.append(nn.MaxPool1d(kernel_size=2))
            in_channels = filters
            length = (length - kernel_size + 1) // 2

        self.conv_layers = nn.Sequential(*layers)
        self.fc1 = nn.Linear(filters*length, 128)
        self.fc2 = nn.Linear(128, 1)
    
    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        return x
    
    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

def calculate_pr_auc(y_true, y_pred):
    precision, recall, _ = precision_recall_curve(y_true, y_pred)
    return auc(recall, precision)

def score_function(engine):
    y_true, y_pred = engine.state.output
    val_pr_auc = calculate_pr_auc(y_true.cpu().numpy(), y_pred.cpu().numpy())
    print(f"Epoch {engine.state.epoch} - val-prAUC: {val_pr_auc}")
    return val_pr_auc

