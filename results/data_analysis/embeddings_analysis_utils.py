import pandas as pd
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
import torch
import paths
from data_preperation.embeddings_creator_utils import esm2_model_names, FULL_DATASET_NAME
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import numpy as np
import os
from datetime import datetime


def load_esm2_embeddings(date):
    """
    Load embeddings from a CSV file.

    Args:
        DATE (str): date of the embeddings file.

    Returns:
        list: A tuple containing embeddings of all models
    """
    embeddings_files = [os.path.join(paths.esm2_embeddings_path, f'{model_name.split("/")[1]}_embedding_{date}.pt') for
                        model_name in esm2_model_names]

    sequence_embeddings = [torch.load(embeddings_file) for embeddings_file in embeddings_files]

    return sequence_embeddings


def load_labels(date):
    """
    Load labels from a CSV file.

    Args:
        DATE (str): date of the labels file.

    Returns:
        ndarray: ndarray containing labels of the sequences
    """
    df = pd.read_csv(os.path.join(paths.full_datasets_path, f'{FULL_DATASET_NAME}_{date}.csv'))

    labels = np.array(df['label'].tolist())

    return labels


def load_sequences(date):
    """
    Load labels from a CSV file.

    Args:
        DATE (str): date of the labels file.

    Returns:
        ndarray: ndarray containing labels of the sequences
    """
    df = pd.read_csv(os.path.join(paths.full_datasets_path, f'{FULL_DATASET_NAME}_{date}.csv'))

    labels = np.array(df['sequence'].tolist())

    return labels


def plot_tsne(embeddings, labels, model_name, save_dir, title='t-SNE plot of embeddings'):
    """
    Plot t-SNE of embeddings and save the plot.

    Args:
        embeddings (torch tensor): Embeddings to plot.
        labels (numpy.ndarray): Labels corresponding to the embeddings.
        model_name (str): Name of the model used to generate embeddings.
        save_dir (str): Directory where the plot will be saved.
        title (str): Title of the plot.
    """
    embeddings = embeddings.numpy()
    tsne = TSNE(n_components=2, random_state=42)
    embeddings_2d = tsne.fit_transform(embeddings)

    plt.figure(figsize=(10, 8))
    for label in np.unique(labels):
        indices = labels == label
        plt.scatter(embeddings_2d[indices, 0], embeddings_2d[indices, 1], label=f'Label {label}', alpha=0.5)

    plt.title(f'{title} - {model_name.split("/")[1]}')
    plt.xlabel(f't-SNE component 1')
    plt.ylabel('t-SNE component 2')
    plt.legend()

    # Save the plot
    current_date = datetime.now().strftime('%d_%m')
    plot_filename = f'tsne_plot_{model_name.split("/")[1]}_{current_date}.png'
    plot_path = os.path.join(save_dir, plot_filename)
    plt.savefig(plot_path)
    plt.close()


# Add this function to `embeddings_analysis_utils.py`
def calculate_sequence_lengths(sequences):
    """
    Calculate the lengths of sequences.

    Args:
        sequences (list): List of sequences.

    Returns:
        list: List of sequence lengths.
    """
    return [len(seq) for seq in sequences]


# Add this function to `embeddings_analysis_utils.py`
def save_sequence_length_histogram(sequences, labels, save_dir, title='Histogram of Sequence Lengths'):
    """
    Plot histogram of sequence lengths for labels 0 and 1.

    Args:
        sequences (list): List of sequences.
        labels (numpy.ndarray): Labels corresponding to the sequences.
        save_dir (str): Directory where the plot will be saved.
        title (str): Title of the plot.
    """
    lengths = calculate_sequence_lengths(sequences)

    lengths_label_0 = [lengths[i] for i in range(len(labels)) if labels[i] == 0]
    lengths_label_1 = [lengths[i] for i in range(len(labels)) if labels[i] == 1]

    plt.figure(figsize=(10, 8))
    plt.hist(lengths_label_0, bins=30, alpha=0.5, label='Label 0', color='blue')
    plt.hist(lengths_label_1, bins=30, alpha=0.5, label='Label 1', color='orange')

    plt.title(title)
    plt.xlabel('Sequence Length')
    plt.ylabel('Frequency')
    plt.legend()

    # Save the plot
    current_date = datetime.now().strftime('%d_%m')
    plot_filename = f'sequence_length_histogram_{current_date}.png'
    plot_path = os.path.join(save_dir, plot_filename)
    plt.savefig(plot_path)
    plt.close()
