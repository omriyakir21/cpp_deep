import os
import sys
import paths
import pandas as pd
import pickle
import numpy as np

def save_as_pickle(data, file_path):
    with open(file_path, 'wb') as file:
        pickle.dump(data, file)

def load_as_pickle(file_path):
    with open(file_path, 'rb') as file:
        data = pickle.load(file)
    return data

def load_labels(date,data_set_name):
    """
    Load labels from a CSV file.

    Args:
        DATE (str): date of the labels file.

    Returns:
        ndarray: ndarray containing labels of the sequences
    """
    df = pd.read_csv(os.path.join(paths.full_datasets_path, f'{data_set_name}_{date}.csv'))

    labels = np.array(df['label'].tolist())

    return labels


def load_sequences(date,data_set_name):
    """
    Load labels from a CSV file.

    Args:
        DATE (str): date of the labels file.

    Returns:
        ndarray: ndarray containing labels of the sequences
    """
    df = pd.read_csv(os.path.join(paths.full_datasets_path, f'{data_set_name}_{date}.csv'))

    labels = np.array(df['sequence'].tolist())

    return labels

def load_ids(date,data_set_name):
    """
    Load IDs from a CSV file.

    Args:
        date (str): date of the IDs file.

    Returns:
        ndarray: ndarray containing IDs of the sequences
    """
    df = pd.read_csv(os.path.join(paths.full_datasets_path, f'{data_set_name}_{date}.csv'))

    ids = np.array(df['id'].tolist())

    return ids