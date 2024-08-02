from sklearn.model_selection import KFold
from sklearn.cluster import AgglomerativeClustering
from Bio import pairwise2
import numpy as np


def calculate_sequence_identity(seq1, seq2):
    """
    Calculate the sequence identity between two sequences.

    Args:
        seq1 (str): The first sequence.
        seq2 (str): The second sequence.

    Returns:
        float: The sequence identity as a fraction.
    """
    alignments = pairwise2.align.globalxx(seq1, seq2)
    best_alignment = alignments[0]
    identity = sum(res1 == res2 for res1, res2 in zip(best_alignment.seqA, best_alignment.seqB)) / len(
        best_alignment.seqA)
    return identity


def cluster_sequences(sequences, threshold=0.5):
    """
    Cluster sequences based on sequence identity using Agglomerative Clustering.

    Args:
        sequences (list): List of sequences to cluster.
        threshold (float): Sequence identity threshold for clustering.

    Returns:
        list: List of clusters, where each cluster is a list of sequence indices.
    """
    # Calculate the distance matrix
    n = len(sequences)
    distance_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            distance_matrix[i, j] = 1 - calculate_sequence_identity(sequences[i], sequences[j])
            distance_matrix[j, i] = distance_matrix[i, j]

    # Perform Agglomerative Clustering
    clustering = AgglomerativeClustering(n_clusters=None, affinity='precomputed', linkage='average',
                                         distance_threshold=1 - threshold)
    cluster_labels = clustering.fit_predict(distance_matrix)

    # Group sequence indices into clusters
    clusters = [[] for _ in range(max(cluster_labels) + 1)]
    for idx, label in enumerate(cluster_labels):
        clusters[label].append(idx)

    return clusters


def divide_clusters_to_folds(sequences, clusters, k):
    """
    Divide clusters into k folds with an equal number of sequences, keeping clusters whole.

    Args:
        sequences(ndarray): The list of sequences.
        clusters (list): List of clusters to divide.
        k (int): Number of folds.

    Returns:
        tuple: A tuple containing:
            - list of k folds, where each fold is a list of sequences.
            - list of k folds, where each fold is a list of cluster labels.
    """
    total_sequences = len(sequences)
    sequences_per_fold = total_sequences // k
    remainder = total_sequences % k

    folds = [[] for _ in range(k)]
    fold_labels = [[] for _ in range(k)]
    fold_sizes = [0] * k
    current_fold = 0

    for cluster_idx, cluster in enumerate(clusters):
        cluster_size = len(cluster)
        if fold_sizes[current_fold] + cluster_size <= sequences_per_fold + (1 if remainder > 0 else 0):
            folds[current_fold].extend(cluster)
            fold_labels[current_fold].extend([cluster_idx] * cluster_size)
            fold_sizes[current_fold] += cluster_size
        else:
            current_fold += 1
            if current_fold >= k:
                current_fold = 0
            folds[current_fold].extend(cluster)
            fold_labels[current_fold].extend([cluster_idx] * cluster_size)
            fold_sizes[current_fold] += cluster_size
        if remainder > 0 and fold_sizes[current_fold] >= sequences_per_fold + 1:
            remainder -= 1

    return folds, fold_labels


def k_fold_cross_validation(sequences, embeddings, labels, k=5, threshold=0.9):
    """
    Perform k-fold cross-validation based on sequence identity.

    Args:
        sequences (list): List of sequences.
        embeddings (numpy.ndarray): Embeddings corresponding to the sequences.
        labels (numpy.ndarray): Labels corresponding to the sequences.
        k (int): Number of folds.
        threshold (float): Sequence identity threshold for clustering.

    Yields:
        tuple: Training and testing sets for each fold.
    """
    clusters = cluster_sequences(sequences, threshold)
    folds = divide_clusters_to_folds(sequences, clusters, k)

    kf = KFold(n_splits=k)
    for train_index, test_index in kf.split(folds):
        # Generate training and testing sets for the current fold
        train_sequences = [sequences[i] for i in train_index]
        test_sequences = [sequences[i] for i in test_index]
        train_embeddings = embeddings[train_index]
        test_embeddings = embeddings[test_index]
        train_labels = labels[train_index]
        test_labels = labels[test_index]

        yield train_sequences, test_sequences, train_embeddings, test_embeddings, train_labels, test_labels
