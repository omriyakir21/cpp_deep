import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import pickle
import paths
from results.data_analysis.embeddings_analysis_utils import load_esm2_embeddings, load_labels, load_sequences
from cross_validation_utils import k_fold_cross_validation

if __name__ == '__main__':
    DATE = '01_08'
    THRESHOLD = 0.5
    K = 5
    sequence_embeddings = load_esm2_embeddings(DATE)
    labels = load_labels(DATE)
    sequences = load_sequences(DATE)
    train_sequences, test_sequences, train_embeddings, test_embeddings, train_labels, test_labels = k_fold_cross_validation(
        sequences, sequence_embeddings, labels, k=K, threshold=THRESHOLD)
    result_dict = {"train_sequences": train_sequences, "test_sequences": test_sequences,
                   "train_embeddings": train_embeddings, "test_embeddings": test_embeddings,
                   "train_labels": train_labels, "test_labels": test_labels}
    with open(f'{os.path.join(paths.cross_validation_path, DATE)}.pkl', 'wb') as f:
        pickle.dump(result_dict, f)
