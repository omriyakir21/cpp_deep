import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils import load_labels, load_sequences, load_ids,save_as_pickle,load_as_pickle
import paths
from results.data_analysis.embeddings_analysis_utils import load_esm2_embeddings
from data_preperation.dataset_creator_utils import FULL_DATASET_NAME
from cross_validation_utils import partition_to_folds_and_save

if __name__ == '__main__':
    DATE = '10_09'
    THRESHOLD = 0.5
    K = 5
    sequence_embeddings = load_esm2_embeddings(DATE)
    labels = load_labels(DATE,FULL_DATASET_NAME)
    sequences = load_sequences(DATE,FULL_DATASET_NAME)
    ids = load_ids(DATE,FULL_DATASET_NAME)
    partition_to_folds_and_save(DATE,FULL_DATASET_NAME)
    folds_trainig_dicts = load_as_pickle(f'{os.path.join(paths.data_for_training_path, DATE,"folds_traning_dicts")}.pkl')
    print(folds_trainig_dicts[0].keys())