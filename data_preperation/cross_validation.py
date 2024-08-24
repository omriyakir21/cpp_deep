import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import utils 
import paths
from results.data_analysis.embeddings_analysis_utils import load_esm2_embeddings
from data_preperation.dataset_creator_utils import FULL_DATASET_NAME
from cross_validation_utils import partition_to_folds_and_save

if __name__ == '__main__':
    DATE = '01_08'
    THRESHOLD = 0.5
    K = 5
    # sequence_embeddings = load_esm2_embeddings(DATE)
    # labels = load_labels(DATE,FULL_DATASET_NAME)
    # sequences = load_sequences(DATE,FULL_DATASET_NAME)
    # ids = load_ids(DATE,FULL_DATASET_NAME)
    # folds_data_dicts = k_fold_cross_validation(sequences, sequence_embeddings, labels, ids, k=K, threshold=THRESHOLD)
    # utils.save_as_pickle(folds_data_dicts, f'{os.path.join(paths.data_for_training, DATE,"folds_data_dicts")}.pkl')
    # partition_to_folds_and_save(DATE,FULL_DATASET_NAME)
    folds_trainig_dicts = utils.load_as_pickle(f'{os.path.join(paths.data_for_training_path, DATE,"folds_traning_dicts")}.pkl')
    print(folds_trainig_dicts[0].keys())