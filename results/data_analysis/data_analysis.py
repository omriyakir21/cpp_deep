import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
import paths
from utils import load_as_pickle
if __name__ == '__main__':
    DATE = '13_09'
    folds_traning_dicts = load_as_pickle(os.path.join(paths.data_for_training_path, DATE, 'folds_traning_dicts.pkl'))
    for i, fold in enumerate(folds_traning_dicts):
        test_labels = fold['labels_test']
        positives_percentage = sum(test_labels) / len(test_labels) * 100
        print(f"Fold {i}: {positives_percentage:.2f}% positives")