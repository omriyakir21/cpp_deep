import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
import numpy as np
import torch
import pandas as pd
from models.baselines.convolution_baseline.convolution_baseline_utils import process_sequences, CNNModel, calculate_pr_auc
from utils import plot_pr_curve, load_as_pickle,save_as_pickle
import paths
import matplotlib.pyplot as plt
from results.result_analysis.predict_sub_sequences import predict_with_convolution,create_dataset_convolution
from scipy.special import softmax

def load_models(model_folder, filters, kernel_size, num_layers, batch_size,pu_iterations, negative_size, num_models=5):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_ensembles = []
    for i in range(num_models):
        fold_models = []
        fold_model_folder = os.path.join(model_folder,f'fold_models_{i}')
        for j in range(pu_iterations):
            model_path = os.path.join(fold_model_folder, f'model_subgroup_{j}.pt')
            model = CNNModel(filters=filters, kernel_size=kernel_size, num_layers=num_layers).to(device)
            model.load_state_dict(torch.load(model_path, map_location=device))
            model.eval()
            fold_models.append(model)
        model_ensembles.append(fold_models)
    return model_ensembles

def create_fold_ensemble_weights(folds_training_dicts,ensemble_models):
    fold_ensemble_weights = []
    for i in range(len(ensemble_models)):
        folds_training_dict = folds_training_dicts[i]
        sequences = create_dataset_convolution(folds_training_dict['sequences_validation'])
        labels = folds_training_dict['labels_validation']
        fold_models = ensemble_models[i]
        fold_pr_aucs = []
        for j in range(len(fold_models)):
            model = fold_models[j]
            predictions = predict_with_convolution(model, sequences)
            pr_auc = calculate_pr_auc(labels, predictions)
            fold_pr_aucs.append(pr_auc)
        # min_pr_auc = min(fold_pr_aucs)
        # pr_aucs_subtracted = [pr_auc - min_pr_auc for pr_auc in fold_pr_aucs]
        # fold_weights = [pr_auc / sum(pr_aucs_subtracted) for pr_auc in pr_aucs_subtracted]
        fold_weights = softmax(fold_pr_aucs)
        fold_ensemble_weights.append(np.array(fold_weights))
        print(f'Fold {i} ensemble weights: {fold_weights}')
        print(f'Fold {i} ensemble weights sum: {sum(fold_weights)}')
    return fold_ensemble_weights

def save_weighted_test_pr_auc(ensemble_models,folds_training_dicts,results_folder,fold_ensemble_weights):
    folds_weighted_predictions = []
    all_test_labels = []
    for i in range(len(ensemble_models)):
        folds_training_dict = folds_training_dicts[i]
        sequences = create_dataset_convolution(folds_training_dict['sequences_test'])
        labels = folds_training_dict['labels_test']
        all_test_labels.append(labels)
        fold_models = ensemble_models[i]
        fold_predictions = []
        for j in range(len(fold_models)):
            model = fold_models[j]
            predictions = predict_with_convolution(model, sequences)
            fold_predictions.append(predictions)
            print(f'Fold {i} model {j} predictions shape: {predictions.shape}')
            print(f'Fold {i} model {j} predictions: {predictions}')
        fold_predictions = np.array(fold_predictions)
        fold_ensemble_weights_reshaped = fold_ensemble_weights[i].reshape(-1, 1)
        weighted_predictions = np.sum(fold_predictions * fold_ensemble_weights_reshaped, axis=0)
        print(f'Fold {i} weighted predictions shape: {weighted_predictions.shape}')
        print(f'Fold {i} weighted predictions: {weighted_predictions}')
        folds_weighted_predictions.append(weighted_predictions)
    folds_weighted_predictions = np.concatenate(folds_weighted_predictions)
    all_test_labels = np.concatenate(all_test_labels)
    plot_pr_curve(all_test_labels, folds_weighted_predictions, save_path=os.path.join(results_folder, 'weighted_bagging_pr_curve_model.png'), title='Convolution bagging weighted Precision-Recall Curve')
    print(f'saved weighted bagging pr curve to {os.path.join(results_folder, "weighted_bagging_pr_curve_model.png")}')



if __name__ == "__main__":
    # Define paths
    DATE = '13_09'  # Update this as needed
    models_folder = os.path.join(paths.convolution_baseline_pu_learning_bagging_models_path, DATE)
    results_path = os.path.join(paths.convolution_baseline_pu_learning_bagging_results_path, DATE)
    if not os.path.exists(results_path):
        os.makedirs(results_path)
    data_for_training_dir = os.path.join(paths.data_for_training_path, DATE)
    folds_traning_dicts = load_as_pickle(os.path.join(data_for_training_dir,'folds_traning_dicts.pkl'))
    pu_iterations = 30
    negative_size = 0.3
    filters = 128  # Update according to model used
    kernel_size = 7  # Update according to model used
    num_layers = 3  # Update according to model used
    batch_size = 128
    architecture_models_folder = os.path.join(models_folder,f'models_{filters}_{kernel_size}_{num_layers}_{batch_size}_{pu_iterations}_{negative_size}')
    model_ensembles = load_models(architecture_models_folder, filters, kernel_size, num_layers, batch_size,pu_iterations, negative_size)
    fold_ensemble_weights = create_fold_ensemble_weights(folds_traning_dicts,model_ensembles)
    save_as_pickle(fold_ensemble_weights,os.path.join(architecture_models_folder,'fold_ensemble_weights.pkl'))
    save_weighted_test_pr_auc(model_ensembles,folds_traning_dicts,results_path,fold_ensemble_weights)
