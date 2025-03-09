import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..','..','..'))
import paths
# from models.baselines.convolution_baseline.convolution_baseline_utils import process_sequences,CNNModel \
#     ,score_function,calculate_pr_auc
from models.baselines.convolution_baseline.convolution_baseline_utils import calculate_pr_auc,process_sequences \
    ,check_if_parameters_can_create_valid_model,pu_metric_calculation,train_architecture_fold,calculate_roc_auc
from utils import load_as_pickle,plot_pr_curve,save_as_pickle,plot_roc_curve
import numpy as np
import torch
from results.result_analysis.predict_sub_sequences import create_dataset_convolution,predict_with_convolution
import pandas as pd
# from ignite.engine import Engine, Events
# from ignite.handlers import EarlyStopping
# from utils import plot_pr_curve
# from torchsummary import summary
# import torch.nn as nn
# import torch.optim as optim
# import matplotlib.pyplot as plt
# from sklearn.utils.class_weight import compute_class_weight
# from torch.utils.data import DataLoader, TensorDataset
import random
import json

def create_groups(indices, fraction, iterations_number):
    random.shuffle(indices)
    groups_size = int(fraction * len(indices))
    groups_for_train = [np.array(random.choices(indices, k=groups_size)) for _ in range(iterations_number)]
    return  groups_for_train


def sample_training_sequences(iteration,negative_size,folds_traning_dicts):
    training_sequences_folds = [fold['sequences_train'] for fold in folds_traning_dicts]
    label_folds = [fold['labels_train'] for fold in folds_traning_dicts]
    positive_indexes = [np.where(label_fold == np.int64(1))[0] for label_fold in label_folds]
    print(f'positive indices example shape: {positive_indexes[0].shape}')
    unlabeled_indexes = [np.where(label_fold == np.int64(0))[0] for label_fold in label_folds]
    print(f'unlabeled indices example shape: {unlabeled_indexes[0].shape}')
    folds_groups_for_train = []
    for i in range(len(training_sequences_folds)):
        groups_for_train = create_groups(unlabeled_indexes[i], negative_size, iteration)
        folds_groups_for_train.append(groups_for_train)
    subsampled_training_sequences_folds = []
    subsampled_labels_folds = []
    for i in range(len(training_sequences_folds)):
        fold_groups_for_train = folds_groups_for_train[i]
        groups_indices = [np.concatenate([positive_indexes[i], group_for_train]) for group_for_train in fold_groups_for_train]
        sequences = [np.array(training_sequences_folds[i])[indices] for indices in groups_indices]
        labels = [np.concatenate([np.ones(positive_indexes[i].shape[0]), np.zeros(len(group_for_train))]) for group_for_train in fold_groups_for_train]
        subsampled_training_sequences_folds.append(sequences)
        subsampled_labels_folds.append(labels)
    return subsampled_training_sequences_folds,subsampled_labels_folds

def save_best_models(best_models,grid_results,best_architecture_index,models_folder,pu_metric,modified_addition,ROC_metric):
    # Save the best models
    modified_addition = '_modified' if modified_metric else ''
    pu_metric_addition = '_pu_metric' if pu_metric else ''
    roc_metric_addition = '_roc_metric' if ROC_metric else ''

    best_filter = grid_results[best_architecture_index]['filters']
    best_kernel_size = grid_results[best_architecture_index]['kernel_size']
    best_num_layers = grid_results[best_architecture_index]['num_layers']
    best_batch_size = grid_results[best_architecture_index]['batch_size']
    best_iteration = grid_results[best_architecture_index]['pu_iterations']
    best_negative_size = grid_results[best_architecture_index]['negative_size']
    best_dialation = grid_results[best_architecture_index]['dialation']
    best_paddign = grid_results[best_architecture_index]['padding']
    print(f'Best model: Filter: {best_filter}, Kernel size: {best_kernel_size}, Num layers: {best_num_layers}, Batch size: {best_batch_size},iteration: {best_iteration},negative size: {best_negative_size},padding: {padding},dialation: {dialation} val_metric_score: {best_architecture_metric_score}')
    architecture_models_folder = os.path.join(models_folder,f'models_{best_filter}_{best_kernel_size}_{best_num_layers}_{best_batch_size}_{best_iteration}_{best_negative_size}_{best_paddign}_{best_dialation}')
    architecture_models_folder = f'{architecture_models_folder}{pu_metric_addition}{modified_addition}{roc_metric_addition}'
    if not os.path.exists(architecture_models_folder):
        os.makedirs(architecture_models_folder)
    for i in range(len(best_models)):
        fold_folder = os.path.join(architecture_models_folder,f'fold_models_{i}')
        os.makedirs(fold_folder, exist_ok=True)
        fold_best_models = best_models[i]
        for index, best_model in enumerate(fold_best_models):
            torch.save(best_model.state_dict(), os.path.join(fold_folder, f'model_subgroup_{index}.pt'))
    return architecture_models_folder

def save_test_metric_score(best_models,folds_traning_dicts,results_folder,pu_metric,modified_metric):
    # Evaluate the best model on the test set
    all_test_outputs = []
    all_test_labels = []
    for i in range(len(best_models)):
        fold_models = best_models[i]
        X_test = create_dataset_convolution(folds_traning_dicts[i]['sequences_test'])
        y_test = folds_traning_dicts[i]['labels_test']
        fold_subgroups_predictions = np.zeros(len(y_test))
        print(f'X_test shape: {X_test.shape}')
        print(f'len(fold_models): {len(fold_models)}')
        for best_model in fold_models:
            subgroup_predictions = predict_with_convolution(best_model,X_test)
            fold_subgroups_predictions += subgroup_predictions
        fold_subgroups_predictions /= len(fold_models)
        print(f'fold_subgroups_predictions shape: {fold_subgroups_predictions.shape}')
        all_test_outputs.append(fold_subgroups_predictions)
        all_test_labels.append(y_test)
    all_test_outputs = np.concatenate(all_test_outputs)
    all_test_labels = np.concatenate(all_test_labels)

    print(f'all_test_outputs shape: {all_test_outputs.shape}')
    print(f'all_test_labels shape: {all_test_labels.shape}')

    sequences = []
    for fold_dict in folds_traning_dicts:
        sequences.extend(fold_dict['sequences_test'])
    print(f'sequences shape: {len(sequences)}')
    df = pd.DataFrame({
        'sequence': sequences,
        'prediction': all_test_outputs,
        'label': all_test_labels
    })

    csv_path = os.path.join(results_folder, 'predictions_labels_sequences.csv')
    df.to_csv(csv_path, index=False)
    print(f'Saved predictions, labels, and sequences to {csv_path}')
    if pu_metric:
        test_metric = pu_metric_calculation(all_test_labels,all_test_outputs,modified_metric)
        #save score as json
        with open(os.path.join(results_folder, 'test_metric.json'), 'w') as f:
            json.dump({'test_metric': test_metric}, f)
    plot_roc_curve(all_test_labels, all_test_outputs, save_path=os.path.join(results_folder, 'roc_curve_model.png'), title='Convolution Baseline ROC Curve')
    # Plot the precision-recall curve
    plot_pr_curve(all_test_labels, all_test_outputs, save_path=os.path.join(results_folder, 'pr_curve_model.png'), title='Convolution Baseline Precision-Recall Curve')

def save_grid_search_results(grid_results,results_folder):
    #save the grid search results to a CSV file
    results_df = pd.DataFrame(grid_results)
    #sort the results by pr_auc
    results_df = results_df.sort_values(by='val_metric_score', ascending=False)
    results_df.to_csv(os.path.join(results_folder, 'grid_search_results.csv'), index=False)


if __name__ == '__main__':
    DATE = '13_02'
    pu_metric = False
    modified_metric = False
    ROC_metric = True
    data_for_training_dir = os.path.join(paths.data_for_training_path, DATE)
    results_folder = os.path.join(paths.convolution_baseline_pu_learning_bagging_results_path, DATE)
    if not os.path.exists(results_folder):
        os.makedirs(results_folder)
    models_folder = os.path.join(paths.convolution_baseline_pu_learning_bagging_models_path, DATE)
    if not os.path.exists(models_folder):
        os.makedirs(models_folder)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    folds_traning_dicts = load_as_pickle(os.path.join(data_for_training_dir,'folds_traning_dicts.pkl'))

        # Define the grid search parameters
    param_grid = {
        'pu_iterations': [30,100],
        'negative_size': [0.3],
        'dialation' : [1],
        'padding': ['valid'],
        'filters': [64, 128],
        'kernel_size': [5],
        'num_layers': [3],
        'batch_size': [256],  
    }
 
    # param_grid = {
    #     'pu_iterations': [10],
    #     'negative_size': [0.1,0.3],
    #     'dialation' : [2],
    #     'padding': ['valid','same'],
    #     'filters': [16],
    #     'kernel_size': [3],
    #     'num_layers': [2],
    #     'batch_size': [64],  
    # }

    epochs = 200    
    best_model = None
    best_architecture_metric_score = 0
    best_architecture_index = 0
    grid_results = []
    cnt = 0
    for iteration in param_grid['pu_iterations']:
        for negative_size in param_grid['negative_size']:
            subsampled_training_sequences_folds,subsampled_labels_folds = sample_training_sequences(iteration,negative_size,folds_traning_dicts)
            for conv_filter in param_grid['filters']:
                for kernel_size in param_grid['kernel_size']:
                    for num_layers in param_grid['num_layers']:
                        for batch_size in param_grid['batch_size']:
                            for dialation in param_grid['dialation']:
                                for padding in param_grid['padding']:
                                    if not check_if_parameters_can_create_valid_model(kernel_size, num_layers, dialation):
                                            continue
                                    architecture_metrics_scores = []
                                    architecture_models = []
                                    for i in range(len(subsampled_training_sequences_folds)):
                                        fold_training_dict = folds_traning_dicts[i]
                                        architecture_fold_models = []
                                        X_val = create_dataset_convolution(folds_traning_dicts[i]['sequences_validation'])
                                        y_val = folds_traning_dicts[i]['labels_validation']
                                        sub_subgroups_predictions = np.zeros(len(y_val))
                                        for j in range(len(subsampled_training_sequences_folds[i])):
                                            _,architecture_subgroup_model = train_architecture_fold(conv_filter,kernel_size,num_layers,batch_size,padding,dialation,epochs,fold_training_dict,subsampled_training_sequences_folds[i][j],subsampled_labels_folds[i][j],pu_metric,ROC_metric)
                                            architecture_fold_models.append(architecture_subgroup_model)
                                            architecture_subgroup_predictions = predict_with_convolution(architecture_subgroup_model,X_val)
                                            sub_subgroups_predictions += architecture_subgroup_predictions
                                        sub_subgroups_predictions /= len(subsampled_training_sequences_folds[i])
                                        if ROC_metric:
                                            val_metric_score = calculate_roc_auc(y_val,sub_subgroups_predictions)
                                        elif pu_metric:
                                            val_metric_score,_ = pu_metric_calculation(y_val,sub_subgroups_predictions,modified_metric)
                                        else:
                                            val_metric_score = calculate_pr_auc(y_val,sub_subgroups_predictions)
                                        print(f'architecture: iteration: {iteration}, negative_size: {negative_size}, filter: {conv_filter}, kernel_size: {kernel_size}, num_layers: {num_layers}, batch_size: {batch_size},padding: {padding},dialation: {dialation}, val_pr_auc: {val_metric_score}')
                                        architecture_models.append(architecture_fold_models)
                                        architecture_metrics_scores.append(val_metric_score)
                                    architecture_metric_score = np.mean(architecture_metrics_scores)
                                    
                                    # Save the grid search results
                                    grid_results.append({
                                        'pu_iterations': iteration,
                                        'negative_size': negative_size,
                                        'filters': conv_filter,
                                        'kernel_size': kernel_size,
                                        'num_layers': num_layers,
                                        'batch_size': batch_size,
                                        'dialation': dialation,
                                        'padding': padding,
                                        'epochs': epochs,
                                        'val_metric_score': architecture_metric_score
                                    })


                                    # Save the best models
                                    if architecture_metric_score > best_architecture_metric_score:
                                        best_architecture_metric_score = architecture_metric_score
                                        best_models = architecture_models
                                        best_architecture_index = cnt
                                    cnt += 1    
    
    architecture_models_folder = save_best_models(best_models,grid_results,best_architecture_index,models_folder,pu_metric,modified_metric,ROC_metric)
    architecture_results_folder = os.path.join(results_folder, architecture_models_folder.split('/')[-1])
    if not os.path.exists(architecture_results_folder):
        os.makedirs(architecture_results_folder)
    save_as_pickle(subsampled_training_sequences_folds,os.path.join(architecture_models_folder,f'subsampled_training_sequences_folds_{iteration}_{negative_size}.pkl'))
    save_test_metric_score(best_models,folds_traning_dicts,architecture_results_folder,pu_metric,modified_metric)
    save_grid_search_results(grid_results,architecture_results_folder)