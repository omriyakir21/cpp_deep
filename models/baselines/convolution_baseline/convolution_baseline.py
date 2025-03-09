import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..','..','..'))
import paths
from models.baselines.convolution_baseline.convolution_baseline_utils import process_sequences,CNNModel \
    ,score_function,calculate_pr_auc \
        ,check_if_parameters_can_create_valid_model,roc_score_function,calculate_roc_auc,train_architecture_fold
from utils import load_as_pickle,plot_pr_curve,plot_roc_curve
import numpy as np
import torch
import pandas as pd
from ignite.engine import Engine, Events
from ignite.handlers import EarlyStopping
from utils import plot_pr_curve
import torch.nn as nn
import matplotlib.pyplot as plt

import json


def save_best_models(best_models,grid_results,best_architecture_index,models_folder,ROC_metric,use_pesg_auc,auc_margin_pretrained):
    # Save the best models
    auc_margin_pretrained_addition = '_aucm_pretrained' if auc_margin_pretrained else ''
    roc_metric_addition = '_roc_metric' if ROC_metric else ''
    use_pesg_auc_addition = '_pesg_auc' if use_pesg_auc else ''    

    best_architecture_dict = grid_results[best_architecture_index]
    val_metric = best_architecture_dict['val_metric']
    del best_architecture_dict['val_metric']
    best_architecture_string = '_'.join([f'{key}_{value}' for key, value in best_architecture_dict.items()])
    best_architecture_dict['val_metric'] = val_metric
    print(f'Best model: {best_architecture_string}')

    models_architecture_folder = os.path.join(models_folder,best_architecture_string)
    models_architecture_folder = f'{models_architecture_folder}{roc_metric_addition}{use_pesg_auc_addition}{auc_margin_pretrained_addition}'
    if not os.path.exists(models_architecture_folder):
        os.makedirs(models_architecture_folder)
    for index, best_model in enumerate(best_models):
        torch.save(best_model.state_dict(), os.path.join(models_architecture_folder, f'model_{best_architecture_string}_{index}.pt'))
    return models_architecture_folder

def save_test_metric(best_models,folds_traning_dicts,results_folder,use_pesg_auc):
    # Evaluate the best model on the test set
    all_test_outputs = []
    all_test_labels = []
    for index, best_model in enumerate(best_models):
        fold_train_dict = folds_traning_dicts[index]
        X_test = torch.tensor(process_sequences(fold_train_dict['sequences_test']), dtype=torch.float32).transpose(1, 2).to(device)
        y_test = torch.tensor(fold_train_dict['labels_test'], dtype=torch.float32).to(device)
        best_model.eval()
        with torch.no_grad():
            test_outputs = best_model(X_test).squeeze(dim=1).cpu().numpy()
        all_test_outputs.append(test_outputs)
        all_test_labels.append(y_test.cpu().numpy())
    all_test_outputs = np.concatenate(all_test_outputs)
    all_test_labels = np.concatenate(all_test_labels)
    sequences = []
    for fold_dict in folds_traning_dicts:
        sequences.extend(fold_dict['sequences_test'])

    df = pd.DataFrame({
        'sequence': sequences,
        'prediction': all_test_outputs,
        'label': all_test_labels
    })

    csv_path = os.path.join(results_folder, 'predictions_labels_sequences.csv')
    df.to_csv(csv_path, index=False)
    print(f'Saved predictions, labels, and sequences to {csv_path}')
    plot_roc_curve(all_test_labels, all_test_outputs, save_path=os.path.join(results_folder, 'roc_curve_model.png'), title='Convolution Baseline ROC Curve')
    plot_pr_curve(all_test_labels, all_test_outputs, save_path=os.path.join(results_folder, 'pr_curve_model.png'), title='Convolution Baseline Precision-Recall Curve')


def save_grid_search_results(grid_results,results_folder):
    #save the grid search results to a CSV file
    results_df = pd.DataFrame(grid_results)
    #sort the results by pr_auc
    results_df = results_df.sort_values(by='val_metric', ascending=False)
    results_df.to_csv(os.path.join(results_folder, 'grid_search_results.csv'), index=False)

def find_fold_model_in_folder(model_folder_path,fold_index):
    for root, dirs, files in os.walk(model_folder_path):
        for file in files:
            if file.endswith(f'{fold_index}.pt'):
                file_path = os.path.join(root, file)
                return file_path

def train_architecture_over_folds(filter,kernel_size,num_layers,batch_size,padding,dialation,epochs,folds_traning_dicts,ROC_metric,pesg_dict,model_folder_path):
    architecture_metrics_score = []
    architecture_models = []
    pretrained_model_path = None
    for index in range(len(folds_traning_dicts)):
        fold_train_dict = folds_traning_dicts[index]
        if model_folder_path is not None:
            pretrained_model_path = find_fold_model_in_folder(model_folder_path,index)
        val_metric,model= train_architecture_fold(filter,kernel_size,num_layers,batch_size,padding,dialation,epochs,fold_train_dict,None,None,ROC_metric,pesg_dict,pretrained_model_path)
        architecture_metrics_score.append(val_metric)
        architecture_models.append(model)

    return architecture_metrics_score,architecture_models

if __name__ == '__main__':
    DATE = '13_09'
    auc_margin_pretrained = True
    model_name = 'filter_128_kernel_size_7_num_layers_3_batch_size_128_padding_same_dialation_1_roc_metric'
    model_folder_path =  None if not auc_margin_pretrained else os.path.join(paths.convolution_baseline_models_path, DATE,model_name)
    ROC_metric = True
    use_pesg_auc = True
    use_pesg_auc_addition = '_pesg_auc' if use_pesg_auc else ''

    data_for_training_dir = os.path.join(paths.data_for_training_path, DATE)
    results_folder = os.path.join(paths.convolution_baseline_results_path, DATE)
    if not os.path.exists(results_folder):
        os.makedirs(results_folder)
    
    models_folder = os.path.join(paths.convolution_baseline_models_path, DATE)
    if not os.path.exists(models_folder):
        os.makedirs(models_folder)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    folds_traning_dicts = load_as_pickle(os.path.join(data_for_training_dir,'folds_traning_dicts.pkl'))

        # Define the grid search parameters
    # param_grid = {
    #     'dialation' : [1,2,3],
    #     'padding': ['valid','same'],
    #     'filters': [64, 128],
    #     'kernel_size': [5,7],
    #     'num_layers': [2,3],
    #     'batch_size': [128, 256],  
    # }

    param_grid = {
        'dialation' : [1],
        'padding': ['same'],
        'filters': [128],
        'kernel_size': [7],
        'num_layers': [3],
        'batch_size': [128],  
    }

    # if use_pesg_auc:
    #     param_grid.update({
    #         'lr': [0.001, 0.0005, 0.0001],
    #         'weight_decay': [1e-4, 1e-5],
    #         'margin': [0.1, 0.3, 0.5, 0.7, 1.0]
    # })
    if use_pesg_auc:
        param_grid.update({
        'lr': [0.001],
        'weight_decay': [1e-4],
        'margin':[1.0]
        })

    epochs = 200    
    best_model = None
    best_architecture_metric = 0
    best_architecture_index = 0
    grid_results = []
    cnt = 0
    for dialation in param_grid['dialation']:
        for padding in param_grid['padding']:
            for filter in param_grid['filters']:
                for kernel_size in param_grid['kernel_size']:
                    for num_layers in param_grid['num_layers']:
                        for batch_size in param_grid['batch_size']:
                            if padding == 'valid':
                                if not check_if_parameters_can_create_valid_model(kernel_size, num_layers, dialation):
                                    continue
                            if not use_pesg_auc:
                                architecture_metrics_score,architecture_models = \
                                    train_architecture_over_folds(filter,kernel_size,num_layers,batch_size,padding,dialation,epochs,folds_traning_dicts,ROC_metric,None,model_folder_path)
                                architecture_metric = np.mean(architecture_metrics_score)
                                # Save the grid search results
                                grid_results.append({
                                    'dialation': dialation,
                                    'padding': padding,
                                    'filters': filter,
                                    'kernel_size': kernel_size,
                                    'num_layers': num_layers,
                                    'batch_size': batch_size,
                                    'epochs': epochs,
                                    'val_metric': architecture_metric
                                })
                                if architecture_metric > best_architecture_metric:
                                    best_architecture_metric = architecture_metric
                                    best_models = architecture_models
                                    best_architecture_index = cnt
                                cnt += 1
                            else:
                                for lr in param_grid['lr']:
                                    for weight_decay in param_grid['weight_decay']:
                                        for margin in param_grid['margin']:
                                            pesg_dict = {
                                                'lr': lr,
                                                'weight_decay': weight_decay,
                                                'margin': margin
                                            }
                                            architecture_metrics_score,architecture_models = \
                                                train_architecture_over_folds(filter,kernel_size,num_layers,batch_size,padding,dialation,epochs,folds_traning_dicts,ROC_metric,pesg_dict,model_folder_path)
                                            architecture_metric = np.mean(architecture_metrics_score)
                            
                                            grid_results.append({
                                                'dialation': dialation,
                                                'padding': padding,
                                                'filters': filter,
                                                'kernel_size': kernel_size,
                                                'num_layers': num_layers,
                                                'batch_size': batch_size,
                                                'lr': lr,
                                                'weight_decay': weight_decay,
                                                'margin': margin,
                                                'epochs': epochs,
                                                'val_metric': architecture_metric
                                            })
                                            if architecture_metric > best_architecture_metric:
                                                best_architecture_metric = architecture_metric
                                                best_models = architecture_models
                                                best_architecture_index = cnt
                                            cnt += 1
    

    models_architecture_folder = save_best_models(best_models,grid_results,best_architecture_index,models_folder,ROC_metric,use_pesg_auc,auc_margin_pretrained)
    results_architecture_folder = os.path.join(results_folder, models_architecture_folder.split('/')[-1])
    os.makedirs(results_architecture_folder, exist_ok=True)

    save_test_metric(best_models,folds_traning_dicts,results_architecture_folder)

    save_grid_search_results(grid_results,results_architecture_folder)