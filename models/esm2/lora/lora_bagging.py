import os 
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..','..','..'))
import paths
import itertools
import torch
from transformers import EsmTokenizer
from peft import LoraConfig, get_peft_model,PeftModel,TaskType
from utils import load_as_pickle, plot_pr_curve, plot_roc_curve,save_as_pickle
from sklearn.utils.class_weight import compute_class_weight
from models.baselines.convolution_baseline.convolution_baseline import save_grid_search_results
import numpy as np
import re
from transformers.trainer_callback import EarlyStoppingCallback
import pandas as pd
from matplotlib import pyplot as plt
from models.esm2.ems2_utils import precision_recall_auc,metrics_evaluation,WeightedTrainer
from datasets import Dataset
from sklearn.metrics import precision_recall_curve, auc
from models.baselines.convolution_pu_learning_bagging.convolution_baseline_pu_learning_bagging import sample_training_sequences,calculate_pr_auc,calculate_roc_auc
from models.esm2.lora.esm2_fine_tune_lora import train_architecture_fold,predict_binary_probs,save_test_metric,save_best_models
# Custom WeightedTrainer


# Define early stopping callback
early_stopping_callback = EarlyStoppingCallback(
    early_stopping_patience=8,  # Number of epochs to wait for improvement
    early_stopping_threshold=0.0,  # Minimum change to consider as improvement
)

def save_models_of_range(models, architecture_grid_info, models_folder,roc_metric,indexes,fold_index):
    roc_metric_addition = "_roc_metric" if roc_metric else ""
    architecture_key_val_string = '_'.join([f'{key}_{value}' for key, value in architecture_grid_info.items()])
    architecture_val_string = '_'.join([str(val) for val in architecture_grid_info.values()])
    architecture_model_folder = os.path.join(models_folder, f'architecture_{architecture_key_val_string}{roc_metric_addition}')
    os.makedirs(architecture_model_folder, exist_ok=True)

    fold_folder = os.path.join(architecture_model_folder,f'fold_models_{fold_index}')
    os.makedirs(fold_folder, exist_ok=True)

    for i in range(len(models)):
        models[i].save_pretrained(os.path.join(fold_folder, f'model_{architecture_val_string}_{i+1+indexes[0]}'))
    return architecture_model_folder

def train_and_save_range(architecture_grid_info,roc_metric,indexes,models_folder,folds_training_dicts,model_name,fine_tune_untrained,tokenizer,subsampled_training_sequences_folds,subsampled_labels_folds,checkpoint_folder):
    models = []
    for fold_index in range(len(folds_training_dicts)):
        for j in range(indexes[0],indexes[1]):
            fold_best_models = train_architecture_fold(architecture_grid_info['batch_size'],architecture_grid_info['num_epochs'],architecture_grid_info['r'],architecture_grid_info['lora_alpha'], folds_training_dicts[j],
                                                    model_name,fine_tune_untrained,tokenizer,subsampled_training_sequences_folds[fold_index][j],subsampled_labels_folds[fold_index][j],checkpoint_folder)
            models.append(fold_best_models)
        architecture_model_folder = save_models_of_range(models, architecture_grid_info, models_folder,roc_metric,indexes,fold_index)
    return architecture_model_folder

def predict_with_ensemble(models_folder,fold_index,base_model,tokenizer,sequences):
    fold_folder = os.path.join(models_folder,f'fold_models_{fold_index}')
    fold_models = []
    for model_folder in os.listdir(fold_folder):
        model = PeftModel.from_pretrained(os.path.join(fold_folder,model_folder))
        fold_models.append(model)
    predictions = np.zeros(len(sequences))
    for model in fold_models:
        X_val = tokenizer(sequences, truncation=True, padding="max_length", max_length=50, return_tensors="pt")
        predictions += predict_binary_probs(model, X_val).flatten()
    predictions /= len(fold_models)
    return predictions



if __name__ == "__main__":
    DATE = '13_09'
    roc_metric = True
    fine_tune_untrained = False
    fine_tune_untrained_addition = "_untrained" if fine_tune_untrained else ""
    my_pretrained = False
    from_scratch = False
    date_for_pretrained = '2025-02-14' if from_scratch else '2025-02-20'
    from_scratch_addition = "_from_scratch" if from_scratch else ""
    folds_training_dicts = load_as_pickle(os.path.join(paths.data_for_training_path, DATE, 'folds_traning_dicts.pkl'))
    esm2_model_name = 'facebook/esm2_t6_8M_UR50D' 
    model_path = os.path.join(paths.esm2_peptide_pretrained_models_path,f'peptide_{esm2_model_name.split("/")[-1]}{from_scratch_addition}_{date_for_pretrained}')
    model_name = model_path if my_pretrained else esm2_model_name
    indexes = (0,2)
    param_grid = {
        'iteration': [100],
        'negative_size': [0.3],   
        'batch_size': [256],
        'num_epochs': [50],
        'r': [100],
        'lora_alpha': [32], 
    }
    architecture_grid_info_range = {
        'pu_iterations': 100,
        'negative_size': 0.3,
        'batch_size': 256,
        'num_epochs':50,
        'r': 100,
        'lora_alpha': 32}
    
    tokenizer = EsmTokenizer.from_pretrained(model_name)
    best_model = None
    best_architecture_pr_auc = 0
    best_architecture_index = 0
    grid_results = []
    cnt = 0
    print(f"Training model: {model_name}")
    print("=========================================")
    print("Grid search parameters:")
    print(param_grid)
        
    models_folder = os.path.join(paths.lora_bagging_models_path, DATE, model_name.split('/')[-1])
    os.makedirs(models_folder, exist_ok=True)
    save_dir = os.path.join(models_folder, 'checkpoints')
    checkpoint_folder = os.path.join(models_folder,'checkpoints')
    os.makedirs(checkpoint_folder, exist_ok=True)
    results_folder = os.path.join(paths.lora_bagging_results_path, DATE, model_name.split('/')[-1])
    os.makedirs(results_folder, exist_ok=True)
    for iteration,negative_size in itertools.product(param_grid['iteration'],param_grid['negative_size']):
        subsampled_training_sequences_folds_path = os.path.join(models_folder,f'sampled_training_sequences_{iteration}_{negative_size}.pkl')
        subsampled_labels_folds_path = os.path.join(models_folder,f'sampled_labels_{iteration}_{negative_size}.pkl')
        if os.path.exists(subsampled_training_sequences_folds_path) and os.path.exists(subsampled_labels_folds_path):
            subsampled_training_sequences_folds = load_as_pickle(subsampled_training_sequences_folds_path)
            subsampled_labels_folds = load_as_pickle(subsampled_labels_folds_path)
        else:
            subsampled_training_sequences_folds,subsampled_labels_folds = sample_training_sequences(iteration,negative_size,folds_training_dicts)
            save_as_pickle(subsampled_training_sequences_folds,os.path.join(models_folder,f'subsampled_training_sequences_folds_{iteration}_{negative_size}.pkl'))
            save_as_pickle(subsampled_labels_folds,os.path.join(models_folder,f'subsampled_labels_folds_{iteration}_{negative_size}.pkl'))
        if indexes is not None:
            print(f"Training with iteration: {iteration}, negative_size: {negative_size}, indexes: {indexes}")

            
            train_and_save_range(architecture_grid_info_range,roc_metric,indexes,models_folder,folds_training_dicts,model_name,fine_tune_untrained,tokenizer,subsampled_training_sequences_folds,subsampled_labels_folds,checkpoint_folder)

        else:
            for batch_size,num_epochs,r,lora_alpha in itertools.product(
                param_grid['batch_size'],param_grid['num_epochs'],param_grid['r'],param_grid['lora_alpha']):
                print(f"Training with batch size: {batch_size}, num_epochs: {num_epochs}, r: {r}, lora_alpha: {lora_alpha}")
                architecture_models = []
                all_predictions = []
                all_labels = []
                for i in range(len(subsampled_training_sequences_folds)):
                    fold_dict = folds_training_dicts[i]
                    architecture_fold_models = []
                    X_val = tokenizer(fold_dict['sequences_validation'], truncation=True, padding="max_length", max_length=50, return_tensors="pt")
                    y_val = folds_training_dicts[i]['labels_validation']
                    all_labels.append(y_val)
                    sub_subgroups_predictions = np.zeros(len(y_val))
                    for j in range(len(subsampled_training_sequences_folds[i])):
                        architecture_subgroup_model = train_architecture_fold(batch_size, num_epochs,r,lora_alpha, fold_dict,model_name,fine_tune_untrained,tokenizer,subsampled_training_sequences_folds[i][j],subsampled_labels_folds[i][j],checkpoint_folder)
                        architecture_fold_models.append(architecture_subgroup_model)
                        architecture_subgroup_predictions = predict_binary_probs(architecture_subgroup_model, X_val)
                        architecture_subgroup_predictions = architecture_subgroup_predictions.flatten()
                        sub_subgroups_predictions += architecture_subgroup_predictions
                    sub_subgroups_predictions /= len(subsampled_training_sequences_folds[i])
                    architecture_models.append(architecture_fold_models)
                    all_predictions.append(sub_subgroups_predictions)
                all_predictions = np.concatenate(all_predictions)
                all_labels = np.concatenate(all_labels)
                if roc_metric:
                    architecture_metric_score = float(calculate_roc_auc(all_labels,all_predictions))
                else:
                    architecture_metric_score = float(calculate_pr_auc(all_labels,all_predictions))
                architecture_grid_info = {
                    'pu_iterations': iteration,
                    'negative_size': negative_size,
                    'batch_size': batch_size,
                    'num_epochs': num_epochs,
                    'r': r,
                    'lora_alpha': lora_alpha,
                    'val_metric': architecture_metric_score
                }
                grid_results.append(architecture_grid_info)
                print(f"architecture results: {grid_results[-1]}")
                if architecture_metric_score > best_architecture_pr_auc:
                    best_architecture_pr_auc = architecture_metric_score
                    best_models = architecture_models
                    best_architecture_grid_info = architecture_grid_info
                    best_architecture_index = cnt
                cnt += 1
    
    if indexes is None:
        architecture_model_folder = save_best_models(best_models, best_architecture_grid_info, models_folder,roc_metric,True)
        architecture_results_folder = os.path.join(results_folder, architecture_model_folder.split('/')[-1])
        os.makedirs(architecture_results_folder, exist_ok=True)
        save_grid_search_results(grid_results, architecture_results_folder)
        save_test_metric(best_models,architecture_results_folder,'esm2 fine tune LoRA bagging',folds_training_dicts,tokenizer,roc_metric,True)