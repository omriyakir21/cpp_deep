import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..','..'))
import torch
print(torch.cuda.is_available())
print(torch.version.cuda)
from setfit import SetFitModel, SetFitTrainer
from models.baselines.convolution_baseline.convolution_baseline import save_grid_search_results
from utils import load_as_pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import paths
from sentence_transformers.losses import CosineSimilarityLoss
from sklearn.utils.class_weight import compute_class_weight
from models.esm2.ems2_utils import CustomDataset, precision_recall_auc
from utils import plot_pr_curve


def save_best_models(best_models, grid_results, best_architecture_index, models_folder):
    # Save the best models
    best_loss_class = str(grid_results[best_architecture_index]['loss_class'].__name__)
    best_num_epochs = grid_results[best_architecture_index]['num_epochs']
    best_num_iterations = grid_results[best_architecture_index]['num_iterations']
    best_batch_size = grid_results[best_architecture_index]['batch_size']

    print(f"Best model: Loss class: {best_loss_class}, Num epochs: {best_num_epochs}, Num iterations: {best_num_iterations}, Batch size: {best_batch_size}")

    # Create architecture-specific folder
    architecture_model_folder = os.path.join(models_folder, f'architecture_{best_loss_class}_{best_num_epochs}_{best_num_iterations}_{best_batch_size}')
    if not os.path.exists(architecture_model_folder):
        os.makedirs(architecture_model_folder)

    for index, best_model in enumerate(best_models):
        model_path = os.path.join(architecture_model_folder, f'model_{index + 1}')
        best_model.save_pretrained(model_path)
    return architecture_model_folder

def train_architecture_over_folds(loss_class, batch_size, num_iterations, num_epochs, folds_training_dicts, model_name):
    architecture_pr_aucs = []
    best_models = []

    # Early stopping variables
    patience = 5  # Number of epochs to wait after last improvement

    for fold_index, fold_dict in enumerate(folds_training_dicts):
        best_model = None
        best_pr_auc = 0
        epochs_no_improve = 0

        print(f"Training fold {fold_index + 1}/{len(folds_training_dicts)}")
        train_dataset = CustomDataset(fold_dict['sequences_train'], fold_dict['labels_train'], model_name).to_hf_dataset()
        eval_dataset = CustomDataset(fold_dict['sequences_validation'], fold_dict['labels_validation'], model_name).to_hf_dataset()

        model = SetFitModel.from_pretrained(model_name)
        model.to(device)

        trainer = SetFitTrainer(
            model=model,
            train_dataset=train_dataset,
            loss_class=loss_class,
            batch_size=batch_size,
            num_iterations=num_iterations,
            num_epochs=1,  # Set to 1 because we're handling our own training loop
            metric=precision_recall_auc,
            column_mapping={"data": "text", "labels": "label"}
        )

        for epoch in range(num_epochs):
            print(f"Epoch {epoch + 1}/{num_epochs}")
            trainer.train()

            with torch.no_grad():
                y_val_pred = model.predict_proba(eval_dataset['data'])[:, 1]  # Get predictions
                y_val_true = eval_dataset['labels']

                pr_auc = precision_recall_auc(y_val_true, y_val_pred)
                print(f"Validation PR AUC at epoch {epoch + 1}: {pr_auc:.4f}")

                if pr_auc > best_pr_auc:
                    best_pr_auc = pr_auc
                    epochs_no_improve = 0
                    best_model = model
                    print(f"New best PR AUC: {pr_auc:.4f}. Model saved.")
                else:
                    epochs_no_improve += 1
                    print(f"No improvement for {epochs_no_improve} epochs.")

                if epochs_no_improve >= patience:
                    print(f"Early stopping at epoch {epoch + 1} after {patience} epochs with no improvement.")
                    break

        best_models.append(best_model)
        architecture_pr_aucs.append(best_pr_auc)
        print(f"Fold {fold_index + 1} done, best PR AUC: {best_pr_auc:.4f}")

    return architecture_pr_aucs, best_models

def save_test_pr_auc(best_models, folds_training_dicts, results_folder, tokenizer_name, title):
    all_test_outputs = []
    all_test_labels = []
    for index, best_model in enumerate(best_models):
        fold_train_dict = folds_training_dicts[index]
        y_test_true = fold_train_dict['labels_test']
        test_dataset = CustomDataset(fold_train_dict['sequences_test'], y_test_true, tokenizer_name).to_hf_dataset()
        with torch.no_grad():
            y_test_proba = best_model.predict_proba(test_dataset['data'])[:, 1]
            test_pr_auc = precision_recall_auc(y_test_true, y_test_proba)
            print(f"Fold {index + 1} Test PR AUC: {test_pr_auc:.4f}")
            all_test_outputs.append(y_test_proba)
            all_test_labels.append(y_test_true)
    all_test_outputs = np.concatenate(all_test_outputs)
    all_test_labels = np.concatenate(all_test_labels)
    np.save(os.path.join(results_folder, 'all_test_outputs.npy'), all_test_outputs)
    np.save(os.path.join(results_folder, 'all_test_labels.npy'), all_test_labels)
    plot_pr_curve(all_test_labels, all_test_outputs, save_path=os.path.join(results_folder, 'pr_curve_model.png'), title=title)

if __name__ == '__main__':
    DATE = '13_09'
    model_name = "facebook/esm2_t6_8M_UR50D"
    models_folder = os.path.join(paths.few_shot_learning_models_path, DATE, model_name.split('/')[-1])
    if not os.path.exists(models_folder):
        os.makedirs(models_folder)
    results_folder = os.path.join(paths.few_shot_learning_results_path, DATE, model_name.split('/')[-1])
    if not os.path.exists(results_folder):
        os.makedirs(results_folder)

    folds_training_dicts = load_as_pickle(os.path.join(paths.data_for_training_path, DATE, 'folds_traning_dicts.pkl'))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    param_grid = {
        'loss_class': [CosineSimilarityLoss],
        'num_epochs': [10],
        'num_iterations': [5, 10, 20],
        'batch_size': [64, 128, 256],
    }

    best_model = None
    best_architecture_pr_auc = 0
    best_architecture_index = 0
    grid_results = []
    cnt = 0

    for loss_class in param_grid['loss_class']:
        for num_epochs in param_grid['num_epochs']:
            for num_iterations in param_grid['num_iterations']:
                for batch_size in param_grid['batch_size']:
                    architecture_pr_aucs, architecture_models = train_architecture_over_folds(
                        loss_class, batch_size, num_iterations, num_epochs, folds_training_dicts, model_name)
                    architecture_pr_auc = np.mean(architecture_pr_aucs)

                    grid_results.append({
                        'loss_class': loss_class,
                        'num_epochs': num_epochs,
                        'num_iterations': num_iterations,
                        'batch_size': batch_size,
                        'val_pr_auc': architecture_pr_auc
                    })

                    if architecture_pr_auc > best_architecture_pr_auc:
                        best_architecture_pr_auc = architecture_pr_auc
                        best_models = architecture_models
                        best_architecture_index = cnt
                    cnt += 1

    architecture_model_folder = save_best_models(best_models, grid_results, best_architecture_index, models_folder)

    architecture_results_folder = os.path.join(results_folder, architecture_model_folder.split('/')[-1])
    if not os.path.exists(architecture_results_folder):
        os.makedirs(architecture_results_folder)
    save_grid_search_results(grid_results, architecture_results_folder)
    save_test_pr_auc(best_models, folds_training_dicts, architecture_results_folder, model_name, 'Few-Shot Learning Precision-Recall Curve')