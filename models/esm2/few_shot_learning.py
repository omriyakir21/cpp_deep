import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
import torch
from setfit import SetFitModel, SetFitTrainer
from sklearn.metrics import precision_recall_curve, auc
from models.baselines.convolution_baseline.convolution_baseline import save_grid_search_results
from utils import load_as_pickle,plot_pr_curve
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import paths
from sentence_transformers.losses import CosineSimilarityLoss
from sklearn.utils.class_weight import compute_class_weight
from transformers import AutoTokenizer
from torch.utils.data import Dataset
from datasets import Dataset as HFDataset


def precision_recall_auc(y_true, y_pred):
    precision, recall, _ = precision_recall_curve(y_true, y_pred)
    return auc(recall, precision)

from sentence_transformers import SentenceTransformer


class CustomDataset(Dataset):
    def __init__(self, data, labels, tokenizer_name):
        if isinstance(data, list):
            data = pd.DataFrame(data, columns=['data'])
        if isinstance(labels, list):
            labels = pd.DataFrame(labels, columns=['labels'])
        
        self.data = data
        self.labels = labels
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data.iloc[idx]['data']  # Get the actual data
        label = int(self.labels.iloc[idx]['labels'])  # Convert label to Python int
        encoded_sample = self.tokenizer(sample, padding='max_length', truncation=True, return_tensors='pt')
        return encoded_sample, label

    def to_hf_dataset(self):
        # Convert the dataset to HuggingFace Dataset for compatibility
        data_dict = {
            'data': self.data['data'].tolist(),
            'labels': [int(label) for label in self.labels['labels'].tolist()]  # Convert labels to Python int
        }
        return HFDataset.from_dict(data_dict)


def save_best_models(best_models,grid_results,best_architecture_index,models_folder):
    # Save the best models
    best_loss_class = str(grid_results[best_architecture_index]['loss_class'])
    best_num_epochs = grid_results[best_architecture_index]['num_epochs']
    best_num_iterations = grid_results[best_architecture_index]['num_iterations']
    best_batch_size = grid_results[best_architecture_index]['batch_size']
    
    print(f'Best model: Loss class: {best_loss_class}, Num epochs: {best_num_epochs}, Num iterations: {best_num_iterations}, Batch size: {best_batch_size}, val_pr_auc: {best_architecture_pr_auc}')
    for index, best_model in enumerate(best_models):
        best_model.save_pretrained(os.path.join(models_folder, f'model_{best_loss_class}_{best_num_epochs}_{best_num_iterations}_{best_batch_size}_{index}'))
        
def save_test_pr_auc(best_models,folds_traning_dicts,results_folder,tokenizer_name):
    # Evaluate the best model on the test set
    all_test_outputs = []
    all_test_labels = []
    for index, best_model in enumerate(best_models):
        fold_train_dict = folds_traning_dicts[index]
        y_test_true = fold_train_dict['labels_test']
        test_dataset = CustomDataset(fold_train_dict['sequences_test'], y_test_true, tokenizer_name).to_hf_dataset()
        best_model.eval()
        with torch.no_grad():
            y_test_proba = best_model.predict_proba(test_dataset['data'])[:, 1]
            test_pr_auc = precision_recall_auc(y_test_true, y_test_proba)
            print(f"Fold {index + 1} Test PR AUC: {test_pr_auc:.4f}")
            all_test_outputs.append(y_test_proba)
            all_test_labels.append(y_test_true)
    all_test_outputs = np.concatenate(all_test_outputs)
    all_test_labels = np.concatenate(all_test_labels)
    # Plot the precision-recall curve
    plot_pr_curve(all_test_labels, all_test_outputs, save_path=os.path.join(results_folder, 'pr_curve_model.png'), title='esm2 few shot learning Precision-Recall Curve')

# def train_architecture_over_folds(loss_class,batch_size,num_iterations,num_epochs,folds_training_dicts):
    
#     architecture_pr_aucs = []
#     best_models = []

#     # Early stopping variables
#     patience = 5  # Number of epochs to wait after last improvement
#     # Iterate through each fold in the training data
#     for fold_index, fold_dict in enumerate(folds_training_dicts):
        
#         class_weights = compute_class_weight('balanced', classes=np.unique(fold_dict['labels_train']), y=fold_dict['labels_train'])
#         def weighted_loss_fn(labels, embeddings):
#             # Apply CosineSimilarityLoss and then weight the loss by class
#             cosine_loss = CosineSimilarityLoss(model)
#             loss = cosine_loss(embeddings, labels)
#             weights = class_weights[labels.long()]  # Get the corresponding class weights
#             weighted_loss = loss * weights
#             return torch.mean(weighted_loss)
        
#         best_model = None
#         best_pr_auc = 0
#         epochs_no_improve = 0
        
#         print(f"Training fold {fold_index + 1}/{len(folds_training_dicts)}")
#         tokenizer_name = "facebook/esm2_t6_8M_UR50D"
#         train_dataset = CustomDataset(fold_dict['sequences_train'], fold_dict['labels_train'], tokenizer_name).to_hf_dataset()
#         eval_dataset = CustomDataset(fold_dict['sequences_validation'], fold_dict['labels_validation'], tokenizer_name).to_hf_dataset()

#         model = SetFitModel.from_pretrained("facebook/esm2_t6_8M_UR50D")
#         model.to(device)

#         trainer = SetFitTrainer(
#             model=model,
#             train_dataset=train_dataset,
#             loss_class= weighted_loss_fn,
#             batch_size= batch_size,
#             num_iterations=num_iterations,
#             num_epochs = num_epochs,  # Set to 1 because we're handling our own training loop
#             metric=precision_recall_auc,
#             column_mapping={"data": "text", "labels": "label"}
#         )

#         # Custom training loop with early stopping
#         for epoch in range(num_epochs):
#             print(f"Epoch {epoch + 1}/{num_epochs}")
#             # Train the model for one epoch
#             trainer.train()

#             model.eval()
#             with torch.no_grad():
#                 # Evaluate on validation set
#                 y_val_pred = model.predict_proba(eval_dataset['data'])[:, 1]  # Get predictions
#                 y_val_true = eval_dataset['labels']  # Ground truth labels

#                 # Calculate Precision-Recall AUC
#                 pr_auc = precision_recall_auc(y_val_true, y_val_pred)
#                 print(f"Validation PR AUC at epoch {epoch + 1}: {pr_auc:.4f}")

#                 # Check if the model has improved
#                 if pr_auc > best_pr_auc:
#                     best_pr_auc = pr_auc
#                     epochs_no_improve = 0  # Reset counter
#                     # Save the model if it improves
#                     best_model = model
#                     print(f"New best PR AUC: {pr_auc:.4f}. Model saved.")
#                 else:
#                     epochs_no_improve += 1
#                     print(f"No improvement for {epochs_no_improve} epochs.")

#                 # Early stopping condition
#                 if epochs_no_improve >= patience:
#                     print(f"Early stopping at epoch {epoch + 1} after {patience} epochs with no improvement.")
#                     break
#         best_models.append(best_model)
#         architecture_pr_aucs.append(best_pr_auc)
    
#     return architecture_pr_aucs,best_models

import numpy as np
from sklearn.utils.class_weight import compute_class_weight
import torch

def train_architecture_over_folds(loss_class, batch_size, num_iterations, num_epochs, folds_training_dicts):
    
    architecture_pr_aucs = []
    best_models = []

    # Early stopping variables
    patience = 5  # Number of epochs to wait after last improvement

    # Iterate through each fold in the training data
    for fold_index, fold_dict in enumerate(folds_training_dicts):
        # Compute class weights
        class_weights = compute_class_weight('balanced', 
                                             classes=np.unique(fold_dict['labels_train']), 
                                             y=fold_dict['labels_train'])
        class_weights = torch.tensor(class_weights, dtype=torch.float32, device=device)
        
        best_model = None
        best_pr_auc = 0
        epochs_no_improve = 0
        
        print(f"Training fold {fold_index + 1}/{len(folds_training_dicts)}")
        tokenizer_name = "facebook/esm2_t6_8M_UR50D"
        train_dataset = CustomDataset(fold_dict['sequences_train'], fold_dict['labels_train'], tokenizer_name).to_hf_dataset()
        eval_dataset = CustomDataset(fold_dict['sequences_validation'], fold_dict['labels_validation'], tokenizer_name).to_hf_dataset()

        model = SetFitModel.from_pretrained("facebook/esm2_t6_8M_UR50D")
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
        

        # Custom training loop with early stopping
        for epoch in range(num_epochs):
            print(f"Epoch {epoch + 1}/{num_epochs}")
            # Train the model for one epoch
            trainer.train()


            with torch.no_grad():
                # Evaluate on validation set
                y_val_pred = model.predict_proba(eval_dataset['data'])[:, 1]  # Get predictions
                y_val_true = eval_dataset['labels']  # Ground truth labels

                # Calculate Precision-Recall AUC
                pr_auc = precision_recall_auc(y_val_true, y_val_pred)
                print(f"Validation PR AUC at epoch {epoch + 1}: {pr_auc:.4f}")

                # Check if the model has improved
                if pr_auc > best_pr_auc:
                    best_pr_auc = pr_auc
                    epochs_no_improve = 0  # Reset counter
                    # Save the model if it improves
                    best_model = model
                    print(f"New best PR AUC: {pr_auc:.4f}. Model saved.")
                else:
                    epochs_no_improve += 1
                    print(f"No improvement for {epochs_no_improve} epochs.")

                # Early stopping condition
                if epochs_no_improve >= patience:
                    print(f"Early stopping at epoch {epoch + 1} after {patience} epochs with no improvement.")
                    break
        best_models.append(best_model)
        architecture_pr_aucs.append(best_pr_auc)
    
    return architecture_pr_aucs, best_models


if __name__ == '__main__':
    DATE = '13_09'
    models_folder = os.path.join(paths.esm2_models_path, DATE)
    if not os.path.exists(models_folder):
        os.makedirs(models_folder)
    results_folder = os.path.join('results', f'esm2_{DATE}')
    if not os.path.exists(results_folder):
        os.makedirs(results_folder)

    folds_traning_dicts = load_as_pickle(os.path.join(paths.data_for_training_path, DATE, 'folds_traning_dicts.pkl'))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Define the grid search parameters
    param_grid = {
        'loss_class': [CosineSimilarityLoss],
        'num_epochs': [10],
        'num_iterations': [10,20,30],
        'batch_size': [64,128,256],  
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
                    architecture_pr_aucs,architecture_models = train_architecture_over_folds(loss_class,batch_size,num_iterations,num_epochs,folds_traning_dicts)
                    architecture_pr_auc = np.mean(architecture_pr_aucs)
                    # Save the grid search results
                    grid_results.append({
                        'loss_class': loss_class,
                        'num_epochs': num_epochs,
                        'num_iterations': num_iterations,
                        'batch_size': batch_size,
                        'val_pr_auc': architecture_pr_auc
                    })

                    # Save the best models
                    if architecture_pr_auc > best_architecture_pr_auc:
                        best_architecture_pr_auc = architecture_pr_auc
                        best_models = architecture_models
                        best_architecture_index = cnt
                    cnt += 1


    save_best_models(best_models,grid_results,best_architecture_index,models_folder)

    tokenizer_name = "facebook/esm2_t6_8M_UR50D"
    save_test_pr_auc(best_models,folds_traning_dicts,results_folder,tokenizer_name)

    save_grid_search_results(grid_results,results_folder)
