import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
import torch
from setfit import SetFitModel, SetFitTrainer
from sklearn.metrics import precision_recall_curve, auc
from models.baselines.convolution_baseline.convolution_baseline_utils import process_sequences
from utils import load_as_pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import paths
from transformers import AutoModel
import transformers.training_args.TrainingArguments

if __name__ == '__main__':
    DATE = '13_09'
    model_folder = os.path.join(paths.esm2_models_path, DATE)
    if not os.path.exists(model_folder):
        os.makedirs(model_folder)
    results_folder = os.path.join('results', f'esm2_{DATE}')
    if not os.path.exists(results_folder):
        os.makedirs(results_folder)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    esm2_model = AutoModel.from_pretrained("facebook/esm2_t6_8M_UR50D")

    model = SetFitModel(model_body=esm2_model)
    # Freeze the body of the ESM2 model
    for param in model.model_body.parameters():
        param.requires_grad = False
        
    
    fold_train_dicts = load_as_pickle(os.path.join(paths.data_for_training_path, DATE, 'folds_traning_dicts.pkl'))[:1]
    fold_pr_aucs = []
    fold_evaluation_results = []
    
    # Variables to accumulate test labels and predictions across all folds
    all_y_test_true = []
    all_y_test_pred = []
    
    # Early stopping variables
    patience = 5  
    best_pr_auc = -float('inf')
    epochs_no_improve = 0
    num_epochs = 200  # Set max number of epochs
    
    # Iterate through each fold in the training data
    for fold_index, fold_dict in enumerate(fold_train_dicts):
        print(f"Training fold {fold_index + 1}/{len(fold_train_dicts)}")
    
        X_train = torch.tensor(process_sequences(fold_dict['sequences_train']), dtype=torch.float32).transpose(1, 2).to(device)
        y_train = torch.tensor(fold_dict['labels_train'], dtype=torch.float32).to(device)
        X_val = torch.tensor(process_sequences(fold_dict['sequences_validation']), dtype=torch.float32).transpose(1, 2).to(device)
        y_val = torch.tensor(fold_dict['labels_validation'], dtype=torch.float32).to(device)
    
        train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
        val_dataset = torch.utils.data.TensorDataset(X_val, y_val)

        # Set up the trainer with the validation dataset for early stopping
        complete_trainer = SetFitTrainer(
            model=model,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,  
            loss_class="bce",
            metric="auc",  
            num_epochs=num_epochs
        )
    
        for epoch in range(num_epochs):
            print(f"Epoch {epoch+1}/{num_epochs}")
            complete_trainer.train()
    
            # Evaluate on the validation set
            model.eval()
            with torch.no_grad():
                y_val_pred = model(X_val).cpu().numpy()
                y_val_true = y_val.cpu().numpy()
            precision, recall, _ = precision_recall_curve(y_val_true, y_val_pred)
            val_pr_auc = auc(recall, precision)
    
            print(f"Validation PR AUC: {val_pr_auc:.4f}")
    
            if val_pr_auc > best_pr_auc:
                best_pr_auc = val_pr_auc
                epochs_no_improve = 0  # Reset patience counter if there's improvement
            else:
                epochs_no_improve += 1  # No improvement in this epoch
    
            # Early stopping condition
            if epochs_no_improve >= patience:
                print(f"Early stopping triggered at epoch {epoch+1}")
                break
    
        # After early stopping or completing all epochs, evaluate on test set
        X_test = torch.tensor(process_sequences(fold_dict['sequences_test']), dtype=torch.float32).transpose(1, 2).to(device)
        y_test = torch.tensor(fold_dict['labels_test'], dtype=torch.float32).to(device)
        model.eval()
        with torch.no_grad():
            y_test_pred = model(X_test).cpu().numpy()  # Get model predictions on test set
            y_test_true = y_test.cpu().numpy()  # Ground truth labels for test set
    
        # Accumulate test labels and predictions across all folds
        all_y_test_true.append(y_test_true)
        all_y_test_pred.append(y_test_pred)
    
        # Calculate precision-recall curve (for test set)
        precision, recall, _ = precision_recall_curve(y_test_true, y_test_pred)
        pr_auc = auc(recall, precision)  # Calculate PR AUC for test set
        fold_pr_aucs.append(pr_auc)
    
        print(f"Test PR AUC for fold {fold_index + 1}: {pr_auc}")
    
        # Save model for the current fold
        model_save_path = os.path.join(model_folder, f'model_fold_{fold_index + 1}.pt')
        torch.save(model.state_dict(), model_save_path)
    
        fold_evaluation_results.append({
            'fold': fold_index + 1,
            'pr_auc': pr_auc
        })
    
    # Combine the true labels and predictions across all folds
    all_y_test_true = np.concatenate(all_y_test_true)
    all_y_test_pred = np.concatenate(all_y_test_pred)
    
    # Calculate combined precision-recall curve
    precision, recall, _ = precision_recall_curve(all_y_test_true, all_y_test_pred)
    combined_pr_auc = auc(recall, precision)
    
    # Plot and save combined PR curve for all folds
    plt.figure()
    plt.plot(recall, precision, label=f"Combined PR AUC: {combined_pr_auc:.2f}")
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Combined Precision-Recall Curve Across All Folds')
    plt.legend()
    combined_pr_curve_path = os.path.join(results_folder, 'combined_pr_curve.png')
    plt.savefig(combined_pr_curve_path)
    plt.close()
    
    print(f"Combined Test PR AUC across all folds: {combined_pr_auc}")
    
    # Calculate mean PR AUC across all folds
    mean_pr_auc = np.mean(fold_pr_aucs)
    print(f"Mean PR AUC across all folds: {mean_pr_auc}")
    
    # Save the fold evaluation results to a CSV file
    results_df = pd.DataFrame(fold_evaluation_results)
    results_csv_path = os.path.join(results_folder, 'fold_evaluation_results.csv')
    results_df.to_csv(results_csv_path, index=False)
