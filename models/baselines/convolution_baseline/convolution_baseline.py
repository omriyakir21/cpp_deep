import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..','..','..'))
import paths
from models.baselines.convolution_baseline.convolution_baseline_utils import process_sequences,CNNModel \
    ,score_function,calculate_pr_auc
from utils import load_as_pickle,plot_pr_curve
import numpy as np
import torch
import pandas as pd
from ignite.engine import Engine, Events
from ignite.handlers import EarlyStopping
from utils import plot_pr_curve
from torchsummary import summary
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader, TensorDataset

def save_best_models(best_models,grid_results,best_architecture_index,models_folder):
    # Save the best models
    best_filter = grid_results[best_architecture_index]['filters']
    best_kernel_size = grid_results[best_architecture_index]['kernel_size']
    best_num_layers = grid_results[best_architecture_index]['num_layers']
    best_batch_size = grid_results[best_architecture_index]['batch_size']
    print(f'Best model: Filter: {best_filter}, Kernel size: {best_kernel_size}, Num layers: {best_num_layers}, Batch size: {best_batch_size}, val_pr_auc: {best_architecture_pr_auc}')
    for index, best_model in enumerate(best_models):
        torch.save(best_model.state_dict(), os.path.join(models_folder, f'model_{best_filter}_{best_kernel_size}_{best_num_layers}_{best_batch_size}_{index}.pt'))

def save_test_pr_auc(best_models,folds_traning_dicts,results_folder):
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
    np.save(os.path.join(results_folder, 'all_test_outputs.npy'), all_test_outputs)
    np.save(os.path.join(results_folder, 'all_test_labels.npy'), all_test_labels)
    # Plot the precision-recall curve
    plot_pr_curve(all_test_labels, all_test_outputs, save_path=os.path.join(results_folder, 'pr_curve_model.png'), title='Convolution Baseline Precision-Recall Curve')

def save_grid_search_results(grid_results,results_folder):
    #save the grid search results to a CSV file
    results_df = pd.DataFrame(grid_results)
    #sort the results by pr_auc
    results_df = results_df.sort_values(by='val_pr_auc', ascending=False)
    results_df.to_csv(os.path.join(results_folder, 'grid_search_results.csv'), index=False)

def train_architecture_over_folds(filter,kernel_size,num_layers,batch_size,epochs,folds_traning_dicts):
    architecture_pr_aucs = []
    architecture_models = []
    for index in range(len(folds_traning_dicts)):
        fold_train_dict = folds_traning_dicts[index]
        # Preprocess the sequences
        X_train = torch.tensor(process_sequences(fold_train_dict['sequences_train']), dtype=torch.float32).transpose(1, 2).to(device)
        y_train = torch.tensor(fold_train_dict['labels_train'], dtype=torch.float32).to(device)
        X_val = torch.tensor(process_sequences(fold_train_dict['sequences_validation']), dtype=torch.float32).transpose(1, 2).to(device)
        y_val = torch.tensor(fold_train_dict['labels_validation'], dtype=torch.float32).to(device)

        train_dataset = TensorDataset(X_train, y_train)
        val_dataset = TensorDataset(X_val, y_val)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size,shuffle = True)
        
        class_weights = compute_class_weight('balanced', classes=np.unique(y_train.cpu().numpy()), y=y_train.cpu().numpy())
        class_weights = torch.FloatTensor(class_weights).to(device)

        print(f'Index:{index},Filter: {filter}, Kernel size: {kernel_size}, Num layers: {num_layers}, Batch size: {batch_size}')
        model = CNNModel(filters=filter, kernel_size=kernel_size, num_layers=num_layers).to(device)
        summary(model, (20, 50))                       
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        def train_step(engine, batch):
            # print('train_step')
            model.train()
            optimizer.zero_grad()
            X_batch, y_batch = batch
            outputs = model(X_batch).squeeze(dim=1)
            batch_weights = class_weights[y_batch.long()]
            criterion = nn.BCELoss(weight=batch_weights)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            return loss.item()

        def validation_step(engine, batch):
            print('validation_step')
            model.eval()
            with torch.no_grad():
                X_batch, y_batch = batch
                y_pred = model(X_batch).squeeze(dim=1)
                return y_batch, y_pred

        trainer = Engine(train_step)
        evaluator = Engine(validation_step)

        @trainer.on(Events.EPOCH_COMPLETED)
        def calculate_pr_auc_engine(engine):
            model.eval()
            print(f'Epoch {engine.state.epoch} - loss: {engine.state.output:.2f}')
            evaluator.run(val_loader)

        handler = EarlyStopping(patience=5, score_function=score_function, trainer = trainer)
        evaluator.add_event_handler(Events.EPOCH_COMPLETED, handler)
        trainer.run(train_loader, max_epochs=epochs)
                        
        # Calculate prAUC
        model.eval()
        evaluator.run([(X_val, y_val)])
        y_true, y_pred = evaluator.state.output
        val_pr_auc = calculate_pr_auc(y_true.cpu().numpy(), y_pred.cpu().numpy())
        architecture_pr_aucs.append(val_pr_auc)
        architecture_models.append(model)

    return architecture_pr_aucs,architecture_models

if __name__ == '__main__':
    DATE = '13_09'
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
    param_grid = {
        'filters': [16,32, 64, 128],
        'kernel_size': [2,3,5],
        'num_layers': [1, 2],
        'batch_size': [128, 256,512],  
    }
    epochs = 200    
    best_model = None
    best_architecture_pr_auc = 0
    best_architecture_index = 0
    grid_results = []
    cnt = 0

    for filter in param_grid['filters']:
        for kernel_size in param_grid['kernel_size']:
            for num_layers in param_grid['num_layers']:
                for batch_size in param_grid['batch_size']:
                    architecture_pr_aucs,architecture_models = train_architecture_over_folds(filter,kernel_size,num_layers,batch_size,epochs,folds_traning_dicts)
                    architecture_pr_auc = np.mean(architecture_pr_aucs)
                    # Save the grid search results
                    grid_results.append({
                        'filters': filter,
                        'kernel_size': kernel_size,
                        'num_layers': num_layers,
                        'batch_size': batch_size,
                        'epochs': epochs,
                        'val_pr_auc': architecture_pr_auc
                    })

                    # Save the best models
                    if architecture_pr_auc > best_architecture_pr_auc:
                        best_architecture_pr_auc = architecture_pr_auc
                        best_models = architecture_models
                        best_architecture_index = cnt
                    cnt += 1
    

    save_best_models(best_models,grid_results,best_architecture_index,models_folder)

    save_test_pr_auc(best_models,folds_traning_dicts,results_folder)

    save_grid_search_results(grid_results,results_folder)