import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..','..','..'))
import torch
import numpy as np
import torch.nn.functional as F
from transformers import AutoTokenizer, EsmForSequenceClassification, TrainingArguments, Trainer,EsmTokenizer,AutoConfig,AutoModelForMaskedLM
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import precision_recall_curve, auc
from datasets import Dataset
from models.baselines.convolution_baseline.convolution_baseline import save_grid_search_results
from utils import load_as_pickle, plot_pr_curve,plot_roc_curve
from models.esm2.ems2_utils import precision_recall_auc,metrics_evaluation,WeightedTrainer
import paths
from transformers.trainer_callback import EarlyStoppingCallback
import pandas as pd


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define early stopping callback
early_stopping_callback = EarlyStoppingCallback(
    early_stopping_patience=8,  # Number of epochs to wait for improvement
    early_stopping_threshold=0.0,  # Minimum change to consider as improvement
)

def create_dataset(sequences, labels,tokenizer, max_length=50):   
    # Tokenize the sequences
    tokenized = tokenizer(sequences, truncation=True, padding=True, max_length=max_length)
    tokenized['input_ids'] = torch.tensor(tokenized['input_ids'])
    tokenized['attention_mask'] = torch.tensor(tokenized['attention_mask'])
    
    # Add labels to the tokenized dictionary
    tokenized['labels'] = torch.tensor(labels)

    # Create a Hugging Face Dataset from the dictionary
    dataset = Dataset.from_dict(tokenized)

    # Set the format to PyTorch to make it compatible with DataLoader
    dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])

    return dataset
    
def save_best_models(trainers, grid_results, best_architecture_index, models_folder,roc_metric):
    roc_metric_addition = "_roc_metric" if roc_metric else ""
    # Save the best models
    best_num_epochs = grid_results[best_architecture_index]['num_epochs']
    best_batch_size = grid_results[best_architecture_index]['batch_size']
    best_architecture_pr_auc = grid_results[best_architecture_index]['val_metric']

    print(f'Best model: Num epochs: {best_num_epochs}, '
          f'Batch size: {best_batch_size}, val_metric: {best_architecture_pr_auc}')
    
    architecture_model_folder = os.path.join(models_folder, f'architecture_{best_num_epochs}_{best_batch_size}{roc_metric_addition}')
    if not os.path.exists(architecture_model_folder):
        os.makedirs(architecture_model_folder)
    for i in range(len(trainers)):
        trainer = trainers[i]
        trainer.save_model(os.path.join(architecture_model_folder, 
                                                f'model_{best_num_epochs}_{best_batch_size}_{i+1}'))
    return architecture_model_folder

def train_architecture_over_folds(batch_size, num_epochs, folds_training_dicts,model_name,fine_tune_untrained):
    architecture_pr_aucs = []
    models = []
    trainers = []
    test_prediction_logits_outputs = []

    # Iterate through each fold in the training data
    for fold_index, fold_dict in enumerate(folds_training_dicts):
        labels_train = np.array(fold_dict['labels_train'])
        # class_weights = calculate_class_weights(labels_train)
        class_weights = WeightedTrainer.calculate_class_weights(torch.tensor(labels_train))

        tokenizer = EsmTokenizer.from_pretrained(model_name)
        if fine_tune_untrained:
            config = AutoConfig.from_pretrained(esm2_model_name)
            config.num_labels = 1
            model = EsmForSequenceClassification(config)
        else:
            model = EsmForSequenceClassification.from_pretrained(model_name, num_labels=1)
        model.to(device)



        print(f"Training fold {fold_index + 1}/{len(folds_training_dicts)}")
        train_dataset = create_dataset(fold_dict['sequences_train'], fold_dict['labels_train'],tokenizer)
        eval_dataset = create_dataset(fold_dict['sequences_validation'],fold_dict['labels_validation'],tokenizer)
        test_dataset = create_dataset(fold_dict['sequences_test'], fold_dict['labels_test'],tokenizer)

        # Define TrainingArguments for this fold
        training_args = TrainingArguments(
            output_dir=checkpoint_folder,  # Directory to save model checkpoints
            learning_rate=1e-3,
            per_device_train_batch_size=batch_size,
            num_train_epochs=num_epochs,
            weight_decay=0.01,
            per_device_eval_batch_size  =  len(fold_dict['sequences_validation']),
            per_gpu_eval_batch_size = len(fold_dict['sequences_validation']),
            greater_is_better=True,
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="roc_auc",
            logging_dir=os.path.join(checkpoint_folder, 'logs'),
            logging_steps=10,
        )

        # Create a WeightedTrainer
        trainer = WeightedTrainer(
            class_weights=class_weights,  # Pass the class weights to the trainer
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            callbacks=[early_stopping_callback],
            compute_metrics=metrics_evaluation,
        )

        trainer.train()
        models.append(trainer.model)
        architecture_pr_aucs.append(trainer.state.best_metric)
        print(f"Fold {fold_index + 1} done, best PR AUC: {trainer.state.best_metric:.4f}")
        trainers.append(trainer)
        predictions_output_logits = trainer.predict(test_dataset)
        test_prediction_logits_outputs.append(predictions_output_logits)

    return architecture_pr_aucs, models,test_prediction_logits_outputs,trainers

def save_test_metric(architecture_test_logits_prediction_outputs, results_folder, title,folds_training_dicts):
    all_test_outputs = []
    all_test_labels = []
    for i in range(len(architecture_test_logits_prediction_outputs)):
        logits = architecture_test_logits_prediction_outputs[i].predictions.squeeze()
        predictions = 1 / (1 + np.exp(-logits))
        print(f' i = {i} predictions = {predictions}')
        labels = architecture_test_logits_prediction_outputs[i].label_ids
        print(f' i = {i} labels = {labels}')
        all_test_outputs.append(predictions)
        all_test_labels.append(labels)
        test_pr_auc = precision_recall_auc(labels, predictions)
        print(f"Fold {i + 1} Test PR AUC: {test_pr_auc:.4f}")

    all_test_outputs = np.concatenate(all_test_outputs)
    all_test_labels = np.concatenate(all_test_labels)
    print(f'all_test_outputs = {all_test_outputs}')
    print(f'all_test_labels = {all_test_labels}')
    # Save the outputs and labels
    np.save(os.path.join(results_folder, 'all_test_outputs.npy'), all_test_outputs)
    np.save(os.path.join(results_folder, 'all_test_labels.npy'), all_test_labels)
    # Save the predictions, labels, and sequences to a CSV file

    sequences = []
    for fold_dict in folds_training_dicts:
        sequences.extend(fold_dict['sequences_test'])

    df = pd.DataFrame({
        'sequence': sequences,
        'prediction': all_test_outputs,
        'label': all_test_labels
    })

    csv_path = os.path.join(results_folder, 'predictions_labels_sequences.csv')
    df.to_csv(csv_path, index=False)
    print(f'Saved predictions, labels, and sequences to {csv_path}')
    test_pr_auc = precision_recall_auc(all_test_labels, all_test_outputs)
    pr_curve_save_path = os.path.join(results_folder, 'pr_curve_model.png')
    roc_curve_save_path = os.path.join(results_folder, 'roc_curve_model.png')
    print(f'save_path_pr : {pr_curve_save_path}, save_path_roc : {roc_curve_save_path}')
    plot_pr_curve(all_test_labels, all_test_outputs, save_path=pr_curve_save_path, title=f'{title} Precision-Recall Curve')
    plot_roc_curve(all_test_labels, all_test_outputs, save_path=roc_curve_save_path, title=f'{title} ROC Curve')

    

if __name__ == '__main__':
    DATE = '11_01'
    roc_metric = True
    fine_tune_untrained = False
    fine_tune_untrained_addition = "_untrained" if fine_tune_untrained else ""
    my_pretrained = False
    from_scratch = False
    date_for_pretrained = '2025-02-14' if from_scratch else '2025-02-20'
    from_scratch_addition = "_from_scratch" if from_scratch else ""
    folds_traning_dicts = load_as_pickle(os.path.join(paths.data_for_training_path, DATE, 'folds_traning_dicts.pkl'))
    esm2_model_name = 'facebook/esm2_t6_8M_UR50D' 
    model_path = os.path.join(paths.esm2_peptide_pretrained_models_path,f'peptide_{esm2_model_name.split("/")[-1]}{from_scratch_addition}_{date_for_pretrained}')
    model_name = model_path if my_pretrained else esm2_model_name
    # Define the grid search parameters
    param_grid = {
        'model_name': [model_name],
        'num_epochs': [100],
        'batch_size': [128, 256,512],
    }
    for model_name in param_grid['model_name']:
        print(f"Training model: {model_name}")
        print("=========================================")
        print("Grid search parameters:")
        print(param_grid)
        
        models_folder = os.path.join(paths.fine_tune_models_path, DATE,os.path.basename(model_path) if my_pretrained else esm2_model_name.split('/')[-1])
        if not os.path.exists(models_folder):
            os.makedirs(models_folder)
        save_dir = os.path.join(models_folder, 'checkpoints')
        checkpoint_folder = os.path.join(models_folder,'checkpoints')
        if not os.path.exists(checkpoint_folder):
            os.makedirs(checkpoint_folder)
        results_folder = os.path.join(paths.fine_tune_results_path, DATE, os.path.basename(model_path) if my_pretrained else esm2_model_name.split('/')[-1])
        if not os.path.exists(results_folder):
            os.makedirs(results_folder)
        best_model = None
        best_architecture_pr_auc = 0
        best_architecture_index = 0
        grid_results = []
        cnt = 0
        # Iterate over grid parameters
        for num_epochs in param_grid['num_epochs']:
            for batch_size in param_grid['batch_size']:
                architecture_pr_aucs, architecture_models,architecture_test_prediction_logits_outputs,trainers = train_architecture_over_folds(batch_size, num_epochs, folds_traning_dicts,model_name,fine_tune_untrained)
                architecture_pr_auc = np.mean(architecture_pr_aucs)
                
                # Save the grid search results
                grid_results.append({
                    'num_epochs': num_epochs,
                    'batch_size': batch_size,
                    'val_metric': architecture_pr_auc
                })

                # Save the best models
                if architecture_pr_auc > best_architecture_pr_auc:
                    best_architecture_pr_auc = architecture_pr_auc
                    best_models = architecture_models
                    best_architecture_index = cnt
                    best_trainers = trainers
                    best_architecture_test_prediction_logits_outputs = architecture_test_prediction_logits_outputs
                cnt += 1
        
        # Save best models and results
        architecture_model_folder = save_best_models(best_trainers, grid_results, best_architecture_index, models_folder,roc_metric)
        architecture_model_folder = architecture_model_folder + fine_tune_untrained_addition
        architecture_results_folder = os.path.join(results_folder, architecture_model_folder.split('/')[-1])
        if not os.path.exists(architecture_results_folder):
            os.makedirs(architecture_results_folder)
        save_grid_search_results(grid_results, architecture_results_folder)
        save_test_metric(best_architecture_test_prediction_logits_outputs,architecture_results_folder,'esm2 fine tune',folds_traning_dicts)

