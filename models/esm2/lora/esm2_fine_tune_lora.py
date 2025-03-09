import os 
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..','..','..'))
import paths
import torch
from transformers import TrainingArguments, EsmForSequenceClassification, Trainer,EsmTokenizer,AutoConfig
from peft import LoraConfig, get_peft_model,PeftModel,TaskType
from utils import load_as_pickle, plot_pr_curve, plot_roc_curve
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

# Custom WeightedTrainer


# Define early stopping callback
early_stopping_callback = EarlyStoppingCallback(
    early_stopping_patience=8,  # Number of epochs to wait for improvement
    early_stopping_threshold=0.0,  # Minimum change to consider as improvement
)

def save_best_models(models, grid_results, best_architecture_index, models_folder,roc_metric):
    # Save the best models
    roc_metric_addition = "_roc_metric" if roc_metric else ""
    best_lora_alpha = str(grid_results[best_architecture_index]['lora_alpha'])
    best_num_epochs = grid_results[best_architecture_index]['num_epochs']
    best_r = grid_results[best_architecture_index]['r']
    best_batch_size = grid_results[best_architecture_index]['batch_size']
    architecture_model_folder = os.path.join(models_folder, f'architecture_{best_batch_size}_{best_num_epochs}_{best_r}_{best_lora_alpha}{roc_metric_addition}')
    if not os.path.exists(architecture_model_folder):
        os.makedirs(architecture_model_folder)
    for i in range(len(models)):
        model = models[i]
        model.save_pretrained(os.path.join(architecture_model_folder, 
                                                f'model_{best_batch_size}_{best_num_epochs}_{best_r}_{best_lora_alpha}_{i+1}'))
    return architecture_model_folder

def save_test_metric(architecture_test_prediction_outputs,results_folder,title,folds_training_dicts):
    all_test_outputs = []
    all_test_labels = []
    for i in range(len(architecture_test_prediction_outputs)):
        predictions = architecture_test_prediction_outputs[i].predictions
        predictions = 1 / (1 + np.exp(-predictions.squeeze()))
        labels = architecture_test_prediction_outputs[i].label_ids
        all_test_outputs.append(predictions)
        all_test_labels.append(labels)
        test_pr_auc = precision_recall_auc(labels, predictions)
        print(f"Fold {i + 1} Test PR AUC: {test_pr_auc:.4f}")
    all_test_outputs = np.concatenate(all_test_outputs)
    all_test_labels = np.concatenate(all_test_labels)
    test_pr_auc = precision_recall_auc(all_test_labels,all_test_outputs)
    pr_curve_save_path = os.path.join(results_folder, 'pr_curve_model.png')
    roc_curve_save_path = os.path.join(results_folder, 'roc_curve_model.png')
    print(f'pr curve save path: {pr_curve_save_path}')
    print(f'roc curve save path: {roc_curve_save_path}')
    np.save(os.path.join(results_folder, 'all_test_outputs.npy'), all_test_outputs)
    np.save(os.path.join(results_folder, 'all_test_labels.npy'), all_test_labels)
    plot_pr_curve(all_test_labels, all_test_outputs, save_path=pr_curve_save_path, title=f'{title} PR Curve')
    plot_roc_curve(all_test_labels, all_test_outputs, save_path=roc_curve_save_path, title=f'{title} ROC Curve')
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

# Tokenize datasets
def tokenize_function(tokenizer,sequences, labels):
    tokens = tokenizer(sequences, truncation=True, padding="max_length", max_length=50)
    tokens["labels"] = labels
    return tokens

def train_architecture_over_folds(batch_size, num_epochs,r,lora_alpha, folds_training_dicts,model_name,fine_tune_untrained):

    architecture_pr_aucs = []
    models = []
    trainers = []
    test_prediction_outputs = []
    # Early stopping variables
    patience = 8  # Number of epochs to wait after last improvement

    # Iterate through each fold in the training data
    for fold_index, fold_dict in enumerate(folds_training_dicts):
        train_sequences = fold_dict['sequences_train']
        train_labels =  fold_dict['labels_train']    # Your list of train labels
        val_sequences = fold_dict['sequences_validation']    # Your list of validation sequences
        val_labels = fold_dict['labels_validation']       # Your list of validation labels
        test_sequences = fold_dict['sequences_test']   # Your list of test sequences
        test_labels = fold_dict['labels_test']     # Your list of test labels

        class_weights = WeightedTrainer.calculate_class_weights(torch.tensor(train_labels))
        tokenizer = EsmTokenizer.from_pretrained(model_name)
        
        train_dataset = Dataset.from_dict(tokenize_function(tokenizer,train_sequences, train_labels))
        val_dataset = Dataset.from_dict(tokenize_function(tokenizer,val_sequences, val_labels))
        test_dataset = Dataset.from_dict(tokenize_function(tokenizer,test_sequences, test_labels))

        # Convert to PyTorch format
        train_dataset.set_format("torch")
        val_dataset.set_format("torch")
        test_dataset.set_format("torch")
        
        if fine_tune_untrained:
            config = AutoConfig.from_pretrained(model_name)
            config.num_labels = 1
            model = EsmForSequenceClassification(config)
        else:
            model = EsmForSequenceClassification.from_pretrained(model_name, num_labels=1)
        target_modules = []
        for name, _ in model.named_modules():
            if re.match(r"esm\.encoder\.layer\.\d+\.attention\.self\.query", name) or re.match(r"esm\.encoder\.layer\.\d+\.attention\.self\.value", name):
                target_modules.append(name)
        print(f'target_modules: {target_modules}')
        # LoRA configuration
        peft_config = LoraConfig(
            task_type=TaskType.SEQ_CLS,
            inference_mode=False,
            target_modules= target_modules,
            r=r,
            lora_alpha=lora_alpha,
            lora_dropout=0.1,
            modules_to_save=["classifier"]
        )
        lora_model = get_peft_model(model, peft_config)

        trainable_params = [name for name, param in model.named_parameters() if param.requires_grad]
        print(f"Trainable parameters: {trainable_params}")
                


        training_args = TrainingArguments(
            output_dir=checkpoint_folder,  # Directory to save model checkpoints
            learning_rate=1e-3,
            per_device_train_batch_size=batch_size,
            num_train_epochs=num_epochs,
            weight_decay=0.01,
            per_device_eval_batch_size  = len(fold_dict['sequences_validation']),
            per_gpu_eval_batch_size = len(fold_dict['sequences_validation']),
            greater_is_better=True,
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="roc_auc",
            logging_dir=os.path.join(checkpoint_folder, 'logs'),
            logging_steps=10,
            report_to="none",
        )

        # Create a WeightedTrainer
        trainer = WeightedTrainer(
            model=lora_model,
            class_weights=class_weights,  # Pass the class weights to the trainer
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            tokenizer=tokenizer,
            callbacks=[early_stopping_callback],
            compute_metrics= metrics_evaluation
        )        
        # Training arguments
        
        lora_model.print_trainable_parameters()
        print(f'trainer.compute_metrics: {trainer.compute_metrics}')   
        trainer.train()
        models.append(trainer.model)
        architecture_pr_aucs.append(trainer.state.best_metric)
        print(f"Fold {fold_index + 1} done, best metric score: {trainer.state.best_metric:.4f}")
        trainers.append(trainer)
        predictions_output = trainer.predict(test_dataset)
        test_prediction_outputs.append(predictions_output)

    return architecture_pr_aucs, models,test_prediction_outputs,trainers



if __name__ == "__main__":
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
    
    param_grid = {
        
        'batch_size': [128,256,512],
        'num_epochs': [50],
        'r': [25,50,100],
        'lora_alpha': [16,32], 
        'model_name': [model_name] 
    }

    best_model = None
    best_architecture_pr_auc = 0
    best_architecture_index = 0
    grid_results = []
    cnt = 0
    for model_name in param_grid['model_name']:
        print(f"Training model: {model_name}")
        print("=========================================")
        print("Grid search parameters:")
        print(param_grid)
        
        models_folder = os.path.join(paths.lora_models_path, DATE, model_name.split('/')[-1])
        if not os.path.exists(models_folder):
            os.makedirs(models_folder)
        save_dir = os.path.join(models_folder, 'checkpoints')
        checkpoint_folder = os.path.join(models_folder,'checkpoints')
        if not os.path.exists(checkpoint_folder):
            os.makedirs(checkpoint_folder)
        results_folder = os.path.join(paths.lora_results_path, DATE, model_name.split('/')[-1])
        if not os.path.exists(results_folder):
            os.makedirs(results_folder)
    
        for batch_size in param_grid['batch_size']:
            for num_epochs in param_grid['num_epochs']:
                for r in param_grid['r']:
                    for lora_alpha in param_grid['lora_alpha']:
                        architecture_pr_aucs, architecture_models,architecture_test_prediction_outputs,trainers = train_architecture_over_folds(batch_size, num_epochs,r,lora_alpha, folds_traning_dicts,model_name,fine_tune_untrained)
                        architecture_pr_auc = np.mean(architecture_pr_aucs)
                        # Save the grid search results
                        grid_results.append({
                            'batch_size': batch_size,
                            'num_epochs': num_epochs,
                            'r': r,
                            'lora_alpha': lora_alpha,
                            'val_metric': architecture_pr_auc
                        })

                        # Save the best models
                        if architecture_pr_auc > best_architecture_pr_auc:
                            best_architecture_pr_auc = architecture_pr_auc
                            best_models = architecture_models
                            best_architecture_index = cnt
                            best_trainers = trainers
                            best_architecture_test_prediction_outputs = architecture_test_prediction_outputs
                        cnt += 1

    

    architecture_model_folder = save_best_models(best_models, grid_results, best_architecture_index, models_folder,roc_metric)
    architecture_results_folder = os.path.join(results_folder, architecture_model_folder.split('/')[-1])
    os.makedirs(architecture_results_folder, exist_ok=True)
    save_grid_search_results(grid_results, architecture_results_folder)
    save_test_metric(best_architecture_test_prediction_outputs,architecture_results_folder,'esm2 fine tune LoRA',folds_traning_dicts)
