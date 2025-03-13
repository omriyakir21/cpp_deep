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
from models.baselines.convolution_baseline.convolution_baseline_utils import calculate_pr_auc,calculate_roc_auc
import itertools

# Custom WeightedTrainer


# Define early stopping callback
early_stopping_callback = EarlyStoppingCallback(
    early_stopping_patience=8,  # Number of epochs to wait for improvement
    early_stopping_threshold=0.0,  # Minimum change to consider as improvement
)



def save_best_models(models, best_architecture_grid_info, models_folder,roc_metric,bagging):
    roc_metric_addition = "_roc_metric" if roc_metric else ""
    val_metric = best_architecture_grid_info['val_metric']    
    del best_architecture_grid_info['val_metric']
    architecture_key_val_string = '_'.join([f'{key}_{value}' for key, value in best_architecture_grid_info.items()])
    architecture_val_string = '_'.join([str(val) for val in best_architecture_grid_info.values()])
    best_architecture_grid_info['val_metric'] = val_metric
    architecture_model_folder = os.path.join(models_folder, f'architecture_{architecture_key_val_string}{roc_metric_addition}')
    os.makedirs(architecture_model_folder, exist_ok=True)
    for i in range(len(models)):
        if bagging:
            fold_folder = os.path.join(architecture_model_folder,f'fold_models_{i}')
            os.makedirs(fold_folder, exist_ok=True)
            fold_best_models = models[i]
            for index, best_model in enumerate(fold_best_models):
                best_model.save_pretrained(os.path.join(fold_folder, f'model_{architecture_val_string}_{index+1}'))
        else:
            model = models[i]
            model.save_pretrained(os.path.join(architecture_model_folder, 
                                                    f'model_{architecture_val_string}_{i+1}'))
    return architecture_model_folder

def save_test_metric(best_models,results_folder,title,folds_training_dicts,tokenizer,roc_metric,bagging):
    all_test_predictions = []
    all_test_labels = []
    for i in range(len(best_models)):
        fold_dict = folds_training_dicts[i]
        X_test = tokenizer(fold_dict['sequences_test'], truncation=True, padding="max_length", max_length=50, return_tensors='pt')
        y_test = fold_dict['labels_test']
        if bagging:
            fold_models = best_models[i]
            test_predictions = np.zeros(len(y_test))
            for model in fold_models:
                test_predictions+= predict_binary_probs(model, X_test).flatten()
            test_predictions /= len(fold_models)
        else:
            model = best_models[i]
            test_predictions = predict_binary_probs(model, X_test).flatten()
        
        all_test_predictions.append(test_predictions)
        all_test_labels.append(y_test)
        fold_metric_score = calculate_roc_auc(y_test,test_predictions) if roc_metric else calculate_pr_auc(y_test,test_predictions)
        print(f"Fold {i + 1} done, best test metric score: {fold_metric_score}")
    
    all_test_predictions = np.concatenate(all_test_predictions)
    all_test_labels = np.concatenate(all_test_labels)

    pr_curve_save_path = os.path.join(results_folder, 'pr_curve_model.png')
    roc_curve_save_path = os.path.join(results_folder, 'roc_curve_model.png')
    print(f'pr curve save path: {pr_curve_save_path}')
    print(f'roc curve save path: {roc_curve_save_path}')
    np.save(os.path.join(results_folder, 'all_test_outputs.npy'), all_test_predictions)
    np.save(os.path.join(results_folder, 'all_test_labels.npy'), all_test_labels)
    plot_pr_curve(all_test_labels, all_test_predictions, save_path=pr_curve_save_path, title=f'{title} PR Curve')
    plot_roc_curve(all_test_labels, all_test_predictions, save_path=roc_curve_save_path, title=f'{title} ROC Curve')
    sequences = []
    for fold_dict in folds_training_dicts:
        sequences.extend(fold_dict['sequences_test'])
    df = pd.DataFrame({
        'sequence': sequences,
        'prediction': list(all_test_predictions),
        'label': list(all_test_labels)
    })

    csv_path = os.path.join(results_folder, 'predictions_labels_sequences.csv')
    df.to_csv(csv_path, index=False)
    print(f'Saved predictions, labels, and sequences to {csv_path}')

# Tokenize datasets
def tokenize_function(tokenizer,sequences, labels):
    tokens = tokenizer(list(sequences), truncation=True, padding="max_length", max_length=50, return_tensors='pt')
    tokens["labels"] = list(labels)
    return tokens

def train_architecture_fold(batch_size, num_epochs,r,lora_alpha, fold_dict,model_name,fine_tune_untrained,tokenizer,train_sequences,train_labels,checkpoint_folder):
    if train_sequences is None or train_labels is None:
        train_sequences = fold_dict['sequences_train']
        train_labels =  fold_dict['labels_train']  
    print(f'train_sequences : {train_sequences}')
    val_sequences = fold_dict['sequences_validation']    
    val_labels = fold_dict['labels_validation']       

    class_weights = WeightedTrainer.calculate_class_weights(torch.tensor(train_labels))    
    train_dataset = Dataset.from_dict(tokenize_function(tokenizer,train_sequences, train_labels))
    val_dataset = Dataset.from_dict(tokenize_function(tokenizer,val_sequences, val_labels))

    # Convert to PyTorch format
    train_dataset.set_format("torch")
    val_dataset.set_format("torch")
    
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
        output_dir=checkpoint_folder,
        learning_rate=1e-3,
        per_device_train_batch_size=batch_size,
        num_train_epochs=num_epochs,
        weight_decay=0.01,
        per_device_eval_batch_size  = batch_size,
        per_gpu_eval_batch_size = batch_size,
        greater_is_better=True,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="roc_auc",
        logging_dir=os.path.join(checkpoint_folder, 'logs'),
        logging_steps=10,
        report_to="none",
        )
    
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
    print(f"Fold done, best metric score: {trainer.state.best_metric:.4f}")
    return lora_model

def predict_binary_probs(lora_model, inputs):
    """
    Perform inference using a LoRA fine-tuned model for binary classification.
    
    Args:
        lora_model: The trained PeftModel (LoRA fine-tuned model).
        inputs: Tokenized inputs (output of tokenizer).

    Returns:
        probabilities (numpy array): Predicted probabilities for the positive class.
    """
    # Ensure the model is in evaluation mode
    lora_model.eval()

    # Move inputs to the same device as the model
    device = next(lora_model.parameters()).device
    inputs = {key: value.to(device) for key, value in inputs.items()}

    with torch.no_grad():
        outputs = lora_model(**inputs)

    logits = outputs.logits

    # Apply sigmoid to get probabilities
    probabilities = torch.sigmoid(logits).cpu().numpy()

    return probabilities

def train_architecture_over_folds(batch_size, num_epochs,r,lora_alpha, folds_training_dicts,model_name,fine_tune_untrained,tokenizer,roc_metric,checkpoint_folder):
    
    models = []
    models = []
    all_val_predictions = []
    all_val_labels = []
    for fold_index, fold_dict in enumerate(folds_training_dicts):
        model = train_architecture_fold(batch_size, num_epochs,r,lora_alpha, fold_dict,model_name,fine_tune_untrained,tokenizer,None,None,checkpoint_folder)
        models.append(model)
        X_val = tokenizer(fold_dict['sequences_validation'], truncation=True, padding="max_length", max_length=50, return_tensors='pt')
        y_val = fold_dict['labels_validation']
        val_predictions = predict_binary_probs(model, X_val) 
        all_val_predictions.append(val_predictions)
        all_val_labels.append(y_val)
        fold_metric_score = calculate_roc_auc(y_val,val_predictions) if roc_metric else calculate_pr_auc(y_val,val_predictions)
        print(f"Fold {fold_index + 1} done, best metric score: {fold_metric_score}")
    
    all_val_predictions = np.concatenate(all_val_predictions)
    all_val_labels = np.concatenate(all_val_labels)
    architecture_metric_score = calculate_roc_auc(all_val_labels,all_val_predictions) if roc_metric else calculate_pr_auc(all_val_labels,all_val_predictions)
    print(f"Architecture done, best metric score: {architecture_metric_score}")
    return models,architecture_metric_score




if __name__ == "__main__":
    DATE = '13_09'
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
        'batch_size': [256],
        'num_epochs': [50],
        'r': [100],
        'lora_alpha': [16], 
        'model_name': [model_name] 
    }

    best_model = None
    best_architecture_val_metric = 0
    best_architecture_index = 0
    grid_results = []
    cnt = 0
    for model_name in param_grid['model_name']:
        print(f"Training model: {model_name}")
        print("=========================================")
        print("Grid search parameters:")
        print(param_grid)
        
        models_folder = os.path.join(paths.lora_models_path, DATE, model_name.split('/')[-1])
        os.makedirs(models_folder, exist_ok=True)
        save_dir = os.path.join(models_folder, 'checkpoints')
        checkpoint_folder = os.path.join(models_folder,'checkpoints')
        os.makedirs(checkpoint_folder, exist_ok=True)
        results_folder = os.path.join(paths.lora_results_path, DATE, model_name.split('/')[-1])
        os.makedirs(results_folder, exist_ok=True)
        tokenizer = EsmTokenizer.from_pretrained(model_name)
        for batch_size, num_epochs, r, lora_alpha in itertools.product(
        param_grid['batch_size'],
        param_grid['num_epochs'],
        param_grid['r'],
        param_grid['lora_alpha']):
            print(f"Training model with the following parameters: batch_size={batch_size}, num_epochs={num_epochs}, r={r}, lora_alpha={lora_alpha}")
            print("=========================================")
            architecture_grid_info = {
                'batch_size': batch_size,
                'num_epochs': num_epochs,
                'r': r,
                'lora_alpha': lora_alpha,}
            architecture_models,architecture_val_metric = train_architecture_over_folds(batch_size, num_epochs,r,lora_alpha, folds_traning_dicts,model_name,fine_tune_untrained,tokenizer,roc_metric,checkpoint_folder)
            architecture_grid_info['val_metric'] = architecture_val_metric
            grid_results.append(architecture_grid_info)
            if architecture_val_metric > best_architecture_val_metric:
                best_architecture_val_metric = architecture_val_metric
                best_models = architecture_models
                best_architecture_grid_info = architecture_grid_info
            cnt += 1
    

    architecture_model_folder = save_best_models(best_models, best_architecture_grid_info, models_folder,roc_metric,False,None)
    architecture_results_folder = os.path.join(results_folder, architecture_model_folder.split('/')[-1])
    os.makedirs(architecture_results_folder, exist_ok=True)
    save_grid_search_results(grid_results, architecture_results_folder)
    save_test_metric(best_models,architecture_results_folder,'esm2 fine tune LoRA',folds_traning_dicts,tokenizer,roc_metric,False)