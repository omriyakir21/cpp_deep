import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..','..','..'))
import torch
from transformers import EsmForSequenceClassification, EsmTokenizer
from peft import LoraConfig, get_peft_model,TaskType,PeftConfig,PeftModel
import paths
import pandas as pd
from models.esm2.fine_tune.fine_tune import create_dataset
import numpy as np
from torch.utils.data import DataLoader
from models.esm2.ems2_utils import precision_recall_auc
from utils import plot_pr_curve, load_as_pickle
from sklearn.metrics import accuracy_score, recall_score, f1_score, matthews_corrcoef
import matplotlib.pyplot as plt
from models.esm2.fine_tune.predict_fine_tune import load_datasets, predict_with_models,predict_over_test_set
from datasets import Dataset
import json

def load_best_models(best_lora_alpha, best_r, best_batch_size, best_num_epochs, models_folder, model_name):
    architecture_model_folder = os.path.join(models_folder, f'architecture_{best_batch_size}_{best_num_epochs}_{best_r}_{best_lora_alpha}')
    models = []
    for i in range(1, 6):
        model_path = os.path.join(architecture_model_folder, f'model_{best_batch_size}_{best_num_epochs}_{best_r}_{best_lora_alpha}_{i}')
        config = PeftConfig.from_pretrained(model_path)
        model = EsmForSequenceClassification.from_pretrained(config.base_model_name_or_path, num_labels=1)
        lora_model = PeftModel.from_pretrained(model, model_path)
        lora_model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        lora_model.print_trainable_parameters()

        models.append(lora_model)
    
    return models

def calculate_best_threshold(predictions, labels):
    best_f1 = 0
    best_threshold = 0
    for threshold in np.arange(0.1, 1.0, 0.01):
        binary_predictions = (predictions >= threshold).astype(int)
        f1 = f1_score(labels, binary_predictions)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
    return best_threshold, best_f1

def tokenize_function(tokenizer,sequences, labels):
    tokens = tokenizer(sequences, truncation=True, padding="max_length", max_length=50)
    tokens["labels"] = labels
    return tokens


def save_top_false_positives(predictions_file, labels_file, folds_training_dicts, results_folder):
    predictions = np.load(predictions_file)
    predictions = 1 / (1 + np.exp(-predictions))
    labels = np.load(labels_file)

    false_positives = []

    index = 0  
    for fold_dict in folds_training_dicts:
        sequences_test = fold_dict['sequences_test']

        for sequence in sequences_test:
            pred = predictions[index]
            label = labels[index]

            if label == 0:
                false_positives.append((float(pred), index, sequence))

            index += 1  

    false_positives.sort(reverse=True, key=lambda x: x[0])

    top_false_positives = false_positives[:50]

    top_false_positives_path = os.path.join(results_folder, 'top_false_positives.json')
    with open(top_false_positives_path, 'w') as f:
        json.dump(top_false_positives, f, indent=4)
    print(f"Top 50 false positives saved to {top_false_positives_path}")

def predict_and_concatenate(folds_training_dicts, models, tokenizers, max_length=50):
    all_predictions = []
    all_labels = []
    for fold_index, fold_dict in enumerate(folds_training_dicts):
        test_sequences = fold_dict['sequences_test']   # Your list of test sequences
        test_labels = fold_dict['labels_test']     # Your list of test labels
        tokenizer = tokenizers[fold_index]
        test_dataset = Dataset.from_dict(tokenize_function(tokenizer,test_sequences, test_labels))
        test_dataset.set_format("torch")
        dataloader = DataLoader(test_dataset, batch_size=32)
        fold_logits = []
        with torch.no_grad():
            for batch in dataloader:
                inputs = {k: v.to(models[0].device) for k, v in batch.items()}
                logits = torch.zeros(len(batch["input_ids"]), device=models[0].device)
                outputs = models[fold_index](**inputs)
                logits += outputs.logits.squeeze()
                logits /= len(models)
                fold_logits.append(logits.cpu().numpy())

        if fold_logits:
            fold_predictions = np.concatenate(fold_logits)
            all_predictions.append(fold_predictions)
            all_labels.extend(test_labels)
        else:
            print(f"No predictions for fold {fold_index}.")

    if not all_predictions:
        raise ValueError("No predictions were generated for any folds.")

    concatenated_predictions = np.concatenate(all_predictions)
    concatenated_predictions = 1 / (1 + np.exp(-concatenated_predictions))
    concatenated_labels = np.array(all_labels)

    print(f"Total predictions: {len(concatenated_predictions)}, Total labels: {len(concatenated_labels)}")
    return concatenated_predictions, concatenated_labels

def main():
    best_batch_size, best_num_epochs, best_r, best_lora_alpha = 128, 50, 100, 32
    model_dir = paths.lora_models_path
    result_dir = paths.lora_results_path
    DATE = '13_09'
    model_name = 'esm2_t6_8M_UR50D'
    models_folder = os.path.join(model_dir, DATE, model_name.split('/')[-1])
    results_folder = os.path.join(result_dir, DATE, model_name.split('/')[-1])    
    models = load_best_models(best_lora_alpha, best_r, best_batch_size, best_num_epochs, models_folder, model_name)
    tokenizers = [EsmTokenizer.from_pretrained('facebook/esm2_t6_8M_UR50D') for _ in range(5)]
    df = pd.read_csv(os.path.join(paths.full_datasets_path, 'cpp924_13_09.csv'))
    datasets_finished = load_datasets(df, tokenizers)
    
    folds_traning_dicts = load_as_pickle(os.path.join(paths.data_for_training_path, DATE, 'folds_traning_dicts.pkl'))
    concatenated_predictions, concatenated_labels = predict_and_concatenate(folds_traning_dicts, models, tokenizers)

    predictions = 1 / (1 + np.exp(-concatenated_predictions))
    pr_auc = precision_recall_auc(concatenated_labels, predictions)

    print(f"Combined PR AUC from all folds: {pr_auc:.4f}")

    all_predictions, all_labels = predict_with_models(models, datasets_finished)

    concatenated_logits = np.concatenate(all_predictions, axis=0)
    concatenated_labels = np.concatenate(all_labels, axis=0)
    predictions = 1 / (1 + np.exp(-concatenated_logits))
    test_pr_auc = precision_recall_auc(concatenated_labels, predictions)
    save_path = os.path.join(results_folder, 'pr_curve_model_CPP924.png')
    title = f'{model_name} Precision-Recall Curve over CPP924'
    plot_pr_curve(concatenated_labels, predictions, save_path=save_path, title=title)

    best_threshold, best_f1 = calculate_best_threshold(predictions, concatenated_labels)
    binary_predictions = (predictions >= best_threshold).astype(int)

    accuracy = accuracy_score(concatenated_labels, binary_predictions)
    sensitivity = recall_score(concatenated_labels, binary_predictions)
    specificity = recall_score(concatenated_labels, binary_predictions, pos_label=0)
    mcc = matthews_corrcoef(concatenated_labels, binary_predictions)
    normalized_mcc = (mcc + 1) / 2 * 100  # Normalize MCC to percentage

    print(f'Best Threshold: {best_threshold:.2f}')
    print(f'Best F1 Score: {best_f1:.4f}')
    print(f'Accuracy: {accuracy * 100:.2f}%')
    print(f'Sensitivity (Recall): {sensitivity * 100:.2f}%')
    print(f'Specificity: {specificity * 100:.2f}%')
    print(f'MCC: {normalized_mcc:.2f}%')


    metrics = {
        'pr_auc': pr_auc,
        'Best F1 Score (%)': best_f1 * 100,
        'Accuracy (%)': accuracy * 100,
        'Sensitivity (%)': sensitivity * 100,
        'Specificity (%)': specificity * 100,
        'MCC (%)': normalized_mcc
    }

    metrics_path = os.path.join(results_folder, 'metrics_lora.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=4)
    print(f"Metrics saved to {metrics_path}")

    folds_traning_dicts = load_as_pickle(os.path.join(paths.data_for_training_path, DATE, 'folds_traning_dicts.pkl'))
    architecture = "architecture_128_50_100_32"
    labels_name = "all_test_labels.npy"
    predictions_name = "all_test_outputs.npy"
    labels_path = os.path.join(paths.lora_results_path, DATE, model_name, architecture, labels_name)
    predictions_path = os.path.join(paths.lora_results_path, DATE, model_name, architecture, predictions_name)
    save_top_false_positives(predictions_path, labels_path, folds_traning_dicts, results_folder)

    fig, ax = plt.subplots()
    ax.bar(metrics.keys(), metrics.values())
    ax.set_ylabel('Score')
    ax.set_title('LoRA Performance Metrics')
    ax.set_ylim(0, 100)
    ax.set_yticks(range(0, 101, 10))
    metrics_save_path = os.path.join(results_folder, 'lora_performance_metrics.png')
    plt.savefig(metrics_save_path)
    plt.close()


if __name__ == '__main__':
    # main()
    best_batch_size, best_num_epochs, best_r, best_lora_alpha = 128, 50, 100, 32
    model_dir = paths.lora_models_path
    result_dir = paths.lora_results_path
    DATE = '13_09'
    model_name = 'esm2_t6_8M_UR50D'
    models_folder = os.path.join(model_dir, DATE, model_name.split('/')[-1])
    results_folder = os.path.join(result_dir, DATE, model_name.split('/')[-1])    
    models = load_best_models(best_lora_alpha, best_r, best_batch_size, best_num_epochs, models_folder, model_name)
    tokenizers = [EsmTokenizer.from_pretrained('facebook/esm2_t6_8M_UR50D') for _ in range(5)]
    folds_traning_dicts = load_as_pickle(os.path.join(paths.data_for_training_path, DATE, 'folds_traning_dicts.pkl'))
    concatenated_predictions, concatenated_labels = predict_and_concatenate(folds_traning_dicts, models, tokenizers)
    print(f'concatenated_predictions {concatenated_predictions.shape}')
    save_path = os.path.join(results_folder, 'pr_curve_model_lora_inference.png')
    title = 'Precision-Recall Curve lora inference'
    # plot_pr_curve(concatenated_labels, concatenated_predictions, save_path=save_path, title=title)
    predictions = np.load("/home/iscb/wolfson/omriyakir/cpp_deep/results/esm2/lora/13_09/esm2_t6_8M_UR50D/architecture_128_50_100_32/all_test_outputs.npy")
    predictions = 1 / (1 + np.exp(-predictions))
    predictions = predictions.squeeze()
    print(f'predictions {predictions.shape}')
    gap = concatenated_predictions - predictions
    for i,g in enumerate(gap):
        print(f'gap {i} {g}')
