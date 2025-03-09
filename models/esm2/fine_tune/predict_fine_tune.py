import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..','..','..'))
from transformers import EsmForSequenceClassification, EsmTokenizer
import paths
import pandas as pd
from models.esm2.fine_tune.fine_tune import create_dataset
import numpy as np
import torch
from torch.utils.data import DataLoader
from models.esm2.ems2_utils import precision_recall_auc
from utils import plot_pr_curve, load_as_pickle
from sklearn.metrics import accuracy_score, recall_score, matthews_corrcoef, f1_score
import matplotlib.pyplot as plt
import json

def load_models_and_tokenizers(architecture_model_folder, architecture, num_models=5):
    models = []
    tokenizers = []
    for i in range(1, num_models + 1):
        model_path = os.path.join(architecture_model_folder, f'model_{architecture}_{i}')
        model = EsmForSequenceClassification.from_pretrained(model_path)
        tokenizer = EsmTokenizer.from_pretrained(model_path)
        models.append(model)
        tokenizers.append(tokenizer)
    return models, tokenizers

def load_datasets(df, tokenizers, max_length=50):
    datasets = {}
    for i in range(5):
        datasets[f'index_{i+1}'] = df[df['folds_index_to_predict'] == i]
    datasets['none'] = df[df['folds_index_to_predict'].isna()]

    datasets_finished = {}
    for key, dataset in datasets.items():
        datasets_finished[key] = create_dataset(dataset['sequence'].tolist(), dataset['label'].tolist(), tokenizers[0], max_length=max_length)
    return datasets_finished

def predict_with_models(models, datasets_finished):
    all_predictions = []
    all_none_predictions = []
    all_labels = []

    for i in range(5):
        model = models[i]
        dataset = datasets_finished[f'index_{i+1}']
        dataloader = DataLoader(dataset, batch_size=len(dataset))
        logits_list = []
        labels_list = []
        for batch in dataloader:
            inputs = {k: v.to(model.device) for k, v in batch.items() if k != 'labels'}
            with torch.no_grad():
                outputs = model(**inputs)
            logits_list.append(outputs.logits.squeeze().cpu().numpy())
            labels_list.append(batch['labels'].cpu().numpy())
        all_predictions.append(np.concatenate(logits_list))
        all_labels.append(np.concatenate(labels_list))

        none_dataset = datasets_finished['none']
        none_dataloader = DataLoader(none_dataset, batch_size=len(none_dataset))
        none_logits_list = []
        none_labels_list = []
        for batch in none_dataloader:
            none_inputs = {k: v.to(model.device) for k, v in batch.items() if k != 'labels'}
            with torch.no_grad():
                none_outputs = model(**none_inputs)
            none_logits_list.append(none_outputs.logits.squeeze().cpu().numpy())
            none_labels_list.append(batch['labels'].cpu().numpy())
        all_none_predictions.append(np.concatenate(none_logits_list))
        all_none_labels = np.concatenate(none_labels_list)
    
    mean_none_predictions = sum(all_none_predictions) / len(all_none_predictions)

    all_predictions.append(mean_none_predictions)
    all_labels.append(all_none_labels)
    
    return all_predictions, all_labels

# def save_top_false_positives(predictions, labels, results_folder, df):
#     false_positives = [(float(pred), int(idx), df.iloc[idx]['sequence']) for idx, (pred, label) in enumerate(zip(predictions, labels)) if label == 0]
#     false_positives.sort(reverse=True, key=lambda x: x[0])  # Sort by prediction confidence
#     top_false_positives = false_positives[:50]

#     top_false_positives_path = os.path.join(results_folder, 'top_false_positives.json')
#     with open(top_false_positives_path, 'w') as f:
#         json.dump(top_false_positives, f, indent=4)
#     print(f"Top 10 false positives saved to {top_false_positives_path}")

def save_top_false_positives(predictions_file, labels_file, folds_training_dicts, results_folder):
    predictions = np.load(predictions_file)
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

def evaluate_predictions(predictions, labels, results_folder, model_name):
    pr_auc = precision_recall_auc(labels, predictions)
    print(f'Precision-Recall AUC: {pr_auc:.4f}')

    pr_curve_path = os.path.join(results_folder, 'pr_curve_model_fine_tune.png')
    title = f'{model_name} Precision-Recall Curve'
    plot_pr_curve(labels, predictions, save_path=pr_curve_path, title=title)

    best_f1 = 0
    best_threshold = 0
    for threshold in np.arange(0.1, 1.0, 0.01):
        binary_predictions = (predictions >= threshold).astype(int)
        f1 = f1_score(labels, binary_predictions)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold

    binary_predictions = (predictions >= best_threshold).astype(int)
    accuracy = accuracy_score(labels, binary_predictions)
    sensitivity = recall_score(labels, binary_predictions)
    specificity = recall_score(labels, binary_predictions, pos_label=0)
    mcc = matthews_corrcoef(labels, binary_predictions)
    normalized_mcc = (mcc + 1) / 2 * 100  # Normalize MCC to percentage

    print(f'Best F1 Score: {best_f1:.4f}')
    print(f'Accuracy: {accuracy * 100:.2f}%')
    print(f'Sensitivity (Recall): {sensitivity * 100:.2f}%')
    print(f'Specificity: {specificity * 100:.2f}%')
    print(f'MCC: {normalized_mcc:.2f}%')

    metrics = {
        'pr_auc': pr_auc * 100,  # Convert to percentage for consistency
        'best_f1': best_f1 * 100,  # Convert to percentage for consistency
        'accuracy': accuracy * 100,  # Convert to percentage for consistency
        'sensitivity': sensitivity * 100,  # Convert to percentage for consistency
        'specificity': specificity * 100,  # Convert to percentage for consistency
        'mcc': normalized_mcc  # Already normalized
    }
    metrics_path = os.path.join(results_folder, 'metrics_fine_tune.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=4)

    plot_metrics(metrics, results_folder)

    return metrics

def plot_metrics(metrics, results_folder):
    metrics_to_plot = {k: v for k, v in metrics.items() if k != 'best_threshold'}  # Exclude threshold
    fig, ax = plt.subplots()
    ax.bar(metrics_to_plot.keys(), [value for value in metrics_to_plot.values()])
    ax.set_ylabel('Score (%)')
    ax.set_title('Model Performance Metrics')
    ax.set_ylim(0, 100)
    ax.set_yticks(range(0, 101, 10))
    metrics_save_path = os.path.join(results_folder, 'model_performance_metrics_fine_tune.png')
    plt.savefig(metrics_save_path)
    plt.close()

def main():
    model_dir = paths.fine_tune_models_path
    result_dir = paths.fine_tune_results_path
    DATE = '13_09'
    model_name = 'esm2_t6_8M_UR50D'
    models_folder = os.path.join(model_dir, DATE, model_name.split('/')[-1])
    results_folder = os.path.join(result_dir, DATE, model_name.split('/')[-1])
    best_num_epochs = '100'
    best_batch_size = '256'
    architecture = f'{best_num_epochs}_{best_batch_size}'
    architecture_model_folder = os.path.join(models_folder, f'architecture_{architecture}')
    
    models, tokenizers = load_models_and_tokenizers(architecture_model_folder, architecture)
    
    df = pd.read_csv(os.path.join(paths.full_datasets_path, 'cpp924_13_09.csv'))
    datasets_finished = load_datasets(df, tokenizers)
    
    all_predictions, all_labels = predict_with_models(models, datasets_finished)
    
    concatenated_logits = np.concatenate(all_predictions, axis=0)
    concatenated_labels = np.concatenate(all_labels, axis=0)
    predictions = 1 / (1 + np.exp(-concatenated_logits))

    folds_traning_dicts = load_as_pickle(os.path.join(paths.data_for_training_path, DATE, 'folds_traning_dicts.pkl'))
    architecture = "architecture_100_256"
    labels_name = "all_test_labels.npy"
    predictions_name = "all_test_outputs.npy"
    labels_path = os.path.join(paths.fine_tune_results_path, DATE, model_name, architecture, labels_name)
    predictions_path = os.path.join(paths.fine_tune_results_path, DATE, model_name,architecture, predictions_name)
    save_top_false_positives(predictions_path, labels_path, folds_traning_dicts, results_folder)

    metrics = evaluate_predictions(predictions, concatenated_labels, results_folder, model_name)
    print(f"Metrics saved to {os.path.join(results_folder, 'metrics_fine_tune.json')}")

def predict_over_test_set(folds_training_dicts, models, tokenizers,save_path,title, max_length=50):
    all_predictions = []
    all_labels = []
    for i in range (len(folds_training_dicts)):
        fold = folds_training_dicts[i]
        model = models[i]
        tokenizer = tokenizers[i]
        sequences_test = fold['sequences_test']
        labels_test = fold['labels_test']
        dataset = create_dataset(sequences_test, labels_test, tokenizer, max_length=max_length)
        dataloader = DataLoader(dataset, batch_size=len(dataset))

        for batch in dataloader:
            inputs = {k: v.to(model.device) for k, v in batch.items() if k != 'labels'}
            with torch.no_grad():
                outputs = model(**inputs)
            logits = outputs.logits.squeeze().cpu().numpy()
            predictions = 1 / (1 + np.exp(-logits))
            all_predictions.append(predictions)
            all_labels.append(batch['labels'].cpu().numpy())
    
    all_predictions = np.concatenate(all_predictions)
    all_labels = np.concatenate(all_labels)
    print(f'all predictions{all_predictions}')
    plot_pr_curve(all_labels, all_predictions, save_path=save_path, title=title)

    

if __name__ == '__main__':
    # main()
    model_dir = paths.fine_tune_models_path
    result_dir = paths.fine_tune_results_path
    DATE = '13_09'
    model_name = 'esm2_t6_8M_UR50D'
    models_folder = os.path.join(model_dir, DATE, model_name.split('/')[-1])
    results_folder = os.path.join(result_dir, DATE, model_name.split('/')[-1])
    best_num_epochs = '100'
    best_batch_size = '256'
    architecture = f'{best_num_epochs}_{best_batch_size}'
    architecture_model_folder = os.path.join(models_folder, f'architecture_{architecture}')
    folds_training_dicts = load_as_pickle(os.path.join(paths.data_for_training_path, DATE, 'folds_traning_dicts.pkl'))
    
    models, tokenizers = load_models_and_tokenizers(architecture_model_folder, architecture)
    save_path = os.path.join(results_folder, 'pr_curve_model_fine_tune_inference.png')
    title = 'Precision-Recall Curve fine tune inference'
    predict_over_test_set(folds_training_dicts, models, tokenizers, results_folder,save_path, title)