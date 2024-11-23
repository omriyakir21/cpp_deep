import os
import sys
import json
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
import torch
import numpy as np
import pandas as pd
from setfit import SetFitModel
from transformers import AutoTokenizer
from utils import plot_pr_curve, load_as_pickle
from models.esm2.ems2_utils import precision_recall_auc, CustomDataset
import paths
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, accuracy_score, recall_score, matthews_corrcoef
from sentence_transformers.losses import CosineSimilarityLoss

def load_models_and_tokenizers(model_folder, loss_class, num_epochs, num_iterations, batch_size, num_models=5):
    models = []
    tokenizers = []
    for i in range(num_models):
        i += 1
        model_path = os.path.join(
            model_folder, f"architecture_CosineSimilarityLoss_{num_epochs}_{num_iterations}_{batch_size}", f"model_{i}"
        )
        print(f"Attempting to load model and tokenizer from: {model_path}")

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model path does not exist: {model_path}")
        else:
            print(f"Contents of model path: {os.listdir(model_path)}")

        try:
            tokenizer_files = ['tokenizer_config.json', 'vocab.txt']
            for tf in tokenizer_files:
                if not os.path.exists(os.path.join(model_path, tf)):
                    raise FileNotFoundError(f"Missing {tf} in {model_path}")

            model = SetFitModel.from_pretrained(model_path)
            tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
            models.append(model)
            tokenizers.append(tokenizer)
        except Exception as e:
            print(f"Failed to load model or tokenizer from {model_path}: {e}")
            raise
    return models, tokenizers

def load_dataset_with_tokenizer(file_path, tokenizer):
    data = pd.read_csv(file_path)
    sequences = data['sequence'].tolist()
    labels = data['label'].tolist()
    return CustomDataset(sequences, labels, tokenizer).to_hf_dataset()

def predict_with_models(models, dataset):
    all_predictions = []
    all_labels = []

    for model in models:
        with torch.no_grad():
            y_pred_proba = model.predict_proba(dataset['data'])[:, 1]  # Get probabilities for the positive class
            y_true = dataset['labels']  # Ground truth labels

            all_predictions.append(y_pred_proba)
            all_labels.append(y_true)

    mean_predictions = np.mean(np.array(all_predictions), axis=0)
    labels = np.array(all_labels[0])  # Assuming labels are the same for all models

    return mean_predictions, labels

def evaluate_predictions(predictions, labels, results_folder, model_name):
    pr_auc = precision_recall_auc(labels, predictions)
    print(f'Precision-Recall AUC: {pr_auc:.4f}')

    pr_curve_path = os.path.join(results_folder, 'pr_curve_model_few_shot.png')
    title = f'{model_name} Precision-Recall Curve'
    plot_pr_curve(labels, predictions, save_path=pr_curve_path, title=title)

    np.save(os.path.join(results_folder, 'predictions_few_shot.npy'), predictions)
    np.save(os.path.join(results_folder, 'labels_few_shot.npy'), labels)

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
    normalized_mcc = (mcc + 1) / 2 * 100  # Normalize MCC to a percentage

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
    metrics_path = os.path.join(results_folder, 'metrics_few_shot.json')
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
    metrics_save_path = os.path.join(results_folder, 'model_performance_metrics_few_shot.png')
    plt.savefig(metrics_save_path)
    plt.close()

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


def main():
    DATE = '13_09'
    model_name = "esm2_t6_8M_UR50D"
    model_folder = os.path.join(
        paths.few_shot_learning_models_path,
        DATE,
        model_name
    )
    print(model_folder)
    results_folder = os.path.join(paths.few_shot_learning_results_path, DATE, model_name)
    test_file_path = os.path.join(paths.full_datasets_path, 'cpp924_13_09.csv')

    if not os.path.exists(results_folder):
        os.makedirs(results_folder)

    loss_class = CosineSimilarityLoss
    num_epochs = 10
    num_iterations = 10
    batch_size = 64

    models, tokenizers = load_models_and_tokenizers(model_folder, loss_class, num_epochs, num_iterations, batch_size)

    datasets = [load_dataset_with_tokenizer(test_file_path, "facebook/esm2_t30_150M_UR50D") for tokenizer in tokenizers]

    predictions, labels = predict_with_models(models, datasets[0])  # Assuming all datasets are identical

    metrics = evaluate_predictions(predictions, labels, results_folder, model_name)
    print(f"Metrics saved to {os.path.join(results_folder, 'metrics_few_shot.json')}")

    folds_traning_dicts = load_as_pickle(os.path.join(paths.data_for_training_path, DATE, 'folds_traning_dicts.pkl'))
    architecture = "architecture_CosineSimilarityLoss_10_10_64"
    labels_name = "all_test_labels.npy"
    predictions_name = "all_test_outputs.npy"
    labels_path = os.path.join(paths.few_shot_learning_results_path, DATE, model_name, architecture, labels_name)
    predictions_path = os.path.join(paths.few_shot_learning_results_path, DATE, model_name,architecture, predictions_name)
    save_top_false_positives(predictions_path, labels_path, folds_traning_dicts, results_folder)


if __name__ == '__main__':
    main()
