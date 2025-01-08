import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
import numpy as np
import torch
import pandas as pd
from models.baselines.convolution_baseline.convolution_baseline_utils import process_sequences, CNNModel, calculate_pr_auc
from utils import plot_pr_curve, load_as_pickle
import paths
import json
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, accuracy_score, recall_score, matthews_corrcoef

def load_models(model_folder, filters, kernel_size, num_layers, batch_size, num_models=5):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    models = []
    for i in range(num_models):
        model_path = os.path.join(model_folder, f'model_{filters}_{kernel_size}_{num_layers}_{batch_size}_{i}.pt')
        model = CNNModel(filters=filters, kernel_size=kernel_size, num_layers=num_layers).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        models.append(model)
    return models

def predict_with_models(models, sequences):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    processed_sequences = torch.tensor(process_sequences(sequences), dtype=torch.float32).transpose(1, 2).to(device)
    all_predictions = []
    for model in models:
        with torch.no_grad():
            predictions = model(processed_sequences).squeeze(dim=1).cpu().numpy()
            all_predictions.append(predictions)
    mean_predictions = np.mean(np.array(all_predictions), axis=0)
    return mean_predictions

def load_test_data(file_path):
    data = pd.read_csv(file_path)
    sequences = data['sequence'].tolist()
    labels = data['label'].tolist()
    return sequences, labels, data

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


def evaluate_predictions(predictions, labels, results_path):
    pr_auc = calculate_pr_auc(np.array(labels), predictions)
    print(f'Precision-Recall AUC: {pr_auc}')

    # Save PR curve
    plot_pr_curve(
        y_true=np.array(labels),
        y_scores=predictions,
        save_path=os.path.join(results_path, 'pr_curve_ensemble.png'),
        title='Ensemble Precision-Recall Curve'
    )

    # Save predictions and labels
    np.save(os.path.join(results_path, 'predictions_ensemble.npy'), predictions)
    np.save(os.path.join(results_path, 'labels_ensemble.npy'), labels)

    # Evaluate metrics
    best_f1 = 0
    best_threshold = 0
    for threshold in np.arange(0.1, 1.0, 0.01):
        binary_predictions = (predictions >= threshold).astype(int)
        f1 = f1_score(labels, binary_predictions)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold

    # Print the best threshold
    print(f'Best Threshold: {best_threshold:.2f}')

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
        'pr_auc': pr_auc,
        'best_f1': best_f1,
        'accuracy': accuracy,
        'sensitivity': sensitivity,
        'specificity': specificity,
        'mcc': normalized_mcc
    }

    metrics_path = os.path.join(results_path, 'metrics_ensemble.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=4)

    plot_metrics(metrics, results_path)

    return metrics

def plot_metrics(metrics, results_path):
    metrics_to_plot = {k: v for k, v in metrics.items() if k != 'best_threshold'}  # Exclude threshold
    fig, ax = plt.subplots()
    
    # Ensure only non-percentage values are scaled
    scaled_metrics = {
        k: (v * 100 if k != 'mcc' else v) for k, v in metrics_to_plot.items()
    }
    
    ax.bar(scaled_metrics.keys(), scaled_metrics.values())
    ax.set_ylabel('Score (%)')
    ax.set_title('Model Performance Metrics')
    ax.set_ylim(0, 100)
    ax.set_yticks(range(0, 101, 10))
    metrics_save_path = os.path.join(results_path, 'model_performance_metrics_ensemble.png')
    plt.savefig(metrics_save_path)
    plt.close()


if __name__ == "__main__":
    # Define paths
    DATE = '13_09'  # Update this as needed
    model_folder = os.path.join(paths.convolution_baseline_models_path, DATE)
    # test_file_path = os.path.join(paths.full_datasets_path, f'cpp924_{DATE}.csv')
    results_path = os.path.join(paths.convolution_baseline_results_path, DATE)

    if not os.path.exists(results_path):
        os.makedirs(results_path)

    # Load test data
    # sequences, labels, data = load_test_data(test_file_path)

    # Load all models (5 in total)
    filters = 128  # Update according to model used
    kernel_size = 5  # Update according to model used
    num_layers = 2  # Update according to model used
    batch_size = 128
    models = load_models(model_folder, filters, kernel_size, num_layers, batch_size)

    # # Predict using ensemble of models
    # predictions = predict_with_models(models, sequences)
    # predictions = np.round(predictions, 2)
    # print(predictions)

    # # Save top 10 false positives
    folds_traning_dicts = load_as_pickle(os.path.join(paths.data_for_training_path, DATE, 'folds_traning_dicts.pkl'))
    # architecture = "architecture_CosineSimilarityLoss_10_10_64"
    # labels_name = "all_test_labels.npy"
    # predictions_name = "all_test_outputs.npy"
    # labels_path = os.path.join(paths.convolution_baseline_results_path, DATE, labels_name)
    # predictions_path = os.path.join(paths.convolution_baseline_results_path, DATE, predictions_name)
    # save_top_false_positives(predictions_path, labels_path, folds_traning_dicts, results_path)

    # # Evaluate predictions
    # evaluate_predictions(predictions, labels, results_path)
    # for fold in folds_training_dicts
