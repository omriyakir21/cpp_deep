import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__),'..','..'))
import paths
import numpy as np
from sklearn.metrics import precision_recall_curve, auc
from matplotlib import pyplot as plt

def plot_pr_curves_multiple(data_dict, title='Precision-Recall Curves', save_path=None):
    """
    Plot multiple precision-recall curves on the same graph.

    Args:
        data_dict (dict): Dictionary where keys are model names and values are tuples of (y_true, y_scores).
        title (str): Title of the plot.
        save_path (str): Path to save the plot. If None, the plot is shown.
    """
    plt.figure()
    for model_name, (y_true, y_scores) in data_dict.items():
        precision, recall, _ = precision_recall_curve(y_true, y_scores)
        pr_auc = auc(recall, precision)
        plt.plot(recall, precision, label=f'{model_name} (PRAUC = {pr_auc:.2f})')

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(title)
    plt.legend(loc='best')
    if save_path:
        plt.savefig(save_path)
        print(f"Plot saved at {save_path}")
    else:
        plt.show()

if __name__ == '__main__':
    # Define model names and paths to your saved numpy files
    fine_tune_model_name = "Fine Tune"
    fine_tune_model = "esm2_t6_8M_UR50D"
    few_shot_model_name = "Few Shot"
    lora_model_name = "LoRA"
    baseline_model_name = "Convolution Baseline"
    baseline_model = "convolution_baseline"

    fine_tune_architecture = "architecture_100_256"
    all_labels = "all_test_labels.npy"
    all_outputs = "all_test_outputs.npy"
    Date = "13_09"

    lora_nodel_name = "LoRA"
    lora_architecture = "architecture_128_50_100_32"

    few_shot_model_name = "Few Shot"
    few_shot_architecture = "architecture_CosineSimilarityLoss_10_10_64"

    # Create a dictionary to hold data paths for each model
    data_paths = {
        fine_tune_model_name: (
            os.path.join(paths.fine_tune_results_path, Date, fine_tune_model, fine_tune_architecture, all_labels),
            os.path.join(paths.fine_tune_results_path, Date, fine_tune_model, fine_tune_architecture, all_outputs)
        ),
        few_shot_model_name: (
            os.path.join(paths.few_shot_learning_results_path, Date, fine_tune_model, few_shot_architecture, all_labels),
            os.path.join(paths.few_shot_learning_results_path, Date, fine_tune_model, few_shot_architecture, all_outputs)
        ),
        lora_model_name: (
            os.path.join(paths.lora_results_path, Date, fine_tune_model, lora_architecture, all_labels),
            os.path.join(paths.lora_results_path, Date, fine_tune_model, lora_architecture, all_outputs)
        ),
        baseline_model_name: (
            os.path.join(paths.baselines_results_path,baseline_model, Date, all_labels),
            os.path.join(paths.baselines_results_path, baseline_model, Date, all_outputs)
        ),
    }

    # Load data and prepare for plotting
    data_dict = {}
    for model_name, (labels_path, outputs_path) in data_paths.items():
        y_true = np.load(labels_path)
        y_scores = np.load(outputs_path)
        data_dict[model_name] = (y_true, y_scores)

    # Save the plot
    script_dir = os.path.dirname(os.path.abspath(__file__))
    save_path = os.path.join(script_dir, "pr_curves_multiple.png")

    plot_pr_curves_multiple(data_dict, title="Precision-Recall Curves Comparison", save_path=save_path)
