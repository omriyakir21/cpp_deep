import os
import sys 
sys.path.append(os.path.join(os.path.dirname(__file__), '..','..'))
import paths
import torch
import numpy as np
from transformers import AutoTokenizer,Trainer
from torch.utils.data import Dataset
from datasets import Dataset as HFDataset
import pandas as pd
from sklearn.metrics import precision_recall_curve, auc, roc_curve
from utils import plot_pr_curve

class CustomDataset(Dataset):
    def __init__(self, data, labels, tokenizer_name):
        if isinstance(data, list):
            data = pd.DataFrame(data, columns=['data'])
        if isinstance(labels, list):
            labels = pd.DataFrame(labels, columns=['labels'])
        
        self.data = data
        self.labels = labels
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data.iloc[idx]['data']  # Get the actual data
        label = int(self.labels.iloc[idx]['labels'])  # Convert label to Python int
        encoded_sample = self.tokenizer(sample, padding='max_length', truncation=True, return_tensors='pt')
        return encoded_sample, label

    def to_hf_dataset(self):
        # Convert the dataset to HuggingFace Dataset for compatibility
        data_dict = {
            'data': self.data['data'].tolist(),
            'labels': [int(label) for label in self.labels['labels'].tolist()]  # Convert labels to Python int
        }
        return HFDataset.from_dict(data_dict)


def calculate_roc_auc(y_true, y_pred):
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    return auc(fpr, tpr)

def precision_recall_auc(y_true, y_pred):
    precision, recall, _ = precision_recall_curve(y_true, y_pred)
    return auc(recall, precision)

def compute_metrics(eval_pred):
    print(f' type eval_pred: {type(eval_pred)}')
    predictions, labels = eval_pred
    predictions = torch.tensor(predictions) 
    print(f' type predictions: {type(predictions)}')
    print(f' type labels: {type(labels)}')
    print(f' predictions: {predictions}')
    with torch.no_grad():
        y_pred = torch.softmax(predictions, dim=-1)[:, 1].cpu().numpy()
    pr_auc = precision_recall_auc(labels, y_pred)
    print(f' precision_recall_auc: {pr_auc}')
    return {"precision_recall_auc": pr_auc}

# Custom evaluation metric
def metrics_evaluation(eval_pred):
    predictions, labels = eval_pred
    y_pred = 1 / (1 + np.exp(-predictions.squeeze()))
    y_true = labels

    precision, recall, _ = precision_recall_curve(y_true, y_pred)
    pr_auc = auc(recall, precision)
    roc_auc = calculate_roc_auc(y_true, y_pred)
    return {"pr_auc": pr_auc, "roc_auc": roc_auc}

def save_test_pr_auc(best_models,folds_traning_dicts,results_folder,tokenizer_name,title):
    # Evaluate the best model on the test set
    all_test_outputs = []
    all_test_labels = []
    for index, best_model in enumerate(best_models):
        fold_train_dict = folds_traning_dicts[index]
        y_test_true = fold_train_dict['labels_test']
        print(f' Fold {index + 1} number of test samples: {len(y_test_true)}')
        test_dataset = CustomDataset(fold_train_dict['sequences_test'], y_test_true, tokenizer_name).to_hf_dataset()
        best_model.eval()
        with torch.no_grad():
            y_test_proba = best_model.predict_proba(test_dataset['data'])[:, 1]
            test_pr_auc = precision_recall_auc(y_test_true, y_test_proba)
            print(f"Fold {index + 1} Test PR AUC: {test_pr_auc:.4f}")

            all_test_outputs.append(y_test_proba)
            all_test_labels.append(y_test_true)
    all_test_outputs = np.concatenate(all_test_outputs)
    all_test_labels = np.concatenate(all_test_labels)
    # Plot the precision-recall curve
    plot_pr_curve(all_test_labels, all_test_outputs, save_path=os.path.join(results_folder, 'pr_curve_model.png'), title=title)

class WeightedTrainer(Trainer):
    def __init__(self, class_weights, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights.to(self.args.device)

    @staticmethod
    def calculate_class_weights(labels):
        # Ensure labels is a PyTorch tensor
        if not isinstance(labels, torch.Tensor):
            labels = torch.tensor(labels)
        # Calculate the number of positives and negatives
        num_negatives = torch.sum(labels == 0).item()  # Count zeros in the tensor
        num_positives = torch.sum(labels == 1).item()  # Count ones in the tensor
        # Compute class weight ratio
        ratio = num_negatives / num_positives
        class_weights = torch.tensor([ratio], dtype=torch.float32)

        return class_weights
    
    def compute_loss(self, model, inputs, return_outputs=False,num_items_in_batch=None):
        labels = inputs.pop("labels").float()
        outputs = model(**inputs)
        logits = outputs.get("logits").squeeze(dim=-1)

        # Apply class weights to BCEWithLogitsLoss
        loss_fct = torch.nn.BCEWithLogitsLoss(pos_weight=self.class_weights)
        loss = loss_fct(logits, labels)

        return (loss, outputs) if return_outputs else loss