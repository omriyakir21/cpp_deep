import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..','..','..'))
import torch
import numpy as np
import torch.nn.functional as F
from transformers import AutoTokenizer, EsmForSequenceClassification, TrainingArguments, Trainer
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import precision_recall_curve, auc
from datasets import Dataset
from models.baselines.convolution_baseline.convolution_baseline import save_grid_search_results
from utils import load_as_pickle, plot_pr_curve
from models.esm2.ems2_utils import precision_recall_auc
import paths

# Custom Dataset class to include input_ids and attention_mask
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, sequences, labels, model_name, tokenizer=None, max_length=512):
        self.sequences = sequences
        self.labels = labels
        self.tokenizer = tokenizer if tokenizer else AutoTokenizer.from_pretrained(model_name)
        self.max_length = max_length

    def __len__(self):
        # Return the number of data samples
        return len(self.labels)

    def __getitem__(self, idx):
        # Return the corresponding input_ids, attention_mask, and label for a given index
        item = {key: torch.tensor(val[idx]) for key, val in self.tokenizer(self.sequences, truncation=True, padding=True, max_length=self.max_length).items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def to_hf_dataset(self):
        # Tokenize the sequences and return a Hugging Face Dataset object
        encodings = self.tokenizer(self.sequences, truncation=True, padding=True, max_length=self.max_length)
        encodings['labels'] = self.labels  # Ensure labels are added to the dictionary
        # Convert to a Hugging Face Dataset
        return Dataset.from_dict(encodings)

# Custom Trainer to include class weights in loss function
class WeightedTrainer(Trainer):
    def __init__(self, class_weights, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights.to(self.args.device)  # Move weights to the same device as the model

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")

        # Apply class weights to CrossEntropyLoss
        loss_fct = torch.nn.CrossEntropyLoss(weight=self.class_weights)
        loss = loss_fct(logits, labels)

        return (loss, outputs) if return_outputs else loss

def precision_recall_auc_for_eval(eval_pred):
    predictions, labels = eval_pred
    print(f'predictions : {predictions}')
    print(f' predictions dim = {predictions.shape}')

    # Get the probabilities by applying softmax (if needed)
    probabilities = torch.softmax(torch.tensor(predictions), dim=-1)
    y_pred = probabilities[:, 1].numpy()  # Assuming binary classification, we take the probability of the positive class
    y_true = labels

    precision, recall, _ = precision_recall_curve(y_true, y_pred)
    pr_auc = auc(recall, precision)
    
    return {"pr_auc": pr_auc}

def save_best_models(best_models, grid_results, best_architecture_index, models_folder):
    # Save the best models
    best_num_epochs = grid_results[best_architecture_index]['num_epochs']
    best_batch_size = grid_results[best_architecture_index]['batch_size']
    best_architecture_pr_auc = grid_results[best_architecture_index]['val_pr_auc']

    print(f'Best model: Num epochs: {best_num_epochs}, '
          f'Batch size: {best_batch_size}, val_pr_auc: {best_architecture_pr_auc}')
    for index, best_model in enumerate(best_models):
        best_model.save_pretrained(os.path.join(models_folder, 
                                                f'model_{best_num_epochs}_{best_batch_size}_{index}'))

def train_architecture_over_folds(batch_size, num_epochs, folds_training_dicts):
    architecture_pr_aucs = []
    best_models = []

    # Early stopping variables
    patience = 5  # Number of epochs to wait after last improvement

    # Iterate through each fold in the training data
    for fold_index, fold_dict in enumerate(folds_training_dicts):
        # Compute class weights
        class_weights = compute_class_weight('balanced',
                                             classes=np.unique(fold_dict['labels_train']),
                                             y=fold_dict['labels_train'])
        class_weights = torch.tensor(class_weights, dtype=torch.float32, device=device)

        best_model = None
        best_pr_auc = 0
        epochs_no_improve = 0

        print(f"Training fold {fold_index + 1}/{len(folds_training_dicts)}")
        model_name = "facebook/esm2_t6_8M_UR50D"
        train_dataset = CustomDataset(fold_dict['sequences_train'], fold_dict['labels_train'], model_name).to_hf_dataset()
        eval_dataset = CustomDataset(fold_dict['sequences_validation'], fold_dict['labels_validation'], model_name).to_hf_dataset()

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = EsmForSequenceClassification.from_pretrained(model_name, num_labels=2)
        model.to(device)

        # Define TrainingArguments for this fold
        training_args = TrainingArguments(
            output_dir=f'output/{model_name}',  # Directory to save model checkpoints
            learning_rate=1e-3,
            per_device_train_batch_size=batch_size,
            num_train_epochs=num_epochs,
            weight_decay=0.01,
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
        )

        # Create a WeightedTrainer
        trainer = WeightedTrainer(
            class_weights=class_weights,  # Pass the class weights to the trainer
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            compute_metrics=precision_recall_auc_for_eval,
        )

        # Custom training loop with early stopping
        trainer.train()

        with torch.no_grad():
            # Extract input tensors from the eval dataset
            input_ids = torch.tensor(eval_dataset['input_ids']).to(device)
            attention_mask = torch.tensor(eval_dataset['attention_mask']).to(device)
            labels = torch.tensor(eval_dataset['labels']).to(device)

            # Forward pass
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits

            # Compute softmax probabilities from logits
            probabilities = torch.softmax(logits, dim=-1)
            y_val_pred = probabilities[:, 1].cpu().numpy()  # Get probability of the positive class
            y_val_true = eval_dataset['labels']

            # Calculate Precision-Recall AUC
            pr_auc = precision_recall_auc(y_val_true, y_val_pred)
            print(f"Validation PR AUC at epoch {num_epochs}: {pr_auc:.4f}")

            # Check if the model has improved
            best_pr_auc = pr_auc
            best_model = model
            print(f"New best PR AUC: {pr_auc:.4f}. Model saved.")

        best_models.append(best_model)
        architecture_pr_aucs.append(best_pr_auc)

    return architecture_pr_aucs, best_models


def save_test_pr_auc(best_models, folds_training_dicts, results_folder, tokenizer_name, title):
    all_test_outputs = []
    all_test_labels = []

    for index, best_model in enumerate(best_models):
        fold_train_dict = folds_training_dicts[index]
        y_test_true = fold_train_dict['labels_test']
        print(f'Fold {index + 1} number of test samples: {len(y_test_true)}')

        # Prepare the test dataset
        test_dataset = CustomDataset(fold_train_dict['sequences_test'], y_test_true, tokenizer_name).to_hf_dataset()

        # Prepare inputs and labels as tensors
        input_ids = torch.tensor(test_dataset["input_ids"]).to(best_model.device)
        attention_mask = torch.tensor(test_dataset["attention_mask"]).to(best_model.device)
        labels = torch.tensor(test_dataset["labels"]).to(best_model.device)

        best_model.eval()
        with torch.no_grad():
            # Forward pass on the entire test set
            outputs = best_model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits

            # Apply softmax to get the probabilities
            probabilities = torch.softmax(logits, dim=-1)
            y_test_proba = probabilities[:, 1].cpu().numpy()

        # Convert `y_test_true` to a numpy array if it's a tensor
        y_test_true = y_test_true.cpu().numpy() if isinstance(y_test_true, torch.Tensor) else y_test_true

        # Ensure that `y_test_proba` and `y_test_true` have the same length and compute the test PR AUC
        test_pr_auc = precision_recall_auc(y_test_true, y_test_proba)
        print(f"Fold {index + 1} Test PR AUC: {test_pr_auc:.4f}")

        all_test_outputs.append(y_test_proba)
        all_test_labels.append(y_test_true)

    # Combine all test results
    all_test_outputs = np.concatenate(all_test_outputs)
    all_test_labels = np.concatenate(all_test_labels)

    # Plot the precision-recall curve
    plot_pr_curve(all_test_labels, all_test_outputs, save_path=os.path.join(results_folder, 'pr_curve_model.png'), title=title)


if __name__ == '__main__':
    DATE = '13_09'
    model_name = "esm2_t6_8M_UR50D"
    models_folder = os.path.join(paths.fine_tune_models_path, DATE, model_name)
    if not os.path.exists(models_folder):
        os.makedirs(models_folder)
    results_folder = os.path.join(paths.fine_tune_results_path, DATE, model_name)
    if not os.path.exists(results_folder):
        os.makedirs(results_folder)
    model_name = "facebook/esm2_t6_8M_UR50D"

    folds_traning_dicts = load_as_pickle(os.path.join(paths.data_for_training_path, DATE, 'folds_traning_dicts.pkl'))[:1]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Define the grid search parameters
    param_grid = {
        'num_epochs': [30],
        'batch_size': [64, 128, 256],
    }

    best_model = None
    best_architecture_pr_auc = 0
    best_architecture_index = 0
    grid_results = []
    cnt = 0
    
    # Iterate over grid parameters
    for num_epochs in param_grid['num_epochs']:
        for batch_size in param_grid['batch_size']:
            architecture_pr_aucs, architecture_models = train_architecture_over_folds(
                batch_size, num_epochs, folds_traning_dicts
            )
            architecture_pr_auc = np.mean(architecture_pr_aucs)
            
            # Save the grid search results
            grid_results.append({
                'num_epochs': num_epochs,
                'batch_size': batch_size,
                'val_pr_auc': architecture_pr_auc
            })

            # Save the best models
            if architecture_pr_auc > best_architecture_pr_auc:
                best_architecture_pr_auc = architecture_pr_auc
                best_models = architecture_models
                best_architecture_index = cnt
            cnt += 1

    # Save best models and results
    save_best_models(best_models, grid_results, best_architecture_index, models_folder)
    tokenizer_name = "facebook/esm2_t6_8M_UR50D"
    save_test_pr_auc(best_models, folds_traning_dicts, results_folder, tokenizer_name, 'esm2 fine tune Precision-Recall Curve')
    save_grid_search_results(grid_results, results_folder)
