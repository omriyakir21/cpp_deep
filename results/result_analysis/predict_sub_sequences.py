import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__),'..','..'))
import paths
import numpy as np
from sklearn.metrics import precision_recall_curve, auc
from matplotlib import pyplot as plt
import pandas as pd
from models.esm2.lora.predict_lora import load_best_models
from models.esm2.fine_tune.predict_fine_tune import load_models_and_tokenizers as load_models_and_tokenizers_fine_tune
from models.esm2.few_shot_learning.predict_few_shot import load_models_and_tokenizers as load_models_and_tokenizers_few_shot
from models.baselines.convolution_baseline.predict_convolution_baseline import load_models as load_models_convolution
from transformers import EsmForSequenceClassification, EsmTokenizer
from models.baselines.convolution_baseline.convolution_baseline_utils import process_sequences
from models.esm2.fine_tune.fine_tune import create_dataset
from models.esm2.lora.esm2_fine_tune_lora import tokenize_function
from models.esm2.few_shot_learning.few_shot_learning import CustomDataset
import torch
from torch.utils.data import DataLoader
from utils import load_as_pickle
from sentence_transformers.losses import CosineSimilarityLoss
from datasets import Dataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def divide_to_sub_sequences(sequence):
    """
    Generate all consecutive subsequences of a given sequence.

    Parameters:
    sequence (str): The input sequence.

    Returns:
    list: A list of all consecutive subsequences, excluding the empty sequence.
    """
    if not sequence:
        raise ValueError("Input sequence cannot be empty")

    sub_sequences = []
    length = len(sequence)

    # Generate subsequences
    for start in range(length):
        for end in range(start + 1, length + 1):
            sub_sequences.append(sequence[start:end])
    return sub_sequences

def get_sequence_fold(sequence):
    DATE = '13_09'
    data_for_training_dir = os.path.join(paths.data_for_training_path, DATE)
    folds_traning_dicts = load_as_pickle(os.path.join(data_for_training_dir,'folds_traning_dicts.pkl'))
    for i in range(len(folds_traning_dicts)):
        fold_train_dict = folds_traning_dicts[i]
        if sequence in fold_train_dict['sequences_test']:
            return i
    return None

def load_lora_models():
    DATE = '13_09'
    best_batch_size, best_num_epochs, best_r, best_lora_alpha = 128, 50, 100, 32
    model_dir = paths.lora_models_path
    result_dir = paths.lora_results_path
    model_name = 'esm2_t6_8M_UR50D'
    models_folder = os.path.join(model_dir, DATE, model_name.split('/')[-1])
    results_folder = os.path.join(result_dir, DATE, model_name.split('/')[-1])    
    models = load_best_models(best_lora_alpha, best_r, best_batch_size, best_num_epochs, models_folder, model_name)
    tokenizers = [EsmTokenizer.from_pretrained('facebook/esm2_t6_8M_UR50D') for _ in range(5)]
    return models

def load_fine_tune_models():
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
    models, tokenizers = load_models_and_tokenizers_fine_tune(architecture_model_folder, architecture)
    return models

def load_few_shot_models():
    DATE = '13_09'
    model_name = "esm2_t6_8M_UR50D"
    model_folder = os.path.join(paths.few_shot_learning_models_path,DATE, model_name)
    results_folder = os.path.join(paths.few_shot_learning_results_path, DATE, model_name)
    loss_class = CosineSimilarityLoss
    num_epochs = 15
    num_iterations = 15
    batch_size = 64
    models, tokenizers = load_models_and_tokenizers_few_shot(model_folder, loss_class, num_epochs, num_iterations, batch_size)
    return models


def load_convolution_models():
    DATE = '13_09'  # Update this as needed
    model_folder = os.path.join(paths.convolution_baseline_models_path, DATE)
    results_path = os.path.join(paths.convolution_baseline_results_path, DATE)
    filters = 128  # Update according to model used
    kernel_size = 5  # Update according to model used
    num_layers = 2  # Update according to model used
    batch_size = 128
    models = load_models_convolution(model_folder, filters, kernel_size, num_layers, batch_size)
    return models

def create_dataset_convolution(sequences):
    dataset = torch.tensor(process_sequences(sequences), dtype=torch.float32).transpose(1, 2).to(device)
    return dataset

def create_dataset_fine_tune(sequences,tokenizer):
    dataset = create_dataset(sequences, [0 for _ in range(len(sequences))], tokenizer, max_length=50)
    return dataset

def predict_fine_tune(model,dataset):
    dataloader = DataLoader(dataset, batch_size=len(dataset))
    all_predictions = []
    for batch in dataloader:
        inputs = {k: v.to(model.device) for k, v in batch.items() if k != 'labels'}
        with torch.no_grad():
            outputs = model(**inputs)
        logits = outputs.logits.squeeze().cpu().numpy()
        predictions = 1 / (1 + np.exp(-logits))
        all_predictions.append(predictions)
    all_predictions = np.concatenate(all_predictions)
    return all_predictions

def create_dataset_lora(sequences,tokenizer):
    labels = [0 for i in range(len(sequences))]
    dataset = Dataset.from_dict(tokenize_function(tokenizer,sequences, labels))
    return dataset

def predict_with_convolution(model,processed_sequences):
    with torch.no_grad():
        predictions = model(processed_sequences).squeeze(dim=1).cpu().numpy()
    return predictions

def create_dataset_few_shot(sequences,tokenizer_name):
    dataset = CustomDataset(sequences, [0 for _ in range(len(sequences))], tokenizer_name).to_hf_dataset()
    return dataset
def predict_few_shot(model,dataset):
    all_predictions = []
    with torch.no_grad():
        proba = model.predict_proba(dataset['data'])[:, 1]  # Get probabilities for the positive class
        all_predictions.append(proba)
    all_predictions = np.concatenate(all_predictions)
    return all_predictions


if __name__ == '__main__':
    sequence = "HADGTFTNDMTSYLDAKAARDFVSWLARSDKS"
    sub_sequences = divide_to_sub_sequences(sequence)
    fold_index = get_sequence_fold(sequence)
    model_name = 'facebook/esm2_t6_8M_UR50D'
    tokenizer = EsmTokenizer.from_pretrained(model_name)
    if fold_index is None:
        print('The sequence is not in the test set')
        foldindices = [0,1,2,3,4]
    else:
        foldindices = [fold_index]
    lora_dataset = create_dataset_fine_tune(sub_sequences,tokenizer)
    fine_tune_dataset = create_dataset_fine_tune(sub_sequences,tokenizer)
    convolution_sub_sequences = create_dataset_convolution(sub_sequences)
    few_shot_dataset = create_dataset_few_shot(sub_sequences,model_name)

    lora_predictions = []
    fine_tune_predictions = []
    few_shot_predictions = []
    convolution_predictions = []
    for fold_index in foldindices:
        lora_model = load_lora_models()[fold_index]
        lora_predictions.append(predict_fine_tune(lora_model,lora_dataset))

        fine_tune_model = load_fine_tune_models()[fold_index]
        fine_tune_predictions.append(predict_fine_tune(fine_tune_model,fine_tune_dataset))
        
        few_shot_model = load_few_shot_models()[fold_index]
        few_shot_predictions.append(predict_few_shot(few_shot_model,few_shot_dataset))

        convolution_model = load_convolution_models()[fold_index]
        convolution_predictions.append(predict_with_convolution(convolution_model,convolution_sub_sequences))
    
    lora_predictions = np.mean(lora_predictions,axis=0)
    fine_tune_predictions = np.mean(fine_tune_predictions,axis=0)
    few_shot_predictions = np.mean(few_shot_predictions,axis=0)
    convolution_predictions = np.mean(convolution_predictions,axis=0)


    # Create a DataFrame with the required data
    data = {
        'sub_sequence': sub_sequences,
        'convolution_prediction': convolution_predictions.round(3),
        'lora_prediction': lora_predictions.round(3),
        'fine_tune_prediction': fine_tune_predictions.round(3),
        'few_shot_prediction': few_shot_predictions.round(3)
    }
    df = pd.DataFrame(data)

    # Calculate the average prediction
    df['average_prediction'] = df[['convolution_prediction', 'lora_prediction', 'fine_tune_prediction', 'few_shot_prediction']].mean(axis=1).round(3)

    # Sort the DataFrame by average prediction in descending order
    df = df.sort_values(by='average_prediction', ascending=False)

    # Write the sorted DataFrame to a CSV file
    subsequences_folder = os.path.join(paths.result_analysis_path, 'sub_sequences')
    if not os.path.exists(subsequences_folder):
        os.makedirs(subsequences_folder)
    csv_path = os.path.join(subsequences_folder,f'predict_sub_sequences_{sequence}.csv')
    df.to_csv(csv_path, index=False)
    print(f'Saved predictions to {csv_path}')



