import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from transformers import EsmTokenizer, EsmModel
import torch
import pandas as pd
from Bio import SeqIO
from bs4 import BeautifulSoup
import re


# Function to load the ESM2 model and tokenizer
esm2_model_names = ['facebook/esm2_t6_8M_UR50D', 'facebook/esm2_t12_35M_UR50D', 'facebook/esm2_t30_150M_UR50D',
                    'facebook/esm2_t33_650M_UR50D']
MAX_LENGTH = 30
FULL_DATASET_NAME = 'full_peptide_dataset'


def read_BIOPEP_file(file_path):
    
    with open(file_path, 'r', encoding='iso-8859-1') as file:
        content = file.read()
    
    soup = BeautifulSoup(content, 'html.parser')
    
    ids_and_sequences = []
    
    # Find all rows in the table
    rows = soup.find_all('tr', class_='info')
    row_num = 0
    for row in rows:
        # Extract the sequence ID and sequence
        seq_id = f'{str(row_num)}_BIOPEP'
        row_num += 1
        seq = row.find('div', class_='info').get_text(strip=True)
        if seq and not (seq.isspace() or seq == 'NA') and re.match(r'^[A-Z]+$', seq):
            ids_and_sequences.append((seq_id, seq))    
    return ids_and_sequences

def load_esm2_model(model_name="facebook/esm2_t6_8M_UR50D"):
    """
    Load the ESM2 model and tokenizer from Hugging Face.

    Args:
        model_name (str): The name of the ESM2 model to load.

    Returns:
        tokenizer: The tokenizer for the ESM2 model.
        model: The ESM2 model.
    """
    print(f'Loading model {model_name}', flush=True)
    tokenizer = EsmTokenizer.from_pretrained(model_name)
    model = EsmModel.from_pretrained(model_name)
    print(f'Model {model_name} loaded', flush=True)
    return tokenizer, model


def get_embeddings(sequences, tokenizer, model, max_length):
    """
    Get embeddings for a list of protein sequences.

    Args:
        sequences (list): A list of protein sequences.
        tokenizer: The tokenizer for the ESM2 model.
        model: The ESM2 model.
        max_length (int): The maximum length for padding/truncation.

    Returns:
        torch.Tensor: The embeddings for the sequences.
    """
    print(f'Getting embeddings for {len(sequences)} sequences', flush=True)
    inputs = tokenizer(sequences, return_tensors="pt", padding='max_length', truncation=True, max_length=max_length)
    with torch.no_grad():
        print('Running model', flush=True)
        outputs = model(**inputs)
        print('Model run complete', flush=True)
        token_embeddings = outputs.last_hidden_state
    sequence_embeddings = token_embeddings.mean(dim=1)
    return sequence_embeddings


def read_fasta(file_path):
    """
    Read sequences from a FASTA file.

    Args:
        file_path (str): The path to the FASTA file.

    Returns:
        list: A list of tuples containing sequence IDs and sequences.
    """
    ids_and_sequences = []
    for record in SeqIO.parse(file_path, "fasta"):
        ids_and_sequences.append((record.id, str(record.seq)))
    return ids_and_sequences


def add_source_and_label(ids_and_sequences, source, label):
    """
    Add source and label to each sequence.

    Args:
        ids_and_sequences (list): A list of tuples containing sequence IDs and sequences.
        source (str): The source of the sequences.
        label (int): The label for the sequences.

    Returns:
        list: A list of tuples containing sequence IDs, sequences, source, and label.
    """
    return [(seq_id, seq, source, label) for seq_id, seq in ids_and_sequences]


def concatenate_sequences(*sequence_lists):
    """
    Concatenate multiple lists of sequences.

    Args:
        *sequence_lists: Multiple lists of sequences.

    Returns:
        list: A concatenated list of sequences.
    """
    concatenated = []
    for seq_list in sequence_lists:
        concatenated.extend(seq_list)
    return concatenated


def save_to_csv(sequences, output_file):
    """
    Save sequences to a CSV file.

    Args:
        sequences (list): A list of sequences.
        output_file (str): The path to the output CSV file.
    """
    df = pd.DataFrame(sequences, columns=['id', 'sequence', 'source', 'label'])
    df.to_csv(output_file, index=False)
    return df


def read_spencer_file(file_path):
    """
    Read sequences from the Spencer file.

    Args:
        file_path (str): The path to the Spencer file.

    Returns:
        list: A list of tuples containing sequence IDs and sequences.
    """
    ids_and_sequences = []
    with open(file_path, 'r') as file:
        for line in file:
            parts = line.strip().split()
            if len(parts) >= 2:
                seq_id = parts[0]
                seq = parts[1]
                ids_and_sequences.append((seq_id, seq))
    return ids_and_sequences

def remove_long_sequences(sequences):
    """
    Args:
        sequences (list): A list of tuples containing sequence IDs, sequences, source, and label.
    
    Returns:
        list: A list of tuples with sequences longer than 50 amino acids removed.
    """
    return [seq for seq in sequences if len(seq[1]) <= 50]

def remove_duplicates(sequences):
    """
    Remove duplicate sequences, keeping only those with label 1 if a sequence has both label 0 and label 1.

    Args:
        sequences (list): A list of tuples containing sequence IDs, sequences, source, and label.

    Returns:
        list: A list of tuples with duplicates removed.
    """
    sequence_dict = {}
    for seq_id, seq, source, label in sequences:
        if seq not in sequence_dict:
            sequence_dict[seq] = (seq_id, seq, source, label)
        else:
            # If the sequence already exists and the current label is 1, replace it
            if label == 1:
                sequence_dict[seq] = (seq_id, seq, source, label)

    return list(sequence_dict.values())


def calculate_max_length(sequences):
    """
    Calculate the maximum sequence length from a list of sequences.

    Args:
        sequences (list): A list of sequences.

    Returns:
        int: The maximum sequence length.
    """
    return max(len(seq) for seq in sequences)




