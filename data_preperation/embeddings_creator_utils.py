from transformers import EsmTokenizer, EsmModel
import torch
import pandas as pd
from Bio import SeqIO

# Function to load the ESM2 model and tokenizer
esm2_model_names = ['facebook/esm2_t6_8M_UR50D', 'facebook/esm2_t12_35M_UR50D', 'facebook/esm2_t30_150M_UR50D',
                    'facebook/esm2_t33_650M_UR50D']


def load_esm2_model(model_name="facebook/esm2_t6_8M_UR50D"):
    """
    Load the ESM2 model and tokenizer from Hugging Face.

    Args:
        model_name (str): The name of the ESM2 model to load.

    Returns:
        tokenizer: The tokenizer for the ESM2 model.
        model: The ESM2 model.
    """
    tokenizer = EsmTokenizer.from_pretrained(model_name)
    model = EsmModel.from_pretrained(model_name)
    return tokenizer, model

def get_embeddings(sequences, tokenizer, model):
    """
    Get embeddings for a list of protein sequences.

    Args:
        sequences (list): A list of protein sequences.
        tokenizer: The tokenizer for the ESM2 model.
        model: The ESM2 model.

    Returns:
        torch.Tensor: The embeddings for the sequences.
    """
    max_length = 30
    inputs = tokenizer(sequences, return_tensors="pt", padding=max_length)
    with torch.no_grad():
        outputs = model(**inputs)
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
