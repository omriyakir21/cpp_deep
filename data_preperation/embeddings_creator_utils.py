from transformers import EsmTokenizer, EsmModel
import torch

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


# Function to get embeddings for a list of sequences
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


# Function to read sequences from a FASTA file
def read_fasta(file_path):
    """
    Read sequences from a FASTA file.

    Args:
        file_path (str): Path to the FASTA file.

    Returns:
        list: A list of sequences.
    """
    sequences = []
    with open(file_path, 'r') as file:
        sequence = ""
        for line in file:
            if line.startswith(">"):
                if sequence:
                    sequences.append(sequence)
                    sequence = ""
            else:
                sequence += line.strip()
        if sequence:
            sequences.append(sequence)
    return sequences
