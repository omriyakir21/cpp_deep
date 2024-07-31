import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import paths
import pandas as pd
import torch
from datetime import datetime
from embeddings_creator_utils import load_esm2_model, get_embeddings, esm2_model_names, read_fasta, read_spencer_file, \
    add_source_and_label, concatenate_sequences, save_to_csv,remove_duplicates



def create_embeddings_esm2(df, model_index):
    """
    Create embeddings for sequences in the DataFrame using the specified ESM2 model.

    Args:
        df (pandas.DataFrame): DataFrame containing sequences.
        model_index (int): Index of the ESM2 model to use.

    Raises:
        ValueError: If model index is not provided or invalid.
    """
    # Get the model index from command-line arguments
    if len(sys.argv) < 2:
        raise ValueError("Model index not provided. Usage: python embeddings_creator.py <model_index>")

    # Validate model index
    if model_index < 0 or model_index >= len(esm2_model_names):
        raise ValueError("Invalid model index")

    # Get the model name
    model_name = esm2_model_names[model_index]

    # Load the ESM2 model and tokenizer
    tokenizer, model = load_esm2_model(model_name)

    # Extract sequences from the DataFrame
    sequences = df['sequence'].tolist()

    # Get embeddings for the sequences
    sequence_embeddings = get_embeddings(sequences, tokenizer, model)

    # Print the shape of the embeddings
    print(sequence_embeddings.shape)

    # Save the embeddings to a file
    current_date = datetime.now().strftime("%d_%m")
    output_file = os.path.join(paths.esm2_embeddings_path, f'{model_name}_embedding_{current_date}.pt')
    torch.save(sequence_embeddings, output_file)


def create_full_dataset():
    """
    Create a full dataset by combining sequences from multiple sources, removing duplicates, and saving them to a CSV file.

    Returns:
        pandas.DataFrame: DataFrame containing the full dataset.
    """
    # Read sequences from FASTA files
    cpp_natural_sequences = read_fasta(paths.cpp_natural_residues_peptides_path)
    peptide_atlas_sequences = read_fasta(paths.PeptideAtlas_path_Human_peptides_path)

    # Read sequences from the Spencer file
    spencer_sequences = read_spencer_file(paths.SPENCER_peptides_path)

    # Add source and label
    cpp_natural_sequences = add_source_and_label(cpp_natural_sequences, 'CPP_Natural', 1)
    peptide_atlas_sequences = add_source_and_label(peptide_atlas_sequences, 'PeptideAtlas', 0)
    spencer_sequences = add_source_and_label(spencer_sequences, 'Spencer', 0)

    # Concatenate sequences
    full_dataset = concatenate_sequences(cpp_natural_sequences, peptide_atlas_sequences, spencer_sequences)

    # Remove duplicates
    full_dataset = remove_duplicates(full_dataset)

    # Save to CSV
    current_date = datetime.now().strftime("%d_%m")
    output_file = os.path.join(paths.full_datasets_path, f'full_peptide_dataset_{current_date}.csv')
    df = save_to_csv(full_dataset, output_file)

    return df


if __name__ == "__main__":
    # Get the model index from command-line arguments
    model_index = int(sys.argv[1])

    # Create the full dataset
    df = create_full_dataset()

    # Create embeddings for the sequences in the dataset
    # Load the dataset from the CSV file
    # DATE = '31_07'
    # dataset_file_path = os.path.join(paths.full_datasets_path, f'full_peptide_dataset_{DATE}.csv')
    # df = pd.read_csv(dataset_file_path)
    # create_embeddings_esm2(df, model_index)
