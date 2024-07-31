import os
import sys
import torch
import pandas as pd
from embeddings_creator_utils import load_esm2_model, get_embeddings, esm2_model_names, read_fasta, read_spencer_file, \
    add_source_and_label, concatenate_sequences, save_to_csv
import path


def create_embeddings(df, model_index):
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
    output_file = os.path.join(path.embeddings_path, f'{model_name}_embedding.pt')
    torch.save(sequence_embeddings, output_file)


def create_full_dataset():
    """
    Create a full dataset by combining sequences from multiple sources and saving them to a CSV file.
    """
    # Read sequences from FASTA files
    cpp_natural_sequences = read_fasta(path.cpp_natural_residues_peptides_path)
    peptide_atlas_sequences = read_fasta(path.PeptideAtlas_path_Human_peptides_path)

    # Read sequences from the Spencer file
    spencer_sequences = read_spencer_file(path.SPENCER_peptides_path)

    # Add source and label
    cpp_natural_sequences = add_source_and_label(cpp_natural_sequences, 'CPP_Natural', 1)
    peptide_atlas_sequences = add_source_and_label(peptide_atlas_sequences, 'PeptideAtlas', 0)
    spencer_sequences = add_source_and_label(spencer_sequences, 'Spencer', 0)

    # Concatenate sequences
    full_dataset = concatenate_sequences(cpp_natural_sequences, peptide_atlas_sequences, spencer_sequences)

    # Save to CSV
    output_file = os.path.join(path.datasets_path, 'full_peptide_dataset.csv')
    df = save_to_csv(full_dataset, output_file)

    return df


if __name__ == "__main__":
    model_index = int(sys.argv[1])
    df = create_full_dataset()
    create_embeddings(df)
