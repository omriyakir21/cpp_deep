import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import paths
import pandas as pd
import torch
from datetime import datetime
from data_preperation.dataset_creator_utils import load_esm2_model, get_embeddings, esm2_model_names, read_fasta, \
    add_source_and_label, concatenate_sequences, save_to_csv, remove_duplicates, \
    calculate_max_length,FULL_DATASET_NAME,read_BIOPEP_file, remove_long_sequences \
        ,save_sequences_descriptions_BIOPEP_to_csv,read_Hmrbase2_files,read_NeuroPep_file \
        ,read_SmProt2
        

def create_embeddings_esm2(df,DATE):
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

    # Get the model index from command-line arguments
    model_index = int(sys.argv[1])

    # Validate model index
    if model_index < 0 or model_index >= len(esm2_model_names):
        raise ValueError("Invalid model index")

    # Get the model name
    model_name = esm2_model_names[model_index]

    # Load the ESM2 model and tokenizer
    tokenizer, model = load_esm2_model(model_name)

    # Extract sequences from the DataFrame
    sequences = df['sequence'].tolist()

    # Calculate and update MAX_LENGTH
    MAX_LENGTH = calculate_max_length(sequences)
    print(f'max length is {MAX_LENGTH}')

    # Get embeddings for the sequences
    sequence_embeddings = get_embeddings(sequences, tokenizer, model, MAX_LENGTH)

    # Print the shape of the embeddings
    print(sequence_embeddings.shape)

    embeddings_dir = os.path.join(paths.esm2_embeddings_path,DATE)
    if not os.path.exists(embeddings_dir):
        os.makedirs(embeddings_dir)
    # Save the embeddings to a file
    output_file = os.path.join(embeddings_dir, f'{model_name.split("/")[1]}_embedding_{DATE}.pt')
    torch.save(sequence_embeddings, output_file)


def create_full_dataset(date):
    """
    Create a full dataset by combining sequences from multiple sources, removing duplicates, and saving them to a CSV file.

    Returns:
        pandas.DataFrame: DataFrame containing the full dataset.
    """
    # Read sequences from FASTA files
    cpp_natural_ids_and_sequences = read_fasta(paths.cpp_natural_residues_peptides_path)
    # peptide_atlas_sequences = read_fasta(paths.PeptideAtlas_path_Human_peptides_path)

    # # Read sequences from the Spencer file
    # spencer_sequences = read_spencer_file(paths.SPENCER_peptides_path)

    # Read sequences from the BIOPRP file
    BIOPEP_ids_sequences_descriptions = read_BIOPEP_file(os.path.join(paths.BIOPEP_UWMix_path, 'BIOPEP_UWMix.html'))
    Hmbrase2_id_sequences_descriptions = read_Hmrbase2_files(paths.Hmrbase2_path)
    neuropep1_ids_sequences_descriptions = read_NeuroPep_file(os.path.join(paths.NeuroPep1_path,"NeuroPep1.txt"))
    SmProt2_id_sequences_descriptions = read_SmProt2(paths.SmProt2_path)
    # Add source and label
    cpp_natural_ids_and_sequences = add_source_and_label(cpp_natural_ids_and_sequences, 'CPP_Natural', 1)
    BIOPEP_ids_sequences_descriptions = add_source_and_label(BIOPEP_ids_sequences_descriptions,'BIOPEP_UWMix',0)
    Hmbrase2_id_sequences_descriptions = add_source_and_label(Hmbrase2_id_sequences_descriptions, 'Hmrbase2', 0)
    neuropep1_ids_sequences_descriptions = add_source_and_label(neuropep1_ids_sequences_descriptions, 'NeuroPep1', 0)
    SmProt2_id_sequences_descriptions = add_source_and_label(SmProt2_id_sequences_descriptions, 'SmProt2', 0)
    # # peptide_atlas_sequences = add_source_and_label(peptide_atlas_sequences, 'PeptideAtlas', 0)
    # spencer_sequences = add_source_and_label(spencer_sequences, 'Spencer', 0)

    # Concatenate sequences
    # full_dataset = concatenate_sequences(cpp_natural_sequences, peptide_atlas_sequences, spencer_sequences)
    full_dataset = concatenate_sequences(cpp_natural_ids_and_sequences, BIOPEP_ids_sequences_descriptions, Hmbrase2_id_sequences_descriptions,neuropep1_ids_sequences_descriptions,SmProt2_id_sequences_descriptions)

    # Remove duplicates
    full_dataset = remove_duplicates(full_dataset)

    full_dataset = remove_long_sequences(full_dataset)
    
    # Save to CSV
    output_file = os.path.join(paths.full_datasets_path, f'{FULL_DATASET_NAME}_{date}.csv')

    df = save_to_csv(full_dataset, output_file)

    return df


if __name__ == "__main__":
    # Create the full dataset
    DATE = datetime.now().strftime("%d_%m")
    DATE = '13_02'
    df = create_full_dataset(DATE)
    
    # Create embeddings for the sequences in the dataset
    # Load the dataset from the CSV file
    
    dataset_file_path = os.path.join(paths.full_datasets_path, f'{FULL_DATASET_NAME}_{DATE}.csv')
    df = pd.read_csv(dataset_file_path)

    # create_embeddings_esm2(df,DATE)
    # csv_path = os.path.join(paths.BIOPEP_UWMix_path, 'BIOPEP_sequences_descriptions.csv')
    # save_sequences_descriptions_BIOPEP_to_csv(csv_path)
