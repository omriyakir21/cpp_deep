import csv
import os
import sys
import pandas as pd 
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from paths import datasets_sources_path, full_datasets_path  # Import paths
import paths
import torch
import numpy as np
from utils import save_as_pickle, load_as_pickle, load_labels, load_sequences, load_ids


def process_neuropep1(input_file, output_file):
    df = pd.read_csv(input_file, sep='\t')
    df.rename(columns={"ID": "id", "Sequence": "sequence"}, inplace=True)
    df['source'] = 'NeuroPep1'
    df['label'] = 0
    df.to_csv(output_file, index=False)
def update_csv_with_folds_index(csv_file, folds_training_dicts):
    """
    Adds a 'folds_index_to_predict' column to the CSV based on test datasets.

    Args:
        csv_file (str): Path to the CSV file.
        folds_training_dicts (list): List of dictionaries containing fold data.
    """
    # Load the CSV into a DataFrame
    df = pd.read_csv(csv_file)
    # Add the new column if it doesn't exist
    if 'folds_index_to_predict' not in df.columns:
        df['folds_index_to_predict'] = 'None'  # Initialize with 'None'

    # Iterate over the folds to get their test datasets
    for fold_index, fold_dict in enumerate(folds_training_dicts):
        print(f"Processing fold {fold_index + 1}")

        # Get the sequences for the current fold's test dataset
        test_sequences = fold_dict['sequences_test']

        # Check each sequence in the test dataset against the CSV
        for sequence in test_sequences:
            # Find matching rows in the DataFrame by sequence
            matching_rows = df[df['sequence'] == sequence]

            # Update the fold index if a match is found
            if not matching_rows.empty:
                df.loc[matching_rows.index, 'folds_index_to_predict'] = fold_index
                print(f"Updated sequence '{sequence}' with fold index {fold_index}")

    # Assign 'None' to any rows that still have missing values
    df['folds_index_to_predict'].fillna('None', inplace=True)

    # Save the updated DataFrame back to the CSV
    df.to_csv(csv_file, index=False)
    print(f"CSV updated with 'folds_index_to_predict' column at: {csv_file}")





if __name__ == "__main__":
    DATE = '13_09'
    # Construct the paths using paths.py
    input_path = os.path.join(paths.NeuroPep1_path, 'NeuroPep1.txt')
    output_path = os.path.join(full_datasets_path, f'NeuroPep1_{DATE}.csv')

    # Call the processing function
    process_neuropep1(input_path, output_path)


    folds_training_dicts = load_as_pickle(
        os.path.join(paths.data_for_training_path, DATE, 'folds_traning_dicts.pkl')
    )
    
    # Define the path to the existing CSV
    csv_path = os.path.join(full_datasets_path,  f'NeuroPep1_{DATE}.csv')

    # Update the CSV with the fold index
    update_csv_with_folds_index(csv_path, folds_training_dicts)

