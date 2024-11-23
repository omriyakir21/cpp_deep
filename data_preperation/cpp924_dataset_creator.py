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


def process_cpp924(input_file, output_file):
    """
    Reads the cpp924.fa file, processes it into pairs of lines, extracts relevant
    information, and saves the results as a CSV file.

    Args:
        input_file (str): Path to the input .fa file.
        output_file (str): Path where the output CSV file will be saved.
    """
    data = []

    # Read the input file and process in pairs
    with open(input_file, 'r') as f:
        lines = f.readlines()

    # Process each pair of lines
    for i in range(0, len(lines), 2):
        id_label_line = lines[i].strip()
        sequence = lines[i + 1].strip()

        # Extract ID and label from the first line
        id_part, label = id_label_line[1:].split('|')
        item_id = f"cpp_294-{id_part}"

        # Add row to data list
        data.append([item_id, sequence, "CPP_924", label])

    # Write the processed data to the output CSV file
    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["id", "sequence", "source", "label"])  # Header
        writer.writerows(data)

    print(f"CSV file saved at: {output_file}")

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

# def modify_numbers_in_text(input_file, output_file):
#     with open(input_file, 'r') as f:
#         lines = f.readlines()

#     modified_lines = []
#     for line in lines:
#         if '|' in line:
#             parts = line.split('|')
#             new_number = '1' if parts[1].strip() == '0' else '0'
#             modified_line = f"{parts[0]}|{new_number}\n"
#             modified_lines.append(modified_line)
#         else:
#             modified_lines.append(line)

#     # Write the modified lines to the output file
#     with open(output_file, 'w') as f:
#         f.writelines(modified_lines)



if __name__ == "__main__":
    # Construct the paths using paths.py
    input_path = os.path.join(datasets_sources_path, 'cpp924', 'cpp924.fa')
    output_path = os.path.join(full_datasets_path, 'cpp924_13_09.csv')

    # Call the processing function
    process_cpp924(input_path, output_path)

    DATE = '13_09'
    folds_training_dicts = load_as_pickle(
        os.path.join(paths.data_for_training_path, DATE, 'folds_traning_dicts.pkl')
    )
    
        # Define the path to the existing CSV
    csv_path = os.path.join(full_datasets_path, 'cpp924_13_09.csv')

    # Update the CSV with the fold index
    update_csv_with_folds_index(csv_path, folds_training_dicts)

