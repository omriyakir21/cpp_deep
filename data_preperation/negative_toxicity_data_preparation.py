import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import paths
import pandas as pd

def get_sequences_with_label_zero(csv_file_path):
    # Read the CSV file into a DataFrame
    df = pd.read_csv(csv_file_path)
    
    # Filter the DataFrame to get rows where the label is 0
    filtered_df = df[df['label'] == 0]
    
    # Return the sequences with label 0
    
    return filtered_df['sequence'].tolist()
def write_sequences_to_fasta(sequences, output_fasta_path):
    with open(output_fasta_path, 'w') as fasta_file:
        for i, sequence in enumerate(sequences):
            fasta_file.write(f">sequence_{i}\n")
            fasta_file.write(f"{sequence}\n")

    # Example usage
if __name__ == "__main__":
    csv_file_path = '/home/iscb/wolfson/omriyakir/cpp_deep/results/esm2/lora/13_09/esm2_t6_8M_UR50D/architecture_128_50_100_32/predictions_labels_sequences.csv'
    output_fasta_path = os.path.join(paths.datasets_toxicity_prediction_path,'negative_sequences.fasta')
    sequences = get_sequences_with_label_zero(csv_file_path)
    write_sequences_to_fasta(sequences, output_fasta_path)