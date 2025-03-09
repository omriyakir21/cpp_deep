import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..','..'))
import paths
import pandas as pd

def csv_to_fasta(csv_file, fasta_file):
    def filter_sequences(df, min_length=10):
        return df[df['sequence'].str.len() >= min_length]
    
    def is_valid_sequence(sequence):
        valid_amino_acids = set("ACDEFGHIKLMNPQRSTVWY")
        return all(char in valid_amino_acids for char in sequence)
    
    df = pd.read_csv(csv_file)
    df = filter_sequences(df)
    df = df[df['sequence'].apply(is_valid_sequence)]
    sequences = df['sequence']
        
    with open(fasta_file, 'w') as f:
        for i, seq in enumerate(sequences):
            f.write(f">sequence{i+1}\n")
            f.write(f"{seq}\n")
    print(f"Saved {len(sequences)} sequences to {fasta_file}")
if __name__ == '__main__':
    csv_file = os.path.join(paths.result_analysis_path,"unlabled_data_predictions_merged_with_descriptions.csv")
    fasta_file = os.path.join(paths.result_analysis_path,"unlabled_sequences.fasta")
    csv_to_fasta(csv_file, fasta_file)
