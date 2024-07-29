import os.path

from data_preperation_utils import load_esm2_model, get_embeddings, read_fasta
import path
# Load the ESM2 model and tokenizer
tokenizer, model = load_esm2_model()

# Read sequences from the FASTA file
file_path = os.path.join(path.cpp_natural_residues_path, 'natural_pep.fa')
sequences = read_fasta(file_path)

# Get embeddings for the sequences
sequence_embeddings = get_embeddings(sequences, tokenizer, model)

# Print the shape of the embeddings
print(sequence_embeddings.shape)  # (number_of_sequences, 1280) for esm2_t33_650M_UR50D