import os.path
import torch
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
print(sequence_embeddings.shape)

# Save the embeddings to a file
output_file = os.path.join(path.embeddings_path, 'sequence_embeddings.pt')
torch.save(sequence_embeddings, output_file)