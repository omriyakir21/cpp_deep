import os
from torch.utils.data import DataLoader
from transformers import EsmTokenizer, EsmModel
from data_preperation.embeddings_creator_utils import load_esm2_model, get_embeddings, read_fasta
import path

class ProteinEmbeddingsDataset(Dataset):
    def __init__(self, embeddings):
        self.embeddings = embeddings

    def __len__(self):
        return len(self.embeddings)

    def __getitem__(self, idx):
        return self.embeddings[idx]


def prepare_training_data(batch_size, num_workers, model_name="facebook/esm2_t6_8M_UR50D"):
    # Load the ESM2 Model and Tokenizer
    tokenizer, model = load_esm2_model(model_name)

    # Read Sequences from FASTA File
    fasta_file_path = os.path.join(path.datasets_path, 'protein_sequences.fasta')
    sequences = read_fasta(fasta_file_path)

    # Get Embeddings for Sequences
    embeddings = get_embeddings(sequences, tokenizer, model)

    # Initialize DataLoader
    dataset = ProteinEmbeddingsDataset(embeddings)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    return dataloader, embeddings.shape[1]