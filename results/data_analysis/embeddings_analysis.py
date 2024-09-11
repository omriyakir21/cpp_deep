import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
from embeddings_analysis_utils import plot_tsne, load_esm2_embeddings, save_sequence_length_histogram
from data_preperation.dataset_creator_utils import esm2_model_names,FULL_DATASET_NAME
import paths
from utils import load_labels, load_sequences


# Load the embeddings data

# Load the embeddings
def save_tsne_plots_all_models(date,data_set_name):
    sequence_embeddings = load_esm2_embeddings(date)
    labels = load_labels(date,data_set_name)

    # Plot t-SNE
    for i, model_name in enumerate(esm2_model_names):
        embeddings = sequence_embeddings[i]
        plot_tsne(embeddings, labels, model_name, os.path.join(paths.data_analysis_plots_path,date),date)


# add if main
if __name__ == '__main__':
    DATE = '10_09'
    FULL_DATASET_NAME = 'full_peptide_dataset'
    save_tsne_plots_all_models(DATE,FULL_DATASET_NAME)

    labels, sequences = load_labels(DATE,FULL_DATASET_NAME), load_sequences(DATE,FULL_DATASET_NAME)

    # Plot the histogram of sequence lengths
    save_sequence_length_histogram(sequences, labels, os.path.join(paths.data_analysis_plots_path),DATE)
