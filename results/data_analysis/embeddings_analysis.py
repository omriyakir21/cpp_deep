import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
from embeddings_analysis_utils import plot_tsne, load_esm2_embeddings, load_labels
from data_preperation.embeddings_creator_utils import esm2_model_names
import paths

# Load the embeddings data

DATE = '01_08'
# Load the embeddings
def save_tsne_plots_all_models(date):
    sequence_embeddings = load_esm2_embeddings(date)
    labels = load_labels(date)

    # Plot t-SNE
    for i, model_name in enumerate(esm2_model_names):
        embeddings = sequence_embeddings[i]
        plot_tsne(embeddings, labels, model_name, paths.data_analysis_plots_path)

save_tsne_plots_all_models(DATE)