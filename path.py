import os

# List of directory paths
current_dir = os.path.dirname(os.path.abspath(__file__))
datasets_path = os.path.join(current_dir, 'datasets')
cpp_natural_residues_path = os.path.join(datasets_path, 'cpp_natural_residues')
embeddings_path = os.path.join(datasets_path, 'embeddings')
models_path = os.path.join(current_dir, 'models')
results_path = os.path.join(current_dir, 'results')

