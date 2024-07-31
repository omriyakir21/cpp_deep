import os

# List of directory and files paths
current_dir = os.path.dirname(os.path.abspath(__file__))

# datasets paths
datasets_path = os.path.join(current_dir, 'datasets')
BIOPEP_UWMix_path = os.path.join(datasets_path, 'BIOPEP_UWMix')
cpp_natural_residues_path = os.path.join(datasets_path, 'cpp_natural_residues')
embeddings_path = os.path.join(datasets_path, 'embeddings')
SPENCER_path = os.path.join(datasets_path, 'SPENCER')
SPENCER_peptides_path = os.path.join(SPENCER_path, 'SPENCER_Peptide_info.txt')
PeptideAtlas_path = os.path.join(datasets_path, 'PeptideAtlas')
PeptideAtlas_path_Human_path = os.path.join(PeptideAtlas_path, 'Human')
PeptideAtlas_path_Human_peptides_path = os.path.join(PeptideAtlas_path_Human_path, 'SPENCER_Peptide_info.txt')

# models
models_path = os.path.join(current_dir, 'models')
# results
results_path = os.path.join(current_dir, 'results')
