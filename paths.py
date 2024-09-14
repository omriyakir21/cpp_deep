import os

# List of directory and files paths
current_dir = os.path.dirname(os.path.abspath(__file__))

# datasets paths
datasets_path = os.path.join(current_dir, 'datasets')
datasets_sources_path = os.path.join(datasets_path, 'sources')
data_for_training_path = os.path.join(datasets_path, 'data_for_training')
full_datasets_path = os.path.join(datasets_path, 'full_datasets')
BIOPEP_UWMix_path = os.path.join(datasets_sources_path, 'BIOPEP_UWMix')
cpp_natural_residues_path = os.path.join(datasets_sources_path, 'cpp_natural_residues')
cpp_natural_residues_peptides_path = os.path.join(cpp_natural_residues_path, 'natural_pep.fa')
SPENCER_path = os.path.join(datasets_sources_path, 'SPENCER')
SPENCER_peptides_path = os.path.join(SPENCER_path, 'SPENCER_Peptide_info.txt')
PeptideAtlas_path = os.path.join(datasets_sources_path, 'PeptideAtlas')
PeptideAtlas_path_Human_path = os.path.join(PeptideAtlas_path, 'Human')
PeptideAtlas_path_Human_peptides_path = os.path.join(PeptideAtlas_path_Human_path, 'APD_Hs_all.fasta')
embeddings_path = os.path.join(datasets_path, 'embeddings')
esm2_embeddings_path = os.path.join(embeddings_path, 'esm2')

# models
models_path = os.path.join(current_dir, 'models')
#   baselines
baselines_models_path = os.path.join(models_path, 'baselines')
#       convolution_baseline
convolution_baseline_models_path = os.path.join(baselines_models_path, 'convolution_baseline')


# results
results_path = os.path.join(current_dir, 'results')
#   baselines
baselines_results_path = os.path.join(results_path, 'baselines')
#       convolution_baseline
convolution_baseline_results_path = os.path.join(baselines_results_path, 'convolution_baseline')
data_analysis_path = os.path.join(results_path, 'data_analysis')
data_analysis_plots_path = os.path.join(data_analysis_path, 'plots')

# tmp
tmp_path = os.path.join(current_dir, 'tmp')

# mafft exec
mafft_exec_path = '/home/iscb/wolfson/omriyakir/anaconda3/envs/ubinet/bin/mafft'

# mmseqs exec
mmseqs_exec_path = '/home/iscb/wolfson/omriyakir/anaconda3/envs/ubinet/bin/mmseqs'