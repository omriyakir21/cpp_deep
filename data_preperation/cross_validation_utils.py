import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from sklearn.model_selection import KFold
from sklearn.cluster import AgglomerativeClustering
from Bio import pairwise2
import numpy as np
import subprocess
import pandas as pd
import paths
from utils import save_as_pickle, load_as_pickle, load_labels, load_sequences, load_ids


def cluster_sequences(list_sequences, seqid=1.0, coverage=0.8, covmode='0', path2mmseqstmp=paths.tmp_path,path2mmseqs=paths.mmseqs_exec_path):
    rng = np.random.randint(0, high=int(1e6))
    tmp_input = os.path.join(path2mmseqstmp, 'tmp_input_file_%s.fasta' % rng)
    tmp_output = os.path.join(path2mmseqstmp, 'tmp_output_file_%s' % rng)

    with open(tmp_input, 'w') as f:
        for k, sequence in enumerate(list_sequences):
            f.write('>%s\n' % k)
            f.write('%s\n' % sequence)

    command = ('{mmseqs} easy-cluster {fasta} {result} {tmp} --min-seq-id %s -c %s --cov-mode %s' % (
        seqid, coverage, covmode)).format(mmseqs=path2mmseqs, fasta=tmp_input, result=tmp_output, tmp=path2mmseqstmp)
    subprocess.run(command.split(' '))

    with open(tmp_output + '_rep_seq.fasta', 'r') as f:
        representative_indices = [int(x[1:-1]) for x in f.readlines()[::2]]
    cluster_indices = np.zeros(len(list_sequences), dtype=int)
    table = pd.read_csv(tmp_output + '_cluster.tsv', sep='\t', header=None).to_numpy(dtype=int)
    for i, j in table:
        if i in representative_indices:
            cluster_indices[j] = representative_indices.index(i)
    for file in [tmp_output + '_rep_seq.fasta', tmp_output + '_all_seqs.fasta', tmp_output + '_cluster.tsv']:
        os.remove(file)
    return np.array(cluster_indices), np.array(representative_indices)

def create_cluster_participants_indices(cluster_indices):
    clustersParticipantsList = []
    for i in range(np.max(cluster_indices) + 1):
        clustersParticipantsList.append(np.where(cluster_indices == i)[0])
    return clustersParticipantsList


def divide_clusters(cluster_sizes):
    """
    :param cluster_sizes: list of tuples (clusterIndex,size)
    :return:  sublists,sublistsSum
    divide the list into 5 sublists such that the sum of each cluster sizes in the sublist is as close as possible
    """
    sublists = [[] for i in range(5)]
    sublistsSum = [0 for i in range(5)]
    cluster_sizes.sort(reverse=True, key=lambda x: x[1])  # Sort the clusters by size descending order.
    for tup in cluster_sizes:
        min_cluster_index = sublistsSum.index(min(sublistsSum))  # find the cluster with the minimal sum
        sublistsSum[min_cluster_index] += tup[1]
        sublists[min_cluster_index].append(tup[0])
    return sublists, sublistsSum


def get_ids_indices_for_groups(clusters_participants_list, sublists, fold_num):
    """
    Get the Uniprot indices for a specific fold.

    :param clusters_participants_list: List of np arrays where each array contains the indices of Uniprots in that fold.
    :param sublists: List of lists where each sublist contains the cluster indices for that fold.
    :param fold_num: The fold number for which to get the Uniprot indices.
    :return: List of Uniprot indices for the specified fold.
    """

    ids_indices = []
    for cluster_index in sublists[fold_num]:
        fold_indices = list(clusters_participants_list[cluster_index])
        ids_indices.append(fold_indices)
    return np.concatenate(ids_indices)

def create_training_folds(groups_indices, sequences,labels,ids):
    folds_training_dicts = []
    for i in range(5):
        training_dict = {}

        training_indices = np.concatenate((groups_indices[(i + 2) % 5] , groups_indices[(i + 3) % 5] , groups_indices[(i + 4) % 5]))
        validation_indices = groups_indices[i]
        test_indices = groups_indices[(i + 1) % 5]

        # When using these indices to index tensors
        training_dict['sequences_train'] = [sequences[j] for j in training_indices]
        training_dict['sequences_validation'] = [sequences[j] for j in validation_indices]
        training_dict['sequences_test'] = [sequences[j] for j in test_indices]
        training_dict['labels_train'] = [labels[j] for j in training_indices]
        training_dict['labels_validation'] = [labels[j] for j in validation_indices]
        training_dict['labels_test'] = [labels[j] for j in test_indices]
        training_dict['ids_train'] = [ids[j] for j in training_indices]
        training_dict['ids_validation'] = [ids[j] for j in validation_indices]
        training_dict['ids_test'] = [ids[j] for j in test_indices]
 
        folds_training_dicts.append(training_dict)
    return folds_training_dicts

def partition_to_folds_and_save(date,data_set_name):
    labels = load_labels(date,data_set_name)
    ids = load_ids(date,data_set_name)
    sequences = load_sequences(date,data_set_name)
    data_for_training_dir = os.path.join(paths.data_for_training_path, date)
    if not os.path.exists(data_for_training_dir):
        os.makedirs(data_for_training_dir)
    
    cluster_indices, _ = cluster_sequences(sequences, seqid=0.5, coverage=0.4)
    print('Clustered sequences')
    save_as_pickle(cluster_indices, os.path.join(data_for_training_dir, 'cluster_indices.pkl'))
    print('Saved cluster indices')
    clusters_participants_list = create_cluster_participants_indices(cluster_indices)
    cluster_sizes = [l.size for l in clusters_participants_list]
    cluster_sizes_and_indices = [(i, cluster_sizes[i]) for i in range(len(cluster_sizes))]
    sublists, _ = divide_clusters(cluster_sizes_and_indices)
    groups_indices = [get_ids_indices_for_groups(clusters_participants_list, sublists, fold_num) for fold_num
                      in
                      range(5)] 
    print('Created groups indices')
    save_as_pickle(groups_indices, os.path.join(data_for_training_dir, 'groups_indices.pkl'))
 
        # CREATE TRAINING DICTS
    folds_training_dicts = create_training_folds(groups_indices,sequences,labels,ids)
                                                   
    save_as_pickle(folds_training_dicts,os.path.join(data_for_training_dir,'folds_traning_dicts.pkl'))


def create_ids_sets():
    ids = load_as_pickle(os.path.join(paths.patch_to_score_data_for_training_path, 'uniprots.pkl'))
    groups_indices = load_as_pickle(os.path.join(paths.patch_to_score_data_for_training_path, 'groups_indices.pkl'))
    ids_sets = []
    for i in range(5):
        ids_set = set([ids[j] for j in groups_indices[(i+1)%5]])
        ids_sets.append(ids_set)
    save_as_pickle(ids_sets, os.path.join(paths.patch_to_score_data_for_training_path, 'uniprots_sets.pkl'))
