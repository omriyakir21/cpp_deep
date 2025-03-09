import os
import sys
import paths
import pandas as pd
import pickle
import numpy as np
from sklearn.metrics import precision_recall_curve, auc, roc_curve
from matplotlib import pyplot as plt
import fileinput
import subprocess
import time


def save_as_pickle(data, file_path):
    with open(file_path, 'wb') as file:
        pickle.dump(data, file)

def load_as_pickle(file_path):
    with open(file_path, 'rb') as file:
        data = pickle.load(file)
    return data

def load_labels(date,data_set_name):
    """
    Load labels from a CSV file.

    Args:
        DATE (str): date of the labels file.

    Returns:
        ndarray: ndarray containing labels of the sequences
    """
    df = pd.read_csv(os.path.join(paths.full_datasets_path, f'{data_set_name}_{date}.csv'))

    labels = np.array(df['label'].tolist())

    return labels


def load_sequences(date,data_set_name):
    """
    Load labels from a CSV file.

    Args:
        DATE (str): date of the labels file.

    Returns:
        ndarray: ndarray containing labels of the sequences
    """
    df = pd.read_csv(os.path.join(paths.full_datasets_path, f'{data_set_name}_{date}.csv'))

    labels = np.array(df['sequence'].tolist())

    return labels

def load_ids(date,data_set_name):
    """
    Load IDs from a CSV file.

    Args:
        date (str): date of the IDs file.

    Returns:
        ndarray: ndarray containing IDs of the sequences
    """
    df = pd.read_csv(os.path.join(paths.full_datasets_path, f'{data_set_name}_{date}.csv'))

    ids = np.array(df['id'].tolist())

    return ids

def load_descriptions(date,data_set_name):
    """
    Load descriptions from a CSV file.

    Args:
        date (str): date of the descriptions file.

    Returns:
        ndarray: ndarray containing descriptions of the sequences
    """
    df = pd.read_csv(os.path.join(paths.full_datasets_path, f'{data_set_name}_{date}.csv'))

    descriptions = np.array(df['description'].tolist())

    return descriptions

def plot_pr_curve(y_true, y_scores, title='Precision-Recall Curve', save_path=None):
    precision, recall, _ = precision_recall_curve(y_true, y_scores)
    pr_auc = auc(recall, precision)
    plt.figure()
    plt.plot(recall, precision, label=f'PRAUC = {pr_auc:.3f}')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(title)
    plt.legend(loc='best')
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()

def plot_roc_curve(y_true, y_scores, title='ROC Curve', save_path=None):
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, label=f'ROCAUC = {roc_auc:.3f}')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc='best')
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()

path2mmseqs = 'mmseqs'
path2mmseqstmp = '/specific/disk2/home/mol_group/tmp/'
path2mmseqsdatabases = '/specific/disk2/home/mol_group/sequence_database/MMSEQS/'

def call_mmseqs(
        input_file,
        output_file,
        database = 'SwissProt',
        nthreads = 6,
        filtermsa = True,
        cov = 0.0,
        qid = 0.0,
        maxseqid = 1.1,
        gapopen = 11,
        gapextend = 1,
        s = 5.7000,
        num_iterations = 1,
        maxseqs = 100000,
        overwrite = True,
        report=None

):
    t = time.time()
    if cov>1:
        cov = cov/100.
    if not overwrite:
        if os.path.exists(output_file):
            print('File %s already exists. Not recomputing' %output_file,file=report)
            return output_file

    ninputs = sum([line.startswith('>') for line in open(input_file,'r').readlines()])

    '''
    Source: https://github.com/soedinglab/MMseqs2/issues/693
    '''
    tmp_folder = '.'.join(output_file.split('.')[:-1]) + '/'
    os.makedirs(tmp_folder,exist_ok=True)
    tmp_input_file = os.path.join(tmp_folder,'input')
    tmp_output_file = os.path.join(tmp_folder, 'output')
    tmp_output_file2 = os.path.join(tmp_folder, 'output2')

    commands = [
        [path2mmseqs, 'createdb', input_file, tmp_input_file],
        [path2mmseqs, 'search', tmp_input_file, os.path.join(path2mmseqsdatabases,database), tmp_output_file, path2mmseqstmp,
        '-s',str(s),'--cov', str(cov),'--cov-mode','1','--diff',str(maxseqs), '--qid', str(qid), '--max-seq-id', str(maxseqid),'--gap-open', str(gapopen), '--gap-extend', str(gapextend), '--threads',str(nthreads),'--num-iterations',str(num_iterations),
           '--max-seqs',str(maxseqs)],
        [path2mmseqs, 'result2msa', tmp_input_file, os.path.join(path2mmseqsdatabases,database), tmp_output_file,tmp_output_file2,
        '--filter-msa',str(int(filtermsa)), '--cov',str(cov),'--diff',str(maxseqs),'--qid', str(qid), '--max-seq-id',str(maxseqid),'--msa-format-mode','5','--gap-open', str(gapopen), '--gap-extend',str(gapextend), '--threads',str(nthreads)],
        [path2mmseqs, 'convertalis', tmp_input_file, os.path.join(path2mmseqsdatabases,database), tmp_output_file,tmp_output_file +'.tab','--format-output',"target,theader"],
        [path2mmseqs, 'unpackdb', tmp_output_file2, tmp_folder, '--unpack-name-mode', '0', '--threads',str(nthreads)]
    ]
    for command in commands:
        print(' '.join(command))
        subprocess.call(command)
    table_labels = pd.read_csv(tmp_output_file +'.tab',sep='\t',header=None,index_col=0).drop_duplicates()

    for n in range(ninputs):
        if ninputs == 1:
            output_file_ = output_file
        else:
            output_file_ = output_file.split('.fasta')[0] + '_%s.fasta'%n
        os.rename(os.path.join(tmp_folder,str(n) ),output_file_)
        with fileinput.input(files=output_file_, inplace=True) as f:
            for line in f:
                if line.startswith('>'):
                    try:
                        newlabel = table_labels.loc[line[1:-1]].item()
                    except:
                        newlabel = 'na|%s|' % line[1:-1]
                    newline = '>' + newlabel + '\n'
                else:
                    newline = line
                print(newline, end='')
        assert os.path.exists(output_file_)
    subprocess.call(['rm', '-r', tmp_folder])
    print('Called mmseqs finished: Duration %.2f s' % (time.time() - t),file=report)
    return output_file
