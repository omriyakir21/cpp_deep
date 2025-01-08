import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
import paths
import pandas as pd
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio import SeqIO
import subprocess
import csv
from Bio import pairwise2
from Bio.pairwise2 import format_alignment


def sequence_has_only_natural_amino_acids(sequence):
    list_of_amino_acids = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']
    for amino_acid in sequence:
        if amino_acid not in list_of_amino_acids:
            return False
    return True

def csv_to_fasta(csv_file, fasta_file):
    # Read the CSV file
    df = pd.read_csv(csv_file)
    
    # Check if the 'sequence' column exists
    if 'sequence' not in df.columns:
        raise ValueError("CSV file must contain a 'sequence' column")
    
    
    # Create a list of SeqRecord objects
    records = []
    for index, row in df.iterrows():
        if sequence_has_only_natural_amino_acids(row['sequence']):
            seq = Seq(row['sequence'])
            record = SeqRecord(seq, id=str(index), description="")
            records.append(record)
    # Write the sequences to a FASTA file
    SeqIO.write(records, fasta_file, "fasta")

def sequence_to_fasta(sequence, fasta_file):
    seq = Seq(sequence)
    record = SeqRecord(seq, id='0', description="")
    SeqIO.write([record], fasta_file, "fasta")

def create_blast_db(fasta_file, db_name):
    command = [
        'makeblastdb',
        '-in', fasta_file,
        '-dbtype', 'prot',
        '-out', db_name
    ]
    subprocess.run(command, check=True)

def perform_local_blast(query_file, db_name, output_csv, output_alignments, threshold=0.1):
    # Command to get the tabular results
    command_tabular = [
        'blastp',
        '-evalue', str(threshold),
        '-query', query_file,
        '-db', db_name,
        '-outfmt', '6 qseqid sseqid pident length mismatch gapopen qstart qend sstart send evalue bitscore qseq sseq'
    ]
    
    # Run the command and capture the tabular output
    result_tabular = subprocess.run(command_tabular, capture_output=True, text=True)
    
    # Print the command output and errors for debugging
    print("Tabular STDOUT:", result_tabular.stdout)
    print("Tabular STDERR:", result_tabular.stderr)
    
    # Define the column headers
    headers = [
        "qseqid", "sseqid", "pident", "length", "mismatch", "gapopen",
        "qstart", "qend", "sstart", "send", "evalue", "bitscore", "qseq", "sseq"
    ]
    
    # Write the headers and the tabular result to the output CSV file
    with open(output_csv, 'w', newline='') as f:
        writer = csv.writer(f, delimiter=',')
        writer.writerow(headers)
        for line in result_tabular.stdout.strip().split('\n'):
            writer.writerow(line.split('\t'))
    #now perform pairwise alignment for each hit with the query sequence, dont use blast
    with open(output_alignments, 'w') as f:
        for line in result_tabular.stdout.strip().split('\n'):
            qseq = line.split('\t')[12]
            sseq = line.split('\t')[13]
            alignments = pairwise2.align.globalxx(qseq, sseq)
            if alignments:
                best_alignment = alignments[0]
                f.write(format_alignment(*best_alignment))
                f.write('\n')

    

if __name__ == '__main__':
    DATE = '13_09'
    data_analysis_path = paths.data_analysis_path
    blast_dir = os.path.join(data_analysis_path, 'blast')
    os.makedirs(blast_dir, exist_ok=True)
    csv_path = os.path.join(paths.full_datasets_path,f'full_peptide_dataset_{DATE}.csv')
    fasta_path = os.path.join(blast_dir, f'full_peptide_dataset_{DATE}.fasta')
    csv_to_fasta(csv_path, fasta_path)
    db_name = os.path.join(blast_dir, f'full_peptide_dataset_{DATE}')
    # create_blast_db(fasta_path, db_name)
    sequence = "HADGTFTNDMTSYLDAKAARDFVSWLARSDKS"
    sequence_dir = os.path.join(blast_dir, f'seq_{sequence}')
    query_path = os.path.join(sequence_dir, f'seq_{sequence}.fasta')
    os.makedirs(sequence_dir, exist_ok=True)
    sequence_to_fasta(sequence, query_path)
    output_csv = os.path.join(sequence_dir, f'blast_results_{sequence}.csv')
    output_alignments = os.path.join(sequence_dir, f'blast_alignments_{sequence}.txt')
    perform_local_blast(query_path, db_name, output_csv, output_alignments)

    