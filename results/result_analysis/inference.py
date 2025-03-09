import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__),'..','..'))
import paths
import numpy as np
import pandas as pd
from transformers import EsmTokenizer
from Bio.SeqUtils.ProtParam import ProteinAnalysis
from results.result_analysis.predict_sub_sequences import (
    load_lora_models,
    load_fine_tune_models,
    load_few_shot_models,
    load_convolution_models,
    create_dataset_fine_tune,
    predict_fine_tune,
    create_dataset_few_shot,
    predict_few_shot,
    predict_with_convolution,
    create_dataset_convolution
)
import subprocess

def predict_sequences(sequences, models):
    # Implement the prediction logic for a batch of sequences using a list of models
    predictions = []
    for model in models:
        predictions.append(model.predict(sequences))
    return predictions

def make_predictions(df,dataset_maker_function,predict_function,models,model_name,tokenizer=None):
    predictions = []
    # Split the DataFrame into subsets based on the folds_index_to_predict column
    for fold_index in range(len(5)):
        subset = df[df['folds_index_to_predict'] == fold_index]
        if not subset.empty:
            sequences = subset['sequence'].tolist()
            if tokenizer is not None:
                if model_name == 'few_shot':
                    dataset = dataset_maker_function(sequences,'facebook/esm2_t6_8M_UR50D')
                else:
                    dataset = dataset_maker_function(sequences,tokenizer)
            else:
                dataset = dataset_maker_function(sequences)
            
            fold_predictions = predict_function(models[fold_index],dataset)
            # print(f'model name : {model_name}, fold predictions shape : {fold_predictions.shape}')
            # print(f' subset is : {subset}')
            i = 0
            for _, row in subset.iterrows():
                print(f'row is : {row}, i is : {i}')
                row_data = row.to_dict()
                row_data.update({
                    f'{model_name}_prediction': fold_predictions[i]
                })
                predictions.append(row_data)
                i+=1

    # Handle sequences where folds_index_to_predict is None
    subset = df[df['folds_index_to_predict'].isnull()]
    if not subset.empty:
        sequences = subset['sequence'].tolist()
        if tokenizer is not None:
            if model_name == 'few_shot':
                dataset = dataset_maker_function(sequences,'facebook/esm2_t6_8M_UR50D')
            else:
                dataset = dataset_maker_function(sequences,tokenizer)
        else:
            dataset = dataset_maker_function(sequences)
        all_fold_predictions = [predict_function(fold ,dataset) for fold in models]
        average_predictions = np.mean(all_fold_predictions, axis=0)
        i = 0
        for _, row in subset.iterrows():
            row_data = row.to_dict()
            row_data.update({
                f'{model_name}_prediction': average_predictions[i]
            })
            predictions.append(row_data)
            i+=1
    predictions_df = pd.DataFrame(predictions)
    return predictions_df

def create_predictions_table_all_models(df,csv_path,dataset_name,tokenizer):
    lora_models = load_lora_models()
    fine_tune_models = load_fine_tune_models()
    few_shot_models = load_few_shot_models()
    convolution_models = load_convolution_models()
    predictions_df = make_predictions(df, create_dataset_fine_tune, predict_fine_tune, fine_tune_models, 'fine_tune',tokenizer)
    predictions_df = make_predictions(predictions_df, create_dataset_fine_tune, predict_fine_tune, lora_models, 'lora',tokenizer)
    predictions_df = make_predictions(predictions_df, create_dataset_few_shot, predict_few_shot, few_shot_models, 'few_shot', tokenizer)
    predictions_df = make_predictions(predictions_df, create_dataset_convolution, predict_with_convolution, convolution_models, 'convolution')
    print(f'predictions_df: {predictions_df}')
    predictions_df['average_prediction'] = predictions_df[['fine_tune_prediction', 'lora_prediction', 'few_shot_prediction', 'convolution_prediction']].mean(axis=1)
    
    predictions_df = predictions_df.sort_values(by='average_prediction', ascending=False)
    csv_path = os.path.join(paths.result_analysis_path, f'predictions{dataset_name}.csv')
    predictions_df.to_csv(csv_path, index=False)
    print(f'saving predictions to {csv_path}')



def merge_neuropep1_training_peptides():
    neuropep1_path = os.path.join(paths.result_analysis_path, 'predictionsNeuroPep1.csv')
    cpp_path = os.path.join(paths.result_analysis_path, 'unlabled_data_predictions_merged_with_descriptions.csv')
    
    neuropep1_df = pd.read_csv(neuropep1_path)
    cpp_df = pd.read_csv(cpp_path)
    
    # Rename 'Name' column in neuropep1_df to 'description'
    neuropep1_df = neuropep1_df.rename(columns={'Name': 'description'})
    
    # Add 'source' column to cpp_df with value 'BIOPEP'
    cpp_df['source'] = 'BIOPEP'
    
    # make dict of both cpp_df where sequence is key and the row with the columns:
    #  sequence,convolution_prediction,lora_prediction,fine_tune_prediction,few_shot_prediction,average_prediction,description 
    # are the values is the value
    combined_dict = {}
    for _, row in cpp_df.iterrows():
        sequence = row['sequence']
        combined_dict[sequence] = {
            'sequence': sequence,
            'source': row['source'],
            'description': row['description'],
            'convolution_prediction': row['convolution_prediction'],
            'lora_prediction': row['lora_prediction'],
            'fine_tune_prediction': row['fine_tune_prediction'],
            'few_shot_prediction': row['few_shot_prediction'],
            'average_prediction': row['average_prediction']
            
        }
    for _, row in neuropep1_df.iterrows():
        sequence = row['sequence']
        combined_dict[sequence] = {
            'sequence': sequence,
            'source': row['source'],
            'description': row['description'],
            'convolution_prediction': row['convolution_prediction'],
            'lora_prediction': row['lora_prediction'],
            'fine_tune_prediction': row['fine_tune_prediction'],
            'few_shot_prediction': row['few_shot_prediction'],
            'average_prediction': row['average_prediction']
        }

    result_df = pd.DataFrame(list(combined_dict.values()))
    result_df = result_df.sort_values(by='average_prediction', ascending=False)
    csv_path = os.path.join(paths.result_analysis_path, 'neuropep1_biopep_predictions.csv')
    result_df.to_csv(csv_path, index=False)
    print(f'Saved merged predictions to {csv_path}')

def extract_features(sequence):
    analysed_seq = ProteinAnalysis(sequence)
    weight = analysed_seq.molecular_weight()
    gravy = analysed_seq.gravy()
    aa_fraction = analysed_seq.get_amino_acids_percent()
    aromaticity = analysed_seq.aromaticity()
    instability_index = analysed_seq.instability_index()
    flexibility = analysed_seq.flexibility()
    isoelectric_point = analysed_seq.isoelectric_point()
    secondary_structure_fraction = analysed_seq.secondary_structure_fraction()
    features = {
        'weight':weight,
        'gravy':gravy,
        'aromaticity':aromaticity,
        'instability_index':instability_index,
        'isoelectric_point':isoelectric_point,
    }
    for k,ss in enumerate(['Helix','Turn','Sheet']):
        features[f'fraction_{ss}'] = secondary_structure_fraction[k]
    for aa in aa_fraction.keys():
        features[f'fraction_{aa}'] = aa_fraction[aa]
    return features

def add_features_and_filter(csv_path):
    df = pd.read_csv(csv_path)
    # Filter out sequences with non-natural amino acids
    natural_amino_acids = set('ACDEFGHIKLMNPQRSTVWY')
    def is_natural(sequence):
        return all(aa in natural_amino_acids for aa in sequence)
    
    df = df[df['sequence'].apply(is_natural)]
    sequences = df['sequence'].tolist()
    features_list = [extract_features(seq) for seq in sequences]
    # add features to the dataframe
    features_df = pd.DataFrame(features_list)
    df = pd.concat([df,features_df],axis=1)
    #calculate the length of the sequences and keep only sequences with 10<=length<= 50
     # Filter out non-string values in the 'sequence' column
    df = df[df['sequence'].apply(lambda x: isinstance(x, str))]
    df['length'] = df['sequence'].apply(len)
    df = df[(df['length'] >= 10) & (df['length'] <= 50)]
    df = df[df['aromaticity'] <= 0.2]
    df = df[df['instability_index'] < 50]
    df = df[df['isoelectric_point'] < 10.5]
    #check if average_prediction is in the columns else sort with prediction column
    if 'average_prediction' in df.columns:
        df = df.sort_values(by='average_prediction', ascending=False)
    else:
        df = df.sort_values(by='prediction', ascending=False)
    csv_folder = os.path.dirname(csv_path)
    file_name = os.path.basename(csv_path)
    csv_path = os.path.join(csv_folder, f'filtered_{file_name}')  
    df.to_csv(csv_path, index=False)
    return csv_path

def remove_positive_examples(csv_path):
    # i have labels column
    df = pd.read_csv(csv_path)
    df = df[df['label'] == 0]
    csv_folder = os.path.dirname(csv_path)
    file_name = os.path.basename(csv_path).split('.')[0]
    csv_path = os.path.join(csv_folder, f'{file_name}_negatives.csv')
    df.to_csv(csv_path, index=False)
    return csv_path


def write_sequences_to_fasta(csv_path):
    df = pd.read_csv(csv_path)
    sequences = df['sequence'].tolist()
    fasta_path = f'{csv_path.split(".csv")[0]}.fasta'
    with open(fasta_path, 'w') as f:
        for i, seq in enumerate(sequences):
            f.write(f'>{seq}\n{seq}\n')
    print(f'Saved sequences to {fasta_path}')
    return fasta_path

def add_toxinity_predictions_and_filter(filtered_csv_path, toxinpred2_output_path,toxciity_threshold):
    df = pd.read_csv(filtered_csv_path)
    toxinpred2_df = pd.read_csv(toxinpred2_output_path)
    merged_df = pd.merge(df, toxinpred2_df[['sequence','Raw predictions']], on='sequence', how='left')
    merged_df.rename(columns={'prediction_score': 'Raw predictions'}, inplace=True)
    merged_df = merged_df[merged_df['Raw predictions'] <= toxciity_threshold]
    csv_path = f'{filtered_csv_path.split(".csv")[0]}_toxicity.csv'
    merged_df.to_csv(csv_path, index=False)
    print(f'Saved predictions to {csv_path}')
    return csv_path

def format_float_columns(csv_path):
    df = pd.read_csv(csv_path)
    float_columns = df.select_dtypes(include=['float64']).columns
    df[float_columns] = df[float_columns].applymap(lambda x: f"{x:.4f}")
    df.to_csv(csv_path, index=False)
    print(f"Formatted float columns in {csv_path}")

def add_source_and_description(predictions_csv_path, full_dataset_path):
    df = pd.read_csv(predictions_csv_path)
    full_dataset_df = pd.read_csv(full_dataset_path)

    # Merge with full dataset to add source and description
    merged_df = pd.merge(df, full_dataset_df[['sequence', 'source', 'description']], on='sequence', how='left')

    # Reorder columns to place source and description as the third and fourth columns
    columns_order = ['sequence', 'Raw predictions', 'source', 'description'] + [col for col in merged_df.columns if col not in ['sequence', 'Raw predictions', 'source', 'description']]
    merged_df = merged_df[columns_order]
    merged_df = merged_df.rename(columns={'Raw predictions': 'toxicity'})
    csv_path = f'{predictions_csv_path.split(".csv")[0]}_with_source_description.csv'
    merged_df.to_csv(csv_path, index=False)
    print(f'Saved merged data to {csv_path}')
    return csv_path

# Example usage within your existing code
if __name__ == '__main__':
    # tokenizer_name = 'facebook/esm2_t6_8M_UR50D'
    # tokenizer = EsmTokenizer.from_pretrained(tokenizer_name)
    DATE = '13_02'
    # dataset_name = 'NeuroPep1'
    model_folder_name = 'models_128_5_3_256_100_0.3_valid_1_roc_metric'
    model_results_folder = os.path.join(paths.convolution_baseline_pu_learning_bagging_results_path, DATE, model_folder_name)
    csv_path = os.path.join(model_results_folder, 'predictions_labels_sequences.csv')
    negatives_csv_path = remove_positive_examples(csv_path)
    filtered_csv_path = add_features_and_filter(negatives_csv_path)
    print(f'filtered_csv_path: {filtered_csv_path}')
    fasta_path = write_sequences_to_fasta(filtered_csv_path)
    print(f'fasta_path: {fasta_path}')
    toxinpred2_output_path = f'{filtered_csv_path.split(".csv")[0]}_toxinpred2_output.csv'
    this_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(paths.toxinpred2_models_path)

    command = [
        "toxinpred2",
        "-i", fasta_path,  # Input FASTA file
        "-o", toxinpred2_output_path,  # Output CSV file
        "-m", "1"  # Model: 1 (AAC based RF) - Since Hybrid mode isn't needed for raw predictions
    ]



    try:
        result = subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        print("Prediction completed successfully.")
        print("Subprocess output:")
        print(result.stdout)
        print("Subprocess errors (if any):")
        print(result.stderr)
    except subprocess.CalledProcessError as e:
        print(f"Error occurred: {e}")
        print("Subprocess output:")
        print(e.stdout)
        print("Subprocess errors:")
        print(e.stderr)
    os.chdir(this_dir)
    toxicity_threshold = 0.6
    filtered_with_toxicity_path = add_toxinity_predictions_and_filter(filtered_csv_path, toxinpred2_output_path,toxicity_threshold)
    final_csv_path = add_source_and_description(filtered_with_toxicity_path,os.path.join(paths.full_datasets_path,f'full_peptide_dataset_{DATE}.csv'))
    # Format float columns in the final CSV file
    format_float_columns(final_csv_path)

    # create_predictions_table_all_models(df, csv_path, dataset_name, tokenizer)
    # merge_neuropep1_training_peptides()

    # add_features_and_filter(os.path.join(paths.result_analysis_path, 'neuropep1_biopep_predictions.csv'))
