import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__),'..','..'))
import paths
import numpy as np
from sklearn.metrics import precision_recall_curve, auc
from matplotlib import pyplot as plt
import pandas as pd

def load_predictions(path):
    file_path = os.path.join(path, 'predictions_labels_sequences.csv')
    return pd.read_csv(file_path)

def filter_negative_sequences(df):
    return df[df['label'] == 0]

def create_unlabled_merge_csv():
    DATE = '13_09'
    model_name = 'esm2_t6_8M_UR50D'
    convolution_results_path = os.path.join(paths.convolution_baseline_results_path,DATE)
    lora_results_path = os.path.join(paths.lora_results_path,DATE,model_name,'architecture_128_50_100_32')
    fine_tune_results_path = os.path.join(paths.fine_tune_results_path,DATE,model_name,'architecture_100_256')
    few_shot_results_path = os.path.join(paths.few_shot_learning_results_path,DATE,model_name,'architecture_15_15_64')
    convolution_df = filter_negative_sequences(load_predictions(convolution_results_path))
    lora_df = filter_negative_sequences(load_predictions(lora_results_path))
    fine_tune_df = filter_negative_sequences(load_predictions(fine_tune_results_path))
    few_shot_df = filter_negative_sequences(load_predictions(few_shot_results_path))
    
    merged_df = pd.DataFrame({
        'sequence': convolution_df['sequence'],
        'convolution_prediction': convolution_df['prediction'],
        'lora_prediction': lora_df['prediction'],
        'fine_tune_prediction': fine_tune_df['prediction'],
        'few_shot_prediction': few_shot_df['prediction']
    })
    
    merged_df['average_prediction'] = merged_df[['convolution_prediction', 'lora_prediction', 'fine_tune_prediction', 'few_shot_prediction']].mean(axis=1)
    
    merged_df = merged_df.sort_values(by='average_prediction',ascending=False)
    
    merged_df['convolution_prediction'] = merged_df['convolution_prediction'].round(3)
    merged_df['lora_prediction'] = merged_df['lora_prediction'].round(3)
    merged_df['fine_tune_prediction'] = merged_df['fine_tune_prediction'].round(3)
    merged_df['few_shot_prediction'] = merged_df['few_shot_prediction'].round(3)
    merged_df['average_prediction'] = merged_df['average_prediction'].round(3)
    print(merged_df)
    csv_path = os.path.join(paths.result_analysis_path,'unlabled_data_predictions_merged.csv')
    merged_df.to_csv(csv_path, index=False)
    print(f'saves csv in {csv_path}')

def add_descriptions(predictions_df,descriptions_df):
        merged_df = pd.merge(predictions_df, descriptions_df, on='sequence', how='left')
        #save the df 
        csv_path = os.path.join(paths.result_analysis_path,'unlabled_data_predictions_merged_with_descriptions.csv')
        merged_df.to_csv(csv_path, index=False)
if __name__ == '__main__':
    # create_unlabled_merge_csv()
    descriptions_df = pd.read_csv(os.path.join(paths.BIOPEP_UWMix_path, 'BIOPEP_sequences_descriptions.csv'))
    predictions_df = pd.read_csv(os.path.join(paths.result_analysis_path,'unlabled_data_predictions_merged.csv'))
    add_descriptions(predictions_df,descriptions_df)


