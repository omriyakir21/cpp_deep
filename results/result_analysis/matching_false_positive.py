import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..','..'))
import paths
import json

def find_matching_components(
    file1_path, file2_path, file3_path, file4_path, output_file_name="matching_components.json"
):
    with open(file1_path, 'r') as file1:
        data1 = json.load(file1)
    with open(file2_path, 'r') as file2:
        data2 = json.load(file2)
    with open(file3_path, 'r') as file3:
        data3 = json.load(file3)
    with open(file4_path, 'r') as file4:
        data4 = json.load(file4)

    predictions1 = {item[2]: item[0] for item in data1}
    predictions2 = {item[2]: item[0] for item in data2}
    predictions3 = {item[2]: item[0] for item in data3}
    predictions4 = {item[2]: item[0] for item in data4}

    sequences1 = set(predictions1.keys())
    sequences2 = set(predictions2.keys())
    sequences3 = set(predictions3.keys())
    sequences4 = set(predictions4.keys())

    matching_sequences = sequences1 & sequences2 & sequences3 & sequences4

    results = []
    for seq in matching_sequences:
        row = {
            "Sequence": seq,
            "Few Shot Prediction": predictions1[seq],
            "Convolution Prediction": predictions2[seq],
            "LoRA Prediction": predictions3[seq],
            "Fine Tune Prediction": predictions4[seq],
        }
        row["Average Prediction"] = sum(
            [row["Few Shot Prediction"], row["Convolution Prediction"], row["LoRA Prediction"], row["Fine Tune Prediction"]]
        ) / 4
        results.append(row)

    results = sorted(results, key=lambda x: x["Average Prediction"], reverse=True)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(script_dir, output_file_name)
    with open(output_path, 'w') as output_file:
        json.dump(results, output_file, indent=4)

    print(f"Matching components saved to {output_path}")


if __name__ == "__main__":
    few_shot_path = "/home/iscb/wolfson/omriyakir/cpp_deep/results/esm2/few_shot_learning/13_09/esm2_t6_8M_UR50D/top_false_positives.json"
    convolution_path = "/home/iscb/wolfson/omriyakir/cpp_deep/results/baselines/convolution_baseline/13_09/top_false_positives.json"
    lora_path = "/home/iscb/wolfson/omriyakir/cpp_deep/results/esm2/lora/13_09/esm2_t6_8M_UR50D/top_false_positives.json"
    fine_tune_path = "/home/iscb/wolfson/omriyakir/cpp_deep/results/esm2/fine_tune/13_09/esm2_t6_8M_UR50D/top_false_positives.json"

    find_matching_components(few_shot_path, convolution_path, lora_path, fine_tune_path)
    