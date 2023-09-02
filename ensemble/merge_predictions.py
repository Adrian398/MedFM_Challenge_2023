import os
from datetime import datetime

import pandas as pd

"""
Aligns all .csv files in the results folder, such that the image IDs are in the correct order, corresponding to 
chest_val.csv / colon_val.csv / endo_val.csv
"""

tasks = ["endo", "chest", "colon"]
shots = ["1-shot", "5-shot", "10-shot"]
experiments = ["exp1", "exp2", "exp3", "exp4", "exp5"]
images_dir = "data/MedFMC_test/"
path = os.path.join('submissions', 'evaluation')

# todo remove below
path = os.path.join('backup')

directories = [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]
format = "%d-%m_%H-%M-%S"
valid_directories = []
for d in directories:
    try:
        valid_directories.append((datetime.strptime(d, format), d))
    except Exception:
        pass

newest_directory = max(valid_directories, key=lambda x: x[0])[1]
print(f"Merging newest submission {newest_directory}")
newest_directory = os.path.join(path, newest_directory)


def merge_predictions_weighted_by_auc(input_directory, output_file, submission_file_name):
    print(f"checking {input_directory}, writing to {output_file}")
    # List all model directories in the input directory

    model_dirs = [d for d in os.listdir(input_directory) if os.path.isdir(os.path.join(input_directory, d))]

    # Initialize a dictionary to store the sum of weighted predictions
    sum_weighted_predictions = None

    # Initialize a dictionary to store the sum of weights
    sum_weights = None

    for model_dir in model_dirs:
        # Read the AUC scores from the auc_scores.txt file
        with open(os.path.join(input_directory, model_dir, 'auc_scores.txt'), 'r') as f:
            auc_scores = [float(score) for score in f.read().split(',')]

        # Read the predictions from the prediction.csv file without headers
        predictions = pd.read_csv(os.path.join(input_directory, model_dir, submission_file_name), header=None)
        print(predictions)

        # Remove the 'image_id' column (first column) and multiply by the AUC scores
        weighted_predictions = predictions.drop(columns=0).multiply(auc_scores, axis=1)
        print("---------------")
        print(weighted_predictions)

        # If the sum_weighted_predictions dictionary is not initialized, do so now
        if sum_weighted_predictions is None:
            sum_weighted_predictions = weighted_predictions
            sum_weights = pd.DataFrame(data=[auc_scores], columns=weighted_predictions.columns)
        else:
            sum_weighted_predictions += weighted_predictions
            sum_weights += auc_scores

    # Compute the final predictions by dividing the sum of weighted predictions by the sum of weights
    final_predictions = sum_weighted_predictions.divide(sum_weights.iloc[0], axis=1)
    final_predictions[0] = predictions[0]
    final_predictions = final_predictions[[0] + [col for col in final_predictions if col != 0]]

    # Save the final predictions to the output file without headers
    final_predictions.to_csv(output_file, index=False, header=False)


for task in tasks:
    for shot in shots:
        for exp in experiments:
            setting_path = os.path.join(newest_directory, "tmp", task, shot, exp)
            destination_path = os.path.join(newest_directory, "result", exp)
            os.makedirs(destination_path, exist_ok=True)
            destination_file = os.path.join(destination_path, f"{task}_{shot}_submission.csv")
            submission_file_name = f"{task}_{shot}_{exp}_submission.csv"
            merge_predictions_weighted_by_auc(setting_path, destination_file, submission_file_name)
            exit()