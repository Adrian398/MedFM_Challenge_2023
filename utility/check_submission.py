import os
from datetime import datetime
import pandas as pd

"""
Checks if all necessary files for submission are in the specified results_dir, checks if they are named correctly,
if they have the right amount of entries, and if the order of image IDs corresponds to the order of IDs in the 
{task}_val.csv, which is a requirement for submission.
"""

experiments = ["exp1", "exp2", "exp3", "exp4", "exp5"]
tasks = ["endo", "colon", "chest"]
n_shots = ["1", "5", "10"]
images_dir = "data/MedFMC_test/"

path = os.path.join('submissions', 'evaluation')

# Get newest submission
directories = [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]
format = "%d-%m_%H-%M-%S"
valid_directories = []
for d in directories:
    try:
        valid_directories.append((datetime.strptime(d, format), d))
    except Exception:
        pass

newest_directory = max(valid_directories, key=lambda x: x[0])[1]
print(f"Checking newest submission {newest_directory}")
predictions_dir = os.path.join(path, newest_directory, "result")

required_file_names = [f"{task}_{n}-shot_submission.csv" for task in tasks for n in n_shots]

file_names_correct = True
file_orders_correct = True
file_lengths_correct = True

for exp in experiments:
    exp_dir = os.path.join(predictions_dir, exp)

    csv_files = [file for file in os.listdir(exp_dir) if file.endswith('.csv')]

    for file in csv_files:
        if file not in required_file_names:
            print(f"Incorrect filename: {file} expected one of {required_file_names}")
            file_names_correct = False
    for file in required_file_names:
        if file not in csv_files:
            print(f"Missing file: {file}")
            file_names_correct = False

    for file in csv_files:
        submission_csv_path = os.path.join(exp_dir, file)

        # Read test_WithoutLabel.txt and remove rows without image ids
        task = file.split("_")[0]
        df_test_order = pd.read_csv(f"{images_dir}{task}/test_WithoutLabel.txt", header=None, names=['img_id'])
        df_test_order.dropna(inplace=True)

        # Read generated submission csv
        df_submission = pd.read_csv(submission_csv_path, header=None)

        if len(df_test_order) != len(df_submission):
            file_lengths_correct = False
            print(f"Incorrect number of entries in {file}, was {len(df_submission['img_id'])}, "
                  f"expected {len(df_test_order['img_id'])}")
            print("You might have done the inference on the wrong image folder (e.g. on train instead of val)")
        # Check if final result is correct
        wrong_order_in_file = False
        for i in range(len(df_test_order)):
            if df_submission[0][i] != df_test_order["img_id"][i]:
                wrong_order_in_file = True
                file_orders_correct = False

        if wrong_order_in_file:
            print(f"Wrong order in file {file}, use align_submission.py to fix this.")

if file_names_correct and file_orders_correct and file_lengths_correct:
    print("Submission complete and in correct format")
