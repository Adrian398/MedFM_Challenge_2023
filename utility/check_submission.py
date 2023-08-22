import os
from datetime import datetime

import pandas as pd

"""
Checks if all necessary files for submission are in the specified results_dir, checks if they are named correctly,
if they have the right amount of entries, and if the order of image IDs corresponds to the order of IDs in the 
{task}_val.csv, which is a requirement for submission.
"""

tasks = ["endo", "colon", "chest"]
n_shots = ["1", "5", "10"]
val_dir = "../data/MedFMC_val/"

path = os.path.join('..', 'submissions')
directories = [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]
format = "%d-%m_%H-%M-%S"

newest_directory = max(directories, key=lambda d: datetime.strptime(d, format))
print(f"Checking newest submission {newest_directory}")
results_dir = os.path.join(path, newest_directory, "predictions")

required_file_names = []
for task in tasks:
    for n in n_shots:
        required_file_names.append(f"{task}_{n}-shot_submission.csv")

csv_files = [file for file in os.listdir(results_dir) if file.endswith('.csv')]

file_names_correct = True
for file in csv_files:
    if file not in required_file_names:
        print(f"Incorrect filename: {file} expected one of {required_file_names}")
        file_names_correct = False
for file in required_file_names:
    if file not in csv_files:
        print(f"Missing file: {file}")
        file_names_correct = False

file_orders_correct = True
file_lengths_correct = True
for file in csv_files:
    submission_csv_path = os.path.join(results_dir, file)

    # Read colon_val.csv/endo_val.csv/chest_val.csv and remove rows without image ids
    task = file.split("_")[0]
    df_val_order = pd.read_csv(f"{val_dir}{task}/{task}_val.csv").dropna(subset="img_id")

    # Read generated submission csv (as one column, since infer generates white-space separation)
    df_submission = pd.read_csv(submission_csv_path, header=None)

    if len(df_val_order) != len(df_submission):
        file_lengths_correct = False
        print(f"Incorrect number of entries in {file}, was {len(df_submission['img_id'])}, "
              f"expected {len(df_val_order['img_id'])}")
        print("You might have done the inference on the wrong image folder (e.g. on train instead of val)")
    # Check if final result is correct
    wrong_order_in_file = False
    for i in range(len(df_val_order)):
        if df_submission[0][i] != df_val_order["img_id"][i]:
            wrong_order_in_file = True
            file_orders_correct = False

    if wrong_order_in_file:
        print(f"Wrong order in file {file}, use align_submission.py to fix this.")

if file_names_correct and file_orders_correct and file_lengths_correct:
    print("Submission complete and in correct format")
