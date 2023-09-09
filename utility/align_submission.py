import pandas as pd
import os
from datetime import datetime


"""
Aligns all .csv files in the results folder, such that the image IDs are in the correct order, corresponding to 
chest_val.csv / colon_val.csv / endo_val.csv
"""

experiments = ["exp1", "exp2", "exp3", "exp4", "exp5"]
images_dir = "data/MedFMC_test/"
path = os.path.join('submissions', 'evaluation')
directories = [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]
format = "%d-%m_%H-%M-%S"
valid_directories = []
for d in directories:
    try:
        valid_directories.append((datetime.strptime(d, format), d))
    except Exception:
        pass

newest_directory = max(valid_directories, key=lambda x: x[0])[1]
print(f"Aligning newest submission {newest_directory}")
results_dir = os.path.join(path, newest_directory, "result")

for exp in experiments:
    exp_dir = os.path.join(results_dir, exp)

    csv_files = [file for file in os.listdir(exp_dir) if file.endswith('.csv')]

    for file in csv_files:
        submission_csv_path = os.path.join(exp_dir, file)

        # Read test_WithoutLabel.txt and remove rows without image ids
        task = file.split("_")[0]
        df_val_order = pd.read_csv(f"{images_dir}{task}/test_WithoutLabel.txt", header=None, names=['img_id'])
        df_val_order.dropna(inplace=True)

        # Read generated submission csv, name first column 'img_id' for easier merge
        df_submission = pd.read_csv(submission_csv_path, header=None).rename(columns={0: 'img_id'})

        # Reorder the rows of df_submission to match those of the given validation csv by merging on 'img_id'
        result = df_val_order.merge(df_submission, on='img_id')
        result = result[df_submission.columns]

        # Check if final result is correct
        for i in range(len(df_val_order)):
            if result["img_id"][i] != df_val_order["img_id"][i]:
                print("Something went wrong, aborting")
                exit()
        result.to_csv(submission_csv_path, header=False, index=False)

print("Successfully aligned submission files to their respective {task}_val.csv")
