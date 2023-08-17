import pandas as pd
import os


"""
Aligns all .csv files in the results folder, such that the image IDs are in the correct order, corresponding to 
chest_val.csv / colon_val.csv / endo_val.csv
"""

results_dir = "../results/"
val_dir = "../data/MedFMC_val/"

csv_files = [file for file in os.listdir(results_dir) if file.endswith('.csv')]

for file in csv_files:
    submission_csv_path = results_dir+file

    # Read colon_val.csv/endo_val.csv/chest_val.csv and remove rows without image ids
    task = file.split("_")[0]
    df_val_order = pd.read_csv(f"{val_dir}{task}/{task}_val.csv").dropna(subset="img_id")

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
