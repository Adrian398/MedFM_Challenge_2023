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

    # Read generated submission csv (as one column, since infer generates white-space separation)
    df_submission = pd.read_csv(submission_csv_path, header=None)

    # Split the last two numbers off to generate proper columns
    classes = 2 if task == "colon" else 19 if task == "chest" else 4

    for i in range(classes):
        df_submission[f"pred_class_{i}"] = df_submission[0].str.rsplit(' ', n=classes).str[-(classes-i)]

    # Extract ID, which is everything to the left of the last two numbers
    df_submission['img_id'] = df_submission[0].str.rsplit(' ', n=classes).str[0]

    required_columns = ['img_id']
    for i in range(classes):
        required_columns.append(f"pred_class_{i}")
    # Rearrange columns and drop the original combined column
    df_submission = df_submission[required_columns]

    # Reorder the rows of df_submission to match those of the given validation csv
    result = df_val_order.merge(df_submission, on='img_id')
    result = result[df_submission.columns]

    # Check if final result is correct
    for i in range(len(df_val_order)):
        if result["img_id"][i] != df_val_order["img_id"][i]:
            print("Something went wrong, aborting")
            exit()
    csv_content = result.apply(lambda row: ' '.join(map(str, row)), axis=1).str.cat(sep='\n')

    # Write the formatted content back to the CSV file
    with open(submission_csv_path, 'w') as f:
        f.write(csv_content)

print("Successfully aligned submission files to their respective {task}_val.csv")
