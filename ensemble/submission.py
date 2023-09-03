import glob
import json
import os
import re

import pandas as pd

root_dir = "/scratch/medfm/medfm-challenge/work_dirs"


def extract_exp_number(string):
    match = re.search(r'exp(\d+)', string)
    return int(match.group(1)) if match else 0


exp_dirs = {}
tasks = ['colon', 'endo', 'chest']
shots = ['1-shot', '5-shot', '10-shot']
exps = ['exp1', 'exp2', 'exp3', 'exp4', 'exp5']
# Traverse through the main categories
for task in tasks:
    # Traverse through 1-shot, 5-shot, 10-shot
    exp_dirs[task] = {}
    for shot in shots:
        exp_dirs[task][shot] = {}
        for exp in exps:
            exp_dirs[task][shot][exp] = []

        # Construct the path pattern for glob
        path_pattern = os.path.join(root_dir, task, shot, '*exp[1-5]*')
        # Get all run directories that match the pattern
        for run_dir in glob.glob(path_pattern):
            print("Checking run directory", run_dir)
            # Check if both csv and json files exist in the run directory
            csv_files = glob.glob(os.path.join(run_dir, "*.csv"))
            json_files = glob.glob(os.path.join(run_dir, "*.json"))
            if csv_files and json_files:
                exp_num = extract_exp_number(run_dir)
                if exp_num != 0:
                    exp_dirs[task][shot][f"exp{exp_num}"].append({'csv': csv_files[0], 'json': json_files[0]})

class_lengths = {"colon": 2, "endo": 4, "chest": 19}


def merge_results_weighted_average_strategy(run_dicts, task, shot, exp):
    pass


def merge_results_expert_model_strategy(run_dicts, task, shot, exp):
    print("merging results for task", task, "shot", shot, "exp", exp)
    num_classes = class_lengths[task]
    # initialize dataframe with image_ids
    merged_df = run_dicts[0][:, 0:1]
    print("Merged df before")
    print(merged_df)
    # Find run with best MAP for each class
    for i in range(num_classes):
        best_run = max(run_dicts, key=lambda x: x['metrics'][f'MAP_class{i+1}'])
        merged_df[i+1] = best_run["prediction"][i+1]
        print("Merged df after adding")
        print(merged_df)
    exit()
    # Merge predictions using class columns from best runs, taking into account first column is image name, no prediction
    # for that column




def extract_data_tuples(run_list):
    data_list = []
    for run in run_list:
        prediction = pd.read_csv(run['csv'], header=None)
        metrics = json.load(open(run['json'], 'r'))
        data_list.append({"prediction": prediction, "metrics": metrics})
    return data_list


for task in tasks:
    for shot in shots:
        for exp in exps:
            data_list = extract_data_tuples(exp_dirs[task][shot][exp])
            merge_results_expert_model_strategy(data_list, task, shot, exp)
