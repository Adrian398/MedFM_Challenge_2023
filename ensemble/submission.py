import glob
from datetime import datetime
import json
import os
import re

import pandas as pd


def extract_exp_number(string):
    match = re.search(r'exp(\d+)', string)
    return int(match.group(1)) if match else 0


def merge_results_weighted_average_strategy(run_dicts, task, shot, exp):
    pass


def merge_results_expert_model_strategy(run_dicts, task, shot, exp, out_path):
    print("merging results for task", task, shot, exp)
    num_classes = class_counts[task]
    # initialize dataframe with image_ids
    merged_df = run_dicts[0]['prediction'].iloc[:, 0:1]
    print("Merged df before")
    print(merged_df)
    # Find run with best MAP for each class
    for i in range(num_classes):
        print("-------------------------------------------------------------------type run dicts")
        print(type(run_dicts))
        print("-------------------------------------------------------------------")
        best_run = max(run_dicts, key=lambda x: x['metrics'][f'MAP_class{i + 1}'])
        best_run_index = run_dicts.index(best_run)
        merged_df[i + 1] = best_run["prediction"][i + 1]
        print(f"Merged df after adding run {best_run_index} {best_run['name']}")
    print(f"Saving merged_df to {out_path}")
    merged_df.to_csv(out_path, index=False, header=False)
    exit()
    # Merge predictions using class columns from best runs, taking into account first column is image name, no prediction
    # for that column


def extract_data_tuples(run_list):
    data_list = []
    for run in run_list:
        prediction = pd.read_csv(run['csv'], header=None)
        metrics = json.load(open(run['json'], 'r'))
        data_list.append({'prediction': prediction, 'metrics': metrics, 'name': run['name']})
    return data_list


def check_run_dir(run_dir, exp_dirs, task, shot):
    print("Checking run directory", run_dir)
    csv_files = glob.glob(os.path.join(run_dir, "*.csv"))
    json_files = glob.glob(os.path.join(run_dir, "*.json"))

    if csv_files and json_files:
        exp_num = extract_exp_number(run_dir)
        if exp_num != 0:
            exp_dirs[task][shot][f"exp{exp_num}"].append(
                {'csv': csv_files[0], 'json': json_files[0], 'name': run_dir})


# Setup
root_dir = "/scratch/medfm/medfm-challenge/work_dirs"
exp_dirs = {}
tasks = ['endo', 'chest', 'colon']
shots = ['1-shot', '5-shot', '10-shot']
experiments = ['exp1', 'exp2', 'exp3', 'exp4', 'exp5']
class_counts = {"colon": 2, "endo": 4, "chest": 19}

# Create submission directory
date_pattern = datetime.now().strftime("%d-%m_%H-%M-%S")
submission_dir = os.path.join("submissions", "evaluation", date_pattern)
print(f"Creating submission directory {submission_dir}")
os.makedirs(submission_dir)
for exp in experiments:
    os.makedirs(os.path.join(submission_dir, "result", f"exp{exp}"), exist_ok=True)

# For each task / shot / experiment combination, find all directories that contain both a csv and json file, and
# add them to the exp_dirs dictionary with keys csv and json
# csv = prediction, json = metrics
for task in tasks:
    exp_dirs[task] = {}
    for shot in shots:
        exp_dirs[task][shot] = {}
        for exp in experiments:
            exp_dirs[task][shot][exp] = []
        path_pattern = os.path.join(root_dir, task, shot, '*exp[1-5]*')
        # Get all run directories that match the pattern
        for run_dir in glob.glob(path_pattern):
            # check if run dir has json + csv, if yes, add info to exp_dirs dict
            check_run_dir(run_dir, exp_dirs, task, shot)

# iterate over exp_dirs_dict, for each task / shot / exp combination, merge results
for task in tasks:
    for shot in shots:
        for exp in experiments:
            if len(exp_dirs[task][shot][exp]) < 2:
                print("not enough runs")
                continue
            out_path = os.path.join(submission_dir, "result", f"exp{exp}", f"{task}_{shot}_submission.csv")
            data_list = extract_data_tuples(exp_dirs[task][shot][exp])
            merge_results_expert_model_strategy(data_list, task, shot, exp, out_path)