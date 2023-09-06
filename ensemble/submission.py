import glob
import json
import os
import re
from datetime import datetime

import pandas as pd
from termcolor import colored


def write_ensemble_report(task, shot, exp, selected_models_for_classes, model_occurrences, root_report_dir):
    """
    Write the ensemble report for a given task, shot, and experiment.

    Args:
        task (str): The task name.
        shot (str): The shot name.
        exp (str): The experiment name.
        selected_models_for_classes (list): List containing the selected models for each class.
        root_report_dir (str): Root directory where the report.txt should be saved.
    """

    # Determine the path for the report.txt file
    report_path = os.path.join(root_report_dir, "report.txt")

    # Append the information to the report.txt file
    with open(report_path, "a") as report_file:
        report_file.write(f"Task: {task}, Shot: {shot}, Experiment: {exp}\n")
        for item in selected_models_for_classes:
            report_file.write(item + "\n")
        report_file.write("\n")

        # Writing model occurrences
        report_file.write("\nModel Summary:\n")
        for model_path, occurrence in model_occurrences.items():
            if occurrence > 1:
                report_file.write(f"{model_path} used {occurrence} times\n")
        report_file.write("\n")



def extract_exp_number(string):
    match = re.search(r'exp(\d+)', string)
    return int(match.group(1)) if match else 0


def merge_results_weighted_average_strategy(run_dicts, task, shot, exp):
    pass


def print_metric_report_for_task(model_list, task):
    print("Report for:", colored(os.path.join(task.capitalize(), shot, exp), 'blue'))

    model_view = []
    for model_info in model_list:
        model_path_rel = model_info['name'].split('work_dirs/')[1]

        agg_name, agg_val = get_aggregate(model_metrics=model_info['metrics'], task=task)
        if agg_val is not None:
            model_view.append((model_path_rel, agg_name, agg_val))

    model_view.sort(key=lambda x: x[2])
    max_char_length = max(len(path) for path, _, _ in model_view)

    for model_path_rel, agg_name, agg_val in model_view:
        print(f"Model: {model_path_rel:{max_char_length}}  {agg_name}: {agg_val:.4f}")


def get_aggregate(model_metrics, task):
    # Dictionary mapping tasks to lambda functions for aggregate calculation
    aggregate_calculations = {
        "colon": lambda metrics: ("AUC-Acc", float((metrics["AUC/AUC_multiclass"] + metrics[
            "accuracy/top1"]) / 2)) if "AUC/AUC_multiclass" in metrics and "accuracy/top1" in metrics else (None, None),
        "chest": lambda metrics: ("AUC-mAP", float((metrics["AUC/AUC_multilabe"] + metrics[
            "multi-label/mAP"]) / 2)) if "AUC/AUC_multilabe" in metrics and "multi-label/mAP" in metrics else (
            None, None),
        "endo": lambda metrics: ("AUC-mAP", float((metrics["AUC/AUC_multilabe"] + metrics[
            "multi-label/mAP"]) / 2)) if "AUC/AUC_multilabe" in metrics and "multi-label/mAP" in metrics else (
            None, None),
    }

    # Get the appropriate aggregate calculation for the task
    calculation = aggregate_calculations.get(task)

    # If there's no calculation for the task, return None for both metric name and value
    if not calculation:
        return (None, None)

    # Calculate and return the aggregate name and value
    return calculation(model_metrics)


def choose_evaluation_type():
    """Prompt the user to select between evaluation or validation.
    Returns True for Evaluation and False for Validation."""

    user_input = input(f"{colored('Evaluation (e)', 'red')} or {colored('Validation (v)', 'blue')}? (Default = e) ").lower() or 'e'

    if user_input == 'e':
        return True
    elif user_input == 'v':
        return False
    else:
        print(f"Invalid choice! Defaulting to Evaluation.")
        return True


# Find the run with the best MAP for a given class, within a list of runs
def find_best_run(run_list, metric):
    best_run_index = 0
    best_run = run_list[0]
    best_run_score = run_list[0]['metrics'][metric]
    for index, run in enumerate(run_list[1:]):
        if run['metrics'][metric] > best_run_score:
            best_run_index = index + 1
            best_run = run
            best_run_score = run['metrics'][metric]
    return best_run, best_run_index


def merge_results_expert_model_strategy(run_dicts, task, shot, exp, out_path):
    print("Merging results for task", task, shot, exp)
    num_classes = class_counts[task]
    merged_df = run_dicts[0]['prediction'].iloc[:, 0:1]

    # List to store which model was selected for each class
    selected_models_for_classes = []

    # Dict to keep track of model occurrences
    model_occurrences = {}

    # Find run with best MAP for each class
    for i in range(num_classes):
        best_run, best_run_index = find_best_run(run_dicts, f'MAP_class{i + 1}')
        merged_df[i + 1] = best_run["prediction"][i + 1]

        # Keeping track of the model used for each class
        model_name = best_run['name']
        if model_name in model_occurrences:
            model_occurrences[model_name] += 1
        else:
            model_occurrences[model_name] = 1

        print(f"Merged dataframe after adding model run {best_run_index} {best_run['name']}")
        selected_models_for_classes.append(f"Class {i + 1}: {best_run['name']}")

    print(f"Saving merged prediction to {out_path}")
    merged_df.to_csv(out_path, index=False, header=False)
    # Merge predictions using class columns from best runs, taking into account first column is image name, no prediction
    # for that column
    return selected_models_for_classes, model_occurrences


def extract_data_tuples(run_list):
    data_list = []
    for run in run_list:
        prediction = pd.read_csv(run['csv'], header=None)
        metrics = json.load(open(run['json'], 'r'))
        data_list.append({'prediction': prediction, 'metrics': metrics, 'name': run['name']})
    return data_list


def check_run_dir(run_dir, exp_dirs, task, shot, subm_type):
    model_path = run_dir.split('work_dirs/')[1]
    print("Checking run directory", model_path)
    csv_path = os.path.join(run_dir, f"{task}_{shot}_{subm_type}.csv")
    csv_files = glob.glob(csv_path)
    json_files = glob.glob(os.path.join(run_dir, "*.json"))

    if csv_files and json_files:
        exp_num = extract_exp_number(run_dir)
        if exp_num != 0:
            print(f"Adding run: {run_dir}")
            exp_dirs[task][shot][f"exp{exp_num}"].append(
                {'csv': csv_files[0], 'json': json_files[0], 'name': run_dir})


# Setup
root_dir = "/scratch/medfm/medfm-challenge/work_dirs"
exp_dirs = {}
tasks = ['endo', 'chest', 'colon']
shots = ['1-shot', '5-shot', '10-shot']
experiments = ['exp1', 'exp2', 'exp3', 'exp4', 'exp5']
class_counts = {"colon": 2, "endo": 4, "chest": 19}

is_evaluation = choose_evaluation_type()
submission_type = 'submission'
if is_evaluation:
    print(f"\nSelected {colored(submission_type.capitalize(), 'red')}\n")
else:
    submission_type = 'validation'
    print(f"\nSelected {colored(submission_type.capitalize(), 'blue')}\n")

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
            check_run_dir(run_dir=run_dir,
                          exp_dirs=exp_dirs,
                          task=task,
                          shot=shot,
                          subm_type=submission_type)

# Count
total_models = 0
least_models = 100000
most_models = -1
most_setting = ""
least_setting = ""
for task in tasks:
    for shot in shots:
        for exp in experiments:
            models_for_setting = len(exp_dirs[task][shot][exp])
            print(f"{task} {shot} {exp} {models_for_setting}")
            total_models += models_for_setting
            if models_for_setting > most_models:
                most_models = models_for_setting
                most_setting = f"{task} {shot} {exp}"
            if models_for_setting < least_models:
                least_models = models_for_setting
                least_setting = f"{task} {shot} {exp}"

print("--------------------------------------")
print(f"| Total models: {total_models}")
print(f"| Most models: {most_models} {most_setting}")
print(f"| Least models: {least_models} {least_setting}")
print("--------------------------------------")

start = input("Continue? (y/n) ")
if start != "y":
    exit()

# Create submission directory
date_pattern = datetime.now().strftime("%d-%m_%H-%M-%S")
submission_dir = os.path.join("submissions", "evaluation", date_pattern)
if not is_evaluation:
    submission_dir = os.path.join("ensemble", f"{submission_type}", date_pattern)
    print(f"Creating {submission_type} directory {submission_dir}")
else:
    print(f"Creating submission directory {submission_dir}")

os.makedirs(submission_dir)
for exp in experiments:
    os.makedirs(os.path.join(submission_dir, "result", f"{exp}"), exist_ok=True)

# iterate over exp_dirs_dict, for each task / shot / exp combination, merge results
for task in tasks:
    for shot in shots:
        for exp in experiments:
            if len(exp_dirs[task][shot][exp]) < 2:
                print("not enough runs")
                continue
            out_path = os.path.join(submission_dir, "result", f"{exp}", f"{task}_{shot}_{submission_type}.csv")
            data_list = extract_data_tuples(exp_dirs[task][shot][exp])

            print_metric_report_for_task(model_list=data_list, task=task)

            # Insert ensemble strategy here
            selected_models, model_occurrences = merge_results_expert_model_strategy(data_list, task, shot, exp, out_path)

            write_ensemble_report(task, shot, exp, selected_models, model_occurrences, submission_dir)
