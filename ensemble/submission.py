import glob
import json
import os
import re
from datetime import datetime

import pandas as pd
from termcolor import colored
from utils.constants import shots, tasks, exps, TASK_2_CLASS_COUNT


def create_ensemble_report_file(task, shot, exp, selected_models_for_classes, model_occurrences, root_report_dir):
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


def weighted_ensemble_strategy(model_runs, task, shot, exp, out_path, k=3):
    """
    Merges model runs using a weighted sum approach based on the N best model runs for each class.
    """
    print("Merging results for task", task, shot, exp)
    num_classes = TASK_2_CLASS_COUNT[task]
    merged_df = model_runs[0]['prediction'].iloc[:, 0:1].copy()

    # List to store which models were selected for each class
    selected_models_for_classes = []

    # Dict to keep track of model occurrences
    model_occurrences = {}

    # For each class, get the N best performing model runs based on the aggregate metric
    for i in range(num_classes):
        class_models = []
        for run in model_runs:
            _, aggregate_value = get_aggregate(run['metrics'], task)
            if aggregate_value is not None:
                class_models.append((run, aggregate_value))

        # Sort the models based on aggregate value and take the top N models
        class_models.sort(key=lambda x: x[1], reverse=True)
        top_n_models = class_models[:k]

        # Record the selected models for the report and update the model_occurrences
        selected_models_for_class = []
        for model, weight in top_n_models:
            model_name = model['name'].split('work_dirs/')[1]
            selected_models_for_class.append(f"Class {i + 1}: {model_name} (Weight: {weight:.4f})")

            if model_name in model_occurrences:
                model_occurrences[model_name] += 1
            else:
                model_occurrences[model_name] = 1

        selected_models_for_classes.extend(selected_models_for_class)

        # Calculate the sum of weights (aggregate values) for normalization
        sum_weights = sum([weight for _, weight in top_n_models])

        # Compute the weighted sum for this class
        weighted_sum_column = pd.Series(0, index=merged_df.index)
        for model, weight in top_n_models:
            weighted_sum_column += (model['prediction'].iloc[:, i + 1] * weight) / sum_weights

        merged_df.loc[:, i + 1] = weighted_sum_column

    print(f"Saving merged prediction to {out_path}")
    merged_df.to_csv(out_path, index=False, header=False)

    return selected_models_for_classes, model_occurrences


def print_report_for_setting(full_model_list, task, shot, exp):
    print("\nReport for:", colored(os.path.join(task.capitalize(), shot, exp), 'blue'))

    model_view = []
    for model_info in full_model_list[task][shot][exp]:
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


def expert_model_strategy(model_runs, task, shot, exp, out_path):
    print("Merging results for task", task, shot, exp)
    num_classes = TASK_2_CLASS_COUNT[task]
    merged_df = model_runs[0]['prediction'].iloc[:, 0:1]

    # List to store which model was selected for each class
    selected_models_for_classes = []

    # Dict to keep track of model occurrences
    model_occurrences = {}

    # Find run with best MAP for each class
    for i in range(num_classes):
        best_run, best_run_index = find_best_run(model_runs, f'MAP_class{i + 1}')
        merged_df[i + 1] = best_run["prediction"][i + 1]

        # Keeping track of the model used for each class
        model_name = best_run['name'].split('work_dirs/')[1]
        if model_name in model_occurrences:
            model_occurrences[model_name] += 1
        else:
            model_occurrences[model_name] = 1

        print(f"Merged dataframe after adding model run {best_run_index} {model_name}")
        selected_models_for_classes.append(f"Class {i + 1}: {model_name}")

    print(f"Saving merged prediction to {out_path}")
    merged_df.to_csv(out_path, index=False, header=False)

    return selected_models_for_classes, model_occurrences


def extract_data_tuples_from_model_runs(run_list):
    data_list = []
    for run in run_list:
        prediction = pd.read_csv(run['csv'], header=None)
        metrics = json.load(open(run['json'], 'r'))
        data_list.append({'prediction': prediction, 'metrics': metrics, 'name': run['name']})
    return data_list


def check_and_extract_data(model_dir_abs, subm_type, task, shot):
    model_dir_rel = model_dir_abs.split('work_dirs/')[1]
    print("Checking run directory", model_dir_rel)

    csv_path = os.path.join(model_dir_abs, f"{task}_{shot}_{subm_type}.csv")
    csv_files = glob.glob(csv_path)
    json_files = glob.glob(os.path.join(model_dir_abs, "*.json"))

    if csv_files and json_files:
        exp_num = extract_exp_number(model_dir_rel)
        if exp_num != 0:
            prediction = pd.read_csv(csv_files[0], header=None)
            metrics = json.load(open(json_files[0], 'r'))
            return {'prediction': prediction, 'metrics': metrics, 'name': model_dir_rel}, exp_num
    return None, None


def create_submission(is_evaluation):
    submission_type = 'submission'
    if is_evaluation:
        print(f"\nSelected {colored(submission_type.capitalize(), 'red')}\n")
    else:
        submission_type = 'validation'
        print(f"\nSelected {colored(submission_type.capitalize(), 'blue')}\n")

    total_models = 0
    least_models = 100000
    most_models = -1
    most_setting = ""
    least_setting = ""

    data_lists = {}
    for task in tasks:
        data_lists[task] = {}
        for shot in shots:
            data_lists[task][shot] = {}
            for exp in exps:
                data_lists[task][shot][exp] = []

                path_pattern = os.path.join(root_dir, task, shot, '*exp[1-5]*')
                for model_dir in glob.glob(path_pattern):
                    data, exp_num = check_and_extract_data(model_dir, submission_type, task=task, shot=shot)
                    if data and exp_num:
                        data_lists[task][shot][f"exp{exp_num}"].append(data)

                # Count and compare
                models_for_setting = len(data_lists[task][shot][exp])
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

    # Print Setting Reports
    continue_query = input("\nPrint report for the best models? (y/n) ")
    if continue_query.lower() == "y":
        for task in tasks:
            for shot in shots:
                for exp in exps:
                    print_report_for_setting(full_model_list=data_lists, task=task, shot=shot, exp=exp)

    # Create Output Directory
    continue_query = input(f"\nCreate {submission_type} directory? (y/n) ")
    submission_dir = os.path.join("submissions", "evaluation", TIMESTAMP)
    if is_evaluation:
        print(f"Creating submission directory {submission_dir}")
    else:
        submission_dir = os.path.join("ensemble", f"{submission_type}", TIMESTAMP)
        print(f"Creating {submission_type} directory {submission_dir}")
    if continue_query.lower() == "y":
        os.makedirs(submission_dir)
        for exp in exps:
            os.makedirs(os.path.join(submission_dir, "result", f"{exp}"), exist_ok=True)

    # Perform Ensemble Strategy
    continue_query = input(f"\nPerform ensemble merge strategy? (y/n) ")
    if continue_query.lower() == "y":
        for task in tasks:
            for shot in shots:
                for exp in exps:
                    model_runs = data_lists[task][shot][exp]
                    if len(model_runs) < 2:
                        print("Not enough runs")
                        continue
                    out_path = os.path.join(submission_dir, "result", f"{exp}", f"{task}_{shot}_{submission_type}.csv")

                    if ENSEMBLE_STRATEGY == "weighted":
                        selected_models, model_occurrences = weighted_ensemble_strategy(model_runs=model_runs,
                                                                                        task=task, shot=shot, exp=exp,
                                                                                        k=TOP_K, out_path=out_path)
                    elif ENSEMBLE_STRATEGY == "expert":
                        selected_models, model_occurrences = expert_model_strategy(model_runs=model_runs,
                                                                                   task=task, shot=shot, exp=exp,
                                                                                   out_path=out_path)
                    else:
                        print("Invalid ensemble strategy!")
                        exit()

                    create_ensemble_report_file(task=task,shot=shot,exp=exp,
                                                selected_models_for_classes=selected_models,
                                                model_occurrences=model_occurrences,
                                                root_report_dir=submission_dir)


def select_top_k_models():
    while True:
        top_k = input("Enter the number of top-k models for the weighted ensemble: ")
        if top_k.isdigit() and int(top_k) > 0:
            return int(top_k)
        else:
            print("Invalid number. Please enter a positive integer.\n")


def select_ensemble_strategy():
    while True:
        print("Choose an ensemble strategy:")
        for idx, strategy in enumerate(ENSEMBLE_STRATEGIES, 1):
            print(f"{idx}. {strategy}")

        choice = input("Enter the number corresponding to your choice: ")

        if choice.isdigit() and 1 <= int(choice) <= len(ENSEMBLE_STRATEGIES):
            choice = ENSEMBLE_STRATEGIES[int(choice) - 1]

            if choice == "weighted":
                top_k_models = select_top_k_models()
                return choice, top_k_models
            else:
                return choice, None
        else:
            print("Invalid choice. Please try again.\n")


# ============================================================
root_dir = "/scratch/medfm/medfm-challenge/work_dirs"
ENSEMBLE_STRATEGIES = ["expert", "weighted"]
# ============================================================


if __name__ == "__main__":
    ENSEMBLE_STRATEGY, TOP_K = select_ensemble_strategy()
    TIMESTAMP = datetime.now().strftime("%d-%m_%H-%M-%S")

    create_submission(is_evaluation=True)
    #create_submission(is_evaluation=False)
