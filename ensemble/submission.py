import glob
import json
import os
import re
from datetime import datetime
from functools import lru_cache

import pandas as pd
from colorama import Fore
from termcolor import colored
from tqdm import tqdm

from utils.constants import shots, tasks, exps, TASK_2_CLASS_COUNT


def create_ensemble_report_file(task, shot, exp, is_eval, selected_models_for_classes, model_occurrences, root_report_dir):
    """
    Write the ensemble report for a given task, shot, and experiment.

    Args:
        task (str): The task name.
        shot (str): The shot name.
        exp (str): The experiment name.
        selected_models_for_classes (list): List containing the selected models for each class.
        model_occurrences (dict): Dict containing the occurences for the selected models.
        root_report_dir (str): Root directory where the report.txt should be saved.
    """
    # Determine the path for the report.txt file
    report_path = os.path.join(root_report_dir, "report.txt")

    # Append the information to the report.txt file
    with open(report_path, "a") as report_file:
        report_file.write(f"Setting: {task}/{shot}/{exp}\n")
        for item in selected_models_for_classes:
            report_file.write(item + "\n")
        report_file.write("\n")

        # Writing model occurrences
        report_file.write("Model Summary:\n")
        for model_path, occurrence in model_occurrences.items():
            if occurrence >= 1:
                report_file.write(f"{model_path} used {occurrence} times\n")
        report_file.write("\n\n")

    if is_eval:
        print("Added ensemble information for setting", colored(f"{task}/{shot}/{exp}", 'red'))
    else:
        print("Added ensemble information for", colored(f"{task}/{shot}/{exp}", 'blue'))


@lru_cache(maxsize=None)
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

    #print(f"Saving merged prediction to {out_path}")
    merged_df.to_csv(out_path, index=False, header=False)

    return selected_models_for_classes, model_occurrences


def print_report_for_setting(full_model_list, task, shot, exp):
    print("\nReport for:", colored(os.path.join(task.capitalize(), shot, exp), 'blue'))

    model_view = []
    for model_info in full_model_list[task][shot][exp]:
        agg_name, agg_val = get_aggregate(model_metrics=model_info['metrics'], task=task)
        if agg_val is not None:
            model_view.append((model_info['name'], agg_name, agg_val))

    model_view.sort(key=lambda x: x[2])
    max_char_length = max(len(path) for path, _, _ in model_view)

    for model_path_rel, agg_name, agg_val in model_view:
        print(f"Model: {model_path_rel:{max_char_length}}  {agg_name}: {agg_val:.4f}")


def print_model_reports():
    continue_query = input("\nPrint report for the best models? (y/n) ")
    if continue_query.lower() == "y":
        for task in tasks:
            for shot in shots:
                for exp in exps:
                    print_report_for_setting(full_model_list=DATA_SUBMISSION, task=task, shot=shot, exp=exp)



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


def compute_colon_aggregate(model_metrics, model_name):
    """
    Compute metrics for the 'colon' task.
    """
    auc_label = "AUC/AUC_multiclass"
    acc_label = "accuracy/top1"

    if auc_label not in model_metrics:
        print(f"Metric '{auc_label}' not found in {model_name}")
        return None

    if acc_label not in model_metrics:
        print(f"Metric '{acc_label}' not found in {model_name}")
        return None

    return (model_metrics[auc_label] + model_metrics[acc_label])/2


def compute_multilabel_aggregate_p_class(model_metrics, model_name, class_idx):
    """Compute metrics for multi-label tasks ('chest' and 'endo')."""
    auc_label = f"AUC/AUC_class{class_idx}"
    map_label = f"MAP_class{class_idx}"

    if auc_label not in model_metrics:
        print(f"Metric '{auc_label}' not found in {model_name}")
        return None

    if map_label not in model_metrics:
        print(f"Metric '{map_label}' not found in {model_name}")
        return None

    return (model_metrics[auc_label] + model_metrics[map_label])/2


# Find the run with the best MAP for a given class, within a list of runs
def find_best_model_for_class(run_list, task, class_idx):
    scores = []
    for model in run_list:
        model_metrics, model_name = model['metrics'], model['name']

        if task == 'colon':
            score = compute_colon_aggregate(model_metrics, model_name)
        elif task in ['chest', 'endo']:
            score = compute_multilabel_aggregate_p_class(model_metrics, model_name, class_idx)
        else:
            raise ValueError(f"Invalid task: {task}")

        scores.append((model, score))

    # Sort the scores in descending order and return the first (highest) tuple
    best_model = sorted(scores, key=lambda x: x[1], reverse=True)[0]
    return best_model[0]


def expert_model_strategy(model_runs, task, out_path):
    num_classes = TASK_2_CLASS_COUNT[task]
    merged_df = model_runs[0]['prediction'].iloc[:, 0:1]

    # List to store which model was selected for each class
    selected_models_for_classes = []

    # Dict to keep track of model occurrences
    model_occurrences = {}

    # Find run with best Aggregate for each class
    for class_idx in range(num_classes):
        best_model_run = find_best_model_for_class(model_runs, task=task, class_idx=class_idx + 1)

        merged_df = merged_df.copy()
        merged_df.loc[:, class_idx + 1] = best_model_run["prediction"].loc[:, class_idx + 1]

        # Keeping track of the model used for each class
        model_name = best_model_run['name'].split('work_dirs/')[0]
        if model_name in model_occurrences:
            model_occurrences[model_name] += 1
        else:
            model_occurrences[model_name] = 1

        selected_models_for_classes.append(f"Class {class_idx + 1}: {model_name}")

    #print(f"Saving merged prediction to {out_path}")
    merged_df.to_csv(out_path, index=False, header=False)

    return selected_models_for_classes, model_occurrences


@lru_cache(maxsize=None)
def extract_data_tuples_from_model_runs(run_list):
    data_list = []
    for run in run_list:
        prediction = pd.read_csv(run['csv'], header=None)
        metrics = json.load(open(run['json'], 'r'))
        data_list.append({'prediction': prediction, 'metrics': metrics, 'name': run['name']})
    return data_list


def check_and_extract_data(model_dir_abs, subm_type, task, shot, pbar=None):
    model_dir_rel = model_dir_abs.split('work_dirs/')[1]
    if pbar:
        pbar.set_description(f"Checking {model_dir_rel}")

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


def extract_data():
    subm_types = ["submission", "validation"]
    data_lists = {stype: {task: {shot: {} for shot in shots} for task in tasks} for stype in subm_types}

    # Total iterations: tasks * shots * exps * model_dirs * subm_types
    total_iterations = len(tasks) * len(shots) * len(exps) * len(
        glob.glob(os.path.join(root_dir, tasks[0], shots[0], '*exp[1-5]*'))) * len(subm_types)

    with tqdm(total=total_iterations, bar_format="{l_bar}%s{bar}%s{r_bar}" % (Fore.BLUE, Fore.RESET)) as pbar:
        for task in tasks:
            for shot in shots:
                path_pattern = os.path.join(root_dir, task, shot, '*exp[1-5]*')
                for model_dir in glob.glob(path_pattern):
                    for subm_type in subm_types:
                        data, exp_num = check_and_extract_data(model_dir_abs=model_dir, subm_type=subm_type,
                                                               task=task, shot=shot, pbar=pbar)
                        if data and exp_num:
                            data_lists[subm_type][task][shot][f"exp{exp_num}"].append(data)
                        pbar.update(1)
    return data_lists["submission"], data_lists["validation"]


def create_submission(is_evaluation):
    submission_type = 'submission'
    if is_evaluation:
        data_lists = DATA_SUBMISSION
        print(f"\n========== Creating {colored('Evaluation', 'red')} Submission ==========")
    else:
        data_lists = DATA_VALIDATION
        submission_type = 'validation'
        print(f"\n========== Creating {colored(submission_type.capitalize(), 'blue')} Submission ==========")

    submission_dir = create_output_dir(is_evaluation, submission_type)

    # Perform Ensemble Strategy
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
                                                                               task=task,
                                                                               out_path=out_path)
                else:
                    print("Invalid ensemble strategy!")
                    exit()

                create_ensemble_report_file(task=task, shot=shot, exp=exp, is_eval=is_evaluation,
                                            selected_models_for_classes=selected_models,
                                            model_occurrences=model_occurrences,
                                            root_report_dir=submission_dir)
    return submission_dir


def print_overall_model_summary():
    """
    Prints the overall model summary. Once is enough since the count for submission and validation is the same.
    """
    total_models = 0
    least_models = 100000
    most_models = -1
    most_setting = ""
    least_setting = ""
    for task in tasks:
        for shot in shots:
            for exp in exps:
                models_for_setting = len(DATA_SUBMISSION[task][shot][exp])
                print(f"| Setting: {task}/{shot}/{exp}\t>> Models: {models_for_setting}")
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


def create_output_dir(is_evaluation, submission_type):
    # Create Output Directory
    submission_dir = os.path.join("submissions", "evaluation", TIMESTAMP)
    if is_evaluation:
        success = f"Created {colored('Evaluation', 'red')} directory {submission_dir}"
    else:
        submission_dir = os.path.join("ensemble", f"{submission_type}", TIMESTAMP)
        success = f"Created {colored(submission_type.capitalize(), 'blue')} directory {submission_dir}"
    os.makedirs(submission_dir)
    for exp in exps:
        os.makedirs(os.path.join(submission_dir, "result", f"{exp}"), exist_ok=True)
    print(success)
    return submission_dir


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
    DATA_SUBMISSION, DATA_VALIDATION = extract_data()

    print_overall_model_summary()
    print_model_reports()

    eval_output_dir = create_submission(is_evaluation=True)
    val_output_dir = create_submission(is_evaluation=False)

    print(f"\nCreated Evaluation at {colored(eval_output_dir,'red')}")
    print(f"Created Validation at {colored(val_output_dir, 'blue')}")
