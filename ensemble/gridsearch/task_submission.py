import glob
import json
import os
import re
from collections import defaultdict
from datetime import datetime
from functools import lru_cache

import numpy as np
import pandas as pd
from colorama import Fore
from termcolor import colored
from tqdm import tqdm

from ensemble.gridsearch.test_task_submission import get_file_by_keyword
from ensemble.utils.constants import TASK_2_CLASS_COUNT, TASK_2_CLASS_NAMES


def print_colored(text, submission_type, depth):
    gradient = COLOR_GRADIENTS.get(submission_type, {})
    color = gradient.get(depth, "white")  # Default to white if depth or submission type is not found
    print(colored(text, color))

    def compute_pairwise_diversity(top_k_models):
        num_models = len(top_k_models)
        diversity_matrix = np.zeros((num_models, num_models))

        for i in range(num_models):
            model_i_predictions = top_k_models[i]['prediction']
            img_id_set1 = set(model_i_predictions.iloc[:, 0])

            for j in range(i + 1, num_models):
                model_j_predictions = top_k_models[j]['prediction']
                img_id_set2 = set(model_j_predictions.iloc[:, 0])

                # Use sets for faster membership checks
                missing_in_df2 = img_id_set1 - img_id_set2
                missing_in_df1 = img_id_set2 - img_id_set1

                if missing_in_df1 or missing_in_df2:
                    corr_idx = i if missing_in_df2 else j
                    # ... rest of the code for corrupted file ...

                try:
                    disagreements = (model_i_predictions != model_j_predictions).sum().sum()
                    diversity_matrix[i, j] = disagreements / len(model_i_predictions)
                    diversity_matrix[j, i] = diversity_matrix[i, j]  # Use the symmetry of the matrix
                except ValueError as e:
                    print(f"Error comparing model {i} and model {j}: {e}")

        return diversity_matrix.sum(axis=1)


def compute_pairwise_diversity(top_k_models):
    """
    Computes the pairwise diversity (disagreement rates) among the top k models.
    """
    num_models = len(top_k_models)
    diversity_matrix = np.zeros((num_models, num_models))

    for i in range(num_models):
        for j in range(num_models):
            if i != j:
                model_i_predictions = top_k_models[i]['prediction']
                model_j_predictions = top_k_models[j]['prediction']

                # Check for label matching
                if not model_i_predictions.columns.equals(model_j_predictions.columns) or \
                   not model_i_predictions.shape[0] == model_j_predictions.shape[0]:

                    img_id_col1 = model_i_predictions.columns[0]
                    img_id_col2 = model_j_predictions.columns[0]

                    # Identify unique image IDs not in the other dataframe
                    missing_in_df2 = model_i_predictions.loc[~model_i_predictions[img_id_col1].isin(model_j_predictions[img_id_col2]), img_id_col1]
                    missing_in_df1 = model_j_predictions.loc[~model_j_predictions[img_id_col2].isin(model_i_predictions[img_id_col1]), img_id_col2]

                    corr_idx = -1
                    if len(missing_in_df2) > 0:
                        # delete 1
                        corr_idx = i
                    elif len(missing_in_df1) > 0:
                        # delete 2
                        corr_idx = j

                    if corr_idx != -1:
                        model_name = top_k_models[corr_idx]['name']
                        model_path = os.path.join("/scratch/medfm/medfm-challenge/work_dirs", model_name)
                        model_split = model_name.split("/")
                        task = model_split[0]
                        shot = model_split[1]
                        corrupted_file = os.path.join(model_path, f"{task}_{shot}_submission.csv")

                        raise ValueError(f"Corrupted CSV File: {corrupted_file} (probably a scuffed line)")

                # Robust comparison
                try:
                    disagreements = (model_i_predictions != model_j_predictions).sum().sum()
                except ValueError as e:
                    print(f"Error comparing model {i} and model {j}: {e}")
                    continue

                diversity_matrix[i, j] = disagreements / len(model_i_predictions)

    # Sum the diversity scores for each model
    diversity_scores = diversity_matrix.sum(axis=1)
    return diversity_scores


def create_ensemble_report_file(task, shot, exp, selected_models_for_classes, model_occurrences, root_report_dir):
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


@lru_cache(maxsize=None)
def extract_exp_number(string):
    match = re.search(r'exp(\d+)', string)
    return int(match.group(1)) if match else 0


def weighted_ensemble_strategy(model_runs, task, shot, exp, out_path, top_k=3):
    """
    Merges model runs using a weighted sum approach based on the N best model runs for each class.
    """
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
        top_n_models = class_models[:top_k]

        # Check if k is greater than the available models and print warning
        if top_k > len(class_models):
            print(colored(
                f"Warning: Requested top {top_k} models, but only {len(class_models)} are available for class {i + 1}",
                'red'))

        # Record the selected models for the report and update the model_occurrences
        selected_models_for_class = []
        for model, weight in top_n_models:
            model_name = model['name']
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

    merged_df.to_csv(out_path, index=False, header=False)

    return selected_models_for_classes, model_occurrences


def performance_diff_weight_ensemble_strategy(model_runs, task, out_path, top_k=3, log_scale=False):
    """
    Merges model runs using a difference in performance weight approach for the N best model runs for each class.
    """
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
        top_n_models = class_models[:top_k]

        # Use the difference in performance from the k-th model as weights
        kth_value = top_n_models[-1][1]

        if log_scale:
            weights = [np.log(value - kth_value + 1) for _, value in top_n_models]
        else:
            weights = [value - kth_value for _, value in top_n_models]

        # Record the selected models for the report and update the model_occurrences
        selected_models_for_class = []
        for (model_run, _), weight in zip(top_n_models, weights):  # Here, we use the difference weights
            model_name = model_run['name']  # Corrected this line
            selected_models_for_class.append(f"Class {i + 1}: {model_name} (Weight: {weight:.4f})")

            if model_name in model_occurrences:
                model_occurrences[model_name] += 1
            else:
                model_occurrences[model_name] = 1

        selected_models_for_classes.extend(selected_models_for_class)

        # Calculate the sum of weights for normalization
        sum_weights = sum(weights)

        # Compute the weighted sum for this class
        weighted_sum_column = pd.Series(0, index=merged_df.index)
        for (model_run, _), weight in zip(top_n_models, weights):  # Use the difference weights
            weighted_sum_column += (model_run['prediction'].iloc[:, i + 1] * weight) / sum_weights

        merged_df.loc[:, i + 1] = weighted_sum_column

    merged_df.to_csv(out_path, index=False, header=False)

    return selected_models_for_classes, model_occurrences


def rank_based_weight_ensemble_strategy(model_runs, task, out_path, top_k=3):
    """
    Merges model runs using a rank-based weight approach for the N best model runs for each class.
    """
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
        top_n_models = class_models[:top_k]

        # Assign rank-based weights
        weights = list(range(top_k, 0, -1))

        # Record the selected models for the report and update the model_occurrences
        selected_models_for_class = []
        for (model_run, _), weight in zip(top_n_models, weights):
            model_name = model_run['name']
            selected_models_for_class.append(f"Class {i + 1}: {model_name} (Rank: {top_k + 1 - weight})")

            if model_name in model_occurrences:
                model_occurrences[model_name] += 1
            else:
                model_occurrences[model_name] = 1

        selected_models_for_classes.extend(selected_models_for_class)

        # Calculate the sum of weights for normalization
        sum_weights = sum(weights)

        # Compute the weighted sum for this class
        weighted_sum_column = pd.Series(0, index=merged_df.index)
        for (model_run, _), weight in zip(top_n_models, weights):
            weighted_sum_column += (model_run['prediction'].iloc[:, i + 1] * weight) / sum_weights

        merged_df.loc[:, i + 1] = weighted_sum_column

    merged_df.to_csv(out_path, index=False, header=False)

    return selected_models_for_classes, model_occurrences


def diversity_weighted_ensemble_strategy(model_runs, task, out_path, top_k=3):
    """
    Merges model runs using a diversity-weighted sum approach based on the N best model runs for each class.
    """
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
            if aggregate_value is not None and aggregate_value >= 0:
                class_models.append((run, aggregate_value))

        # Sort the models based on aggregate value and take the top N models
        class_models.sort(key=lambda x: x[1], reverse=True)
        top_n_models = class_models[:top_k]

        # Compute diversity scores for the top k models
        top_k_model_data = [model for model, _ in top_n_models]

        diversity_scores = compute_pairwise_diversity(top_k_model_data)

        if diversity_scores is None:
            return None, None

        # Ensure diversity scores match the expected length
        if len(diversity_scores) != len(top_k_model_data):
            raise ValueError("Mismatch between diversity scores length and top k models.")

        # Check if k is greater than the available models and print warning
        if top_k > len(class_models):
            print(colored(
                f"Warning: Requested top {top_k} models, but only {len(class_models)} are available for class {i + 1}",
                'red'))

        # Normalize diversity scores to be within the same range as weights (optional step based on diversity scores)
        max_weight = max([weight for _, weight in top_n_models])
        diversity_scores = [score / max(diversity_scores) * max_weight for score in diversity_scores]

        # Record the selected models for the report and update the model_occurrences
        selected_models_for_class = []
        for model, weight in top_n_models:
            model_name = model['name']
            selected_models_for_class.append(f"Class {i + 1}: {model_name} (Weight: {weight:.4f})")

            if model_name in model_occurrences:
                model_occurrences[model_name] += 1
            else:
                model_occurrences[model_name] = 1

        selected_models_for_classes.extend(selected_models_for_class)

        weights = [value for _, value in top_n_models]

        # Calculate the sum of weights (aggregate values) for normalization
        sum_weights = sum([weight + diversity_score for weight, diversity_score in zip(weights, diversity_scores)])

        # Compute the diversity-weighted sum for this class
        weighted_sum_column = pd.Series(0, index=merged_df.index)
        for model, weight, diversity_score in zip(top_k_model_data, weights, diversity_scores):
            adjusted_weight = weight + diversity_score
            weighted_sum_column += (model['prediction'].iloc[:, i + 1] * adjusted_weight) / sum_weights

        merged_df.loc[:, i + 1] = weighted_sum_column

    merged_df.to_csv(out_path, index=False, header=False)

    return selected_models_for_classes, model_occurrences


def print_report_for_setting(full_model_list, task, shot, exp):
    print("\nReport for:", colored(os.path.join(task.capitalize(), shot, exp), 'blue'))

    model_view = []
    for model_info in full_model_list[task][shot][exp]:
        agg_name, agg_val = get_aggregate(model_metrics=model_info['metrics'], task=task)
        m1_name, m1_val, m2_name, m2_val = get_metrics(model_metrics=model_info['metrics'], task=task)

        if agg_val is not None and m1_val is not None and m2_val is not None:
            model_view.append((model_info['name'], agg_name, agg_val, m1_name, m1_val, m2_name, m2_val))

    model_view.sort(key=lambda x: x[2])
    max_char_length = max(len(path) for path, _, _, _, _, _, _ in model_view)

    for model_path_rel, agg_name, agg_val, m1_name, m1_val, m2_name, m2_val in model_view:
        print(f"{model_path_rel:{max_char_length}} {agg_name}: {agg_val:.4f}  "
              f"{m1_name}: {m1_val:.4f}  {m2_name}: {m2_val:.4f}")


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
        return None, None

    return calculation(model_metrics)


def get_metrics(model_metrics, task):
    # Dictionary mapping tasks to lambda functions for aggregate calculation
    aggregate_calculations = {
        "colon": lambda metrics:
        ("AUC", metrics["AUC/AUC_multiclass"], "Acc", metrics["accuracy/top1"])
        if "AUC/AUC_multiclass" in metrics and "accuracy/top1" in metrics else (None, None, None, None),

        "chest": lambda metrics:
        ("AUC", metrics["AUC/AUC_multilabe"], "mAP", metrics["multi-label/mAP"])
        if "AUC/AUC_multilabe" in metrics and "multi-label/mAP" in metrics else (None, None, None, None),

        "endo": lambda metrics:
        ("AUC", metrics["AUC/AUC_multilabe"], "mAP", metrics["multi-label/mAP"])
        if "AUC/AUC_multilabe" in metrics and "multi-label/mAP" in metrics else (None, None, None, None),
    }

    # Get the appropriate aggregate calculation for the task
    calculation = aggregate_calculations.get(task)

    # If there's no calculation for the task, return None for all four values
    if not calculation:
        return None, None, None, None

    return calculation(model_metrics)


def compute_binary_classification_aggregate(model_data):
    """
    Compute metrics for a binary classification setting (e.g., 'colon').
    AUC and ACC metrics are considered.
    """
    model_metrics, model_name = model_data['metrics'], model_data['name']

    auc_label = "AUC/AUC_multiclass"
    acc_label = "accuracy/top1"

    if auc_label not in model_metrics:
        print(f"Metric '{auc_label}' not found in {model_name}")
        return None

    if acc_label not in model_metrics:
        print(f"Metric '{acc_label}' not found in {model_name}")
        return None

    return (model_metrics[auc_label] + model_metrics[acc_label]) / 2


def compute_multilabel_aggregate(model_data, class_idx=None):
    """
    Compute metrics for multi-label setting (e.g., 'chest' and 'endo').
    AUC and mAP metrics are considered.
    """
    model_metrics, model_name = model_data['metrics'], model_data['name']

    auc_label = "AUC/AUC_multilabe"
    map_label = "multi-label/mAP"

    if class_idx:
        auc_label = f"AUC/AUC_class{class_idx}"
        map_label = f"MAP_class{class_idx}"

    if auc_label not in model_metrics:
        auc_label = "AUC/AUC_multiclass"

        if auc_label not in model_metrics:
            print(f"Metric 'AUC/AUC_multilabe' and '{auc_label}' not found in {model_name}")
            return None

    if map_label not in model_metrics:
        print(f"Metric '{map_label}' not found in {model_name}")
        return None

    return (model_metrics[auc_label] + model_metrics[map_label]) / 2


def softmax(x):
    e_x = np.exp(x - np.max(x))  # subtracting np.max for numerical stability
    return e_x / e_x.sum(axis=0)


def exponential_scaling(x, p=2):
    return [xi**p for xi in x]


def log_scaling(x):
    return [np.log(xi + 1) for xi in x]


def find_top_k_models(model_list, num_classes, class_idx=None, top_k=1):
    """
    Find the K best performing model runs according to the 'Aggregate' metric.
    If class_idx is None, the aggregate is calculated among all classes.
    If class_idx is not None, the aggregate is calculated for one specific class.

    Args:
        model_list:
        num_classes:
        class_idx:
        top_k:

    Returns:

    """
    scores = []

    for model_data in model_list:

        if num_classes == 2:
            score = compute_binary_classification_aggregate(model_data=model_data)
        elif num_classes > 2:
            score = compute_multilabel_aggregate(model_data=model_data, class_idx=class_idx)
        else:
            raise ValueError(f"Invalid amount of classes: {num_classes}")

        if score is None:
            raise ValueError(f"Could not calculate score: {model_data['name']}")

        scores.append((model_data, score))

    # Sort the scores in descending order
    sorted_scores = sorted(scores, key=lambda x: x[1], reverse=True)

    # Return the top-k models. If less than k models are available, return all of them.
    return sorted_scores[:top_k]


def find_best_model(model_list, num_classes, class_idx=None):
    scores = []

    for model_data in model_list:

        if num_classes == 2:
            score = compute_binary_classification_aggregate(model_data=model_data)
        elif num_classes > 2:
            score = compute_multilabel_aggregate(model_data=model_data, class_idx=class_idx)
        else:
            raise ValueError(f"Invalid amount of classes: {num_classes}")

        if score is None:
            raise ValueError(f"Could not calculate score: {model_data['name']}")

        scores.append((model_data, score))

    # Sort the scores in descending order and return the first (highest) tuple
    best_model = sorted(scores, key=lambda x: x[1], reverse=True)[0]
    return best_model[0]


def weighted_exp_per_class_ensemble_strategy(model_runs, task, out_path, top_k=3, scaling_func=None):
    num_classes = TASK_2_CLASS_COUNT[task]
    merged_df = model_runs[0]['prediction'].iloc[:, 0:1].copy()

    selected_models_for_classes = []
    model_occurrences = {}

    for class_idx in range(num_classes):

        class_models = find_top_k_models(model_runs, num_classes, class_idx=class_idx + 1, top_k=top_k)

        if top_k > len(class_models):
            print(colored(
                f"Warning: Requested top {top_k} models, but only {len(class_models)} are available for class {class_idx + 1}",
                'red'))

        # Record the selected models for the report and update the model_occurrences
        selected_models_for_class = []
        for model, weight in class_models:
            model_name = model['name']
            selected_models_for_class.append(f"Class {class_idx + 1}: {model_name} (Weight: {weight:.4f})")

            if model_name in model_occurrences:
                model_occurrences[model_name] += 1
            else:
                model_occurrences[model_name] = 1

        selected_models_for_classes.extend(selected_models_for_class)

        # Compute the weighted sum for this class
        weighted_sum_column = pd.Series(0, index=merged_df.index)
        for model, weight in class_models:
            weighted_sum_column += model['prediction'].iloc[:, class_idx + 1] * weight

        # Scale
        if scaling_func:
            weighted_sum_column = pd.Series(scaling_func(weighted_sum_column), index=weighted_sum_column.index)

        # Normalize
        weighted_sum_column /= weighted_sum_column.sum()

        merged_df.loc[:, class_idx + 1] = weighted_sum_column

    merged_df.to_csv(out_path, index=False, header=False)

    return selected_models_for_classes, model_occurrences


def expert_per_class_model_strategy(model_runs, task, out_path):
    num_classes = TASK_2_CLASS_COUNT[task]
    merged_df = model_runs[0]['prediction'].iloc[:, 0:1]

    selected_models_for_classes = []
    model_occurrences = {}

    for class_idx in range(num_classes):
        best_model_run = find_best_model(model_runs, num_classes=num_classes, class_idx=class_idx + 1)

        merged_df = merged_df.copy()
        merged_df.loc[:, class_idx + 1] = best_model_run["prediction"].loc[:, class_idx + 1]

        # Keeping track of the model used for each class
        model_name = best_model_run['name'].split('work_dirs/')[0]
        if model_name in model_occurrences:
            model_occurrences[model_name] += 1
        else:
            model_occurrences[model_name] = 1

        selected_models_for_classes.append(f"Class {class_idx + 1}: {model_name}")

    merged_df.to_csv(out_path, index=False, header=False)

    return selected_models_for_classes, model_occurrences


def expert_per_task_model_strategy(model_runs, task, out_path):
    best_model_run = find_best_model(model_list=model_runs, num_classes=TASK_2_CLASS_COUNT[task])

    merged_df = best_model_run['prediction']
    merged_df.to_csv(out_path, index=False, header=False)

    # Extract and return the name of the best model
    best_model_name = best_model_run['name'].split('work_dirs/')[0]

    # Dict to keep track of model occurrences (in this case, just the best model)
    model_occurrences = {best_model_name: 1}

    return [f"Overall Best Model: {best_model_name}"], model_occurrences


@lru_cache(maxsize=None)
def extract_data_tuples_from_model_runs(run_list):
    data_list = []
    for run in run_list:
        prediction = pd.read_csv(run['csv'], header=None)
        metrics = json.load(open(run['json'], 'r'))
        data_list.append({'prediction': prediction, 'metrics': metrics, 'name': run['name']})
    return data_list


def get_gt_df(gt_dir, task):
    gt_file_path = get_file_by_keyword(directory=gt_dir, keyword=task, file_extension='csv')

    if not gt_file_path:
        raise ValueError(f"Ground truth file for task {task} not found.")
    try:
        cols_2_keep = TASK_2_CLASS_NAMES.get(task, None)
        if not cols_2_keep:
            raise ValueError(f"No matching class names for task {task} found")
        cols_2_keep.insert(0, 'img_id')

        gt_df = pd.read_csv(gt_file_path, usecols=cols_2_keep)

    except Exception as e:
        raise ValueError(f"Error reading CSV files: {e}")

    return gt_df


def check_and_extract_data(model_dir_abs, subm_type, task, shot):
    model_dir_rel = model_dir_abs.split('work_dirs/')[1]

    csv_path = os.path.join(model_dir_abs, f"{task}_{shot}_{subm_type}.csv")
    csv_files = glob.glob(csv_path)
    json_files = glob.glob(os.path.join(model_dir_abs, "*.json"))

    if csv_files and json_files:
        exp_num = extract_exp_number(model_dir_rel)
        if exp_num != 0:
            pred_df = pd.read_csv(csv_files[0], header=None)
            metrics = json.load(open(json_files[0], 'r'))

            return {'prediction': pred_df,
                    'metrics': metrics,
                    'name': model_dir_rel}, exp_num
    return None, None


def load_data():
    data_lists = {
        stype: {
            task: {
                shot: {
                    exp: [] for exp in EXPS
                } for shot in SHOTS
            } for task in TASKS
        } for stype in SUBM_TYPES
    }

    # Total iterations: tasks * shots * exps * model_dirs * subm_types
    total_iterations = 0
    for task in TASKS:
        for shot in SHOTS:
            total_iterations += len(glob.glob(os.path.join(ROOT_DIR, task, shot, '*exp[1-5]*')))

    print(f"Checking and extracting data for {colored(str(total_iterations), 'blue')} models:")

    with tqdm(total=total_iterations, bar_format="{l_bar}%s{bar}%s{r_bar}" % (Fore.BLUE, Fore.RESET)) as pbar:
        for task in TASKS:
            for shot in SHOTS:
                path_pattern = os.path.join(ROOT_DIR, task, shot, '*exp[1-5]*')

                for model_dir in glob.glob(path_pattern):
                    for subm_type in SUBM_TYPES:
                        data, exp_num = check_and_extract_data(model_dir_abs=model_dir,
                                                               subm_type=subm_type,
                                                               task=task,
                                                               shot=shot)
                        if data and exp_num and f"exp{exp_num}" in EXPS:
                            data_lists[subm_type][task][shot][f"exp{exp_num}"].append(data)
                    pbar.update(1)
    return data_lists


def create_submission_cfg_dump(top_k, total_models, strategy, root_report_dir):
    config_data = {
        'timestamp': TIMESTAMP,
        'top-k': top_k,
        'model-count': total_models,
        'strategy': strategy,
    }
    cfg_file_path = os.path.join(root_report_dir, "config.json")

    with open(cfg_file_path, 'w') as cfg_file:
        json.dump(config_data, cfg_file, indent=4)

    return cfg_file_path


def process_top_k(subm_type, task, shot, exp, strategy, top_k):
    submission_dir = create_output_dir(subm_type=subm_type,
                                       task=task, shot=shot, exp=exp,
                                       strategy=strategy,
                                       top_k=top_k)

    model_runs = DATA[subm_type][task][shot][exp]
    if len(model_runs) < 2:
        print("Not enough runs")
        return
    out_path = os.path.join(submission_dir, "result", f"{exp}", f"{task}_{shot}_{subm_type}.csv")

    if strategy == "weighted":
        selected_models, model_occurrences = weighted_ensemble_strategy(model_runs=model_runs,
                                                                        task=task, shot=shot, exp=exp,
                                                                        top_k=top_k, out_path=out_path)
    elif strategy == "expert-per-class":
        selected_models, model_occurrences = expert_per_class_model_strategy(model_runs=model_runs,
                                                                             task=task,
                                                                             out_path=out_path)
    elif strategy == "expert-per-task":
        selected_models, model_occurrences = expert_per_task_model_strategy(model_runs=model_runs,
                                                                            task=task,
                                                                            out_path=out_path)
    elif strategy == "weighted-exp-per-class":
        selected_models, model_occurrences = weighted_exp_per_class_ensemble_strategy(model_runs=model_runs, top_k=top_k,
                                                                                      task=task, out_path=out_path)
    elif strategy == "log-weighted-exp-per-class":
        selected_models, model_occurrences = weighted_exp_per_class_ensemble_strategy(model_runs=model_runs, top_k=top_k,
                                                                                      task=task, out_path=out_path,
                                                                                      scaling_func=log_scaling)
    elif strategy == "sm-weighted-exp-per-class":
        selected_models, model_occurrences = weighted_exp_per_class_ensemble_strategy(model_runs=model_runs, top_k=top_k,
                                                                                      task=task, out_path=out_path,
                                                                                      scaling_func=softmax)
    elif strategy == "expo-weighted-exp-per-class":
        selected_models, model_occurrences = weighted_exp_per_class_ensemble_strategy(model_runs=model_runs, top_k=top_k,
                                                                                      task=task, out_path=out_path,
                                                                                      scaling_func=exponential_scaling)
    elif strategy == "pd-weighted":
        selected_models, model_occurrences = performance_diff_weight_ensemble_strategy(model_runs=model_runs,
                                                                                       task=task,
                                                                                       out_path=out_path,
                                                                                       top_k=top_k)
    elif strategy == "pd-log-weighted":
        selected_models, model_occurrences = performance_diff_weight_ensemble_strategy(model_runs=model_runs,
                                                                                       task=task,
                                                                                       out_path=out_path,
                                                                                       top_k=top_k,
                                                                                       log_scale=True)
    elif strategy == "rank-based-weighted":
        selected_models, model_occurrences = rank_based_weight_ensemble_strategy(model_runs=model_runs,
                                                                                 task=task,
                                                                                 out_path=out_path,
                                                                                 top_k=top_k)
    elif strategy == "diversity-weighted":
        selected_models, model_occurrences = diversity_weighted_ensemble_strategy(model_runs=model_runs,
                                                                                  task=task,
                                                                                  out_path=out_path,
                                                                                  top_k=top_k)
    else:
        print(f"Invalid ensemble strategy {strategy}!")
        exit()

    if model_occurrences and submission_dir:
        create_ensemble_report_file(task=task, shot=shot, exp=exp,
                                    selected_models_for_classes=selected_models,
                                    model_occurrences=model_occurrences,
                                    root_report_dir=submission_dir)
    if subm_type == "validation":
        create_submission_cfg_dump(top_k=top_k,
                                   strategy=strategy,
                                   total_models=MODEL_COUNTS[subm_type][task][shot][exp]['total'],
                                   root_report_dir=submission_dir)

    return submission_dir


def extract_least_model_counts(task, subm_type, shot, exp):
    if (task in MODEL_COUNTS[subm_type]
            and shot in MODEL_COUNTS[subm_type][task]
            and exp in MODEL_COUNTS[subm_type][task][shot]):
        return

    total_models = len(DATA[subm_type][task][shot][exp])
    least_models = total_models

    result_dict = {'top-k': least_models, 'total': total_models}
    MODEL_COUNTS[subm_type][task][shot][exp] = result_dict


def create_output_dir(subm_type, task, shot, exp, top_k, strategy):
    base_path = "ensemble/gridsearch"
    submission_dir = os.path.join(base_path, TIMESTAMP, subm_type, task, shot, exp, strategy)

    if top_k:
        submission_dir = os.path.join(submission_dir, f"top-{str(top_k)}")
    else:
        submission_dir = os.path.join(submission_dir)

    if not os.path.isdir(submission_dir):
        os.makedirs(submission_dir)

    os.makedirs(os.path.join(submission_dir, "result", f"{exp}"), exist_ok=True)

    return submission_dir


def select_top_k_models():
    while True:
        top_k = input("Enter the number of top-k models for the weighted ensemble: ")
        if top_k.isdigit() and int(top_k) > 0:
            return int(top_k)
        else:
            print("Invalid number. Please enter a positive integer.\n")


def select_task():
    tasks = ["colon", "endo", "chest"]
    while True:
        print("Choose a task:")
        for idx, task in enumerate(tasks, 1):
            print(f"{idx}. {task}")

        choice = input("Enter your choice: ")

        if choice.isdigit() and 1 <= int(choice) <= len(tasks):
            choice = tasks[int(choice) - 1]
            return choice
        else:
            print("Invalid choice. Please try again.\n")


def process_strategy(subm_type, task, shot, exp, strategy):
    print_colored(f"\t\t\t\tProcessing Strategy {strategy}", subm_type, 4)
    top_k = MODEL_COUNTS[subm_type][task][shot][exp]['top-k']

    top_k_values = [None] if "expert" in strategy else range(2, min(top_k, MAX_TOP_K))

    for top_k in top_k_values:
        process_top_k(subm_type=subm_type,
                      task=task, shot=shot, exp=exp,
                      strategy=strategy, top_k=top_k)


def main():
    # 1st Level Iteration
    for subm_type in SUBM_TYPES:
        print_colored(f"Processing Submission Type {subm_type.capitalize()}", subm_type, 0)

        # 2nd Level Iteration
        for task in TASKS:
            print_colored(f"\tProcessing Task {task.capitalize()}", subm_type, 1)
            for shot in SHOTS:
                print_colored(f"\t\tProcessing Shot {shot}", subm_type, 2)
                for exp in EXPS:
                    print_colored(f"\t\t\tProcessing Experiment {exp}", subm_type, 3)
                    extract_least_model_counts(subm_type=subm_type,
                                               task=task,
                                               shot=shot,
                                               exp=exp)

                    # 3rd Level Iteration
                    for strategy in ENSEMBLE_STRATEGIES:
                        process_strategy(subm_type=subm_type,
                                         task=task, shot=shot, exp=exp,
                                         strategy=strategy)

        dir_path = os.path.join("ensemble/gridsearch", TIMESTAMP, subm_type)
        print(f"Created directory {dir_path}")


# ===================  DEFAULT PARAMS  =================
ROOT_DIR = "/scratch/medfm/medfm-challenge/work_dirs"
SUBM_TYPES = ["validation", "submission"]
TASKS = ["colon", "endo", "chest"]
SHOTS = ["1-shot", "5-shot", "10-shot"]
EXPS = ["exp1", "exp2", "exp3", "exp4", "exp5"]
ENSEMBLE_STRATEGIES = ["expert-per-task",
                       "expert-per-class",
                       "weighted",
                       "weighted-exp-per-class",
                       "log-weighted-exp-per-class",
                       "sm-weighted-exp-per-class",
                       "expo-weighted-exp-per-class",
                       "pd-weighted",
                       "pd-log-weighted",
                       "rank-based-weighted",
                       "diversity-weighted"]
COLOR_GRADIENTS = {
    "submission": {
        0: "red",
        1: "magenta",
        2: "cyan",
        3: "light_blue",
        4: "light_magenta"
    },
    "validation": {
        0: "blue",
        1: "magenta",
        2: "cyan",
        3: "light_blue",
        4: "light_magenta"
    }
}
# ======================================================


if __name__ == "__main__":
    # 1st Level Params
    #SUBM_TYPES = ["validation"]

    # 2nd Level Params
    #TASKS = ["endo"]
    #SHOTS = ["1-shot"]
    #EXPS = ["exp1"]

    # 3rd Level Params
    ENSEMBLE_STRATEGIES = ["expert-per-task",
                           "expert-per-class",
                           "weighted",
                           "weighted-exp-per-class",
                           "log-weighted-exp-per-class",
                           "sm-weighted-exp-per-class",
                           "expo-weighted-exp-per-class",
                           "pd-weighted",
                           "pd-log-weighted",
                           "rank-based-weighted",
                           #"diversity-weighted"
                           ]

    MODEL_COUNTS = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))

    TIMESTAMP = datetime.now().strftime("%d-%m_%H-%M-%S")
    MAX_TOP_K = 10
    DATA = load_data()

    main()
