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
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from termcolor import colored
from tqdm import tqdm

from ensemble.gridsearch.test_task_submission import get_file_by_keyword
from ensemble.utils.constants import shots, exps, TASK_2_CLASS_COUNT, TASK_2_CLASS_NAMES


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
                disagreements = (model_i_predictions != model_j_predictions).sum().sum()  # Fixed here
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
    print("Executing weighted ensemble for ", task, shot, exp, top_k)

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

        # # Debug print #1: Print the top k model names and their weights
        # print(f"Top {top_k} models for class {i + 1}:")
        # for (model_run, _), weight in zip(top_n_models, weights):
        #     print(f"Model: {model_run['name']}, Weight: {weight:.4f}")
        #
        # # Debug print #2: Print the sum of weights before normalization
        # print(f"Sum of weights for class {i + 1}: {sum(weights):.4f}\n")

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

        # # Debug print #3: Print the weighted sum for the first 5 data points
        # print(f"Weighted sum for the first 5 data points in class {i + 1}: {weighted_sum_column[:5].tolist()}\n")

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


def print_model_reports(tasks):
    continue_query = input("\nPrint report for the best models? (y/n) ")
    if continue_query.lower() == "y":
        for task in tasks:
            for shot in shots:
                for exp in exps:
                    print_report_for_setting(full_model_list=DATA_VALIDATION, task=task, shot=shot, exp=exp)


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


def stacking_strategy(model_runs, task, shot, subm_type, out_path):
    print(f"Executing Stacking for {os.path.join(task, shot)}")

    # Step 1: Generate Meta-Features for the Validation Set
    num_classes = TASK_2_CLASS_COUNT[task]

    # The ground truth of the test split from the training set
    gt_df = model_runs[list(model_runs.keys())[0]][0]['train-test_gt']
    print(gt_df, gt_df.shape, gt_df.columns)
    exit()

    # Extracting the img_id column from the first model run of the first experiment
    meta_features_df = model_runs[list(model_runs.keys())[0]][0]['prediction'][[0]].copy()
    meta_features_df.rename({0: 'img_id'}, axis='columns', inplace=True)

    # Create an empty list to store meta-features for each model
    meta_features_df_list = [meta_features_df]

    for exp in model_runs:
        for model_run in model_runs[exp]:
            predictions = model_run['train-test_prediction'].iloc[:, 1 : num_classes + 1]

            # Using a unique identifier for each prediction column, e.g. model name or index
            renamed_predictions = predictions.copy()
            for col_name in predictions.columns:
                new_col_name = f"{model_run['name']}_{col_name}"  # Adjust naming convention as needed
                renamed_predictions.rename(columns={col_name: new_col_name}, inplace=True)

            meta_features_df_list.append(renamed_predictions)

    # Base Data
    meta_features_df = pd.concat(meta_features_df_list, axis=1)
    gt_df = gt_df[gt_df['img_id'].isin(meta_features_df['img_id'])]

    X_train = meta_features_df.drop(columns=['img_id'])
    y_train = gt_df.drop(columns=['img_id'])

    # Set up the meta-model based on the task
    if task == "colon":
        y_train = y_train['tumor'].ravel()
        meta_model = LogisticRegression(solver='lbfgs', max_iter=1000)
    elif task == "endo" or task == "chest":
        base_classifier = LogisticRegression(solver='lbfgs', max_iter=1000)
        meta_model = OneVsRestClassifier(base_classifier)
    else:
        raise f"Invalid task specified."

    meta_model.fit(X=X_train, y=y_train)

    # Step 3 & 4: Generate Meta-Features for the Test Set and Make Predictions for Each Experiment
    for exp in model_runs:
        meta_features_test_list = []
        for model_run in model_runs[exp]:
            predictions = model_run['prediction'].iloc[:, 1:num_classes+1]
            meta_features_test_list.append(predictions)

        meta_features_test = pd.concat(meta_features_test_list, axis=1)

        final_predictions = meta_model.predict(meta_features_test)

        # Save the predictions to the specified output path for the current experiment
        exp_out_path = os.path.join(out_path, "result", f"{exp}", f"{task}_{shot}_{subm_type}.csv")
        final_predictions_df = pd.DataFrame(final_predictions)
        final_predictions_df.to_csv(exp_out_path, index=False, header=False)


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

    train_test_pred_df = None
    if "stacking" in ENSEMBLE_STRATEGIES:
        train_test_csv = os.path.join(model_dir_abs, f"{task}_{shot}_train-test.csv")
        train_test_csv_file = glob.glob(train_test_csv)

        if train_test_csv_file:
            train_test_pred_df = pd.read_csv(train_test_csv_file[0], header=None)

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
                    'name': model_dir_rel,
                    'train-test_prediction': train_test_pred_df}, exp_num
    return None, None


def extract_data(root_dir):
    subm_types = ["submission", "validation"]

    # Total iterations: tasks * shots * exps * model_dirs * subm_types
    total_iterations = 0
    for task in TASKS:
        for shot in shots:
            total_iterations += len(glob.glob(os.path.join(root_dir, task, shot, '*exp[1-5]*')))

    print(f"Checking and extracting data for {colored(str(total_iterations), 'blue')} models:")
    data = load_data(total_iterations=total_iterations, root_dir=root_dir, subm_types=subm_types)

    return data['submission'], data['validation']


def load_data(total_iterations, root_dir, subm_types):
    data_lists = {
        stype: {
            task: {
                shot: {
                    exp: [] for exp in exps
                } for shot in shots
            } for task in TASKS
        } for stype in subm_types
    }

    with tqdm(total=total_iterations, bar_format="{l_bar}%s{bar}%s{r_bar}" % (Fore.BLUE, Fore.RESET)) as pbar:
        for task in TASKS:
            # Load ground truth for test split of training set once per task
            train_test_gt_dir = f"/scratch/medfm/medfm-challenge/data/MedFMC_train/{task}"
            train_test_gt_df = get_gt_df(gt_dir=train_test_gt_dir, task=task)

            for shot in shots:
                path_pattern = os.path.join(root_dir, task, shot, '*exp[1-5]*')
                for model_dir in glob.glob(path_pattern):
                    for subm_type in subm_types:
                        data, exp_num = check_and_extract_data(model_dir_abs=model_dir, subm_type=subm_type, task=task,
                                                               shot=shot)
                        if data and exp_num:
                            data['train-test_gt'] = train_test_gt_df  # Add ground truth data to the dictionary
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


def process_top_k(strategy, top_k, task, subm_type):
    if subm_type == "submission":
        data = DATA_SUBMISSION
    elif subm_type == "validation":
        data = DATA_VALIDATION
    else:
        raise f"Unknown submission type {subm_type}!"

    submission_dir = create_output_dir(strategy=strategy,
                                       top_k=top_k,
                                       task=task,
                                       submission_type=subm_type)

    # Perform Ensemble Strategy Task/Shot Wise
    for shot in shots:
        model_runs = data[task][shot]
        if len(model_runs) < 2:
            print("Not enough runs")
            continue

        if strategy == "stacking":
            stacking_strategy(model_runs=model_runs, task=task, shot=shot, subm_type=subm_type, out_path=submission_dir)
            continue

        # Perform Ensemble Strategy Task/Shot/Exp Wise
        for exp in exps:
            model_runs = data[task][shot][exp]
            if len(model_runs) < 2:
                print("Not enough runs")
                continue
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
                print("Invalid ensemble strategy!")
                exit()

            if model_occurrences and submission_dir:
                create_ensemble_report_file(task=task, shot=shot, exp=exp,
                                            selected_models_for_classes=selected_models,
                                            model_occurrences=model_occurrences,
                                            root_report_dir=submission_dir)
    if subm_type == "validation":
        create_submission_cfg_dump(top_k=top_k,
                                   strategy=strategy,
                                   total_models=TOTAL_MODELS[task],
                                   root_report_dir=submission_dir)

    return submission_dir


def get_least_model_count(task):
    total_models = 0
    least_models = 100000
    least_setting = ""
    for shot in shots:
        for exp in exps:
            models_for_setting = len(DATA_SUBMISSION[task][shot][exp])
            total_models += models_for_setting
            if models_for_setting < least_models:
                least_models = models_for_setting
                least_setting = f"{task} {shot} {exp}"
    return total_models, least_models, least_setting


def print_overall_model_summary(tasks):
    """
    Prints the overall model summary. Once is enough since the count for submission and validation is the same.
    """
    total_models = 0
    least_models = 100000
    most_models = -1
    most_setting = ""
    least_setting = ""
    print(f"""\n============== Overall Model Summary ==============""")
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
    print("===================================================")
    print(f"| Total models: {total_models}")
    print(f"| Most models: {most_models} {most_setting}")
    print(f"| Least models: {least_models} {least_setting}")
    print("===================================================")
    return total_models


def create_output_dir(task, top_k, strategy, submission_type):
    base_path = os.path.join("ensemble", "gridsearch")
    submission_dir = os.path.join(base_path, TIMESTAMP, submission_type, task, strategy)

    color = 'red'
    if submission_type == "validation":
        color = 'blue'

    if top_k:
        submission_dir = os.path.join(submission_dir, f"top-{str(top_k)}")
    else:
        submission_dir = os.path.join(submission_dir)

    if not os.path.isdir(submission_dir):
        os.makedirs(submission_dir)

    for exp in exps:
        os.makedirs(os.path.join(submission_dir, "result", f"{exp}"), exist_ok=True)

    print(f"Created {colored(task.capitalize(), color)} {submission_type} directory {submission_dir}")

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


def process_strategy(strategy, task, subm_type):
    if "expert" in strategy or "stacking" == strategy:
        process_top_k(strategy=strategy,
                      task=task,
                      top_k=None,
                      subm_type=subm_type)
    else:
        for top_k in range(2, TOP_K_MAX[task]):
            process_top_k(strategy=strategy,
                          top_k=top_k,
                          task=task,
                          subm_type=subm_type)


def process_subm_type(task, subm_type):
    for strategy in ENSEMBLE_STRATEGIES:
        process_strategy(strategy=strategy, task=task, subm_type=subm_type)


def process_task(task):
    TOTAL_MODELS[task], TOP_K_MAX[task], _ = get_least_model_count(task=task)

    for subm_type in SUBMISSION_TYPES:
        process_subm_type(task=task, subm_type=subm_type)


def main():
    for task in TASKS:
        process_task(task=task)


# ======================================================
ENSEMBLE_STRATEGIES = ["expert-per-task",
                       "expert-per-class",
                       "stacking",
                       "weighted",
                       "pd-weighted",
                       "pd-log-weighted",
                       "rank-based-weighted",
                       "diversity-weighted"]
SUBMISSION_TYPES = ["submission", "validation"]
TASKS = ["colon", "endo", "chest"]
# ======================================================


if __name__ == "__main__":
    root_dir = "/scratch/medfm/medfm-challenge/work_dirs"

    ENSEMBLE_STRATEGIES = ["stacking"]
    TASKS = ["colon"]
    SUBMISSION_TYPES = ["validation"]

    TOTAL_MODELS = defaultdict()
    TOP_K_MAX = defaultdict()
    TIMESTAMP = datetime.now().strftime("%d-%m_%H-%M-%S")
    DATA_SUBMISSION, DATA_VALIDATION = extract_data(root_dir=root_dir)

    main()
