import glob
import os

import pandas as pd
import math
import numpy as np
import torch
from sklearn import metrics
from sklearn.metrics import average_precision_score


def process_experiment(exp, task, shot):
    gt_path = get_gt_csv_filepath(task=task)
    if not gt_path:
        print(f"Ground truth file for task {task} not found.")
        return None

    pred_path = get_pred_csv_filepath(exp=exp, task=task, shot=shot)
    if not pred_path:
        print(f"Prediction file for {exp} and task {task} with shot {shot} not found.")
        return None
    return None
    #return compute_task_specific_metrics(pred_path, gt_path, task)


def get_gt_csv_filepath(task):
    return get_file_by_keyword(directory=GT_DIR, keyword=task, file_extension='csv')


def get_pred_csv_filepath(exp, task, shot):
    pred_dir = os.path.join(PREDICTION_DIR, exp)
    file_name = f"{task}_{shot}_validation"
    return get_file_by_keyword(directory=pred_dir, keyword=file_name, file_extension='csv')


def get_file_by_keyword(directory, keyword, file_extension=None):
    """Get the path of a unique file in the specified directory that contains the given keyword in its name.

    Args:
    - directory (str): The directory to search in.
    - keyword (str): The keyword to look for in file names.
    - file_extension (str, optional): The desired file extension (e.g., 'csv'). If None, any extension is considered.

    Returns:
    - str/None: The path of the unique file, or None if there's no such file or if multiple such files exist.
    """

    # Build the search pattern based on the presence of a file_extension
    search_pattern = os.path.join(directory, f"*{keyword}*")
    if file_extension:
        search_pattern += f".{file_extension}"

    # Using glob to fetch all files in the directory that match the pattern (case-insensitive)
    matching_files = [f for f in glob.glob(search_pattern, recursive=True) if keyword.lower() in f.lower()]

    # Results handling
    if len(matching_files) == 1:
        return matching_files[0]
    elif len(matching_files) > 1:
        print(
            f"More than one file found in {directory} containing the keyword '{keyword}' with the specified extension.")
    else:
        print(f"No file found in {directory} containing the keyword '{keyword}' with the specified extension.")

    return None


def compute_auc(cls_scores, cls_labels):
    cls_aucs = []
    for i in range(cls_scores.shape[1]):
        scores_per_class = cls_scores[:, i]
        labels_per_class = cls_labels[:, i]
        try:
            auc_per_class = metrics.roc_auc_score(labels_per_class,
                                                  scores_per_class)
            # print('class {} auc = {:.2f}'.format(i + 1, auc_per_class * 100))
        except ValueError:
            pass
        cls_aucs.append(auc_per_class * 100)

    return cls_aucs


def cal_metrics_multilabel(target, cosine_scores):
    """Calculate mean AUC with given dataset information and cosine scores."""

    sample_num = target.shape[0]
    cls_num = cosine_scores.shape[1]

    gt_labels = np.zeros((sample_num, cls_num))
    for k in range(target.shape[0]):
        label = target[k]
        gt_labels[k, :] = label

    cls_scores = np.zeros((sample_num, cls_num))
    for k in range(target.shape[0]):
        cos_score = cosine_scores[k]
        norm_scores = [1 / (1 + math.exp(-1 * v)) for v in cos_score]
        cls_scores[k, :] = np.array(norm_scores)

    cls_aucs = compute_auc(cls_scores, gt_labels)
    mean_auc = np.mean(cls_aucs)

    return mean_auc


def compute_task_specific_metrics(pred_path, gt_path, task):
    """
    Compute metrics based on the provided task.

    Args:
    - pred_path: Path to the predictions CSV.
    - gt_path: Path to the ground truth CSV.
    - task: Name of the task.

    Returns:
    - metrics: Dictionary containing the computed metrics.
    """

    # Read CSVs
    try:
        predictions = pd.read_csv(pred_path)
        ground_truth = pd.read_csv(gt_path)
    except Exception as e:
        print(f"Error reading CSV files: {e}")
        return

    # Check for necessary columns and equal lengths
    for column_name in ['label', 'score']:
        if column_name not in predictions.columns or column_name not in ground_truth.columns:
            print(f"Missing '{column_name}' column in CSV files.")
            return

    if len(predictions) != len(ground_truth):
        print("Predictions and Ground Truth have different lengths.")
        return

    # Compute metrics
    target = torch.tensor(ground_truth['label'].values)
    pred = torch.tensor(predictions['score'].values)

    metrics = {'AUC': cal_metrics_multilabel(target, pred)}

    if task in ['chest', 'endo']:
        metrics['mAP'] = average_precision_score(target.numpy(), pred.numpy()) * 100

    if task == 'colon':
        correct_predictions = sum(predictions['label'] == ground_truth['label'])
        metrics['ACC'] = correct_predictions / len(predictions)

    return metrics


# Directory paths
PREDICTION_DIR = "ensemble/validation/05-09_14-17-34/result"
GT_DIR = "/scratch/medfm/medfm-challenge/data/MedFMC_trainval_annotation/"

shots = ['1-shot', '5-shot', '10-shot']
tasks = ["colon", "endo", "chest"]
exps = ["exp1", "exp2", "exp3", "exp4", "exp5"]

# Iterate over experiments, tasks and shots
results = {exp: {} for exp in exps}

for exp in exps:
    for task in tasks:
        for shot in shots:
            metrics = process_experiment(exp=exp, task=task, shot=shot)
            if metrics:
                results[exp][f"{task}_{shot}"] = metrics

# Display the results
# for exp, metrics in results.items():
#     print(f"Experiment: {exp}")
#     for task, task_metrics in metrics.items():
#         print(f"\tTask: {task}")
#         for metric_name, metric_value in task_metrics.items():
#             print(f"\t\t{metric_name}: {metric_value}")
