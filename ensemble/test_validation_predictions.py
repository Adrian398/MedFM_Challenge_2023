import glob
import json
import os

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score
from utils.constants import shots, exps, tasks, TASK_2_CLASS_NAMES, TASK_2_CLASS_COUNT
from medfmc.evaluation.metrics.auc import cal_metrics_multiclass, cal_metrics_multilabel


def generate_json(results):
    """Generate a refined JSON object with aggregates."""

    # Define expected metrics for each task type
    expected_metrics = {
        'colon': ['ACC', 'AUC'],
        'chest': ['AUC', 'mAP'],
        'endo': ['AUC', 'mAP']
    }

    json_results = {"task": {}}
    total_sum = 0
    total_metrics_count = 0

    # Helper function to safely get metric value
    def get_metric_value(metrics, metric_name):
        return metrics.get(metric_name, None)

    # Iterate over experiments
    for exp, task_data in results.items():
        json_results["task"][exp] = {}

        # For each combination of task and shot
        for task_shot, metrics in task_data.items():
            task_type = task_shot.split('_')[0]
            json_results["task"][exp][task_shot] = {}

            # Ensure expected metrics are present
            for metric_name in expected_metrics[task_type]:
                metric_value = get_metric_value(metrics, metric_name)
                if metric_value is not None:
                    total_sum += metric_value
                    total_metrics_count += 1
                json_results["task"][exp][task_shot][f"{metric_name}_metric"] = str(metric_value)

    # Calculate the aggregate value
    aggregate_value = total_sum / total_metrics_count if total_metrics_count > 0 else 0
    json_results["aggregates"] = str(aggregate_value)

    return json.dumps(json_results, indent=2)


def process_experiment(exp, task, shot):
    print(f"Processing Setting:", os.path.join(exp, task, shot))
    gt_path = get_gt_csv_filepath(task=task)
    if not gt_path:
        print(f"Ground truth file for task {task} not found.")
        return None

    pred_path = get_pred_csv_filepath(exp=exp, task=task, shot=shot)
    if not pred_path:
        print(f"Prediction file for {exp} and task {task} with shot {shot} not found.")
        return None

    return compute_task_specific_metrics(pred_path=pred_path, gt_path=gt_path, task=task)


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

    # Search pattern only based on the file extension (if present)
    search_pattern = os.path.join(directory, f"*.{file_extension}" if file_extension else "*")

    # Fetch all files in the directory that match the pattern
    all_files = glob.glob(search_pattern)

    # Filter files based on case-insensitive presence of the keyword
    matching_files = [f for f in all_files if keyword.lower() in os.path.basename(f).lower()]

    # Results handling
    if len(matching_files) == 1:
        return matching_files[0]
    elif len(matching_files) > 1:
        print(
            f"More than one file found in {directory} containing the keyword '{keyword}' with the specified extension.")
    else:
        print(f"No file found in {directory} containing the keyword '{keyword}' with the specified extension.")

    return None


def read_and_validate_files(pred_path, gt_path, task):
    """Read prediction and ground truth files, then validate the necessary columns."""
    try:
        pred_df = pd.read_csv(pred_path)
        gt_df = pd.read_csv(gt_path)
    except Exception as e:
        raise ValueError(f"Error reading CSV files: {e}")

    score_cols = [f'score_{i}' for i in range(TASK_2_CLASS_COUNT.get(task, 2))]
    pred_df.columns = ['img_id'] + score_cols

    # Validate columns in prediction
    missing_cols = [col for col in ['img_id'] + score_cols if col not in pred_df.columns]
    if missing_cols:
        raise ValueError(f"Missing columns in the predictions: {missing_cols}")

    return pred_df, gt_df, score_cols


def compute_colon_metrics(df, score_cols):
    """
    Compute metrics for the 'colon' task.
    """
    target = df['tumor'].values
    pred_scores = df[score_cols].values

    # Calculate AUC using multiclass method
    metrics_dict = {'AUC': cal_metrics_multiclass(target, pred_scores)}

    # Calculate ACC (accuracy)
    pred_labels = np.argmax(pred_scores, axis=1)  # Get the predicted class (0 or 1)
    correct_predictions = sum(pred_labels == target)
    metrics_dict['ACC'] = correct_predictions / len(pred_labels) * 100  # Express accuracy as a percentage

    return metrics_dict


def compute_multilabel_metrics(merged_df, target_columns, score_cols, num_classes):
    """Compute metrics for multi-label tasks ('chest' and 'endo')."""
    target = merged_df[target_columns].values
    pred_scores = merged_df[score_cols].values

    metrics_dict = {'AUC': cal_metrics_multilabel(target, pred_scores)}

    # Compute mAP for each label and then average them
    ap_scores = [average_precision_score(target[:, i], pred_scores[:, i]) for i in range(num_classes)]
    metrics_dict['mAP'] = np.mean(ap_scores) * 100

    return metrics_dict


def compute_task_specific_metrics(pred_path, gt_path, task):
    pred_df, gt_df, score_cols = read_and_validate_files(pred_path, gt_path, task)

    target_columns = TASK_2_CLASS_NAMES.get(task, [])

    # Merge predictions and ground truth based on img_id
    merged_df = pd.merge(pred_df, gt_df, on='img_id', how='inner')

    if task == 'colon':
        return compute_colon_metrics(merged_df, score_cols)
    elif task in ['chest', 'endo']:
        num_classes = TASK_2_CLASS_COUNT.get(task, 2)
        return compute_multilabel_metrics(merged_df, target_columns, score_cols, num_classes)
    else:
        raise ValueError(f"Invalid task: {task}")


def get_prediction_dir():
    while True:
        timestamp = input("Please enter the timestamp (format: DD-MM_HH-MM-SS): ")
        dir_path = f"ensemble/validation/{timestamp}/result"

        if os.path.exists(dir_path):
            return dir_path
        else:
            print(f"Directory '{dir_path}' does not exist. Please enter a valid timestamp.")


# ==========================================================================================
#PREDICTION_DIR = "ensemble/validation/06-09_22-45-28/result"  # Weighted Sum Model Ensemble Top7: 61.32564695023872 (eval=)
#PREDICTION_DIR = "ensemble/validation/06-09_21-14-47/result"  # Weighted Sum Model Ensemble Top3: 61.3906 (eval=06-09_21-07-17)
#PREDICTION_DIR = "ensemble/validation/05-09_14-17-34/result"  # Expert per Class Model Ensemble
#PREDICTION_DIR = "ensemble/validation/02-09_00-32-41/result"  # Expert
GT_DIR = "/scratch/medfm/medfm-challenge/data/MedFMC_trainval_annotation/"
# ==========================================================================================


if __name__ == "__main__":
    PREDICTION_DIR = get_prediction_dir()

    results = {exp: {} for exp in exps}

    for exp in exps:
        for task in tasks:
            for shot in shots:
                metrics = process_experiment(exp=exp, task=task, shot=shot)
                if metrics:
                    results[exp][f"{task}_{shot}"] = metrics

    json_result = generate_json(results=results)
    print(json_result)
