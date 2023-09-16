import glob
import json
import os
import re
from datetime import datetime
from multiprocessing import cpu_count, Pool

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score
from termcolor import colored

from utils.constants import shots, exps, tasks, TASK_2_CLASS_NAMES, TASK_2_CLASS_COUNT
from medfmc.evaluation.metrics.auc import cal_metrics_multiclass, cal_metrics_multilabel

TIMESTAMP_PATTERN = re.compile(r"\d{2}-\d{2}_\d{2}-\d{2}-\d{2}")


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

    return json.dumps(json_results, indent=2), json_results["aggregates"]


def process_experiment(pred_dir, exp, task, shot):
    #print(f"Processing Setting:", os.path.join(exp, task, shot))
    gt_path = get_gt_csv_filepath(task=task)
    if not gt_path:
        print(f"Ground truth file for task {task} not found.")
        return None

    pred_path = get_pred_csv_filepath(pred_dir=pred_dir, exp=exp, task=task, shot=shot)
    if not pred_path:
        print(f"Prediction file for {exp} and task {task} with shot {shot} not found.")
        return None

    return compute_task_specific_metrics(pred_path=pred_path, gt_path=gt_path, task=task)


def get_gt_csv_filepath(task):
    return get_file_by_keyword(directory=GT_DIR, keyword=task, file_extension='csv')


def get_pred_csv_filepath(pred_dir, exp, task, shot):
    pred_dir = os.path.join(pred_dir, exp)
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


def sort_key(timestamp):
    """Convert timestamp string to datetime object for sorting."""
    return datetime.strptime(timestamp, "%d-%m_%H-%M-%S")


def get_prediction_timestamp_dirs(base_path):
    all_dirs = [d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))]
    timestamp_dirs = [d for d in all_dirs if TIMESTAMP_PATTERN.match(d)]

    # Filter timestamp directories that have a 'result' sub-folder
    valid_dirs = [d for d in timestamp_dirs if os.path.exists(os.path.join(base_path, d, 'validation', "result"))]

    # If no valid directories are found, return an empty list
    if not valid_dirs:
        print("No valid timestamp directories found.")
        return []

    sorted_valid_dirs = sorted(valid_dirs, key=sort_key)

    return sorted_valid_dirs


def build_pred_log_string(pred_dict):
    timestamp = pred_dict.get('timestamp', "None")
    model_cnt = pred_dict.get('model_count', "None")
    strategy = pred_dict.get('strategy', "None")
    top_k = pred_dict.get('top_k', "None")
    prediction_dir = pred_dict.get('prediction_dir', "None")
    aggregate_value = pred_dict.get('aggregate_value', "None")

    try:
        aggregate_value = float(aggregate_value)
        value_string = f"{aggregate_value:<10.4f}"
    except ValueError:
        value_string = f"{aggregate_value:<10}"

    top_k_str = str(top_k) if top_k is not None else "None"
    model_cnt = str(model_cnt) if model_cnt is not None else "None"

    return f"{timestamp:<20} {model_cnt:<20} {strategy:<20} {top_k_str:<10} {prediction_dir:<40} {value_string}\n"


def load_submission_cfg_dump(dir):
    cfg_file_path = os.path.join(dir, "config.json")

    if not os.path.exists(cfg_file_path):
        return None

    with open(cfg_file_path, 'r') as cfg_file:
        config_data = json.load(cfg_file)
    return config_data


def process_prediction_dir(base_path, timestamp_dir):
    prediction_root_path = os.path.join(base_path, timestamp_dir)
    prediction_result_path = os.path.join(prediction_root_path, 'validation', 'result')

    ensemble_cfg = load_submission_cfg_dump(dir=prediction_root_path)

    results = {exp: {} for exp in exps}

    for exp in exps:
        for task in tasks:
            for shot in shots:
                metrics = process_experiment(pred_dir=prediction_result_path, exp=exp, task=task, shot=shot)
                if metrics:
                    results[exp][f"{task}_{shot}"] = metrics

    json_result, aggregates = generate_json(results=results)

    strategy = "None"
    top_k = "None"
    model_count = "None"
    if ensemble_cfg:
        strategy = ensemble_cfg.get('strategy', strategy)
        top_k = ensemble_cfg.get('top-k', top_k)
        model_count = ensemble_cfg.get('model-count', model_count)

    # Save JSON result to the corresponding timestamp folder
    with open(os.path.join(prediction_root_path, 'validation', 'results.json'), 'w') as json_file:
        json_file.write(json_result)

    return {
            'timestamp': timestamp_dir,
            'model_count': model_count,
            'strategy': strategy,
            'top_k': top_k,
            'prediction_dir': prediction_root_path,
            'aggregate_value': aggregates
    }


def worker_func(base_path, timestamp_dir):
    print(colored(f"Processing Timestamp {timestamp_dir}", 'blue'))
    if timestamp_dir != "02-09_00-32-41":
        return
    return process_prediction_dir(base_path=base_path, timestamp_dir=timestamp_dir)



# ==========================================================================================
GT_DIR = "/scratch/medfm/medfm-challenge/data/MedFMC_trainval_annotation/"
# ==========================================================================================


def main():
    base_path = "ensemble/gridsearch"
    timestamp_dirs = get_prediction_timestamp_dirs(base_path)

    # Number of processes to spawn. You can adjust this value as needed.
    num_processes = min(cpu_count(), len(timestamp_dirs))

    args = [(base_path, timestamp_dir) for timestamp_dir in timestamp_dirs]

    with Pool(num_processes) as pool:
        log_pred_dicts = pool.starmap(worker_func, args)

    log_file_path = os.path.join(base_path, 'log.txt')

    with open(log_file_path, 'w') as log_file:
        log_file.write(f"{'Timestamp':<20} {'Model-Count':<20} {'Strategy':<20} {'Top-K':<10} {'PredictionDir':<40} {'Aggregate':<10}\n")

        for log_pred_dict in log_pred_dicts:
            log_pred_str = build_pred_log_string(log_pred_dict)
            log_file.write(log_pred_str)


if __name__ == "__main__":
    main()
