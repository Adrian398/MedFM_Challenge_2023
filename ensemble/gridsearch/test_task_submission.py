import fnmatch
import glob
import json
import os
import re
import shutil
from collections import defaultdict
from datetime import datetime
from multiprocessing import cpu_count, Pool

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score
from termcolor import colored

from ensemble.gridsearch.task_submission import ENSEMBLE_STRATEGIES
from ensemble.utils.constants import shots, exps, TASK_2_CLASS_NAMES, TASK_2_CLASS_COUNT, tasks
from medfmc.evaluation.metrics.auc import cal_metrics_multiclass, cal_metrics_multilabel
from utility.softmax_submission import process_csv, softmax

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


def get_gt_csv_filepath(task):
    return get_file_by_keyword(directory=GT_DIR, keyword=task, file_extension='csv')


def get_pred_csv_filepath(pred_csv_path, exp, task, shot):
    exp_path = os.path.join(pred_csv_path, exp)
    file_name = f"{task}_{shot}_validation"
    return get_file_by_keyword(directory=exp_path, keyword=file_name, file_extension='csv')


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


def read_and_validate_files(pred_path, gt_path, task, pred_is_df):
    """Read prediction and ground truth files, then validate the necessary columns."""
    try:
        gt_df = pd.read_csv(gt_path)
    except Exception as e:
        raise ValueError(f"Error reading CSV files: {e}")

    if pred_is_df:
        pred_df = pred_path
    else:
        try:
            pred_df = pd.read_csv(pred_path)
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


def compute_task_specific_metrics(pred_path, gt_path, task, pred_is_df=False):
    pred_df, gt_df, score_cols = read_and_validate_files(pred_path, gt_path, task, pred_is_df)

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


def find_result_folder(directory):
    """Recursively search for a 'result' folder in the given directory."""
    for root, dirs, files in os.walk(directory):
        if "result" in dirs:
            return True
    return False


def get_prediction_timestamp_dirs(base_path):
    all_dirs = [d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))]
    timestamp_dirs = [d for d in all_dirs if TIMESTAMP_PATTERN.match(d) and d not in TIMESTAMPS_2_IGNORE]

    valid_dirs = [d for d in timestamp_dirs if find_result_folder(os.path.join(base_path, d))]

    # If no valid directories are found, return an empty list
    if not valid_dirs:
        print("No valid timestamp directories found.")
        return []

    sorted_valid_dirs = sorted(valid_dirs, key=sort_key)

    return sorted_valid_dirs


def build_log_string(data, task):
    model_cnt = data.get('model_count', "None")
    strategy = data.get('strategy', "None")
    top_k = data.get('top_k', "None")
    prediction_dir = data.get('prediction_dir', "None")
    aggregate_value = data.get('aggregate_value', "None")

    try:
        aggregate_value = float(aggregate_value)
        value_string = f"{aggregate_value:<10.4f}"
    except ValueError:
        value_string = f"{aggregate_value:<10}"

    top_k_str = str(top_k) if top_k is not None else "None"
    model_cnt = str(model_cnt) if model_cnt is not None else "None"
    prediction_dir = prediction_dir.split(f"{task}/")[1]

    return f"{model_cnt:<15} {strategy:<20} {top_k_str:<10} {prediction_dir:<40} {value_string}\n"


def load_submission_cfg_dump(dir):
    cfg_file_path = os.path.join(dir, "config.json")

    if not os.path.exists(cfg_file_path):
        return None

    with open(cfg_file_path, 'r') as cfg_file:
        config_data = json.load(cfg_file)
    return config_data


def extract_number_from_string(s):
    return int(''.join(filter(str.isdigit, s)))


def compile_results_to_json(base_path, timestamp, tasks):
    output_json_path = os.path.join(base_path, timestamp, 'validation', "best_ensembles.json")
    print(f"Wrote Result JSON file to {output_json_path}")

    final_results = {
        "task": {},
        "aggregates": 0
    }

    best_ensembles_per_task = {}

    metrics_sum = 0.0
    metrics_count = 0

    for task in tasks:
        task_log_path = os.path.join(base_path, timestamp, 'validation', task, 'log.txt')
        with open(task_log_path, 'r') as file:
            lines = file.readlines()

        # Skip the header
        lines = lines[1:]

        best_aggregate = float('-inf')
        best_result = None

        for line in lines:
            model_count, strategy, top_k, _, aggregate = line.split()
            aggregate_value = float(aggregate)

            if aggregate_value > best_aggregate:
                best_aggregate = aggregate_value
                best_result = {
                    "Model-Count": model_count,
                    "Strategy": strategy,
                    "Top-K": top_k,
                    "Aggregate": aggregate
                }

        best_ensembles_per_task[task] = best_result

        strategy = best_result['Strategy']
        top_k = best_result['Top-K']

        if strategy == "expert":
            results_file_path = os.path.join(base_path, timestamp, 'validation', task, strategy, "results.json")
        else:
            results_file_path = os.path.join(base_path, timestamp, 'validation', task, strategy,
                                             f"top-{top_k}", "results.json")

        with open(results_file_path, 'r') as results_file:
            results_data = json.load(results_file)

        # Merge the task results into the main results
        for exp_key, exp_value in results_data['task'].items():
            if exp_key not in final_results['task']:
                final_results['task'][exp_key] = {}
            final_results['task'][exp_key].update(exp_value)

            # Accumulate metrics for aggregate computation
            for _, metrics in exp_value.items():
                for metric_score in metrics.values():
                    metrics_sum += float(metric_score)
                    metrics_count += 1

    # Compute the aggregate value
    final_results["aggregates"] = metrics_sum / metrics_count if metrics_count != 0 else 0

    # Save the final results to the timestamp directory
    output_json_path = os.path.join(base_path, timestamp, 'validation', "results.json")
    with open(output_json_path, 'w') as file:
        json.dump(final_results, file, indent=4)

    # Save the best ensembles to the timestamp directory
    best_ensembles_output_path = os.path.join(base_path, timestamp, 'validation', "best_ensemble_per_task.json")
    with open(best_ensembles_output_path, 'w') as file:
        json.dump(best_ensembles_per_task, file, indent=4)

    print(f"Wrote Final Result JSON file to {output_json_path}")
    print(json.dumps(final_results, indent=4))
    print(f"\nWrote Best Ensembles JSON file to {best_ensembles_output_path}")
    print(json.dumps(best_ensembles_per_task, indent=4))

    return final_results, best_ensembles_per_task, output_json_path, best_ensembles_output_path


def process_csv_to_df(filename):
    df = pd.read_csv(filename)
    df[['col2', 'col3']] = softmax(df[['col2', 'col3']].values)

    return df


def process_experiment(top_k_path, exp, task, shot):
    gt_path = get_gt_csv_filepath(task=task)
    if not gt_path:
        print(f"Ground truth file for task {task} not found.")
        return None

    pred_csv_path = os.path.join(top_k_path, "result")
    pred_csv_file_path = get_pred_csv_filepath(pred_csv_path=pred_csv_path, exp=exp, task=task, shot=shot)
    if not pred_csv_file_path:
        print(f"Prediction file for {exp} and task {task} with shot {shot} not found.")
        return None

    metrics_dict = compute_task_specific_metrics(pred_path=pred_csv_file_path, gt_path=gt_path, task=task)

    # Perform Softmax before score calculation
    if task == "colon":
        colon_df = process_csv_to_df(pred_csv_file_path)
        print(f"Processed {pred_csv_file_path} to dataframe")
        metrics_softmaxed_dict = compute_task_specific_metrics(pred_path=colon_df,
                                                               gt_path=gt_path,
                                                               task=task,
                                                               pred_is_df=True)
        print(f"""Colon Metrics Normal:\tAUC:{metrics_softmaxed_dict['AUC']:.4f}\tACC:{metrics_softmaxed_dict['ACC']:.4f}""")
        print(f"""Colon Metrics Softmax:\tAUC:{metrics_softmaxed_dict['AUC']:.4f}\tACC:{metrics_softmaxed_dict['ACC']:.4f}""")
    return metrics_dict


def process_top_k(top_k, strategy_path, task):
    top_k_path = strategy_path
    if top_k:
        top_k_path = os.path.join(strategy_path, top_k)
        #print(colored(f"\t\t\tProcessing Top-K {top_k}", 'light_grey'))

    ensemble_cfg = load_submission_cfg_dump(dir=top_k_path)

    results = {exp: {} for exp in exps}
    for exp in exps:
        for shot in shots:
            metrics = process_experiment(top_k_path=top_k_path, exp=exp, task=task, shot=shot)
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
    with open(os.path.join(top_k_path, 'results.json'), 'w') as json_file:
        json_file.write(json_result)

    return {
        'model_count': model_count,
        'strategy': strategy,
        'top_k': top_k,
        'prediction_dir': top_k_path,
        'aggregate_value': aggregates
    }


def process_strategy(task_path, strategy, task):
    print(colored(f"\t\tProcessing Strategy {strategy}", 'light_red'))

    strategy_path = os.path.join(task_path, strategy)

    result_dicts = []
    if strategy == "expert":
        result_dict = process_top_k(top_k=None, strategy_path=strategy_path, task=task)
        result_dicts.append(result_dict)
    else:
        for top_k in sorted(os.listdir(strategy_path), key=extract_number_from_string):
            result_dict = process_top_k(top_k=top_k, strategy_path=strategy_path, task=task)
            result_dicts.append(result_dict)

    return result_dicts


def process_task(timestamp_path, task):
    print(colored(f"\tProcessing Task {task}", 'cyan'))

    task_path = os.path.join(timestamp_path, task)

    task_result_dicts = defaultdict()
    for strategy in ENSEMBLE_STRATEGIES:
        strategy_result_dicts = process_strategy(task_path=task_path,
                                                 strategy=strategy,
                                                 task=task)
        task_result_dicts[strategy] = strategy_result_dicts

    return task_result_dicts


def process_timestamp(base_path, timestamp, tasks):
    print(colored(f"Processing Timestamp {timestamp}", 'blue'))

    timestamp_path = os.path.join(base_path, timestamp, 'validation')

    timestamp_result_dicts = {timestamp: {}}
    for task in tasks:
        task_result_dicts = process_task(timestamp_path=timestamp_path, task=task)
        timestamp_result_dicts[timestamp][task] = task_result_dicts

    return timestamp_result_dicts


def worker_func(base_path, timestamp, tasks):
    return process_timestamp(base_path=base_path, timestamp=timestamp, tasks=tasks)


def create_subm_target_dir(timestamp):
    # Create submission target directory
    submission_target_path = os.path.join("submissions/evaluation", timestamp)
    if not os.path.isdir(submission_target_path):
        os.makedirs(submission_target_path)

    for exp in exps:
        os.makedirs(os.path.join(submission_target_path, "result", f"{exp}"), exist_ok=True)
    print(f"Created {colored(timestamp, 'red')} submission directory {submission_target_path}")

    return submission_target_path


def build_final_submission(base_path, timestamp, strategies, ensemble_path, json_path):
    submission_path = os.path.join(base_path, timestamp, 'submission')
    target_dir = create_subm_target_dir(timestamp=timestamp)

    for task in tasks:
        strategy = strategies[task]['Strategy']
        top_k = strategies[task]['Top-K']
        csv_file_pattern = f"{task}_*.csv"

        for exp in exps:
            if strategy == "expert":
                result_path = os.path.join(submission_path, task, strategy)
            else:
                result_path = os.path.join(submission_path, task, strategy, f"top-{top_k}")

            csv_file_dir = os.path.join('result', exp)
            source_csv_file_dir = os.path.join(result_path, csv_file_dir)

            for csv_file in os.listdir(source_csv_file_dir):
                if fnmatch.fnmatch(csv_file, csv_file_pattern):
                    source_csv_file = os.path.join(source_csv_file_dir, csv_file)
                    destination = os.path.join(target_dir, csv_file_dir, csv_file)
                    shutil.copy(source_csv_file, destination)
                    print(f"Copied {csv_file} from {source_csv_file} to {destination}")

    # Copy results.json
    shutil.copy(json_path, target_dir)
    shutil.copy(ensemble_path, target_dir)


# ==========================================================================================
GT_DIR = "/scratch/medfm/medfm-challenge/data/MedFMC_trainval_annotation/"
WORK_DIR = "/scratch/medfm/medfm-challenge/work_dirs"
TIMESTAMPS_2_IGNORE = ["02-09_00-32-41"]
# ==========================================================================================


def main():
    base_path = "ensemble/gridsearch"
    timestamps = get_prediction_timestamp_dirs(base_path)

    # Number of processes to spawn. You can adjust this value as needed.
    num_processes = min(cpu_count(), len(timestamps))

    tasks = ["colon", "endo", "chest"]
    args = [(base_path, timestamp, tasks) for timestamp in timestamps]

    with Pool(num_processes) as pool:
        results_list = pool.starmap(worker_func, args)

    timestamps_dict = {key: value for d in results_list for key, value in d.items()}

    for timestamp_key, timestamp_dict in timestamps_dict.items():
        for task_key, task_dict in timestamp_dict.items():
            log_file_path = os.path.join(base_path, timestamp_key, 'validation', task_key, 'log.txt')

            lines = []
            for strategy_key, strategy_list in task_dict.items():
                for top_k_item in strategy_list:
                    log_pred_str = build_log_string(top_k_item, task_key)
                    lines.append(log_pred_str)

            lines = sorted(lines, key=lambda x: float(x.split()[-1]))

            with open(log_file_path, 'w') as log_file:
                log_file.write(
                    f"{'Model-Count':<15} {'Strategy':<20} {'Top-K':<10} {'PredictionDir':<40} {'Aggregate':<10}\n")
                for line in lines:
                    log_file.write(line)
                print(f"Wrote Log file to {timestamp_key}/{task_key}/log.txt")

        _, strategy_per_task, json_path, ensemble_path = compile_results_to_json(base_path=base_path, timestamp=timestamp_key, tasks=tasks)

        build_final_submission(base_path=base_path,
                               timestamp=timestamp_key,
                               strategies=strategy_per_task,
                               json_path=json_path,
                               ensemble_path=ensemble_path)


if __name__ == "__main__":
    main()
