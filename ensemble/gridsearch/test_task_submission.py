import argparse
import fnmatch
import glob
import json
import os
import re
import shutil
from collections import defaultdict
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score
from termcolor import colored

from ensemble.utils.constants import TASK_2_CLASS_NAMES, TASK_2_CLASS_COUNT
from medfmc.evaluation.metrics.auc import cal_metrics_multiclass, cal_metrics_multilabel

TIMESTAMP_PATTERN = re.compile(r"\d{2}-\d{2}_\d{2}-\d{2}-\d{2}")


def get_timestamp(arguments):
    if arguments.ts:
        return arguments.ts

    return get_newest_timestamp(BASE_PATH)


def get_newest_timestamp(base_path):
    """Get the newest timestamp from directory names in base_path."""
    valid_directories = []

    for d in os.listdir(base_path):
        if os.path.isdir(os.path.join(base_path, d)):
            try:
                date = datetime.strptime(d, "%d-%m_%H-%M-%S")
                valid_directories.append((date, d))
            except ValueError:
                pass

    return max(valid_directories, key=lambda x: x[0])[1]


def create_subm_target_dir():
    submission_target_path = os.path.join("submissions/evaluation", TIMESTAMP)

    if not os.path.isdir(submission_target_path):
        os.makedirs(submission_target_path)

    for exp in EXPS:
        os.makedirs(os.path.join(submission_target_path, "result", f"{exp}"), exist_ok=True)
    print(f"Created {colored(TIMESTAMP, 'red')} submission directory {submission_target_path}")

    return submission_target_path


def build_final_submission(strategies):
    if not BUILD_SUBMISSION:
        return

    subm_base_path = os.path.join(BASE_PATH, TIMESTAMP, 'submission')
    target_dir = create_subm_target_dir()

    for task in TASKS:
        for shot in SHOTS:
            for exp in EXPS:
                strategy = strategies[task][shot][exp]['Strategy']
                top_k = strategies[task][shot][exp]['Top-K']
                csv_file_pattern = f"{task}_{shot}_*.csv"

                if "expert" in strategy:
                    result_path = os.path.join(subm_base_path, task, shot, exp, strategy)
                else:
                    result_path = os.path.join(subm_base_path, task, shot, exp, strategy, f"top-{top_k}")

                csv_file_dir = os.path.join('result', exp)
                source_csv_file_dir = os.path.join(result_path, csv_file_dir)

                for csv_file in os.listdir(source_csv_file_dir):
                    if fnmatch.fnmatch(csv_file, csv_file_pattern):
                        source_csv_file = os.path.join(source_csv_file_dir, csv_file)
                        destination = os.path.join(target_dir, csv_file_dir, csv_file)
                        shutil.copy(source_csv_file, destination)
                        print(f"Copied {csv_file} from {source_csv_file} to {destination}")

    best_strategies_path = os.path.join(VAL_BASE_PATH, "best_strategies_per_task.json")
    final_results_path = os.path.join(VAL_BASE_PATH, "results.json")

    # Copy results.json
    shutil.copy(best_strategies_path, target_dir)
    shutil.copy(final_results_path, target_dir)


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


def get_gt_csv_filepath_for_task(task):
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

    return f"{model_cnt:<15} {strategy:<35} {top_k_str:<10} {prediction_dir:<55} {value_string}\n"


def load_submission_cfg_dump(dir):
    cfg_file_path = os.path.join(dir, "config.json")

    if not os.path.exists(cfg_file_path):
        return None

    with open(cfg_file_path, 'r') as cfg_file:
        config_data = json.load(cfg_file)
    return config_data


def extract_number_from_string(s):
    return int(''.join(filter(str.isdigit, s)))


def get_best_strategy_for_setting(task, shot, exp):
    # Read the log.txt file
    log_path = os.path.join(BASE_PATH, TIMESTAMP, 'validation', task, shot, exp, 'log.txt')
    with open(log_path, 'r') as file:
        lines = file.readlines()

    # Skip the header
    lines = lines[1:]

    best_aggregate = float('-inf')
    best_result = None

    # Extract the best ensemble strategy for the given setting
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

    return best_result


def extract_setting_specific_result(result_file, task, shot, exp):
    task_shot = f'{task}_{shot}'
    return result_file['task'][exp][task_shot]


def load_metrics_for_setting(task, shot, exp, strategy_info):
    strategy = strategy_info['Strategy']
    top_k = strategy_info['Top-K']

    # Load the metric scores for the best ensemble strategy for the current setting
    if "expert" in strategy:
        results_file_path = os.path.join(BASE_PATH, TIMESTAMP, 'validation', task, shot, exp,
                                         strategy, "results.json")
    else:
        results_file_path = os.path.join(BASE_PATH, TIMESTAMP, 'validation', task, shot, exp,
                                         strategy, f"top-{top_k}", "results.json")
    with open(results_file_path, 'r') as results_file:
        results_data = json.load(results_file)

    return results_data


def compile_results_to_json():
    best_strategy_path = os.path.join(VAL_BASE_PATH, "best_strategies_per_task.json")

    if FROM_FILE:
        best_strategy_per_setting = load_best_strategies_from_json(best_strategy_path)
    else:
        best_strategy_per_setting = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))

    final_results = {
        "task": defaultdict(lambda: defaultdict(dict)),
        "aggregates": 0
    }

    metrics_sum = 0.0
    metrics_count = 0

    for task in TASKS:
        for shot in SHOTS:
            for exp in EXPS:

                if FROM_FILE:
                    strategy_info = best_strategy_per_setting[task][shot][exp]

                else:
                    strategy_info = get_best_strategy_for_setting(task=task, shot=shot, exp=exp)
                    best_strategy_per_setting[task][shot][exp] = strategy_info

                results_json = load_metrics_for_setting(task=task, shot=shot, exp=exp,
                                                        strategy_info=strategy_info)

                task_shot = f'{task}_{shot}'
                result = results_json['task'][exp][task_shot]
                final_results['task'][exp][task_shot] = result

                # Accumulate setting's metrics for aggregate computation
                for metric_val in result.values():
                    metrics_sum += float(metric_val)
                    metrics_count += 1

    # Compute the aggregate value
    final_results["aggregates"] = metrics_sum / metrics_count if metrics_count != 0 else 0

    print(json.dumps(final_results, indent=4))

    # Save the final results to the timestamp directory
    output_json_path = os.path.join(VAL_BASE_PATH, "results.json")
    with open(output_json_path, 'w') as file:
        json.dump(final_results, file, indent=4)
    print(f"Wrote Final Result JSON file to {output_json_path}")
    print(json.dumps(final_results, indent=4))

    if not FROM_FILE:
        # Save the best ensembles to the timestamp directory
        best_strategies_out_path = os.path.join(VAL_BASE_PATH, "best_strategies_per_task.json")
        with open(best_strategies_out_path, 'w') as file:
            json.dump(best_strategy_per_setting, file, indent=4)
        print(f"\nWrote Best Ensembles JSON file to {best_strategies_out_path}")
    print(json.dumps(best_strategy_per_setting, indent=4))

    return best_strategy_per_setting


def softmax(values):
    exp_values = np.exp(values)
    return exp_values / np.sum(exp_values, axis=1, keepdims=True)


def process_csv_to_df(filename):
    df = pd.read_csv(filename, header=None)
    df[[1, 2]] = softmax(df[[1, 2]].values)
    return df


def process_experiment(top_k_path, exp, task, shot):
    gt_path = get_gt_csv_filepath_for_task(task=task)
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
    if COLON_SOFTMAX_PRINT and task == "colon":
        colon_df = process_csv_to_df(pred_csv_file_path)
        print(f"Processed {pred_csv_file_path} to dataframe")
        metrics_softmaxed_dict = compute_task_specific_metrics(pred_path=colon_df,
                                                               gt_path=gt_path,
                                                               task=task,
                                                               pred_is_df=True)
        print(f"""Colon Metrics Normal:\tAUC:{metrics_dict['AUC']:.4f}\tACC:{metrics_dict['ACC']:.4f}""")
        print(f"""Colon Metrics Softmax:\tAUC:{metrics_softmaxed_dict['AUC']:.4f}\tACC:{metrics_softmaxed_dict['ACC']:.4f}""")
    return metrics_dict


def process_top_k(top_k_num, strategy_path, task, shot, exp):
    if top_k_num:
        top_k_path = os.path.join(strategy_path, f"top-{str(top_k_num)}")
    else:
        top_k_path = strategy_path

    ensemble_cfg = load_submission_cfg_dump(dir=top_k_path)

    results = defaultdict(lambda: defaultdict(dict))
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


def recursive_defaultdict():
    return defaultdict(recursive_defaultdict)


def extract_top_k_from_folder(folder_name):
    """Extracts the top-k number from the folder name."""
    prefix = "top-"
    if folder_name.startswith(prefix):
        return int(folder_name[len(prefix):])
    return None  # Return None if the folder doesn't match the expected format


def get_top_k_dirs(strategy_path, strategy):
    """Compute the top-k values for the given strategy."""
    if "expert" in strategy:
        return [None]

    folder_names = os.listdir(strategy_path)
    return sorted(extract_top_k_from_folder(folder) for folder in folder_names)


def process_timestamp():
    print(colored(f"Processing Timestamp {TIMESTAMP}", 'blue'))

    if FROM_FILE:
        return

    result_dicts = recursive_defaultdict()

    for task in TASKS:
        print(colored(f"\tProcessing Task {task}", 'cyan'))

        for shot in SHOTS:
            print(colored(f"\t\tProcessing Shot {shot}", 'cyan'))

            for exp in EXPS:
                print(colored(f"\t\t\tProcessing Experiment {exp}", 'cyan'))

                for strategy in ENSEMBLE_STRATEGIES:
                    print(colored(f"\t\t\t\tProcessing Strategy {strategy}", 'light_red'))

                    strategy_path = os.path.join(BASE_PATH, TIMESTAMP, 'validation', task, shot, exp, strategy)
                    top_k_dirs = get_top_k_dirs(strategy_path, strategy)

                    results = []
                    for top_k_num in top_k_dirs:
                        top_k_result = process_top_k(top_k_num=top_k_num, strategy_path=strategy_path,
                                                     task=task, shot=shot, exp=exp)
                        results.append(top_k_result)

                    result_dicts[task][shot][exp][strategy] = results

    return result_dicts


def create_log_files(data):
    if FROM_FILE:
        return

    for task in TASKS:
        for shot in SHOTS:
            for exp in EXPS:
                log_file_path = os.path.join(BASE_PATH, TIMESTAMP, 'validation', task, shot, exp, 'log.txt')

                lines = []
                for strategy in ENSEMBLE_STRATEGIES:
                    for top_k in data[task][shot][exp][strategy]:
                        log_pred_str = build_log_string(top_k, task=task)
                        lines.append(log_pred_str)

                lines = sorted(lines, key=lambda x: float(x.split()[-1]))

                with open(log_file_path, 'w') as log_file:
                    log_file.write(
                        f"{'Model-Count':<15} {'Strategy':<35} {'Top-K':<10} {'PredictionDir':<55} {'Aggregate':<10}\n")
                    for line in lines:
                        log_file.write(line)
                    print(f"Wrote Log file to {TIMESTAMP}/validation/{task}/{shot}/{exp}/log.txt")


def load_best_strategies_from_json(path):
    with open(path, 'r') as file:
        best_strategies = json.load(file)

    return best_strategies


# ===================  DEFAULT PARAMS  ====================================================
BASE_PATH = "ensemble/gridsearch"
GT_DIR = "/scratch/medfm/medfm-challenge/data/MedFMC_trainval_annotation/"
WORK_DIR = "/scratch/medfm/medfm-challenge/work_dirs"
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
                       # "diversity-weighted"
                       ]
COLON_SOFTMAX_PRINT = False
FROM_FILE = False
BUILD_SUBMISSION = True
# ==========================================================================================


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test validation submission directories for their performance.")
    parser.add_argument("--ts", help="Timestamp for the directory.")
    args = parser.parse_args()

    TIMESTAMP = get_timestamp(args)
    VAL_BASE_PATH = os.path.join(BASE_PATH, TIMESTAMP, 'validation')

    # ===== MAIN LOOP =====
    # Creates results.json for each setting * strategy * top-k directory
    result = process_timestamp()

    # Creates log.txt for each setting directory
    create_log_files(data=result)

    # Creates ONE best_strategies_per_setting.json & ONE results.json in validation root dir
    best_strategies = compile_results_to_json()
    build_final_submission(strategies=best_strategies)
