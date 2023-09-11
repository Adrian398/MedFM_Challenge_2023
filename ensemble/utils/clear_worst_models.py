import json
import os
import re
import shutil
import sys
from collections import defaultdict
from multiprocessing import Pool

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from termcolor import colored


EXP_PATTERN = re.compile(r'exp(\d+)')


def print_report(invalid_model_dirs, best_scores, model_performance):
    if len(invalid_model_dirs) == 0:
        print(colored(f"\nNo models found that are at least {SCORE_INTERVAL}x worse than the BEST_SCORE found!\n",
                      'green'))
        exit()
    else:
        # Find the maximum path length to adjust the table formatting dynamically
        max_path_length = max([len(entry.split("/work_dirs/")[-1]) for entry in invalid_model_dirs])
        padding = 38
        sorted_report_entries = sorted([model_dir for model_dir in invalid_model_dirs], key=sort_key)
        print("\n" + "-" * (max_path_length + padding))
        print(f"| Models that were found with a {SCORE_INTERVAL}x worse score than the best:")
        print("-" * (max_path_length + padding))
        print(f"| {'Path':<{max_path_length}} | Metric    | Score  | Best Score |")
        print(f"| {'-' * max_path_length} |-----------|--------|------------|")
        for entry in sorted_report_entries:
            relative_path = entry.split("/work_dirs/")[-1]
            parts = entry.split('/')
            task = parts[5]
            shot = parts[6].split('-')[0]
            exp_num = extract_exp_number(entry)  # Extracting the exp number from the model directory path
            best_score_for_task_shot_exp = best_scores.get((task, shot, exp_num))
            metric_for_task = task_specific_metrics.get(task, "Aggregate")
            score_for_model = model_performance.get(entry)
            print(
                f"| {relative_path:<{max_path_length}} | {metric_for_task:9} | {score_for_model:6.2f} | {best_score_for_task_shot_exp:10.2f} |")
        print("-" * (max_path_length + padding))
        print(f"| Found {len(invalid_model_dirs)} bad model runs.")
        print("-" * (max_path_length + padding))



def extract_exp_number(string):
    match = EXP_PATTERN.search(string)
    return int(match.group(1)) if match else None


def sort_key(entry):
    # Extract task, shot, and exp number from the entry
    parts = entry.split('/')
    task = parts[5]
    shot = int(parts[6].split('-')[0])
    exp_num = extract_exp_number(entry)  # Extracting the exp number from the model directory path
    score = model_performance.get(entry, float('-inf'))  # If no score is found, default to negative infinity
    return task, shot, exp_num, score


def my_print(message):
    sys.stdout.write(str(message) + '\n')
    sys.stdout.flush()


def get_event_file_from_run_dir(run_dir):
    try:
        for entry in os.listdir(run_dir):
            full_path = os.path.join(run_dir, entry)
            if os.path.isdir(full_path):
                full_path = os.path.join(full_path, "vis_data")
                event_file = os.listdir(full_path)[0]
                return os.path.join(full_path, event_file)
    except Exception:
        return None


def compute_colon_aggregate(data, filepath):
    """
    Compute metrics for the 'colon' task.
    """
    auc_label = "AUC/AUC_multiclass"
    acc_label = "accuracy/top1"

    if auc_label not in data:
        print(f"Metric '{auc_label}' not found in {filepath}")
        return None

    if acc_label not in data:
        print(f"Metric '{acc_label}' not found in {filepath}")
        return None

    return (data[auc_label] + data[acc_label])/2


def compute_multilabel_aggregate(data, filepath):
    """Compute metrics for multi-label tasks ('chest' and 'endo')."""
    auc_label = "AUC/AUC_multilabe"
    map_label = "multi-label/mAP"

    if auc_label not in data:
        print(f"Metric '{auc_label}' not found in {filepath}")
        return None

    if map_label not in data:
        print(f"Metric '{map_label}' not found in {filepath}")
        return None

    return (data[auc_label] + data[map_label])/2


def extract_metric_from_performance_json(model_dir, task):
    for dirpath, _, filenames in os.walk(model_dir):
        for filename in filenames:
            if filename == "performance.json":
                filepath = os.path.join(dirpath, filename)
                try:
                    with open(filepath, 'r') as file:
                        data = json.load(file)

                    if task == 'colon':
                        return compute_colon_aggregate(data, filepath)
                    elif task in ['chest', 'endo']:
                        return compute_multilabel_aggregate(data, filepath)
                    else:
                        raise ValueError(f"Invalid task: {task}")

                except json.JSONDecodeError:
                    print(f"Cannot load JSON from: {filepath}")
                    return None
                except Exception as e:
                    print(f"Error encountered: {e}")
                    return None
    return None


def find_and_validate_json_files(model_dir, task):
    json_files_found = False  # To track if we found any JSON files
    performance_json_count = 0  # To track the number of "performance.json" files found

    for dirpath, dirnames, filenames in os.walk(model_dir):
        for filename in filenames:
            if filename.endswith('.json'):
                json_files_found = True
                filepath = os.path.join(dirpath, filename)

                try:
                    with open(filepath, 'r') as file:
                        data = json.load(file)

                    # If filename is "performance.json", further check for "MAP_Class1"
                    if filename == "performance.json":
                        performance_json_count += 1

                        if "MAP_class1" not in data:
                            print(f"Found 'performance.json' but mAP per class (e.g. 'MAP_class1') missing")
                            return False

                except json.JSONDecodeError:
                    print(f"Cannot load JSON from: {filepath}")
                    print(f"Deleting {filepath}")
                    os.remove(filepath)  # Deleting the corrupted JSON file
                    return False
                except PermissionError as permission_error:
                    print(f"Permission Error encountered: {permission_error}")
                    return False
                except Exception as e:
                    print(f"Error encountered: {e}")
                    return False

    if not json_files_found:
        print("No JSON files found.")
        return False

    if performance_json_count != 1:
        print(f"Multiple 'performance.json' found: {performance_json_count}")
        return False

    return True


def get_worst_performing_model_dirs(task, shot):
    bad_performing_models = []
    best_scores_for_each_setting = {}

    setting_directory = os.path.join(work_dir_path, task, f"{shot}-shot")

    try:
        setting_model_dirs = os.listdir(setting_directory)
    except Exception:
        return None, None

    model_performance = {}
    for model_dir in setting_model_dirs:
        abs_model_dir = os.path.join(setting_directory, model_dir)

        # Only consider model directories with a performance.json file
        if find_and_validate_json_files(abs_model_dir, task):
            aggregate_metric = extract_metric_from_performance_json(abs_model_dir, task)
            if aggregate_metric is not None:
                model_performance[abs_model_dir] = aggregate_metric

    # Group by exp number
    exp_grouped_scores = {}
    invalid_runs = []
    for model_dir, score in model_performance.items():
        exp_num = extract_exp_number(model_dir)

        if exp_num:
            if exp_num not in exp_grouped_scores:
                exp_grouped_scores[exp_num] = []
            exp_grouped_scores[exp_num].append((model_dir, score))
        else:
            invalid_runs.append((model_dir, score))

    sorted_exp_nums = sorted(exp_grouped_scores.keys(), key=lambda x: int(x))

    for exp_num in sorted_exp_nums:
        scores = exp_grouped_scores[exp_num]

        best_score_for_exp_group = max(score for _, score in scores)
        best_scores_for_each_setting[(task, shot, exp_num)] = best_score_for_exp_group

        print(colored(f"\nHighest Aggregate for {task}/{shot}-shot/exp-{exp_num} = {best_score_for_exp_group:.4f}", 'blue'))

        threshold_score = SCORE_INTERVAL * best_score_for_exp_group

        models_to_print = []

        for model_dir, model_score in scores:
            max_char_length = max(len(m_dir.split('shot/')[1]) for m_dir, _ in scores)
            m_name = model_dir.split('shot/')[1]
            m_name = f"{m_name:{max_char_length + 2}}"

            print_str = f"| {m_name} Aggregate: {model_score:.2f}  Threshold: {threshold_score:.2f}"

            if model_score < threshold_score:
                models_to_print.append(colored(print_str, 'red'))
                bad_performing_models.append(model_dir)
            else:
                models_to_print.append(print_str)

        # Sort the list by scores in descending order and print
        for model_info in sorted(models_to_print, key=lambda x: float(x.split("Aggregate: ")[1].split()[0]),
                                 reverse=True):
            print(model_info)

    return bad_performing_models, best_scores_for_each_setting


def remove_model_dir(directory):
    """Removes the specified directory after getting confirmation from the user."""
    directory_name = directory.split('work_dirs/')[1]
    try:
        shutil.rmtree(directory)
        print(f"Directory {directory_name} deleted successfully.")
    except Exception as e:
        print(f"Error while deleting directory {directory_name}. Error: {e}")


def process_task_shot_combination_for_worst_models(args):
    task, shot = args
    bad_performing_models, best_scores_for_each_setting = get_worst_performing_model_dirs(task=task, shot=shot)
    return task, shot, bad_performing_models, best_scores_for_each_setting


# ========================================================================================
work_dir_path = os.path.join("/scratch", "medfm", "medfm-challenge", "work_dirs")
tasks = ["colon", "endo", "chest"]
shots = ["1", "5", "10"]
batch_size = 4
metric_tags = {"auc": "AUC/AUC_multiclass",
               "aucl": "AUC/AUC_multilabe",
               "map": "multi-label/mAP",
               "agg": "Aggregate"}
SCORE_INTERVAL = 0.7  # Assuming you want to keep models that achieved at least 90% of the best score
task_specific_metrics = {
    "colon": "Aggregate",
    "endo": "Aggregate",
    "chest": "Aggregate"
}
best_scores = {}
model_performance = {}
# ========================================================================================


if __name__ == "__main__":
    try:
        SCORE_INTERVAL = float(input("Enter the threshold (e.g., 0.7): "))
    except ValueError:
        print("Invalid threshold. Using default value of 0.7.")
        SCORE_INTERVAL = 0.7

    with Pool() as pool:
        combinations = [(task, shot) for task in tasks for shot in shots]
        results_worst = list(pool.imap_unordered(process_task_shot_combination_for_worst_models, combinations))

    worst_model_dirs = []
    for task, shot, model_list, best_scores_for_setting in results_worst:
        for (task_key, shot_key, exp_num), best_score in best_scores_for_setting.items():
            best_scores[(task_key, shot_key, exp_num)] = best_score

        local_model_performance = {model: extract_metric_from_performance_json(model, task) for model in model_list}
        model_performance.update(local_model_performance)

        for model_name in model_list:
            model_path = os.path.join(work_dir_path, task, f"{shot}-shot", model_name)
            worst_model_dirs.append(model_path)

    print_report(worst_model_dirs, best_scores, model_performance)

    user_input = input(
        "\nDo you want to delete the worst-performing model runs? (y)es / (n)o: ")
    input = user_input.strip().lower()

    if input == 'y':
        for model_dir in worst_model_dirs:
            remove_model_dir(model_dir)
    else:
        exit()
