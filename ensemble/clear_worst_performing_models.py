import json
import os
import re
import sys
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
            best_score_for_task_shot = best_scores.get((task, shot))
            metric_for_task = task_specific_metrics.get(task, "Aggregate")
            score_for_model = model_performance.get(entry)
            print(
                f"| {relative_path:<{max_path_length}} | {metric_for_task:9} | {score_for_model:6.2f} | {best_score_for_task_shot:10.2f} |")
        print("-" * (max_path_length + padding))
        print(f"| Found {len(invalid_model_dirs)} bad model runs.")
        print("-" * (max_path_length + padding))


def extract_exp_number(string):
    match = EXP_PATTERN.search(string)
    return int(match.group(1)) if match else 0


def sort_key(entry):
    # Extract task, shot from the entry
    parts = entry.split('/')
    task = parts[5]
    shot = int(parts[6].split('-')[0])
    score = model_performance.get(entry, float('-inf'))  # If no score is found, default to negative infinity
    return task, shot, score


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


def extract_metric_from_performance_json(model_dir, task):
    for dirpath, _, filenames in os.walk(model_dir):
        for filename in filenames:
            if filename == "performance.json":
                filepath = os.path.join(dirpath, filename)
                try:
                    with open(filepath, 'r') as file:
                        data = json.load(file)

                    metric_tag = task_specific_metrics.get(task, "Aggregate")
                    if metric_tag not in data:
                        print(f"Metric '{metric_tag}' not found in {filepath}")
                        return None
                    return data[metric_tag]
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
    model_dirs = []
    model_performance = {}
    setting_directory = os.path.join(work_dir_path, task, f"{shot}-shot")

    try:
        setting_model_dirs = os.listdir(setting_directory)
    except Exception:
        return None, None

    for model_dir in setting_model_dirs:
        my_print(f"Checking {task}/{shot}-shot/{model_dir}")
        abs_model_dir = os.path.join(setting_directory, model_dir)

        # Only consider model directories with a performance.json file
        if find_and_validate_json_files(abs_model_dir, task):
            aggregate_metric = extract_metric_from_performance_json(abs_model_dir, task)
            if aggregate_metric is not None:
                model_performance[abs_model_dir] = aggregate_metric

    # Group by exp number
    exp_grouped_scores = {}
    for model_dir, score in model_performance.items():
        exp_num = extract_exp_number(model_dir)
        if exp_num not in exp_grouped_scores:
            exp_grouped_scores[exp_num] = []
        exp_grouped_scores[exp_num].append((model_dir, score))

    print(exp_grouped_scores.items()[0])
    exit()

    for _, scores in exp_grouped_scores.items():
        best_score_for_exp = max(score for _, score in scores)
        threshold_score = get_score_interval_for_exp(best_score_for_exp)

        # Consider for deletion the models with scores below the threshold for each exp
        for model_dir, score in scores:
            if score < threshold_score:
                model_dirs.append(model_dir)

    best_score = max(model_performance.values()) if model_performance else None

    return model_dirs, best_score


def get_score_interval_for_exp(best_score_for_exp):
    return SCORE_INTERVAL * best_score_for_exp


def process_task_shot_combination_for_worst_models(args):
    task, shot = args
    model_dirs, best_score = get_worst_performing_model_dirs(task=task, shot=shot)
    return task, shot, model_dirs, best_score


# ========================================================================================
work_dir_path = os.path.join("/scratch", "medfm", "medfm-challenge", "work_dirs")
tasks = ["colon", "endo", "chest"]
shots = ["1", "5", "10"]
N_inferences_per_task = 10
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
    for task, shot, model_list, best_score in results_worst:
        best_scores[(task, shot)] = best_score
        local_model_performance = {model: extract_metric_from_performance_json(model, task) for model in model_list}
        model_performance.update(local_model_performance)
        for model_name in model_list:
            model_path = os.path.join(work_dir_path, task, f"{shot}-shot", model_name)
            worst_model_dirs.append(model_path)

    print_report(worst_model_dirs, best_scores, model_performance)

    # user_input = input(f"\nDo you want to delete the worst-performing model runs? (yes/no): ")
    # if user_input.strip().lower() == 'yes':
    #    for model_dir in worst_model_dirs:
    #        remove_model_dir(model_dir)