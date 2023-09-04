import os
import re
import sys
from multiprocessing import Pool

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from termcolor import colored


EXP_PATTERN = re.compile(r'exp(\d+)')


def print_report(invalid_model_dirs):
    if len(invalid_model_dirs) == 0:
        print(colored(f"\nAll models have a checkpoint file!\n", 'green'))
        exit()
    else:
        sorted_report_entries = sorted([model_dir for model_dir in invalid_model_dirs], key=sort_key)
        print("\n---------------------------------------------------------------------------------------------------------------")
        print("| Invalid Models with missing checkpoint or event file:")
        print("---------------------------------------------------------------------------------------------------------------")
        for entry in sorted_report_entries:
            print(f"| {entry}")
        print("---------------------------------------------------------------------------------------------------------------")
        print(f"| Found {len(invalid_model_dirs)} invalid model runs.")
        print("---------------------------------------------------------------------------------------------------------------")


def extract_exp_number(string):
    match = EXP_PATTERN.search(string)
    return int(match.group(1)) if match else 0


def sort_key(entry):
    # Extract task, shot, and experiment number from the entry
    parts = entry.split('/')
    task = parts[5]
    shot = int(parts[6].split('-')[0])
    exp_number = extract_exp_number(parts[-1])
    return task, shot, exp_number


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


def extract_aggregate_metric_from_tensorboard(model_dir):
    event_file = get_event_file_from_run_dir(model_dir)
    if not event_file:
        return None
    try:
        event_acc = EventAccumulator(event_file)
        event_acc.Reload()
        aggregate_metric_values = event_acc.Scalars(metric_tags["agg"])
        if aggregate_metric_values:
            # Assuming the last value is the final aggregate metric value
            return aggregate_metric_values[-1].value
        return None
    except Exception as e:
        print(f"Error reading tensorboard event file for {model_dir}: {e}")
        return None


def get_worst_performing_model_dirs(task, shot):
    model_dirs = []
    model_performance = {}
    setting_directory = os.path.join(work_dir_path, task, f"{shot}-shot")

    try:
        setting_model_dirs = os.listdir(setting_directory)
    except Exception:
        return None

    for model_dir in setting_model_dirs:
        my_print(f"Checking {task}/{shot}-shot/{model_dir}")
        abs_model_dir = os.path.join(setting_directory, model_dir)

        aggregate_metric = extract_aggregate_metric_from_tensorboard(abs_model_dir)
        if aggregate_metric is not None:
            model_performance[model_dir] = aggregate_metric

    threshold_score = get_score_interval(model_performance)

    # Consider for deletion the models with scores below the threshold
    for model_dir, score in model_performance.items():
        if score < threshold_score:
            model_dirs.append(model_dir)

    return model_dirs


def get_score_interval(model_performance):
    best_score = max(model_performance.values())
    threshold_score = SCORE_INTERVAL * best_score
    return threshold_score


def process_task_shot_combination_for_worst_models(args):
    task, shot = args
    return task, shot, get_worst_performing_model_dirs(task=task, shot=shot)


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
SCORE_INTERVAL = 0.9  # Assuming you want to keep models that achieved at least 90% of the best score
# ========================================================================================


if __name__ == "__main__":
    with Pool() as pool:
        combinations = [(task, shot) for task in tasks for shot in shots]
        results_worst = list(pool.imap_unordered(process_task_shot_combination_for_worst_models, combinations))

    worst_model_dirs = []
    for task, shot, model_list in results_worst:
        for model_name in model_list:
            model_path = os.path.join(work_dir_path, task, f"{shot}-shot", model_name)
            worst_model_dirs.append(model_path)

    print_report(worst_model_dirs)

    # user_input = input(f"\nDo you want to delete the worst-performing model runs? (yes/no): ")
    # if user_input.strip().lower() == 'yes':
    #    for model_dir in worst_model_dirs:
    #        remove_model_dir(model_dir)