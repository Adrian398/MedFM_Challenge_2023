import argparse
import os
import re
import shutil
from datetime import datetime

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from termcolor import colored

parser = argparse.ArgumentParser(description='Choose by which metric the best runs should be picked: map / auc / agg)')
parser.add_argument('--metric', type=str, default='map', help='Metric type, default is map')
parser.add_argument('--eval', action='store_true', help='If this flag is set, no files will be created, simply the best runs will be listed. (default false)')
args = parser.parse_args()
metric = args.metric
print(metric)


#work_dir_path = os.path.join("/scratch", "medfm", "medfm-challenge", "dataset_creation","work_dirs")
work_dir_path = os.path.join("dataset_creation", "work_dirs")

metric_tags = {"auc": "AUC/AUC_multiclass",
               "aucl": "AUC/AUC_multilabe",
               "map": "multi-label/mAP",
               "agg": "Aggregate"}

metric = metric_tags[metric]

tasks = ["colon", "endo", "chest"]
shots = ["1", "5", "10"]


# DEBUG
# tasks = ["colon"]
# shots = ["1"]

def get_max_metric_from_event_file(file_path, metric):

    event_acc = EventAccumulator(file_path)
    event_acc.Reload()
    scalar_tags = event_acc.Tags()['scalars']

    # skip, no auc
    if metric_tags["auc"] not in scalar_tags and metric_tags["aucl"] not in scalar_tags:
        return -1

    # skip no map
    if metric_tags["map"] not in scalar_tags:
        return -1

    # determine auc type
    auc_tag = metric_tags["auc"] if metric_tags["auc"] in scalar_tags else metric_tags["aucl"]

    if metric.__contains__("AUC"):
        metric = auc_tag

    if metric == "Aggregate" and metric not in scalar_tags:
        map_values = [item.value for item in event_acc.Scalars(metric_tags["map"])]
        auc_values = [item.value for item in event_acc.Scalars(auc_tag)]
        max_index = map_values.index(max(map_values))
        return float((map_values[max_index] + auc_values[max_index]) / 2)

    # Extract relevant values
    values = event_acc.Scalars(metric)
    return max([item.value for item in values])


def get_ckpt_file_from_run_dir(run_dir):
    for entry in os.listdir(run_dir):
        if entry.__contains__(f"best"):
            return entry
    return None


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


def get_5_best_run_dirs(task, shot, metric):
    setting_directory = os.path.join(work_dir_path, task, f"{shot}-shot")
    # a setting is a combination of task and shot, e.g. 1-shot colon
    try:
        setting_run_dirs = os.listdir(setting_directory)
    except Exception:
        return None, -1

    run_score_list = []

    for run_dir in setting_run_dirs:
        print(f"checking {task}/{shot}-shot/{run_dir}")
        run_dir_path = os.path.join(setting_directory, run_dir)

        # skip if no event file
        event_file = get_event_file_from_run_dir(run_dir_path)
        if event_file is None:
            continue

        # skip if metric not in event file
        score = get_max_metric_from_event_file(event_file, metric)
        if score == -1:
            continue

        run_score_list.append((run_dir, score))

    run_score_list.sort(key=lambda x: x[1])

    return run_score_list[:5]


def extract_exp_number(path):
    match = re.search(r'_exp(\d+)', path)
    return int(match.group(1)) if match else 0

best_settings = {}

for task in tasks:
    best_settings[task] = {}

    for shot in shots:
        best_runs = get_5_best_run_dirs(task, shot, metric)
        assert len(best_runs) == 5
        best_settings[task][shot] = best_runs

print(best_settings)

base_path_source = os.path.join("dataset_creation", "candidate_data")
base_path_target = os.path.join("data_anns", "MedFMC")

for task in tasks:
    keep_files = []
    for shot in shots:
        best_runs = best_settings[task][shot]
        for run_path in best_runs:
            exp_num = extract_exp_number(run_path)
            keep_files.append(f"{task}_{shot}-shot_train_exp{exp_num}.txt")
            keep_files.append(f"{task}_{shot}-shot_val_exp{exp_num}.txt")

    target_task_dir = os.path.join(base_path_target, f"{task}_tuned")
    base_task_dir = os.path.join(base_path_source, task)

    if os.path.isdir(target_task_dir):
        shutil.copytree(base_task_dir, target_task_dir)
    else:
        shutil.copy2(base_task_dir, target_task_dir)


