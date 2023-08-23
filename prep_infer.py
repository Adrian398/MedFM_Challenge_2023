import argparse
import os
import shutil
from datetime import datetime

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from termcolor import colored

parser = argparse.ArgumentParser(description='Choose by which metric the best runs should be picked: map / auc / agg)')
parser.add_argument('--metric', type=str, default='map', help='Metric type, default is map')
args = parser.parse_args()
metric = args.metric
print(metric)
print(type(metric))

work_dir_path = os.path.join("/scratch", "medfm", "medfm-challenge", "work_dirs")
metric_tags = {"auc": "AUC/AUC_multiclass",
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
    if metric not in scalar_tags:
        return -1

    # Extract relevant values
    values = event_acc.Scalars(metric)
    return max([item.value for item in values])


def get_ckpt_file_from_run_dir(run_dir):
    for entry in os.listdir(run_dir):
        #if entry.__contains__(f"best_{metric.replace('/', '_')}"):
        #    return entry
        # TODO: Do not make checkpoint file dependent on metric, since even if the checkpoint is named
        # TODO: aggregate, it still can contain AUC and mAP
        if entry.__contains__(f"best_"):
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


def get_best_run_dir(task, shot, metric):
    setting_directory = os.path.join(work_dir_path, task, f"{shot}-shot")
    # a setting is a combination of task and shot, e.g. 1-shot colon
    try:
        setting_run_dirs = os.listdir(setting_directory)
    except Exception:
        return None, -1

    best_score = 0
    best_run = None

    for run_dir in setting_run_dirs:
        print(f"checking {task}/{shot}-shot/{run_dir}")
        run_dir_path = os.path.join(setting_directory, run_dir)

        # skip if no checkpoint
        ckpt_file = get_ckpt_file_from_run_dir(run_dir_path)
        if ckpt_file is None:
            continue

        # skip if no event file
        event_file = get_event_file_from_run_dir(run_dir_path)
        if event_file is None:
            continue

        # skip if metric not in event file
        score = get_max_metric_from_event_file(event_file, metric)
        if score == -1:
            continue

        if score > best_score:
            best_score = score
            best_run = run_dir
    return best_run, best_score


report = []
best_runs = []

for task in tasks:
    for shot in shots:
        best_run, best_score = get_best_run_dir(task, shot, metric)
        if best_run is None:
            report.append(f"| {shot}-shot_{task}\t No run found")
        else:
            report.append(f"| {shot}-shot_{task}\t{metric}: {best_score}\t{best_run}")
            best_runs.append(os.path.join(task, f"{shot}-shot", best_run))

print("")
print("---------------------------------------------------------------------------------------------------------------")
print("|\t\t\tBest runs for each setting:")
print("---------------------------------------------------------------------------------------------------------------")
for line in report:
    if line.__contains__("No run found"):
        print(colored(line, 'red'))
    else:
        print(line)
print("---------------------------------------------------------------------------------------------------------------")

# create dir for submission and config
date_pattern = datetime.now().strftime("%d-%m_%H-%M-%S")
submission_dir = os.path.join("submissions", date_pattern)
configs_dir = os.path.join(submission_dir, "configs")
predictions_dir = os.path.join(submission_dir, "predictions")

os.makedirs(submission_dir)
os.makedirs(configs_dir)
os.makedirs(predictions_dir)

bash_script = "#!/bin/bash\n"
for given_run_path in best_runs:
    scratch_repo_path = os.path.join("/scratch", "medfm", "medfm-challenge")

    task = given_run_path.split(os.sep)[0]
    if task.__contains__("-"):
        task = task.split("-")[0]
    shot = given_run_path.split(os.sep)[1]

    given_run_path = os.path.join("work_dirs", given_run_path)
    path_components = given_run_path.split(os.sep)
    run_dir = os.path.join(*path_components[:4])
    run_dir = os.path.join(scratch_repo_path, run_dir)

    config_filename = [file for file in os.listdir(run_dir) if file.endswith(".py")][0]
    checkpoint_filename = [file for file in os.listdir(run_dir) if file.endswith(".pth") and file.__contains__("best")][0]

    config_path = os.path.join(run_dir, config_filename)
    checkpoint_path = os.path.join(run_dir, checkpoint_filename)
    images_path = os.path.join(scratch_repo_path, "data", "MedFMC_val", task, "images")
    csv_name = f"{task}_{shot}_submission.csv"
    out_path = os.path.join(predictions_dir, csv_name)

    # copy config into submission directory
    shutil.copy(config_path, configs_dir)
    command = f"python tools/infer.py {config_path} {checkpoint_path} {images_path} --out {out_path}\n"
    bash_script += command

print(f"Saved respective configs to {configs_dir}")
print("Created infer.sh")
print(f"Run ./infer.sh to create prediction files in {predictions_dir}")
with open("infer.sh", "w") as file:
    file.write(bash_script)
os.chmod("infer.sh", 0o755)
