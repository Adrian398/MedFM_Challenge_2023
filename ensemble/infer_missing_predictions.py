"""
This script does the following steps:
- walk through the whole work_dirs directory
- detect and print all model folders that have not yet been inferred (i.e. the prediction csv file is missing)
- prompt the user to start the infer process
- generate the infer commands
- batch all commands on the corresponding gpus, whereas each gpu is dedicated for a specific task

Prediction File Naming Scheme:  TASK_N-shot_submission.csv
Example:                        chest_10-shot_submission.csv
"""
import os
import re

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def extract_exp_number(string):
    match = re.search(r'exp(\d+)', string)
    return int(match.group(1)) if match else 0


def contains_csv_file(task, shot, model_dir):
    expected_filename = f"{task}_{shot}-shot_submission.csv"

    try:
        return expected_filename in os.listdir(model_dir)
    except FileNotFoundError:
        pass
    except PermissionError as permission_error:
        print(f"Permission Error encountered: {permission_error}")
        return False

    return False


def is_metric_in_event_file(file_path, metric):
    event_acc = EventAccumulator(file_path)
    event_acc.Reload()
    scalar_tags = event_acc.Tags()['scalars']

    # skip no map
    if metric in scalar_tags:
        return True
    else:
        return False


def get_event_file_from_model_dir(model_dir):
    try:
        for entry in os.listdir(model_dir):
            full_path = os.path.join(model_dir, entry)
            if os.path.isdir(full_path):
                full_path = os.path.join(full_path, "vis_data")
                event_file = os.listdir(full_path)[0]
                return os.path.join(full_path, event_file)
    except Exception:
        return None


def get_model_dirs_without_prediction(task, shot):
    model_dirs = []
    setting_directory = os.path.join(work_dir_path, task, f"{shot}-shot")

    try:
        setting_model_dirs = os.listdir(setting_directory)
    except Exception:
        return None

    for model_dir in setting_model_dirs:
        print(setting_model_dirs)
        print(f"Checking {task}/{shot}-shot/{model_dir}")
        model_dir = os.path.join(setting_directory, model_dir)

        # Skip/Delete if no event file
        event_file = get_event_file_from_model_dir(model_dir)
        if event_file is None:
            # TODO Delete if no event file!
            #print("No event file found, skipping..")
            continue

        # Skip if metric not in event file
        if not is_metric_in_event_file(event_file, metric_tags['map']):
            #print("Metric map not present, skipping..")
            continue

        # Skip if prediction csv file is present
        if contains_csv_file(task, shot, model_dir):
            continue

        model_dirs.append(model_dir)
    return model_dirs


work_dir_path = os.path.join("/scratch", "medfm", "medfm-challenge", "work_dirs")
#work_dir_path = os.path.join("work_dirs")
tasks = ["colon", "endo", "chest"]
shots = ["1", "5", "10"]
#exps = [1, 2, 3, 4, 5]

metric_tags = {"auc": "AUC/AUC_multiclass",
               "aucl": "AUC/AUC_multilabe",
               "map": "multi-label/mAP",
               "agg": "Aggregate"}

report = [
    "\n---------------------------------------------------------------------------------------------------------------",
    f"| Valid Models without an existing prediction CSV file:",
    "---------------------------------------------------------------------------------------------------------------"]

model_dirs = []

for task in tasks:
    for shot in shots:
        model_list = get_model_dirs_without_prediction(task=task, shot=shot)

        if model_list is not None:
            model_dirs.append(model_list)

            for model in model_list:
                model_path_to_print = model.split(os.sep)[-1]
                exp = extract_exp_number(model_path_to_print)
                report.append(f"| {task}/{shot}-shot/exp{exp}\t{model_path_to_print}")

report.append(
    "---------------------------------------------------------------------------------------------------------------")
for line in report:
    print(line)
