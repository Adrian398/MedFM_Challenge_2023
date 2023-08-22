import os

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

metric = "agg"  # map, agg

work_dir_path = "work_dirs"
work_dir_path = os.path.join("/scratch", "medfm", "medfm-challenge", "work_dirs")
metric_tags = {"auc": "AUC/AUC_multiclass",
               "map": "multi-label/mAP",
               "agg": "Aggregate"}

metric = metric_tags[metric]

tasks = ["colon", "endo", "chest"]
shots = ["1", "5", "10"]

# DEBUG
tasks = ["colon"]
shots = ["1"]


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
        if entry.__contains__(f"best_{metric}"):
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
    setting_run_dirs = os.listdir(setting_directory)

    best_score = 0
    best_run = None

    for run_dir in setting_run_dirs:
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


print("Best runs for each setting:")
for task in tasks:
    for shot in shots:
        best_run, best_score = get_best_run_dir(task, shot, metric)
        if best_run is None:
            print(f"{shot}-shot_{task}: No run found")
        else:
            print(f"{shot}-shot_{task} - {metric}: {best_score} - {best_run}")