import argparse
import os

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from termcolor import colored

parser = argparse.ArgumentParser(description='Choose by which metric the best runs should be picked: map / auc / agg)')
parser.add_argument('--metric', type=str, default='agg', help='Metric type, default is agg')
parser.add_argument('--exclude', type=str, default='', help='Comma separated model names to exclude')
parser.add_argument('--eval', action='store_true',
                    help='If this flag is set, no files will be created, simply the best runs will be listed. (default false)')
args = parser.parse_args()
metric = args.metric

exclude_models = []
if len(args.exclude) > 0:
    exclude_models = args.exclude.split(",")

work_dir_path = os.path.join("/scratch", "medfm", "medfm-challenge", "work_dirs")
metric_tags = {"auc": "AUC/AUC_multiclass",
               "aucl": "AUC/AUC_multilabe",
               "map": "multi-label/mAP",
               "agg": "Aggregate"}

metric = metric_tags[metric]

tasks = ["colon", "endo", "chest"]
shots = ["1", "5", "10"]
exps = [1, 2, 3, 4, 5]


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


def exclude_model(name):
    for exclude in exclude_models:
        if name.__contains__(exclude):
            return True
    return False


def filter_directories_by_exp(directories, exp_number):
    exp_str = f"exp{exp_number}"
    filtered_dirs = [dir for dir in directories if exp_str in dir]
    return filtered_dirs


def get_5_best_exp_run_dirs(task, shot, exp, metric):
    setting_directory = os.path.join(work_dir_path, task, f"{shot}-shot")
    # a setting is a combination of task and shot, e.g. colon/1-shot
    # within such a setting we then also have different experiments -> colon/1-shot/...exp1...
    try:
        setting_run_dirs = os.listdir(setting_directory)
    except Exception:
        return None, -1

    setting_run_dirs = filter_directories_by_exp(setting_run_dirs, exp)

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

    run_score_list.sort(key=lambda x: x[1], reverse=True)

    return run_score_list[:min(5, len(run_score_list))]


report = []
best_settings = {}

for task in tasks:
    best_settings[task] = {}

    for shot in shots:
        best_settings[task][shot] = {}

        for exp in exps:
            best_runs = get_5_best_exp_run_dirs(task=task, shot=shot, exp=exp, metric=metric)
            best_settings[task][shot][exp] = best_runs
            report.append("---------------------------------------------------------------------------------------------------------------")

            if len(best_runs) == 0:
                report.append(f"| {task}/{shot}-shot/exp{exp}\tNo run found")
            else:
                for run in best_settings[task][shot][exp]:
                    report.append(f"| {task}/{shot}-shot/exp{exp}\t{metric}: {run[1]:.2f}\t{run[0]}")

print("")
print("---------------------------------------------------------------------------------------------------------------")
print(f"| Best runs for each setting, ranked by {metric}:")
print("---------------------------------------------------------------------------------------------------------------")
for line in report:
    if line.__contains__("No run found"):
        print(colored(line, 'red'))
    else:
        print(line)
print("---------------------------------------------------------------------------------------------------------------")

# if args.eval:
#     exit()

# # create dir for submission and config
# date_pattern = datetime.now().strftime("%d-%m_%H-%M-%S")
# submission_dir = os.path.join("submissions", date_pattern)
# configs_dir = os.path.join(submission_dir, "configs")
# predictions_dir = os.path.join(submission_dir, "predictions")
#
# os.makedirs(submission_dir)
# with open(os.path.join(submission_dir, "report.txt"), "w") as file:
#     txt_report = ""
#     for line in report:
#         txt_report += line + "\n"
#     file.write(txt_report)
#
# os.makedirs(configs_dir)
# os.makedirs(predictions_dir)
#
# bash_script = "#!/bin/bash\n"
# for given_run_path in best_runs:
#     scratch_repo_path = os.path.join("/scratch", "medfm", "medfm-challenge")
#
#     task = given_run_path.split(os.sep)[0]
#     if task.__contains__("-"):
#         task = task.split("-")[0]
#     shot = given_run_path.split(os.sep)[1]
#
#     given_run_path = os.path.join("work_dirs", given_run_path)
#     path_components = given_run_path.split(os.sep)
#     run_dir = os.path.join(*path_components[:4])
#     run_dir = os.path.join(scratch_repo_path, run_dir)
#
#     config_filename = [file for file in os.listdir(run_dir) if file.endswith(".py")][0]
#     checkpoint_filename = [file for file in os.listdir(run_dir) if file.endswith(".pth") and file.__contains__("best")][0]
#
#     config_path = os.path.join(run_dir, config_filename)
#     checkpoint_path = os.path.join(run_dir, checkpoint_filename)
#     images_path = os.path.join(scratch_repo_path, "data", "MedFMC_val", task, "images")
#     csv_name = f"{task}_{shot}_submission.csv"
#     out_path = os.path.join(predictions_dir, csv_name)
#
#     # copy config into submission directory
#     shutil.copy(config_path, configs_dir)
#     command = f"python tools/infer.py {config_path} {checkpoint_path} {images_path} --out {out_path}\n"
#     bash_script += command
#
# print(f"Saved respective configs to {configs_dir}")
# print("Created infer.sh")
# print(f"Run ./infer.sh to create prediction files in {predictions_dir}")
# with open("infer.sh", "w") as file:
#     file.write(bash_script)
# os.chmod("infer.sh", 0o755)
