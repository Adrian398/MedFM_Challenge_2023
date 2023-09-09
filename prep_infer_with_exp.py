import argparse
import itertools
import os
import re
import shutil
import subprocess
import time
from datetime import datetime

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from termcolor import colored


def extract_exp_number(string):
    match = re.search(r'exp(\d+)', string)
    return int(match.group(1)) if match else 0


def run_commands_on_cluster(commands, gpu=None, delay_seconds=1):
    """
    Runs the generated commands on the cluster.
    If no GPU is specified, the commands are queued on the cluster in the following scheme:
    gpuc -> gpua / gpub -> gpua / gpub -> gpuc -> ...
    """

    if gpu == 'c':
        gpus = ['rtx4090']
    elif gpu == 'ab':
        gpus = ['rtx3090']
    elif gpu == '8a':
        gpus = ['rtx2080ti']
    elif gpu == 'all':
        gpus = ['rtx4090', 'rtx3090', 'rtx4090', 'rtx3090']
    else:
        raise ValueError(f'Invalid gpu type {gpu}.')

    gpu_cycle = itertools.cycle(gpus)

    # Ensure the log directory exists
    log_dir = os.path.join(submission_dir, "logs")
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    for command in commands:
        gpu = next(gpu_cycle)

        command_splitted = command.split(" ")[2].split("/")
        task, shot, exp = command_splitted[5], command_splitted[6], extract_exp_number(command_splitted[7])
        log_file_name = f"{task}_{shot}_exp{exp}_slurm-%j"

        slurm_cmd = f'sbatch -p ls6 --gres=gpu:{gpu}:1 --wrap="{command}" -o "{log_dir}/{log_file_name}.out"'
        subprocess.run(slurm_cmd, shell=True)
        time.sleep(delay_seconds)


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


def filter_directories_by_exp(directories, exp_number):
    exp_str = f"exp{exp_number}"
    filtered_dirs = [dir for dir in directories if exp_str in dir]
    return filtered_dirs


def get_N_best_exp_run_dirs(task, shot, exp, metric):
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

        run_score_list.append((run_dir_path, score))

    run_score_list.sort(key=lambda x: x[1], reverse=True)

    return run_score_list[:min(N_best, len(run_score_list))]


def run_on_bash(commands):
    if len(commands) == 0:
        print(colored('No commands found to run on the bash.', 'red'))
        exit()
    bash_script = "#!/bin/bash\n"
    for c in commands:
        bash_script += c
    bash_script += f"echo {submission_dir}"
    print(f"Saved respective configs to {configs_dir}")
    print("Created infer.sh")
    print(f"Run ./infer.sh to create prediction files in {predictions_dir}")
    with open("infer.sh", "w") as file:
        file.write(bash_script)
    os.chmod("infer.sh", 0o755)


parser = argparse.ArgumentParser(description='Choose by which metric the best runs should be picked: map / auc / agg)')
parser.add_argument('--metric', type=str, default='agg', help='Metric type, default is agg')
parser.add_argument('--exclude', type=str, default='', help='Comma separated model names to exclude')
parser.add_argument('--n_best', type=int, default=1, help='Returns the N best models per setting')
parser.add_argument('--bs', type=int, default=4, help='The batch size for inference')
parser.add_argument("--gpu", type=str, default='c', help="GPU type: 'c'=rtx4090, '8a'=rtx2070ti or 'ab'=rtx3090")
args = parser.parse_args()

metric = args.metric
gpu_type = args.gpu
batch_size = args.bs

exclude_models = []
if len(args.exclude) > 0:
    exclude_models = args.exclude.split(",")

work_dir_path = os.path.join("/scratch", "medfm", "medfm-challenge", "work_dirs")
metric_tags = {"auc": "AUC/AUC_multiclass",
               "aucl": "AUC/AUC_multilabe",
               "map": "multi-label/mAP",
               "agg": "Aggregate"}

metric = metric_tags[metric]

N_best = args.n_best
tasks = ["colon", "endo", "chest"]
shots = ["1", "5", "10"]
exps = [1, 2, 3, 4, 5]
model_soup = False

report = [
    "\n---------------------------------------------------------------------------------------------------------------",
    f"| Best runs for each setting, ranked by {metric}:",
    "---------------------------------------------------------------------------------------------------------------"]

best_settings = {}

for task in tasks:
    best_settings[task] = {}

    for shot in shots:
        best_settings[task][shot] = {}

        for exp in exps:
            best_runs = get_N_best_exp_run_dirs(task=task, shot=shot, exp=exp, metric=metric)
            best_settings[task][shot][exp] = best_runs
            if N_best > 1:
                report.append(
                    "---------------------------------------------------------------------------------------------------------------")

            for run in best_settings[task][shot][exp]:
                run_path_to_print = run[0].split(os.sep)[-1]
                report.append(f"| {task}/{shot}-shot/exp{exp}\t{metric}: {run[1]:.2f}\t{run_path_to_print}")
            if len(best_runs) < N_best:
                for i in range(N_best - len(best_runs)):
                    report.append(f"| {task}/{shot}-shot/exp{exp}\tNo run found")

data_complete = True
report.append(
    "---------------------------------------------------------------------------------------------------------------")
for line in report:
    if line.__contains__("No run found"):
        data_complete = False
        print(colored(line, 'red'))
    else:
        print(line)

if not data_complete:
    print(colored(f"Could not find {N_best} runs for every setting, aborting...", 'red'))
    exit()

if N_best > 1:
    model_soup = True
    print(f"The {colored(N_best, 'red')} best experiments for each setting have been selected.")
    user_input = input(f"\nDo you want to continue with {colored('Model Soup', 'blue')}? (yes/no): ")
    if user_input.strip().lower() == 'no':
        exit()

if model_soup:
    print(colored("Model Soup not implemented yet!", 'red'))
    exit()

# Prompt the user
user_input = input(f"\nDo you want to generate the inference commands? (yes/no): ")

if user_input.strip().lower() == 'no':
    exit()

# create dir for submission and config
date_pattern = datetime.now().strftime("%d-%m_%H-%M-%S")

submission_dir = os.path.join("submissions", "evaluation", date_pattern)
configs_dir = os.path.join(submission_dir, "configs")
predictions_dir = os.path.join(submission_dir, "predictions")
scratch_repo_path = os.path.join("/scratch", "medfm", "medfm-challenge")
os.makedirs(submission_dir)
os.makedirs(configs_dir)
os.makedirs(predictions_dir)

with open(os.path.join(submission_dir, "report.txt"), "w") as file:
    file.write("\n".join(report))
best_runs = []
for task in tasks:
    for shot in shots:
        for exp in exps:
            for run in best_settings[task][shot][exp]:
                best_runs.append(run[0])

commands = []
for run_path in best_runs:
    split_path = run_path.split(os.sep)
    task, shot, exp = split_path[5], split_path[6], extract_exp_number(split_path[7])

    config_filename = [file for file in os.listdir(run_path) if file.endswith(".py")][0]
    checkpoint_filename = \
        [file for file in os.listdir(run_path) if file.endswith(".pth") and file.__contains__("best")][0]

    config_path = os.path.join(run_path, config_filename)
    checkpoint_path = os.path.join(run_path, checkpoint_filename)

    images_path = os.path.join(scratch_repo_path, "data", "MedFMC_test", task, "images")

    predictions_dest_dir = os.path.join(predictions_dir, f"exp{exp}")
    if not os.path.exists(predictions_dest_dir):
        os.makedirs(predictions_dest_dir)

    csv_name = f"{task}_{shot}_submission.csv"
    out_path = os.path.join(predictions_dir, f"exp{exp}", csv_name)

    config_dest_dir = os.path.join(configs_dir, f"exp{exp}")
    if not os.path.exists(config_dest_dir):
        os.makedirs(config_dest_dir)

    # copy config into submission directory
    shutil.copy(config_path, os.path.join(config_dest_dir, config_filename))
    command = f"python tools/infer.py {config_path} {checkpoint_path} {images_path} --batch-size {batch_size} --out {out_path}\n"
    commands.append(command)

print(f"Saved respective configs to {configs_dir}\n")
print("Generated Infer Commands:")
for command in commands:
    print(command)

user_input_medium = input(f"Do you want to run the commands on the {colored('cluster', 'red')} or {colored('bash', 'blue')}? (cluster/bash): ")

if user_input_medium.strip().lower() == 'cluster':
    medium_txt = colored('cluster', 'red')
    medium = 'cluster'
elif user_input_medium.strip().lower() == 'bash':
    medium_txt = colored('bash', 'blue')
    medium = 'bash'
else:
    print(colored('No valid medium selected...', 'red'))
    exit()

user_input_start = input(f"Start {len(commands)} commands on the {medium_txt}? (yes/no): ")

if user_input.strip().lower() == 'yes':
    if medium == 'cluster':
        run_commands_on_cluster(commands, gpu=gpu_type)
    elif medium == 'bash':
        run_on_bash(commands)
    else:
        print(colored('No valid medium or answer...', 'red'))
        exit()
