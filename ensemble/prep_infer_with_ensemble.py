import argparse
import itertools
import os
import re
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


def get_map_and_class_scores(file_path):
    event_acc = EventAccumulator(file_path)
    event_acc.Reload()
    scalar_tags = event_acc.Tags()['scalars']

    # skip no map
    if metric_tags["map"] not in scalar_tags:
        return -1

    # Extract relevant values
    map_scalar = event_acc.Scalars(metric_tags["map"])
    map_values = [item.value for item in map_scalar]
    max_map = max(map_values)
    max_map_index = map_values.index(max_map)

    # Check if MAP values for classes are available
    use_map_for_class_scores = False
    for tag in scalar_tags:
        if tag.startswith("MAP_class"):
            use_map_for_class_scores = True
            break

    if use_map_for_class_scores:
        class_tags = [tag for tag in scalar_tags if tag.startswith('MAP_class')]
    else:
        class_tags = [tag for tag in scalar_tags if tag.startswith('AUC/AUC_class')]
    class_tags.sort()
    class_values = []
    for class_tag in class_tags:
        class_values.append(event_acc.Scalars(class_tag)[max_map_index].value)
    print(f"Class values: {class_values}")
    print(f"MAP: {max_map}")

    return max_map, class_values


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


def get_N_best_exp_run_dirs(task, shot, exp):
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
            print("Skip, no event file")
            continue

        score, class_values = get_map_and_class_scores(event_file)

        if score == -1:
            print("Skip, no map")
            continue

        run_score_list.append((run_dir_path, score, class_values))

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
    print("Created infer.sh")
    print(f"Run ./infer.sh to create prediction files in {submission_dir}")
    with open("infer.sh", "w") as file:
        file.write(bash_script)
    os.chmod("infer.sh", 0o755)


parser = argparse.ArgumentParser(description='Prepares the inference for the ensemble model.')
parser.add_argument('--exclude', type=str, default='', help='Comma separated model names to exclude')
parser.add_argument('--n_best', type=int, default=1, help='Returns the N best models per setting')
parser.add_argument('--bs', type=int, default=4, help='The batch size for inference')
parser.add_argument("--gpu", type=str, default='c', help="GPU type: 'c'=rtx4090, '8a'=rtx2070ti or 'ab'=rtx3090")
args = parser.parse_args()

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

N_best = args.n_best
tasks = ["endo", "colon", "chest"]
shots = ["1", "5", "10"]
experiments = [1, 2, 3, 4, 5]
model_soup = False

report = [
    "\n---------------------------------------------------------------------------------------------------------------",
    f"| Best runs for each setting, ranked by MAP:",
    "---------------------------------------------------------------------------------------------------------------"]

best_settings = {}

for task in tasks:
    best_settings[task] = {}

    for shot in shots:
        best_settings[task][shot] = {}

        for exp in experiments:
            best_runs = get_N_best_exp_run_dirs(task=task, shot=shot, exp=exp)
            best_settings[task][shot][exp] = best_runs
            if N_best > 1:
                report.append(
                    "---------------------------------------------------------------------------------------------------------------")

            for run in best_settings[task][shot][exp]:
                run_path_to_print = run[0].split(os.sep)[-1]
                report.append(f"| {task}/{shot}-shot/exp{exp}\tMAP: {run[1]:.2f}\t{run_path_to_print}")
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

# Prompt the user
user_input = input(f"\nDo you want to generate the inference commands? This step will already set up a sub"
                   f"mission directory (yes/no): ")

if user_input.strip().lower() == 'no':
    exit()

'''
----------------------------------------Set up directory structure below
'''

# create dir for submission and config
date_pattern = datetime.now().strftime("%d-%m_%H-%M-%S")
submission_dir = os.path.join("submissions", "evaluation", date_pattern)
os.makedirs(submission_dir)

for task in tasks:
    for shot in shots:
        for exp in experiments:
            os.makedirs(os.path.join(submission_dir, "tmp", task, f"{shot}-shot", f"exp{exp}"), exist_ok=True)

scratch_repo_path = os.path.join("/scratch", "medfm", "medfm-challenge")
os.makedirs(submission_dir)

with open(os.path.join(submission_dir, "report.txt"), "w") as file:
    file.write("\n".join(report))

best_runs = []
for task in tasks:
    for shot in shots:
        for exp in experiments:
            for run_info in best_settings[task][shot][exp]:
                best_runs.append(run_info)

commands = []
for run_info in best_runs:
    run_path = run_info[0]
    split_path = run_path.split(os.sep)
    run_name = split_path[7]
    task, shot, exp = split_path[5], split_path[6], extract_exp_number(run_name)

    run_out_dir = os.path.join(submission_dir, "tmp", task, f"{shot}-shot", f"exp{exp}", run_name)
    os.makedirs(run_out_dir, exist_ok=True)

    config_filename = [file for file in os.listdir(run_path) if file.endswith(".py")][0]
    checkpoint_filename = \
        [file for file in os.listdir(run_path) if file.endswith(".pth") and file.__contains__("best")][0]

    config_path = os.path.join(run_path, config_filename)
    checkpoint_path = os.path.join(run_path, checkpoint_filename)

    images_path = os.path.join(scratch_repo_path, "data", "MedFMC_test", task, "images")

    csv_name = f"{task}_{shot}_exp{exp}_submission.csv"
    out_path_csv = os.path.join(run_out_dir, csv_name)

    # save auc class values to run directory
    auc_class_values = run_info[2]
    with open(os.path.join(run_out_dir, 'auc_class_values.txt'), 'w') as f:
        f.write(','.join([str(value) for value in auc_class_values]))

    command = f"python tools/infer.py {config_path} {checkpoint_path} {images_path} --batch-size {batch_size} --out {out_path_csv}\n"
    commands.append(command)

print("Generated Infer Commands:")
for command in commands:
    print(command)

user_input_medium = input(
    f"Do you want to run the commands on the {colored('cluster', 'red')} or {colored('bash', 'blue')}? (cluster/bash): ")

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
