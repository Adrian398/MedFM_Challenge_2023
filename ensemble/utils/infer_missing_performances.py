import argparse
import itertools
import json
import multiprocessing
import os
import subprocess
import sys
from collections import Counter
from functools import lru_cache
from multiprocessing import Pool

from termcolor import colored

from ensemble.utils.constants import *


def my_print(message):
    sys.stdout.write(str(message) + '\n')
    sys.stdout.flush()


def determine_gpu(gpu_type):
    """
    Determines the GPUs to be used based on the provided type.

    Args:
        gpu_type (str): Type of GPU. Can be 'c', 'ab', '8a', or 'all'.

    Returns:
        list: List of GPUs to be used.
    """
    gpu_mappings = {
        'c': ['rtx4090'],
        'ab': ['rtx3090'],
        'a': ['rtx3090'],
        'b': ['rtx3090'],
        '8a': ['rtx2080ti'],
        'all': ['rtx4090', 'rtx3090', 'rtx4090', 'rtx3090']
    }
    gpu = gpu_mappings.get(gpu_type, None)
    if not gpu:
        raise ValueError(f'Invalid gpu type {gpu_type}.')
    return gpu


def get_file_from_directory(directory, extension=None, contains_string=None):
    for file in os.listdir(directory):
        if file.endswith(extension) and (not contains_string or contains_string in file):
            return os.path.join(directory, file)
    return None


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


def extract_exp_number(string):
    match = EXP_PATTERN.search(string)
    return int(match.group(1)) if match else 0


def run_single_command(command, gpu, gpu_type, task_counter, num_commands):
    """
    Runs a single command on the specified GPU.

    Args:
        command (str): The command to run.
        gpu (str): GPU type on which the command will be executed (e.g., 'rtx2080ti').
        gpu_type (str): GPU name (e.g., 'a' or 'b').
        task_counter (dict): A counter dict to keep track of tasks processed.
        num_commands (int): Number of commands to run for each task.
    """
    task = command.split("/")[6]
    if task_counter[task] >= num_commands:
        return

    cfg_path = command.split(" ")[3]
    cfg_path_split = cfg_path.split("/")
    shot, exp = cfg_path_split[6], extract_exp_number(cfg_path_split[7])

    log_dir = cfg_path.rsplit("/", 1)[0]
    log_file_name = f"{task}_{shot}_exp{exp}_performance_slurm-%j"

    # Ensure the log directory exists
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    if gpu_type == 'a':
        gpu_str = "--gres=gpu:1 --nodelist=gpu1a"
    elif gpu_type == 'b':
        gpu_str = "--gres=gpu:1 --nodelist=gpu1b"
    else:
        gpu_str = f"--gres=gpu:{gpu}:1"

    slurm_cmd = f'sbatch -p ls6 {gpu_str} --wrap="{command}" -o "{log_dir}/{log_file_name}.out"'
    print(f"{slurm_cmd}\n")

    task_counter[task] += 1
    subprocess.run(slurm_cmd, shell=True)


def run_commands_on_cluster(commands, num_commands, gpu_type='all'):
    """
    Runs the provided commands on the cluster using specified GPUs.

    Args:
    - commands (list): List of commands to be executed.
    - num_commands (int): Number of commands to execute for each task.
    - gpu_type (str, optional): Type of GPU to use. Defaults to 'all'.

    Returns:
    - None
    """
    gpus = determine_gpu(gpu_type)
    if not gpus:
        raise ValueError(f'Invalid gpu type {gpu_type}.')

    gpu_cycle = itertools.cycle(gpus)
    task_counter = Counter({'colon': 0, 'chest': 0, 'endo': 0})

    for command in commands:
        gpu = next(gpu_cycle)
        run_single_command(command, gpu, gpu_type, task_counter, num_commands)


def find_and_validate_json_files(model_dir, task):
    json_files_found = False  # To track if we found any JSON files
    performance_json_count = 0  # To track the number of "performance.json" files found

    model_name = model_dir.split('work_dirs/')[1]
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
                            print(colored(f"'MAP_class1' missing: {model_name}", 'red'))
                            return False

                        if task == "endo" and "AUC/AUC_multilabe" not in data:
                            print(colored(f"'AUC/AUC_multilabe' missing: {model_name}", 'red'))
                            return False

                        for index in range(1, TASK_2_CLASS_COUNT[task]):
                            if f"MAP_class{index}" in data and data[f"MAP_class{index}"] == -0.0:
                                print(colored(f"Value of 'MAP_class{index}' is -0.0: {model_name}", 'red'))
                                return False

                        if task == "colon" and "accuracy/top1" not in data:
                            print(colored(f"'accuracy/top1' missing: {model_name}", 'red'))
                            return False

                        if task == "colon" and "accuracy/top1" in data:
                            model_path = dirpath.split("work_dirs/")[1]
                            #print(colored(f"Accuracy found in colon: {model_path}", 'blue'))

                except json.JSONDecodeError:
                    my_print(f"Cannot load JSON from: {filepath}")
                    my_print(f"Deleting {filepath}")
                    os.remove(filepath)  # Deleting the corrupted JSON file
                    return False
                except PermissionError as permission_error:
                    my_print(f"Permission Error encountered: {permission_error}")
                    return False
                except Exception as e:
                    my_print(f"Error encountered: {e}")
                    return False

    if not json_files_found:
        my_print(f"No JSON files found for {model_name}")
        return False

    if performance_json_count != 1:
        my_print(f"Multiple 'performance.json' found: {performance_json_count}")
        return False

    return True


def is_valid_model_dir(abs_model_dir, task):
    """
    Checks if a directory is a valid model directory.

    Args:
        abs_model_dir (str): Absolute path to the model directory.
        task (str): The task type, e.g., 'colon', 'chest', etc.

    Returns:
        bool: True if it's a valid model directory, False otherwise.
    """
    # Skip if no best checkpoint file
    checkpoint_path = get_file_from_directory(abs_model_dir, ".pth", "best")
    if checkpoint_path is None:
        return False

    # Skip/Delete if no event file
    event_file = get_event_file_from_model_dir(abs_model_dir)
    if event_file is None:
        return False

    # Skip if performance json file is present and its present
    if find_and_validate_json_files(abs_model_dir, task):
        return False

    return True


def print_report(model_infos):
    """
    Prints a report of model directories without a performance JSON.

    Args:
    - model_infos (dict): Information about the models.

    Returns:
    - None
    """
    model_dirs = [model["path"] for model in model_infos.values()]
    if len(model_dirs) == 0:
        print(colored(f"\nAll valid models have an existing performance JSON!\n", 'green'))
        exit()
    else:
        sorted_report_entries = sorted([model_dir for model_dir in model_dirs], key=sort_key)
        print(
            "\n---------------------------------------------------------------------------------------------------------------")
        print("| Valid Models without an existing performance JSON file:")
        print(
            "---------------------------------------------------------------------------------------------------------------")
        for entry in sorted_report_entries:
            print(f"| {entry}")
        print(
            "---------------------------------------------------------------------------------------------------------------")
        print(f"| Found {len(model_dirs)} model runs without existing performance JSON.")
        print(
            "---------------------------------------------------------------------------------------------------------------")


def sort_key(entry):
    # Extract task, shot, and experiment number from the entry
    parts = entry.split('/')
    task = parts[5]
    shot = int(parts[6].split('-')[0])
    exp_number = extract_exp_number(parts[-1])
    return task, shot, exp_number


def process_task_shot_combination(args):
    task, shot = args
    return task, shot, get_model_dirs_without_performance(task=task, shot=shot)


@lru_cache(maxsize=None)
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


def get_model_dirs_without_performance(task, shot):
    """
    Retrieves a list of model directories without a performance file.

    Args:
        task (str): The task type.
        shot (str): The shot type or number.

    Returns:
        list: List of model directories without performance.
    """
    model_dirs = []
    setting_directory = os.path.join(work_dir_path, task, shot)

    try:
        setting_model_dirs = os.listdir(setting_directory)
    except Exception:
        return []

    for model_dir in setting_model_dirs:
        abs_model_dir = os.path.join(setting_directory, model_dir)

        if not os.access(abs_model_dir, os.W_OK):
            print(f"{colored('Missing Write Access:', 'red')} Skipping {abs_model_dir.split('work_dirs/')[1]}")
            continue

        if is_valid_model_dir(abs_model_dir, task):
            model_dirs.append(model_dir)

    return model_dirs


def generate_test_commands(model_infos):
    """
    Generates test commands for given model information.

    Args:
        model_infos (dict): Dictionary containing model information.

    Returns:
        list: List of generated commands.
    """
    commands = []
    for model in model_infos.values():
        model_path = model['path']
        config_filepath = get_file_from_directory(model_path, ".py")
        checkpoint_filepath = get_file_from_directory(model_path, ".pth", "best")
        out_filepath = os.path.join(model_path, "performance.json")

        command = (f"python ensemble/create_performance_file_sample.py "
                   f"--config_path {config_filepath} "
                   f"--checkpoint_path {checkpoint_filepath} "
                   f"--output_path {out_filepath}")
        commands.append(command)
    return commands


# ========================================================================================
work_dir_path = os.path.join("/scratch", "medfm", "medfm-challenge", "work_dirs")
batch_size = 16
# ========================================================================================


def main(args):
    """
    Main method for the script.
    """
    gpu_type = args.gpu
    selected_tasks = args.task

    with Pool() as pool:
        combinations = [(task, shot) for task in selected_tasks for shot in shots]
        results = [result for result in pool.imap_unordered(process_task_shot_combination, combinations)]

    model_infos = {}
    for task, shot, model_list in results:
        for model_name in model_list:
            model_path = os.path.join(work_dir_path, task, shot, model_name)
            exp_num = extract_exp_number(model_name)
            model_infos[model_name] = {
                "task": task,
                "shot": shot,
                "exp_num": exp_num,
                "name": model_name,
                "path": model_path,
            }

    print_report(model_infos)

    user_input = input(f"\nDo you want to generate the testing commands? (yes/no): ")
    if user_input.strip().lower() == 'no':
        exit()

    commands = generate_test_commands(model_infos)
    task_counts = Counter(model["task"] for model in model_infos.values())

    print("Task Counts:")
    for task, count in task_counts.items():
        print(f"{task.capitalize()}: {count}")

    while True:
        user_input = input("\nHow many testing commands per task do you want to generate? ").strip().lower()

        if user_input == 'no':
            exit()

        try:
            num_commands = int(user_input)
            break
        except ValueError:
            print("Invalid input. Please enter a number or 'no' to exit.")

    run_commands_on_cluster(commands, num_commands, gpu_type=gpu_type)


def parse_args():
    """
    Parses command line arguments.

    Returns:
        Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description='Infers missing performances from model runs on the test set.')
    parser.add_argument("--gpu", type=str, default='all',
                        help="GPU type: \n- 'c'=rtx4090,\n- '8a'=rtx2070ti\n"
                             "- 'ab'=rtx3090\n - 'a'=gpu1a\n - 'b'=gpu1b\n- 'all'=rtx4090, rtx3090 cyclic")
    parser.add_argument("--task", type=str, nargs='*', default=["colon", "endo", "chest"],
                        choices=tasks,
                        help="Task type: 'colon', 'chest', or 'endo'. "
                             "Multiple tasks can be provided separated with a whitespace. "
                             "Default is all tasks.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
