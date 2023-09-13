import os
import re
import shutil
import sys
from multiprocessing import Pool
from termcolor import colored

EXP_PATTERN = re.compile(r'exp(\d+)')


def remove_model_dir(model_dir):
    try:
        shutil.rmtree(model_dir)
        print(f"Successfully removed {model_dir}")
    except PermissionError:
        print(f"Permission denied: Unable to delete {model_dir}. Please check your permissions.")
    except Exception as e:
        print(f"Error: {e}")


def print_report(invalid_model_dirs, total_gb):
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
        print(colored(f"Total size of checkpoint files: {total_gb:.2f} GB", 'green'))

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


def process_task_shot_combination(args):
    task, shot = args
    model = get_non_valid_model_dirs(task=task, shot=shot)
    return task, shot, model


def extract_exp_number(string):
    match = EXP_PATTERN.search(string)
    return int(match.group(1)) if match else 0


def get_file_from_directory(directory, ext, keyword):
    """Helper function to get files from a directory based on extension and keyword."""
    return [f for f in os.listdir(directory) if f.endswith(ext) and keyword in f]


def get_non_valid_model_dirs(task, shot):
    model_dirs = []
    setting_directory = os.path.join(work_dir_path, task, f"{shot}-shot")

    print(f"\nProcessing Setting {task}/{shot}-shot")

    try:
        setting_model_dirs = os.listdir(setting_directory)
    except Exception:
        return None

    for model_dir in setting_model_dirs:
        abs_model_dir = os.path.join(setting_directory, model_dir)

        checkpoint_files = [f for f in os.listdir(abs_model_dir) if f.endswith(".pth")]

        # If no checkpoint files
        if not checkpoint_files:
            print(colored(f"No checkpoint file found for {model_dir}", 'red'))
            continue

        # If only "best" checkpoints
        best_checkpoints = get_file_from_directory(abs_model_dir, ".pth", "best")

        if len(best_checkpoints) > 1:
            print(colored(f"More than one 'best' checkpoint found in {abs_model_dir}", 'yellow'))
            continue

        if len(best_checkpoints) == len(checkpoint_files):
            continue

        # Calculate the total size of the checkpoint files in this directory
        total_ckpt_gb = 0
        for chkpt_file in checkpoint_files:
            ckpt_in_gb = os.path.getsize(os.path.join(abs_model_dir, chkpt_file)) / (1024 ** 3)
            total_ckpt_gb += ckpt_in_gb  # Convert bytes to GB

        model_dirs.append((model_dir, total_ckpt_gb))

    return model_dirs


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
# ========================================================================================


if __name__ == "__main__":  # Important when using multiprocessing
    with Pool() as pool:
        combinations = [(task, shot) for task in tasks for shot in shots]
        results_invalid = list(pool.imap_unordered(process_task_shot_combination, combinations))

    invalid_model_dirs = []
    total_gb = 0

    for task, shot, model_list in results_invalid:
        task_gb = 0

        for model_info in model_list:
            if len(model_info) != 2:
                print(f"Unexpected data in model_list: {model_info}")
                continue

            model_name, model_gb = model_info
            print(model_name, model_gb)


        #     model_path = os.path.join(work_dir_path, task, f"{shot}-shot", model_name)
        #     task_gb += model_gb
        #     invalid_model_dirs.append(model_path)
        # print(f"Task {task} non-best Checkpoint GB:  {task_gb:.2f}")
    #
    #     total_gb += task_gb
    # print(f"Total non-best Checkpoint GB:  {total_gb:.2f}")
    #print_report(invalid_model_dirs, total_gb)

    # user_input = input(f"\nDo you want to delete those model runs? (yes/no): ")
    # if user_input.strip().lower() == 'yes':
    #     for model_dir in invalid_model_dirs:
    #         remove_model_dir(model_dir)
    # else:
    #     exit()
