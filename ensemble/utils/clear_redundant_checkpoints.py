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
    return task, shot, get_non_valid_model_dirs(task=task, shot=shot)


def extract_exp_number(string):
    match = EXP_PATTERN.search(string)
    return int(match.group(1)) if match else 0


def get_file_from_directory(directory, ext, keyword):
    """Helper function to get files from a directory based on extension and keyword."""
    return [f for f in os.listdir(directory) if f.endswith(ext) and keyword in f]


def get_non_valid_model_dirs(task, shot):
    model_dirs = []
    setting_directory = os.path.join(work_dir_path, task, f"{shot}-shot")

    try:
        setting_model_dirs = os.listdir(setting_directory)
    except Exception:
        return None

    for model_dir in setting_model_dirs:
        abs_model_dir = os.path.join(setting_directory, model_dir)

        checkpoint_files = [f for f in os.listdir(abs_model_dir) if f.endswith(".pth")]

        # If no checkpoint files
        if not checkpoint_files:
            print(colored(f"No checkpoint file found for {abs_model_dir}", 'red'))
            model_dirs.append(model_dir)
            continue

        # If only "best" checkpoints
        best_checkpoints = get_file_from_directory(abs_model_dir, ".pth", "best")

        if len(best_checkpoints) > 1:
            print(colored(f"More than one 'best' checkpoint found in {abs_model_dir}", 'yellow'))
            model_dirs.append(model_dir)
            continue

        if len(best_checkpoints) == len(checkpoint_files):
            continue

        # If there are checkpoints other than "best"
        print(colored(f"Found non-best checkpoints in {abs_model_dir}", 'cyan'))
        model_dirs.append(model_dir)

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
    for task, shot, model_list in results_invalid:
        for model_name in model_list:
            model_path = os.path.join(work_dir_path, task, f"{shot}-shot", model_name)
            invalid_model_dirs.append(model_path)

    print_report(invalid_model_dirs)

    # user_input = input(f"\nDo you want to delete those model runs? (yes/no): ")
    # if user_input.strip().lower() == 'yes':
    #     for model_dir in invalid_model_dirs:
    #         remove_model_dir(model_dir)
    # else:
    #     exit()
