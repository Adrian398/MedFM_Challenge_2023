import json
import os
import re
import shutil
import sys
from multiprocessing import Pool
from functools import lru_cache
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
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


def get_file_from_directory(directory, extension, contains_string=None):
    """Get a file from an absolute directory (i.e. from /scratch/..) with the given extension and optional substring."""
    for file in os.listdir(directory):
        if file.endswith(extension) and (not contains_string or contains_string in file):
            return os.path.join(directory, file)
    return None


def print_report(invalid_model_dirs):
    if len(invalid_model_dirs) == 0:
        print(colored(f"\nAll models have a checkpoint and an event file!\n", 'green'))
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


def find_and_validate_json_files(model_dir):
    json_files_found = False  # To track if we found any JSON files
    performance_json_count = 0  # To track the number of "performance.json" files found

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
                            print(f"Found 'performance.json' but mAP per class (e.g. 'MAP_class1') missing")
                            return False

                except json.JSONDecodeError:
                    print(f"Cannot load JSON from: {filepath}")
                    print(f"Deleting {filepath}")
                    os.remove(filepath)  # Deleting the corrupted JSON file
                    return False
                except PermissionError as permission_error:
                    print(f"Permission Error encountered: {permission_error}")
                    return False
                except Exception as e:
                    print(f"Error encountered: {e}")
                    return False

    if not json_files_found:
        print("No JSON files found.")
        return False

    if performance_json_count != 1:
        print(f"Multiple 'performance.json' found: {performance_json_count}")
        return False

    return True

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


def get_non_valid_model_dirs(task, shot):
    model_dirs = []
    setting_directory = os.path.join(work_dir_path, task, f"{shot}-shot")

    try:
        setting_model_dirs = os.listdir(setting_directory)
    except Exception:
        return None

    for model_dir in setting_model_dirs:
        my_print(f"Checking {task}/{shot}-shot/{model_dir}")
        abs_model_dir = os.path.join(setting_directory, model_dir)

        checkpoint = get_file_from_directory(abs_model_dir, ".pth", "best")

        if checkpoint is None:
            print(colored("No checkpoint file found", 'light_red'))
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

        # Use imap_unordered and directly iterate over the results
        results = []
        for result in pool.imap_unordered(process_task_shot_combination, combinations):
            results.append(result)

    invalid_model_dirs = []
    for task, shot, model_list in results:
        for model_name in model_list:
            model_path = os.path.join(work_dir_path, task, f"{shot}-shot", model_name)
            invalid_model_dirs.append(model_path)

    print_report(invalid_model_dirs)

    user_input = input(f"\nDo you want to delete those model runs? (yes/no): ")
    if user_input.strip().lower() == 'yes':
        for model_dir in invalid_model_dirs:
            remove_model_dir(model_dir)
    else:
        exit()
