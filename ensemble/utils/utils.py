import os
import sys

from ensemble.utils.constants import EXP_PATTERN


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
        '8a': ['rtx2080ti'],
        'all': ['rtx4090', 'rtx3090', 'rtx4090', 'rtx3090']
    }
    return gpu_mappings.get(gpu_type, [])


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


def find_files_in_directory(directory, extension=None, contains_string=None):
    found_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if (not extension or file.endswith(extension)) and (not contains_string or contains_string in file):
                found_files.append(os.path.join(root, file))
    return found_files


def extract_exp_number(string):
    match = EXP_PATTERN.search(string)
    return int(match.group(1)) if match else 0