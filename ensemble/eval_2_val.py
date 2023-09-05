import os
import shutil


def is_valid_csv(path):
    # 1. Check if file exists
    if not os.path.exists(path):
        print(f"File does not exist: {path}")
        return False

    # 2. Check if file is not empty
    if os.path.getsize(path) == 0:
        print(f"File is empty.: {path}")
        return False

    return True


def construct_model_paths(report):
    lines = [line for line in report if any(t in line for t in TASKS)]

    model_infos = []
    for line in lines:
        parts = line.split('\t')

        model_name = parts[-1].strip()
        task_shot_exp = parts[0].split("/")

        # Extract task, shot, and exp number
        model_task = task_shot_exp[0].strip('| ').strip()
        model_shot = task_shot_exp[1].strip()
        model_exp = task_shot_exp[2].strip()

        model_infos.append((model_name, model_task, model_shot, model_exp))
    return model_infos


# ================================================================================
SCRATCH_BASE_PATH = '/scratch/medfm/medfm-challenge'
VAL_TARGET_PATH = 'ensemble/validation'
EVAL_BASE_PATH = 'submissions/evaluation'
TASKS = ["colon", "endo", "chest"]
# ================================================================================

# TODO: Read TIMESTAMP during runtime
TIMESTAMP = '02-09_00-32-41'
EVAL_REPORT_PATH = os.path.join(EVAL_BASE_PATH, TIMESTAMP, 'report.txt')


with open(EVAL_REPORT_PATH, 'r') as f:
    report_content = f.readlines()

model_infos = construct_model_paths(report_content)

for path in model_infos:
    print(path)

for name, task, shot, exp in model_infos:
    # Search for validation prediction csv and validate it
    file_name = f"{task}_{shot}_validation.csv"
    source_file_path = os.path.join(SCRATCH_BASE_PATH, 'work_dirs', task, shot, name, file_name)

    if not is_valid_csv(source_file_path):
        continue

    # Construct target path and ensure that the directory exists
    target_path = os.path.join(VAL_TARGET_PATH, TIMESTAMP, 'result', exp)
    os.makedirs(target_path, exist_ok=True)

    target_file_path = os.path.join(target_path, file_name)
    shutil.copy(source_file_path, target_file_path)
    print(f"Copied from {source_file_path} to {source_file_path}")
