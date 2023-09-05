import os


def construct_model_paths(report):
    lines = [line for line in report if any(task in line for task in TASKS)]

    model_infos = []
    for line in lines:
        parts = line.split('\t')

        model_name = parts[-1].strip()
        task_shot_exp = parts[0].split("/")

        # Extract task, shot, and exp number
        task = task_shot_exp[0].strip('| ').strip()
        shot = task_shot_exp[1].strip()
        exp = task_shot_exp[2].strip()

        model_infos.append({
            'model_name': model_name,
            'task': task,
            'shot': shot,
            'exp': exp,
        })
    return model_infos


# ================================================================================
SCRATCH_BASE_PATH = '/scratch/medfm/medfm-challenge'
VAL_TARGET_PATH = 'ensemble/validation'
EVAL_BASE_PATH = 'submission/evaluation'
TASKS = ["colon", "endo", "chest"]
# ================================================================================

# TODO: Read TIMESTAMP during runtime
TIMESTAMP = '02-09_00-32-41'
EVAL_REPORT_PATH = os.path.join(EVAL_BASE_PATH, TIMESTAMP, 'predictions', 'report.txt')


with open(EVAL_REPORT_PATH, 'r') as f:
    report_content = f.readlines()

model_infos = construct_model_paths(report_content)

for path in model_infos:
    print(path)

for model_name, task, shot, exp in model_infos:
    # Search for validation prediction csv and validate it
    file_name = f"{task}_{shot}_validation.csv"
    source_path = os.path.join(SCRATCH_BASE_PATH, 'work_dirs', task, shot, model_name, file_name)
    print(source_path)

    # Construct target path and ensure that the directory exists
    target_path = os.path.join(VAL_TARGET_PATH, TIMESTAMP, 'result', exp)
    os.makedirs(target_path, exist_ok=True)
    print(target_path)
    exit()
