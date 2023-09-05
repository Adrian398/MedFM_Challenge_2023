import os


def construct_model_paths(report):
    lines = [line for line in report if any(task in line for task in TASKS)]

    model_paths = []
    for line in lines:
        parts = line.split('\t')

        model_name = parts[-1].strip()
        task_shot_exp = parts[0].split("/")

        # Extract task, shot, and exp number
        task = task_shot_exp[0].strip('| ').strip()
        shot = task_shot_exp[1].strip()
        exp = task_shot_exp[2].split("exp")[1].strip()  # extract the number after "exp"

        # Constructing the path
        path = os.path.join(EVAL_BASE_PATH, "work_dirs", task, shot, model_name)
        model_paths.append(path)
    return model_paths


# ================================================================================
TIMESTAMP = '02-09_00-32-41'
SCRATCH_BASE_PATH = '/scratch/medfm/medfm-challenge'
EVAL_BASE_PATH = 'submissions/evaluation'
EVAL_FOLDER_PATH = os.path.join(EVAL_BASE_PATH, TIMESTAMP)
EVAL_REPORT_PATH = os.path.join(EVAL_FOLDER_PATH, 'report.txt')
TASKS = ["colon", "endo", "chest"]
# ================================================================================


with open(EVAL_REPORT_PATH, 'r') as f:
    report_content = f.readlines()

model_paths = construct_model_paths(report_content)

for path in model_paths:
    print(path)

