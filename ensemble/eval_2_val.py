import os
import re


def construct_model_paths(report):
    model_paths = []
    for line in report:
        task, shot = line.split('/')[0:2]
        model_name = line.strip().split()[-1]

        model_path = os.path.join(SCRATCH_BASE_PATH, "work_dirs", task, shot, model_name)
        model_paths.append(model_path)
    return model_paths


# ================================================================================
TIMESTAMP = '02-09_00-32-41'
SCRATCH_BASE_PATH = '/scratch/medfm/medfm-challenge'
EVAL_BASE_PATH = 'submissions/evaluation'
EVAL_FOLDER_PATH = os.path.join(EVAL_BASE_PATH, TIMESTAMP)
EVAL_REPORT_PATH = os.path.join(EVAL_FOLDER_PATH, 'report.txt')
# ================================================================================


with open(EVAL_REPORT_PATH, 'r') as f:
    report_content = f.readlines()

model_paths = construct_model_paths(report_content)


# Extracting the necessary details from the report content using regex
pattern = re.compile(r"\| (\w+)/(\d+-shot)/\w+     Aggregate: [\d.]+        (\w+)")
matches = pattern.findall(report_content)

for path in model_paths:
    print(path)

