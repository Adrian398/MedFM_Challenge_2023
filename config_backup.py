import os
import sys
from datetime import datetime
import shutil

scratch_repo_path = os.path.join("/scratch", "medfm", "medfm-challenge")

given_run_path = sys.argv[1]

given_run_path = os.path.join("work_dirs", given_run_path)
path_components = given_run_path.split(os.sep)
run_dir = os.path.join(*path_components[:4])
run_dir = os.path.join(scratch_repo_path, run_dir)

config_filename = [file for file in os.listdir(run_dir) if file.endswith(".py")][0]
config_path = os.path.join(run_dir, config_filename)

# create dir for submission and config
submission_path = os.path.join("submissions")
date_pattern = datetime.now().strftime("%d-%m")
submission_dir = os.path.join(submission_path, date_pattern)
configs_dir = os.path.join(submission_dir, "configs")
if not any(s.startswith(date_pattern) for s in os.listdir(submission_path)):
    os.makedirs(submission_dir)
    print(f"Directory {submission_dir} created.")
    os.makedirs(configs_dir)
else:
    print(f"Directory with pattern {date_pattern} already exists.")

# copy config into directory
print(f"Copying config from {config_path} to {configs_dir}")
shutil.copy(config_path, configs_dir)