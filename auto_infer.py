import os
from datetime import datetime
import shutil
import subprocess
import sys


def find_repo(root_dir, target):
    for root, dirs, _ in os.walk(root_dir):
        if target in dirs:
            return os.path.join(root, target)
    return None


home_directory = os.path.expanduser("~")
target_directory = "medfm-challenge"

home_repo_path = find_repo(home_directory, target_directory)

path = sys.argv[1]
delimiter = "/"
path = os.path.join("work_dirs", path)

task = path.split(delimiter)[0]
shot = path.split(delimiter)[1]
run_dir = delimiter.join(path.split(delimiter)[:4])

config_filename = [file for file in os.listdir(run_dir) if file.endswith(".py")][0]
checkpoint_filename = [file for file in os.listdir(run_dir) if file.endswith(".pth") and file.__contains__("best")][0]

config_path = os.path.join(run_dir, config_filename)
checkpoint_path = os.path.join(run_dir, checkpoint_filename)
images_path = f"data/MedFMC_val/{task}/images"
csv_name = f"{task}_{shot}_submission.csv"
out_path = os.path.join(home_repo_path, "results", csv_name)

# create dir for submission and config
submission_path = os.path.join(home_repo_path, "submissions")
date_pattern = datetime.now().strftime("%d-%m")
submission_dir = os.path.join(submission_path, date_pattern)
configs_dir = os.path.join(submission_dir, "configs")
if not any(s.startswith(date_pattern) for s in os.listdir(submission_path)):
    os.makedirs(submission_dir)
    print(f"Directory {submission_dir} created.")
    # os.makedirs(configs_dir)
else:
    print(f"Directory with pattern {date_pattern} already exists.")

print(f"Copying config from {config_path} to {configs_dir}")
print(f"Starting infer with {config_path} {checkpoint_path} {images_path} {out_path}")
# copy config into directory
# shutil.copy(config_path, configs_dir)
# subprocess.run(["python", "tools/infer.py", config_path, checkpoint_path, images_path, out_path])
