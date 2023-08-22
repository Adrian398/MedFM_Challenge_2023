import os
import sys
from datetime import datetime

scratch_repo_path = os.path.join("scratch", "medfm", "medfm-challenge")

given_run_path = sys.argv[1]
given_run_path = os.path.join("work_dirs", given_run_path)

task = os.path.split(given_run_path)[0]
shot = os.path.split(given_run_path)[1]
path_components = os.path.split(given_run_path)
print(path_components)
print(path_components[:4])
x = path_components[:4]
print(os.path.join(*x))
print("end test")
run_dir = os.path.join(*path_components[:4])
print(f"Run dir {run_dir}")

config_filename = [file for file in os.listdir(run_dir) if file.endswith(".py")][0]
checkpoint_filename = [file for file in os.listdir(run_dir) if file.endswith(".pth") and file.__contains__("best")][0]

config_path = os.path.join(run_dir, config_filename)
checkpoint_path = os.path.join(run_dir, checkpoint_filename)
images_path = f"data/MedFMC_val/{task}/images"
csv_name = f"{task}_{shot}_submission.csv"
out_path = os.path.join("results", csv_name)

# create dir for submission and config
submission_path = os.path.join("submissions")
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
