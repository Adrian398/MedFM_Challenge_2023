import os
import shutil
from datetime import datetime

# todo, fill this array with run_dirs (from workdir, e.g. "colon/1-shot/swin_93878")
best_runs = [
    "colon/1-shot/swinv2_bs8_lr0.0001_exp5_exp-test_20230823-002433",
    "colon/5-shot/swin_bs8_lr0.0005_exp4__exp-test_20230824-102425",
    "colon/10-shot/swin_bs8_lr0.05_exp1_20230820-133202",
    "endo/1-shot/swinv2_bs8_lr1e-05_exp3__exp-test_20230824-132431_20230824-171326",  # swin-b testen (swinv2 in queue)
    "endo/5-shot/swinv2_bs8_lr1e-05_exp4__exp-test_20230824-151647",  # swinv2 testen (swinv2 in queue)
    "endo/10-shot/swin_bs8_lr0.0005_exp1_20230821_201554",  # swinv2 UND alle exps durchtesten
    "chest/1-shot/vit-b_1-shot_ptokens-1_chest_baseline_test_exp2",
    "chest/5-shot/vit-b_5-shot_ptokens-1_chest_baseline_test_exp2",
    "chest/10-shot/clip-b_1_bs4_lr0.001_10-shot_chest_exp1_baseline_test/20230824_010301",
]

# create dir for submission and config
date_pattern = datetime.now().strftime("%d-%m_%H-%M-%S")
submission_dir = os.path.join("submissions", date_pattern)
configs_dir = os.path.join(submission_dir, "configs")
predictions_dir = os.path.join(submission_dir, "predictions")

os.makedirs(submission_dir)
with open(os.path.join(submission_dir, "report.txt"), "w") as file:
    txt_report = ""
    for run in best_runs:
        txt_report += run + "\n"
    file.write(txt_report)

os.makedirs(configs_dir)
os.makedirs(predictions_dir)

bash_script = "#!/bin/bash\n"
for given_run_path in best_runs:
    scratch_repo_path = os.path.join("/scratch", "medfm", "medfm-challenge")

    task = given_run_path.split(os.sep)[0]
    if task.__contains__("-"):
        task = task.split("-")[0]
    shot = given_run_path.split(os.sep)[1]
    print(given_run_path)
    given_run_path = os.path.join("work_dirs", given_run_path)
    path_components = given_run_path.split(os.sep)
    run_dir = os.path.join(*path_components[:4])
    run_dir = os.path.join(scratch_repo_path, run_dir)

    config_filename = [file for file in os.listdir(run_dir) if file.endswith(".py")][0]
    checkpoint_filename = [file for file in os.listdir(run_dir) if file.endswith(".pth") and file.__contains__("best")][
        0]

    config_path = os.path.join(run_dir, config_filename)
    checkpoint_path = os.path.join(run_dir, checkpoint_filename)
    images_path = os.path.join(scratch_repo_path, "data", "MedFMC_val", task, "images")
    csv_name = f"{task}_{shot}_submission.csv"
    out_path = os.path.join(predictions_dir, csv_name)

    # copy config into submission directory
    shutil.copy(config_path, configs_dir)
    command = f"python tools/infer.py {config_path} {checkpoint_path} {images_path} --out {out_path}\n"
    bash_script += command

bash_script += f"echo {submission_dir}"
print(f"Saved respective configs to {configs_dir}")
print("Created infer.sh")
print(f"Run ./infer.sh to create prediction files in {predictions_dir}")
with open("infer.sh", "w") as file:
    file.write(bash_script)
os.chmod("infer.sh", 0o755)
