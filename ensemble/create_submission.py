import glob
import os
import re

root_dir = "/scratch/medfm/medfm-challenge/work_dirs"


def extract_exp_number(string):
    match = re.search(r'exp(\d+)', string)
    return int(match.group(1)) if match else 0


exp_dirs = {}
tasks = ['colon', 'endo', 'chest']
shots = ['1-shot', '5-shot', '10-shot']
exps = ['exp1', 'exp2', 'exp3', 'exp4', 'exp5']
# Traverse through the main categories
for task in tasks:
    # Traverse through 1-shot, 5-shot, 10-shot
    exp_dirs[task] = {}
    for shot in shots:
        exp_dirs[task][shot] = {}
        for exp in exps:
            exp_dirs[task][shot][exp] = []

        # Construct the path pattern for glob
        path_pattern = os.path.join(root_dir, task, shot, '*exp[1-5]*')
        # Get all run directories that match the pattern
        for run_dir in glob.glob(path_pattern):
            print("Checking run directory", run_dir)
            # Check if both csv and json files exist in the run directory
            csv_files = glob.glob(os.path.join(run_dir, "*.csv"))
            json_files = glob.glob(os.path.join(run_dir, "*.json"))
            if csv_files and json_files:
                exp_num = extract_exp_number(run_dir)
                if exp_num != 0:
                    exp_dirs[task][shot][f"exp{exp_num}"].append({'csv': csv_files[0], 'json': json_files[0]})

# Now, exp_dirs will have the desired output
for task in tasks:
    for shot in shots:
        for exp in exps:
            print(f"For {task} {shot} {exp}: {len(exp_dirs[task][shot][exp])}")