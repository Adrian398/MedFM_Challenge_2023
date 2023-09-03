import os
import glob

root_dir = "/scratch/medfm/medfm-challenge/work_dirs"

exp_dirs = {}
# Traverse through the main categories
for category in ['colon', 'endo', 'chest']:
    # Traverse through 1-shot, 5-shot, 10-shot
    for shot in ['1-shot', '5-shot', '10-shot']:
        # Construct the path pattern for glob
        path_pattern = os.path.join(root_dir, category, shot, '*exp[1-5]*')
        # Get all run directories that match the pattern
        for run_dir in glob.glob(path_pattern):
            # Check if both csv and json files exist in the run directory
            csv_files = glob.glob(os.path.join(run_dir, "*.csv"))
            json_files = glob.glob(os.path.join(run_dir, "*.json"))
            if csv_files and json_files:
                exp_num = next(filter(lambda x: x.startswith("exp"), run_dir.split(os.sep)), None)
                if exp_num:
                    exp_num = exp_num[3:]  # Get the number after "exp"
                    if exp_num not in exp_dirs:
                        exp_dirs[exp_num] = []
                    for csv_file, json_file in zip(csv_files, json_files):
                        exp_dirs[exp_num].append((csv_file, json_file))

# Now, exp_dirs will have the desired output
for exp_num, file_tuples in exp_dirs.items():
    print(f"For exp{exp_num}:")
    for csv, json in file_tuples:
        print(f"CSV: {csv}, JSON: {json}")
    print("---------")
