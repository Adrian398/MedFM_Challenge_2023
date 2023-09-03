import glob
import os

work_dir_path = "/scratch/medfm/medfm-challenge/work_dirs"

categories = ["colon", "endo", "chest"]
shots = ["1-shot", "5-shot", "10-shot"]


def find_run_dirs(category, shot):
    """Generator that yields run directories with 'exp' in their name."""
    run_dirs_pattern = os.path.join(work_dir_path, category, shot, "*exp*")
    for run_dir in glob.glob(run_dirs_pattern):
        yield run_dir


def get_csv_json_pair(run_dir):
    """Returns tuple containing matched .csv and .json files."""
    csv_file = glob.glob(os.path.join(run_dir, "*.csv"))
    json_file = glob.glob(os.path.join(run_dir, "*.json"))

    # If both a .csv and a .json file are found, return them as a tuple
    if csv_file and json_file:
        return (csv_file[0], json_file[0])
    return None


def group_by_exp(category, shot):
    """Groups (csv, json) pairs by their exp number."""
    exp_dict = {}
    for run_dir in find_run_dirs(category, shot):
        pair = get_csv_json_pair(run_dir)
        if pair:
            exp_num = next((s[3:] for s in run_dir.split("/") if "exp" in s), None)
            if exp_num:
                exp_dict.setdefault(exp_num, []).append(pair)
    return exp_dict


def main():
    for category in categories:
        for shot in shots:
            exp_grouped_pairs = group_by_exp(category, shot)
            for exp, pairs in exp_grouped_pairs.items():
                print(f"For {category}/{shot} in exp{exp}:")
                for csv, json in pairs:
                    # Your processing code here
                    print(csv, json)


if __name__ == "__main__":
    main()
