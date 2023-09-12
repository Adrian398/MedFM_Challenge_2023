import glob
import json
import os
from datetime import datetime

from colorama import Fore
from termcolor import colored
from tqdm import tqdm

from ensemble.submission import print_report_for_setting, extract_exp_number
from ensemble.utils.constants import tasks, shots, exps


def check_and_extract_data(model_dir_abs):
    model_dir_rel = model_dir_abs.split('work_dirs/')[1]

    json_files = glob.glob(os.path.join(model_dir_abs, "*.json"))

    if json_files:
        exp_num = extract_exp_number(model_dir_rel)
        if exp_num != 0:
            metrics = json.load(open(json_files[0], 'r'))
            return {'metrics': metrics, 'name': model_dir_rel}, exp_num
    return None, None


def extract_data(root_dir):
    data_dict = {
        task: {
            shot: {
                exp: [] for exp in EXPS
            } for shot in SHOTS
        } for task in TASKS
    }

    # Total iterations: tasks * shots * exps * model_dirs * subm_types
    total_iterations = 0
    for task in tasks:
        for shot in shots:
            total_iterations += len(glob.glob(os.path.join(root_dir, task, shot, '*exp[1-5]*')))

    print(f"Checking {colored(str(total_iterations), 'blue')} models:")
    with tqdm(total=total_iterations, bar_format="{l_bar}%s{bar}%s{r_bar}" % (Fore.BLUE, Fore.RESET)) as pbar:
        for task in tasks:
            for shot in shots:
                path_pattern = os.path.join(root_dir, task, shot, '*exp[1-5]*')
                for model_dir in glob.glob(path_pattern):
                    data, exp_num = check_and_extract_data(model_dir_abs=model_dir)
                    if data and exp_num:
                        data_dict[task][shot][f"exp{exp_num}"].append(data)
                    pbar.update(1)
    return data_dict


if __name__ == "__main__":
    root_dir = "/scratch/medfm/medfm-challenge/work_dirs"
    TASKS = ["colon", "endo", "chest"]
    SHOTS = ["1-shot", "5-shot", "10-shot"]
    EXPS = ["exp1", "exp2", "exp3", "exp4", "exp5"]
    TIMESTAMP = datetime.now().strftime("%d-%m_%H-%M-%S")
    DATA = extract_data(root_dir=root_dir)

    total_models = 0
    least_models = 100000
    most_models = -1
    most_setting = ""
    least_setting = ""
    print(f"""\n============== Overall Model Summary ==============""")
    for task in tasks:
        for shot in shots:
            for exp in exps:
                models_for_setting = len(DATA[task][shot][exp])
                print(f"| Setting: {task}/{shot}/{exp}\t>> Models: {models_for_setting}")
                total_models += models_for_setting
                if models_for_setting > most_models:
                    most_models = models_for_setting
                    most_setting = f"{task} {shot} {exp}"
                if models_for_setting < least_models:
                    least_models = models_for_setting
                    least_setting = f"{task} {shot} {exp}"
    print("===================================================")
    print(f"| Total models: {total_models}")
    print(f"| Most models: {most_models} {most_setting}")
    print(f"| Least models: {least_models} {least_setting}")
    print("===================================================")

    for task in tasks:
        for shot in shots:
            for exp in exps:
                print_report_for_setting(full_model_list=DATA, task=task, shot=shot, exp=exp)

