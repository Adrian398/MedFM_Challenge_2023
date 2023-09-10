from datetime import datetime

from ensemble.gridsearch.task_submission import extract_data
from ensemble.submission import print_report_for_setting
from ensemble.utils.constants import tasks, shots, exps


if __name__ == "__main__":
    root_dir = "/scratch/medfm/medfm-challenge/work_dirs"
    TIMESTAMP = datetime.now().strftime("%d-%m_%H-%M-%S")
    DATA_SUBMISSION, DATA_VALIDATION = extract_data(root_dir=root_dir)

    total_models = 0
    least_models = 100000
    most_models = -1
    most_setting = ""
    least_setting = ""
    print(f"""\n============== Overall Model Summary ==============""")
    for task in tasks:
        for shot in shots:
            for exp in exps:
                models_for_setting = len(DATA_SUBMISSION[task][shot][exp])
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
                print_report_for_setting(full_model_list=DATA_SUBMISSION, task=task, shot=shot, exp=exp)

