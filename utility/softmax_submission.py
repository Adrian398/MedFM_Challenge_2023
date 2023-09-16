import os
import csv
import math
from datetime import datetime


def softmax(values):
    exp_values = [math.exp(v) for v in values]
    sum_exp_values = sum(exp_values)
    return [ev / sum_exp_values for ev in exp_values]


def process_csv(filename):
    with open(filename, 'r') as file:
        reader = csv.reader(file)
        rows = list(reader)

    with open(filename, 'w', newline='') as file:
        writer = csv.writer(file)
        for row in rows:
            values = list(map(float, row[1:3]))
            sm_values = softmax(values)
            row[1], row[2] = sm_values
            writer.writerow(row)


def main(start_path):
    experiments = [f'exp{i}' for i in range(1, 6)]

    for exp in experiments:
        exp_path = os.path.join(start_path, exp)

        for root, dirs, files in os.walk(exp_path):
            for file in files:
                if file.endswith('.csv') and "colon" in file:
                    filepath = os.path.join(root, file)
                    process_csv(filepath)
                    print(f"Processed {filepath}")


base_path = "submissions/evaluation/"


if __name__ == "__main__":
    path = os.path.join('submissions', 'evaluation')
    directories = [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]
    format = "%d-%m_%H-%M-%S"
    valid_directories = []
    for d in directories:
        try:
            valid_directories.append((datetime.strptime(d, format), d))
        except Exception:
            pass
    newest_directory = max(valid_directories, key=lambda x: x[0])[1]
    print(f"Applying Softmax for colon task on newest submission {newest_directory}")
    results_dir = os.path.join(path, newest_directory, "result")

    main(results_dir)
