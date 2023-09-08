import json
from collections import defaultdict

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from ensemble.utils.constants import exps, tasks, shots

# Load the JSON files
with open("worse.json", "r") as worse_file:
    worse_data = json.load(worse_file)

with open("better.json", "r") as better_file:
    better_data = json.load(better_file)


def compare_metric_jsons(src_data, trg_data):
    """Simplified comparison of metric values between two datasets with structured keys."""
    # Initialize differences dictionary with structured keys
    differences = {'task': {exp: {f"{task}_{shot}": {} for task in tasks for shot in shots} for exp in exps}}

    acc_map_heatmap_data = defaultdict(lambda: defaultdict(float))
    auc_heatmap_data = defaultdict(lambda: defaultdict(float))

    for exp in exps:
        for task in tasks:
            for shot in shots:
                setting_key = f"{task}_{shot}"
                src_metrics = src_data['task'][exp][setting_key]
                trg_metrics = trg_data['task'][exp][setting_key]

                col_key = f"{exp}_{shot}"

                for metric, src_value in src_metrics.items():
                    diff = float(trg_metrics.get(metric, 0)) - float(src_value)
                    differences['task'][exp][setting_key][metric] = diff

                    # Combine ACC_metric, mAP_metric, and AUC_metric data into their respective dictionaries
                    if metric == "ACC_metric" and task == "colon":
                        acc_map_heatmap_data[task][col_key] = diff
                    elif metric == "mAP_metric" and task != "colon":
                        acc_map_heatmap_data[task][col_key] = diff
                    elif metric == "AUC_metric":
                        auc_heatmap_data[task][col_key] = diff
    return acc_map_heatmap_data, auc_heatmap_data


acc_map_data, auc_heatmap_data = compare_metric_jsons(src_data=worse_data,
                                                      trg_data=better_data)
acc_map_heatmap_df = pd.DataFrame(acc_map_data)
auc_heatmap_df = pd.DataFrame(auc_heatmap_data)

# Plotting the heatmap for ACC_metric (colon) & mAP_metric (chest, endo) using the 'coolwarm' colormap
plt.figure(figsize=(15, 7))
sns.heatmap(acc_map_heatmap_df, cmap='RdBu', center=0, annot=True, fmt=".2f")
plt.title('Differences in ACC_metric (colon) & mAP_metric (chest, endo)')
plt.show()

# Plotting the heatmap for AUC_metric across all tasks using the 'coolwarm' colormap
plt.figure(figsize=(15, 7))
sns.heatmap(auc_heatmap_df, cmap='RdBu', center=0, annot=True, fmt=".2f")
plt.title('Differences in AUC_metric for All Tasks')
plt.show()
