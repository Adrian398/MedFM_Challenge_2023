import argparse
import json
import os
from collections import defaultdict

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.gridspec as gridspec

from ensemble.utils.constants import exps, tasks, shots

def load_json_from_directory(base_path, timestamp):
    path = os.path.join(base_path, timestamp, "results.json")
    with open(path, 'r') as file:
        data = json.load(file)
    return data


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

def main():
    # Step 1: Accept timestamps as input arguments
    parser = argparse.ArgumentParser(description="Compare two JSON files based on timestamps.")
    parser.add_argument("timestamp1", help="First timestamp for the directory containing the first JSON file.")
    parser.add_argument("timestamp2", help="Second timestamp for the directory containing the second JSON file.")
    args = parser.parse_args()

    base_path = "ensemble/validation"
    src_data = load_json_from_directory(base_path, args.timestamp1)
    trg_data = load_json_from_directory(base_path, args.timestamp2)

    acc_map_data, auc_heatmap_data = compare_metric_jsons(src_data=src_data,trg_data=trg_data)

    acc_map_heatmap_df = pd.DataFrame(acc_map_data)
    auc_heatmap_df = pd.DataFrame(auc_heatmap_data)

    fig = plt.figure(figsize=(15, 14))
    gs = gridspec.GridSpec(2, 1, height_ratios=[1, 1])

    # Create the first heatmap in the top section of the grid
    ax0 = plt.subplot(gs[0])
    sns.heatmap(acc_map_heatmap_df, cmap='RdBu', center=0, annot=True, fmt=".2f", ax=ax0)
    ax0.set_title('Differences in ACC_metric (colon) & mAP_metric (chest, endo)')

    # Create the second heatmap in the bottom section of the grid
    ax1 = plt.subplot(gs[1])
    sns.heatmap(auc_heatmap_df, cmap='RdBu', center=0, annot=True, fmt=".2f", ax=ax1)
    ax1.set_title('Differences in AUC_metric for All Tasks')

    # Adjust the space between the plots
    plt.tight_layout()

    out_path = "ensemble/validation/compare"

    # Check if the directory exists, and if not, create it
    if not os.path.exists(out_path):
        os.makedirs(out_path)

    # Save the combined image
    plt.savefig(os.path.join(out_path, f"{args.timestamp1}_vs_{args.timestamp2}_combined_comparison.png"))


if __name__ == "__main__":
    main()
