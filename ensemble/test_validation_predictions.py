import os

import pandas as pd
import math
import numpy as np
import torch
from sklearn import metrics
from sklearn.metrics import average_precision_score

def compute_auc(cls_scores, cls_labels):
    cls_aucs = []
    for i in range(cls_scores.shape[1]):
        scores_per_class = cls_scores[:, i]
        labels_per_class = cls_labels[:, i]
        try:
            auc_per_class = metrics.roc_auc_score(labels_per_class,
                                                  scores_per_class)
            # print('class {} auc = {:.2f}'.format(i + 1, auc_per_class * 100))
        except ValueError:
            pass
        cls_aucs.append(auc_per_class * 100)

    return cls_aucs


def cal_metrics_multilabel(target, cosine_scores):
    """Calculate mean AUC with given dataset information and cosine scores."""

    sample_num = target.shape[0]
    cls_num = cosine_scores.shape[1]

    gt_labels = np.zeros((sample_num, cls_num))
    for k in range(target.shape[0]):
        label = target[k]
        gt_labels[k, :] = label

    cls_scores = np.zeros((sample_num, cls_num))
    for k in range(target.shape[0]):
        cos_score = cosine_scores[k]
        norm_scores = [1 / (1 + math.exp(-1 * v)) for v in cos_score]
        cls_scores[k, :] = np.array(norm_scores)

    cls_aucs = compute_auc(cls_scores, gt_labels)
    mean_auc = np.mean(cls_aucs)

    return mean_auc

def compute_task_specific_metrics(pred_path, gt_path, task):
    predictions = pd.read_csv(pred_path)
    ground_truth = pd.read_csv(gt_path)

    target = torch.tensor(ground_truth['label'].values)
    pred = torch.tensor(predictions['score'].values)

    auc = cal_metrics_multilabel(target, pred)

    metrics = {
        'AUC': auc
    }

    if task in ['chest', 'endo']:
        map_value = average_precision_score(target.numpy(), pred.numpy()) * 100
        metrics['mAP'] = map_value

    if task == 'colon':
        correct_predictions = sum(predictions['label'] == ground_truth['label'])
        acc = correct_predictions / len(predictions)
        metrics['ACC'] = acc

    return metrics


# Directory paths
PREDICTION_DIR = "ensemble/validation/"
GT_DIR = "/scratch/medfm/medfm-challenge/data/MedFMC_trainval_annotation/"

# Iterate over experiments and tasks
results = {}

experiments = [folder for folder in os.listdir(PREDICTION_DIR) if folder.startswith('exp')]
for exp in experiments:
    exp_dir = os.path.join(PREDICTION_DIR, exp)
    tasks = [task.split('_')[0] for task in os.listdir(exp_dir) if task.endswith('_validation.csv')]

    results[exp] = {}
    for task in tasks:
        pred_path = os.path.join(exp_dir, f"{task}_*shot_validation.csv")
        gt_path = os.path.join(GT_DIR, f"{task}_trainval.txt")
        print("task:",task)
        print("pred_path:", pred_path)
        print("gt_path:", gt_path)
        exit()
        metrics = compute_task_specific_metrics(pred_path, gt_path, task)
        results[exp][task] = metrics

# Display the results
for exp, metrics in results.items():
    print(f"Experiment: {exp}")
    for task, task_metrics in metrics.items():
        print(f"\tTask: {task}")
        for metric_name, metric_value in task_metrics.items():
            print(f"\t\t{metric_name}: {metric_value}")
