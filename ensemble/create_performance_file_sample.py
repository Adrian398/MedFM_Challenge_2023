import json
import argparse
from mmengine.config import Config
from mmengine.runner import Runner

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test a model with given checkpoint and config.")

    # Adding command-line arguments
    parser.add_argument("--checkpoint_path", type=str, required=True, help="Path to the model checkpoint.")
    parser.add_argument("--config_path", type=str, required=True, help="Path to the model configuration.")
    parser.add_argument("--output_path", type=str, default="performance.json", help="Path to save the output metrics. Default is 'performance.json'.")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size of the test dataloader.")

    args = parser.parse_args()

    cfg = Config.fromfile(args.config_path)
    task = cfg.dataset
    cfg.test_dataloader.batch_size = args.batch_size
    cfg.test_dataloader.dataset.data_prefix = f'/scratch/medfm/medfm-challenge/data/MedFMC_train/{task}/images'

    if task == "colon":
        cfg.test_evaluator = [
            dict(type='AveragePrecision'),
            dict(type='Accuracy', topk=(1,)),
            dict(type='SingleLabelMetric', items=['precision', 'recall']),
            dict(type='Aggregate'),
            dict(type='AUC')
        ]
    else:
        cfg.test_evaluator = [
            dict(type='AveragePrecision'),
            dict(type='MultiLabelMetric', average='macro'),  # class-wise mean
            dict(type='MultiLabelMetric', average='micro'),  # overall mean
            dict(type='AUC', multilabel=True),
            dict(type='Aggregate', multilabel=True)]

    cfg.load_from = args.checkpoint_path
    runner = Runner.from_cfg(cfg)
    metrics = runner.test()
    with open(args.output_path, 'w') as f:
        json.dump(metrics, f, indent=4)
