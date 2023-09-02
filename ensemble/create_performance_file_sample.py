import json
from mmengine.config import Config
from mmengine.runner import Runner


if __name__ == "__main__":
    # todo read from run directory
    checkpoint_path = "work_dirs/endo/1-shot/resnet101_bs4_lr1e-06_exp3__20230902-125110/best_Aggregate_epoch_5.pth"
    config_path = "configs/resnet101/1-shot_endo.py"

    cfg = Config.fromfile(config_path)
    cfg.load_from = checkpoint_path
    runner = Runner.from_cfg(cfg)
    metrics = runner.test()
    with open('performance.json', 'w') as f:
        json.dump(metrics, f, indent=4)
