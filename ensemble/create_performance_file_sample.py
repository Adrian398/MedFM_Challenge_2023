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

    args = parser.parse_args()

    cfg = Config.fromfile(args.config_path)
    cfg.load_from = args.checkpoint_path
    runner = Runner.from_cfg(cfg)
    metrics = runner.test()
    with open(args.output_path, 'w') as f:
        json.dump(metrics, f, indent=4)
