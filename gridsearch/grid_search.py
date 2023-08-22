import importlib.util
import logging
import os
import subprocess
import argparse
import sys

logging.basicConfig(level=logging.INFO)

gridsearch_config_path = "gridsearch/configs/config.py"

BASE_PARAMS_CONFIG = {
    'model': ["clip-b_vpt", "dinov2-b_vpt", "eva-b_vpt", "swin-b_vpt", "swinv2-b", "vit-b_vpt"],
    'dataset': ["chest", "colon", "endo"],
    'shot': [1, 5, 10],
    'exp_num': [1, 2, 3, 4, 5],
    'lr': [1e-4, 1e-5, 1e-6],
}


def run_training(params, exp_suffix, dry_run=False):
    cfg_path = generate_config_path(params['model'], params['shot'], params['dataset'])
    command = ["python", "tools/train.py", cfg_path]

    for key, value in params.items():
        # only add params as command arg if required
        if key not in ['model', 'shot', 'dataset']:
            command.extend([f"--{key}", str(value)])
    print(exp_suffix)
    if exp_suffix:
        print(exp_suffix)
        command.extend(["--exp_suffix", str(exp_suffix)])

    logging.info(f"Generated command: {' '.join(command)}")

    if not dry_run:
        logging.info(f"Starting training with config: {params}")
        subprocess.run(command)
        logging.info(f"Training for config {params} completed")


def generate_config_path(model, shot, dataset):
    return f"configs/{model}/{shot}-shot_{dataset}.py"


def generate_combinations(params_config, combination={}, index=0, dry_run=False, exp_suffix=''):
    if index == len(params_config):
        run_training(combination, dry_run, exp_suffix)
        return

    param_name, values = list(params_config.items())[index]

    # Wrap non-iterables (or strings) in a list
    if not is_iterable(values) or isinstance(values, str):
        values = [values]

    for value in values:
        combination[param_name] = value
        generate_combinations(params_config, combination, index + 1, dry_run)


def merge_configs(base, override):
    effective_config = base.copy()
    for key, value in override.items():
        effective_config[key] = value
    return effective_config


def is_iterable(obj):
    try:
        iter(obj)
    except TypeError:
        return False
    return True


def check_pythonpath_from_cwd():
    if os.getcwd() not in sys.path:
        logging.error(f"PYTHONPATH not set.")
        sys.exit(1)


if __name__ == "__main__":
    check_pythonpath_from_cwd()

    parser = argparse.ArgumentParser(description="Run grid search for training.")
    parser.add_argument("--config", type=str, default=gridsearch_config_path, help="Path to the configuration file.")
    args = parser.parse_args()

    config_path = args.config
    spec = importlib.util.spec_from_file_location("config", config_path)
    config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config)

    # Extract overrides
    USER_OVERRIDE = config.OVERRIDE

    # Extract additional settings
    exp_suffix = config.SETTINGS['exp_suffix']
    dry_run = config.SETTINGS['dry_run']
    log_level = config.SETTINGS['log_level']

    print(exp_suffix)

    logging.getLogger().setLevel(log_level)
    effective_config = merge_configs(BASE_PARAMS_CONFIG, USER_OVERRIDE)
    generate_combinations(effective_config, dry_run=dry_run, exp_suffix=exp_suffix)
