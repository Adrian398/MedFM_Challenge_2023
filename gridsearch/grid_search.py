import importlib.util
import itertools
import logging
import os
import subprocess
import argparse
import sys
import time

logging.basicConfig(level=logging.INFO)

gridsearch_config_path = "gridsearch/configs/config.py"

BASE_PARAMS_CONFIG = {
    'model': ["clip-b_vpt", "dinov2-b_vpt", "eva-b_vpt", "swin-b_vpt", "swinv2-b", "vit-b_vpt"],
    'dataset': ["chest", "colon", "endo"],
    'shot': [1, 5, 10],
    'exp_num': [1, 2, 3, 4, 5],
    'lr': [1e-6, 1e-7, 1e-8],
    'train_bs': [2, 4, 6, 8]
}


def create_command(params, exp_suffix):
    cfg_path = generate_config_path(params['model'], params['shot'], params['dataset'])
    command = ["python", "tools/train.py", cfg_path]

    for key, value in params.items():
        # only add params as command arg if required
        if key not in ['model', 'shot', 'dataset']:
            command.extend([f"--{key}", str(value)])

    if exp_suffix:
        command.extend(["--exp_suffix", str(exp_suffix)])

    return command


def generate_config_path(model, shot, dataset):
    return f"configs/{model}/{shot}-shot_{dataset}.py"


def generate_combinations(params_config, exp_suffix, combination={}, index=0):
    commands = []
    if index == len(params_config):
        command = create_command(params=combination, exp_suffix=exp_suffix)
        commands.append(command)
    else:
        param_name, values = list(params_config.items())[index]

        # Wrap non-iterables (or strings) in a list
        if not is_iterable(values) or isinstance(values, str):
            values = [values]

        for value in values:
            combination_copy = combination.copy()
            combination_copy[param_name] = value
            commands.extend(generate_combinations(params_config=params_config,
                                                  exp_suffix=exp_suffix,
                                                  combination=combination_copy,
                                                  index=index + 1))
    return commands


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


def run_commands_on_cluster(commands, gpu=None):
    """
    Runs the generated commands on the cluster.
    If no GPU is specified, the commands are queued on the cluster in the following scheme:
    gpuc -> gpua / gpub -> gpua / gpub -> gpuc -> ...
    """

    if gpu == 'c':
        gpus = ['rtx4090']
    elif gpu == 'ab':
        gpus = ['rtx3090']
    elif gpu == '8a':
        gpus = ['rtx2080ti']
    elif gpu is None:
        gpus = ['rtx4090', 'rtx4090', 'rtx3090', 'rtx3090']
    else:
        raise ValueError(f'Invalid gpu type {gpu}.')

    gpu_cycle = itertools.cycle(gpus)

    # Ensure the log directory exists
    log_dir = "gridsearch/output"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    for command in commands:
        cmd_str = " ".join(command)

        gpu = next(gpu_cycle)

        slurm_cmd = f'sbatch -p ls6 \
        --gres=gpu:{gpu}:1 \
        --wrap="{cmd_str}" \
        -o "{log_dir}/slurm-%j.out"'

        subprocess.run(slurm_cmd, shell=True)


if __name__ == "__main__":
    check_pythonpath_from_cwd()

    parser = argparse.ArgumentParser(description="Run grid search for training.")
    parser.add_argument("--config", type=str, default=gridsearch_config_path, help="Path to the configuration file.")
    parser.add_argument("--gpu", type=str, default=None, help="GPU type: 'c'=rtx4090 or 'ab'=rtx3090.")
    args = parser.parse_args()

    gpu_type = args.gpu
    config_path = args.config
    spec = importlib.util.spec_from_file_location("config", config_path)
    config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config)

    # Extract overrides
    USER_OVERRIDE = config.OVERRIDE

    # Extract additional settings
    try:
        exp_suffix = config.SETTINGS['exp_suffix']
    except:
        exp_suffix = None
    log_level = config.SETTINGS['log_level']

    logging.getLogger().setLevel(log_level)
    effective_config = merge_configs(BASE_PARAMS_CONFIG, USER_OVERRIDE)

    commands = generate_combinations(params_config=effective_config, exp_suffix=exp_suffix)

    # Display the generated commands
    print("Generated Commands:")
    for command in commands:
        print(f"{' '.join(command)}")

    # Prompt the user
    user_input = input(f"Do you want to run {len(commands)} commands on the cluster? (yes/no): ")

    if user_input.strip().lower() == 'yes':
        run_commands_on_cluster(commands, gpu=gpu_type)
