import logging
import subprocess

logging.basicConfig(level=logging.INFO)

BASE_PARAMS_CONFIG = {
    'model': ["clip-b_vpt", "dinov2-b_vpt", "eva-b_vpt", "swin-b_vpt", "swinv2-b", "vit-b_vpt"],
    'dataset': ["chest", "colon", "endo"],
    'shot': [1, 5, 10],
    'exp_suffix': [1, 2, 3, 4, 5],
    'lr': [1e-4, 1e-5, 1e-6],
}


def run_training(params):
    cfg_path = generate_config_path(params['model'], params['shot'], params['dataset'])
    command = ["python", "tools/train.py", cfg_path]

    for key, value in params.items():
        # only add params as command arg if required
        if key not in ['model', 'shot', 'dataset']:
            command.extend([f"--{key}", str(value)])

    logging.info(f"Starting training with config: {params}")
    subprocess.run(command)
    logging.info(f"Training for config {params} completed")

def generate_config_path(model, shot, dataset):
    return f"configs/{model}/{shot}-shot_{dataset}.py"


def generate_combinations(params_config, combination={}, index=0):
    if index == len(params_config):
        run_training(combination)
        return

    param_name, values = list(params_config.items())[index]
    for value in values:
        combination[param_name] = value
        generate_combinations(params_config, combination, index + 1)


def merge_configs(base, override):
    effective_config = base.copy()
    for key, value in override.items():
        effective_config[key] = value
    return effective_config


if __name__ == "__main__":
    # User's customization: They can modify this as per their needs.
    USER_OVERRIDE = {
        'model': ['swinv2-b'],
        'dataset': ['colon'],
        'exp_suffix': [1],
        'shot': [1],
        'lr': [1e-6]
    }

    effective_config = merge_configs(BASE_PARAMS_CONFIG, USER_OVERRIDE)
    generate_combinations(effective_config)
