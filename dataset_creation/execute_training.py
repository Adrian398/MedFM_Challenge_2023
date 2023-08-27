import argparse
import itertools
import logging
import os
import subprocess
import sys
import time

from evaluate_candidates import generate_train_commands


def extract_filename(path):
    """Extract the filename without the preceding path and file extension."""
    base_name = os.path.basename(path)  # Get the filename with extension
    file_name, _ = os.path.splitext(base_name)  # Split off the file extension
    return file_name


def check_pythonpath_from_cwd():
    if os.getcwd() not in sys.path:
        logging.error(f"PYTHONPATH not set.")
        sys.exit(1)


def run_commands_on_cluster(commands, gpu=None, delay_seconds=1):
    """
    Runs the generated commands on the cluster.
    If no GPU is specified, the commands are queued on the cluster in the following scheme:
    gpuc -> gpua / gpub -> gpua / gpub -> gpuc -> ...
    """

    if gpu == 'c':
        gpus = ['rtx4090']
    elif gpu == 'ab':
        gpus = ['rtx3090']
    elif gpu is None:
        gpus = ['rtx4090', 'rtx3090', 'rtx4090', 'rtx3090']
    else:
        raise ValueError(f'Invalid gpu type {gpu}.')

    gpu_cycle = itertools.cycle(gpus)

    # Ensure the log directory exists
    log_dir = "dataset_creation/output_logs"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    for cmd_str in commands:
        gpu = next(gpu_cycle)

        file_name = extract_filename(cmd_str)
        slurm_cmd = f'sbatch -p ls6 --gres=gpu:{gpu}:1 --wrap="{cmd_str}" -o "{log_dir}/slurm-%j_{file_name}.out"'

        subprocess.run(slurm_cmd, shell=True)

        time.sleep(delay_seconds)


if __name__ == "__main__":
    check_pythonpath_from_cwd()

    parser = argparse.ArgumentParser(description="Run grid search for training.")
    parser.add_argument("--gpu", type=str, default=None, help="GPU type: 'c'=rtx4090 or 'ab'=rtx3090.")
    args = parser.parse_args()

    gpu_type = args.gpu

    commands = generate_train_commands()

    # Display the generated commands
    print("Generated Commands:")
    print("\n".join(commands))

    # Prompt the user
    user_input = input(f"Do you want to run {len(commands)} commands on the cluster? (yes/no): ")

    if user_input.strip().lower() == 'yes':
        run_commands_on_cluster(commands=commands, gpu=gpu_type)
