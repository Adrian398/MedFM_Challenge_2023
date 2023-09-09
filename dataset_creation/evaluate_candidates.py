import os
import re
from generate_candidates import delete_all_files_from_directory

config_source_dir = os.path.join('configs', 'swinv2-b')
config_target_dir = os.path.join('configs', 'dataset_creation')
candidate_data_dir = os.path.join('dataset_creation', 'candidate_data')
exp_configs = []

shot_2_max_epochs = {
    1: 20,
    5: 10,
    10: 5
}

'''
    Notes on configs for dataset evaluation:
    
    They all have to be in the above defined config_dir and be named N-shot_task.py, e.g. "1-shot_chest.py"
    
    Use relatively high learning rates without warm_up to get quick results in a low number of epochs.
    The following Adam optimizer with a LR of 0.002, no parameter scheduling worked well for a BS of 8 in colon,
    to produce decent results within 20 epochs.
   
    > lr = 0.002 
    > optim_wrapper = dict(
        optimizer=dict(
            betas=(
                0.9,
                0.999,
            ),
            eps=1e-08,
            lr=lr,
            type='AdamW',
            weight_decay=0.05),
        paramwise_cfg=dict(
            bias_decay_mult=0.0,
            custom_keys=dict({
                '.absolute_pos_embed': dict(decay_mult=0.0),
                '.relative_position_bias_table': dict(decay_mult=0.0)
            }),
            flat_decay_mult=0.0,
            norm_decay_mult=0.0))

    > param_scheduler = []
'''


def extract_config_info(cfg):
    match = re.match(r'(\d+)-shot_([\w]+).py', cfg)
    if match:
        return int(match.group(1)), match.group(2)
    print(f"Aborting, wrong name for config {cfg}")
    exit()


def extract_exp_number(filename):
    match = re.search(r'(\d+)\.txt$', filename)
    return int(match.group(1)) if match else 0


def create_config(train_f, val_f, config, shot, dataset_type):
    with open(os.path.join(config_source_dir, config), 'r') as f:
        exp = extract_exp_number(train_f)
        src_config = f.read()

        max_epochs = shot_2_max_epochs[shot]

        config_injection = f'''
train_dataloader = dict(
    batch_size=8,
    dataset=dict(ann_file='{candidate_data_dir}/{dataset_type}/{train_f}'),
)

val_dataloader = dict(
    batch_size=16,
    dataset=dict(ann_file='{candidate_data_dir}/{dataset_type}/{val_f}'),
)

default_hooks = dict(
    checkpoint=dict(type='CheckpointHook', interval=1000, max_keep_ckpts=-1, save_last=False),
    logger=dict(interval=10),
)

randomness = dict(seed=0)

train_cfg = dict(by_epoch=True, val_interval={max_epochs}, max_epochs={max_epochs})
        '''

        optimizer_content = '''
lr = 0.002
optim_wrapper = dict(
optimizer=dict(
    betas=(
        0.9,
        0.999,
    ),
    eps=1e-08,
    lr=lr,
    type='AdamW',
    weight_decay=0.05),
paramwise_cfg=dict(
    bias_decay_mult=0.0,
    custom_keys=dict({
        '.absolute_pos_embed': dict(decay_mult=0.0),
        '.relative_position_bias_table': dict(decay_mult=0.0)
    }),
    flat_decay_mult=0.0,
    norm_decay_mult=0.0))

param_scheduler = []
        '''
        target_config = src_config + config_injection + optimizer_content

        # ==================================================================================
        # =========================       Config Adjustments       =========================

        # Replace work_dir in config
        target_config = target_config.replace("work_dir = f'work_dirs/",
                                              "work_dir = f'dataset_creation/work_dirs/")

        # Replace experiment number in config
        target_config = re.sub(r"(exp_num\s*=\s*)(\d+)", lambda m: m.group(1) + str(exp), target_config)

        # ==================================================================================

        new_config_name = f'{shot}-shot_{dataset_type}_exp{exp}.py'
        print(f"Created config:\t{new_config_name}")
        exp_configs.append(new_config_name)

        target_dir = os.path.join('configs', 'dataset_creation')
        if not os.path.exists(target_dir):
            os.makedirs(target_dir)

        with open(os.path.join(target_dir, new_config_name), 'w') as config_file:
            config_file.write(target_config)


def generate_train_commands():
    delete_all_files_from_directory(os.path.join(config_target_dir, 'chest'))
    delete_all_files_from_directory(os.path.join(config_target_dir, 'colon'))
    delete_all_files_from_directory(os.path.join(config_target_dir, 'endo'))

    config_list = os.listdir(config_source_dir)

    for config in config_list:
        shot, dataset_type = extract_config_info(config)
        dataset_candidate_data_dir = os.path.join(candidate_data_dir, dataset_type)

        txt_files = os.listdir(dataset_candidate_data_dir)
        txt_files_train = list(filter(lambda x: f'{shot}-shot_train' in x, txt_files))
        txt_files_train.sort(key=extract_exp_number)
        txt_files_val = list(filter(lambda x: f'{shot}-shot_val' in x, txt_files))
        txt_files_val.sort(key=extract_exp_number)

        for train_file, val_file in zip(txt_files_train, txt_files_val):
            create_config(train_file, val_file, config=config, shot=shot, dataset_type=dataset_type)

    return [f'python tools/train.py configs/dataset_creation/{exp_config}' for exp_config in exp_configs]


if __name__ == "__main__":
    train_commands = generate_train_commands()
    command_str = "\n".join(train_commands)
    output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'train_commands.sh')

    with open(output_path, 'w') as file:
        file.write(command_str)
        print(f"Saving train commands to {output_path}")
