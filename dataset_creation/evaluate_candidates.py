import os
import re

config_dir = os.path.join('configs', 'swinv2-b')
candidate_data_dir = os.path.join('dataset_creation', 'candidate_data')
exp_configs = []

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

config_list = os.listdir(config_dir)


def read_dataset_type_from_config(cfg):
    if cfg.__contains__("colon"):
        return "colon"
    if cfg.__contains__("endo"):
        return "endo"
    if cfg.__contains__("chest"):
        return "chest"
    print(f"Aborting, wrong name for config {cfg}")
    exit()


def extract_exp_number(filename):
    match = re.search(r'(\d+)\.txt$', filename)
    return int(match.group(1)) if match else 0


def create_config(train_f, val_f):
    with open(os.path.join(config_dir, config), 'r') as f:
        exp = extract_exp_number(train_f)
        config_content = f.read()

        # This indentation is necessary, don't refactor
        # Todo inject proper work_dir, to save results
        config_injection = f'''
train_dataloader = dict(
    batch_size=8,
    dataset=dict(ann_file='{candidate_data_dir}/{task}/{train_f}'),
)

val_dataloader = dict(
    batch_size=128,
    dataset=dict(ann_file='{candidate_data_dir}/{task}/{val_f}'),
)

default_hooks = dict(
    checkpoint=dict(type='CheckpointHook', interval=1000, max_keep_ckpts=-1, save_last=False),
    logger=dict(interval=10),
)

randomness = dict(seed=0)

train_cfg = dict(by_epoch=True, val_interval=20, max_epochs=20)
        '''

        additional_content = '''
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
        config_injection += additional_content

        new_config_name = f'{shot}-shot_{task}_exp{exp}.py'
        exp_configs.append(new_config_name)

        target_dir = os.path.join('configs', 'dataset_creation')
        if not os.path.exists(target_dir):
            os.makedirs(target_dir)

        with open(os.path.join(target_dir, new_config_name), 'w') as config_file:
            config_file.write(config_content + config_injection)


for config in config_list:
    print(config)
    shot = config[:1]
    task = read_dataset_type_from_config(config)
    dataset_candidate_data_dir = os.path.join(candidate_data_dir, task)

    txt_files = os.listdir(dataset_candidate_data_dir)
    txt_files_train = list(filter(lambda x: f'{shot}-shot_train' in x, txt_files))
    txt_files_train.sort(key=extract_exp_number)
    txt_files_val = list(filter(lambda x: f'{shot}-shot_val' in x, txt_files))
    txt_files_val.sort(key=extract_exp_number)

    for train_file, val_file in zip(txt_files_train, txt_files_val):
        create_config(train_file, val_file)

# Create bash file for training runs
bash_content = ""
for config in exp_configs:
    bash_content += f'python tools/train.py {os.path.join("configs", "dataset_creation", config)}\n'
with open('evaluate.sh', 'w') as bash_file:
    bash_file.write(bash_content)
os.chmod("evaluate.sh", 0o755)

# Todo write separate script (reusing logic from prep_infer) to find best runs => use their exp_numbers to
#  extract best datasets
