import os

import torch
from mmengine.config import Config
from mmengine.runner import Runner


# This method takes a list of state_dicts and a list of weights and returns the weighted sum of the state_dicts
# It does this by multiplying each state_dict with its corresponding weight and then summing them up
def get_sd(state_dicts, alphal):
    sd = {}  # Initialize an empty dictionary
    for k in state_dicts[0]['state_dict'].keys():
        sd[k] = state_dicts[0]['state_dict'][k].clone() * alphal[0]

    for i in range(1, len(state_dicts)):
        for k in state_dicts[i]['state_dict'].keys():
            sd[k] = sd[k] + state_dicts[i]['state_dict'][k].clone() * alphal[i]

    return sd


def get_valid_files_from_directory(directory, file_suffix, keyword=None):
    filenames = [f for f in os.listdir(directory) if f.endswith(file_suffix)]
    if keyword:
        filenames = [f for f in filenames if keyword in f]
    return filenames


def join_files_with_directory(directory, filenames):
    return [os.path.join(directory, filename) for filename in filenames]


def find_checkpoints_in_config(directory, config_filename, seed, exp_num, use_seed):
    config_path = os.path.join(directory, config_filename)
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"The file '{config_path}' does not exist.")

    with open(config_path, 'r') as file:
        config = file.read()
    seed_string = "seed = " + str(seed)
    exp_num_string = f"exp{exp_num}.txt"

    if (use_seed and seed_string in config and exp_num_string in config) or (not use_seed and exp_num_string in config):
        checkpoint_filenames = get_valid_files_from_directory(directory, ".pth", "best")
        print("Adding checkpoint files: " + str(directory))
        return join_files_with_directory(directory, checkpoint_filenames)
    return []


# Parameters
nshot = 5
dataset = 'endo'
model_name = 'resnet101'
exp_num = 1
seed = -1000
use_seed = False

start_dir = f"/scratch/medfm/medfm-challenge/work_dirs/{dataset}/{nshot}-shot"
checkpoint_filenames = []
configs_for_checkpoints_filenames = []

print(f"Checking {start_dir}")
# Walk through the base directory and its subdirectories
for dirpath, dirnames, filenames in os.walk(start_dir):
    # Check if the directory starts with the model name
    # todo potentially remove the soup check
    if os.path.basename(dirpath).startswith(model_name) and not os.path.basename(dirpath).__contains__("soup"):
        config_filename = f"{nshot}-shot_{dataset}.py"
        if config_filename in filenames:
            configs_for_checkpoints_filenames.append(os.path.join(dirpath, config_filename))
            checkpoint_filenames.extend(find_checkpoints_in_config(dirpath, config_filename, seed, exp_num, use_seed))

# checkpoint_filenames = ["/scratch/medfm/medfm-challenge/work_dirs/endo/10-shot/swin_bs4_lr0.0005_exp1_20230821-004750/best_multi-label_mAP_epoch_11.pth", "/scratch/medfm/medfm-challenge/work_dirs/endo/10-shot/swin_bs8_lr0.0005_exp1_20230821-172020/best_multi-label_mAP_epoch_100.pth"]


# take just three
###############REMOVE LATER!!!!!##############################################
# checkpoint_filenames = checkpoint_filenames[:5]
device = 'cuda' if torch.cuda.is_available() else 'cpu'
state_dicts = []
for f in checkpoint_filenames:
    print(f'Loading {f}')
    state_dicts.append(torch.load(f, map_location=device))

####create greedy soup
val_results = []
# create val results of all models which could be included in the soup
#

for i, filename in enumerate(checkpoint_filenames):
    cfg = Config.fromfile(configs_for_checkpoints_filenames[i])
    cfg.load_from = filename
    runner = Runner.from_cfg(cfg)
    metrics = runner.test()
    print(metrics)
    val_results.append(metrics['Aggregate'])


print(val_results)

# rank all those models
ranked_candidates = [i for i in range(len(state_dicts))]
ranked_candidates.sort(key=lambda x: -val_results[x])

# run greedy soup algorithm
current_best = val_results[ranked_candidates[0]]
best_ingredients = ranked_candidates[:1]
print("Starting model soup search")
# this for loop will always create the same model because
for i in range(1, len(state_dicts)):
    # add current index to the ingredients
    ingredient_indices = best_ingredients + [ranked_candidates[i]]
    print("Trying ingredients: " + str(ingredient_indices))
    alphal = [0 for i in range(len(state_dicts))]
    for j in ingredient_indices:
        alphal[j] = 1 / len(ingredient_indices)

    # benchmark and conditionally append
    sd = get_sd(state_dicts, alphal)
    # exit()
    folder_path = checkpoint_filenames[0].split("-shot")[0] + "-shot/modelsoup"

    if not os.path.exists(folder_path):
        # If the folder doesn't exist, create it
        os.makedirs(folder_path)
    model_soup_path = folder_path + "/" + model_name + "_soup.pth"
    torch.save(sd, model_soup_path)

    #### do validate with model soup state dict:
    cfg = Config.fromfile(configs_for_checkpoints_filenames[0])
    cfg.load_from = model_soup_path
    runner = Runner.from_cfg(cfg)
    metrics = runner.test()
    current = metrics['Aggregate']

    print(f'Models {ingredient_indices} got {current}% on validation.')
    if current > current_best:
        print('New best model soup')
        current_best = current
        best_ingredients = ingredient_indices
    else:
        print('No improvement')


alphal = [0 for i in range(len(state_dicts))]
for j in best_ingredients:
    alphal[j] = 1 / len(best_ingredients)
sd = get_sd(state_dicts, alphal)

best_model_soup_path = folder_path + "/" + model_name + "exp" + str(exp_num) + str(seed) + "_soup_best.pth"
torch.save(sd, best_model_soup_path)
cfg = Config.fromfile(configs_for_checkpoints_filenames[0])
cfg.load_from = best_model_soup_path
runner = Runner.from_cfg(cfg)
metrics = runner.test()
best_result = metrics['Aggregate']

print(val_results)
print("Best_ingredients: " + str(best_ingredients))
print("Best result: " + str(best_result))

# print(type(state_dicts[0]))

########stadard soup#########
# alphal = [1 / len(state_dicts) for i in range(len(state_dicts))]
# sd = get_sd(state_dicts, alphal)

# folder_path =  checkpoint_filenames[0].split("-shot")[0] + "-shot/modelsoup"
# model = "swin"


# if not os.path.exists(folder_path):
#     # If the folder doesn't exist, create it
#     os.makedirs(folder_path)
# model_soup_path = folder_path + "/" + model + "_soup.pth"
# torch.save(sd, model_soup_path)

##### create validation #####
### config file auch angeben und dann validation machen! 
# runn tool/test.py with config file
# cfg = Config.fromfile("configs/swinv2-b/10-shot_endo.py")
# cfg.load_from = model_soup_path
# runner = Runner.from_cfg(cfg)
# metrics = runner.test()
# print(metrics)'''
