import torch 
import os
import mmengine
from mmengine.config import Config, ConfigDict, DictAction
from mmengine.evaluator import DumpResults
from mmengine.runner import Runner

def get_sd(state_dicts, alphal):
  sd = {}  # Initialize an empty dictionary
  for k in state_dicts[0]['state_dict'].keys():
      #print(k)
      #print(state_dicts[0][k])
      sd[k] = state_dicts[0]['state_dict'][k].clone() * alphal[0]
  for i in range(1, len(state_dicts)):
      for k in state_dicts[i]['state_dict'].keys():
          sd[k] = sd[k] + state_dicts[i]['state_dict'][k].clone() * alphal[i]
  return sd


#parameter
nshot = 10
dataset = 'endo'
model_name = 'resnet101'
exp_num = 1
seed = -1000


checkpoint_filenames = []
configs_for_checkpoints_filenames = []

use_seed = False

start_dir = "/scratch/medfm/medfm-challenge/work_dirs/" + dataset + "/" + str(nshot) + "-shot"
# Walk through the base directory and its subdirectories
for dirpath, dirnames, filenames in os.walk(start_dir):
    # Check if the directory starts with "swin_bs"
    if os.path.basename(dirpath).startswith(model_name):
        # For each file in the directory
        for filename in filenames:
            # Check if the file ends with ".pth"
            if filename == (str(nshot) + "-shot_" + dataset + ".py"):
                if not os.path.exists(os.path.join(start_dir, dirpath, filename)):
                    raise FileNotFoundError(f"The file '{os.path.join(start_dir, dirpath, filename)}' does not exist.")
                else:
                    with open(os.path.join(start_dir, dirpath, filename), 'r') as file:
                        config = file.read()
                    
                    seed_string = "seed = " + str(seed)
                    exp_num_string = "exp_num = " + str(exp_num) 
                    if use_seed:
                        if seed_string in config and exp_num_string in config:
                            configs_for_checkpoints_filenames.append(os.path.join(start_dir, dirpath, filename))
                            filenames_to_get_pth = os.listdir(os.path.join(start_dir, dirpath))
                            for file_name_to_get_pth in filenames_to_get_pth:
                                if file_name_to_get_pth.endswith(".pth") and "best" in file_name_to_get_pth:
                                    # Append the full path of the file to the list
                                    checkpoint_filenames.append(os.path.join(start_dir, dirpath, file_name_to_get_pth))
                    else: 
                        if exp_num_string in config:
                            configs_for_checkpoints_filenames.append(os.path.join(start_dir, dirpath, filename))
                            filenames_to_get_pth = os.listdir(os.path.join(start_dir, dirpath))
                            for file_name_to_get_pth in filenames_to_get_pth:
                                if file_name_to_get_pth.endswith(".pth") and "best" in file_name_to_get_pth:
                                    # Append the full path of the file to the list
                                    checkpoint_filenames.append(os.path.join(start_dir, dirpath, file_name_to_get_pth))


#checkpoint_filenames = ["/scratch/medfm/medfm-challenge/work_dirs/endo/10-shot/swin_bs4_lr0.0005_exp1_20230821-004750/best_multi-label_mAP_epoch_11.pth", "/scratch/medfm/medfm-challenge/work_dirs/endo/10-shot/swin_bs8_lr0.0005_exp1_20230821-172020/best_multi-label_mAP_epoch_100.pth"]

print(checkpoint_filenames)
print(configs_for_checkpoints_filenames)

#take just three 
###############REMOVE LATER!!!!!##############################################
#checkpoint_filenames = checkpoint_filenames[:5]
device = 'cuda' if torch.cuda.is_available() else 'cpu'
state_dicts = []
for f in checkpoint_filenames:
    print(f'Loading {f}')
    state_dicts.append(torch.load(f, map_location=device))

####create greedy soup
val_results = []
#create val results of all models which could be included in the soup
for i, filename in enumerate(checkpoint_filenames):
    cfg = Config.fromfile(configs_for_checkpoints_filenames[i])
    cfg.load_from = filename
    runner = Runner.from_cfg(cfg)
    metrics = runner.test()
    print(metrics)
    val_results.append(metrics['Aggregate'])

print(val_results)

#rank all those models
ranked_candidates = [i for i in range(len(state_dicts))]
ranked_candidates.sort(key=lambda x: -val_results[x])


# run greedy soup algorithm
current_best = val_results[ranked_candidates[0]]
best_ingredients = ranked_candidates[:1]
for i in range(1, len(state_dicts)):
  # add current index to the ingredients
  ingredient_indices = best_ingredients \
    + [ranked_candidates[i]]
  alphal = [0 for i in range(len(state_dicts))]
  for j in ingredient_indices:
    alphal[j] = 1 / len(ingredient_indices)
  
  # benchmark and conditionally append
  sd = get_sd(state_dicts, alphal)
  folder_path =  checkpoint_filenames[0].split("-shot")[0] + "-shot/modelsoup"


  if not os.path.exists(folder_path):
    # If the folder doesn't exist, create it
    os.makedirs(folder_path)
  model_soup_path = folder_path + "/" + model_name + "_soup.pth"
  torch.save(sd, model_soup_path)


  #### do validate with model soup state dict:
  cfg = Config.fromfile("configs/swinv2-b/10-shot_endo.py")
  cfg.load_from = model_soup_path
  runner = Runner.from_cfg(cfg)
  metrics = runner.test()
  current = metrics['Aggregate']

  print(f'Models {ingredient_indices} got {current}% on validation.')
  if current > current_best:
    current_best = current
    best_ingredients = ingredient_indices


alphal = [0 for i in range(len(state_dicts))]
for j in best_ingredients:
  alphal[j] = 1 / len(best_ingredients)
sd = get_sd(state_dicts, alphal)

best_model_soup_path = folder_path + "/" + model_name + "exp" + str(exp_num) + str(seed) +  "_soup_best.pth"
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
#runn tool/test.py with config file
#cfg = Config.fromfile("configs/swinv2-b/10-shot_endo.py")
#cfg.load_from = model_soup_path
#runner = Runner.from_cfg(cfg)
#metrics = runner.test()
#print(metrics)'''