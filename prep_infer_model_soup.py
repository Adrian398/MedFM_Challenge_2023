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



checkpoint_filenames = []


start_dir = "/scratch/medfm/medfm-challenge/work_dirs/endo/10-shot"

# Walk through the base directory and its subdirectories
for dirpath, dirnames, filenames in os.walk(start_dir):
    # Check if the directory starts with "swin_bs"
    if os.path.basename(dirpath).startswith("swin_bs"):
        # For each file in the directory
        for filename in filenames:
            # Check if the file ends with ".pth"
            if filename.endswith(".pth") and "best" in filename:
                # Append the full path of the file to the list
                checkpoint_filenames.append(os.path.join(start_dir, dirpath, filename))

#checkpoint_filenames = ["/scratch/medfm/medfm-challenge/work_dirs/endo/10-shot/swin_bs4_lr0.0005_exp1_20230821-004750/best_multi-label_mAP_epoch_11.pth", "/scratch/medfm/medfm-challenge/work_dirs/endo/10-shot/swin_bs8_lr0.0005_exp1_20230821-172020/best_multi-label_mAP_epoch_100.pth"]

print(checkpoint_filenames)

checkpoint_filenames = checkpoint_filenames[:2]

device = 'cuda' if torch.cuda.is_available() else 'cpu'
state_dicts = []

for f in checkpoint_filenames:
    print(f'Loading {f}')
    state_dicts.append(torch.load(f, map_location=device))

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
#print(metrics)




####create greedy soup
val_results = []
#create val results

for filename in checkpoint_filenames:
    cfg = Config.fromfile("configs/swinv2-b/10-shot_endo.py")
    cfg.load_from = filename
    runner = Runner.from_cfg(cfg)
    metrics = runner.test()
    print(metrics)
    val_results.append(metrics['Aggregate'])

print(val_results)
'''
'''
ranked_candidates = [i for i in range(len(state_dicts))]
ranked_candidates.sort(key=lambda x: -val_results[x])

print(ranked_candidates)
print(val_results)


# current_best = val_results[ranked_candidates[0]]
# best_ingredients = ranked_candidates[:1]
# for i in range(1, len(state_dicts)):
#   # add current index to the ingredients
#   ingredient_indices = best_ingredients \
#     + [ranked_candidates[i]]
#   alphal = [0 for i in range(len(state_dicts))]
#   for j in ingredient_indices:
#     alphal[j] = 1 / len(ingredient_indices)
  
#   # benchmark and conditionally append
#   model = get_model(state_dicts, alphal)
#   current = validate(model)
#   print(f'Models {ingredient_indices} got {current*100}% on validation.')
#   if current > current_best:
#     current_best = current
#     best_ingredients = ingredient_indices



#os.system('python tools/test.py "configs/swinv2-b/10-shot_endo.py" "' + model_soup_path + '" --out "model_soup_results/out.pkl" --out-item "metrics"')