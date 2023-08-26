import torch 
import os

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

checkpoint_filenames = ["/scratch/medfm/medfm-challenge/work_dirs/endo/10-shot/swin_bs4_lr0.0005_exp1_20230821-004750/best_multi-label_mAP_epoch_11.pth", "/scratch/medfm/medfm-challenge/work_dirs/endo/10-shot/swin_bs8_lr0.0005_exp1_20230821-172020/best_multi-label_mAP_epoch_100.pth"]

device = 'cuda' if torch.cuda.is_available() else 'cpu'
state_dicts = []

for f in checkpoint_filenames:
    print(f'Loading {f}')
    state_dicts.append(torch.load(f, map_location=device))

print(type(state_dicts[0]))


alphal = [1 / len(state_dicts) for i in range(len(state_dicts))]
sd = get_sd(state_dicts, alphal)

folder_path =  checkpoint_filenames[0].split("-shot")[0] + "-shot/modelsoup"
model = "swin"


if not os.path.exists(folder_path):
    # If the folder doesn't exist, create it
    os.makedirs(folder_path)
model_soup_path = folder_path + "/" + model + "_soup.pth"
torch.save(sd, model_soup_path)

##### create validation #####
### config file auch angeben und dann validation machen! 
#runn tool/test.py with config file
os.system('python tools/test.py "configs/swinv2-b/10-shot_endo.py" "' + model_soup_path + '"')
