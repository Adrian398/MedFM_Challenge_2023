# 1 Dataset Creation
Note: You have to be in the root of your local repository on slurmmaster.

### 1.1 Generate few-shot files for each setting (1,5 and 10-shot; colon, chest and endo):
```bash
python dataset_creation/generate_candidates.py --num_exp 100
```
### 1.2 Create the corresponding config files and queue the training on the cluster nodes:
```bash
python dataset_creation/execute_training.py
```
### 1.3 Extract the 5 (hardcoded) best datasets for each setting and copy them over to data_anns:
```bash
python dataset_creation/find_best_data.py
```
# Tensorboard
```bash
ssh -N -L 6008:localhost:6008 slurmmaster-ls6
```
```bash
tensorboard --logdir dataset_creation/work_dirs/ --port 6008
```