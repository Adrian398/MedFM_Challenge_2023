# Slurmmaster
Documentation:
[doku-ls6.informatik.uni-wuerzburg.de/docs/venv.html]()

Project directory: ```/scratch/medfm/medfm-challenge``` <br>

Compute Nodes:
````commandline
NAME         GPUs
gpu1a,gpu1b  gpu:rtx3090:1
gpu8a        gpu:rtx2080ti:7
gpu1c        gpu:rtx4090:1
````

## Commands
Show running jobs:
````commandline
squeue 
````
Open bash session on node (here gpu1b):
````commandline
srun --pty -w gpu1b bash
````
Activate env (when in ```/scratch/medfm/medfm-challenge```):
````commandline
source venv/bin/activate
````
Export python path
````commandline
export PYTHONPATH="$PWD:$PYTHONPATH"
````
### Tensorboard
When on Slurmmaster (not on compute node):
````commandline
tensorboard --logdir work_dirs/ --port 6006
````
Locally set up port forwarding (Only active while console is open, this command will not print anything):

````commandline
ssh -N -L 6006:localhost:6006 slurmmaster-ls6
````
Then connect to Tensorboard by going to [localhost:6006]()