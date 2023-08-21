# Slurmmaster
```bash
ssh slurmmaster-ls6
```

Documentation:
[https://doku-ls6.informatik.uni-wuerzburg.de/docs/venv.html](https://doku-ls6.informatik.uni-wuerzburg.de/docs/venv.html)

Project directory: ```cd /scratch/medfm/medfm-challenge``` <br>

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
When on Slurmmaster (not on compute node) in directory medfm-challenge and venv is active:
````commandline
tensorboard --logdir work_dirs/ --port 6006
````
Optionally add the following line to ~/.bashrc to automate this process by typing ``tb``
````commandline
alias tb="cd /scratch/medfm/medfm-challenge && source venv/bin/activate && tensorboard --logdir work_dirs/ --port 6006"
````
Locally set up port forwarding (Only active while console is open, this command will not print anything):


````bash
ssh -N -L 6006:localhost:6006 slurmmaster-ls6
````
Then connect to Tensorboard by going to [localhost:6006]()

Copy the following into the search field inside tensorboard to display relevant metrics (or manually select and pin them)
````commandline
loss|auc|base_lr|map|accuracy
````
When the plots seem to be cut off, uncheck ``ignore outliers`` and set ``smoothing`` to 0.

Sometimes tensorboard crashes but keeps blocking the port, in this case use ``lsof -i :6006`` to check which process is blocking the
port then use ``kill -9 <PID>``
## Set up git via SSH

Create ssh key on server (Confirm filename, password etc. with enter => default values)
````commandline
ssh-keygen -t rsa -b 4096 -C "your_email@example.com"
````
Copy SSH key and add it to gitlab
````commandline
cat ~/.ssh/id_rsa.pub
````
### Before working in the dir, make sure to type
````commandline
umask 000
````
or add this line to your ./bash_profile
### Fix files without write permission for others:
From the root directory execute
````commandline
find . -user YOUR_USERNAME -exec chmod 777 {} \;
````
This will set full permissions for all files you own in the current directory and its subdirectories.
## Basic Linux commands
Remove file
````commandline
rm filename
````

Remove directory and contents
````commandline
rm -r directory
````

Show only files currently staged to be committed:
````commandline
git diff --cached --name-only
````