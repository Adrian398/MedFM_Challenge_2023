# 0. Preliminary Stuff
Before being able to use the python command, the python path must be added to the environment variables.

Windows Powershell (copy & paste into console)
```bash
$env:PYTHONPATH = "$PWD;" + $env:PYTHONPATH
```

MacOS and Linux:
```bash
export PYTHONPATH="$PWD:$PYTHONPATH"
```

# 1. Training
## 1.1 Hyperparameters | Config
In order to adjust training hyperparameters such as validation interval, max_epochs etc., one must add the following 
line of code to the config:
````python
train_cfg = dict(by_epoch=True, val_interval=10, max_epochs=100)
````

## 1.2 Training example
```bash
python .\tools\train.py .\configs\swin-b_vpt\1-shot_colon.py
```

## 1.3 Inference example
The inference assumes that one model was already trained and produced a checkpoint file within the `work_dirs` directory. Note that the checkpoint path must be added to the following command.

Approximate command sequence: infer, config, checkpoint, images, out
```bash
python .\tools\infer.py .\configs\swin-b_vpt\1-shot_colon.py .\work_dirs\colon\1-shot\{CHECKPOINT_PATH} .\data\MedFMC_
val\chest\images --out .\results\chest_1-shot_submission.csv
```

### 1.3.1 Check and fix the generated submission
Check submission files using
```bash
python .\uitilty\check_submission.py
```
Fix alignments if necessary using
```bash
python .\uitilty\align_submission.py
```

## 1.4 Visualization with Tensorboard
Add the following line to any desired config file
````python
visualizer = dict(type='Visualizer', vis_backends=[dict(type='TensorboardVisBackend')])
````

Start tensorboard using
```bash
tensorboard --logdir .\work_dirs\
```
Sidenote: If plots appear to be cut off, uncheck `Ignore outliers in chart scaling` under scalar settings and set smoothing to 0.
