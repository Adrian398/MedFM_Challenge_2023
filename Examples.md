# Install
- Use Python 3.8
- Install pytorch with the newest version
- Install the following packages:
```bash
pip install openmim scipy scikit-learn ftfy regex tqdm tensorboard future pandas
```
```bash
mim install mmpretrain
```

# Examples

 Python path setup, Windows Powershell (copy & paste into console if it does not work)
```bash
$env:PYTHONPATH = "$PWD;" + $env:PYTHONPATH
```
MacOS and Linux:
```bash
export PYTHONPATH="$PWD:$PYTHONPATH"
```

 Train example
```bash
python .\tools\train.py .\configs\dinov2-b_vpt\1-shot_colon.py
```
To adjust parameters like validation interval, max_epochs etc, add the following to the config:
````python
train_cfg = dict(by_epoch=True, val_interval=10, max_epochs=100)
````

 Inference example
 (infer, config, checkpoint, images, out)
```bash
python .\tools\infer.py .\configs\dinov2-b_vpt\1-shot_chest.py .\work_dirs\dinov2-b\exp1\dinov2-b_1-shot_ptokens-1_chest\best_multi-label_mAP_epoch_1.pth .\data\MedFMC_
val\chest\images --out .\results\chest_1-shot_submission.csv
```

### Check and fix generated submission
Check submission files using
```bash
python .\uitilty\check_submission.py
```
Fix alignments if necessary using
```bash
python .\uitilty\align_submission.py
```

## Visualize training using tensorboard
Install required dependencies
````bash
pip install future tensorboard
````
Add the following line to the config 
````python
visualizer = dict(type='Visualizer', vis_backends=[dict(type='TensorboardVisBackend')])
````
Start tensorboard using
```bash
tensorboard --logdir .\work_dirs\
```
Sidenote: If plots appear to be cut off, uncheck `Ignore outliers in chart scaling` under scalar settings
