# Examples

 Python path setup, Windows Powershell
```bash
$env:PYTHONPATH = "$PWD;" + $env:PYTHONPATH
```

 Train example
```bash
python .\tools\train.py .\configs\dinov2-b_vpt\1-shot_chest.py
```

 Inference example
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
