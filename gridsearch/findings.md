- swinv2-b:
  - Exp-Num:        2, 5 perform significantly worse than 1, 3, and 4

# Hyperparameters
## Independently Good Performing:
  - LR: 1e-6

## Dataset Specific
### Colon
- 1-Shot:
  - Model: Swinv2-b (max. 92.7 AUC) > Swin-b (max. 84.3 AUC)
  - Exp: 1,4,5 >>> ..
- 5-Shot:
- 10-Shot:

### Endo
- 1-Shot:
    - Model: 
    - Exp: 3 > 1 > 2 >>> 4, 5
- 5-Shot:
    - Model: Swinv2-b > Swin-b_vpt
    - Exp: 4 > 3 > ..

### Chest
- 10-Shot:
    - Exp: 2 >>> ..