OVERRIDE = {
    'model': ["pre_processed_swin-b_vpt"],
    'dataset': ["endo"],
    'shot': [1],
    #'exp_num': [3],
    'lr': [5e-4],
    'train_bs': [8],
    'seed': [0, 1, 2, 3, 4, 5, 6]
}

SETTINGS = {
    'exp_suffix': "pre_processed",
    'log_level': "INFO"
}
