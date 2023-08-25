OVERRIDE = {
    'model': ["swinv2-b"],
    'dataset': "endo",
    'shot': [10],
    'exp_num': [2,3,4,5],
    'lr': [1e-6],  # Start learning rate that increases up to 1e-5 (until max_epochs) with cosine annealing
    'train_bs': [8]
}

SETTINGS = {
    'exp_suffix': "exp-test_",
    'log_level': "INFO"
}
