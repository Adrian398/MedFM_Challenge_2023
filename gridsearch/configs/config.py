OVERRIDE = {
    'model': ["resnet101"],
    'dataset': ["colon"],
    'shot': [1],
    #'exp_num': [1],
    'lr': [1e-6],  # Start learning rate that increases up to 1e-5 (until max_epochs) with cosine annealing
    'train_bs': [1, 2, 8, 16],
    'seed': [0]
}

SETTINGS = {
    'exp_suffix': "bs_test_",
    'log_level': "INFO"
}
