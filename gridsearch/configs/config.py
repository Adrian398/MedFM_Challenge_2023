OVERRIDE = {
    'model': ["resnet101"],
    'dataset': ["endo"],
    'shot': [5, 10],
    #'exp_num': [2],
    'lr': [5e-6],  # Start learning rate that increases up to 1e-5 (until max_epochs) with cosine annealing
    #'train_bs': [32],
    'seed': [0, 1, 2, 3, 4]
}

SETTINGS = {
    'exp_suffix': "exp+seed_test_",
    'log_level': "INFO"
}
