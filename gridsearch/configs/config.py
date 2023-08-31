OVERRIDE = {
    'model': ["swinv2-b"],
    'dataset': ["endo", "chest"],
    'shot': [1, 5, 10],
    #'exp_num': [2],
    'lr': [1e-6],  # Start learning rate that increases up to 1e-5 (until max_epochs) with cosine annealing
    #'train_bs': [32],
    'seed': [0, 1, 2]
}

SETTINGS = {
    'exp_suffix': "exp+seed_test_",
    'log_level': "INFO"
}
