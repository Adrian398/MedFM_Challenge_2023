OVERRIDE = {
    'model': ["resnet101"],
    'dataset': ["endo"],
    'shot': [1],
    #'exp_num': [3],
    'lr': [5e-6],  # Start learning rate that increases up to 1e-5 (until max_epochs) with cosine annealing
    #'train_bs': [32],
    'seed': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
}

SETTINGS = {
    'exp_suffix': "seed_test_",
    'log_level': "INFO"
}
