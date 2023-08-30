OVERRIDE = {
    'model': ["resnet101"],
    'dataset': ["colon"],
    #'shot': [10],
    #'exp_num': [1],
    'lr': [1e-6],  # Start learning rate that increases up to 1e-5 (until max_epochs) with cosine annealing
    'train_bs': [32],
    'seed': [2, 3, 4, 5, 6, 7]
}

SETTINGS = {
    'exp_suffix': "seed_test_",
    'log_level': "INFO"
}
