OVERRIDE = {
    'model': ["convnext-v2-b"],
    'dataset': ["colon"],
    #'shot': [1],
    #'exp_num': [1],
    'lr': [5e-5],  # Start learning rate that increases up to 1e-5 (until max_epochs) with cosine annealing
    'train_bs': [1],
    'seed': [0]
}

SETTINGS = {
    #'exp_suffix': "exp+seed_test_",
    'log_level': "INFO"
}
