OVERRIDE = {
    'model': ["swinv2-b"],
    'dataset': ["colon"],
    #'shot': [5, 10],
    #'exp_num': [1],
    'lr': [1e-6],  # Start learning rate that increases up to 1e-5 (until max_epochs) with cosine annealing
    'train_bs': [1, 2, 8, 16],
    'seed': [0]
}

SETTINGS = {
    'exp_suffix': "bs_test_with_jitter",
    'log_level': "INFO"
}
