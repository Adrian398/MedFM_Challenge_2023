OVERRIDE = {
    'model': ["swinv2-b"],
    'dataset': "endo",
    'shot': [10],
    'exp_num': [1],
    'lr': [1e-8],  # Start learning rate that increases up to 1e-5 (until max_epochs) with cosine annealing
    #'train_bs': [2, 4, 6, 8]
}

SETTINGS = {
    'exp_suffix': "lr-test_",
    'log_level': "INFO"
}
