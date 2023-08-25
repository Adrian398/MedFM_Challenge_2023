OVERRIDE = {
    'model': ["swinv2-b"],
    'dataset': "chest",
    'shot': [5, 10],
    'exp_num': [2],
    #'lr': [1e-8],  # Start learning rate that increases up to 1e-5 (until max_epochs) with cosine annealing
    'train_bs': [8]
}

SETTINGS = {
    'exp_suffix': "exp-test_",
    'log_level': "INFO"
}
