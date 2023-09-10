OVERRIDE = {
    'model': ["pre_processed_swinv2"],
    'dataset': ["endo"],
    #'shot': [5, 10],
    #'exp_num': [1],
    'lr': [1e-6],  # Start learning rate that increases up to 1e-5 (until max_epochs) with cosine annealing
    'train_bs': [1, 2],
    'seed': [1, 42]
}

SETTINGS = {
    'exp_suffix': "pre_processed",
    'log_level': "INFO"
}
