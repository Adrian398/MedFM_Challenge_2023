OVERRIDE = {
    'model': ["swinv2-b"],
    #'dataset': ["chest", "endo"],
    #'shot': [10],
    #'exp_num': [1],
    'lr': [1e-6],  # Start learning rate that increases up to 1e-5 (until max_epochs) with cosine annealing
    'train_bs': [8],
    'seed': [0]
}

SETTINGS = {
    'exp_suffix': "eval-submission_",
    'log_level': "INFO"
}
