OVERRIDE = {
    'model': ["convnext-v2-b"],
    'dataset': ["chest"],
    #'shot': [5, 10],
    #'exp_num': [1],
    'lr': [1e-6],  # Start learning rate that increases up to 1e-5 (until max_epochs) with cosine annealing
    'train_bs': [1, 2],
    'seed': [0, 42]
}

SETTINGS = {
    'exp_suffix': "",
    'log_level': "INFO"
}
