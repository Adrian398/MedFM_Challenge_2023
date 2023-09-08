OVERRIDE = {
    'model': ["pre_processed_resnet"],
    'dataset': ["endo"],
    #'shot': [5, 10],
    #'exp_num': [1],
    'lr': [1e-6],  # Start learning rate that increases up to 1e-5 (until max_epochs) with cosine annealing
    'train_bs': [32],
    'seed': [0]
}

SETTINGS = {
    'exp_suffix': "pre_processed_",
    'log_level': "INFO"
}
