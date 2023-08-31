OVERRIDE = {
    'model': ["resnet101"],
    'dataset': ["endo"],
    'shot': [1],
    'exp_num': [3],
    'lr': [1e-6, 5e-6, 5e-5],  # Start learning rate that increases up to 1e-5 (until max_epochs) with cosine annealing
    #'train_bs': [32],
    'seed': [0]
}

SETTINGS = {
    'exp_suffix': "resnet_sanity_test_",
    'log_level': "INFO"
}
