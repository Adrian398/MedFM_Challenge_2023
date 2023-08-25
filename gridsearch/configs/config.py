OVERRIDE = {
    'model': ["swinv2-b"],
    'dataset': "endo",
    'shot': [10],
    #'exp_num': [1,2,3,4,5],
    'lr': [1e-8]  # Start learning rate that increases up to 1e-5 (until max_epochs) with cosine annealing
}

SETTINGS = {
    'exp_suffix': "exp-test_",
    'log_level': "INFO"
}
