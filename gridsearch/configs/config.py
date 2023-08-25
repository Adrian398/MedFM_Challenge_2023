OVERRIDE = {
    'model': ["swinv2-b"],
    'dataset': "colon",
    'shot': [1, 5, 10],
    'exp_num': [1],
    'lr': [1e-6],  # Start learning rate that increases up to 1e-5 (until max_epochs) with cosine annealing
    'train_bs': [8]
}

SETTINGS = {
    'exp_suffix': "sanity-test_",
    'log_level': "INFO"
}
