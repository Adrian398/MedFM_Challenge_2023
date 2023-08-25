OVERRIDE = {
    'model': ["swinv2-b"],
    'dataset': ["endo", "chest"],
    'shot': [10],
    'exp_num': [1],
    'lr': [1e-6],  # Start learning rate that increases up to 1e-5 (until max_epochs) with cosine annealing
    'train_bs': [8]
}

SETTINGS = {
    'exp_suffix': "sanity-test_",
    'log_level': "INFO"
}
