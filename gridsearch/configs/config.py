OVERRIDE = {
    'model': ["convnext-v2-b"],
    'dataset': ["colon"],
    'shot': [5],
    'exp_num': [3],
    'lr': [5e-5, 5e-6, 1e-6],  # Start learning rate that increases up to 1e-5 (until max_epochs) with cosine annealing
    'train_bs': [1],
    'seed': [0, 1, 2, 3, 4, 5, 6, 12, 42]
}

SETTINGS = {
    'exp_suffix': "seed_test",
    'log_level': "INFO"
}
