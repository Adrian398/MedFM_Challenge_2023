OVERRIDE = {
    'model': ["swin-b_vpt"],
    'dataset': "colon",
    'shot': [5],
    'exp_num': [1,2,3,4,5],
    'lr': 1e-5,
}

SETTINGS = {
    'exp_suffix': "-test_",
    'log_level': "INFO"
}
