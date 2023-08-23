OVERRIDE = {
    'model': ["swinv2-b", "swin-b_vpt"],
    'dataset': "colon",
    'shot': [1],
    'exp_num': [1,3,4],
    'lr': 1e-5,
}

SETTINGS = {
    'exp_suffix': "exp-test_",
    'log_level': "INFO"
}
