OVERRIDE = {
    'model': ["swinv2-b"],
    'dataset': "colon",
    'shot': [1,5],
    #'exp_num': [1],
    'lr': 1e-5
}

SETTINGS = {
    'exp_suffix': "neck-test_",
    'log_level': "INFO"
}
