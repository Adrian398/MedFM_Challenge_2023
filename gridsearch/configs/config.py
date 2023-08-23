OVERRIDE = {
    'model': ["swinv2-b"],
    'dataset': "colon",
    'shot': 5,
    'exp_num': [1],
    'lr': 1e-5
}

SETTINGS = {
    'exp_suffix': "exp-test_",
    # Use dry run to only generate the python commands but not execute them
    'dry_run': False,
    'log_level': "INFO"
}
