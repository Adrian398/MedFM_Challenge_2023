OVERRIDE = {
    'model': ["swinv2-b"],
    'dataset': "colon",
    'shot': 1,
    'exp_num': [1,2,3,4,5],
    'lr': 1e-6
}

SETTINGS = {
    'exp_suffix': "swinv2-exp-test_",
    # Use dry run to only generate the python commands but not execute them
    'dry_run': False,
    'log_level': "INFO"
}
