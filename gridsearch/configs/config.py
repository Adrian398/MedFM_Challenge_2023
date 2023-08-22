OVERRIDE = {
    'model': ["swinv2-b", "swin-b_vpt"],
    'dataset': "colon",
    'shot': 1,
    'exp_num': [1,2,3,4,5],
    'lr': 1e-6
}

SETTINGS = {
    'exp_suffix': "Exp_Test",
    'dry_run': True,
    'log_level': "INFO"
}
