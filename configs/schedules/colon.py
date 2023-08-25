optimizer = dict(betas=(0.9, 0.999), eps=1e-08, lr=0.05, type='AdamW', weight_decay=0.01)

optim_wrapper = dict(
    optimizer=optimizer,
    paramwise_cfg=dict(
        norm_decay_mult=0.0,
        bias_decay_mult=0.0,
        flat_decay_mult=0.0,
        custom_keys={
            '.absolute_pos_embed': dict(decay_mult=0.0),
            '.relative_position_bias_table': dict(decay_mult=0.0)
        }),
)

param_scheduler = [
    dict(by_epoch=True, end=1, start_factor=1, type='LinearLR'),
    dict(begin=1, by_epoch=True, eta_min=1e-05, type='CosineAnnealingLR'),
]

# train, val, test setting
train_cfg = dict(by_epoch=True, max_epochs=20, val_interval=1)
val_cfg = dict()
test_cfg = dict()

auto_scale_lr = dict(base_batch_size=1024)
