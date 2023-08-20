
## General:

- Validation / saving checkpoints takes a lot of time, adjust ``val_interval`` in ``train_cfg`` and ``interval`` in default_hooks/checkpoint accordingly


- When using cosine annealing (default in swin schedule), the learning rate decrease will orient itself on ``max_epochs``, i.e. larger max_epoch => slower decrease of LR


- Final Inference on Validation set very slow on cluster (tested on gpu1c), probably due to bottleneck when reading data from drive (15 min on cluster with RTX 4090 vs 5 min at home on GTX1080)


## For Swin + Colon 1-shot:

- Adding the following Augmentation did not improve results 
````python
dict(type='RandomFlip', prob=0.5, direction='vertical'),
dict(type='ColorJitter', hue=0.1, brightness=0.2, contrast=0.2, saturation=0.2)
#Same for higher values like:
dict(type='ColorJitter', hue=0.3, brightness=0.4, contrast=0.4, saturation=0.4),
````
	
 - on ``gpu1c``, (4090), validation batch size of ``128`` works, ``256`` => out of memory


- in config, uncommment ``optim_wrapper=...`` so that Adam optimizer is used, as defined in 
	swin_schedule, which is imported at the top => training more stable


- PromptedSwinTransformer parameter ``window_size`` can't be changed without changing other parameters like padding


- Changing visual prompt length ``vpl`` from 5 to 50 => no noticeable difference

### Interval combos
which work well for colon for certain batch sizes (more data in higher n-shot => decrease val interval since more learning happens in less epochs):

- 1-shot, BS 8, val 250 or 125
- 5-shot, BS 8, val 50 or 25
    
