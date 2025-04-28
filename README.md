# C2C

This code implements the training proposed in the paper "Clean-to-clean: pretraining vision transformers without additional data(https://doi.org/10.1007/s00138-025-01693-w)".

## for C2Cpretraining,
```bash
python train.py --task clear --bs 100 --net vit_designing_clear --lr 1e-4 --aug --n_epochs 100 --watermark 'lr_1e-4_clear'
```

## for CLS finetuning,
```bash
python train.py --task cls --bs 100 --net vit_designing_clear --lr 1e-4 --aug --n_epochs 100 --watermark 'lr_1e-4_clear_finetuned'
```

## if you want comparison with CLS pretraining,
```bash
python train.py --task cls --bs 100 --net vit_designing_clear --lr 1e-4 --aug --n_epochs 100 --watermark 'lr_1e-4_cls'
python train.py --task cls --bs 100 --net vit_designing_clear --lr 1e-4 --aug --n_epochs 100 --watermark 'lr_1e-4_cls_finetuned'
```
