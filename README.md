# C2C

for C2Cpretraining,
python train.py --task clear --bs 100 --net vit_designing_clear --lr 1e-4 --aug --n_epochs 100 --watermark 'lr_1e-4_clear'

for CLS finetuning,
python train.py --task cls --bs 100 --net vit_designing_clear --lr 1e-4 --aug --n_epochs 100 --watermark 'lr_1e-4_clear_finetuned'

if you want comparison with CLS pretraining,
python train.py --task cls --bs 100 --net vit_designing_clear --lr 1e-4 --aug --n_epochs 100 --watermark 'lr_1e-4_cls'
python train.py --task cls --bs 100 --net vit_designing_clear --lr 1e-4 --aug --n_epochs 100 --watermark 'lr_1e-4_cls_finetuned'
