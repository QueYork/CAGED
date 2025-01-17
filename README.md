# CAGED
 
This is the PyTorch implementation for the paper:

> Causality-aware Graph Aggregation Weight Estimator for Popularity Debiasing in Top-K Recommendation

## To Start With

Run the following commands to reproduce the performances of CAGED model reported in our paper. We provide the training code for `CAGED-LightGCN` and original `LightGCN`. Hyper-parameter arguments in the commands have been tested to be the optimal settings.

<li> <b>CAGED-LightGCN training:</b>
  
```
python train.py --dataset movie --lr 5e-4 --weight 1e-3 --lr2 1e-4 --eps_decay 5e-4 --sc_var 1 --beta 0 --epoch 1000
python train.py --dataset pinterest --lr 5e-4 --weight 1e-3 --lr2 5e-4 --eps_decay 5e-4 --sc_var 1 --beta 0 --epoch 1000
python train.py --dataset epinions --lr 1e-3 --weight 2e-4 --lr2 1e-4 --eps_decay 1e-2 --sc_var 1.2 --beta 1 --epoch 1000
```

<li> <b>LightGCN baseline training:</b>
  
```
python train.py --dataset movie --enable_vae 0 --lr 5e-4 --weight 1e-3 --epoch 800
python train.py --dataset pinterest --enable_vae 0 --lr 5e-4 --weight 1e-3 --epoch 800
python train.py --dataset epinions --enable_vae 0 --lr 1e-3 --weight 2e-4 --epoch 800
```

Thank you for your interest in our work!
