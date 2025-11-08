# CAGED
 
This is the PyTorch implementation for the paper:

> Yue Que, Yingyi Zhang, Xiangyu Zhao, Chen Ma (2025). Causality-aware Graph Aggregation Weight Estimator for Popularity Debiasing in Top-K Recommendation. In CIKM'25, Seoul, Korea, Nov 10-14, 2025. [Paper link](https://arxiv.org/abs/2510.04502).

## To Start With

Run the following commands to reproduce the performances of CAGED model reported in our paper. We provide the training code for `CAGED-LightGCN` and original `LightGCN`. Hyper-parameter arguments in the commands have been tested to be the optimal settings.

<li> <b>CAGED-LightGCN training:</b>
  
```
python train.py --dataset movie --lr 5e-4 --weight 1e-3 --lr2 1e-4 --eps_decay 5e-4 --sc_var 1 --beta 0 --epoch 1000
python train.py --dataset pinterest --lr 5e-4 --weight 1e-3 --lr2 5e-4 --eps_decay 5e-4 --sc_var 1 --beta 0 --epoch 800
python train.py --dataset epinions --lr 1e-3 --weight 2e-4 --lr2 1e-4 --eps_decay 1e-2 --sc_var 1.2 --beta 1 --epoch 1000
```

<li> <b>LightGCN baseline training:</b>
  
```
python train.py --dataset movie --enable_caged 0 --lr 5e-4 --weight 1e-3 --epoch 800
python train.py --dataset pinterest --enable_caged 0 --lr 5e-4 --weight 1e-3 --epoch 800
python train.py --dataset epinions --enable_caged 0 --lr 1e-3 --weight 2e-4 --epoch 800
```

## Citation
Please kindly cite our paper if you find this code useful for your research:

```
@inproceedings{caged2025,
author = {Que, Yue and Zhang, Yingyi and Zhao, Xiangyu and Ma, Chen},
title = {Causality-aware Graph Aggregation Weight Estimator for Popularity Debiasing in Top-K Recommendation},
year = {2025},
isbn = {9798400720406},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3746252.3761155},
doi = {10.1145/3746252.3761155},
booktitle = {Proceedings of the 34th ACM International Conference on Information and Knowledge Management},
pages = {2471–2481},
numpages = {11},
location = {Seoul, Republic of Korea},
series = {CIKM '25}
}
```

Thank you for your interest in our work!
