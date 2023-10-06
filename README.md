# M-MAE (Matrix Variational Masked Autoencoder)

Official implementation of paper "Information Flow in Self-Supervised Learning" (https://arxiv.org/abs/2309.17281).

This repository includes a PyTorch implementation of the Matrix Variational Masked Autoencoder (M-MAE). M-MAE is an extension of [MAE (He et al., 2022)](https://arxiv.org/pdf/2111.06377.pdf) and [U-MAE (Zhang et al., 2022)](https://arxiv.org/pdf/2210.08344.pdf) by further encouraging the feature uniformity of MAE from a matrix information theoretic perspective. 

## Instructions
This repo is based on the [official code of MAE](https://github.com/facebookresearch/mae) and [official code of U-MAE](https://github.com/zhangq327/U-MAE) with minor modifications below, and we follow all the default training and evaluation configurations of MAE. Please see their instructions [README_mae.md](README_mae.md) for details.

**Main differences.** In M-MAE, we introduce a ``uniformity_loss_TCR``  (implemented in ``loss_func.py``) as a uniformity regularization to the MAE loss. It also introduces an additional hyper-parameter ``lamb`` (default to ``1e-2``) in ``pretrain.sh``, which represents the coefficient of the uniformity regularization in the M-MAE loss. 

**Minor points:**
1. We add a linear classifier to monitor the online linear accuracy and its gradient will not be backpropagated to the backbone encoder.
2. For efficiency, we only train M-MAE for 200 epochs, and accordingly, we adopt 20 warmup epochs.



## Acknowledgement

Our code follows the official implementations of MAE (https://github.com/facebookresearch/mae) and U-MAE (https://github.com/zhangq327/U-MAE). We thank the authors for their great work.
