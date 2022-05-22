# CVPR-Track-2

## Installation
- Install ``aw_nas`` following ``../aw_nas/README.md``
- Run ``pip install -r requirements.txt``
- Run ``mkdir ~/awnas/plugins`` and ``ln -s `readlink -f ../CVPR-Track-2` ~/awnas/plugins``.

## Reproduction
Save ``CVPR_2022_NAS_Track2_train.json`` and ``CVPR_2022_NAS_Track2_test.json`` at ``./data``.

Run ``python main.py configuration --split-num <SPLIT_NUM> --seed <SEED> --train-dir <TRAIN_DIR>`` to reproduce the experiment.

The result submission file will be saved as ``TRAIN_DIR/CVPR_2022_NAS_Track2_submit_A.json``. 

## Notation
We set ``<SPLIT_NUM>=7, <SEED>=100`` for all tasks except Veri\_rank in our experiments.

For Veri\_rank, we are sorry that we forgot to save the concrete hyper-parameters. Therefore, the result on Veri\_rank would be slightly different from the original submission result.

## Pretrain Model Path
https://cloud.tsinghua.edu.cn/d/61c9d918ef114d0bb625/

Download the folder and run ``python reproduce_from_ckpt.py --train-dir checkpoint`` to reproduce the results from the checkpoints.

## 方案简介
### 概述
本方案使用基于深度学习技术的架构性能预测器对架构性能进行预测。具体而言，预测器包含架构编码器与多层感知机（MLP）两部分。其中架构编码器对输入的架构表示进行编码，然后由后层的MLP对编码的架构隐表征进行性能预测。
### 编码器架构
优异的架构编码器能有效地提取架构特征，在有限的训练数据下有利于性能预测。本方案主要针对 ViT 搜索空间进行编码器设计。

与最基本的 LSTM 编码器不同，本方案不直接用 LSTM 对输入的架构表示进行编码，而是首先用两组共用的MLP，分别对架构每层的 mlp\_ratio以及num\_head进行编码。其次，本方案还引入了每层的深度（0-11）编码，具体地使用 Embedding方法将深度映射到隐空间。将每层的 mlp\_ratio / num\_head / 深度编码拼接在一起，作为这一层模型整体的编码。将所有层的编码沿着深度方向拼接在一起，过LSTM，得到最终的架构编码。
