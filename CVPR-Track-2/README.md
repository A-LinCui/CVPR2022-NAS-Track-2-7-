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
