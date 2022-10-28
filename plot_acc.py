# -*-coding:utf-8-*-
import matplotlib.pyplot as plt
import numpy as np
import torch

from utils import setup_seed, plot_acc

if __name__ == '__main__':
    setup_seed()
    moco_v3_frozen = [float(v) for v in torch.load('ckpt/finetuning/ft_v3_frozen_acc.log')]
    sup_retrain = [float(v) for v in torch.load('ckpt/retrain/rt_acc.log')]
    moco_v3 = [float(v) for v in torch.load('ckpt/finetuning/ft_v3_acc.log')]
    moco_v3_mix = [float(v) for v in torch.load('ckpt/ft_v3_mix_acc.log')]
    plot_acc([moco_v3_frozen, moco_v3, sup_retrain, moco_v3_mix],
             tags=['moco v3 frozen', 'moco_v3', 'retrain', 'moco_v3_mix'])
