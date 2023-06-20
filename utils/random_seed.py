#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@file: random_seed.py
@time: 2022/07/22 20:03
@desc:
"""

import random

import numpy as np
import torch
from transformers import set_seed

try:
    from pytorch_lightning import seed_everything
except:
    pass


def set_train_random_seed(seed: int):
    """set seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    seed_everything(seed=seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    set_seed(seed)


def set_predict_random_seed(seed: int):
    """set seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    set_seed(seed)


def set_basic_random_seed(seed: int):
    """set basic random seed when working with python3"""
    random.seed(seed)
    np.random.seed(seed)


if __name__ == '__main__':
    # without this line, x would be different in every execution.
    set_train_random_seed(0)

    x = np.random.random()
    print(x)
