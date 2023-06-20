#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@file: utils/get_logger.py
@time: 2022/12/06 20:03
@desc:
"""

import logging
import os


def get_info_logger(name: str, save_log_file: str = None, print_to_console=True,
                    log_format: str = '%(asctime)s - %(name)s : - %(message)s', ):
    """get information logger"""
    logging.basicConfig(format=log_format, level=logging.INFO)
    logger = logging.getLogger(name)
    # remove duplicate infos
    # logger.propagate = False
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter(log_format, datefmt='%Y-%m-%d %H:%M:%S')

    if save_log_file is not None:
        output_dir = os.path.dirname(save_log_file)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        fh = logging.FileHandler(save_log_file)
        fh.setLevel(logging.INFO)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    if not print_to_console:
        return logger

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    return logger
