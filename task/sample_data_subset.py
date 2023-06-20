#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@file: task/sample_data_subset.py
@time: 2022/12/06 20:03
@desc:
"""

import argparse

from data.dataloader import SST2Dataloader, AGNewsDataloader, TwentyNewsGroupDataloader, R8Dataloader, R52Dataloader, \
    MRDataloader
from utils.random_seed import set_basic_random_seed


def get_argparser():
    parser = argparse.ArgumentParser(description="Sample-Subset-Data")
    parser.add_argument("--seed", default=2333, type=int, help="random seed")
    parser.add_argument("--data_dir", default="../sst2", type=str, )
    parser.add_argument("--save_file_path", default="./test.txt", type=str)
    parser.add_argument("--data_type", default="test", type=str)
    parser.add_argument("--sample_ratio", default=0.25, type=float)
    parser.add_argument("--sample_strategy", default="dist", type=str)
    parser.add_argument("--dataset_name", default="sst2", type=str)
    return parser


def main():
    parser = get_argparser()
    args = parser.parse_args()
    set_basic_random_seed(args.seed)
    if "sst2" in args.dataset_name.lower():
        dataloader = SST2Dataloader(args.data_dir)
    elif "agnews" in args.dataset_name.lower():
        dataloader = AGNewsDataloader(args.data_dir)
    elif "20news_expire" in args.dataset_name.lower():
        dataloader = TwentyNewsGroupDataloader(args.data_dir)
    elif "r8" in args.dataset_name.lower():
        dataloader = R8Dataloader(args.data_dir)
    elif "r52" in args.dataset_name.lower():
        dataloader = R52Dataloader(args.data_dir)
    elif "mr" in args.dataset_name.lower():
        dataloader = MRDataloader(args.data_dir)
    else:
        raise ValueError(args.dataset_name.lower())
    file_format = args.save_file_path.split(".")[-1]
    dataloader.split_and_save_subset(args.save_file_path, data_type=args.data_type,
                                     sample_ratio=args.sample_ratio,
                                     sample_strategy=args.sample_strategy, file_format=file_format)
    print("=*" * 20)
    print(f"successfully sample {args.sample_ratio} {args.dataset_name}")
    print("=*" * 20)


if __name__ == "__main__":
    main()
