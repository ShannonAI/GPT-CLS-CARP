#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@file: task/save_nearest_neighbors.sh
@time: 2022/12/06 20:03
@desc:
"""

import argparse

from data.data_retriever import FinetunedMLMRetriever, SimCSERetriever
from data.dataloader import SST2Dataloader, AGNewsDataloader, TwentyNewsGroupDataloader, R8Dataloader, R52Dataloader, \
    MRDataloader
from utils.random_seed import set_basic_random_seed


def get_argparser():
    parser = argparse.ArgumentParser(description="Sample-Subset-Data")
    parser.add_argument("--seed", default=2333, type=int, help="random seed")
    parser.add_argument("--data_dir", default="../sst2", type=str, )
    parser.add_argument("--mlm_dir", default="./test.txt", type=str)
    parser.add_argument("--encoder_ckpt_path", default="test", type=str)
    parser.add_argument("--candidate_type", default="train", type=str)
    parser.add_argument("--query_type", default="test", type=str)
    parser.add_argument("--search_threshold", default=0.0, type=float)
    parser.add_argument("--top_k", default=24, type=int)
    parser.add_argument("--ranking_model", default="finetuned-roberta-large", type=str)
    parser.add_argument("--save_nearest_neighbor_path", type=str, default="")
    parser.add_argument("--dataset_name", type=str, default="sst2")
    parser.add_argument("--max_len", type=int, default=280)
    parser.add_argument("--retriever_type", type=str, default="ft")
    return parser


def main():
    parser = get_argparser()
    args = parser.parse_args()
    set_basic_random_seed(args.seed)

    if args.dataset_name == "sst2":
        data_loader = SST2Dataloader(args.data_dir)
    elif args.dataset_name == "agnews":
        data_loader = AGNewsDataloader(args.data_dir)
    elif args.dataset_name == "20news_expire":
        data_loader = TwentyNewsGroupDataloader(args.data_dir)
    elif args.dataset_name == "r8":
        data_loader = R8Dataloader(args.data_dir)
    elif args.dataset_name == "r52":
        data_loader = R52Dataloader(args.data_dir)
    elif args.dataset_name == "mr":
        data_loader = MRDataloader(args.data_dir)
    else:
        raise ValueError(args.dataset_name)

    if args.retriever_type == "ft":
        data_retriever = FinetunedMLMRetriever(args.mlm_dir, args.encoder_ckpt_path, max_len=args.max_len,
                                               num_labels=len(data_loader.get_labels()))
    elif args.retriever_type == "simcse":
        data_retriever = SimCSERetriever("/data2/lixiaoya/gpt_data_models/models/sup-simcse-roberta-large",
                                         max_len=args.max_len)
    else:
        raise ValueError
    data_retriever.search_nearest_neighbors_and_save_to_file(args.save_nearest_neighbor_path,
                                                             dataloader=data_loader,
                                                             candidate_type=args.candidate_type,
                                                             query_type=args.query_type,
                                                             search_threshold=args.search_threshold,
                                                             top_k=args.top_k,
                                                             ranking_model=args.ranking_model,
                                                             )
    print("=$" * 20)
    print(args.save_nearest_neighbor_path)
    print("=$" * 20)


if __name__ == "__main__":
    main()
