#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@file: utils/argparse_utils.py
@time: 2022/12/06 20:03
@desc:
"""

import argparse


def get_finetune_mlm_parser():
    parser = argparse.ArgumentParser(description="Training")
    parser.add_argument("--bert_path", required=True, type=str, help="bert config file")
    parser.add_argument("--dataset_name", default="sst2", type=str)
    parser.add_argument("--data_dir", required=True, type=str, help="train data path")
    parser.add_argument("--batch_size", type=int, default=8, help="batch size")
    parser.add_argument("--lr", type=float, default=2e-5, help="learning rate")
    parser.add_argument("--workers", type=int, default=0, help="num workers for dataloader")
    parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--warmup_steps", default=0, type=int, help="warmup steps")
    parser.add_argument("--use_memory", action="store_true", help="load dataset to memory to accelerate.")
    parser.add_argument("--max_length", default=512, type=int, help="max length of dataset")
    parser.add_argument("--save_path", required=True, type=str, help="train data path")
    parser.add_argument("--save_topk", default=10, type=int, help="save topk checkpoint")
    parser.add_argument("--pretrain_checkpoint", default="", type=str, help="train data path")
    parser.add_argument("--warmup_proportion", default=0.01, type=float)
    parser.add_argument("--hidden_dropout_prob", default=0.1, type=float, help="dropout probability")
    parser.add_argument("--model_sign", default="bert", type=str, )
    parser.add_argument("--only_eval", action="store_true")
    parser.add_argument("--eval_ckpt_path", default="", type=str)
    parser.add_argument("--test_file_name", default="test", type=str)
    parser.add_argument("--train_file_name", default="train", type=str)
    parser.add_argument("--lr_scheduler", default="default", type=str)
    parser.add_argument("--optimizer", default="adamw", type=str)
    return parser


def get_evaluate_mlm_parser():
    parser = argparse.ArgumentParser(description="Arguments using when evals")
    parser.add_argument("--bert_path", required=True, type=str, help="bert config file")
    parser.add_argument("--max_length", default=512, type=int, help="max length of dataset")
    parser.add_argument("--data_dir", required=True, type=str, help="train data path")
    parser.add_argument("--model_sign", default="bert", type=str, )
    parser.add_argument("--save_path", required=True, type=str, help="train data path")
    parser.add_argument("--eval_ckpt_path", default="", type=str, help="train data path")
    parser.add_argument("--dataset_name", default="sst2", type=str)
    parser.add_argument("--batch_size", type=int, default=1, help="batch size")
    parser.add_argument("--warmup_proportion", default=0.01, type=float)
    parser.add_argument("--lr", type=float, default=2e-5, help="learning rate")
    parser.add_argument("--workers", type=int, default=0, help="num workers for dataloader")
    parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--warmup_steps", default=0, type=int, help="warmup steps")
    parser.add_argument("--use_memory", action="store_true", help="load dataset to memory to accelerate.")

    return parser
