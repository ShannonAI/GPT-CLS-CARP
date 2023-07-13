#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@file: tests/debug/evaluate_inference.py
@time: 2022/12/06 20:03
@desc:
"""

import json
import os
from glob import glob

import evaluate
from tqdm import tqdm


def eval_xiaofei_folder(repo_path):
    file_path_lst = glob(os.path.join(repo_path, "*.json"))
    sacrebleu = evaluate.load("sacrebleu")
    for file_path in file_path_lst:
        with open(file_path, "r") as f:
            data_items = json.load(f)
            eval_bleu_lst = []
            counter = 0
            for data_item in tqdm(data_items):
                tmp_scores = []
                counter += 1
                for res in data_item["results"]:
                    src = data_item["sentence"]
                    res = res.replace(" ", "")
                    scores = sacrebleu.compute(predictions=[res], references=[src], tokenize="zh")
                    tmp_scores.append(scores["score"])
                eval_bleu_lst.append(min(tmp_scores))
            assert len(eval_bleu_lst) == counter
            print(file_path)
            print(sum(eval_bleu_lst) / float(counter))
            print("=" * 40)


def eval_yicheng_folder(repo_path):
    file_path_lst = glob(os.path.join(repo_path, "*.txt"))
    sacrebleu = evaluate.load("sacrebleu")
    for file_path in file_path_lst:
        with open(file_path, "r") as f:
            datalines = f.read().split("输入:\n")
            eval_bleu_lst = []
            counter = 0

            for data in tqdm(datalines[1:]):
                tmp_scores = []
                counter += 1
                lines = data.split("输出:")
                assert len(lines) == 2
                src = lines[0]
                res = lines[-1].split("\n")
                res = [item for item in res if len(item) != 0]
                for tmp_res in res:
                    src = src.replace(" ", "")
                    tmp_res = tmp_res.replace(" ", "")
                    scores = sacrebleu.compute(predictions=[tmp_res], references=[src], tokenize="zh")
                    
                    tmp_scores.append(scores["score"])
                eval_bleu_lst.append(min(tmp_scores))

            assert len(eval_bleu_lst) == counter
            print(file_path)
            print(sum(eval_bleu_lst) / float(counter))
            print("=" * 40)


if __name__ == "__main__":
    # repo_path = "/data2/lixiaoya/datasets/inference/xiaofei_inference"
    # eval_xiaofei_folder(repo_path)

    repo_path = "/data2/lixiaoya/datasets/inference/yicheng_inference"
    eval_yicheng_folder(repo_path)
